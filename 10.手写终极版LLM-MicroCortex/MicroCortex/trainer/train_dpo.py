import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_microcortex import MicroCortexConfig, MicroCortexForCausalLM
# 导入DPODataset
from dataset.llm_dpo_dataset import DPODataset

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# logits转labels对应的probs
def logits_to_probs(logits,     #(batch_size, seq_len, vocab_size)
                    labels):    #(batch_size, seq_len)

    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs    #(batch_size, seq_len)，即为每个位置预测下一个token为labels中结果的概率

# dpo loss计算函数，可以参照手写DPO笔记中的公式进行理解
def dpo_loss(
        ref_probs,  #对于label 基准模型输出的概率分布 (batch_size, seq_len)
        probs,      #对于label 训练模型输出的概率分布 (batch_size, seq_len)
        mask,       #掩码
        beta):      #系数
    # ref_probs 和 probs 都是 shape: (batch_size, seq_len)
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    # log概率相加是计算assistant整个句子输出的概率，为了平衡序列长度的影响再除以以下序列长度
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze() # (batch_size)
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze() # (batch_size)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]  #基准模型对正样本输出的概率(batch_size//2)
    reject_ref_probs = ref_probs[batch_size // 2:]  #基准模型对负样本输出的概率(batch_size//2)
    chosen_probs = probs[:batch_size // 2]          #训练模型对正样本输出的概率(batch_size//2)
    reject_probs = probs[batch_size // 2:]          #训练模型对负样本输出的概率(batch_size//2)

    #(batch_size//2)
    pi_logratios = chosen_probs - reject_probs #ln pi(y_w) - ln pi(y_l)，训练模型对正负样本输出的概率差，也就是训练模型认为的正负样本的差距
    ref_logratios = chosen_ref_probs - reject_ref_probs #ln ref(y_w) - ln ref(y_l)，基准模型对正负样本输出的概率差，也就是基准模型认为的正负样本的差距
    #训练模型的公正度-基准模型的公正度，公正都就代表模型对于好坏样本的区分程度
    logits = pi_logratios - ref_logratios #{ ln pi(y_w) - ln pi(y_l) } - { ln ref(y_w) - ln ref(y_l) }

    # (batch_size//2)
    loss = -F.logsigmoid(beta * logits) #乘以beta后计算sigmod再取负号
    return loss.mean()#对多个batch取均值作为最终这个batch的loss

def init_model(lm_config):
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    # 初始化模型
    model = MiniMindForCausalLM(lm_config)

    moe_path = '_moe' if lm_config.use_moe else ''
    # 模型名
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    # 加载模型参数
    state_dict = torch.load(ckp, map_location=args.device)
    # 加载模型参数到模型
    model.load_state_dict(state_dict, strict=False)

    # 初始化参考模型
    ref_model = MiniMindForCausalLM(lm_config)
    # 加载参考模型参数
    ref_model.load_state_dict(state_dict, strict=False)
    # 设置参考模型为eval模式
    ref_model.eval()
    # 设置参考模型不计算梯度
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def train_epoch(epoch, wandb):
    start_time = time.time()
    # 取一个batch的数据
    for step, batch in enumerate(train_loader):

        x_chosen = batch['x_chosen'].to(args.device)            #正样本输入
        x_rejected = batch['x_rejected'].to(args.device)        #负样本输入
        y_chosen = batch['y_chosen'].to(args.device)            #正样本标签
        y_rejected = batch['y_rejected'].to(args.device)        #负样本标签
        mask_chosen = batch['mask_chosen'].to(args.device)      #正样本标签掩码
        mask_rejected = batch['mask_rejected'].to(args.device)  #负样本标签掩码
        x = torch.cat([x_chosen, x_rejected], dim=0)#正负样本输入在batch维度拼接
        y = torch.cat([y_chosen, y_rejected], dim=0)#正负样本标签在batch维度拼接
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 计算并设置学习率（step级别）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # -------- 1. 参考模型（冻结） --------
            with torch.no_grad():# ref_model 只做前向，不要梯度
                #正负样本一起被ref_model前向传播
                ref_outputs = ref_model(x)#[batch , seq_len-1]
                #ref_model的ref_logits输出
                ref_logits = ref_outputs.logits#[batch , seq_len-1 , vocab]
            # 取出参考模型在每个时间步「预测下一 token=y」的概率
            ref_probs = logits_to_probs(ref_logits, y)  # logits_to_probs 会先 softmax，再 gather 到标签位置
            ref_probs = ref_probs * mask    # 用 loss_mask 过滤掉 <pad> token

            # -------- 2. 当前训练模型（可更新） --------
            # 同样把输入序列喂给当前策略
            outputs = model(x)#[batch , seq_len-1]
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask

            # -------- 3. 计算 DPO 损失 --------
            # ref_probs、probs : [batch , seq_len-1] 分别为基准模型和训练模型每个时间步「预测下一 token=y」的概率
            # 其中batch一分为2，一部分是正样本，一部分是负样本，
            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
            loss = loss / args.accumulation_steps

        # 自动混合精度训练，把反向传播前的loss乘以放大因子，来避免(FP16/BP16)下的下溢出
        scaler.scale(loss).backward()

        # 每 accumulation_steps 个 minibatch 更新一下权重
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度除以放大因子，将梯度恢复到真实尺度
            scaler.unscale_(optimizer)
            # 裁剪过大梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器更新参数并且更新scaler
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度节约显存
            optimizer.zero_grad(set_to_none=True)

        # 日志打印与wandb可视化
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            # 同步到wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]['lr'],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                    }
                )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'

            # DDP 下需要取 module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度存储，节约磁盘空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")

    args = parser.parse_args()

    ####################
    # 模型配置
    ####################
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    # 创建输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb run 名称 = 超参组合
    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # AMP 上下文：CPU 下为 nullcontext()，GPU下为 torch.cuda.amp.autocast()
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 判断是否为 ddp 进程（RANK 环境变量由 torchrun 注入）
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    ####################
    # 随机种子
    ####################
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # 若为 DDP，需要调用初始化并根据 rank 调整种子/设备
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    ####################
    # wandb初始化
    ####################
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        # 初始化wandb的项目和训练名
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    ####################
    # 模型、数据集初始化
    ####################
    model, ref_model, tokenizer = init_model(lm_config)

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 若为分布式，使用 DistributedSampler 保证各进程拿到不同切片
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # AMP 梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 若启用 DDP，包裹模型；排除不需要同步的 pos_cis（静态预计算矩阵），但是其实原作者这里有错误，它的旋转矩阵变量注册名并不是pos_cis
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    # 每 epoch 的 step 数
    iter_per_epoch = len(train_loader)

    ####################
    # 开始训练
    ####################
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)