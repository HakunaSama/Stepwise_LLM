"""
和train_pretrain基本相同，只是要加载模型和权重，并且只对训练数据中模型回答部分计算loss
"""
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
# 导入SFTDataset
from dataset.llm_full_sft_dataset import SFTDataset

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def init_model(lm_config):
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model')
    #初始化模型
    model = MicroCortexForCausalLM(lm_config)

    moe_path = '_moe' if lm_config.use_moe else ''
    # 模型名
    ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'
    # 加载模型参数
    state_dict = torch.load(ckp, map_location=args.device)
    # 加载模型参数到模型
    model.load_state_dict(state_dict, strict=False)

    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer

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
    # 依旧使用 token‑level 交叉熵；不在这里求平均以便后续乘以 loss_mask
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    # 从数据集中取出一个batch的X，Y，loss_mask，lossmask和padding是对应的
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将batch数据全部搬到指定设备上
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算并设置学习率（step级别）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 前向传播
            res = model(X)
            # 计算tokenlevel的交叉熵损失
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            # 只统计训练数据中模型回答区域的token_level级别交叉熵，然后求平均
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 损失加上MoE的正则loss
            loss += res.aux_loss
            # 当前minibatch的梯度，要除以batch/minibatch
            loss = loss / args.accumulation_steps

        # 自动混合精度训练会把反向传播前的loss乘以放大因子，来避免(FP16/BP16)下的下溢出
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
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            # 同步到wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
            # DDP 下需要取 module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度存储，节约磁盘空间
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroCortex Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MicroCortex-Full-SFT")
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
    parser.add_argument('--use_moe', default=True, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")
    args = parser.parse_args()

    ####################
    # 模型配置
    ####################
    lm_config = MicroCortexConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                  use_moe=args.use_moe)
    # 创建输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb run 名称 = 超参组合
    args.wandb_run_name = f"MicroCortex-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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
    model, tokenizer = init_model(lm_config)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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