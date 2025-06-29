import torch
from torch import optim, nn

class LoRA(nn.Module):
    def __init__(self,
                 in_features,   #输入特征
                 out_features,  #输出特征
                 rank):         #压缩维度
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A in_features -> rank
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B rank -> out_features
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    # 遍历模型的所有子模块
    for name, module in model.named_modules():
        # 只给权重是方阵的 nn.Linear 注入 LoRA
        #   ─ 假设它们是 Q / K / V / O 投影；
        #   ─ 非方阵（mlp down/up）不处理，节省显存。
        # 但是请注意，本项目中没有使用LoRA训练模型，因为我们的是MLA架构，没有方阵Linear，这里需要修改才行
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)