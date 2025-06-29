"""
自定义模型和参数转换Transformer兼容格式脚本，转换后可以开源至huggingface、modelscope
1.权重格式互相转换
    将pytorch MicroCortexForCausalLM（我们自定义的模型，可以用from_pretrained加载）.pth文件转成Transformers兼容格式(save_pretrained目录)
2.精度转换：在保存时可直接把权重转换成bfloat16或者其他，降低显存占用
3.统计参数量和保存tokenizer：转换完成后打印模型可训练参数总量；把对应的tokenizer连同模型一起保存确保加载端可直接使用
4.反向转换：提供convert_transformers2torch：把已保存的Transformers目录重新导出成单个.pth文件（便于Flash-Attention、纯Pytorch推理等场景）
"""
# 导入相关包
import os
import sys

# 把项目根目录加入 PYTHONPATH，方便 import
__package__ = "deploy"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

# 导入我们自己实现的MiniMind模型
from model.model_microcortex import MicroCortexConfig, MicroCortexForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)

"""
把pytorch checkpoint转换成Transformers-MicroCortex格式
"""
def convert_torch2transformers_microcortex(torch_path, transformers_path, dtype=torch.bfloat16):
    """
    将.pth权重保存为自定义MicroCortex的Transformers兼容格式
    Args:
        torch_path: 本地.pth文件路径
        transformers_path: 保存为Transformer格式的目录
        dtype: 保存时转换成什么精度
    """
    # 注册到AutoClass，方便AutoModel.from_pretrained直接识别到
    MicroCortexConfig.register_for_auto_class()
    MicroCortexForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    # 构建模型
    lm_model = MicroCortexForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载权重
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)

    #转换模型权重精度
    lm_model = lm_model.to(dtype)

    # 统计模型参数
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')

    # 保存transformers格式模型
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")

"""
把Transformers转换成pytorch checkpoint格式
"""
def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")

"""
转换示例
"""
if __name__ == '__main__':
    # 这里根据需要写死一个配置，也可以 argparse 读超参
    lm_config = MicroCortexConfig(hidden_size=768, num_hidden_layers=16, max_seq_len=8192, use_moe=False)

    # 源权重和目标保存路径
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"

    transformers_path = '../MicroCortex'

    convert_torch2transformers_microcortex(torch_path, transformers_path)