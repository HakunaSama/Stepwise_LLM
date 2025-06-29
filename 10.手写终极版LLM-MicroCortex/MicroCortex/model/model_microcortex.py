"""
首先导入相关包
"""
#config
from transformers import PretrainedConfig

#model
import math
import torch
from torch import nn
from transformers.activations import ACT2FN #huggingface工具，将字符串（“gelu”、“relu”...）映射到对应激活函数
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F

# Hugging Face 抽象基类
# PreTrainedModel：提供加载 / 保存权重、配置等通用逻辑
# GenerationMixin：注入 generate() 等文本生成方法
# PretrainedConfig：存放模型结构、超参数并支持 .from_pretrained() 解析
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig

# Hugging Face 统一输出 dataclass
# 用于自回归语言模型（Causal LM），包含 logits、past_key_values、hidden_states 等字段
from transformers.modeling_outputs import CausalLMOutputWithPast

"""
定义模型配置
"""
class MicroCortexConfig(PretrainedConfig):
    model_type = "microcortex"

    def __init__(
            self,
            dropout: float = 0.0,       #随机失活率
            hidden_act: str = 'silu',   #隐藏层激活函数
            hidden_size: int = 512,     #隐藏层维度
            intermediate_size: int = None,          #中间层维度
            num_hidden_layers: int = 8,     #隐藏层数
            vocab_size: int = 6400,         #词表大小
            rms_norm_eps: float = 1e-05,    #RMS归一化eps
            flash_attn: bool = True,        #是否启用flash注意力
            ####################################################
            # 这里是MOE的特殊配置，当use_moe是false时，下面都将无效
            ####################################################
            use_moe: bool = False,          #是否启用moe
            num_experts_per_tok: int = 2,   #每个token总的专家数量
            n_routed_experts: int = 4,      #被路由的专家数
            n_shared_experts: int = 1,      #共享专家数
            scoring_func: str = 'softmax',  #评分函数，默认为'softmax'
            aux_loss_alpha: float = 0.1,    #辅助损失的alpha参数
            seq_aux: bool = True,           #是否在序列级别上计算辅助损失
            norm_topk_prob: bool = True,    #是否标准化topk的概率
            mlp_bias=False,
            ####################################################
            # 这里是MLA的特殊配置
            ####################################################
            attention_dropout: float=0.0,
            rope_theta: int = 1000000.0,            #位置编码常量
            max_position_embeddings: int = 32768,   #最大位置编码
            attention_bias: bool=False,
            num_heads: int=8,                       # 注意力头数
            q_lora_rank: int=4,                     # Q压缩后的维度
            qk_rope_head_dim:int=32,                 # Q和K位置特征head维度
            kv_lora_rank:int=4,                     # KV压缩后的维度
            v_head_dim: int=64,                      # Vhead的维度
            qk_nope_head_dim: int=64,                # QKhead的维度，不带位置特征
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.flash_attn = flash_attn
        ####################################################
        # 这里是MOE的特殊配置，当use_moe是false时，下面都将无效
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.mlp_bias = mlp_bias
        ####################################################
        # 这里是MLA的特殊配置，本项目
        ####################################################
        self.attention_dropout=attention_dropout
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias=attention_bias
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim

"""
定义模型结构
"""
# 与layernorm不同的是，计算layernorm时是减去均值除以标准差，然后乘以权重。而rmsnorm没有减去均值，是直接除以均值平方和然后开根号。
# 我们将使用RMSNorm来代替nn.LayerNorm(hidden_size)层
class MicroCortexRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# 获取所有位置的旋转矩阵
# 该层将会被包裹在每一个MLP层中，并且该层将会注册inv_freq，cos_cached，sin_cached参数，并且每一个MLP层中这些参数是一样的，这里可以进行优化，让所有的MLP层共享同一份位置编码的参数。
class MicroCortexRotaryEmbedding(nn.Module):
    def __init__(self, dim, #要进行编码的向量的维度
                 max_position_embeddings=2048, #能够编码的最大位置
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 较小索引位置对应较低频率
        # 较大的索引位置有较高的频率

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = max_position_embeddings

    # 注册从0到seq_len位置的旋转矩阵
    def _set_cos_sin_cache(self,
                           seq_len, #要编码的最大位置
                           device,
                           dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册cos_cached，sin_cached参数
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    """
    Args:
        x:需要被旋转位置编码的向量，这里传入其实只是要用它的类型而已
        seq_len:序列长度
    Raise:
        从0到seq_len位置的旋转矩阵
    """
    def forward(self, x, seq_len=None):
        # x: [bs, num_heads, seq_len, head_size]
        # 如果传入的seq_len比我们缓存的max_seq_len还要大，那就只能重新计算并缓存一份从0到seq_len位置的旋转矩阵了
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            #print("seq_len:", seq_len, self.max_seq_len_cached)
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    # 将最后一维一分为二
    x1 = x[..., : x.shape[-1] // 2]# 取前半部分（偶数维）
    x2 = x[..., x.shape[-1] // 2 :]# 取后半部分（奇数维）
    # 返回按旋转矩阵规则组合的新向量 [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# 对于qk进行position_ids位置的旋转位置编码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# 对于q进行position_ids位置的旋转位置编码
def apply_rotary_pos_emb_v2(q: torch.Tensor, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


"""
这是带有矩阵吸收的版本
"""
class MicroCortexMLA(nn.Module):
    def __init__(self,config: MicroCortexConfig):
        super().__init__()
        #### part1 , mha 部分 ####
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads   #头数
        self.v_head_dim = config.v_head_dim # Vhead的维度


        #### part2MLA 压缩部分 ####
        # 在 deepseek v3 中， hidden_size 7168
        # 压缩后的kv 为 512，压缩比例 1/14
        # 压缩后的q 为 1536，压缩比例 1/4.7
        # rope部分是64

        ### part2.1 down 压缩
        # 其实qk_nope_head_dim、q_nope_head_dim、k_nope_head_dim是一样的，这里为了更好对应上公式和图示中的过程，从名称上进行区分
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_nope_head_dim = config.qk_nope_head_dim  #Qhead维度（不带位置特征）
        self.k_nope_head_dim = config.qk_nope_head_dim  #Khead维度（不带位置特征）
        # 其实qk_rope_head_dim、q_rope_head_dim、k_rope_head_dim是一样的，这里为了更好对应上公式和图示中的过程，从名称上进行区分
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_rope_head_dim = config.qk_rope_head_dim  #Q位置特征head维度
        self.k_rope_head_dim = config.qk_rope_head_dim  #K位置特征head维度

        # deepseek v3中hidden_size=7168，q=1536，从7168->1536； 压缩比是 1/4.7
        self.q_lora_rank = config.q_lora_rank   # 压缩了qnope和qrope
        # kv=512，压缩比例 1/14
        self.kv_lora_rank = config.kv_lora_rank # 压缩了knope和vnope

        # WDQ压缩矩阵：hidden_size -> q_lora_rank(压缩qnope和qrope)
        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
        )
        self.q_down_layernorm = MicroCortexRMSNorm(self.q_lora_rank)

        #WDKV压缩矩阵和WKR压缩矩阵：hidden_size -> kv_lora_rank(压缩knope, v) + k_rope_head_dim(压缩krope)
        #这里是把两个矩阵合并到一起了
        #WDKV矩阵：hidden_size -> kv_lora_rank(压缩knope,v)
        #WKR矩阵： hidden_size -> k_rope_head_dim(压缩krope)
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.k_rope_head_dim,
            bias=config.attention_bias,
        )
        # q_rope_head_dim和k_rope_head_dim一般设置的很小，一般是 64
        # 这里的输出是kv_lora_rank(knope,v) + k_rope_head_dim(krope)拼接，后期要进行split

        #只对CKV进行Norm，不对KR继续Norm
        self.kv_down_layernorm = MicroCortexRMSNorm(self.kv_lora_rank)


        ### part2.2 升维
        # q_head_dim也是k_head_dim，两者相等
        self.qk_head_dim = self.q_nope_head_dim + self.q_rope_head_dim
        #WUQ矩阵和WQR矩阵：q_lora_rank(qnope和qrope) -> (q_nope_head_dim + q_rope_head_dim) * num_heads
        #WUQ矩阵：q_lora_rank(压缩qnope和qrope) -> q_nope_head_dim * num_heads(qnope)
        #WQR矩阵：q_lora_rank(压缩qnope和qrope) -> q_rope_head_dim * num_heads(qrope)
        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=config.attention_bias,
        ) # 因为这里的结果是(q_nope1+q_rope1) * num_heads拼接起来的，这里也要split开q_rope和q_nope

        #WUK矩阵和WUV矩阵：kv_lora_rank -> num_heads * (k_nope_head_dim + v_head_dim)
        #WUK矩阵：kv_lora_rank(压缩knope, v) -> num_heads * k_nope_head_dim(knope)
        #WUV矩阵：kv_lora_rank(压缩knope, v) -> num_heads * v_head_dim(v)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * ((self.qk_head_dim - self.qk_rope_head_dim) + self.v_head_dim),
            bias=config.attention_bias,
        )#后期要split开

        #### part3: rope部分 ####
        self.max_postion_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rotary_emb = MicroCortexRotaryEmbedding(
            #每一层都存有一个旋转矩阵，并且都是一样的，这里可以优化成所有层共享一个MicroCortexRotaryEmbedding
            self.qk_rope_head_dim,
            self.max_postion_embeddings,
            self.rope_theta,
        )

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.attention_dropout)

        #### part4: 最后的输出线性层 ####
        #WO矩阵：num_heads * v_head_dim -> hidden_size
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
    """
    MLA (Multi-head Linearized Attention) forward pass
    Args:
        hidden_states:  #输入数据，训练时为一整个seq，推理时就是当前的token的hidden_state
        attention_mask: #输入数据的因果掩码，推理时为None
        position_ids:   #输入数据的position_ids，应该和hidden_states长度相同，代表hidden_states的位置，训练时为[0,1,2,3,4...]，推理时应该为[654]
        past_key_value: #kvcache
        use_cache:      #是否使用kvcache

    Raise:
        attn_output:    #注意力层的输出
        past_kv:        #包括了当前token的kvcache
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # hidden_states [b, seq_len, hidden_dim]
        bsz, q_len, _ = hidden_states.size()

        #### 获取q_nope、q_rope ####
        # WDQ矩阵压缩：hidden_size -> q_lora_rank(qnope和qrope)
        # WUQ矩阵和WQR矩阵解压：q_lora_rank(qnope和qrope) -> (q_nope_head_dim + q_rope_head_dim) * num_heads
        q = self.q_up_proj(self.q_down_layernorm(self.q_down_proj(hidden_states)))

        # 对q进行head分组
        # [b, seq_len, num_heads * qk_head_dim]
        # qk_head_dim=q_nope_head_dim + q_rope_head_dim
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        # [b, num_head, seq_len, qk_head_dim]

        # 将q_nope和q_rope分离
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        #获取旋转矩阵
        cos, sin = self.rotary_emb(q_rope)

        #对q_rope进行旋转位置编码
        #print("q_rope shape:", q_rope.shape)
        q_rope = apply_rotary_pos_emb_v2(q_rope, cos, sin, position_ids)

        #### 获取compressed_kv、k_rope ####
        # WDKV压缩矩阵和WKR压缩矩阵：hidden_size -> kv_lora_rank(knope，v) + k_rope_head_dim(krope)
        # WDKV矩阵：hidden_size -> kv_lora_rank(knope，v)
        # WKR矩阵： hidden_size -> k_rope_head_dim(krope)
        # 训练阶段，compressed_kv和k_rope是seq中每一个token对应一个
        # 推理阶段，compressed_kv和k_rope是当前token的，之前token的compressed_kv和k_rope需要在kvcache中取得
        compressed_kv = self.kv_down_proj(hidden_states)
        # 将c_kv, k_rope(未广播) 分离
        compressed_kv, k_rope = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.k_rope_head_dim],
            dim=-1,
        )
        # [b, seq_len, k_rope_head_dim]
        #对k_rope进行旋转位置编码
        #print("k_rope shape:", k_rope.shape)    #[b, seq_len, k_rope_head_dim]
        k_rope = k_rope.unsqueeze(1)#扩展head维度,
        #print("k_rope shape:", k_rope.shape)    #[b, seq_len, 1, k_rope_head_dim]
        k_rope = apply_rotary_pos_emb_v2(k_rope, cos, sin, position_ids)
        #print("k_rope shape:", k_rope.shape)
        k_rope = k_rope.squeeze(1)  #压缩掉head维度
        #print("k_rope shape:", k_rope.shape)    #[b, seq_len, k_rope_head_dim]

        # 在推理阶段，我们要将当前token的c_kv和位置编码后的k_rope放入cache中
        # 并且拿到kvcache中已经生成序列中所有token的compressed_kv和k_rope，也就是kvcache中的完整内容
        if past_key_value is not None:
            compressed_kv = torch.cat([past_key_value[0],compressed_kv], dim=1)
            k_rope = torch.cat([past_key_value[1],k_rope], dim=1)
        past_kv=(compressed_kv,k_rope) if use_cache else None

        # compressed_kv [b, kv_seq_len, kv_lora_rank]
        kv_seq_len = compressed_kv.size(1)

        # k_rope：[b, kv_seq_len, k_rope_head_dim]，扩展维度用于后面广播
        # 备注：这里是的 ke_rope 长度和原来不一样了，用的不是 seq_len, 而是 kv_seq_len
        k_rope = k_rope.view(bsz, kv_seq_len, 1, self.k_rope_head_dim).transpose(1, 2)
        # [b, 1, kv_seq_len, k_rope_head_dim]

        #！！！因为我们要做矩阵吸收，所以我们要把WUK矩阵和WUV矩阵分离出来，但是其实这里直接一开始就定义两个线性层也可以
        #WUK矩阵和WUV矩阵：kv_lora_rank -> num_heads * (k_nope_head_dim + v_head_dim)
        #WUK矩阵：kv_lora_rank(knope，v) -> num_heads * k_nope_head_dim(knope)
        #WUV矩阵：kv_lora_rank(knope，v) -> num_heads * v_head_dim(v)

        # kv_up_proj [num_heads * (k_nope_head_dim + v_head_dim), kv_lora_rank]
        kv_up_proj = self.kv_up_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        # kv_up_proj [num_heads, k_nope_head_dim + v_head_dim, kv_lora_rank]

        # q_absorb [num_heads, k_nope_head_dim, kv_lora_rank]
        q_absorb = kv_up_proj[:, : self.q_nope_head_dim, :]     #WUK矩阵
        # out_absorb [num_heads, v_head_dim, kv_lora_rank]
        out_absorb = kv_up_proj[:, self.q_nope_head_dim :, :]   #WUV矩阵


        #### 开始计算 ####
        #print("q_rope shape:", q_rope.shape)
        #print("k_rope shape:", k_rope.shape)
        #print("k_rope mT shape:", k_rope.mT.shape)
        #print("compressed_kv shape:", compressed_kv.shape)
        #print("q_nope shape:", q_nope.shape)
        #print("torch.matmul(q_rope, k_rope.mT) shape", torch.matmul(q_rope, k_rope.mT).shape)

        # q_nope = (q_C * W_UK)
        q_nope = torch.matmul(q_nope, q_absorb)

        # compressed_kv [b, kv_seq_len, kv_lora_rank]
        # attn_weights = ( (q_R * k_RT) + ( (q_C * W_UK) * c_KVT) ) / 根号 head_dim，请注意这里要用qk_head_dim，也就是q_rope_head_dim+q_nope_head_dim
        attn_weights = (
            torch.matmul(q_rope, k_rope.mT)
            + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)
        ) / math.sqrt(self.qk_head_dim)

        if attention_mask is not None:
            # causal mask #
            attn_weights = torch.masked_fill(
                attn_weights,
                attention_mask == 0,
                float('-inf')
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_nope.dtype)
        # attn_weights [bsz, num_heads, q_len, kv_seq_len]

        attn_weights = self.attn_dropout(attn_weights)

        # attention输出
        # Aweight * c_KV
        attn_output = torch.einsum("bhql,blc->bhqc", attn_weights, compressed_kv)

        # (Aweight * c_KV) * W_UVT
        attn_output = torch.matmul(
            attn_output, out_absorb.mT
        )  #torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)
        # h,q维度调换，并且拉平
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

        # ((Aweight * c_KV) * W_UVT) *WO
        attn_output = self.o_proj(attn_output)

        attn_output=self.resid_dropout(attn_output)

        return attn_output, past_kv


class MicroCortexMLP(nn.Module):
    def __init__(self, config:MicroCortexConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if config.intermediate_size is None:
            tmp = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((tmp + 63) // 64)
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        #dropout( Down( act( Gate( x ) )*Up( x ) ) )
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MicroCortexConfig):
        super().__init__()
        self.config = config
        #每个token的专家数
        self.num_experts_per_tok = config.num_experts_per_tok
        #会被路由到的专家数
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)

        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.num_experts_per_tok
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MicroCortexMOE(nn.Module):
    def __init__(self, config: MicroCortexConfig):
        super().__init__()
        self.config = config
        #定义专家
        self.experts = nn.ModuleList([
            MicroCortexMLP(config)
            for _ in range(config.n_routed_experts)
        ])
        #定义路由网络
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                #定义共享专家
                MicroCortexMLP(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self,
                hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, _ = hidden_states.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=torch.float16)
            #对每个专家进行循环
            for i, expert in enumerate(self.experts):
                #提取所有需要该专家处理的token，用该专家一次性进行处理，提升了处理性能
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            #乘以专家的路由权重
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        #如果有共享专家
        if self.config.n_shared_experts > 0:
            #在结果上加上共享专家对所有token的处理结果
            for expert in self.shared_experts:
                y = y + expert(identity)
        #加上辅助负载均衡损失
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self,
                  hidden_states,
                  flat_expert_indices,
                  flat_expert_weights):
        expert_cache = torch.zeros_like(hidden_states)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = hidden_states[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, hidden_states.shape[-1]), expert_out)

        return expert_cache


class MicroCortexBlock(nn.Module):
    def __init__(self, layer_id: int, config: MicroCortexConfig):
        super().__init__()
        self.num_heads = config.num_heads   #注意力头数
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads    #注意力头维度
        self.self_attn = MicroCortexMLA(config) #注意力计算层

        self.layer_id = layer_id    #层id
        self.input_layernorm = MicroCortexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MicroCortexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 使用MLP还是使用MOE
        self.mlp = MicroCortexMLP(config) if not config.use_moe else MicroCortexMOE(config)

    """
    Args:
        hidden_states:  [batch, seq_len, hidden_size]
        attention_mask: 输入数据的padding mask [batch, seq_len, hidden_size]
        position_ids:    输入数据的position_ids
        past_key_values:过去所有token的KVCache List[ Tuple[[batch, past_seq_len, kv_lora_rank], [batch, past_seq_len, k_rope_head_dim] ]
        use_cache:      是否使用cache

    Raises:
        hidden_states:  输入token最后输出的向量，训练时输出的是[batch, seq_len, hidden_size]，推理时输出的是[1, 1, hidden_size]
        presents_kv_cache: 当前token及其之前所有的kvcache
        aux_loss:   额外的正则化损失
    """
    def forward(self,
                hidden_states,
                position_ids=None,
                past_key_value=None,
                use_cache=False,
                attention_mask=None):
        residual = hidden_states
        # 经过一层mla后输出序列所有位置的hidden_states和新的present_key_value
        hidden_states, present_key_value = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states), position_ids=position_ids,
            past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MicroCortexModel(nn.Module):
    def __init__(self, config: MicroCortexConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MicroCortexBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = MicroCortexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    """
    Args:
        input_ids:      输入数据的token_ids [batch, seq_len, hidden_size]
        attention_mask: 输入数据的padding mask [batch, seq_len, hidden_size]
        past_key_values:过去所有token的KVCache List[ Tuple[[batch, past_seq_len, kv_lora_rank], [batch, past_seq_len, k_rope_head_dim] ]
        use_cache:      是否使用cache

    Raises:
        hidden_states:  输入token最后输出的向量，训练时输出的是[batch, seq_len, hidden_size]，推理时输出的是[1, 1, hidden_size]
        presents_kv_cache: 当前token及其之前所有的kvcache
        aux_loss:   额外的正则化损失
    """
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        # 如果没有past_key_values就初始化一个列表
        past_key_values = past_key_values or [None] * len(self.layers)

        #计算past_seq_len，有kvcache就是从kvcache中获取，没有就是0
        if past_key_values and past_key_values[0] is not None:
            past_seq_len = past_key_values[0][0].size(1)
        else:
            past_seq_len=0

        #构造position_ids，从start到start+seq_length
        position_ids = torch.arange(
            past_seq_len,
            past_seq_len + seq_length,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 用来存储当前的kvcache
        presents_kv_cache = []# List[ Tuple[[batch, past_seq_len, kv_lora_rank], [batch, past_seq_len, k_rope_head_dim] ]

        # 传递每个层
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states=hidden_states,  #[batch, seq_len, hidden_size]
                position_ids=position_ids,
                past_key_value=past_key_value,  #Tuple[[batch, past_seq_len, kv_lora_rank], [batch, past_seq_len, k_rope_head_dim]
                use_cache=use_cache,
                attention_mask=attention_mask   #[batch, seq_len, hidden_size]
            )
            presents_kv_cache.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MicroCortexMOE)
        )

        return hidden_states, presents_kv_cache, aux_loss


class MicroCortexForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MicroCortexConfig

    def __init__(self, config: MicroCortexConfig = None):
        self.config = config or MicroCortexConfig()
        super().__init__(self.config)
        self.model = MicroCortexModel(self.config)
        # 最终的分类头层
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # tie weight，将embedding的参数和lm_head参数共享，可以减少显存
        self.model.embed_tokens.weight = self.lm_head.weight
        #封装模型输出的标准输出容器
        self.OUT = CausalLMOutputWithPast()

    """
    Args:
        input_ids:      输入数据的token_ids [batch, seq_len, hidden_size]
        attention_mask: 输入数据的padding mask [batch, seq_len, hidden_size]
        past_key_values:过去所有token的KVCache List[ Tuple[[batch, past_seq_len, kv_lora_rank], [batch, past_seq_len, k_rope_head_dim] ]
        use_cache:      是否使用cache
        logits_to_keep:

    Raises:
        OUT:  记录这最后一层输出向量、logits(未归一化输出)、正则化损失、kvcache
    """
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        # 直接传入模型中，获取最后一层的输出
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 控制否是只计算最后几个token的logits，logits_to_keep表示保留几个时间步，我们这里就是用最后一个时间步的hidden_states做预测，logits_to_keep 为1
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        # 将最后一层输出放入标准输出容器中，并且进行返回，可以就当成一个字典，但是可以兼容huggingface的模型调用与生成流程
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT