import torch
import torch.nn as nn
import math

# 忽略了 attention_mask, attention_dropout; 
# 假设nums_head为8，nums_key_value_head为4
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0 # 可以整除
        assert nums_head % nums_key_value_head == 0  # N 个 query head 为一组

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        # 初始化 qkv o
        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)  # out feature_size (8 * head_dim) 
        # k v out shape (4 * head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim) # input_size nums_head * head_dim

    def forward(self, X, attention_mask=None):
        # X shape (batch, seq, hidden_dim)
        batch_size, seq, _ = X.size()

        # qkv projection
        q = self.q_proj(X)  # （batch, seq, hidden_dim)
        k = self.k_proj(X)
        v = self.v_proj(X) 

        # attention_weight 目标shape 是 (batch, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # 关注: nums_head 和 nums_key_value_head 的关系
        q = q.transpose(1, 2) # (b, nums_head, seq, head_dim)
        k = k.transpose(1, 2) # (b, nums_key_value_head, seq, head_dim)
        v = v.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)

        # k v repeat； （广播操作）8/4=2
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        attention_weight = torch.softmax(attention_score, dim=-1)
        # （attention_mask 忽略） # 可以看前面的视频

        output = attention_weight @ v  # (b, nums_head, seq, head_dim)

        # output projection 变成 (b, seq, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        final_output = self.o_proj(output.view(batch_size, seq, -1))

        return final_output

# 测试
x = torch.rand(3, 2, 128)
net = GroupQueryAttention(128, 8, 4)
net(x).shape