import torch
import math
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self,hidden_dim:int = 512) -> None:
        super().__init__()
        self.hidden_dim=hidden_dim

        self.query_proj=nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,hidden_states):# hidden_states [batch_size, seq_len, hidden_dim]

        q=self.query_proj(hidden_states)# q [batch_size, seq_len, hidden_dim]
        k = self.key_proj(hidden_states)# k [batch_size, seq_len, hidden_dim]
        v = self.value_proj(hidden_states)# v [batch_size, seq_len, hidden_dim]

        attention_value=torch.matmul(# attention_value [batch_size, seq_len, seq_len]
            q,k.transpose(-1,-2)# k 转置后 [batch_size, hidden_dim, seq_len]
        )

        attention_weight=torch.softmax(# attention_weight [batch_size, seq_len, seq_len]
            attention_value/math.sqrt(self.hidden_dim),
            dim=-1
        )

        print(attention_weight)

        # output [batch_size, seq_len, hidden_dim]
        output=torch.matmul(attention_weight.v)

        return output

x=torch.rand(3,2,4)
self_att_net=SelfAttentionV1(4)
self_att_net(x)


#效率优化
class SelfAttentionV2(nn.Module):
    def __init__(self,hidden_dim:int = 512) -> None:
        super().__init__()
        self.hidden_dim=hidden_dim

        self.proj=nn.Linear(hidden_dim,hidden_dim*3)

    def forward(self, hidden_states):  # hidden_states [batch_size, seq_len, hidden_dim]
        qkv=self.proj(hidden_states) # qkv [batch_size, seq_len, hidden_dim*3]
        q,k,v=torch.split(qkv,self.hidden_dim,dim=-1)

        attention_weight = torch.softmax(# attention_weight [batch_size, seq_len, seq_len]
            torch.matmul(  # attention_value [batch_size, seq_len, seq_len]
                q, k.transpose(-1, -2)  # k 转置后 [batch_size, hidden_dim, seq_len]
            )/math.sqrt(self.hidden_dim)
        )

        print(attention_weight)

        # output [batch_size, seq_len, hidden_dim]
        output=attention_weight@v

        return output

x=torch.rand(3,2,4)
self_att_net=SelfAttentionV2(4)
self_att_net(x)

#加入dropout、attention_mask、output线性变换
class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout_rate=0.1, *args, **kwargs) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_dropout=nn.Dropout(dropout_rate)

        self.output_proj=nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None):  # hidden_states [batch_size, seq_len, hidden_dim]
        qkv = self.proj(hidden_states)  # qkv [batch_size, seq_len, hidden_dim*3]
        q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)

        attention_value=q@k.transpos(-1,-2)/math.sqrt(self.hidden_dim)

        if attention_mask is not None:
            attention_value=attention_value.marked_fill(
                attention_mask==0,
                float("-1e20")
            )

        attention_weight=torch.softmax(attention_value,dim=-1)

        attention_weight=self.attention_dropout(attention_weight)
        print(attention_weight)

        # output [batch_size, seq_len, hidden_dim]
        output = attention_weight @ v

        output=self.output_proj(output)

        return output

x=torch.rand(3,4,2)
# mask [batch_size, seq_len, seq_len] [3,4,4]
mask=torch.tensor(
    [
        [1,1,1,0],
        [1,1,0,0],
        [1,0,0,0]
    ]
)
mask=mask.unsqueeze(dim=1).repeat(1,4,1)

self_att_net=SelfAttentionV3(2,mask)
self_att_net(x)

#面试写法
class SelfAttentionInterview(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout_rate=0.1) -> None:
        super().__init__()
        self.hidden_dim=hidden_dim

        self.query_proj=nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,hidden_states,attention_mask=None):# hidden_states [batch_size, seq_len, hidden_dim]

        q = self.query_proj(hidden_states)  # q [batch_size, seq_len, hidden_dim]
        k = self.key_proj(hidden_states)  # k [batch_size, seq_len, hidden_dim]
        v = self.value_proj(hidden_states)  # v [batch_size, seq_len, hidden_dim]

        attention_value = q @ k.transpos(-1, -2) / math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            attention_value=attention_value.marked_fill(
                attention_mask==0,
                float("-inf")
            )

        attention_weight = torch.softmax(attention_value, dim=-1)

        attention_weight = self.attention_dropout(attention_weight)

        # output [batch_size, seq_len, hidden_dim]
        output = attention_weight @ v

        output = self.output_proj(output)

        return output

x=torch.rand(3,4,2)
# mask [batch_size, seq_len, seq_len] [3,4,4]
mask=torch.tensor(
    [
        [1,1,1,0],
        [1,1,0,0],
        [1,0,0,0]
    ]
)
mask=mask.unsqueeze(dim=1).repeat(1,4,1)

self_att_net=SelfAttentionV3(2,mask)
self_att_net(x)