import torch
import torch.nn as nn
import math

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, max_len=512):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.max_len = max_len
        
        # 初始化相对位置编码矩阵
        self.relative_pos_emb = nn.Parameter(torch.randn(max_len * 2, self.d_k))
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 生成相对位置索引
        range_vec = torch.arange(seq_len)
        relative_pos = range_vec[None, :] - range_vec[:, None]
        relative_pos += self.max_len  # 转换为正索引
        
        # 投影查询、键、值
        q = self._split_heads(self.q_linear(q))
        k = self._split_heads(self.k_linear(k))
        v = self._split_heads(self.v_linear(v))
        
        # 计算相对位置注意力分数
        rel_emb = self.relative_pos_emb[relative_pos]
        content_score = torch.matmul(q, k.transpose(-2, -1))
        pos_score = torch.matmul(q, rel_emb.transpose(-2, -1))
        scores = (content_score + pos_score) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = self._combine_heads(output)
        return self.out_linear(output)

    def _split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)