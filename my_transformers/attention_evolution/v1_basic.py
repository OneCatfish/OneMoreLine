import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out_linear(output)