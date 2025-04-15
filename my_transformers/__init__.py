# 注意力机制演进实现模块

from .attention_evolution.v1_basic import MultiHeadAttention
from .attention_evolution.v2_transformer_xl import RelativeMultiHeadAttention

__all__ = [
    'MultiHeadAttention',
    'RelativeMultiHeadAttention'
]