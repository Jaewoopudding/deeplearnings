from functools import partial

from torch import nn
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (k h c) x y -> k b h c (x y)", h=self.heads, k=3) # qkv 분할 / 배치 / 헤드 / 채널 / 각 픽셀값이 token이 된다. 
        attention_map = torch.einsum("b h c x, b h c y -> b h x y", q * self.scale, k).softmax(dim=-1) # 각 픽셀 별 attention이 진행되었음.
        output = torch.einsum('b h x y, b h c y -> b h x c', attention_map, v)
        output = rearrange(output, 'b h (x y) c -> b (h c) x y', x=h, y=w)
        output = self.to_out(output)
        return output
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (k h c) x y -> k b h c (x y)", h=self.heads, k=3)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)