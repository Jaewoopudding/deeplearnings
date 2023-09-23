import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Embedding(nn.Module):
    def __init__(self, 
                 input_channel,
                 img_size,
                 patch_size,
                 dim):
        assert(img_size%patch_size == 0)
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_count = int(self.img_size / self.patch_size)

        self.projection_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, 
                      out_channels=dim, 
                      kernel_size=patch_size, 
                      stride=patch_size),
            Rearrange('b c w d -> b (w d) c')
        )

        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_count**2+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection_layer(x)
        cls_tokens = repeat(self.cls_token, 'c d -> b c d', b=b)
        return torch.cat([x, cls_tokens], dim=1) + self.positional_embedding
    

class ViTMultiHeadAttention(nn.Module):
    def __init__(self,
                 num_of_heads,
                 dim,
                 dropout
                 ):
        assert(dim % num_of_heads == 0)
        super().__init__()

        self.num_of_heads = num_of_heads
        self.dim = dim
        self.to_qkv = nn.Linear(dim, dim*3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b t (k h d) -> k b h t d', k=3, h=self.num_of_heads, d=int(self.dim//self.num_of_heads)) # (qkv), batch, num of head, token num, dim
        qk = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attention = torch.softmax(qk / self.dim ** (0.5), dim=-1)
        output = torch.einsum('b h t i, b h i d -> b h t d', attention, v)
        output = rearrange(output, 'b h t d -> b t (h d)')
        return output
    

class ViTLinear(nn.Module):
    def __init__(self, 
                 dim,
                 expansion,
                 dropout):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(dim, dim*expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*expansion, dim)
        )
    
    def forward(self, x):
        return self.layer(x)
    

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_of_heads,
                 dropout,
                 expansion):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

        self.mha = ViTMultiHeadAttention(
            num_of_heads=num_of_heads,
            dim=dim,
            dropout=dropout
        )

        self.linear = ViTLinear(
            dim=dim,
            expansion=expansion,
            dropout=dropout
        )

    def forward(self, x):
        tmp = self.layer_norm(x)
        tmp = self.mha(tmp)
        x = x + tmp
        tmp = self.layer_norm(x)
        tmp = self.linear(x)
        return x + tmp
    
class TransformerEncoder(nn.Sequential):
    def __init__(self, 
                 depth,
                 **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ViTClassifier(nn.Module):
    def __init__(self, 
                 dim, 
                 output_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_size)
        )
    
    def forward(self, x):
        x = x.mean(dim=1)
        return self.layer(x)
    

class ViT(nn.Module):
    def __init__(self, 
                 input_channel: int,
                 img_size: int,
                 patch_size: int,
                 dim: int,
                 num_of_heads: int,
                 dropout: float,
                 expansion: int,
                 depth: int,
                 output_size: int):
        super().__init__()
        
        self.embedding = Embedding(input_channel=input_channel,
            img_size=img_size,
            patch_size=patch_size,
            dim=dim
        )
        
        self.encoder = TransformerEncoder(
            depth=depth,
            dim=dim,
            num_of_heads=num_of_heads,
            dropout=dropout,
            expansion=expansion
        )
        
        self.classifier = ViTClassifier(
            dim=dim,
            output_size=output_size
        )

    def forward(self, x):
        return self.classifier(self.encoder(self.embedding(x)))
