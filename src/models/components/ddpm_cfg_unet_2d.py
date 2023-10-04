import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nn.attention import Attention, LinearAttention

def one_param(m):
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_model_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
        
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, conn, t):
        x = self.up(x)
        x = torch.cat([x, conn], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, block_count, init_depth=64, depth_multiplier=2, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim

        self.encoding_conv = DoubleConv(in_channels, init_depth)
        self.downs = nn.ModuleList([])
        self.midconv1 = DoubleConv(init_depth * depth_multiplier ** block_count, init_depth * depth_multiplier ** block_count)
        self.midattn = LinearAttention(init_depth * depth_multiplier ** block_count)
        self.midconv2 = DoubleConv(init_depth * depth_multiplier ** block_count, init_depth * depth_multiplier ** block_count)
        self.ups = nn.ModuleList([])
        self.decoding_conv = DoubleConv(init_depth, out_channels)

        for idx in range(block_count):
            inp = depth_multiplier ** (idx) * (init_depth)
            out = depth_multiplier ** (idx + 1) * (init_depth)
            self.downs.append(
                nn.ModuleList([
                    Down(inp, out),
                    LinearAttention(out)
                ])
            )

            self.ups.insert(0, 
                nn.ModuleList([
                    Up(int(out * 1.5), inp),
                    LinearAttention(inp)
                ])
            )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = self.pos_encoding(t.unsqueeze(-1), self.time_dim)
        return self.unet_forward(x, t)
        
    def unet_forward(self, x, t):
        h = []
        x = self.encoding_conv(x)
        for conv, attn in self.downs:
            h.append(x)
            x = conv(x, t)
            x = attn(x)

        x = self.midconv1(x)
        x = self.midattn(x)
        x = self.midconv2(x) 

        for conv, attn in self.ups:
            x = conv(x, h.pop(), t)
            x = attn(x)
        x = self.decoding_conv(x)
        return x


class ConditionedUnet(Unet):
    def __init__(self, in_channels, out_channels, block_count, num_classes=None,
                 init_depth=64, depth_multiplier=2, time_dim=256, remove_deep_conv=False,
                 **kwargs):
        super().__init__(in_channels, out_channels, block_count, init_depth, depth_multiplier, time_dim, remove_deep_conv)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
    
    def forward(self, x, t, y=None):
        t = self.pos_encoding(t.unsqueeze(-1), self.time_dim)
        if y is not None:
            emb = self.label_emb(y)
        if y is None:
            y = torch.zeros(x.shape[0]).long().to(x.device)
            emb = self.label_emb(y)
            emb[:] =0.
        t += emb 
        return self.unet_forward(x, t)