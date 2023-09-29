from typing import Any
from tqdm import tqdm

import torch
import torch.nn.functional as F


class DiffusionScheduler:
    def __init__(self,
                 timesteps: int,
                 std_schedule: str):

        self.timesteps = timesteps
        self.betas = self.scheduler_mapping(std_schedule)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def scheduler_mapping(self, schedule_string):
        beta_start = 0.0001
        beta_end = 0.02
        s = 0.008

        if schedule_string == 'cosine':
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alpha_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        elif schedule_string == 'linear':
            return torch.linspace(beta_start, beta_end, self.timesteps)

        elif schedule_string == 'quadratic':
            return torch.linspace(beta_start**0.5, beta_end**0.5, self.timesteps) ** 2

        elif schedule_string == 'sigmoid':
            betas = torch.linspace(-6, 6, self.timesteps)
            return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        
        else:
            raise(ValueError("scheduling string must be one of 'cosin', 'linear', 'quadratic', 'sigmoid'. "))
