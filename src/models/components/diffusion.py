import torch

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, 
                img_size=256, num_classes=10, in_channels=3, out_channels=3, 
                noise_schedule='linear', **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.beta = self.prepare_noise_schedule(noise_schedule)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, noise_schedule):
        s = 0.008
        if noise_schedule == 'cosine':
            steps = self.noise_steps + 1
            x = torch.linspace(0, self.noise_steps, steps)
            alpha_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif noise_schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif noise_schedule == 'quadratic':
            return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.noise_steps) ** 2
        elif noise_schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, self.noise_steps)
            return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        raise(ValueError("scheduling string must be one of 'cosin', 'linear', 'quadratic', 'sigmoid'. "))
    





