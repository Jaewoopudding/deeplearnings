from typing import Any
import copy

from fastprogress import progress_bar
from tqdm import tqdm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import wandb

from src.models.components.diffusion import Diffusion


class DDPM_CFG_Module(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 diffusion: Diffusion,
                 cfg_scale: float,
                 label_drop_rate: float,
                 num_classes: int,
                 in_channels: int,
                 img_size: int,
                 noise_steps: int,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activation
        self.net = net 
        self.diffusion = diffusion

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        
    def forward(self, x, t):
        return self.net(x, t)
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        
    def model_step(self, batch: Any):
        x, y = batch
        t = self.sample_timestep(x.shape[0])
        x_t, noise = self.get_noise_image(x, t)
        if torch.rand(1) < self.hparams.label_drop_rate:
            y = None
        predicted_noise = self.net(x_t, t, y)
        loss = F.mse_loss(noise, predicted_noise)
        return loss


    def training_step(self, batch: Any, batch_idx):
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.val_loss(loss)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        
    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        self.log_image()
        
    def test_step(self, batch, batch_index: int):
        loss = self.model_step(batch)
        self.test_loss(loss)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer=self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler(optimizer=optimizer):
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer":optimizer,
            }
        return {"optimizer": optimizer}


    def sample_timestep(self, n):
        return torch.randint(low=1, high=self.hparams.noise_steps, size=(n,)).to(self.device)
    

    def get_noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.diffusion.alpha_hat.to(self.device)[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.diffusion.alpha_hat.to(self.device)[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return (sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon).to(self.device), epsilon

    
    @torch.inference_mode()
    def sample_image(self, labels):
        n = len(labels)
        self.net.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.hparams.in_channels, self.hparams.img_size, self.hparams.img_size)).to(self.device)
            for i in reversed(range(1, self.hparams.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = self.net(x, t, labels)
                if self.hparams.cfg_scale > 0:
                    uncond_predicted_noise = self.net(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, self.hparams.cfg_scale)
                alpha = self.diffusion.alpha.to(self.device)[t][:, None, None, None]
                alpha_hat = self.diffusion.alpha_hat.to(self.device)[t][:, None, None, None]
                beta = self.diffusion.beta.to(self.device)[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1 / 2)
        return x


    def log_image(self):
        sampled_images = self.sample_image(labels=torch.tensor(list(range(self.hparams.num_classes))).to(self.device))
        self.logger.experiment.log({
            "sampled_images": [wandb.Image(img) for img in sampled_images]
        })



    
    
if __name__ == "__main__":
    pass    
        