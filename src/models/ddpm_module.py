from typing import Any
from tqdm import tqdm

import torch
import torch.nn.functional as F

from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

from src.models.components.diffusion_scheduler import DiffusionScheduler


class DDPMModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 diffusion_scheduler: DiffusionScheduler,
                 timesteps: int,
                 img_size: int
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activation
        self.net = net # GPU 마법같이 . 내부 원리가 설명되어 있지 않아. 
        self.criterion = torch.nn.CrossEntropyLoss()

        self.img_size = img_size
        self.timesteps = timesteps
        self.diffusion_scheduler = diffusion_scheduler

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
        t = torch.randint(0, self.timesteps, (x.shape[0],)).long().type_as(x)
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x=x, t=t, noise=noise)
        predicted_noise = self.forward(x_noisy, t)
        loss = F.smooth_l1_loss(noise, predicted_noise)

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
        self.log_image()
        
    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True), 
        
    def test_step(self, batch, batch_index: int):
        loss = self.model_step(batch)
        self.test_loss(loss)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_test_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        optimizer=self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler(optimizer=optimizer):
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer":optimizer,
            }
        return {"optimizer": optimizer}
    
    

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.long())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x, t, noise=None): 
        if noise is None:
            noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.extract(self.diffusion_scheduler.sqrt_alphas_cumprod.type_as(x) , t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.diffusion_scheduler.sqrt_one_minus_alphas_cumprod.type_as(x), t, x.shape)
        return sqrt_alphas_cumprod_t.to(x.device) * x + sqrt_one_minus_alphas_cumprod_t.to(noise.device) * noise
    
    @torch.no_grad()
    def p_sample(self, x, t,  t_index):
        betas_t = self.extract(self.diffusion_scheduler.betas.type_as(x), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.diffusion_scheduler.sqrt_one_minus_alphas_cumprod.type_as(x), t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.diffusion_scheduler.sqrt_recip_alphas.type_as(x), t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.net(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.diffusion_scheduler.posterior_variance.type_as(x), t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
    @torch.no_grad()
    def p_sample_loop(self, shape):
        b = shape[0]
        img = torch.randn(shape).to(self.device)
        imgs = []
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((b,), i, dtype=torch.long).to(self.device), i)
            imgs.append(img.cpu().numpy())
        return imgs

    def sample(self, image_size, batch_size, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))
    

    def log_image(self):
        import wandb
        sampled_images = self.sample(self.img_size, 32, channels=3)
        self.logger.experiment.log({
            "sampled_images": [wandb.Image((img.transpose(1,2,0)) / 2 + 0.5) for img in sampled_images[-1]]
        })



    
    
if __name__ == "__main__":
    pass    
        