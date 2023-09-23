from typing import List

from torch import nn
import torch


class DenseLayer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 growth_rate: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.SiLU(),
            nn.Conv2d(input_dim, 4 * growth_rate, 1, 1, 0),
            nn.BatchNorm2d(4 * growth_rate),
            nn.SiLU(),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1)
        )
        
    def forward(self, x):
        z = self.model(x)
        x = torch.cat([x, z], dim=1)
        return x
    
    
class DenseBlock(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 num_of_layers: int,
                 growth_rate: int):
        super().__init__()
        self.input_dim = input_dim
        
        self.model = nn.Sequential()
        for i in range(num_of_layers):
            self.model.add_module(
                f'dense_layer_{i+1}', DenseLayer(self.input_dim, growth_rate)
            )
            self.input_dim = self.input_dim + growth_rate
            
    def forward(self, x):
        x = self.model(x)
        return x
    

class TransitionLayer(nn.Module):
    def __init__(self, 
                 input_dim: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.SiLU(),
            nn.Conv2d(input_dim, input_dim//2, 1, 1, 0),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
class DenseNet(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 growth_rate: int,
                 num_of_layers: List[int],
                 output_size: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, 1, 1))
        
        for i, layers in enumerate(num_of_layers):
            self.model.add_module(f'dense_block_{i+1}', DenseBlock(self.hidden_dim,
                                                                   layers,
                                                                   growth_rate))
            self.hidden_dim = self.hidden_dim + layers * growth_rate
            if i != (len(num_of_layers) - 1):
                self.model.add_module(f'transition_{i+1}', TransitionLayer(self.hidden_dim))
                self.hidden_dim = self.hidden_dim // 2
        
        self.final_norm_and_activ = nn.Sequential(
            nn.BatchNorm2d(self.hidden_dim),
            nn.SiLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, output_size)
            )

    def forward(self, x):
        x = self.model(x)
        x = self.final_norm_and_activ(x)
        x = x.mean(dim=(-1, -2))
        return self.classifier(x)
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
