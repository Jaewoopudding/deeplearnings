from torch import nn
import torch


class InceptionBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 conv11_channel: int,
                 conv33_channel: int,
                 conv55_channel: int,
                 pool_conv_channel: int):
        super().__init__()
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channel, conv11_channel, 1, 1, 0),
            nn.BatchNorm2d(conv11_channel),
            nn.SiLU()
        )
        
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channel, int(conv33_channel/4*3), 1, 1, 0),
            nn.BatchNorm2d(int(conv33_channel/4*3)),
            nn.SiLU(),
            nn.Conv2d(int(conv33_channel/4*3), conv33_channel, 3, 1, 1),
            nn.BatchNorm2d(conv33_channel),
            nn.SiLU()
        )
        
        self.conv55 = nn.Sequential(
            nn.Conv2d(in_channel, int(conv55_channel/4), 1, 1, 0),
            nn.BatchNorm2d(int(conv55_channel/4)),
            nn.SiLU(),
            nn.Conv2d(int(conv55_channel/4), conv55_channel, 5, 1, 2),
            nn.BatchNorm2d(conv55_channel),
            nn.SiLU()
        )
        
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channel, pool_conv_channel, 1, 1, 0),
            nn.BatchNorm2d(pool_conv_channel),
            nn.SiLU()
        )
        
    def forward(self, x):
        # conv11, conv33, conv55, pool_conv = self.conv11(x), self.conv33(x), self.conv55(x), self.pool_conv(x)
        conv11 = self.conv11(x)
        conv33 = self.conv33(x)
        conv55 = self.conv55(x)
        pool_conv = self.pool_conv(x)
        return torch.cat([conv11, conv33, conv55, pool_conv], dim=1)
    

class GoogleNet(nn.Module):
    def __init__(self, input_channel=3, output_size=10):
        super().__init__()
        
        self.model = nn.Sequential(
            InceptionBlock(in_channel=input_channel, conv11_channel=16*2, conv33_channel=32*2, conv55_channel=16*2, pool_conv_channel=16*2),
            InceptionBlock(in_channel=80*2, conv11_channel=16*2, conv33_channel=32*2, conv55_channel=16*2, pool_conv_channel=16*2),
            nn.MaxPool2d(2), # 16
            InceptionBlock(in_channel=80*2, conv11_channel=48*2, conv33_channel=32*2, conv55_channel=16*2, pool_conv_channel=32*2),
            InceptionBlock(in_channel=128*2, conv11_channel=48*2, conv33_channel=32*2, conv55_channel=16*2, pool_conv_channel=32*2),
            nn.MaxPool2d(2), # 8
            InceptionBlock(in_channel=128*2, conv11_channel=64*2, conv33_channel=48*2, conv55_channel=32*2, pool_conv_channel=32*2),
            InceptionBlock(in_channel=176*2, conv11_channel=64*2, conv33_channel=48*2, conv55_channel=32*2, pool_conv_channel=32*2),
            nn.MaxPool2d(2), # 4
            InceptionBlock(in_channel=176*2, conv11_channel=64*2, conv33_channel=48*2, conv55_channel=32*2, pool_conv_channel=64*2),
            InceptionBlock(in_channel=208*2, conv11_channel=64*2, conv33_channel=48*2, conv55_channel=32*2, pool_conv_channel=64*2),
            nn.MaxPool2d(2) # 2
        )
        
        self.classifier = nn.Linear(416, output_size)
        
    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=(-1, -2))
        return self.classifier(x)
            
    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        


if __name__ == '__main__':
    _ = GoogleNet(10)