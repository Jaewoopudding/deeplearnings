from functools import partial

from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean") # o: Channel
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False)) 
        normed_weight = (weight - mean) / var

        return F.conv2d(
            x, 
            normed_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )