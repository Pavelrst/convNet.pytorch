import torch.nn as nn
from torch.functional import F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd

from utils.mixup import MixUp

# Targeted dropout imports
from targetedDropout import targeted_unit_dropout
from targetedDropout import targeted_weight_dropout
from targetedDropout import ramping_targeted_unit_dropout
from targetedDropout import ramping_targeted_weight_dropout
# end imports


class Conv2d_with_td(_ConvNd):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', dropout_fn=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_with_td, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.dropout_fn = dropout_fn

    def forward(self, input):
        if self.dropout_fn is not None:
            self.weight = self.dropout_fn.forward(self.weight, self.training)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)