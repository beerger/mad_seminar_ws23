import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor

def binary_max_pooling(mask, kernel_size, stride, padding):
    # Use max pooling on the mask, which will return 1 if any of the input within the pooling window is 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=stride, padding=padding)

class GlobalNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Image size is 256 x 256 x 3, outputs feature map size 1 x 1 x 128

        self.partial_conv0 = PartialConv2d(in_channels=3, out_channels=16, 
                                           kernel_size=7, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.partial_conv1_1 = PartialConv2d(in_channels=16, out_channels=32,
                                              kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()

        self.partial_conv1_2 = PartialConv2d(in_channels=32, out_channels=32, 
                                             kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()

        self.partial_conv2_1 = PartialConv2d(in_channels=32, out_channels=64, 
                                             kernel_size=3, stride=2, padding=0)
        self.relu4 = nn.ReLU()

        self.partial_conv2_2 = PartialConv2d(in_channels=64, out_channels=64, 
                                             kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()

        self.partial_conv3_1 = PartialConv2d(in_channels=64, out_channels=128, 
                                             kernel_size=3, stride=2, padding=0)
        self.relu6 = nn.ReLU()

        self.partial_conv3_2 = PartialConv2d(in_channels=128, out_channels=128, 
                                             kernel_size=3, stride=1, padding=0)
        self.relu7 = nn.ReLU()

        self.partial_conv4 = PartialConv2d(in_channels=128, out_channels=128, 
                                             kernel_size=3, stride=1, padding=0)

    def forward(self, x : Tensor, mask):
        
        x, mask = self.partial_conv0(x, mask)
        x = self.relu1(x)

        x = self.max_pool(x)
        # Update mask manually since pooling doesn't do this
        mask = binary_max_pooling(mask, kernel_size=3, stride=2, padding=0)

        x, mask = self.partial_conv1_1(x, mask)
        x = self.relu2(x)

        x, mask = self.partial_conv1_2(x, mask)
        x = self.relu3(x)

        x, mask = self.partial_conv2_1(x, mask)
        x = self.relu4(x)

        x, mask = self.partial_conv2_2(x, mask)
        x = self.relu5(x)

        x, mask = self.partial_conv3_1(x, mask)
        x = self.relu6(x)

        x, mask = self.partial_conv3_2(x, mask)
        x = self.relu7(x)

        x = self.partial_conv4(x, mask)

        return x


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################



class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output