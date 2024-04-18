# -*- coding: utf-8 -*-
import importlib
import re
import torch
import torch.nn as nn

class ConvolutionBatchNorm(nn.Module):
    def __init__(self, channels, filters, kernel_size, stride=1, dilation=1):
        super(ConvolutionBatchNorm, self).__init__()

        padding = kernel_size // 2
        padding += padding * (dilation-1)

        self.conv = nn.Conv1d(
            channels,
            filters,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )

        self.bn = nn.BatchNorm1d(filters)

    def forward(self, x):
        return self.bn(self.conv(x))

