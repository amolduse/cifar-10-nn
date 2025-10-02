import torch.nn as nn
from depthwise_separable_conv import DepthwiseSeparableConv

# Define a block with multiple conv layers
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, kernel_size=3, padding = 1, stride = 1, pool=True, dilation = 1, use_depthwise_separable=False):
        super(ConvBlock, self).__init__()
        layers = []
        dropout_value = 0.1
        for i in range(num_convs):
            if use_depthwise_separable:
              layers.append(DepthwiseSeparableConv(
                  in_channels if i == 0 else out_channels,  # first layer uses in_channels
                  out_channels,
                  kernel_size = kernel_size,
                  stride = stride,
                  padding = padding
              ))
            else:
              layers.append(nn.Conv2d(
                  in_channels if i == 0 else out_channels,  # first layer uses in_channels
                  out_channels,
                  kernel_size = kernel_size,
                  padding = dilation if i == (num_convs - 1) else padding,
                  dilation = dilation if i == (num_convs - 1) else 1
              ))
            if use_depthwise_separable == False:
              layers.append(nn.ReLU(inplace=True))
              layers.append(nn.BatchNorm2d(out_channels))
            
            if(i < num_convs - 1):
              layers.append(nn.Dropout(dropout_value))

        if pool:
            if use_depthwise_separable:
              layers.append(DepthwiseSeparableConv(
                  out_channels,
                  out_channels,                  
                  stride = 2,
                  padding = padding
              ))
            else:
              layers.append(nn.Conv2d(
                  out_channels,  # first layer uses in_channels
                  out_channels,                  
                  kernel_size = 3,
                  padding=0,
                  stride=2
              ))  # downsampling

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)