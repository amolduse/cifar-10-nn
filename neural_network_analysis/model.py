
import torch.nn as nn
import torch.nn.functional as F
from conv_block import ConvBlock

# The model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Input Block
        self.block1 = ConvBlock(32, 64, num_convs=2, use_depthwise_separable = True)
        self.block2 = ConvBlock(64, 128, num_convs=2, use_depthwise_separable = True)
        self.block3 = ConvBlock(128, 144, num_convs=2, use_depthwise_separable = True, dilation = 2)
        self.block4 = ConvBlock(144, 144, num_convs=2, use_depthwise_separable = True, dilation = 2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))        
        self.dropout_fc = nn.Dropout(0.2)
        self.fc = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)        
        x = self.gap(x)        
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        x = self.fc(x)

        #x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
