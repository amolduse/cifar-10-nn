"""
Target:
    1. Improve the architecture of previously trained network by adding following features one by one and observing the impact- 
        a. Add max pooling at correct location
        b. Add image augmentation
        c. Add schedulers / step LR
    2. Keep parameters under 8k and achieve accuracy of 99.4% on test set consistently.
Result:
    1. Parameters: 
        a. Max pooling after receptive field of 5x5 (after 2 conv layers) - 8k
        b. After image augmentation and some parameter changing 7.1k
        c. Added more image augmentation to increase data size so that model can learn better and not underfit. - 7.1k
        d. By adding schedular with StepLR 7.1k
    2. Best Train Accuracy: 
        a. Max pooling after receptive field of 5x5 (after 2 conv layers) -  98.59
        b. After image augmentation and some parameter changing 7.1k 98.69
        c. Added more image augmentation to increase data size so that model can learn better and not underfit. - 98.33
        d. By adding schedular with StepLR 99.01
    3. Best Test Accuracy:
        a. Max pooling after receptive field of 5x5 (after 2 conv layers) -  99.27
        b. After image augmentation and some parameter changing - 99.29
        c. Added more image augmentation to increase data size so that model can learn better and not underfit. - 99.34
        d. By adding schedular with StepLR 99.43

Analysis:
    1. After adding max pooling after 2 conv layers (receptive field of 5x5) accuracy but gap between train and test accuracy didn't improve.
       The network was not overfitting but it was underfitting. So, I added image augmentation to increase the data size.
    2. After adding image augmentation accuracy improved but still there was gap between train and test accuracy and didn't achieve the desired result.
    3. As network was not learning in lower parameters I increased the dataset by adding more image augmentation 
       (RandomAffine to shift image horizontally or vertically by few pixels) which gave promissing result of test accuracy of 99.34%.
    4. Finally after adding schedular with StepLR I achieved the desired result of 99.43% test accuracy.

Summary:
    Tried three models with different architecture and parameters. The final model is defined in model3.py which has 7.1k parameters and achieves
    99.43% accuracy on test set and 99.01% accuracy on train set. 
    In this excersise I learnt how important it is to choose the right architecture and parameters to achieve the desired result.


"""
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
    
"""
Logs:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,440
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9            [-1, 8, 24, 24]             128
        MaxPool2d-10            [-1, 8, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]             864
             ReLU-12           [-1, 12, 10, 10]               0
      BatchNorm2d-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 12, 8, 8]           1,296
             ReLU-16             [-1, 12, 8, 8]               0
      BatchNorm2d-17             [-1, 12, 8, 8]              24
          Dropout-18             [-1, 12, 8, 8]               0
           Conv2d-19             [-1, 12, 6, 6]           1,296
             ReLU-20             [-1, 12, 6, 6]               0
      BatchNorm2d-21             [-1, 12, 6, 6]              24
          Dropout-22             [-1, 12, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           1,728
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 7,158
Trainable params: 7,158
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.62
Params size (MB): 0.03
Estimated Total Size (MB): 0.65
----------------------------------------------------------------
EPOCH: 0
Loss=0.27918386459350586 Batch_id=937 Accuracy=88.30: 100%|██████████| 938/938 [01:56<00:00,  8.02it/s]

Test set: Average loss: 0.0813, Accuracy: 9747/10000 (97.47%)

EPOCH: 1
Loss=0.3599886894226074 Batch_id=937 Accuracy=96.32: 100%|██████████| 938/938 [01:47<00:00,  8.76it/s]

Test set: Average loss: 0.0403, Accuracy: 9879/10000 (98.79%)

EPOCH: 2
Loss=0.09605827182531357 Batch_id=937 Accuracy=97.10: 100%|██████████| 938/938 [01:46<00:00,  8.82it/s]

Test set: Average loss: 0.0411, Accuracy: 9879/10000 (98.79%)

EPOCH: 3
Loss=0.09392913430929184 Batch_id=937 Accuracy=97.45: 100%|██████████| 938/938 [01:46<00:00,  8.84it/s]

Test set: Average loss: 0.0350, Accuracy: 9900/10000 (99.00%)

EPOCH: 4
Loss=0.04100244864821434 Batch_id=937 Accuracy=97.82: 100%|██████████| 938/938 [01:46<00:00,  8.79it/s]

Test set: Average loss: 0.0405, Accuracy: 9867/10000 (98.67%)

EPOCH: 5
Loss=0.10842973738908768 Batch_id=937 Accuracy=98.08: 100%|██████████| 938/938 [01:46<00:00,  8.82it/s]

Test set: Average loss: 0.0233, Accuracy: 9935/10000 (99.35%)

EPOCH: 6
Loss=0.014932522550225258 Batch_id=937 Accuracy=98.22: 100%|██████████| 938/938 [01:46<00:00,  8.81it/s]

Test set: Average loss: 0.0250, Accuracy: 9931/10000 (99.31%)

EPOCH: 7
Loss=0.0775967389345169 Batch_id=937 Accuracy=98.26: 100%|██████████| 938/938 [01:45<00:00,  8.87it/s]

Test set: Average loss: 0.0240, Accuracy: 9934/10000 (99.34%)

EPOCH: 8
Loss=0.08878305554389954 Batch_id=937 Accuracy=98.25: 100%|██████████| 938/938 [01:45<00:00,  8.90it/s]

Test set: Average loss: 0.0224, Accuracy: 9938/10000 (99.38%)

EPOCH: 9
Loss=0.09185191243886948 Batch_id=937 Accuracy=98.27: 100%|██████████| 938/938 [01:44<00:00,  8.95it/s]

Test set: Average loss: 0.0224, Accuracy: 9936/10000 (99.36%)

EPOCH: 10
Loss=0.08158264309167862 Batch_id=937 Accuracy=98.31: 100%|██████████| 938/938 [01:45<00:00,  8.90it/s]

Test set: Average loss: 0.0230, Accuracy: 9934/10000 (99.34%)

EPOCH: 11
Loss=0.06454384326934814 Batch_id=937 Accuracy=98.31: 100%|██████████| 938/938 [01:46<00:00,  8.84it/s]

Test set: Average loss: 0.0235, Accuracy: 9937/10000 (99.37%)

EPOCH: 12
Loss=0.026850048452615738 Batch_id=937 Accuracy=98.33: 100%|██████████| 938/938 [01:46<00:00,  8.79it/s]

Test set: Average loss: 0.0227, Accuracy: 9937/10000 (99.37%)

EPOCH: 13
Loss=0.25904005765914917 Batch_id=937 Accuracy=98.28: 100%|██████████| 938/938 [01:46<00:00,  8.80it/s]

Test set: Average loss: 0.0222, Accuracy: 9939/10000 (99.39%)

EPOCH: 14
Loss=0.04412204027175903 Batch_id=937 Accuracy=98.29: 100%|██████████| 938/938 [01:45<00:00,  8.90it/s]

Test set: Average loss: 0.0220, Accuracy: 9943/10000 (99.43%)

"""