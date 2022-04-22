# googlenet

import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        #
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)

        return x


# in_channels  channels 渠道   输入特征矩阵的深度  ch1_1   是纵深向的排布  ，是由512  到24个的一个纵深向排布
# 卷积核的个数  ？   inception 模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1_1, ch3_3red, ch3_3, ch5_5red, ch5_5, pool_prej):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1_1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3_3red, kernel_size=1),
            BasicConv2d(ch3_3red, ch3_3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5_5red, kernel_size=1),
            BasicConv2d(ch5_5red, ch5_5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            # 最大池化也可以有卷积核
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_prej, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 1  代表了深度方向上的合并     tensor torch   [batch, 深度，高，宽]
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x


# googlenet的分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # 平均池化下采样
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 含有两个类器，有不同的参数值
        # aux1 N*512* 14*14      aux2  N*528*14*14
        x1 = self.averagePool(x)
        # aux1 N*512*4*4         aux2   N*5112*4*4
        x2 = self.conv(x1)
        #   N*128*4*4       N*128*4*4
        # 1 代表了 channels这个维度战平的
        x3 = torch.flatten(x2, 1)
        x = F.dropout(x3, 0.5, training=self.training)
        x4 = self.fc1(x)
        x4 = F.relu(x4, inplace=True)

        x5 = F.dropout(x4, 0.5, training=self.training)
        x = self.fc2(x5)

        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        # 这里的3是一般3通道图像  ceil_mode
        self.conv = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv1 = BasicConv2d(64, 64, kernel_size=1)
        self.conv2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        # 最大池化没有进行填充
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 卷积核个数

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        # 自适应的平均池化下采样层  得到高和宽都为   1的特征矩阵
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.maxpool3(x)
        x = self.inception4a(x)
        # 判断是否在训练模式还是在测试模式
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)

        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        # 这里需要对线形层进行拉直
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        # x = nn.Softmax(x)  这一层不能存在，是错误的
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    # 初始化权重函数
    def _initialize_weights(self):
        for m in self.modules():
            # 比对是否为卷积操作,进行初始化的操作
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # 偏执的初始化
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# input1 = torch.rand([2, 3, 224, 224])
# model = GoogleNet()
# output = model(input1)
# print(output)