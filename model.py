# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    model.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了CustomNet类，用于定义神经网络模型
#               ★★★请在空白处填写适当的语句，将CustomNet类的定义补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    """自定义神经网络模型。
    请完成对__init__、forward方法的实现，以完成CustomNet类的定义。
    """

    def __init__(self):
        """初始化方法。
        在本方法中，请完成神经网络的各个模块/层的定义。
        请确保每层的输出维度与下一层的输入维度匹配。
        """
        super(CustomNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)

        self.fc2 = nn.Linear(128, 6)
        # END------------------------------------------------------------

    def forward(self, x):
        """前向传播过程。
        在本方法中，请完成对神经网络前向传播计算的定义。
        """
        # START----------------------------------------------------------
        x = self.pool1(F.relu(self.conv1(x)))  # [B, 32, 32, 32]
        x = self.pool2(F.relu(self.conv2(x)))  # [B, 64, 16, 16]
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # END------------------------------------------------------------


if __name__ == "__main__":
    # 测试
    from dataset import CustomDataset
    from torchvision.transforms import ToTensor

    c = CustomDataset('./images/train.txt', './images/train', ToTensor)
    net = CustomNet()                                # 实例化
    x = torch.unsqueeze(c[10]['image'], 0)      # 模拟一个模型的输入数据
    print(net.forward(x))                            # 测试forward方法
