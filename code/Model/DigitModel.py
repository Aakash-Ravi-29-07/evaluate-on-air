import torch
from torch.nn import Module
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, BatchNorm1d,Flatten
from torch.nn import functional as F

class DigitModel(Module):
    def __init__(self, kernel_size, pool_size, num_classes):
        super(DigitModel, self).__init__()

        self.cnn1 = Conv2d(1, 10, kernel_size=kernel_size)
        self.cnn2 = Conv2d(10, 20, kernel_size)
        self.pool1 = MaxPool2d(pool_size)
        self.pool2 = MaxPool2d(pool_size)
        self.fc1 = Linear(320, 80)
        self.fc2 = Linear(80, 10)
        self.batchnorm1 = BatchNorm1d(80)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        # x = self.batchnorm1(x)
        x = self.fc2(x)
        return F.log_softmax(x)