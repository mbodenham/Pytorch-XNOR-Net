import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from util import BinLinear
from util import BinConv2d



class Bin_LeNet_train(nn.Module):
    def __init__(self):
        super(Bin_LeNet_train, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=20)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = BinConv2d(in_channels=20, out_channels=50, kernel_size=5, bias=False, istrain=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = BinLinear(in_features=50 * 4 * 4, out_features=500, bias=False, istrain=True)
        self.fc2 = nn.Linear(in_features=500, out_features=10, bias=True)

        self.bn2 = nn.BatchNorm2d(num_features=50)
        self.bn3 = nn.BatchNorm1d(num_features=500)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 4 * 4 * 50)

        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        return self.fc2(x)


class Bin_LeNet_test(nn.Module):
    def __init__(self):
        super(Bin_LeNet_test, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=20)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = BinConv2d(in_channels=20, out_channels=50, kernel_size=5, bias=False, istrain=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = BinLinear(in_features=50 * 4 * 4, out_features=500, bias=False, istrain=False)
        self.fc2 = nn.Linear(in_features=500, out_features=10, bias=True)

        self.bn2 = nn.BatchNorm2d(num_features=50)
        self.bn3 = nn.BatchNorm1d(num_features=500)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 4 * 4 * 50)

        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        return self.fc2(x)