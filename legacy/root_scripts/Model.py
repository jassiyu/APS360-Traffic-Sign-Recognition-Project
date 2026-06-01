import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


class ColorFilterLayer(nn.Module):
    def __init__(self):
        super(ColorFilterLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x

# Define the InceptionModule
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()

        #self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        #self.branch3x3_1 = nn.Conv2d(in_channels, red_3x3, kernel_size=1)
        #self.branch3x3_2 = nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)

        #self.branch5x5_1 = nn.Conv2d(in_channels, red_5x5, kernel_size=1)
        #self.branch5x5_2 = nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        #self.branch_pool = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU()
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.BatchNorm2d(red_3x3),
            nn.ReLU(),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU()
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.BatchNorm2d(red_5x5),
            nn.ReLU(),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=3, padding=1),  # 3x3 instead of 5x5
            nn.BatchNorm2d(out_5x5),
            nn.ReLU()
        )

        self.branch_pool = nn.Sequential(
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU()
        )


    def forward(self, x):
        #branch1x1 = F.relu(self.branch1x1(x))

        #branch3x3 = F.relu(self.branch3x3_1(x))
        #branch3x3 = F.relu(self.branch3x3_2(branch3x3))

        #branch5x5 = F.relu(self.branch5x5_1(x))
        #branch5x5 = F.relu(self.branch5x5_2(branch5x5))

        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.relu(self.branch_pool(branch_pool))

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        return outputs

# Define the TrafficSignGoogLeNet model
class TrafficSignGoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignGoogLeNet, self).__init__()
        self.color_filter = ColorFilterLayer()

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        #self.inception3a = InceptionModule(64, 64, 96, 128, 16, 32, 32)
        #self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionModule(64, 32, 48, 64, 8, 16, 16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        #self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        #self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4 = InceptionModule(128, 64, 64, 96, 16, 32, 32)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        #self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Linear(224, num_classes)

    def forward(self, x):
        x = self.color_filter(x)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        #x = self.inception3a(x)
        #x = self.inception3b(x)
        x = self.inception3(x)
        x = self.maxpool2(x)
        x = self.inception4(x)
        #x = self.inception4a(x)
        #x = self.inception4b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    print("Model")