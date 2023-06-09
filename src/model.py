import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(29952, 8192)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(8192, 1024)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 3)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.float()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = Model()
    print(count_parameters(model))
    print(count_trainable_parameters(model))
