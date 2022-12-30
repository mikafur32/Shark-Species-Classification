from Load_Dataset_Folder import *
import torch
import torch.nn as nn
import torch.nn.functional as F


# Depreciated File
# Contains contents for our attempt at a neural network based of LeNet
# This can be ignored as it is not used for any of the program

'''
class MMNet_336_224(nn.Module):
    def __init__(self, output_dim=9):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=32,
                               kernel_size=2,
                               )

        self.conv4 = nn.Conv2d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               stride=2
                               )

        self.conv5 = nn.Conv2d(in_channels=16,
                               out_channels=8,
                               kernel_size=3
                               )

        self.fc_1 = nn.Linear(8 * 18 * 11, 6336)
        self.fc_2 = nn.Linear(6336, 99)
        self.fc_3 = nn.Linear(99, output_dim)

    def forward(self, x):
        # x = [batch size, 1, 28, 28]

        x = self.conv1(x)

        # x = [batch size, 6, 24, 24]

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # x = [batch size, 6, 12, 12]

        x = F.relu(x)

        x = self.conv2(x)

        # x = [batch size, 16, 8, 8]

        x = self.conv3(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(x)

        # x = [batch size, 16, 4, 4]

        x = self.conv4(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(x)

        x = self.conv5(x)

        x = x.view(x.shape[0], -1)

        # x = [batch size, 16*4*4 = 256]

        h = x

        x = self.fc_1(x)

        # x = [batch size, 120]

        x = F.relu(x)

        x = self.fc_2(x)

        # x = batch size, 84]

        x = F.relu(x)

        x = self.fc_3(x)

        # x = [batch size, output dim]

        return x, h

'''
'''
class MMNet_224(nn.Module):
    def __init__(self, output_dim= 9):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding= 1)

        self.conv3 = nn.Conv2d(in_channels = 64,
                               out_channels = 32,
                               kernel_size = 2,
                               )

        self.conv4 = nn.Conv2d(in_channels = 32,
                               out_channels = 16,
                               kernel_size = 3,
                               stride= 2
                               )

        self.conv5 = nn.Conv2d(in_channels = 16,
                               out_channels = 8,
                               kernel_size = 3
                               )

        self.fc_1 = nn.Linear(8 * 18 * 11, 6336)
        self.fc_2 = nn.Linear(6336, 99)
        self.fc_3 = nn.Linear(99, output_dim)

    def forward(self, x):

        # x = [batch size, 1, 28, 28]

        x = self.conv1(x)

        # x = [batch size, 6, 24, 24]

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # x = [batch size, 6, 12, 12]

        x = F.relu(x)

        x = self.conv2(x)

        # x = [batch size, 16, 8, 8]

        x = self.conv3(x)

        x = F.max_pool2d(x, kernel_size=2, stride= 2)

        x = F.relu(x)

        # x = [batch size, 16, 4, 4]

        x = self.conv4(x)

        x = F.max_pool2d(x, kernel_size=2, stride= 2)

        x = F.relu(x)

        x = self.conv5(x)

        x = x.view(x.shape[0], -1)

        # x = [batch size, 16*4*4 = 256]

        h = x

        x = self.fc_1(x)

        # x = [batch size, 120]

        x = F.relu(x)

        x = self.fc_2(x)

        # x = batch size, 84]

        x = F.relu(x)

        x = self.fc_3(x)

        # x = [batch size, output dim]

        return x, h
'''
