from PIL import Image
from torchvision.transforms import InterpolationMode

from Load_Dataset_Folder import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

import torchvision.transforms as transforms
import numpy as np
import random


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class ImageLoader():
    def __init__(self, dataset_path='./Modern shark teeth', resize=(224, 224)):
        self.dataset_path = dataset_path
        # load dataset
        self.x, self.y = load_dataset_folder(self.dataset_path)
        self.transform_x = transforms.Compose([
                                      #transforms.Grayscale(num_output_channels=1),
                                      transforms.Resize(resize, InterpolationMode.BILINEAR),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        y = self.y[idx]
        return x, y

    def __len__(self):
        return len(self.x)
    
    
def Dataset_Splitter(ratio, dataset):
    train_list = int(len(dataset)*ratio)
    test_list = len(dataset) - train_list
    
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_list, test_list])
    
    return train_dataset, test_dataset


class MMNet_336_224(nn.Module):
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

