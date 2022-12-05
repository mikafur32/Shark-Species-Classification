import os
from PIL import Image
from Load_Dataset_Folder import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split

import torchvision.transforms as transforms

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
"from tqdm.notebook import tqdm, trange"
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class ImageLoader():
    def __init__(self, dataset_path='./Modern shark teeth'):
        self.dataset_path = dataset_path
        # load dataset
        self.x, self.y = load_dataset_folder(self.dataset_path)
        self.transform_x = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])]) #Figure out how to calculate our actual mean and std
    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        y = self.y[idx]
        return x, y

    def __len__(self):
        return len(self.x)
    
    
def Dataset_Splitter(ratio, dataset):
    train_list  = int(len(dataset)*ratio)
    test_list  = len(dataset) - train_list
    
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_list, test_list])
    
    return train_dataset, test_dataset

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):

        # x = [batch size, 1, 28, 28]

        x = self.conv1(x)

        # x = [batch size, 6, 24, 24]

        x = F.max_pool2d(x, kernel_size=2)

        # x = [batch size, 6, 12, 12]

        x = F.relu(x)

        x = self.conv2(x)

        # x = [batch size, 16, 8, 8]

        x = F.max_pool2d(x, kernel_size=2)

        # x = [batch size, 16, 4, 4]

        x = F.relu(x)

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
