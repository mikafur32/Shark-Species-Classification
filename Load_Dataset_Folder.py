import os
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import random_split
import torch


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_dataset_folder(data_path):
    x = []
    y = []
    i = 0
    for file1 in os.listdir(data_path):
        file2 = os.path.join(data_path, file1)
        for file3 in os.listdir(file2):
            if file3 == "unused teeth":
                continue
            else:
                simage = os.path.join(file2, file3)
                x.append(simage)
                y.append(i)
        i += 1
    return list(x), list(y)


class ImageLoader:
    def __init__(self, dataset_path='./Modern shark teeth', resize=(336, 224), cropsize=224):
        self.dataset_path = dataset_path
        # load dataset
        self.x, self.y = load_dataset_folder(self.dataset_path)
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, InterpolationMode.BILINEAR),
            transforms.CenterCrop(cropsize),
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
    train_list = int(len(dataset) * ratio)
    test_list = len(dataset) - train_list

    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_list, test_list])

    return train_dataset, test_dataset
