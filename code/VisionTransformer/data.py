import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
class Mango_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform, crop_by_pos = False):
        self.df = pd.read_csv(csvFile)
        self.data_path = data_path
        self.xTrain = self.df['image_id']

        if 'grade' in self.df:
            self.yTrain, self.labels = pd.factorize(self.df['grade'], sort=True)
        elif 'label' in self.df:
            self.yTrain, self.labels = pd.factorize(self.df['label'], sort=True)
        else:
            print('no proper field for label'), exit(0)

        if crop_by_pos and 'pos_x' in self.df:
            self.pos_x, self.pos_y, width, height = \
                self.df['pos_x'], self.df['pos_y'], \
                self.df['width'], self.df['height']
        else:
            self.pos_x, self.pos_y, width, height = None, None, None, None

        self.data_transform = data_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.xTrain[index]))
        img = img.convert('RGB')
        if self.data_transform:
            if self.pos_x:
                origin_transform = data_transform.transforms
                new_transform = [transforms.functional.crop(
                                    self.pos_y[index],
                                    self.pos_x[index],
                                    self.height[index],
                                    self.width[index]
                                    )] + origin_transform
                img = transforms.Compose(new_transform)(img)
            else:
                img = self.data_transform(img)
        return img, self.yTrain[index]

    def __len__(self):
        return len(self.xTrain.index)

    def __class__(self):
        return self.labels

class Eval_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform):
        self.df = pd.read_csv(csvFile)
        self.data_path = data_path
        self.xTrain = self.df['image_id']
        self.data_transform = data_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.xTrain[index]))
        img = img.convert('RGB')
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img, index

    def __len__(self):
        return len(self.xTrain.index)
