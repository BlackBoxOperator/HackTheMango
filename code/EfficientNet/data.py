import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import albumentations as A
import imageio

class Mango_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform, crop_by_pos = False, blur = False):
        self.df = pd.read_csv(csvFile)
        self.data_path = data_path
        self.xTrain = self.df['image_id']
        self.blur = blur

        if 'grade' in self.df:
            self.yTrain, self.labels = pd.factorize(self.df['grade'], sort=True)
        elif 'label' in self.df:
            self.yTrain, self.labels = pd.factorize(self.df['label'], sort=True)
        else:
            print('no proper field for label'), exit(0)

        if crop_by_pos and 'pos_x' in self.df:
            self.pos_x, self.pos_y, self.width, self.height = \
                self.df['pos_x'], self.df['pos_y'], \
                self.df['width'], self.df['height']
        else:
            self.pos_x, self.pos_y, self.width, self.height = None, None, None, None

        self.data_transform = data_transform

    def __getitem__(self, index):

        if self.blur:
            img = imageio.imread(os.path.join(self.data_path, self.xTrain[index]))
        else:
            img = Image.open(os.path.join(self.data_path, self.xTrain[index]))
            img = img.convert('RGB')

        if self.data_transform:
            if self.blur: # use albumentations
                xm = max(self.pos_x[index], 0)
                ym = max(self.pos_y[index], 0)
                xmx = min(1280, xm + self.width[index])
                ymx = min(720, ym + self.height[index])
                aug = A.Compose([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.Crop(
                        x_min = xm,
                        y_min = ym, 
                        x_max = xmx,
                        y_max = ymx),
                    self.data_transform,
                ], p=1)
                img = aug(**{'image': img, 'label': self.yTrain[index]})
                img = img['image']
            else: # torchvision transforms
                if self.pos_x is not None:
                    img = transforms.functional.crop(
                            img, self.pos_y[index], self.pos_x[index],
                            self.height[index], self.width[index])
                img = self.data_transform(img)
        return img, self.yTrain[index]

    def __len__(self):
        return len(self.xTrain.index)

    def __class__(self):
        return self.labels

class Eval_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform, crop_by_pos = False):
        self.df = pd.read_csv(csvFile)
        self.data_path = data_path
        self.xTrain = self.df['image_id']

        if crop_by_pos and 'pos_x' in self.df:
            self.pos_x, self.pos_y, self.width, self.height = \
                self.df['pos_x'], self.df['pos_y'], \
                self.df['width'], self.df['height']
        else:
            self.pos_x, self.pos_y, self.width, self.height = None, None, None, None

        self.data_transform = data_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.xTrain[index]))
        img = img.convert('RGB')

        if self.data_transform:
            if self.pos_x is not None:
                img = transforms.functional.crop(
                        img, self.pos_y[index], self.pos_x[index],
                        self.height[index], self.width[index])
            img = self.data_transform(img)

        return img, index

    def __len__(self):
        return len(self.xTrain.index)
