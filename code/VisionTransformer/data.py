import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
class Mango_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform):
        self.df = pd.read_csv(csvFile)
        self.data_path = data_path
        self.xTrain = self.df['image_id']
        self.yTrain, self.labels = pd.factorize(self.df['label'], sort=True)
        self.data_transform = data_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.xTrain[index]))
        img = img.convert('RGB')
        if self.data_transform is not None:
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