import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import imageio

class Mango_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform):
        self.df = pd.read_csv(csvFile)
        self.data_path = data_path
        self.xTrain = self.df['image_id']
        self.yTrain, self.labels = pd.factorize(self.df['label'], sort=True)
        self.data_transform = data_transform

    def __getitem__(self, index):
        #img = Image.open(os.path.join(self.data_path, self.xTrain[index]))
        #img = img.convert('RGB')
        img = imageio.imread(os.path.join(self.data_path, self.xTrain[index]))

        if self.data_transform is not None:
            #img = self.data_transform(img)
            img = self.data_transform(**{'image': img, 'label': self.yTrain[index]})
            img = img['image']
        return img, self.yTrain[index]

    def __len__(self):
        return len(self.xTrain.index)
    
    def __class__(self):
        return self.labels
