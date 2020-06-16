import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import imageio

class Mango_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform,
            sample_count = 0, sample_frac = 0, balance = False):
        self.df = pd.read_csv(csvFile)

        # show count by label
        # self.df['label'].value_counts()

        if sample_count or sample_frac or balance:
            if sample_frac:
                self.df = self.df.groupby('label')\
                            .apply(lambda x: x.sample(g.size().min())\
                                              .sample(frac=sample_frac)\
                                              .reset_index(drop=True))
            elif sample_count:
                self.df = self.df.groupby('label')\
                            .apply(lambda x: x.sample(sample_count).reset_index(drop=True))
            else:
                self.df = self.df.groupby('label')\
                            .apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

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

class Eval_dataset(Dataset):
    def __init__(self, csvFile, data_path, data_transform):
        self.df = pd.read_csv(csvFile)
        self.data_path = data_path
        self.xTrain = self.df['image_id']
        self.data_transform = data_transform

    def __getitem__(self, index):
        
        img = imageio.imread(os.path.join(self.data_path, self.xTrain[index]))
        if self.data_transform is not None:
            img = self.data_transform(**{'image': img})
            img = img['image']
        return img, index

    def __len__(self):
        return len(self.xTrain.index)