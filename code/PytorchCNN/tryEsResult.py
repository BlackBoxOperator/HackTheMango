import os
import numpy as np

# Deap
import random
import array
from deap import algorithms, base, creator, tools, benchmarks

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from mlp import VGG16_model, ResNet, ResidualBlock
from data import Mango_dataset

# import Albumentations package
import albumentations as A
# Import pytorch utilities from albumentations
from albumentations.pytorch import ToTensor

# ======== Evolutionray Strategy ============
# for reproducibility
random.seed(64)

MU, LAMBDA = 4, 8
NGEN = 30  # number of generations

IND_SIZE = 6
MIN_VALUE = 0
MAX_VALUE = 1
MIN_STRATEGY = 0.3
MAX_STRATEGY = 0.8

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", array.array, typecode="d")

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

# ==========================================

class train():
    def __init__(self,classes = ["A","B","C"], max_epoch = 100, lr = 1e-4, batch_size = 32,
                    image_size= 128, validation_frequency = 5, weight_path = "weight", data_path="data"):
        if not os.path.isdir(weight_path):
            os.makedirs(weight_path)

        self.seed = 42
        self.data_path = data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = classes
        self.image_size = image_size
        self.validation_frequency = validation_frequency
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size = batch_size

        # self.classifier_net = VGG16_model(numClasses=len(self.classes)).to(self.device)
        self.classifier_net = ResNet(ResidualBlock=ResidualBlock,numClasses=len(self.classes)).to(self.device)
        self.optimizer = optim.Adam(self.classifier_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.98)
        self.cross = nn.CrossEntropyLoss().to(self.device)

        """
        fix the seed
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        """
        image transformation / augmentation
        """
# Blur
# Transpose
# RandomGridShuffle
# RGBShift
# MotionBlur
# MedianBlur
# GaussianBlur (Noise) Glass
# RGBShift ??
# CLAHE
# ChannelShuffle
# InvertImg
# ISONoise
# FancyPCA (class)
# HueSaturationValue
# ShiftScaleRotate?

# Equalize
# OpticalDistortion
# GridDistortion
# CoarseDropout
# RandomGamma

        train_trans = lambda ind: [

            A.RandomResizedCrop(
                height=self.image_size,
                width=self.image_size,
                scale=(0.9, 1.0), # 0.08 to 0.8 become worse
                ratio=(0.75, 1.3333333333333333),
                interpolation=2, # non default
                p=1,
            ),

            A.Flip(p=.5),

            #A.Transpose(p=.5),

            #A.ISONoise(
            #    color_shift=(0.01, 0.05),
            #    intensity=(0.1, 0.5),
            #    p=0.5),

            A.RandomGamma(p=.5),
            A.GridDistortion(p=.3),
            A.OpticalDistortion(p=.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=8,
                max_width=8,
                min_holes=None,
                min_height=None,
                min_width=None,
                fill_value=0,
                p=0.5
            ),
            #A.RandomGridShuffle(grid=(32, 32), p=0.5),
            #A.Transpose(),
            #A.OneOf([
            #    A.IAAAdditiveGaussianNoise(),
            #    A.GaussNoise(),
            #], p=0.2),
            #A.OneOf([
            #    A.MotionBlur(p=.2),
            #    A.MedianBlur(blur_limit=3, p=.1),
            #    A.Blur(blur_limit=3, p=.1),
            #], p=0.2),
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
            #A.OneOf([
            #    A.OpticalDistortion(p=0.3),
            #    A.GridDistortion(p=.1),
            #    A.IAAPiecewiseAffine(p=0.3),
            #], p=0.2),
            #A.OneOf([
            #    A.CLAHE(clip_limit=2),
            #    A.IAASharpen(),
            #    A.IAAEmboss(),
            #    A.RandomContrast(),
            #    A.RandomBrightness(),
            #], p=0.3),

            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                brightness_by_max=False,
                p=0.5
            ),
            
            A.ShiftScaleRotate(
                #shift_limit=0.0625,
                #scale_limit=0.1,
                shift_limit=0,
                scale_limit=0,
                rotate_limit=180,
                interpolation=1,
                p=0.5
            ), 

            #A.Equalize(p=0.2),

            A.Normalize(
                mean=[ind[0], ind[1], ind[2]],
                std=[ind[3], ind[4], ind[5]],
            ),
            ToTensor(), # convert the image to PyTorch tensor
        ]

        valid_trans = lambda ind: [
            A.RandomBrightnessContrast(
                brightness_limit = 0,
                contrast_limit = 0,
                brightness_by_max = False,
                always_apply = True,
                p = 1
            ),
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean = [ind[0], ind[1], ind[2]],
                std = [ind[3], ind[4], ind[5]],
            ),
            ToTensor(),
        ]

        """
        new_trans = [
                A.HorizontalFlip(p = 0.5), # apply horizontal flip to 50% of images
                A.OneOf(
                    [
                        # apply one of transforms to 50% of images
                        A.RandomContrast(), # apply random contrast
                        A.RandomGamma(), # apply random gamma
                        A.RandomBrightness(), # apply random brightness
                    ],
                    p = 0.5
                ),
                A.RandomResizedCrop(self.image_size, interpolation=2),
                A.RandomRotation(degrees=(-180,180)),
                ToTensor(), # convert the image to PyTorch tensor
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    #mean=[ind[0], ind[1], ind[2]],
                    std=[0.229, 0.224, 0.225],
                    #std=[ind[3], ind[4], ind[5]],
                ),
            ]
        """

        """
        [
                #A.HorizontalFlip(p = 0.5), # apply horizontal flip to 50% of images
                #A.OneOf(
                #    [
                #        # apply one of transforms to 50% of images
                #        A.RandomContrast(), # apply random contrast
                #        A.RandomGamma(), # apply random gamma
                #        A.RandomBrightness(), # apply random brightness
                #    ],
                #    p = 0.5
                #),

                A.RandomResizedCrop(self.image_size, interpolation=2),
                A.RandomRotation(degrees=(-180,180)),

                ToTensor(), # convert the image to PyTorch tensor

                #A.Normalize(, )
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    #mean=[ind[0], ind[1], ind[2]],
                    std=[0.229, 0.224, 0.225],
                    #std=[ind[3], ind[4], ind[5]],
                ),

            ]
        """
        self.augmentation_pipeline = lambda ind: A.Compose(train_trans(ind), p = 1)
        self.validation_pipeline = lambda ind: A.Compose(valid_trans(ind), p = 1)

    def run(self, ind):

        ind = [x % 1 for x in ind]

        print("parameter = {}".format(ind))

        # Define the augmentation pipeline

        """
        Image data augmentation is typically only applied to the training dataset,
        and not to the validation or test dataset.
        This is different from data preparation such as image resizing and pixel scaling;
        they must be performed consistently across all datasets that interact with the model.

        https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
        """

        dataTransformsTrain = self.augmentation_pipeline(ind)

        trainDatasets = Mango_dataset(os.path.join(self.data_path,"train.csv"), os.path.join(self.data_path,"C1-P1_Train"), dataTransformsTrain)
        dataloadersTrain = torch.utils.data.DataLoader(trainDatasets, batch_size=self.batch_size, shuffle=True, num_workers=2)
        for step in range(self.max_epoch):
            self.classifier_net.train()
            totalLoss = 0
            count = 0
            accuracy = 0
            for x, label in dataloadersTrain:
                x = x.to(self.device)
                label = label.to(self.device, dtype=torch.long)
                outputs = self.classifier_net(x)
                loss = self.cross(outputs,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                _, predicted = torch.max(outputs.data, 1)
                count += len(x)
                accuracy += (predicted == label).sum().item()
                totalLoss += loss.item()
            print("Training loss = {}".format(totalLoss/count))
            print("step = {}, Training Accuracy: {}".format(step,accuracy / count))
            if step % self.validation_frequency == 0 or step == self.max_epoch-1:
                result = self.validation(ind)
                self.store_weight(step)

        return result,

    def validation(self, ind):
        with torch.no_grad():

            """
            dataTransformsValid = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([ind[0], ind[1], ind[2]], [ind[3], ind[4], ind[5]])
            ])
            """

            dataTransformsValid = self.validation_pipeline(ind)

            validDatasets = Mango_dataset(os.path.join(self.data_path,"dev.csv"), os.path.join(self.data_path,"C1-P1_Dev"), dataTransformsValid)
            dataloadersValid = torch.utils.data.DataLoader(validDatasets, batch_size=self.batch_size, shuffle=False)
            accuracy = 0
            count = 0
            self.classifier_net.eval()
            for x, label in dataloadersValid:
                x = x.to(self.device)
                label = label.to(self.device, dtype=torch.long)
                outputs = self.classifier_net(x)
                _, predicted = torch.max(outputs.data, 1)
                count += len(x)
                accuracy += (predicted == label).sum().item()

            # ===============================================
            # export the accurancy for evolutionary algorithm
            exportAccurancy = accuracy / count
            # ===============================================

            print("Validation Accuracy: {}".format(accuracy / count))
            return exportAccurancy

    def store_weight(self,step):
        with open("weight/weight_{}".format(step), "wb") as f:
            torch.save(self.classifier_net.state_dict(), f)
    def load_weight(self,param):
        with open(os.path.join("weight/weight_" + str(param)), "rb") as f:
            print("loading weight_{}".format(param))
            self.classifier_net.load_state_dict(torch.load(f,map_location="cpu"))
    def predict(self,x):
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.classifier_net(x)
            return self.classes[torch.argmax(outputs)]

if __name__ == "__main__":
    #_78625 = [0.9937351930880042, 0.6999774952670302, 0.4506417252015659,
    #        0.23635548700778367, 0.07663879046228922, 0.6776228457941125]
    bench = [0.4914, 0.4822, 0.4465, 0.2023, 0.1994, 0.2010]
    train().run(bench)
