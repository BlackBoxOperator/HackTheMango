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
    def __init__(self,classes = ["A","B","C"], max_epoch = 100, lr = 1e-4, batch_size = 25,
                    image_size= 256, validation_frequency = 5, weight_path = "weight", data_path="data"):
        if not os.path.isdir(weight_path):
            os.makedirs(weight_path)
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

    def run(self, ind):

        ind = [x % 1 for x in ind]

        print("parameter = {}".format(ind))

        dataTransformsTrain = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, interpolation=2),
            transforms.RandomRotation(degrees=(-180,180)),
            transforms.ToTensor(),
            transforms.Normalize([ind[0], ind[1], ind[2]], [ind[3], ind[4], ind[5]])
            ])
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
            dataTransformsValid = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([ind[0], ind[1], ind[2]], [ind[3], ind[4], ind[5]])
            ])
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
    bench = [0.4914, 0.4822, 0.4465, 0.2023, 0.1994, 0.2010]
    train().run(bench)
