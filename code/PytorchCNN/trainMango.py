import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from mlp import VGG16_model
from data import Mango_dataset

class train():
    def __init__(self,classes = ["A","B","C"], max_epoch = 100, lr = 1e-4, batch_size = 32,
                    image_size= 256, validation_frequency = 1, weight_path = "weight", data_path="data"):
        if not os.path.isdir(weight_path):
            os.makedirs(weight_path)
        self.data_path = data_path
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.classes = classes
        self.image_size = image_size
        self.validation_frequency = validation_frequency
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size = batch_size

        self.classifier_net = VGG16_model(numClasses=len(self.classes)).to(self.device)
        self.optimizer = optim.Adam(self.classifier_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.98)
        self.cross = nn.CrossEntropyLoss().to(self.device)

    def run(self):
        dataTransformsTrain = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomResizedCrop(self.image_size, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        trainDatasets = Mango_dataset(os.path.join(self.data_path,"train.csv"), os.path.join(self.data_path,"C1-P1_Train"), dataTransformsTrain)
        dataloadersTrain = torch.utils.data.DataLoader(trainDatasets, batch_size=self.batch_size, shuffle=True, num_workers=4)
        for step in range(self.max_epoch):
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
            print("Training Accuracy: {}".format(accuracy / count))
            if step % self.validation_frequency == 0:
                self.validation()
                self.store_weight(step)

    def validation(self):
        with torch.no_grad():
            dataTransformsValid = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            validDatasets = Mango_dataset(os.path.join(self.data_path,"dev.csv"), os.path.join(self.data_path,"C1-P1_Dev"), dataTransformsValid)
            dataloadersValid = torch.utils.data.DataLoader(validDatasets, batch_size=self.batch_size, shuffle=False)
            accuracy = 0
            count = 0
            for x, label in dataloadersValid:
                x = x.to(self.device)
                label = label.to(self.device, dtype=torch.long)
                outputs = self.classifier_net(x)
                _, predicted = torch.max(outputs.data, 1)
                count += len(x)
                accuracy += (predicted == label).sum().item()
            print("Validation Accuracy: {}".format(accuracy / count))

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
    a = train()
    a.run()
    