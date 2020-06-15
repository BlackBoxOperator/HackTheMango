import os
import math
import numpy as np

# Deap
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from mlp import VGG16_model, ResNet, ResidualBlock
from data import Mango_dataset

from esPipeline import idxList2trainPipeline, idxList2validPipeline, printPipeline, defaultParametersByPipeline, newPipelineWithParams

target_pipeline = [83, 46, 51, 42, 44, 60, 23, 14, 63, 77, 61, 20, 36, 24, 58, 30]
target_parameters = [] # give the parameters here

class train():
    def __init__(self,classes = ["A","B","C"], max_epoch = 100, lr = 1e-4, batch_size = 32,
                    image_size= 128, validation_frequency = 5, weight_path = "weight", data_path="data"):

        #return
        self.seed = 42

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

        """
        fix the seed
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def run(self, pipeline, parameters):

        result = 0.0

        """
        Image data augmentation is typically only applied to the training dataset,
        and not to the validation or test dataset.
        This is different from data preparation such as image resizing and pixel scaling;
        they must be performed consistently across all datasets that interact with the model.

        https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
        """

        dataTransformsTrain = newPipelineWithParams(pipeline, parameters)

        trainDatasets = Mango_dataset(
                os.path.join(self.data_path,"train.csv"),
                os.path.join(self.data_path,"C1-P1_Train"),
                dataTransformsTrain,
                sample_count = 100,
        )
        dataloadersTrain = torch.utils.data.DataLoader(
                trainDatasets, batch_size=self.batch_size, shuffle=True, num_workers=2)

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
            if(math.isnan(totalLoss/count)): return 0,
            print("step = {}, Training Accuracy: {}".format(step,accuracy / count))
            if step % self.validation_frequency == 0 or step == self.max_epoch-1:
                result = self.validation(pipeline, parameters)
                self.store_weight(step)

        return result,

    def validation(self, pipeline, parameters):
        with torch.no_grad():

            dataTransformsValid = idxList2validPipeline(pipeline)

            validDatasets = Mango_dataset(
                    os.path.join(self.data_path,"dev.csv"),
                    os.path.join(self.data_path,"C1-P1_Dev"),
                    dataTransformsValid
            )
            dataloadersValid = torch.utils.data.DataLoader(
                    validDatasets, batch_size=self.batch_size, shuffle=False)
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

            exportAccurancy = accuracy / count

            print("Validation Accuracy: {}".format(exportAccurancy))
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

    result = train().run(pipeline, parameters)
    print(result[0])
