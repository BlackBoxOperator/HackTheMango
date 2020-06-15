import os
import numpy as np
import math

# Deap
import random
import array
from deap import algorithms, base, creator, tools, benchmarks, cma

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

from esPipeline import idxList2trainPipeline, idxList2validPipeline, printPipeline

target_pipe = [0, 2, 4, 6, 8, 12, 14, 16, 18, 20, 22, 28, 32, 34, 36, 38, 40, 44, 46, 48, 50, 54, 56, 58, 60, 62, 66, 68, 72, 74, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]


# ======== Evolutionray Strategy ============
# for reproducibility
random.seed(64)
NGEN = 30
IND_SIZE = 0 # evalute in the bottom
CENTROID = 0 # evalute in the bottom
SIGMA = 0.3
LAMBDA = 8

global_a = 0
# ==========================================

class train():
    def __init__(self,classes = ["A","B","C"], max_epoch = 15, lr = 1e-4, batch_size = 32,
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

    def run(self, ind):
        global global_a

        result = 0.0

        #return sum([1 / (((ind[i] - i) ** 2) + 1) for i in range(len(ind))]),

        """
        Image data augmentation is typically only applied to the training dataset,
        and not to the validation or test dataset.
        This is different from data preparation such as image resizing and pixel scaling;
        they must be performed consistently across all datasets that interact with the model.

        https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
        """

        dataTransformsTrain = newPipelineWithParams(target_pipe, ind)

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
            dataTransformsValid = idxList2validPipeline(target_pipe)

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

# ea


# should be NUM_PARAMETER
NUM_PIPE = len(defaultParametersByPipeline(target_pipe))

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
strategy = 0



def adjust(x):
    if x > 0.99:
        return 0.99
    if x < 0.01:
        return 0.01
    return x

def update_new_pop(y):
    global strategy
    strategy.update(y)
    for group in y:
        group[:] = [adjust(x) for x in group]

def gen_new_pop(y):
    global strategy
    pop = strategy.generate(y)
    for group in pop:
        group[:] = [adjust(x) for x in group]
    return pop



def main():
    global global_a
    global strategy
    global IND_SIZE
    global CENTROID

    strategy = cma.Strategy(centroid=CENTROID, sigma=SIGMA, lambda_=LAMBDA)

    random.seed(64)

    global_a = train()

    toolbox.register("generate", gen_new_pop, creator.Individual)
    toolbox.register("update", update_new_pop)
    toolbox.register("evaluate", global_a.run)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)

    return pop, logbook, hof

if __name__ == "__main__":
    global IND_SIZE
    global CENTROID
    pl = [83, 46, 51, 42, 44, 60, 23, 14, 63, 77, 61, 20, 36, 24, 58, 30]

    pl_params = ep.defaultParametersByPipeline(pl)
    IND_SIZE = len(pl_params)
    CENTROID = [0.5] * IND_SIZE
    pop, log, hof = main(pl_params)

    print("hof: ", hof[0])

    new_pl_params = ep.newPipelineWithParams(pl, hof[0])

    print("new pl params", new_pl_params)

    logbook = open('logbook.txt', 'w')

    for h in hof:
        print("individual: ", [x % 1 for x in h], " value: ", h.fitness.values)
        print("individual: ", [x % 1 for x in h], " value: ", h.fitness.values, file = logbook)

    print(log)
    print(log, file = logbook)
    logbook.close()
