import os
import numpy as np

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

# ======== Evolutionray Strategy ============
# for reproducibility
random.seed(64)
IND_SIZE = 6
NGEN = 30
CENTROID = [0.5]*IND_SIZE
SIGMA = 0.3
LAMBDA = 8

global_a = 0

class train():
    def __init__(self,classes = ["A","B","C"], max_epoch = 10, lr = 1e-4, batch_size = 32,
                    image_size= 256, validation_frequency = 5, weight_path = "weight", data_path="data"):
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
        self.classifier_net = ResNet(ResidualBlock=ResidualBlock,numClasses=len(self.classes)).to(self.device)
        self.optimizer = optim.Adam(self.classifier_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.98)
        self.cross = nn.CrossEntropyLoss().to(self.device)

    def run(self, ind):
        global global_a

        ind = [x % 1 for x in ind]
        result = 0.0

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


        global_a = train()
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

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


strategy = cma.Strategy(centroid=CENTROID, sigma=SIGMA, lambda_=LAMBDA)

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

    random.seed(64)

    global_a = train()

    # new-design es start from this
    global strategy
    toolbox.register("generate", gen_new_pop, creator.Individual)
    toolbox.register("update", update_new_pop)
    toolbox.register("evaluate", global_a.run)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)    #random.s

    return pop, logbook, hof


if __name__ == "__main__":
    pop, log, hof = main()

    logbook = open('logbook.txt', 'w')

    for h in hof:
        print("individual: ", [x % 1 for x in h], " value: ", h.fitness.values)
        print("individual: ", [x % 1 for x in h], " value: ", h.fitness.values, file = logbook)

    print(log)
    print(log, file = logbook)
    logbook.close()
