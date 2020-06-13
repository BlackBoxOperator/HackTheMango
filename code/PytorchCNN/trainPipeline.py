import os
import numpy as np
import math

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

from esPipeline import numOfPipeline, idxList2trainPipeline, idxList2validPipeline, printPipeline


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

#useless
#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
#creator.create("Strategy", array.array, typecode="d")

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
    def __init__(self,classes = ["A","B","C"], max_epoch = 10, lr = 1e-4, batch_size = 32,
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
        #return sum([1 / (((ind[i] - i) ** 2) + 1) for i in range(len(ind))]),
        printPipeline(ind, idxList2trainPipeline)

        """
        Image data augmentation is typically only applied to the training dataset,
        and not to the validation or test dataset.
        This is different from data preparation such as image resizing and pixel scaling;
        they must be performed consistently across all datasets that interact with the model.

        https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
        """

        dataTransformsTrain = idxList2trainPipeline(ind)

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

            dataTransformsValid = idxList2validPipeline(ind)

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
NUM_PIPE = numOfPipeline()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def main():
    #_78625 = [0.9937351930880042, 0.6999774952670302, 0.4506417252015659,
    #        0.23635548700778367, 0.07663879046228922, 0.6776228457941125]
    random.seed(64)

    #Since there is only one queen per line,
    #individual are represented by a permutation
    toolbox = base.Toolbox()
    toolbox.register("permutation", random.sample, range(NUM_PIPE), NUM_PIPE)

    #Structure initializers
    #An individual is a list that represents the position of each queen.
    #Only the line is stored, the column is the index of the number in the list.
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", train().run)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0/NUM_PIPE)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=6)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=30, stats=stats,
                        halloffame=hof, verbose=True)

    return pop, logbook, hof

if __name__ == "__main__":
    pop, log, hof = main()

    logbook = open('logbook.txt', 'w')

    print(log)
    print(log, file = logbook)
    logbook.close()

    print(hof[0])
