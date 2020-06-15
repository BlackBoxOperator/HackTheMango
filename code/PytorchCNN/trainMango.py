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
# for reproducibility
random.seed(64)
'''
# ======== Evolutionray Strategy ============


MU, LAMBDA = 4, 6
NGEN = 30  # number of generations

IND_SIZE = 6
MIN_VALUE = 4
MAX_VALUE = 2048
MIN_STRATEGY = 64
MAX_STRATEGY = 512

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls([114.0, 36.0, 195.0, 273.0, 1713.0, 1290.0])
    # ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.randint(smin, smax) for _ in range(size))
    return ind

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
'''

##############################
####CMA########
NGEN = 30
IND_SIZE = 6
CENTROID = [64,128,256,512,1024,1024]
SIGMA = 64
LAMBDA = 8
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

################################



class train():
    def __init__(self,classifier_hp_dict = {},classes = ["A","B","C"], max_epoch = 10, lr = 1e-4, batch_size = 16,
                    image_size= 128, validation_frequency = 10, weight_path = "weight", data_path="data"):
        if not os.path.isdir(weight_path):
            os.makedirs(weight_path)
        self.data_path = data_path
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.classes = classes
        self.image_size = image_size
        self.validation_frequency = validation_frequency
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size = batch_size

        # self.classifier_net = VGG16_model(numClasses=len(self.classes)).to(self.device)
        self.classifier_net = ResNet(ResidualBlock=ResidualBlock,numClasses=len(self.classes),
                                        hp_dic=classifier_hp_dict).to(self.device)
        self.optimizer = optim.Adam(self.classifier_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.98)
        self.cross = nn.CrossEntropyLoss().to(self.device)
        self.seed = 0
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def run(self):
        result = 0.0

        dataTransformsTrain = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, interpolation=2),
            transforms.RandomRotation(degrees=(-180,180)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        trainDatasets = Mango_dataset(os.path.join(self.data_path,"train.csv"), os.path.join(self.data_path,"C1-P1_Train"), dataTransformsTrain,sample_count=300)
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
            if step+1 % self.validation_frequency == 0 or step == self.max_epoch-1:
                result = self.validation()
                self.store_weight(step)


        return result,

    def validation(self):
        with torch.no_grad():
            dataTransformsValid = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
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
            print("Validation Accuracy: {}".format(accuracy / count), file=logfile)
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


def es_hidden(ind):
    for i in range(6):
        if i < 4:
            ind[i] = int(np.abs(ind[i] % 513))
        else:
            ind[i] = int(np.abs(ind[i] % 2049))
        if ind[i] < 16:
            ind[i] = 16
    
    print("individual: ",ind)
    print("individual: ",ind, file=logfile)
    hp_dic = {
        "layer1" : int(ind[0]),
        "layer2" : int(ind[1]),
        "layer3" : int(ind[2]),
        "layer4" : int(ind[3]),
        "linear1" : int(ind[4]),
        "linear2" : int(ind[5])
    }
    a = train(classifier_hp_dict=hp_dic)
    result = a.run()
    return result

def main():

    random.seed(64)
    '''
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", es_hidden)

    toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    '''
    toolbox = base.Toolbox()

    strategy = cma.Strategy(centroid=CENTROID, sigma=SIGMA, lambda_=LAMBDA)


    def gen_new_pop(y):
        pop = strategy.generate(y)
        for i in range(len(pop)):
            for j in range(len(pop[i])):
                if j < 4:
                    pop[i][j] = int(pop[i][j] % 513)
                else:
                    pop[i][j] = int(pop[i][j] % 2049)
                if pop[i][j] < 16:
                    pop[i][j] = 16
        return pop
    toolbox.register("evaluate", es_hidden)

    toolbox.register("generate", gen_new_pop, creator.Individual)
    toolbox.register("update", strategy.update)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
    #     cxpb=0.2, mutpb=0.8, ngen=NGEN, stats=stats, halloffame=hof)
    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)

    return pop, logbook, hof

logfile = open('logfile.txt', 'w')
if __name__ == "__main__":
    pop, log, hof = main()

    logbook = open('logbook.txt', 'w')

    for h in hof:
        print("individual: ", [int(x)  for x in h], " value: ", h.fitness.values)
        print("individual: ", [int(x)  for x in h], " value: ", h.fitness.values, file = logbook)

    print(log)
    print(log, file = logbook)
    logbook.close()
    logfile.close()
