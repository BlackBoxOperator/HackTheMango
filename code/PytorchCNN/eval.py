import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from mlp import VGG16_model, ResNet, ResidualBlock
from data import Eval_dataset
from trainPipelineParameter import train
from esPipeline import idxList2trainPipeline, idxList2validPipeline, printPipeline, defaultParametersByPipeline, newPipelineWithParams

target_pipe = [56, 25, 86, 62]
    
def eval(model, imgsize, data_path, device, classes=["A","B","C"]):
    df = pd.read_csv(os.path.join(data_path,"test.csv"))
    with torch.no_grad():
            dataTransformsValid = idxList2validPipeline(target_pipe)

            validDatasets = Eval_dataset(os.path.join(data_path,"test.csv"), os.path.join(data_path,"C1-P1_Test"), dataTransformsValid)
            dataloadersValid = torch.utils.data.DataLoader(validDatasets, batch_size=1, shuffle=False)
            model.eval()
            for x, idx in dataloadersValid:
                x = x.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu()
                df["label"][idx] = classes[predicted[0]]
    df.to_csv("result.csv", header=True, index=False)

if __name__ == "__main__":
    model = train()
    model.load_weight(75)
    eval(model.classifier_net, imgsize=128,data_path="data/",device=model.device)

