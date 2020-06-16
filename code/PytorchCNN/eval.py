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
ind = [0.4491086820559641, 0.5110112673570865, 0.8731044710710582, 0.18218755082068294, 0.3650997462797822, 0.21605293979567114, 0.99, 0.5389306222703668, 0.233419236545455, 0.17544182845835327, 0.5731093655427737, 0.5015190586068116, 0.99, 0.01, 0.49726851364050745, 0.694593240228059, 0.39460950335574463, 0.1463446569283618, 0.23974226312300628, 0.5161554718255529, 0.434586626846959, 0.99, 0.42466523379905513, 0.33664723473173774]
    
def eval(model, imgsize, data_path, device, classes=["A","B","C"]):
    df = pd.read_csv(os.path.join(data_path,"test.csv"))
    with torch.no_grad():
            dataTransformsValid = newPipelineWithParams(target_pipe, ind)

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
    model.load_weight(70)
    eval(model.classifier_net, imgsize=128,data_path="data/",device=model.device)

