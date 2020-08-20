import numpy as np
import time
import torch
from torch import nn,optim
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import json

import argparse

import train_model

parser = argparse.ArgumentParser(description="Train model")

parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu",type=str)
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.01)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=20)
parser.add_argument('--arch', dest="arch", action="store", default="vgg13", type = str, help ="vgg13, vgg16, vgg19")
parser.add_argument('--hidden_units', type=int, dest="hidden_unit", action="store", default=512,type=int)

ag = parser.parse_args()

load_path = ag.data_dir
save_path = ag.save_dir
lr = ag.learning_rate
arch = ag.arch
hidden_layer = ag.hidden_unit
gpu_set = ag.gpu
epochs = ag.epochs

trainloader, validloader, testloader, image_dataset = train_model.load_data(load_path)

model, optimizer, criterion  = train_model.train_setup(structure,hidden_layer,lr,gpu_set)

train_model.nn_train(model, epochs, trainloader,testloader, validloader, criterion, gpu_set)

train_model.save(model, optimizer, arch, image_datasets, save_path)
