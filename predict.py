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

parser = argparse.ArgumentParser(description="Predict images")
parser.add_argument('images', default='flowers/test/102/image_08030.jpg', nargs='*', action="store", type = str)
parser.add_argument('checkpoint', default="./checkpoint.pth", nargs='*', action="store",type = str)
parser.add_argument('--top_k', default=3, dest="topk", action="store", type=int)
parser.add_argument('--category_names',default='cat_to_name.json', dest="category_names", action="store", type = str)
parser.add_argument('--gpu', default="gpu", dest="gpu", action="store", type = str)


ag = parser.parse_args()

output_num = ag.topk
set_gpu = ag.gpu
images= ag.images
path = ag.checkpoint
flower_name = ag.category_names



pred_model = predict_function.load_checkpoint(path)

label = predict_function.load_label(flower_name)

probs, result = predict_function.predict(images, pred_model, output_num, set_gpu)

classes = label[result]