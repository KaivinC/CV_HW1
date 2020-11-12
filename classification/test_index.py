# oding=utf-8
import os
import datetime
import pandas as pd
from dataset.dataset_index_test import collate_fn2, dataset
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.test_util_index import test
from torch.nn import CrossEntropyLoss
from models.classifier import Classifier
from models.resnet_index import resnet_swap_2loss_add as Extractor
from PIL import Image
import argparse
import numpy as np
import glob
import normalize

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--desc', default='_')
parser.add_argument('--resume', default=None)
args = parser.parse_args()

# load the mean and std of all imgs of dataset
if(not os.path.exists("./normalize_value.txt")):
    normalize.normalize_img(img_h=128, img_w=128)
normalize_file = open("normalize_value.txt", "r")
temp = normalize_file.read().splitlines()
means = [float(i) for i in temp[1].split(",")]
stdevs = [float(i) for i in temp[2].split(",")]


# prepare dataset
rawdata_root = './datasets/testing_data'
train_pd = pd.read_csv("./datasets/anno/anno.txt",
                       header=None, names=['ImageName'])
cfg_numcls = 196
numimage = 5000

# set transform
print('Set transform')
data_transforms = {
    'swap': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
    ]),
    'unswap': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
    ]),
    'totensor': transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(means, stdevs),
    ]),

}

# Load data
data_set = {}
data_set['test'] = dataset(
    imgroot=rawdata_root,
    anno_pd=train_pd,
    unswap=data_transforms["unswap"],
    totensor=data_transforms["totensor"],
    train=False)

dataloader = {}
dataloader['test'] = torch.utils.data.DataLoader(
    data_set['test'],
    batch_size=8,
    shuffle=False,
    num_workers=12,
    collate_fn=collate_fn2)
print('done')
print('**********************************************')

# Set cache dir
filename = args.desc + 'test'
save_dir = './net_model' + filename
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# choose model and train set
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set model
model = [Extractor(stage=3), Classifier(2048, cfg_numcls)]
model = [e.cuda() for e in model]
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = [nn.DataParallel(e) for e in model]

# load trained model
resume = args.resume
start_epoch = 0
if resume is not None:
    state_dicts = torch.load(resume)
    [m.load_state_dict(d) for m, d in zip(model, state_dicts)]
    start_epoch = None


test(model=model,
     data_set=data_set,
     data_loader=dataloader,
     save_dir=save_dir
     )
