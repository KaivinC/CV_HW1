# oding=utf-8
import os
import datetime
import pandas as pd
from dataset.dataset_index import collate_fn1, collate_fn2, dataset
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util_index import train, trainlog
from utils.warmup_scheduler import WarmupMultiStepLR
from utils.auto_resume import AutoResumer
from torch.nn import CrossEntropyLoss
import logging
from models.resnet_index import resnet_swap_2loss_add as Extractor
from models.classifier import Classifier
from PIL import Image
import argparse
import numpy as np
import random
import normalize
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--stage', '-s', default=3, type=int, choices=[1, 2, 3])
parser.add_argument('--num_positive', '-n', default=3, type=int)
parser.add_argument('--desc', default='_')
args = parser.parse_args()

# load the mean and std of all imgs of dataset,the value of mean and std
# of 128*128 is the same with 512*512
if(not os.path.exists("./normalize_value.txt")):
    normalize.normalize_img(img_h=128, img_w=128)
normalize_file = open("normalize_value.txt", "r")
temp = normalize_file.read().splitlines()
means = [float(i) for i in temp[1].split(",")]
stdevs = [float(i) for i in temp[2].split(",")]

# Set the stage size
stage_to_size = {3: '7x', 2: '14x', 1: '28x'}
num_positive = args.num_positive
cfg = {}
time = datetime.datetime.now()
print("USE DATASET           <<< {} >>>".format("CV_HW1"))
sssize = stage_to_size[args.stage]
print("CALCULATE FEATURES OF <<< {} >>>".format(sssize))
stage = args.stage


# prepare dataset
rawdata_root = './datasets/training_data'
train_pd = pd.read_csv("./datasets/anno/train.txt", sep=",",
                       header=None, names=['ImageName', 'label'])
test_pd = pd.read_csv("./datasets/anno/test.txt", sep=",",
                      header=None, names=['ImageName', 'label'])
cfg['numcls'] = 196
numimage = 11185

# print training information
print('Dataset:', "CV_HW1")
print('train images:', train_pd.shape)
print('test images:', test_pd.shape)
print('num classes:', cfg['numcls'])
print("********************************************************")

# set transform
print('Set transform')
data_transforms = {
    'swap': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
    ]),
    'swap2': None,
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
    'None': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop((448, 448)),
    ]),
}

# Set dataloader
data_set = {}
data_set['train'] = dataset(
    cfg,
    imgroot=rawdata_root,
    anno_pd=train_pd,
    stage=stage,
    num_positive=num_positive,
    unswap=data_transforms["unswap"],
    swap=data_transforms["None"],
    swap2=data_transforms["swap2"],
    totensor=data_transforms["totensor"],
    train=True)
data_set['val'] = dataset(
    cfg,
    imgroot=rawdata_root,
    anno_pd=test_pd,
    stage=stage,
    num_positive=num_positive,
    unswap=data_transforms["None"],
    swap=data_transforms["None"],
    swap2=data_transforms["swap2"],
    totensor=data_transforms["totensor"],
    train=False)
dataloader = {}
dataloader['train'] = torch.utils.data.DataLoader(
    data_set['train'],
    batch_size=10,
    shuffle=True,
    num_workers=16,
    collate_fn=collate_fn1)
dataloader['val'] = torch.utils.data.DataLoader(
    data_set['val'],
    batch_size=10,
    shuffle=False,
    num_workers=16,
    collate_fn=collate_fn2)
print('done')
print('**********************************************')

# Set cache dir
print('Set cache dir')
filename = args.desc + '_' + \
    str(time.month) + str(time.day) + str(time.hour) + \
    sssize + '_num_' + str(num_positive)
save_dir = './net_model/' + filename
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = save_dir + '/' + filename + '.log'
trainlog(logfile)
print('done')
print('*********************************************')


print('choose model and train set')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = [Extractor(stage=stage), Classifier(2048, cfg['numcls'])]
print('swap + 2 loss')

model = [e.cuda() for e in model]
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = [nn.DataParallel(e) for e in model]

base_lr = 0.001
resume = None
start_epoch = 0
if resume is not None:
    logging.info('resuming finetune from %s' % resume)
    state_dicts = torch.load(resume)
    [m.load_state_dict(d) for m, d in zip(model, state_dicts)]
    start_epoch = None

params = []
for idx, m in enumerate(model):
    for key, value in m.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        momentum = 0.9
        if isinstance(m.module, Classifier) or 'lrx' in key:
            print('[learning rate] {} is set to x10'.format(key))
            lr = base_lr * 10
        params += [{"params": [value], "lr": lr, "momentum": momentum}]
optimizer = optim.SGD(params)

criterion = CrossEntropyLoss()
scheduler = WarmupMultiStepLR(optimizer,
                              warmup_epoch=2,
                              milestones=[40, 80, 120])
resumer = AutoResumer(scheduler, save_dir)

train(cfg,
      model,
      epoch_num=180,
      start_epoch=start_epoch,
      optimizer=optimizer,
      criterion=criterion,
      scheduler=scheduler,
      resumer=resumer,
      data_set=data_set,
      data_loader=dataloader,
      num_positive=num_positive,
      save_dir=save_dir)
