# coding=utf8
from __future__ import division
import torch
import os
import time
import datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil
from torch.nn import L1Loss, MultiLabelSoftMarginLoss, BCELoss
from torch import nn
from .rela import calc_rela
import torch.nn.functional as F
import make_submission


def test(model,
         data_set,
         data_loader,
         save_dir):

    storepath = os.path.join(save_dir, 'store5.txt')
    storefile = open(storepath, 'w')
    model = [e.train(False) for e in model]
    extractor, classifier = model

    LL = len(data_loader['test'])
    for batch_cnt_val, data_val in enumerate(data_loader['test']):

        # print data
        inputs, items = data_val
        inputs = Variable(inputs.cuda())

        # forward
        cls_feature = extractor(inputs)
        outputs = classifier(cls_feature)
        _, preds1 = torch.max(outputs, 1)

        # statistics
        for i in range(preds1.size(0)):
            storefile.write(str(preds1[i].item()) + "," + str(items[i]) + "\n")
        print('[TEST]: Batch {:03d} / {:03d}'.format(batch_cnt_val + 1, LL))

    storefile.close()
