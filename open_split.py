#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:19:04 2022

@author: gloupit
"""

import numpy as np

def decompose(gt):
    #décompose une image de n classes en n images (1 par classe)
    n_classes = np.amax(gt)
    gt_classes = []
    for i in range(1,n_classes+1):
        # print(i)
        gt_temp = np.zeros(gt.shape)
        gt_temp = np.where(gt==i,gt,0.0)
        gt_classes.append(gt_temp)
    return gt_classes
        
def compare_gt_train(gt,train):
    gt_classes = decompose(gt)
    train_classes = decompose(train)
    diff_classes = []
    for i in range(0,12):
        diff_temp = np.zeros(gt_classes[0].shape)
        diff_temp = np.where(gt_classes[i] == train_classes[i], train_classes[i], 0.0)
        diff_classes.append(diff_temp)
    return diff_classes

def compare_gt_train_class(gt,train,valid,n):
    # diff_classes = compare_gt_train(gt,train)
    gt_classes = decompose(gt)
    train_classes = decompose(train)
    valid_classes = decompose(valid)
    # valid_classes = decompose(valid)
    # gt_temp = gt_classes[n]
    # gt_temp = np.where(train_classes[n] != 0,100.0,gt_temp)
    return gt_classes[n], train_classes[n], valid_classes[n]
    
    
    

# gt = np.load('/tmp_user/ldota704h/gloupit/Datasets/Mauzac/mauzac_gt.npy')
gt = np.load('/tmp_user/ldota704h/gloupit/Datasets/Toulouse/toulouse_gt.npy')

train = np.load('./split/Toulouse/split/train_gt3.npy')
valid = np.load('./split/Toulouse/split/val_gt3.npy')
# train = np.load('./split/Mauzac/train_gt1.npy')

# gt_classes = decompose(gt)
# gt_1 = gt_classes[0]

# train_classes = decompose(train)
# train_1 = train_classes[0]

# verif = compare_gt_train(gt, train)
# verif1 = verif[0]
# verif2 = verif[1]
# verif3 = verif[2]
# verif4 = verif[3]
# verif5 = verif[4]
# verif6 = verif[5]
# verif7 = verif[6]
# verif8 = verif[7]
# verif9 = verif[8]
# verif10 = verif[9]
# verif11 = verif[10]
# verif12 = verif[11]

a,b,c = compare_gt_train_class(gt, train, valid,1)

