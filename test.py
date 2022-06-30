#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:42:53 2022

@author: gloupit
"""

from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from skimage import io

#Attributes Profiles
import compute_profiles

# Visualization
import seaborn as sns
import visdom

import ast
import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
    save_prediction,
    save_components,
)
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model

import argparse

import joblib

# Debug 
import pdb
import matplotlib.pyplot as plt

DATASET1 = 'Mauzac'
DATASET2 = 'Toulouse'
FOLDER = "/scratchs/gloupit/Datasets/"

# Load the datasets
mauzac_img, mauzac_gt, mauzac_LABEL_VALUES, mauzac_IGNORED_LABELS, mauzac_RGB_BANDS, mauzac_palette = get_dataset(DATASET1, FOLDER)
tlse_img, tlse_gt, tlse_LABEL_VALUES, tlse_IGNORED_LABELS, tlse_RGB_BANDS, tlse_palette = get_dataset(DATASET2, FOLDER)

pca_mauzac = PCA(n_components=4)
# pca_tlse = PCA(n_components=4)

pca_mauzac.fit(mauzac_img.reshape(-1, mauzac_img.shape[-1]))
# pca_tlse.fit(tlse_img.reshape(-1, tlse_img.shape[-1]))

# imgR_mauzac = np.dot(mauzac_img,np.transpose(pca_mauzac.components_))

mauzac_img = np.dot(mauzac_img, np.transpose(pca_mauzac.components_))
tlse_img = np.dot(tlse_img, np.transpose(pca_mauzac.components_))

# pdb.set_trace()

classes = np.unique(mauzac_gt)
for class_id in classes:
    if class_id != 0:
        sp1 = mauzac_img[mauzac_gt == class_id]
        sp2 = tlse_img[tlse_gt == class_id]
        fig = plt.figure()
        plt.scatter(sp1[:,0], sp1[:,1], color='red', alpha=0.25)
        plt.scatter(sp2[:,0], sp2[:,1], color='blue', alpha=0.25)
        plt.show()








































