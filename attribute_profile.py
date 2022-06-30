# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 19:11:07 2020

@author: Deise Santana Maia
"""

import sys
sys.path.append('/d/gloupit/Documents/code/sap/tools')
# import pandas as pd
# from sklearn.datasets import fetch_mldata
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns

import numpy as np
# import higra as hg
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
# from ClassificationEvaluation import classificationEvaluation
import sap
# from PIL import Image
from matplotlib import colors
import sys
import time
import compute_profiles as cp
import compute_area_thresholds 



#%%

################################################################################################################   
def compute_FP(image, lamb_area, lamb_moi, adj, method, nb_PCA):
    if (method == "FP_AREA_MEAN"):
        out_feature = {'mean_vertex_weights','area'}
    elif (method == "FP_AREA"):
        out_feature = {'area'}
    elif (method == "FP_MEAN"):
        out_feature = {'mean_vertex_weights'}
        
    FP_area = sap.feature_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, out_feature=out_feature,adjacency=adj)
    FP_moi = sap.feature_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj, out_feature=out_feature,filtering_rule='subtractive')
    FP = sap.concatenate((FP_area,FP_moi))
    for i in range(1,nb_PCA):
        FP_area = sap.feature_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, out_feature=out_feature, adjacency=adj)
        FP_moi = sap.feature_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj, out_feature=out_feature,filtering_rule='subtractive')
        FP = sap.concatenate((FP,FP_area,FP_moi))
        
    final_FP = sap.vectorize(FP)
    
    return final_FP
    

# method = "FP_AREA"
# method = "FP_AREA_MEAN"
method = "FP_MEAN"
adj = 4
quantization = 8
# training_set = "STANDARD"
training_set = "New"
nb_tree = 100
nb_PCA = 4

# file = open("/d/gloupit/Documents/code/attributes-profiles-survey-source-codes/Results/pavia_"+method+"_"+str(adj)+"_"+str(quantization)+"_"+training_set+"_"+str(nb_tree)+".txt",'w')

# paviaU_mat = sio.loadmat('./Data/PaviaU/PaviaU.mat')
# paviaU_mat = sio.loadmat('/d/gloupit/Documents/code/attributes-profiles-survey-source-codes/Data/Pavia/PaviaU.mat')
# paviaU_image = paviaU_mat['paviaU']
image = np.load('/d/gloupit/Documents/code/sap/sub_samples/mauzac_vegetation_1.npy')

d1 = image.shape[0]
d2 = image.shape[1]
d3 = image.shape[2]
image_reshape = np.reshape(image, (d1*d2, d3))

pca = PCA()
pca.fit(image_reshape)
imgR = np.dot(image, np.transpose(pca.components_))
imgR = np.reshape(imgR, (d1, d2, d3))
img_PCA_ = imgR[:, :, 0:nb_PCA]

#Normalization
delta = 1 / (np.amax(img_PCA_) - np.amin(img_PCA_))
img_PCA = img_PCA_ * delta + (-np.amin(img_PCA_)*delta)

if (quantization == 8):
        img_PCA = np.double(np.round(img_PCA*255))
elif (quantization == 16):
        img_PCA = np.double(np.round(img_PCA*65535))

    # cmap = colors.ListedColormap([(0,0,0,1),(0,0,1,1),(0,0.3,1,1),(0,1,1,1),(0.3,1,0.7,1),(0,1,0,1),(0.7,1,0.3,1),(1,1,0,1),(1,0.3,0,1),(1,0,0,1)])

lamb_area = compute_area_thresholds.compute_area_thresholds(0.55, 14)
    # lamb_area=[770, 1538, 2307, 3076, 3846, 4615, 5384, 6153, 6923, 7692, 8461, 9230, 10000, 10769]
lamb_moi = [0.2, 0.3, 0.4, 0.5]

    ################################################################################################################

    # start_time=time.time()

# if (method == "GRAY"):
#         features = cp.compute_GRAY(img_PCA)
# elif (method == "AP"):
#         features = cp.compute_AP(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "MAX"):
#         features = cp.compute_MAX(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "MIN"):
#         features = cp.compute_MIN(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "SDAP"):
#         features = cp.compute_SDAP(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "LFAP"):
#         features = cp.compute_LFAP(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "LFSDAP"):
#         features = cp.compute_LFSDAP(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "HAP"):
#         features = cp.compute_HAP(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "HSDAP"):
#         features = cp.compute_HSDAP(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "ALPHA"):
#         features = cp.compute_ALPHA(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "OMEGA"):
#         features = cp.compute_OMEGA(img_PCA, lamb_area, lamb_moi, adj)
# elif (method == "FP_AREA" or method == "FP_AREA_MEAN" or method == "FP_MEAN"):
#         features = cp.compute_FP(img_PCA, lamb_area, lamb_moi, adj, method)
# else:
#         print("Method not implemented!")


features = compute_FP(img_PCA, lamb_area, lamb_moi, adj, method, nb_PCA)
#%%

