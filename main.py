# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
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

# museotoolbox
# from museotoolbox.cross_validation import SpatialLeaveAsideOut
# from museotoolbox import datasets,processing

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
    spatial_k_fold,
)
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model

import argparse

import joblib

# Debug 
import pdb

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use."
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model to train. Available:\n"
    "SVM (linear), "
    "SVM_grid (grid search on linear, poly and RBF kernels), "
    "baseline (fully connected NN), "
    "hu (1D CNN), "
    "hamida (3D CNN + 1D classifier), "
    "lee (3D FCN), "
    "chen (3D CNN), "
    "li (3D CNN), "
    "he (3D CNN), "
    "luo (3D CNN), "
    "sharma (2D CNN), "
    "boulch (1D semi-supervised CNN), "
    "liu (3D semi-supervised CNN), "
    "mou (1D RNN)",
)

parser.add_argument(
    "--method", 
    type=str, 
    default='GRAY', 
    help="Method to use with AP model (Attributes Profiles):\n"
    "GRAY (only 4 PCA)"
    "AP"
    "MAX"
    "MIN"
    "SDAP"
    "LFAP"
    "LFSDAP"
    "HAP"
    "HSDAP"
    "ALPHA"
    "OMEGA"
    "FP_AREA or FP_AREA_MEAN or FP_MEAN"
)

parser.add_argument(
    "--adj",
    type=int,
    help="Adjacency (for AP model), by default 4, can be 8",
    default=4,
)

parser.add_argument(
    "--quantization",
    type=int,
    help="Quantization (for AP model) by default 8, can be 8",
    default=8,
)

parser.add_argument(
    "--nb_tree",
    type=int,
    help="Number of Tree (for AP model) by default 100",
    default=100,
)

parser.add_argument(
    "--nb_PCA",
    type=int,
    help="Number of PCA (for AP model) by default 4",
    default=4,
)

parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="/scratchs/gloupit/Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

parser.add_argument(
    "--restore_pca",
    type=str,
    default=None,
    help="PCA to use for testing",
)

parser.add_argument(
    "--only_test",
    action="store_true",
    # type=bool,
    # default=False,
    help="For applying a test without training",
)

parser.add_argument(
    "--cv",
    action="store_true",
    # type=bool,
    # default=False,
    help="For using Cross Validation and GridSearch for RF",
)

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=0.2,
    help="Percentage of samples to use for training (default: 10%%)",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="disjoint",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    help="Training epochs (optional, if" " absent will be set by the model)",
)
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, help="Learning rate, set by the model if not specified."
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
group_train.add_argument(
    "--pretrained_indices",
    type=str,
    default='[]',
    help="Paths to pretrained indices for HSI model",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Method name (in case of AP model)
METHOD = args.method
# Adjacency (in case of AP model)
ADJ = args.adj
# Quantization (in case of AP model)
QUANTIZATION = args.quantization
# Number of Trees (in case of AP model)
NB_TREE = args.nb_tree
# Number of PCA (in case of AP model)
NB_PCA = args.nb_PCA
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Pre-computed PCA
PCA_COMPONENTS = args.restore_pca
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set 
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
#Only testing
ONLY_TEST = args.only_test
#CrossValidation
CV = args.cv
    

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

if MODEL == 'AP':
    viz = visdom.Visdom(env=DATASET + " " + MODEL + " " + METHOD)
else:
    viz = visdom.Visdom(env=DATASET + " " + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [
    {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [3], "gamma": [1e-1, 1e-2, 1e-3]},
]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)

#Paths for model selection
path = FOLDER + DATASET + "/split/"
train_split_paths = [path + "train_gt1.npy",
                     path + "train_gt2.npy",
                     path + "train_gt3.npy",
                     path + "train_gt4.npy",
                     path + "train_gt5.npy"]

val_split_paths =   [path + "val_gt1.npy",
                     path + "val_gt2.npy",
                     path + "val_gt3.npy",
                     path + "val_gt4.npy",
                     path + "val_gt5.npy"]

# Instantiate the experiment based on predefined networks
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
        "pretrained_indices": ast.literal_eval(args.pretrained_indices)
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(
        img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
    )
    plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

results = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    if ONLY_TEST:
        test_gt = gt
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(train_gt), np.count_nonzero(gt)
        )
    )
    print(
        "Running an experiment with the {} model".format(MODEL),
        "run {}/{}".format(run + 1, N_RUNS),
    )
    
    if not ONLY_TEST:
        display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

    if MODEL == "SVM_grid":
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(
            clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "SVM":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'AP':
        
        # method = 'FP_MEAN'
        nb_PCA = NB_PCA
        adj = ADJ
        quantization = QUANTIZATION
        nb_tree = NB_TREE
        spatial_resolution = 0.55
        nb_thresholds = 14
        
        d1, d2, d3 = img.shape
        
        #Preprocessing
        pca = PCA(n_components=nb_PCA)
        pca.fit(img.reshape(-1, img.shape[-1]))
        # img_reshape = img.reshape(-1, img.shape[-1])
        # pdb.set_trace()
        
        if PCA_COMPONENTS != None:
            pca_components = np.load(PCA_COMPONENTS)
            imgR = np.dot(img,np.transpose(pca_components))
        else:
            if not ONLY_TEST:
                save_components(pca.components_, MODEL, METHOD, DATASET)
            imgR = np.dot(img,np.transpose(pca.components_))
            
        
        img_PCA_ = np.reshape(imgR,(d1,d2,nb_PCA))
        
        #Normalization
        delta = 1 / (np.amax(img_PCA_) - np.amin(img_PCA_))
        img_PCA  = img_PCA_ * delta + (-np.amin(img_PCA_)*delta)
    
        if (quantization == 8):
            img_PCA = np.double(np.round(img_PCA*255))
        elif (quantization == 16):
            img_PCA = np.double(np.round(img_PCA*65535))
        
        lamb_area = compute_profiles.compute_area_thresholds(spatial_resolution, nb_thresholds)
        lamb_area = np.dot(lamb_area,1/30)
        # lamb_area=[770, 1538, 2307, 3076, 3846, 4615, 5384, 6153, 6923, 7692, 8461, 9230, 10000, 10769]
        lamb_moi = [0.2, 0.3, 0.4, 0.5]
        
        if   (METHOD == "GRAY"):
            features = compute_profiles.compute_GRAY(img_PCA)
        elif (METHOD == "AP"):
            features = compute_profiles.compute_AP(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "MAX"):
            features = compute_profiles.compute_MAX(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "MIN"):
            features = compute_profiles.compute_MIN(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "SDAP"):
            features = compute_profiles.compute_SDAP(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "LFAP"):
            features = compute_profiles.compute_LFAP(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "LFSDAP"):
            features = compute_profiles.compute_LFSDAP(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "HAP"):
            features = compute_profiles.compute_HAP(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "HSDAP"):
            features = compute_profiles.compute_HSDAP(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "ALPHA"):
            features = compute_profiles.compute_ALPHA(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "OMEGA"):
            features = compute_profiles.compute_OMEGA(img_PCA, lamb_area, lamb_moi, adj)
        elif (METHOD == "FP_AREA" or METHOD == "FP_AREA_MEAN" or METHOD == "FP_MEAN"):
            features = compute_profiles.compute_FP(img_PCA, lamb_area, lamb_moi, adj, METHOD)
        else:
            print("Method not implemented!")

        features = features.transpose(1, 2, 0)
        # pdb.set_trace()
        
        if not ONLY_TEST:    
            X_train, y_train = build_dataset(features, train_gt, ignored_labels=IGNORED_LABELS)
            # pdb.set_trace()
            
            if not CV:
                clf = RandomForestClassifier(n_estimators=nb_tree, random_state=0)
                clf.fit(X_train,y_train)
                save_model(clf, MODEL, DATASET)
            else:
                param_grid = {'max_depth': [10,50,100],
                      'max_features': ['sqrt', 'log2'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10],
                      'n_estimators': [10,100,200]}
                
                scores = ["precision", "recall"]
                for score in scores:
                    print("# Tuning hyper-parameters for %s" % score)
                    print()
                    truc = spatial_k_fold(train_split_paths, val_split_paths)
                    # pdb.set_trace()
                    # clf = sklearn.model_selection.GridSearchCV(RandomForestClassifier(random_state=0), param_grid , cv=spatial_k_fold(train_split_paths, val_split_paths),scoring="%s_macro" % score)
                    # clf = sklearn.model_selection.RandomizedSearchCV(RandomForestClassifier(random_state=0), param_distributions=param_grid, n_iter = 1, cv=spatial_k_fold(train_split_paths, val_split_paths),scoring="%s_macro" % score,random_state=0)
                    clf = sklearn.model_selection.RandomizedSearchCV(RandomForestClassifier(random_state=0), param_distributions=param_grid, n_iter = 1)#, cv=spatial_k_fold(train_split_paths, val_split_paths))
                    print(clf.get_params())
                    # clf.fit(X_train, y=y_train,groups=None, fit_params=clf.get_params)   #build the forest of trees
                    # pdb.set_trace()
                    clf.fit(X_train,y=y_train)
                    print("Best parameters set found on development set:")
                    print()
                    print(clf.best_params_)
                    print()
                    print("Grid scores on development set:")
                    print()
                    means = clf.cv_results_["mean_test_score"]
                    stds = clf.cv_results_["std_test_score"]
                    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                    print()
                
                    print("Detailed classification report:")
                    print()
                    print("The model is trained on the full development set.")
                    print("The scores are computed on the full evaluation set.")
                    print()
                    # y_true, y_pred = test_gt, clf.predict(features.reshape(-1, features.shape[2]))
                    # print(sklearn.metrics.classification_report(y_true, y_pred))
                    # print()
                    save_model(clf, MODEL, DATASET, score)
                
                # pdb.set_trace()
            # save_model(clf, MODEL, DATASET)
        else:
            clf = joblib.load(CHECKPOINT)
            
        prediction = clf.predict(features.reshape(-1, features.shape[2]))
        prediction = prediction.reshape(features.shape[:2])
        
    elif MODEL == "SGD":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(
            class_weight=class_weight, learning_rate="optimal", tol=1e-3, average=10
        )
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "nearest":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.neighbors.KNeighborsClassifier(weights="distance")
        clf = sklearn.model_selection.GridSearchCV(
            clf, {"n_neighbors": [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2]) 
                       
    else:
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams["weights"] = torch.from_numpy(weights)
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        # Split train set in train/val
        train_gt, val_gt = sample_gt(train_gt, 0.95, mode="random")
        # Generate the dataset
        train_dataset = HyperX(img, train_gt, **hyperparams)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            # pin_memory=hyperparams['device'],
            shuffle=True,
        )
        val_dataset = HyperX(img, val_gt, **hyperparams)
        val_loader = data.DataLoader(
            val_dataset,
            # pin_memory=hyperparams['device'],
            batch_size=hyperparams["batch_size"],
        )

        print(hyperparams)
        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            #summary(model.to(hyperparams["device"]), input.size()[1:])
            # We would like to use device=hyperparams['device'] altough we have
            # to wait for torchsummary to be fixed first.

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))
        
        if not ONLY_TEST:
            try:
                train(
                    model,
                    optimizer,
                    loss,
                    train_loader,
                    hyperparams["epoch"],
                    scheduler=hyperparams["scheduler"],
                    device=hyperparams["device"],
                    supervision=hyperparams["supervision"],
                    val_loader=val_loader,
                    display=viz,
                )
            except KeyboardInterrupt:
                # Allow the user to stop the training
                pass

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        
        
    prediction_total = np.copy(prediction)
    
    run_results = metrics(
        prediction,
        test_gt,
        ignored_labels=hyperparams["ignored_labels"],
        n_classes=N_CLASSES,
    )

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    color_prediction_total = convert_to_color(prediction_total)
    
    display_predictions(
        color_prediction,
        viz,
        gt=convert_to_color(test_gt),
        caption="Prediction vs. test ground truth",
    )
    
    display_predictions(
        color_prediction_total,
        viz,
        gt=convert_to_color(gt),
        caption="Prediction total vs. ground truth",
    )
    
    diff_prediction = np.where((color_prediction-convert_to_color(gt)) !=0, color_prediction, np.nan)
    diff_gt = np.where((color_prediction-convert_to_color(gt)) !=0, convert_to_color(gt), np.nan)
    # diff_prediction = np.where((prediction-gt) !=0, prediction, np.nan)
    # diff_gt = np.where((prediction-gt) !=0, gt, np.nan)
    
    display_predictions(
        diff_prediction,
        viz,
        diff_gt,
        caption="(Difference between) Prediction vs. ground truth",
    )
    
    save_prediction(prediction, "prediction" ,MODEL, DATASET, METHOD)
    save_prediction(np.where((prediction-gt) !=0, prediction, np.nan), "diff_prediction" ,MODEL, DATASET, METHOD)

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES)

if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
