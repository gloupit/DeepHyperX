from utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "DFC2018_HSI": {
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "download": False,
        "loader": lambda folder: dfc2018_loader(folder),
    },
    "Mauzac": {
        "img": "mauzac_img.npy",
        "gt": "mauzac_gt.npy",
        "download": False,
        "loader": lambda folder: mauzac_loader(folder),
    },
    "Mauzac_maison_1": {
        "img": "mauzac_maison_1_img.npy",
        "gt": "mauzac_maison_1_gt.npy",
        "download": False,
        "loader": lambda folder: mauzac_maison_1_loader(folder),
    },
    "Mauzac_quartier": {
        "img": "mauzac_quartier_img.npy",
        "gt": "mauzac_quartier_gt.npy",
        "download": False,
        "loader": lambda folder: mauzac_quartier_loader(folder),
    },
    "Toulouse": {
        "img": "toulouse_img.npy",
        "gt": "toulouse_gt.npy",
        "download": False,
        "loader": lambda folder: toulouse_loader(folder),
    }
}


def dfc2018_loader(folder):
    img = open_file(folder + "2018_IEEE_GRSS_DFC_HSI_TR.HDR")[:, :, :-2]
    gt = open_file(folder + "2018_IEEE_GRSS_DFC_GT_TR.tif")
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Healthy grass",
        "Stressed grass",
        "Artificial turf",
        "Evergreen trees",
        "Deciduous trees",
        "Bare earth",
        "Water",
        "Residential buildings",
        "Non-residential buildings",
        "Roads",
        "Sidewalks",
        "Crosswalks",
        "Major thoroughfares",
        "Highways",
        "Railways",
        "Paved parking lots",
        "Unpaved parking lots",
        "Cars",
        "Trains",
        "Stadium seats"
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette


def mauzac_loader(folder):
    img = open_file(folder + "mauzac_img.npy")
    gt = open_file(folder + "mauzac_gt.npy")
    gt = gt.astype("uint8")

    rgb_bands = (70, 50, 25)

    label_values = [
        'Untitled', 
        'Vegetation shadows', 
        'High vegetation', 
        'Ground vegetation', 
        'Dry vegetation',
        'Bare soil', 
        'Water body', 
        'Swimming pool', 
        'Pool cover', 
        'Curbstone', 
        'Tile', 
        'Asphalt', 
        'Other shadows'
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette

def toulouse_loader(folder):
    img = open_file(folder + "toulouse_img.npy")
    gt = open_file(folder + "toulouse_gt.npy")
    gt = gt.astype("uint8")

    rgb_bands = (70, 50, 25)

    label_values = [
        'Untitled', 
        'Vegetation shadows', 
        'High vegetation', 
        'Ground vegetation', 
        'Dry vegetation',
        'Bare soil', 
        'Water body', 
        'Swimming pool', 
        'Pool cover', 
        'Curbstone', 
        'Tile', 
        'Asphalt', 
        'Other shadows'
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette

def mauzac_maison_1_loader(folder):
    img = open_file(folder + "mauzac_maison_1_img.npy")
    gt = open_file(folder + "mauzac_maison_1_gt.npy")
    gt = gt.astype("uint8")

    rgb_bands = (70, 50, 25)

    label_values = [
        'Untitled', 
        'Vegetation shadows', 
        'High vegetation', 
        'Ground vegetation', 
        'Dry vegetation',
        'Bare soil', 
        'Water body', 
        'Swimming pool', 
        'Pool cover', 
        'Curbstone', 
        'Tile', 
        'Asphalt', 
        'Other shadows'
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette

def mauzac_quartier_loader(folder):
    img = open_file(folder + "mauzac_quartier_img.npy")
    gt = open_file(folder + "mauzac_quartier_gt.npy")
    gt = gt.astype("uint8")

    rgb_bands = (70, 50, 25)

    label_values = [
        'Untitled', 
        'Vegetation shadows', 
        'High vegetation', 
        'Ground vegetation', 
        'Dry vegetation',
        'Bare soil', 
        'Water body', 
        'Swimming pool', 
        'Pool cover', 
        'Curbstone', 
        'Tile', 
        'Asphalt', 
        'Other shadows'
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
