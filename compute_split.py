# -*- coding: utf-8 -*-

from museotoolbox.cross_validation import SpatialLeaveAsideOut
from museotoolbox import datasets,processing
import numpy as np
import scipy.spatial.distance as distance

if 1==1:
    #Mauzac
    gt = np.load('/tmp_user/ldota704h/gloupit/Datasets/Mauzac/mauzac_gt.npy')
    folder = "/scratchs/gloupit/Datasets/Mauzac/"
    name = "mauzac_"
else:
    #Toulouse
    gt = np.load('/tmp_user/ldota704h/gloupit/Datasets/Toulouse/toulouse_gt.npy')
    folder = "/scratchs/gloupit/Datasets/Toulouse/"
    name = "toulouse_"

#gt.shape (h,w)
coords = np.where(gt != 0)
#coords --> tuple des coordonnées des pixels annotés
#coords[0] --> npy.array de taille n_samples
#coords[1] --> idem
coords = np.array([coords[0], coords[1]])
coordsT = coords.T
d = np.asarray(distance.cdist(coordsT, coordsT, 'euclidean'), dtype=np.uint64)
SLOPO = SpatialLeaveAsideOut(valid_size=1/5,distance_matrix=d,random_state=4)

y = np.zeros(coords.shape[1])
y[:] = gt[coords[0],coords[1]] #valeur de gt pour tous les pixels annotés


i = 1
for tr,vl in SLOPO.split(coords,y):
    print(tr.shape,vl.shape)
    
    coords_split = tuple((coordsT[tr,[0]], coordsT[tr,[1]]))
    train_split = np.zeros_like(gt)
    train_split[coords_split] = gt[coords_split]
    
    coords_split_vl = tuple((coordsT[vl,[0]], coordsT[vl,[1]]))
    val_split = np.zeros_like(gt)
    val_split[coords_split_vl] = gt[coords_split_vl]
    
    # np.save('./split/Mauzac/train_gt{}.npy'.format(i), train_split)
    # np.save('./split/Mauzac/val_gt{}.npy'.format(i), val_split)
    # np.save('./split/Toulouse/train_gt{}.npy'.format(i), train_split)
    # np.save('./split/Toulouse/val_gt{}.npy'.format(i), val_split)
    np.save(folder + name + 'train_gt{}.npy'.format(i), train_split)
    np.save(folder + name + 'val_gt{}.npy'.format(i), val_split)
    i += 1
    
# train1 = np.load('./split/Toulouse/train_gt1.npy')



# tr.shape (n_samples) indices entre 0 et n_samples-1
# coords_split = tuple((coordsT[tr][0], coordsT[tr][1]))

# train_split = np.zeros_like(gt)
# train_split[coords_split] = gt[coords_split]
# np.where(train_split == 4 )


































