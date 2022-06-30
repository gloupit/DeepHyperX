#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:24:54 2022

@author: gloupit
"""

from museotoolbox.cross_validation import SpatialLeaveAsideOut
from museotoolbox import datasets,processing
# import pdb
# pdb.set_trace()

raster,vector = datasets.load_historical_data()
field = 'Class'
# raster, vector et field sont des str

X,y = processing.extract_ROI(raster,vector,field)
distance_matrix = processing.get_distance_matrix(raster,vector)

SLOPO = SpatialLeaveAsideOut(valid_size=1/3,
                              distance_matrix=distance_matrix,random_state=4)

print(SLOPO.get_n_splits(X,y))

for tr,vl in SLOPO.split(X,y):
    print(tr.shape,vl.shape)
    
processing.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
trvl = SLOPO.save_to_vector('/tmp/pixels.gpkg',field,out_vector='/tmp/SLOPO.gpkg')
for tr,vl in trvl:
    print(tr,vl)