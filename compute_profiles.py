# -*- coding: utf-8 -*-
import numpy as np
import sap

def compute_area_thresholds(spatial_resolution, number_of_thresholds):
    list_of_thresholds = []
    for i in range(1,number_of_thresholds+1):
        list_of_thresholds.append(round((1000/spatial_resolution)*i))
    return list_of_thresholds

def compute_GRAY(image):
    # new_image = np.zeros(4,image.shape(1),image.shape(2))
    new_image = np.moveaxis(image, -1, 0)
    return new_image


def compute_AP(image, lamb_area, lamb_moi, adj):
    nb_thresholds_area = len(lamb_area)
    nb_thresholds_moi = len(lamb_moi)
    AP_area = np.zeros((nb_thresholds_area*2+1,image.shape[0],image.shape[1],4))
    AP_moi = np.zeros((nb_thresholds_moi*2+1,image.shape[0],image.shape[1],4))

    for i in range(0,4):
        AP_area[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj).data
        AP_moi[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive').data
    final_AP = np.concatenate([AP_area[:,:,:,0],AP_moi[:,:,:,0]],axis=0)
    for i in range(1,4):
        final_AP = np.concatenate([final_AP,AP_area[:,:,:,i],AP_moi[:,:,:,i]],axis=0)    

    return final_AP

def compute_MAX(image, lamb_area, lamb_moi, adj):
    nb_thresholds_area = len(lamb_area)
    nb_thresholds_moi = len(lamb_moi)
    AP_area = np.zeros((nb_thresholds_area*2+1,image.shape[0],image.shape[1],4))
    AP_moi = np.zeros((nb_thresholds_moi*2+1,image.shape[0],image.shape[1],4))

    for i in range(0,4):
        AP_area[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj).data
        AP_moi[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive').data
    final_AP_max = np.concatenate([AP_area[nb_thresholds_area:nb_thresholds_area*2+1,:,:,0],AP_moi[nb_thresholds_moi:nb_thresholds_moi*2+1,:,:,0]],axis=0)
    for i in range(1,4):
        final_AP_max = np.concatenate([final_AP_max,AP_area[nb_thresholds_area:nb_thresholds_area*2+1,:,:,i],AP_moi[nb_thresholds_moi:nb_thresholds_moi*2+1,:,:,i]],axis=0)

    return final_AP_max

def compute_MIN(image, lamb_area, lamb_moi, adj):
    nb_thresholds_area = len(lamb_area)
    nb_thresholds_moi = len(lamb_moi)
    AP_area = np.zeros((nb_thresholds_area*2+1,image.shape[0],image.shape[1],4))
    AP_moi = np.zeros((nb_thresholds_moi*2+1,image.shape[0],image.shape[1],4))

    for i in range(0,4):
        AP_area[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj).data
        AP_moi[:,:,:,i] = sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive').data
    final_AP_min = np.concatenate([AP_area[0:nb_thresholds_area+1,:,:,0],AP_moi[0:nb_thresholds_moi+1,:,:,0]],axis=0)
    for i in range(1,4):
        final_AP_min = np.concatenate([final_AP_min,AP_area[0:nb_thresholds_area+1,:,:,i],AP_moi[0:nb_thresholds_moi+1,:,:,i]],axis=0)

    return final_AP_min


def compute_SDAP(image, lamb_area, lamb_moi, adj):
    SDAP_area = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    SDAP_moi = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
    SDAP = sap.concatenate((SDAP_area,SDAP_moi))
    
    for i in range(1,4):
        SDAP_area = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)
        SDAP_moi = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
        SDAP = sap.concatenate((SDAP,SDAP_area,SDAP_moi))
    
    final_SDAP = sap.vectorize(SDAP)
    
    return final_SDAP

def compute_ALPHA(image, lamb_area, lamb_moi, adj):
    ALPHA_area = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    ALPHA_moi = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
   
    ALPHA_profile = sap.concatenate((ALPHA_area,ALPHA_moi))
    
    for i in range(1,4):
        ALPHA_area = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)
        ALPHA_moi = sap.profiles.alpha_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
        ALPHA_profile = sap.concatenate((ALPHA_profile,ALPHA_area,ALPHA_moi))
    
    final_ALPHA = sap.vectorize(ALPHA_profile)
    
    return final_ALPHA

def compute_OMEGA(image, lamb_area, lamb_moi, adj):
    OMEGA_area = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    OMEGA_moi = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
    OMEGA_profile = sap.concatenate((OMEGA_area,OMEGA_moi))
    
    for i in range(1,4):
        OMEGA_area = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)
        OMEGA_moi = sap.profiles.omega_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj,filtering_rule='subtractive')
        OMEGA_profile = sap.concatenate((OMEGA_profile,OMEGA_area,OMEGA_moi))
    
    final_OMEGA = sap.vectorize(OMEGA_profile)
    
    return final_OMEGA

def compute_LFAP(image, lamb_area, lamb_moi, adj):
    AP = sap.attribute_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    for i in range(1,4):
        AP = sap.concatenate((AP, sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)))
    for i in range(0,4):
        AP = sap.concatenate((AP, sap.attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive')))
    
    LFAP = sap.local_features(AP, local_feature=(np.mean, np.std), patch_size=7) #patch_size changed from 5 to 7 to match with the FP article
    LFAP = sap.vectorize(LFAP)
    
    return LFAP

def compute_LFSDAP(image, lamb_area, lamb_moi, adj):
    SDAP = sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, adjacency=adj)
    for i in range(1,4):
        SDAP = sap.concatenate((SDAP, sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, adjacency=adj)))
    for i in range(0,4):
        SDAP = sap.concatenate((SDAP, sap.self_dual_attribute_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi},adjacency=adj,filtering_rule='subtractive')))
    
    LFSDAP = sap.local_features(SDAP, local_feature=(np.mean, np.std), patch_size=7) #patch_size changed from 5 to 7 to match with the FP article
    LFSDAP = sap.vectorize(LFSDAP)
    
    return LFSDAP

def compute_FP(image, lamb_area, lamb_moi, adj, method):
    if (method == "FP_AREA_MEAN"):
        out_feature = {'mean_vertex_weights','area'}
    elif (method == "FP_AREA"):
        out_feature = {'area'}
    elif (method == "FP_MEAN"):
        out_feature = {'mean_vertex_weights'}
        
    FP_area = sap.feature_profiles(np.ascontiguousarray(image[:,:,0]), {'area': lamb_area}, out_feature=out_feature,adjacency=adj)
    FP_moi = sap.feature_profiles(np.ascontiguousarray(image[:,:,0]), {'moment_of_inertia': lamb_moi}, adjacency=adj, out_feature=out_feature,filtering_rule='subtractive')
    FP = sap.concatenate((FP_area,FP_moi))
    for i in range(1,4):
        FP_area = sap.feature_profiles(np.ascontiguousarray(image[:,:,i]), {'area': lamb_area}, out_feature=out_feature, adjacency=adj)
        FP_moi = sap.feature_profiles(np.ascontiguousarray(image[:,:,i]), {'moment_of_inertia': lamb_moi}, adjacency=adj, out_feature=out_feature,filtering_rule='subtractive')
        FP = sap.concatenate((FP,FP_area,FP_moi))
        
    final_FP = sap.vectorize(FP)
    
    return final_FP