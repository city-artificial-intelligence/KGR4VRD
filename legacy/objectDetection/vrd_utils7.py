#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
Utility functions for object detection on the NeSy4VRD dataset.
'''

#%%

import torch
import pandas as pd
import os
import sys


#%% set the file path root

platform = 'hyperion'
#platform = 'MacStudio'

filepath_root = '..'


#%% import vrd utility function modules

if platform == 'hyperion':
    vrd_utils_dir_path = os.path.join('~','archive', 'repo-vrd-kg', 'extensibility', 'analysis')
else:
    vrd_utils_dir_path = os.path.join(filepath_root, 'extensibility', 'analysis')
vrd_utils_dir_path = os.path.expanduser(vrd_utils_dir_path)
sys.path.insert(0, vrd_utils_dir_path)

import nesy4vrd_utils as vrdu

if platform == 'hyperion':
    vrd_utils_dir_path = os.path.join('~','archive', 'repo-vrd-kg', 'predicatePrediction')
else:
    vrd_utils_dir_path = os.path.join(filepath_root, 'predicatePrediction')
vrd_utils_dir_path = os.path.expanduser(vrd_utils_dir_path)
sys.path.insert(0, vrd_utils_dir_path)

import vrd_utils5 as vrdu5


#%% function to restructure the ground-truth objects for an image

def restructure_gt_objects(gt_objects):
    
    targets = {}
    
    # extract bbox specifications into a list, and convert the specifications
    # from the VRD format [ymin, ymax, xmin, xmax] to the
    # [xmin, ymin, xmax, ymax] used by Faster R-CNN object detection models
    bbox_coords = list(gt_objects.keys())
    bboxes = []
    for bbox in bbox_coords:
        bbox2 = (bbox[2], bbox[0], bbox[3], bbox[1])
        bboxes.append(bbox2)
    
    # extract object classes into a list
    bbox_classes = list(gt_objects.values())

    targets['boxes'] = bboxes
    targets['labels'] = bbox_classes
    
    return targets


#%% function to compute F-measure metric

def calculate_F_measure(b, p, r):
    
    numerator = p * r
    denominator = b**2 * p + r
    
    if denominator == 0:
        F_b_score = 0.0
    else:
        F_b_score = (1 + b**2) * (numerator / denominator)
    
    return F_b_score


#%% function for object detection performance evaluation for one image

def calculate_performance_on_image(detections, gt_objects):
      
    # restructure the annotated (ground-truth) object bboxes so they have the 
    # same format as the detected (predicted) object bboxes
    targets = restructure_gt_objects(gt_objects)
    
    # initialise hit lists
    n_targets = len(targets['boxes'])
    n_detections = len(detections['boxes'])
    dhits = [0] * n_detections   # detection hits
    thits = [0] * n_targets      # target hits
 
    # set IoU threshold for bbox matches
    iou_thresh = 0.5
    
    # iterate over the detected objects and match the detected objects
    # (pairs of (bbox, label)) with the ground-truth objects
    for didx, dbox in enumerate(detections['boxes']):
        
        best_idx = -1
        best_iou = -1
        
        # iterate over the ground-truth (gt) pairs of (bbox, label)
        for tidx, tbox in enumerate(targets['boxes']):
            
            # ----- check if the class labels match ----- 
            
            # if the class labels do not match, the current gt pair (bbox,
            # label) cannot be a hit, so skip it and consider the next one
            if detections['labels'][didx] == targets['labels'][tidx]:
                pass
            else:
                continue
            
            # ----- check if the current gt pair has already been hit ----- 
            
            if thits[tidx] > 0:  # gt pair already hit (already matched)
                continue

            # ----- check if the bboxes match -----
        
            iou = vrdu5.bb_intersection_over_union(dbox, tbox)
            if iou > iou_thresh:
                if iou > best_iou:
                    best_iou = iou
                    best_idx = tidx
        
        # if we found a gt match for the current detected object, then we
        # have a hit, so record it
        if best_idx >= 0:
            dhits[didx] = 1
            thits[best_idx] = 1
    
    # count the hits
    n_hits = sum(dhits)
    
    # store statistics for return
    e_score = round(n_detections / n_targets, 3)
    g_score = round(n_hits / n_targets, 3)         # recall (hit rate)
    h_score = abs(1 - e_score) + (1 - g_score)
    if n_detections == 0:
        j_score = 0
    else:
        j_score = round(n_hits / n_detections, 3)      # precision
    k_score = (1 - j_score) + (1 - g_score)        # alternate to F1 metric
    F2_score = round(calculate_F_measure(2, j_score, g_score), 3)
    F3_score = round(calculate_F_measure(3, j_score, g_score), 3)
    target_ratio = 1.5
    q_score = abs(target_ratio - e_score) + (1 - g_score)
    statistics = {}
    statistics['n_targets'] = n_targets
    statistics['n_detections'] = n_detections
    statistics['n_hits'] = n_hits
    statistics['detections_to_targets_ratio'] = e_score
    statistics['recall_iou_only_hits'] = 0                # obsolete
    statistics['recall'] = g_score
    statistics['h_score'] = h_score
    statistics['precision'] = j_score
    statistics['k_score'] = k_score
    statistics['F2_score'] = F2_score
    statistics['F3_score'] = F3_score
    statistics['q_score'] = q_score
    
    return statistics


#%% function to drive calculation of performance statistics over all images

def calculate_per_image_performance_statistics(model_results, vrd_img_anno):
    
    statistics = {}
    
    # iterate over each image entry in the object detection results dictionary
    for imname, detections in model_results.items():
        
        # get the VR annotations for the current image 
        imanno = vrd_img_anno[imname]
    
        # get the annotated (ground-truth) objects for the image
        # (nb: the gt_objects dictionary has bbox tuples as 'keys' and
        #  integer VRD object class labels as 'values')
        gt_objects = vrdu.get_bboxes_and_object_classes(imname, imanno)

        # calculate the image-level object detection performance statistics
        # using the two sets of 1) predicted objects and 2) ground-truth objects
        img_stats = calculate_performance_on_image(detections, gt_objects)
        
        # save the performance statistics for the current image
        statistics[imname] = img_stats
    
    return statistics


#%% function for performing inference using a particular model

def perform_inference(model, dataset, vrd_img_names,
                      device, n_images_to_process):
    '''
    Detect objects in a set of images using a trained ODNN model.

    Parameters
    ----------
    model : nn.Module
        A trained FRCNN object detector.
    dataset : torch.Dataset
        A torch.Dataset object that retrieves images and their corresponding
        ground-truth annotated objects (bboxes and object class labels).
    vrd_img_names : List
        A list of of VRD image names. The contents of the list will either
        be the image names allocated to the NeSy4VRD validation set or the
        image names for the VRD (and NeSy4VRD) test set.
    device : torch.Device
        A PyTorch device.
    n_images_to_process : integer
        An argument to optionally limit the number of images to process
        (for development and testing purposes).

    
    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''
    
    # nb: the caller is responsible for putting the model in evaluation 
    # mode and for disabling gradient computation

    # initialise a dictionary to store the objects detected in each image
    # (this becomes a dictionary of dictionaries; one dictionary per image)
    detected_objects_per_image = {}

    for idx, imname in enumerate(vrd_img_names):
    
        # get the next image to be processed
        idx2, img, _ = dataset[idx]
        if idx2 != idx:
            raise ValueError('Problem: idx2 {idx2} != idx {idx} for image {imname}')
    
        # capture the image's size for saving to results dictionary
        img_height = img.size()[1]
        img_width = img.size()[2]

        # if we're using a GPU, the model is on the GPU; so push the image 
        # data to the GPU as well
        if device != torch.device('cpu'):
            img = img.to(device)

        # perform inference on the image (ie detect objects in the image)
        out = model([img])
        
        # extract the individual components from the output
        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']
        
        # if we're using a GPU, copy the output components to the CPU
        # before trying to manipulate them
        if device != torch.device('cpu'):
            boxes = boxes.cpu()
            labels = labels.cpu()
            scores = scores.cpu()
        
        # convert the output tensors to numpy arrays and then to lists
        # (note: the predicted bboxes have format [xmin, ymin, xmax, ymax])
        # (note: convert numpy arrays to lists so json can serialise the data)
        # (note: decrement the object class labels by 1 so they correspond to
        #  the NeSy4VRD object class labels; the NeSy4VRD Dataset class
        #  increments them by 1 for the Faster R-CNN obj detection models, so
        #  we need to decrement them here before storing the class labels
        #  predicted for the bboxes)
        boxes = boxes.detach().numpy().tolist()
        labels = labels.numpy().tolist()
        labels = [label-1 for label in labels]
        scores = scores.detach().numpy().tolist()
        
        # store the objects detected in the current image
        detected_objects_per_image[imname] = {'imwidth': img_width,
                                              'imheight': img_height,
                                              'boxes': boxes,
                                              'labels': labels,
                                              'scores': scores}
        
        # check early-stopping condition
        if n_images_to_process > 0:
            if idx >= n_images_to_process:
                break 
        
    return detected_objects_per_image


#%% function to calculate per-model global performance statistics

def calculate_global_per_model_performance_statistics(model_per_img_stats):
    
    # initialise lists for holding per-image statistics
    b_stats = []
    e_stats = []
    f_stats = []
    g_stats = []
    h_stats = []
    j_stats = []
    k_stats = []
    F2_stats = []
    F3_stats = []
    q_stats = []
    
    # iterate over the image entries in the performance statistics dictionary
    # for the current model and build lists of the statistics for the metrics
    # (e), (f) and (g) in which we are interested
    for imname, statistics in model_per_img_stats.items():   
        b_stats.append(statistics['n_detections'])
        e_stats.append(statistics['detections_to_targets_ratio'])
        f_stats.append(statistics['recall_iou_only_hits'])  # obsolete
        g_stats.append(statistics['recall'])
        h_stats.append(statistics['h_score'])
        j_stats.append(statistics['precision'])
        k_stats.append(statistics['k_score'])
        F2_stats.append(statistics['F2_score'])
        F3_stats.append(statistics['F3_score'])
        q_stats.append(statistics['q_score'])
    
    # convert the lists of image-level statistics to Pandas Series
    b_stats_series = pd.Series(b_stats)
    e_stats_series = pd.Series(e_stats)
    f_stats_series = pd.Series(f_stats)   
    g_stats_series = pd.Series(g_stats)
    h_stats_series = pd.Series(h_stats)
    j_stats_series = pd.Series(j_stats)
    k_stats_series = pd.Series(k_stats)
    F2_stats_series = pd.Series(F2_stats)
    F3_stats_series = pd.Series(F3_stats)
    q_stats_series = pd.Series(q_stats)

    # calculate summary statistics describing the distributions of
    # the samples of statistics (e), (f) and (g)
    b_distribution = b_stats_series.describe().to_list()
    e_distribution = e_stats_series.describe().to_list()
    f_distribution = f_stats_series.describe().to_list()
    g_distribution = g_stats_series.describe().to_list()
    h_distribution = h_stats_series.describe().to_list()
    j_distribution = j_stats_series.describe().to_list()
    k_distribution = k_stats_series.describe().to_list()
    F2_distribution = F2_stats_series.describe().to_list()
    F3_distribution = F3_stats_series.describe().to_list()
    q_distribution = q_stats_series.describe().to_list()
    
    # ----- round (shorten) certain values -----
    
    b_distribution[1] = round(b_distribution[1], 4)    # mean
    
    e_distribution[1] = round(e_distribution[1], 4)    # mean
    e_distribution[2] = round(e_distribution[2], 4)    # std
    e_distribution[4] = round(e_distribution[4], 4)    # 25th percentile
    e_distribution[5] = round(e_distribution[5], 4)    # 50th percentile (median)
    e_distribution[6] = round(e_distribution[6], 4)    # 75th percentile

    f_distribution[1] = round(f_distribution[1], 4)    # mean
    f_distribution[2] = round(f_distribution[2], 4)    # std
    f_distribution[4] = round(f_distribution[4], 4)    # 25th percentile
    f_distribution[5] = round(f_distribution[5], 4)    # 50th percentile (median)
    f_distribution[6] = round(f_distribution[6], 4)    # 75th percentile
      
    g_distribution[1] = round(g_distribution[1], 4)    # mean
    g_distribution[2] = round(g_distribution[2], 4)    # std
    g_distribution[4] = round(g_distribution[4], 4)
    g_distribution[5] = round(g_distribution[5], 4)
    g_distribution[6] = round(g_distribution[6], 4)
    
    h_distribution[1] = round(h_distribution[1], 4)    # mean

    j_distribution[1] = round(j_distribution[1], 4)    # mean

    k_distribution[1] = round(k_distribution[1], 4)    # mean

    F2_distribution[1] = round(F2_distribution[1], 4)    # mean

    F3_distribution[1] = round(F3_distribution[1], 4)    # mean

    q_distribution[1] = round(q_distribution[1], 4)    # mean

    # ----- package the global (per-model) performance statistics -----
    
    global_stats = {}
    global_stats['b-mean'] = b_distribution[1]
    global_stats['e-dist'] = e_distribution
    global_stats['g-dist'] = g_distribution
    global_stats['h-mean'] = h_distribution[1]
    global_stats['j-mean'] = j_distribution[1]    
    global_stats['k-mean'] = k_distribution[1]
    global_stats['F2-mean'] = F2_distribution[1]
    global_stats['F3-mean'] = F3_distribution[1]
    global_stats['q-mean'] = q_distribution[1]
    
    return global_stats






