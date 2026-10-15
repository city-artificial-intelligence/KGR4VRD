#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script evalutates the predictive performance of a single, specific trained 
Faster RCNN ResNet50-FPN object detection model using a range of different
user-defined inference hyperparameter settings.

This script is the 3rd step, step (c), in the series of 4 related training & 
inference/evaluation scripts described here:

a) vrd_frcnn_v2_1_a_training.py
   - trains and saves multiple model checkpoint files for a fixed configuration of
     training hyperparameter settings

b) vrd_frcnn_v2_1_b_inference_eval_1.py
   - performs inference and evaluates performance on a range of related trained model
     checkpoint files produced by (a), all models of which share a common naming
     pattern that varies only in terms of the number of epochs used for training;
     (ie where only the value of Z changes); all inference is done using a single,
     fixed (default) configuration of inference hyperparameter settings
   - produces a .csv with global performance statistics for each model checkpoint file
     processed 
   - the analyst reviews the global performance statistics across the range of model
     checkpoint files and selects the best performing model for subsequent tuning
     in step (c)

c) vrd_frcnn_v2_1_c_inference_eval_2.py
   - facilitates the tuning of the inference hyperparameter settings of 
     FASTERRCNN_RESNET50_FPN models, for a given, single 'best performing' model
     checkpoint selected by the analyst in step (b)
   - multiple successive runs of this script on a single 'best performing' model
     checkpoint are required; each run performs a grid search over a range of
     user-defined settings for a range of user-selected inference hyperparameters
   - each unique configuration of inference hyperparameter settings is referred to
     as a 'trial set'; each run explores multiple related 'trial sets'
   - for each unique trial set of inference hyperparameter settings, inference is
     performed and predictive performance is evaluated
   - each run of the script produces a .csv reporting performance stats unique to
     each unique trial set of inference hyperparameter settings
   - the analyst reviews the predictive performance statistics in a trial set stats
     .csv, analyses the inference hyperparameter settings for the best performing
     trial sets, and uses the observed patterns to refine and shape the range of 
     inference hyperparameter settings to be explored in the next run of the script
   - the analyst iterates through this process of progressively tuning the inference
     hyperparameter settings using successive runs of this script until predictive
     performance is judged to have converged
   - the 'optimal' set of inference hyperparameter settings is selected and 
     preserved

d) vrd_frcnn_v2_1_d_inference_eval_3.py
   - performs inference on and evaluates the predictive performance of a given
     trained Faster RCNN ResNet50-FPN object detection model checkpoint against
     VRD images
   - The inference is performed with respect to a particular FRCNN inference
     hyperparameter configuration, as selected using script 
     vrd_frcnn_v2_1_c_inference_eval_2.py

--------

This script uses a trained Faster RCNN ResNet50-FPN object detection model
trained on VRD dataset images to do inference on other VRD dataset images
and to evaluate the predictive performance of the inference.

It is designed to facilitate the tuning of the inference-related 
hyperparameters of a trained Faster RCNN ResNet50-FPN object detection 
model. It operates on a single trained object detection model and explores
the inference/performance effects of a user-defined range of inference-related
hyperparameter settings on this single trained model.
'''

#%%

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import os
import json
import pandas as pd
import time
import sys

from vrd_dataset_frcnn import NeSy4VRD_Dataset
import vrd_utils7 as vrdu7


#%%

platform = 'hyperion'
#platform = 'MacStudio'

filepath_root = '..'


#%% specify model checkpoint to process

model_checkpoint_name_base = 'vrd_frcnn_v2_1_1_checkpoint_250'

model_checkpoint_name_full = model_checkpoint_name_base + '.pth'

# specify where the model checkpoint files are located
if platform == 'hyperion':
    #model_checkpoint_dir = os.path.join('~', 'sharedscratch', 'research', 'frcnn')
    model_checkpoint_dir = os.path.join('~', 'sharedscratch', 'research', 'frcnnKeep', 'trial_v2_1_1')
else:
    #model_checkpoint_dir = os.path.join('~', 'research', 'frcnn')
    model_checkpoint_dir = os.path.join('~', 'research', 'frcnnKeep', 'trial_v2_1_1')

model_checkpoint_dir = os.path.expanduser(model_checkpoint_dir)

model_checkpoint_path = os.path.join(model_checkpoint_dir, model_checkpoint_name_full)


#%% specify the trial set id

inf2_trial_set = 4


#%%

redirect_stdout = True

if redirect_stdout:
    inference_log_filename = 'vrd_frcnn_v2_1_1_inference2_' + str(inf2_trial_set) +'_log.txt'
else:
    inference_log_filename = ''

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(model_checkpoint_dir, inference_log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%%

print('*** FRCNN model inference run - part 2 ***')
print()
print('--- Inference run configuration start ---')
print()


#%%

device = torch.device('cpu')

#device = torch.device('mps')

if torch.cuda.is_available():
    device = torch.device('cuda')

if platform == 'hyperion' and device != torch.device('cuda'):
    print('cuda device on hyperion not allocated as expected')
    raise Exception('cuda device on hyperion not allocated')


#%%

print(f"Inference2 hyperparameter trial set: {inf2_trial_set}")

print(f'Platform: {platform}')

print(f'Device: {device}')

print(f'Model checkpoint dir: {model_checkpoint_dir}')

print(f'Model checkpoint name: {model_checkpoint_name_full}')


#%% configure the hyperparameter values for grid search (trial set)

rpn_scores = [0.0, 0.1]
n_rpn_scores = len(rpn_scores)

box_scores = [0.6, 0.7, 0.8]
n_box_scores = len(box_scores)

box_nms = [0.1, 0.2, 0.3]
n_box_nms = len(box_nms)

box_dets = [4, 8, 12]
n_box_dets = len(box_dets)

print('Inf2 trial set hyperparameter configuration:')
print(f'* rpn_scores: {rpn_scores}')
print(f'* box_scores: {box_scores}')
print(f'* box_nms: {box_nms}')
print(f'* box_dets: {box_dets}')


#%% set the directory containing the VRD dataset training images

# The images allocated to the validation set reside within the single
# directory of VRD training images.

if platform == 'hyperion':
    vrd_img_dir = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'train_images')
else:
    vrd_img_dir = os.path.join(filepath_root, 'data', 'train_images')

vrd_img_dir = os.path.expanduser(vrd_img_dir)

print(f'Image dir: {vrd_img_dir}')


#%% get the names of the training set images allocated to the validation set

if platform == 'hyperion':
    filepath = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'annotations',
                            'nesy4vrd_image_names_train_validation.json')
else:
    filepath = os.path.join(filepath_root, 'data', 'annotations',
                            'nesy4vrd_image_names_train_validation.json')

filepath = os.path.expanduser(filepath)

# load the image names defining the validation set
with open(filepath, 'r') as fp:
    vrd_img_names_validation = json.load(fp)

print(f'Number of VRD training images for validation: {len(vrd_img_names_validation)}')


#%% get the NeSy4VRD VR annotations

# nb: the validation set was carved out of the VRD training set; the VR
# annotations for the validation set images still reside within the
# overall VR annotations file for the overall training set

if platform == 'hyperion':
    vrd_anno_file = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'annotations', 
                                 'nesy4vrd_annotations_train.json')
else:
    vrd_anno_file = os.path.join(filepath_root, 'data', 'annotations', 
                                 'nesy4vrd_annotations_train.json')

vrd_anno_file = os.path.expanduser(vrd_anno_file)

# load the VR annotations
with open(vrd_anno_file, 'r') as fp:
    vrd_img_anno = json.load(fp)

print(f'Annotations file: {vrd_anno_file}')


#%% prepare our VRD Dataset object for loading images and target annotations

# instantiate our VRD Dataset class
dataset = NeSy4VRD_Dataset(vrd_img_dir=vrd_img_dir, 
                           nesy4vrd_anno_file=vrd_anno_file,
                           img_names=vrd_img_names_validation)

print('NeSy4VRD Dataset object instantiated')

# Note:
# when called, a NeSy4VRD Dataset object returns a 3-tuple: (idx, img, targets)
# idx - the index of an image's entry within the dataset annotations
# img - an image (formatted for the Faster RCNN model)
# targets - a dictionary with keys 'boxes' and 'labels' (formatted for
#           the Faster RCNN model); during inference, however, the targets
#           are NOT passed into the model


#%% function to instantiate a Faster R-CNN ResNet-50 FPN model

def instantiate_model(kwargs):

    num_classes = 113      # nr of classes for use with VRD dataset: 112 + 1
    trainable_layers = 0   # used for training only, so value won't matter
    
    model = fasterrcnn_resnet50_fpn(weights=None,
                                    num_classes=num_classes,
                                    trainable_backbone_layers=trainable_layers,
                                    **kwargs)
    
    return model


#%% function to load trained model checkpoint file

def load_model_checkpoint(filepath, model):
    
    # load the model checkpoint file
    # (assume, for now, that we are on a platform that is not using a GPU;
    #  so load all tensors to the CPU; if we are using a GPU, the model will
    #  be pushed to the GPU elsewhere; this approach prevents the throwing of
    #  an Exception on platforms where a GPU is not available)
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    # load the parameters representing our trained object detector 
    # into our (customised) model instance
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #print(f"Model checkpoint file loaded: {filepath}")
    
    return None


#%% function to save trial performance statistics to a file

def save_trial_stats_to_file(filepath, stats_df):
    
    stats_df.to_csv(filepath, index=False)


#%% generate the set of trials (combinations of hyperparameter values)

hyperparam_trials = {}

cnt = -1
for idx1 in range(n_rpn_scores):
    for idx2 in range(n_box_scores):
        for idx3 in range(n_box_nms):
            for idx4 in range(n_box_dets):
                cnt += 1
                if cnt <= 9:
                    tid = '00' + str(cnt)
                elif cnt <= 99:
                    tid = '0' + str(cnt)
                else:
                    tid = str(cnt)
                tid = 't' + tid
                hyperparam_trials[tid] = {'rpn_score_thresh': rpn_scores[idx1],
                                          'box_score_thresh': box_scores[idx2],
                                          'box_nms_thresh': box_nms[idx3],
                                          'box_detections_per_img': box_dets[idx4]}


#%% (CAUTION) display the generated hyperparameter tuning trials

# NOTE: run this section only so we can copy the trial definitions into
# a text file for safe keeping and later analysis; prior to submitting a
# job to run the script, comment-out this section; we don't need these
# details cluttering the log file

#for trial_id, kwargs in hyperparam_trials.items():
#    print(trial_id, kwargs)


#%%

print()
print('--- Inference run configuration end ---')
print()

if redirect_stdout:
    sys.stdout.flush()


#%%

# set the number of images to process per model
# (0 means process all images)
n_images_to_process = 0


#%% initialise dataframe for storing performance statistics

cols = ['model','trial-set', 'trial',
                   'b-mean',
        'e-count', 'e-mean', 'e-std', 'e-min', 'e-25%', 'e-50%', 'e-75%', 'e-max',
        'g-count', 'g-mean', 'g-std', 'g-min', 'g-25%', 'g-50%', 'g-75%', 'g-max',
                   'h-mean',
                   'j-mean',
                   'k-mean',
                   'F2-mean',
                   'F3-mean',
                   'q-mean']

df = pd.DataFrame(columns=cols)

df_idx = 0


#%% main processing loop

start_time_total = time.time()

# iterate over the hyperparameter setting trials we wish to run
for trial_id, kwargs in hyperparam_trials.items():
   
    # capture start time for current model
    start_time_trial = time.time()
    
    print(f"\nInf2 trial: {trial_id}")

    # prepare model
    #print("Preparing model with trial hyperparameters")
    model = instantiate_model(kwargs)
    load_model_checkpoint(filepath=model_checkpoint_path, model=model)
    
    # if we have a GPU, push model to GPU
    if device != torch.device('cpu'):
        model = model.to(device)
    
    # put model in evaluation (inference) mode
    # (to disable things like Dropout and Batch Normalisation)
    model.eval()
    
    # disable gradient computation
    # (to eliminate redundant processing and reduce memory consumption)
    torch.set_grad_enabled(False)
    
    # perform inference (detect objects) with the current model
    #print("Performing object detection inference")
    results = vrdu7.perform_inference(model, dataset, 
                                      vrd_img_names_validation, 
                                      device, n_images_to_process)
    
    # Using the object detection results for each image, calculate the
    # detection performance statistics for each image
    #print("Calculating per-image performance statistics")
    statistics = vrdu7.calculate_per_image_performance_statistics(results,
                                                                  vrd_img_anno)
    
    # Using the per-image performance statistics, calculate the global
    # performance statistics for the model for the current Inf2 trial
    #print("Computing per-model performance statistics")
    global_stats = vrdu7.calculate_global_per_model_performance_statistics(statistics)
    
    # assemble the global stats into one list representing a new row to be
    # added to our dataframe of performance statistics for all trials 
    row = [model_checkpoint_name_base, inf2_trial_set, trial_id] + \
          [global_stats['b-mean']] + \
           global_stats['e-dist'] + \
           global_stats['g-dist'] + \
          [global_stats['h-mean']] + \
          [global_stats['j-mean']] + \
          [global_stats['k-mean']] + \
          [global_stats['F2-mean']] + \
          [global_stats['F3-mean']] + \
          [global_stats['q-mean']]
    
    # append new row to DataFrame
    df_idx += 1
    df.loc[df_idx] = row
    
    # measure elapsed time
    end_time_trial = time.time()
    trial_time = (end_time_trial - start_time_trial) / 60
    print(f"Trial processing time: {trial_time:.2f} minutes")
    
    if redirect_stdout:
        sys.stdout.flush()


# save DataFrame of trial statistics to .csv file
trial_stats_filename = model_checkpoint_name_base + '_trialset_' + str(inf2_trial_set) + '_stats.csv'
trial_stats_path = os.path.join(model_checkpoint_dir, trial_stats_filename)
print(f"\nSaving trial statistics to file: {trial_stats_path}")
save_trial_stats_to_file(trial_stats_path, df)

print()
print('\nProcessing complete')

# print total training time (in minutes)
end_time_total = time.time()
time_total = (end_time_total - start_time_total) / 60
print(f"Total time: {time_total:.2f} minutes\n")


#%%

if redirect_stdout:
    # flush stdout buffer
    sys.stdout.flush()
    # close redirected output file
    sys.stdout.close()
    # restore sys.stdout to original file handler
    sys.stdout = stdout_file_saved

print('Processing completed')
















