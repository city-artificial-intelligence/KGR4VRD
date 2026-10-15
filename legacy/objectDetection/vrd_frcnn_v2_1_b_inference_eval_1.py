#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script evaluates the predictive performance of a set of trained
Faster RCNN ResNet50-FPN object detection model checkpoints using a single,
fixed (default) set of inference hyperparameter settings.

This script is the 2nd step, step (b), in the series of 4 related training & 
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

-------

This script uses trained Faster RCNN ResNet50-FPN object detection models
trained on VRD dataset images to do inference on other VRD dataset images
and to evaluate the predictive performance of the inference of these models.

It is designed to evaluate the predictive performance of the inference of
a range of trained models. All inference is performed using a common set
of user-defined inference-related hyperparameter settings.

This script:
* uses a trained FASTERRCNN_RESNET50_FPN (v1) model
'''

#%%

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import os
import json
import pandas as pd
import glob
import time
import sys

from vrd_dataset_frcnn import NeSy4VRD_Dataset
import vrd_utils7 as vrdu7


#%%

platform = 'hyperion'
#platform = 'MacStudio'


#%% specify the target set of model checkpoint files to be processed

# specify the filename pattern of the model checkpoint files
model_name_pattern = "vrd_frcnn_v2_1_1_checkpoint_*.pth"

# specify where the model checkpoint files are located
if platform == 'hyperion':
    #model_checkpoint_dir = os.path.join('~', 'sharedscratch', 'research', 'frcnn')
    model_checkpoint_dir = os.path.join('~', 'sharedscratch', 'research', 'frcnnKeep', 'trial_v2_1_1')
else:
    model_checkpoint_dir = os.path.join('~', 'research', 'frcnn')

model_checkpoint_dir = os.path.expanduser(model_checkpoint_dir)


#%%

redirect_stdout = True

if redirect_stdout:
    inference_log_filename = 'vrd_frcnn_v2_1_1_inference1_1_log.txt'
else:
    inference_log_filename = ''

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(model_checkpoint_dir, inference_log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%%

print('*** FRCNN model inference run - part 1 ***')
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

print(f'Platform: {platform}')

print(f'Device: {device}')

print(f"Model checkpoint filename pattern: {model_name_pattern}")

print(f'Model checkpoint dir: {model_checkpoint_dir}')


#%% set the file path root

filepath_root = '..'


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


#%% prepare our Dataset object

# instantiate our NeSy4VRD_Dataset 
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


#%% function to save performance statistics to a file

def save_stats_to_file(filepath, stats_df):
    
    stats_df.to_csv(filepath, index=False)


#%% Specify the common set of inference hyperparameter settings

# training-only hyperparameters
# rpn_batch_size_per_image : default 256    
# bot_batch_size_per_image : default 512

# inference-only hyperparameters
# rpn_score_thresh : default 0.0
# box_score_thresh : default 0.05
# box_nms_thresh   : default 0.5
rpn_score_thresh = 0.0
box_score_thresh = 0.75
box_nms_thresh = 0.5

# training & inference hyperparameters
# box_detections_per_image : default 100
box_detections_per_image = 50

# specify the inference hyperparameter settings for this stage of 
# inference & evaluation; these are common settings that will be used
# to perform inference and evaluate predictive performance across all
# model checkpoint files being processed in a given run of this script
kwargs = {'rpn_score_thresh': rpn_score_thresh,
          'box_score_thresh': box_score_thresh,
          'box_nms_thresh': box_nms_thresh,
          'box_detections_per_image': box_detections_per_image}          

print('FRCNN model inference hyperparameter settings:')
print(f'* rpn_score_thresh: {rpn_score_thresh}')
print(f'* box_score_thresh: {box_score_thresh}')
print(f'* box_nms_thresh: {box_nms_thresh}')
print(f'* box_detections_per_image: {box_detections_per_image}')


#%% get the names of the model checkpoint files to be processed

# gather the names of all of the model checkpoint files to be processed
filepath_pattern = os.path.join(model_checkpoint_dir, model_name_pattern)
model_filename_paths = glob.glob(filepath_pattern)
model_filename_paths = sorted(model_filename_paths)

print(f'Number of model checkpoints to process: {len(model_filename_paths)}')

# set the number of images to process per model
# (= 0 means process all images; > 0 is for testing)
n_images_to_process = 0

print(f'Number of images to process per model: {n_images_to_process}')


print()
print('--- Inference run configuration end ---')
print()

if redirect_stdout:
    sys.stdout.flush()


#%% initialise dataframe for storing performance statistics

cols = ['model',
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
for item in model_filename_paths:
   
    # capture start time for current model
    start_time_model = time.time()
    
    # prepare model name
    model_name = item.removeprefix(model_checkpoint_dir)
    
    print(f"\nProcessing model: {model_name}")

    # prepare model
    model = instantiate_model(kwargs)
    load_model_checkpoint(filepath=item, model=model)
    
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
    detected_objects = vrdu7.perform_inference(model, dataset,
                                               vrd_img_names_validation, 
                                               device, 
                                               n_images_to_process)
    
    # Using the object detection results for each image, calculate the
    # detection performance statistics for each image
    statistics = vrdu7.calculate_per_image_performance_statistics(detected_objects,
                                                                  vrd_img_anno)
    
    # Using the per-image performance statistics, calculate the global
    # performance statistics for the model
    global_stats = vrdu7.calculate_global_per_model_performance_statistics(statistics)
    
    # assemble the global stats into one list representing a new row to be
    # added to our dataframe of performance statistics
    row = [model_name] + \
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
    end_time_model = time.time()
    model_time = (end_time_model - start_time_model) / 60
    print(f"Model processing time: {model_time:.2f} minutes")

    if redirect_stdout:
        sys.stdout.flush()


# save DataFrame of performance evaluation statistics to .csv file
stats_filename_base = model_name_pattern.removesuffix('*.pth')
stats_filename = stats_filename_base + 'global_stats_1.csv'
stats_filename_path = os.path.join(model_checkpoint_dir, stats_filename)
print(f"\nSaving statistics to file: {stats_filename_path}")
save_stats_to_file(stats_filename_path, df)

# print total training time (in minutes)
end_time_total = time.time()
time_total = (end_time_total - start_time_total) / 60
print(f"Total time: {time_total:.2f} minutes\n")
print()
print('Processing completed')


#%%

if redirect_stdout:
    # flush stdout buffer
    sys.stdout.flush()
    # close redirected output file
    sys.stdout.close()
    # restore sys.stdout to original file handler
    sys.stdout = stdout_file_saved

print('Processing completed')















