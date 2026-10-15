#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script performs inference on and evaluates the predictive performance of 
a given trained Faster RCNN ResNet50-FPN object detection model checkpoint
against VRD mages.

The inference is performed with respect to a particular FRCNN inference
hyperparameter configuration, as selected using script:
vrd_frcnn_v2_1_c_inference_eval_2.py

It is the 4th script in the sequence of scripts:
    a) vrd_frcnn_v2_1_a_training.py  
    b) vrd_frcnn_v2_1_b_inference_eval_1.py
    c) vrd_frcnn_v2_1_c_inference_eval_2.py
    d) vrd_frcnn_v2_1_d_inference_eval_3.py

The script is used to perform inference on VRD training and test sets.
'''

#%%

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import os
import json
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


#%% specify the Inference Hyperparameter Configuration id

ihc_id = 4


#%%

redirect_stdout = True

if redirect_stdout:
    inference_log_filename = 'vrd_frcnn_v2_1_1_inference3_ihc_' + str(ihc_id) + '_trainset_log.txt'
    #inference_log_filename = 'vrd_frcnn_v2_1_1_inference3_ihc_' + str(ihc_id) + '_testset_log.txt'
else:
    inference_log_filename = ''

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(model_checkpoint_dir, inference_log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%%

print('*** FRCNN model inference run - part 3 (training set) ***')
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

print(f'Platform: {platform}')

print(f'Device: {device}')

print(f'Model checkpoint dir: {model_checkpoint_dir}')

print(f'Model checkpoint name: {model_checkpoint_name_full}')


#%% apply the inference hyperparameter configuration (IHC) specified above

# set the values of the FRCNN model inference hyperparameters


if ihc_id == 1:
    rpn_score_thresh = 0.0
    box_score_thresh = 0.6
    box_nms_thresh = 0.3
    box_detections_per_image = 8
elif ihc_id == 2:
    rpn_score_thresh = 0.0
    box_score_thresh = 0.5
    box_nms_thresh = 0.3
    box_detections_per_image = 10
elif ihc_id == 3:
    rpn_score_thresh = 0.0
    box_score_thresh = 0.4
    box_nms_thresh = 0.3
    box_detections_per_image = 20
elif ihc_id == 4:
    rpn_score_thresh = 0.0
    box_score_thresh = 0.3
    box_nms_thresh = 0.4
    box_detections_per_image = 25
else:
    raise ValueError('ihc_id not recognised')

print(f'FRCNN model inference hyperparameter configuration (IHC): {ihc_id}')
print(f'* rpn_score_thresh: {rpn_score_thresh}')
print(f'* box_score_thresh: {box_score_thresh}')
print(f'* box_nms_thresh: {box_nms_thresh}')
print(f'* box_detections_per_image: {box_detections_per_image}')


#%% set the directory containing the VRD dataset test images

if platform == 'hyperion':
    vrd_img_dir = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'train_images')
    #vrd_img_dir = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'test_images')
else:
    vrd_img_dir = os.path.join(filepath_root, 'data', 'train_images')
    #vrd_img_dir = os.path.join(filepath_root, 'data', 'test_images')

vrd_img_dir = os.path.expanduser(vrd_img_dir)

print(f'Image dir: {vrd_img_dir}')


#%% get the NeSy4VRD VR annotations for the names of the VRD images

if platform == 'hyperion':
    vrd_anno_file = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'annotations', 
                                 'nesy4vrd_annotations_train.json')
    #vrd_anno_file = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'annotations', 
    #                             'nesy4vrd_annotations_test.json')
else:
    vrd_anno_file = os.path.join(filepath_root, 'data', 'annotations', 
                                 'nesy4vrd_annotations_train.json')
    #vrd_anno_file = os.path.join(filepath_root, 'data', 'annotations', 
    #                             'nesy4vrd_annotations_test.json')

vrd_anno_file = os.path.expanduser(vrd_anno_file)

# load the VR annotations
with open(vrd_anno_file, 'r') as fp:
    vrd_img_anno = json.load(fp)

print(f'Annotations file: {vrd_anno_file}')

vrd_img_names = list(vrd_img_anno.keys())

print(f'Number of VRD images to be processed: {len(vrd_img_names)}')


#%% prepare our Dataset object

# instantiate our NeSy4VRD_Dataset 
dataset = NeSy4VRD_Dataset(vrd_img_dir=vrd_img_dir, 
                           nesy4vrd_anno_file=vrd_anno_file,
                           img_names=vrd_img_names)

print('NeSy4VRD Dataset class instantiated')

# Note:
# when called, a VRDDataset object returns a 3-tuple: (idx, img, targets)
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


#%% function to save detected objects

def save_detected_objects_to_file(filepath, detected_objects):
    
    with open(filepath, 'w') as fp:
        json.dump(detected_objects, fp)
    
    return None


#%% specify the common set of inference hyperparameter settings

# package the FRCNN model inference hyperparameters for 
# instantiating the FRCNN model
kwargs = {'rpn_score_thresh': rpn_score_thresh,
          'box_score_thresh': box_score_thresh,
          'box_nms_thresh': box_nms_thresh,
          'box_detections_per_image': box_detections_per_image}          


#%% 

# set the number of images to process per model
# (= 0 means process all images; > 0 is for testing)
n_images_to_process = 0

print(f'Number of images to process per model: {n_images_to_process}')


#%%

print()
print('--- Inference run configuration end ---')
print()

if redirect_stdout:
    sys.stdout.flush()


#%% main processing cell

start_time_total = time.time()

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
print("Performing object detection inference")
detected_objects = vrdu7.perform_inference(model, dataset, 
                                           vrd_img_names, 
                                           device, n_images_to_process)

# save the detected objects to a file for later downstream processing
filename = 'vrd_frcnn_v2_1_1_checkpoint_250_ihc_' + str(ihc_id) + \
           '_trainset_detected_objects.json'
#filename = 'vrd_frcnn_v2_1_1_checkpoint_250_ihc_' + str(ihc_id) + \
#           '_testset_detected_objects.json'
filepath = os.path.join(model_checkpoint_dir, filename)
save_detected_objects_to_file(filepath, detected_objects)
print(f"Detected objects saved to file: {filepath}")

# using the object detection results for each image, calculate the
# detection performance statistics for each image
print("Calculating per-image performance statistics")
statistics = vrdu7.calculate_per_image_performance_statistics(detected_objects,
                                                              vrd_img_anno)
        
# Using the per-image performance statistics, calculate the global
# performance statistics for the model for the current Inf2 trial
#print("Computing per-model performance statistics")
global_stats = vrdu7.calculate_global_per_model_performance_statistics(statistics)

print()
print(f'FRCNN object detection predictive performance on full train set for IHC: {ihc_id}')
print()

print(f"b-mean: {global_stats['b-mean']}")     # mean detections per image
print(f"e-mean: {global_stats['e-dist'][1]}")  # mean pred-to-target ratio per image
print(f"g-mean: {global_stats['g-dist'][1]}")  # mean recall per image
print(f"j-mean: {global_stats['j-mean']}")     # mean precision per image
print(f"F2-mean: {global_stats['F2-mean']}")   # mean F2 score per image
print(f"F3-mean: {global_stats['F3-mean']}")   # mean F3 score per image
print(f"q-mean: {global_stats['q-mean']}")     # mean q per image
print()

# print total inference processing time (in minutes)
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




