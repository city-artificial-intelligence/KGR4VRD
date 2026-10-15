#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script trains PyTorch FASTERRCNN_RESNET50_FPN models on the NeSy4VRD 
training set.

It saves model checkpoint files every N epochs, where N is usually 10. These
model checkpoint files use a naming pattern such as vrd_frcnn_v1_X_Y_checkpoint_Z.pth,
where
* X is an integer indicating a certain version of this training script
* Y is an integer that maps to a particular configuration of settings for key training 
  hyperparameters, particularly 'trainable_backbone_layers' and 'box_batch_size_per_image' 
* Z is an integer indicating the number of epochs for which a particular model checkpoint
  was trained

This script is the 1st step, step (a), in the series of 4 related training & 
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

Object detection on the customised VRD dataset.

This is a script version of Jupyter Notebook 'vrd_frcnn_2.ipynb', designed
to run on a remote server like City's Camber server cluster.

This script trains the PyTorch Torchvision implementation of a
faster-rcnn-resnet50-fpn object detection model on the VRD
(visual relationship detection) dataset whose annotations have been 
heavily customised (quality-improved / de-noised) by Dave Herron.

We start with a Torchvision Faster R-CNN ResNet50 FPN model pre-trained
on the COCO train2017 datatset. The pre-trained model is trained to detect
only the first 91 (1 'background' class, plus 90 object classes) of the
full 183 object classes in the COCO 2017 dataset. We adapt the dual
(classification and bbox regression) output layers to fit our customised
VRD dataset and then train the model further on the customised VRD dataset.

This training script:
* was coded for Torchvision 0.14
* uses the original FASTERRCNN_RESNET50_FPN model
'''

#%%

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

import os
import json
import time
import sys

from vrd_dataset_frcnn import NeSy4VRD_Dataset


#%%

platform = 'hyperion'


#%% specify model checkpoint file name pattern and storage directory

# specify the filename pattern to be shared by all model checkpoint files
# saved by the current run of this training script; the choice of filename
# pattern should correspond to (map to) a particular and unique 
# configuration of training hyperparameter settings

model_checkpoint_filename_base = "vrd_frcnn_v2_1_1_checkpoint_"

# specify where the model checkpoint files produced by the current run of
# this training script are to be saved; if we are training a given model
# using a succession of training runs (rather than one large one),
# this is also the directory from which saved model checkpoint files will be
# loaded for onward training

if platform == 'hyperion':
    model_checkpoint_dir = os.path.join('~', 'sharedscratch', 'research', 'frcnn')
else:
    model_checkpoint_dir = os.path.join('~', 'research', 'frcnn')

model_checkpoint_dir = os.path.expanduser(model_checkpoint_dir)


#%%

redirect_stdout = True

if redirect_stdout:
    training_log_filename = 'vrd_frcnn_v2_1_1_training_log_x.txt'
else:
    training_log_filename = ''

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(model_checkpoint_dir, training_log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%%

print('*** FRCNN model training run ***')
print()
print('--- Training run configuration start ---')
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

print(f"Model checkpoint filename base: {model_checkpoint_filename_base}")

print(f'Model checkpoint dir: {model_checkpoint_dir}')


#%% set the file path root

filepath_root = '..'


#%% get the NeSy4VRD object classes

# set path to NeSy4VRD object class names
if platform == 'hyperion':
    vrd_obj_file = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'annotations',
                                'nesy4vrd_objects.json')
else:
    vrd_obj_file = os.path.join(filepath_root, 'data', 'annotations',
                                'nesy4vrd_objects.json')

vrd_obj_file = os.path.expanduser(vrd_obj_file)

# get the VRD object class names 
with open(vrd_obj_file, 'r') as file:
    vrd_object_class_names = json.load(file)

# The object classes of the VRD dataset do not include a 'background' class.
# A Faster R-CNN object detection model expects a 'background' class 
# at index 0. So, if we create a new 'background' class at index 0,
# we must increment the indices of the existing VRD object classes 
# by 1.

print(f"Nr of NeSy4VRD object class names (excl. 'background'): {len(vrd_object_class_names)}")

n_nesy4vrd_object_classes = len(vrd_object_class_names) + 1

print(f"Nr of object class names for training Faster R-CNN model: {n_nesy4vrd_object_classes}")


#%% set the NeSy4VRD dataset image directory

if platform == 'hyperion':
    vrd_img_dir = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'train_images')
else:
    vrd_img_dir = os.path.join(filepath_root, 'data', 'train_images')

vrd_img_dir = os.path.expanduser(vrd_img_dir)

print(f'Image dir: {vrd_img_dir}')


#%% get the names of the training set images allocated to the (sub)training set

if platform == 'hyperion':
    filepath = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'annotations',
                            'nesy4vrd_image_names_train_training.json')
else:
    filepath = os.path.join(filepath_root, 'data', 'annotations',
                            'nesy4vrd_image_names_train_training.json')

filepath = os.path.expanduser(filepath)

# load the image names defining the (sub)training set that remains from the
# original (full) training set after having carved out a validation set
with open(filepath, 'r') as fp:
    vrd_img_names_training = json.load(fp)

print(f'Number of VRD training images to train on: {len(vrd_img_names_training)}')


#%% get the NeSy4VRD VR annotations for the training set images

if platform == 'hyperion':
    vrd_anno_file = os.path.join('~', 'archive', 'repo-vrd-kg', 'data', 'annotations', 
                                 'nesy4vrd_annotations_train.json')
else:
    vrd_anno_file = os.path.join(filepath_root, 'data', 'annotations', 
                                 'nesy4vrd_annotations_train.json')

vrd_anno_file = os.path.expanduser(vrd_anno_file)

print(f'Annotations file: {vrd_anno_file}')


#%%

# Instantiate a pre-trained model with the desired hyperparameter configuration
# - set the values of key model training hyperparameters
# - load a pre-trained Faster R-CNN ResNet50 FPN model
# - define the model checkpoint filename base that corresponds to (indicates) 
#   the particular configuration of training hyperparameter settings specified

# Definitions of selected model hyper-parameters:
# box_score_thresh : only return proposals with score > thresh (inference only)
# num_classes : nr of object classes (including a 'background' class at index 0)
# rpn_batch_size_per_image : nr anchors sampled during training of the RPN, re RPN loss calc
# box_batch_size_per_image : nr proposals sampled during training of ROI classification head

# training-only hyperparameters
# rpn_batch_size_per_image : default 256    
# box_batch_size_per_image : default 512
rpn_batch_size_per_image = 256  
box_batch_size_per_image = 512

# inference-only hyperparameters
# rpn_score_thresh : default 0.0
# box_score_thresh : default 0.05
# box_nms_thresh   : default 0.5

# training & inference hyperparameters
# box_detections_per_image : default 100
box_detections_per_image = 100

# specify the training hyperparameter settings to be used for training
kwargs = {'rpn_batch_size_per_image': rpn_batch_size_per_image,   
          'box_batch_size_per_image': box_batch_size_per_image,
          'box_detections_per_image': box_detections_per_image}

# note: we use a much lower setting for 'box_detections_per_image'
# during inference & evaluation

# nb: we must set num_classes=91 here, because that's the number of classes
# used when the model was pretrained; so the pre-trained weights will
# only fit (be compatible with) a model configured for 91 classes
num_classes = 91

# nb: the number of trainable backbone layers is a key training hyperparameter;
# experience shows it has a huge influence on ultimate predictive performance
trainable_bbone_layers = 0

# prepare the pre-trained weights ready for model instantiation
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

# instantiate a model with the specified training hyperparameter settings and
# the pre-trained weights
model = fasterrcnn_resnet50_fpn(weights=weights,
                                num_classes=num_classes,
                                trainable_backbone_layers=trainable_bbone_layers,
                                **kwargs)

print('Model configured and initialised with pre-trained weights')
print('FRCNN model training hyperparameter settings:')
print(f'* rpn_batch_size_per_image: {rpn_batch_size_per_image}')
print(f'* box_batch_size_per_image: {box_batch_size_per_image}')
print(f'* box_detections_per_image: {box_detections_per_image}')


#%%  customise the pre-trained FRCNN model for the VRD dataset

# adjust the number of output features in the classification output layer (for
# the object class predictions) to match the number of object classes in the 
# VRD dataset (plus 1 for a 'background' class)
model.roi_heads.box_predictor.cls_score.out_features = n_nesy4vrd_object_classes

# adjust the number of output features in the regression output layer (for the
# bounding box predictions) to match 4 times the number of object classes in
# the VRD dataset (plus 1 for a 'background' class)
model.roi_heads.box_predictor.bbox_pred.out_features = n_nesy4vrd_object_classes * 4

# replace the classification output layer's weights and biases with new
# random ones whose dimensions match the layer's revised output size
in_feat = model.roi_heads.box_predictor.cls_score.in_features
out_feat = model.roi_heads.box_predictor.cls_score.out_features
weights = torch.rand(out_feat, in_feat)
model.roi_heads.box_predictor.cls_score.weight = \
                torch.nn.Parameter(weights, requires_grad=True)
biases = torch.rand(out_feat)
model.roi_heads.box_predictor.cls_score.bias = \
                torch.nn.Parameter(biases, requires_grad=True)

# replace the bbox regression output layer's weights and biases with new
# random ones whose dimensions match the layer's revised output size
in_feat = model.roi_heads.box_predictor.bbox_pred.in_features
out_feat = model.roi_heads.box_predictor.bbox_pred.out_features
weights = torch.rand(out_feat, in_feat)
model.roi_heads.box_predictor.bbox_pred.weight = \
                torch.nn.Parameter(weights, requires_grad=True)
biases = torch.rand(out_feat)
model.roi_heads.box_predictor.bbox_pred.bias = \
                torch.nn.Parameter(biases, requires_grad=True)

print('Model architecture customised for NeSy4VRD dataset')

if redirect_stdout:
    sys.stdout.flush()


#%% define collate function for use by Dataloader

# Here we define a custom 'collate()' function to pass to our Dataloader so 
# that it won't complain about the input images being of different size and 
# throw a runtime Exception.  Without defining this custom 'collate()' 
# function, our only option for avoiding the runtime Exception would be to 
# train with a mini-batch size of just 1.
#
# sources:
# https://discuss.pytorch.org/t/
#         torchvision-and-dataloader-different-images-shapes/41026/3
# https://discuss.pytorch.org/t/
#         how-to-create-a-dataloader-with-variable-size-input/8278/2

def frcnn_collate(batch):
    '''
    Arguments:
       batch : a List of (idx, img, targets) 3-tuples created by calling 
               VRDDataset[idx]
    '''
    idxs = [item[0] for item in batch]
    imgs = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    return idxs, imgs, targets


#%% prepare Dataset and Dataloader objects

# instantiate our NeSy4VRD_Dataset 
dataset = NeSy4VRD_Dataset(vrd_img_dir=vrd_img_dir, 
                           nesy4vrd_anno_file=vrd_anno_file,
                           img_names=vrd_img_names_training)

# set batch size
# Camber server cluster GPUs (eg Nvidia Titan V) can cope with a 
# batch size of 4
# Hyperion HPC GPUs (Nvidia A100) can cope with a batch size of 8, and
# perhaps more

batch_size = 8
print(f'Batch size: {batch_size}')

# prepare a Dataloader
args = {'batch_size': batch_size, 'shuffle': True, 'collate_fn': frcnn_collate}
dataloader = torch.utils.data.DataLoader(dataset, **args)

print(f'Number of minibatches per epoch: {len(dataloader)}')


#%%

# put model in training mode
model.train()

# push model to correct device
if device != torch.device('cpu'):
    model = model.to(device)


#%%

# establish an optimiser
# (note: we do this AFTER having pushed our model to the GPU, which is a 
#  recommended convention; it's required especially when the optimiser 
#  maintains internal state, like the Adagrad optimiser)
opt_args = {'lr': 1e-5, 'weight_decay': 1e-3}
optimiser = torch.optim.Adam(model.parameters(), **opt_args)


#%%

def save_model_checkpoint(epoch, model, optimiser, avg_loss_per_mb):
    checkpoint = {'epoch': epoch, 
                  'model_state_dict': model.state_dict(),
                  'optimiser_state_dict': optimiser.state_dict(),
                  'avg_loss_per_mb': avg_loss_per_mb}
    
    filename = model_checkpoint_filename_base + str(epoch).zfill(3) + ".pth"
    path = os.path.join(model_checkpoint_dir, filename)
    torch.save(checkpoint, path)
    print(f"Model checkpoint file saved: {filename}")
    
    return None


#%%

def load_model_checkpoint(filepath, model, optimiser):
    # load checkpoint file; note that we don't set parameter map_location=device, 
    # which indicates the location where ALL tensors should be loaded;
    # this is because we want our model on the GPU but our optimiser on the CPU,
    # which is where they are both saved from when torch.save() is executed
    checkpoint = torch.load(filepath)
    
    # initialise the model and optimiser state (in-place) to the saved state 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    
    # get other saved variables
    epoch = checkpoint['epoch']
    avg_loss_per_mb = checkpoint['avg_loss_per_mb']
    
    print(f"Model checkpoint file loaded: {filepath}")
    
    return epoch, avg_loss_per_mb


#%% configure training loop parameters

# set training mode: 'start' or 'continue'
# - 'start' : no checkpoint file to load; begin training from epoch 0
# - 'continue' : load checkpoint file and resume training from last epoch
training_mode = 'continue'
continue_from_epoch = 210

# set number of training epochs
n_epochs_train = 100

# set frequency for saving checkpoint files (in number of epochs)
n_epochs_checkpoint = 5

# initialise training epoch range
if training_mode == 'start':
    last_epoch = 0
elif training_mode == 'continue':
    filename = model_checkpoint_filename_base + str(continue_from_epoch) + ".pth"
    filepath = os.path.join(model_checkpoint_dir, filename)
    args = {'filepath': filepath, 'model': model, 'optimiser': optimiser}
    last_epoch, last_avg_loss_per_mb_over_epoch = load_model_checkpoint(**args)
    print(f"last epoch: {last_epoch}; last avg loss per mb over epoch: {last_avg_loss_per_mb_over_epoch:.4f}")
else:
    raise ValueError('training mode not recognised')

# initialise range of epoch numbers
first_epoch = last_epoch + 1
final_epoch = first_epoch + n_epochs_train

# minibatch group size
#mb_group_size = 10

# don't start saving model checkpoint files until we've reached this epoch
start_saving_checkpoints_epoch = 1

print(f'Training mode: {training_mode}')
print(f'Continue from epoch: {continue_from_epoch}')
print(f'Number of epochs to train: {n_epochs_train}')
print(f'First epoch number in range: {first_epoch}')
print(f'Final epoch number in range: {final_epoch - 1}')

print()
print('--- Training run configuration end ---')
print()

if redirect_stdout:
    sys.stdout.flush()


#%% attempt at using multiple GPUs

#print(f'We are using {torch.cuda.device_count()} GPU(s)')

# NOTE: use of multiple GPUs is an unsolved problem; this code block
#       here is not sufficient by itself; we get run-time errors later
#
# if using multiple GPUs, prepare the model for parallelisation
#if torch.cuda.device_count() > 1:
#    print('About to prepare model for GPU parallelisation with DataParallel')
#    model = torch.nn.DataParallel(model)
#    print('Model prepared for GPU parallelisation with DataParallel')


#%% Training loop

start_time = time.time()

for epoch in range(first_epoch, final_epoch):
    
    #print(f'\nepoch {epoch} starting ...')
    
    epoch_loss = 0
    #mb_group_loss = 0
    
    start_time_epoch = time.time()
    
    for bidx, batch in enumerate(dataloader):
        
        #if bidx+1 % 20 == 0:
            #print(f'processing minibatch: {bidx+1}')
        #print(f'processing minibatch: {bidx+1}')
        
        # split the batch into its 3 components
        idxs, images, targets = batch
        
        # push the training data to the correct device
        if device != torch.device('cpu'):
            images_gpu = []
            targets_gpu = []
            for i in range(len(images)):
                image_gpu = images[i].to(device)
                images_gpu.append(image_gpu)
                target_gpu = {}
                target_gpu['labels'] = targets[i]['labels'].to(device)
                target_gpu['boxes'] = targets[i]['boxes'].to(device)
                targets_gpu.append(target_gpu)
            images = images_gpu
            targets = targets_gpu
        
        # forward pass through model
        loss_components = model(images, targets)
        
        # sum the 4 loss components to get total mini-batch loss
        mb_loss = 0
        for k in loss_components.keys():
            mb_loss += loss_components[k]
            
        # backpropagate and update model parameters
        optimiser.zero_grad()
        mb_loss.backward()
        optimiser.step()
        
        # accumulate loss
        #mb_group_loss += mb_loss.item()
        epoch_loss += mb_loss.item()
        
        # print something periodically so we can monitor progress
        #if (bidx+1) % mb_group_size == 0:
        #    avg_loss_per_mb_over_group = mb_group_loss / mb_group_size
        #    #print(f"batch {bidx+1:4d}; avg loss per mb: {avg_loss_per_mb_over_group:.4f}")
        #    mb_group_loss = 0

        # print mini-batch idx and loss so we can monitor progress
        #print(f"mb: {bidx:3d}; mb_loss: {mb_loss}")
        #if (bidx) >= 2:
        #    break
        
        if redirect_stdout:
            sys.stdout.flush()


    # compute average loss per minibatch over the current epoch
    avg_loss_per_mb = epoch_loss / len(dataloader)
    
    # capture epoch training time
    end_time_epoch = time.time()
    epoch_time = (end_time_epoch - start_time_epoch) / 60
    
    # write to training log file
    print(f"epoch {epoch:3d}; train loss: {avg_loss_per_mb:.4f}; time: {epoch_time:.2f} min")

    # periodically save of model checkpoint file
    if epoch % n_epochs_checkpoint == 0:
        if epoch >= start_saving_checkpoints_epoch:
            # models for early epochs will NOT be near convergence and so
            # won't be candidates for selection; so there's no need to save
            # checkpoint files until we reach a certain epoch number
            save_model_checkpoint(epoch, model, optimiser, avg_loss_per_mb)

    if redirect_stdout:
        sys.stdout.flush()


# print total training time (in minutes)
end_time = time.time()
train_time = (end_time - start_time) / 60
print(f"\nTotal training time: {train_time:.2f} minutes\n")

# save a checkpoint file if training epochs have been processed since
# the last time a checkpoint file was saved
if epoch % n_epochs_checkpoint != 0:
    save_model_checkpoint(epoch, model, optimiser, avg_loss_per_mb)


#%%

if redirect_stdout:
    # flush stdout buffer
    sys.stdout.flush()
    # close redirected output file
    sys.stdout.close()
    # restore sys.stdout to original file handler
    sys.stdout = stdout_file_saved

print('Processing completed')
