#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script performs inference using trained Predicate Prediction neural 
network (PPNN) models.

The inference will normally be run on the NeSy4VRD test set image data, 
but the script can process the NeSy4VRD training set image data as well, 
if desired.  The script does NOT currently support the sub(training) and
validation sets of image names carved out of the original (full) training set.

It can perform inference on a set of trained PPNN models if they share
a common filename pattern.

The script stores the inference outputs of the trained PPNN models in 
JSON files.  

This script does not evaluate performance; a separate Python script exists
for that purpose.
'''

#%%

import torch

import os
import json
import glob
import time
import sys
from datetime import date

import vrd_ppnn_models as vpm
from vrd_ppnn_dataset import PPNNDataset
import vrd_utils10 as vrdu10
import vrd_utils14 as vrdu14


#%% gather arguments supplied to this script

# get or set the experiment space training region cell id
if len(sys.argv) > 1:
    trc_id = sys.argv[1]
else:
    trc_id = 'trc0460'
if not trc_id.startswith('trc'):
    raise ValueError(f'training region cell id {trc_id} not recognised')

# get or set the platform on which the training script is running
if len(sys.argv) > 2:
    platform = sys.argv[2]
else:
    platform = 'macstudio'
if not platform in vrdu14.supported_platforms:
    raise ValueError(f'platform {platform} not recognised')

# # get or set the work directory (the folder in which to store output files)
if len(sys.argv) > 3:
    workdir = sys.argv[3]
else:
    workdir = 'ppnn'


#%% lookup config of the experiment space training region cell

# specify the experiment family
experiment_family = 'nnkgs0'      # NN+KG_S0

#
# get the training region cell configuration (dimension levels)
#

cfg = vrdu14.get_training_region_cell_config(experiment_family, trc_id)

# set the training region dimension levels
tr_d_model1 = cfg['D_model1_level']
tr_d_model2 = cfg['D_model2_level']      
tr_d_dataCat = cfg['D_dataCat_level']
tr_d_dataFeat = cfg['D_dataFeat_level']
tr_d_kgS0 = cfg['D_kgS0_level']                  
tr_d_kgS1 = cfg['D_kgS1_level']
tr_d_kgS2 = cfg['D_kgS2_level']
tr_d_kgS3 = cfg['D_kgS3_level']
tr_d_nopredTarget = cfg['D_nopredTarget_level']  
tr_d_onto = cfg['D_onto_level']


#%% build the name of the work directory (where model checkpoint files are)

if platform == 'hyperion':
    model_checkpoint_dir = os.path.join('~', 'sharedscratch', 'research', workdir)
else:
    model_checkpoint_dir = os.path.join('~', 'research', workdir)

scriptName = os.path.basename(__file__)
print(f'script name: {scriptName}')

model_checkpoint_dir = os.path.expanduser(model_checkpoint_dir)
print(f'work dir   : {model_checkpoint_dir}')


#%% choose whether or not to redirect stdout to a file

# redirecting stdout allows one to retain an inference log text file
# for documentation

redirect_stdout = True


#%% set the name of the file to which stdout is to be redirected

if redirect_stdout:
    inference_log_filename = 'vrd_ppnn_' + trc_id + '_inference_log.txt'
else:
    inference_log_filename = ''


#%% redirect stdout

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(model_checkpoint_dir, inference_log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%% declare (record) experiment meta data

print()
print(f'# experiment family: {experiment_family}')
print(f'# training region cell id: {trc_id}')
print()
print(f'Date: {date.today()}')


#%% establish the device (CPU, GPU, MPS) on which we're doing inference

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# force use of cpu    
#device = torch.device('cpu')


#%% record key info in the inference log file

# the platform on which we're running
print(f'Platform: {platform}')

# the name of this Python script
#scriptName = sys.argv[0]
scriptName = os.path.basename(__file__)
print(f'Script: {scriptName}')

# the device (CPU, GPU, MPS) upon which we're running
print(f'Device: {device}')

# the name of the conda environment in which we're running
print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")

# the version of PyTorch we're using
print(f'PyTorch version: {torch.__version__}')


#%% get and count the VRD object class names

# set path to VRD object class names
vrd_obj_file = os.path.join('..', 'data', 'annotations',
                            'nesy4vrd_objects.json')

# get the VRD object class names
with open(vrd_obj_file, 'r') as file:
    vrd_object_class_names = json.load(file)

# count the object class names
nr_object_classes = len(vrd_object_class_names)

print(f"Nr of object classes: {nr_object_classes}")


#%% get and count the VRD predicate names

# set path to VRD predicate names
vrd_pred_filepath = os.path.join('..', 'data', 'annotations',
                                 'nesy4vrd_predicates.json')

# get the VRD predicates names
with open(vrd_pred_filepath, 'r') as file:
    vrd_predicate_names = json.load(file)

# count the predicate names
nr_predicates = len(vrd_predicate_names)

print(f"Nr of predicates: {nr_predicates}")


#%% get the test set image names

# Note: we use the initial (sparse & arbitrary) test set VR annotations to
# get the list of test set images names.  We could get them from 
# KG-augmented test set VR annotations as well, since the set of test set
# image names is fixed and constant across different versions of the
# VR annotations.  But to avoid confusion we use only the initial test set
# VR annotations for this purpose.

anno_filename = 'nesy4vrd_annotations_test.json'
anno_filepath = os.path.join('..', 'data', 'annotations', anno_filename)

print(f'Annotations file used to get image names: {anno_filepath}')

# load the VR annotations
with open(anno_filepath, 'r') as file:
    vrd_img_anno = json.load(file)

# extract the image names from the annotations
vrd_img_names = list(vrd_img_anno.keys())

print(f'Number of images available for processing: {len(vrd_img_names)}')


#%% configure the correct PPNN input data file for inference

# The level of dimension D_dataCat of the experiment space training region
# identifies the type/version of PPNN input data file to be used for the 
# current experiment.

# The level of dimension D_dataCat determines the PPNN input data file
# used for both training and inference. This way, we avoid unintended
# mismatches.
#
# If D_dataCat_1 is used in training, we want D_dataCat_1 for inference.
# This way we stay solidly within the 'relationship detection' regime of
# experiments. A mismatch here would be a nonsense.
#
# If D_dataCat_2 is used in training, we want D_dataCat_2 for inference.
# This way we stay solidly within the 'predicate detection' regime of
# experiments.  A mismatch here would be a nonsense.
#
# If D_dataCat_3 is used in training, we want D_dataCat_3 for inference.

if tr_d_dataCat == 1:
    # use ppnn input data derived from the objects detected in the 
    # test set images by our FRCNN object detector; this level of
    # dimension D_dataCat implies that the current experiment belongs to
    # the 'relationship detection' regime of experiments
    data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_1_testset.json'
    #data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_2_testset.json'
    #data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_3_testset.json'
    #data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_4_testset.json'
elif tr_d_dataCat == 2:
    # use ppnn input data derived from the initial (sparse, arbitrary)
    # VR annotations of the test set images; this level of dimension 
    # D_dataCat implies that the current experiment belongs to the 
    # 'predicate detection' regime of experiments
    data_filename = 'ppnn_input_pp_per_nesy4vrd_annotations_test.json'
#elif tr_d_dataCat == 3:
    # use ppnn input data derived from the KG-augmented VR annotations of 
    # the test set images; this level of dimension D_dataCat implies
    # that the current experiment belongs to the 'predicate detection'
    # regime of experiments    
#    data_filename = 'to be determined'
else:
    raise ValueError(f'level {tr_d_dataCat} of dimension D_dataCat not recognised')

data_path = os.path.join('..', 'data', 'ppnn_input_data', data_filename)

print(f'PPNN input data file: {data_path}')


#%% instantiate a PPNNDataset object

# The level of dimension D_dataFeat of the experiment space training region
# identifies the particular PPNN input data feature set that is to be used
# for the current experiment.

if not tr_d_dataFeat in [1,2,3,4]:
    raise ValueError(f'PPNN input data feature set {tr_d_dataFeat} not recognised')

# the feature set of PPNN input data to be used for inference
featureset = tr_d_dataFeat

print(f'PPNN input data feature set: {featureset}')

# instantiate a PPNNDataset to retrieve mini-batches of PPNN input data
dataItem = PPNNDataset(data_filepath=data_path,
                       featureset=featureset,
                       anno_filepath=anno_filepath,
                       nr_object_classes=nr_object_classes,
                       nr_predicates=nr_predicates,
                       targets_required=False)

print(f'PPNNDataset object has data for this many images: {len(dataItem)}')


#%% get the size of our PPNN input data feature vectors

# We need to know the size of the input data feature vectors so we can
# instantiate our PPNN models correctly.

# specify an arbitrary image in terms of its index position
img_idx = 1

# access the ppnn input data for an arbitrary image
results = dataItem[img_idx]
inputdata = results['inputdata']

in_features = inputdata.shape[1]

print(f'Size of input data vectors (per feature set): {in_features}')


#%% function to instantiate a PPNN model

def instantiate_model(tr_d_model1, tr_d_model2, in_features, nr_predicates):
    '''
    Instantiate a designated PPNN model. The instantiated model will be
    initialised with the state dictionary of a particular PPNN model 
    checkpoint file created previously during PPNN model training.
    
    Dimensions D_model1 and D_model2 together specify the precise PPNN
    model architecture to be used.
    '''

    # apply the policy indicated by the level of dimension D_model2
    # (nb: this is the primary purpose of dimension D_model2 --- to indicate
    # whether or not our PPNN model is to have a special neuron in its output
    # layer that represents a notional 'no predicate' predicate)
    if tr_d_model2 == 1:
        out_size = nr_predicates
    elif tr_d_model2 == 2:
        out_size = nr_predicates + 1
    else:
        raise ValueError(f'level {tr_d_model2} of dimension D_model2 not recognised')
    
    # apply the policy indicated by the level of dimension D_model1
    if tr_d_model1 == 1:
        model = vpm.PPNN_1(in_features=in_features, out_size=out_size)
    elif tr_d_model1 == 2:
        raise ValueError(f'level {tr_d_model1} of dimension D_model1 not yet implemented')
        #model = vpm.PPNN_2(in_features=in_features, out_size=out_size)
    elif tr_d_model1 == 3:
        raise ValueError(f'level {tr_d_model1} of dimension D_model1 not yet implemented')
        #model = vpm.PPNN_3(in_features=in_features, out_size=out_size)
    else:
        raise ValueError(f'level {tr_d_model1} of dimension D_model1 not recognised')
    
    return model


#%% function to load trained model checkpoint file

def load_model_checkpoint(filepath, model):
    
    # load the model checkpoint file
    # (assume, for now, that we are on a platform that is not using a GPU;
    #  so load all tensors to the CPU; if we are using a GPU, the model will
    #  be pushed to the GPU elsewhere; this approach prevents the throwing of
    #  an Exception on platforms where a GPU is not available)
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    # load the checkpoint (ie the parameters representing our trained PPNN)
    # into our model instance
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


#%% function to save inference results to a file

def save_results_to_file(filepath, results):
    
    with open(filepath, 'w') as fp:
        json.dump(results, fp)

    return None


#%% set order for sorting of model checkpoint filenames

# choose whether to have the model checkpoint files processed in ascending
# order by epoch number or in descending order by epoch number:
# - in development mode, 'ascending' is convenient
# - in production mode, 'descending' is what we want 
sort_models_ascending = False


#%% get the target set of model checkpoint files to be processed

# We specify the target set of model checkpoint files by specifying the
# common pattern of the file names. We also specify the directory in which
# these model checkpoint files have been stored.

# set the filename pattern to identify the set of model checkpoint files to process
model_checkpoint_pattern = 'vrd_ppnn_' + trc_id + '_ckpt_*_model.pth'

path = os.path.join(model_checkpoint_dir, model_checkpoint_pattern)

# gather the names of all of the model checkpoint files to be processed
model_checkpoint_paths = glob.glob(path)

if sort_models_ascending:
    model_checkpoint_paths = sorted(model_checkpoint_paths)
else:
    model_checkpoint_paths = sorted(model_checkpoint_paths, reverse=True)

print(f'Number of model checkpoint files to be processed: {len(model_checkpoint_paths)}')


#%% set number of model checkpoint files to process (evaluate=N)

# Our model selection policy can be summarised as 'patience=30, evaluate=35',
# where 'patience' relates to the 'early stopping' of training, and where
# 'evaluate' refers to the number of models whose performance we evaluate
# once training has stopped.

# The variable initialised here corresponds to the 'evaluate=N' portion of
# our model selection policy. 
#
# A value of N=0 means process 'all' available model checkpoint files. This
# setting is convenient during development and testing.
#
# A value of N>0 means process only up to the first N model checkpoint files.
# This setting normal for production.  It presumes that the model
# checkpoint files have been sorted in 'descending' order by epoch number,
# which means we'll process the N 'latest' models --- the ones with the 
# largest epoch numbers.

#n_models_to_process = 0
n_models_to_process = 35


#%% set number of images-worth of data to process per model checkpoint

# in development mode, it can sometimes be convenient to process only the 
# first N images-worth of input data (eg to facilitate quick testing), in
# which case we set a value 'greater than zero' (> 0)
# in production mode, we want to process all of the input data, in which
# case we set a value 'equal to zero' (= 0)

n_images_to_process = 0

if n_images_to_process == 0: 
    print('Number of images to process per model: all')
else:
    print(f'Number of images to process per model: {n_images_to_process}')


#%% flush all the print statements above to the log file

if redirect_stdout:
    sys.stdout.flush()


#%% main processing loop

start_time_total = time.time()

print(f"Performing inference for models: {model_checkpoint_pattern}")

# initialise parameters for call to vrdu10.perform_inference()
inference_mode = 'test'
calculate_loss = False
loss_reduction = None
nopred_target_policy = None

# initialise a counter
cnt = 0

# iterate over the models with which to perform inference
for item in model_checkpoint_paths:

    # capture start time for current model
    start_time_model = time.time()

    # prepare model name
    prefix = model_checkpoint_dir + os.sep
    ckpt_filename = item.removeprefix(prefix)

    print(f"\nProcessing model: {ckpt_filename}")

    # prepare model
    model = instantiate_model(tr_d_model1, tr_d_model2, in_features, nr_predicates)
    model = load_model_checkpoint(filepath=item, model=model)

    # if a non-CPU device is available, push the model onto it
    if device != torch.device('cpu'):
        model = model.to(device)
    
    # put model in evaluation (inference) mode
    # (to disable things like Dropout and Batch Normalisation)
    model.eval()
    
    # disable gradient computation
    # (to eliminate redundant processing and reduce memory consumption)
    torch.set_grad_enabled(False)

    # perform inference with the current model
    inference_outputs = vrdu10.perform_inference(model, dataItem, 
                                                 vrd_img_names,
                                                 vrd_img_names,
                                                 device, n_images_to_process,
                                                 inference_mode,
                                                 calculate_loss,
                                                 loss_reduction,
                                                 tr_d_model2, 
                                                 nopred_target_policy)
    
    # save the PPNN inference output to disk
    output_path_filename = item.removesuffix('model.pth') + 'output.json'
    #output_path_filename = os.path.join(model_checkpoint_dir, output_filename)
    save_results_to_file(output_path_filename, inference_outputs)
    prefix = model_checkpoint_dir + os.sep
    output_filename = output_path_filename.removeprefix(prefix)
    print(f'Inference output: {output_filename}')
    
    # measure elapsed time
    end_time_model = time.time()
    model_time = (end_time_model - start_time_model) / 60
    print(f"Processing time : {model_time:.2f} minutes")
    
    if redirect_stdout:
        sys.stdout.flush()
    
    # check if it's time to stop processing models
    cnt += 1
    if n_models_to_process > 0:
        if cnt >= n_models_to_process:
            break 


print()
print('\nProcessing complete')  # goes to redirected stdout, if used

# print total training time (in minutes)
end_time_total = time.time()
time_total = (end_time_total - start_time_total) / 60
print(f"Total time: {time_total:.2f} minutes\n")

if redirect_stdout:
    # flush current stdout buffer
    sys.stdout.flush()
    # close current (redirected) stdout file
    sys.stdout.close()
    # restore sys.stdout to its original file handler
    sys.stdout = stdout_file_saved

print('processing complete')  # goes to default stdout always



# %%
