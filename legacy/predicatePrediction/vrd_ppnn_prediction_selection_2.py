#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script performs prediction selection on the outputs captured by doing
inference using trained PPNN models.

It can perform prediction selection on a set of PPNN output files if they 
share a common filename pattern.

The PPNN outputs that are processed by this script are logits representing
the likelihoods that the different VRD predicates describe visual 
relationships between ordered pair of objects. For each ordered pair of
objects there is a vector of predicate logits. From that set of logits,
this script decides which ones shall be deemed to represent 'predictions'.
Once the set of 'predicted predicates' is established for a given ordered
pair of objects, the script then fashions a set of 'predicted visual
relationships' in a format compatible with performance evaluation. These
'predicted visual relationships' are saved to a file for subsequent
performance evaluation processing.

This script does not evaluate performance; a separate Python script exists
for that purpose.
'''

#%%

import os
import json
import glob
import time
import sys
from datetime import date

import vrd_utils11 as vrdu11
import vrd_utils14 as vrdu14


#%% gather arguments supplied to this script

# get or set the experiment space training region cell id
if len(sys.argv) > 1:
    trc_id = sys.argv[1]
else:
    trc_id = 'trc0460'
if not trc_id.startswith('trc'):
    raise ValueError(f'training region cell id {trc_id} not recognised')

# get or set the experiment space prediction region cell id
if len(sys.argv) > 2:
    prc_id = sys.argv[2]
else:
    prc_id = 'prc018'
if not prc_id.startswith('prc'):
    raise ValueError(f'prediction region cell id {prc_id} not recognised')

# get or set the platform on which the training script is running
if len(sys.argv) > 3:
    platform = sys.argv[3]
else:
    platform = 'macstudio'

if not platform in vrdu14.supported_platforms:
    raise ValueError(f'platform {platform} not recognised')

# get or set the work directory (the folder in which to store output files)
if len(sys.argv) > 4:
    workdir = sys.argv[4]
else:
    workdir = 'ppnn'


#%% lookup config of the experiment space prediction region cell

# specify the experiment family
experiment_family = 'nnkgs0'      # NN+KG_S0

#
# get the training region cell configuration (dimension levels)
#

cfg = vrdu14.get_training_region_cell_config(experiment_family, trc_id)

# get the level of the dimension that tells us whether or not the
# PPNN model has an output layer neuron representing a notional 
# 'no predicate' prediate
tr_d_model2 = cfg['D_model2_level']      

#
# get the prediction region cell configuration (dimension levels)
#

cfg = vrdu14.get_prediction_region_cell_config(experiment_family, prc_id)

# assign the dimension levels to more familiar variable names

# prediction confidence threshold
pred_conf_thresh = cfg['D_predConf_level']
        
# maximum number of predicted predicates per ordered pair of objects
max_preds_per_obj_pair = cfg['D_predMax_level']

# use of a KG as a final filter for selecting predictions, or not
if cfg['D_predKG_level'] == 1: 
    kg_filtering = False                          
else:
    kg_filtering = True

# policy for handling 'no predicate' predictions    
nopred_prediction_policy = cfg['D_predNoPred_level']       

# package the prediction region configuration for passing to functions
prediction_region_config = {}
prediction_region_config['pred_region_id'] = prc_id
prediction_region_config['pred_conf_thresh'] = pred_conf_thresh
prediction_region_config['max_preds_per_obj_pair'] = max_preds_per_obj_pair
prediction_region_config['kg_filtering'] = kg_filtering
prediction_region_config['nopred_prediction_policy'] = nopred_prediction_policy


#%% build the name of the work directory (where the PPNN inference outputs are)

root_dir = '~'   # local hard drive
#root_dir = os.path.join(os.sep, 'Volumes', 'My Passport for Mac') # external hard drive

if platform == 'hyperion':
    ppnn_output_dir = os.path.join('~', 'sharedscratch', 'research', workdir)
else:
    ppnn_output_dir = os.path.join(root_dir, 'research', workdir)
    
    # nb: the following setting is for reprocessing archived cells, if required
    #ppnn_output_dir = os.path.join(root_dir, 'research', 'results-archive', 'nnkgs0')
    #ppnn_output_dir = os.path.join(ppnn_output_dir, 'setID-01', 'trial-02')
    #ppnn_output_dir = os.path.join(ppnn_output_dir, 'cell-trc2470-prc018-rrc01')
    #ppnn_output_dir = os.path.join(ppnn_output_dir, workdir)

scriptName = os.path.basename(__file__)
print(f'script name: {scriptName}')

ppnn_output_dir = os.path.expanduser(ppnn_output_dir)
print(f'work dir   : {ppnn_output_dir}')


#%% choose whether or not to redirect stdout to a file

# redirecting stdout allows one to retain an inference log text file
# for documentation

redirect_stdout = True


#%% set the name of the file for the redirected stdout

if redirect_stdout:
    log_filename = 'vrd_ppnn_' + trc_id + '_' + prc_id + '_prediction_log.txt'
else:
    log_filename = ''


#%% redirect stdout

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(ppnn_output_dir, log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%% declare (record) experiment meta data

print()
print(f'# experiment family: {experiment_family}')
print(f'# training region cell id: {trc_id}')
print(f'# prediction region cell id: {prc_id}')
print()
print(f'Date: {date.today()}')


#%% record key info in the inference log file

# the platform on which we're running
print(f'Platform: {platform}')

# the name of this Python script
#scriptName = sys.argv[0]
scriptName = os.path.basename(__file__)
print(f'Script: {scriptName}')

# the name of the conda environment in which we're running
print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")


#%% function to save VR predictions to a file

def save_predictions_to_file(filepath, results):
    
    #print(results)
    
    with open(filepath, 'w') as fp:
        json.dump(results, fp)

    return None


#%% function to load a PPNN output file

def load_ppnn_output(filepath):
    
    with open(filepath, 'r') as fp:
        ppnn_output = json.load(fp)

    return ppnn_output    


#%% get the set of PPNN output files to be processed

# set the filename pattern
ppnn_output_pattern = 'vrd_ppnn_' + trc_id + '_ckpt_*_output.json'
print(f"PPNN output filename pattern: {ppnn_output_pattern}")

path = os.path.join(ppnn_output_dir, ppnn_output_pattern)

# gather the names of all of the files to be processed
ppnn_output_paths = glob.glob(path)
ppnn_output_paths = sorted(ppnn_output_paths)
print(f'Number of PPNN output files to be processed: {len(ppnn_output_paths)}')


#%% set mode: production or testing

# set the number of images to process per ppnn output file: just a few (for 
# test purposes) or all of them (for a production run)
# = 0 is for production --- it means process all images
# > 0 is for testing --- it means process the first N images only
n_images_to_process = 0

if n_images_to_process == 0: 
    print('Number of images to process per PPNN output file: all')
else:
    print(f'Number of images to process per PPNN output file: {n_images_to_process}')


#%% main processing loop

#cnt = 0

start_time_total = time.time()

# iterate over the ppnn output files generated by various different
# PPNN models (checkpoints) 
for item in ppnn_output_paths:

    #cnt += 1
    #if cnt >  1:
    #    break

    # capture start time
    start_time_model = time.time()

    # get the filename of the current PPNN output file
    prefix = ppnn_output_dir + os.sep
    ppnn_output_filename = item.removeprefix(prefix)
    print(f"\nProcessing file: {ppnn_output_filename}")

    # load the current ppnn output file (predicate probabilities)
    ppnn_output_per_image = load_ppnn_output(item)

    # from the predicate probabilities calculated from the logits output by
    # a PPNN, select the predicates that we deem to have been 'predicted' 
    # and use these to fashion 'predicted visual relationships'; do this
    # for each image in a dataset
    predicted_vrs = vrdu11.select_predictions(ppnn_output_per_image,
                                              tr_d_model2, 
                                              prediction_region_config,
                                              n_images_to_process)
    
    # save the predicted visual relationships to a file
    filename = item.removesuffix('output.json')
    pred_path_filename = filename + prc_id + '_preds.json'
    save_predictions_to_file(pred_path_filename, predicted_vrs)
    prefix = ppnn_output_dir + os.sep
    pred_filename = pred_path_filename.removeprefix(prefix)
    print(f'Predicted VRs  : {pred_filename}')

    # measure elapsed time
    end_time_model = time.time()
    model_time = (end_time_model - start_time_model) / 60
    print(f"Processing time: {model_time:.2f} minutes")
    
    if redirect_stdout:
        sys.stdout.flush()

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


