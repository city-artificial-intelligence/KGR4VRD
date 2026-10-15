#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script consolidates multiple .json files containing measures of 
recall@N-based predictive performance metrics into a single .csv file. There
is one row per training epoch, each with its associated performance measures.

The script also merges the per epoch training loss, validation loss, and
validation performance scores already saved in a separate .csv file.

The .csv file can later be opened in a spreadsheet app for easy viewing of
a collection of performance results delivered by different experiment
space regions. This allows the distribution of performance results to be
visualised and reviewed, and allows the best performing models (for each
of the different metrics) to be visually identified.

This script is designed to be run interactively.

The .json files that are consolidated all share a common filename pattern. 
All .json files whose names correspond to the pattern are effectively 
'combined' into one consolidated .csv file.
'''

#%%

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from datetime import date

import vrd_utils14 as vrdu14


#%% gather arguments supplied to this script

# get or set the experiment space training region cell id
if len(sys.argv) > 1:
    trc_id = sys.argv[1]
else:
    trc_id = 'trc0460'  # 'trc0460'  
if not trc_id.startswith('trc'):
    raise ValueError(f'training region cell id {trc_id} not recognised')

# get or set the experiment space prediction region cell id
if len(sys.argv) > 2:
    prc_id = sys.argv[2]
else:
    prc_id = 'prc018'
if not prc_id.startswith('prc'):
    raise ValueError(f'prediction region cell id {prc_id} not recognised')

# get or set the experiment space results region cell id
if len(sys.argv) > 3:
    rrc_id = sys.argv[3]
else:
    rrc_id = 'rrc01'
if not rrc_id.startswith('rrc'):
    raise ValueError(f'results region cell id {rrc_id} not recognised')

# get or set the value of topN for our recall@N-based performance metrics
if len(sys.argv) > 4:
    topN = int(sys.argv[4])
else:
    topN = 25

# get or set the platform on which the training script is running
if len(sys.argv) > 5:
    platform = sys.argv[5]
else:
    platform = 'macstudio'

if not platform in vrdu14.supported_platforms:
    raise ValueError(f'platform {platform} not recognised')

# get or set the work directory (the folder in which to store output files)
if len(sys.argv) > 6:
    workdir = sys.argv[6]
else:
    workdir = 'ppnn'
    

#%% lookup config of the experiment space prediction region cell

# specify the experiment family
experiment_family = 'nnkgs0'      # NN+KG_S0

# no lookups required


#%% build the name of the work directory (where the performance results are)

root_dir = '~'   # local hard drive
#root_dir = os.path.join(os.sep, 'Volumes', 'My Passport for Mac') # external hard drive

if platform == 'hyperion':
    performance_results_dir = os.path.join('~', 'sharedscratch', 'research', workdir)
else:
    performance_results_dir = os.path.join(root_dir, 'research', workdir)
    
    # nb: the following setting is for reprocessing archived cells, if required
    #performance_results_dir = os.path.join(root_dir, 'research', 'results-archive', 'nnkgs0')
    #performance_results_dir = os.path.join(performance_results_dir, 'setID-01', 'trial-02')
    #performance_results_dir = os.path.join(performance_results_dir, 'cell-trc5500-prc018-rrc01')
    #performance_results_dir = os.path.join(performance_results_dir, workdir)


scriptName = os.path.basename(__file__)
print(f'script name: {scriptName}')

performance_results_dir = os.path.expanduser(performance_results_dir)
print(f'work dir   : {performance_results_dir}')


#%% declare (record) experiment meta data

print()
print(f'# experiment family: {experiment_family}')
print(f'# training region cell id: {trc_id}')
print(f'# prediction region cell id: {prc_id}')
print(f'# results region cell id: {rrc_id}')
print(f'# topN for recall@N: {topN}')
print()
print(f'Date: {date.today()}')


#%% display key info 

# the platform on which we're running
print(f'Platform: {platform}')

# the name of this Python script
#scriptName = sys.argv[0]
scriptName = os.path.basename(__file__)
print(f'Script: {scriptName}')

# the name of the conda environment in which we're running
print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")


#%%

def load_performance_results(path):

    with open(path, 'r') as fp:
        results_data = json.load(fp)

    return results_data


#%% get the set of test set performance results files we wish to process

# set the filename pattern
perf_results_pattern = 'vrd_ppnn_' + trc_id + '_ckpt_*_' + prc_id
perf_results_pattern = perf_results_pattern + '_' + rrc_id  + '_results_topN_'
perf_results_pattern = perf_results_pattern + str(topN).zfill(3) + '.json'
print(f"Performance results filename pattern: {perf_results_pattern}")

path = os.path.join(performance_results_dir, perf_results_pattern)

# gather the names of all of the performance results files to be processed
perf_results_paths = glob.glob(path)
perf_results_paths = sorted(perf_results_paths)
print(f'Number of performance results files to process: {len(perf_results_paths)}')


#%% get the training log summary data to be merged with the test results

# set the training log file name and path
training_log_filename = 'vrd_ppnn_' + trc_id + '_ckpt_training_loss_summary.csv'
training_log_filepath = os.path.join(performance_results_dir, 
                                     training_log_filename)

# specify the datatypes to be applied to certain columns
#col_data_types = {'tl_dir': str, 'vl_dir': str, 'vp_dir': str}

# load the training log summary stats into a dataframe
df_stats = pd.read_csv(training_log_filepath)
#df_stats = pd.read_csv(training_log_filepath, dtype=col_data_types)


#%% remove any NaN values from our initial dataframe

# NOTE: this section became obsolete; we can remove it

#mask = df_stats['tl_dir'] != 'up'
#df_stats.loc[mask, 'tl_dir'] = ''     # blank out NaN values

#mask = df_stats['vl_dir'] != 'up'
#df_stats.loc[mask, 'vl_dir'] = ''       # blank out NaN values

#mask = df_stats['vp_dir'] != 'up'
#df_stats.loc[mask, 'vp_dir'] = ''       # blank out NaN values


#%% our consolidation & merge strategy

# The dataframe just initialised with the data from the training loss
# summary .csv (epoch, training loss, validation loss, validation performance)
# is our starting point. We augment this dataframe by preparing and 
# introducing new, additional columns of data containing the recall@N
# scores produced by evaluating test set performance in the previous stage
# of the PPNN pipeline.

# We build lists of the recall@N scores and then introduce these lists to
# the dataframe as new columns. Then we save the augmented dataframe as 
# our output .csv file.

# We use the epoch numbers associated with the two sets of input data to
# ensure that the statistics in our dataframe are properly aligned by epoch.


#%% initialise the lists that will become new dataframe columns

n_elems = df_stats.shape[0]

mean_gt_vrs = [0] * n_elems
mean_pred_preds = [0] * n_elems
mean_pred_vrs_avail = [0] * n_elems
mean_pred_vrs = [0] * n_elems
mean_hits = [0] * n_elems
recallN_m1 = [0] * n_elems
rm1_dir = [' '] * n_elems
recallN_m2 = [0] * n_elems
rm2_dir = [' '] * n_elems
recallN_m3 = [0] * n_elems
rm3_dir = [' '] * n_elems

mean_avg_precision = [0] * n_elems
mean_ap_dir = [' '] * n_elems


#%% main processing loop

# initialise scores to be tracked
previous_global_recall = float('-inf')
previous_mean_per_image_recall = float('-inf')
previous_mean_avg_recall_k_topn = float('-inf')

previous_mean_avg_precision = float('-inf')

# iterate over the test set performance results files
for item in perf_results_paths:

    results = load_performance_results(item)

    # extract the model checkpoint epoch number from the filename in 'item'
    prefix = performance_results_dir + os.sep
    prefix = prefix + 'vrd_ppnn_' + trc_id + '_ckpt_'
    epoch_plus_suffix = item.removeprefix(prefix)
    suffix = '_' + prc_id + '_' + rrc_id 
    suffix = suffix + '_results_topN_' + str(topN).zfill(3) + '.json'
    ckpt_epoch = epoch_plus_suffix.removesuffix(suffix)

    # Using the epoch of the performance results, find the index of
    # of the corresponding epoch in our dataframe of training log
    # summary statistics. This index gives the location in our lists
    # for storing the corresponding performance results.
    mask = df_stats['epoch'] == int(ckpt_epoch)
    if sum(mask) != 1:
        raise ValueError(f'matching problem for epoch: {ckpt_epoch}')
    epoch_idx = mask[mask == True].index[0]
      
    # check the directions of change of our test set performance scores
    # (nb: this code relies on the performance results files having been
    #  sorted in ascending order by epoch number)
    if results['global_recallN'] < previous_global_recall:
        gr_dir = 'down'
    else:
        gr_dir = ' '
    
    if results['mean_per_image_recallN'] < previous_mean_per_image_recall:
        mpir_dir = 'down'
    else:
        mpir_dir = ' '
    
    if results['mean_avg_recallK_topN'] < previous_mean_avg_recall_k_topn:
        mark_dir = 'down'
    else:
        mark_dir = ' '    

    if results['mean_avg_precision'] <   previous_mean_avg_precision:
        map_dir = 'down'
    else:
        map_dir = ' '

    # place all of the performance results scores for the current epoch
    # in the correct location in our results score lists 
    mean_gt_vrs[epoch_idx] = results['mean_gt_vrs_per_img']
    mean_pred_preds[epoch_idx] = results['mean_pred_preds_per_img']
    mean_pred_vrs_avail[epoch_idx] = results['mean_pred_vrs_avail_per_img']
    mean_pred_vrs[epoch_idx] = results['mean_pred_vrs_per_img']
    mean_hits[epoch_idx] = results['mean_hits_per_img']
    recallN_m1[epoch_idx] = results['global_recallN']
    rm1_dir[epoch_idx] = gr_dir
    recallN_m2[epoch_idx] = results['mean_per_image_recallN']
    rm2_dir[epoch_idx] = mpir_dir
    recallN_m3[epoch_idx] = results['mean_avg_recallK_topN']
    rm3_dir[epoch_idx] = mark_dir
    
    mean_avg_precision[epoch_idx] = results['mean_avg_precision']
    mean_ap_dir[epoch_idx] = map_dir
       
    # save current scores for next iteration
    previous_global_recall = results['global_recallN']
    previous_mean_per_image_recall = results['mean_per_image_recallN']
    previous_mean_avg_recall_k_topn = results['mean_avg_recallK_topN']
    
    previous_mean_avg_precision = results['mean_avg_precision']


#%% mark the relevant MIN and MAX scores for easy identification

# find and mark the minimum validation loss score
trgt_idx = np.argmin(df_stats['val_loss'])
df_stats.loc[trgt_idx,'vl_dir'] = 'MIN'

# find and mark the maximum validation performance score
trgt_idx = np.argmax(df_stats['val_perf'])
df_stats.loc[trgt_idx,'vp_dir'] = 'MAX'

# find and mark the maximum scores for the 3 test set performance metrics
trgt_idx = np.argmax(recallN_m1)
rm1_dir[trgt_idx] = 'MAX'
trgt_idx = np.argmax(recallN_m2)
rm2_dir[trgt_idx] = 'MAX'
trgt_idx = np.argmax(recallN_m3)
rm3_dir[trgt_idx] = 'MAX'

# find and mark the maximum score for the mAP metric
trgt_idx = np.argmax(mean_avg_precision)
mean_ap_dir[trgt_idx] = 'MAX'



#%% augment the dataframe with new columns containing the performance scores

df_stats['mean_gt_vrs'] = mean_gt_vrs
df_stats['mean_pred_preds'] = mean_pred_preds
df_stats['mean_pred_vrs_avail'] = mean_pred_vrs_avail
df_stats['mean_pred_vrs'] = mean_pred_vrs
df_stats['mean_hits'] = mean_hits
df_stats['recall@N_m1'] = recallN_m1
df_stats['rm1_dir'] = rm1_dir
df_stats['recall@N_m2'] = recallN_m2
df_stats['rm2_dir'] = rm2_dir
df_stats['recall@N_m3'] = recallN_m3
df_stats['rm3_dir'] = rm3_dir

df_stats['mean_ap'] = mean_avg_precision
df_stats['map_dir'] = mean_ap_dir


#%%

# build output path and filename for consolidated results .csv file
out_filename = 'vrd_ppnn_' + trc_id + '_' + prc_id
out_filename = out_filename + '_' + rrc_id + '_results_topN_'
out_filename = out_filename + str(topN).zfill(3) + '_summary.csv'
outfilepath = os.path.join(performance_results_dir, out_filename)

# save data to .csv file
#df = pd.DataFrame(dd)
df_stats.to_csv(outfilepath, index=False)

print(f'consolidated results: {out_filename}')

print('processing complete')


