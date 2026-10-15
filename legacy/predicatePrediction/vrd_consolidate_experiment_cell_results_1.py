#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script represents step 9 of the PPNN pipeline v4.

This script consolidates MAX recall@N-based metric scores for multiple runs
of a given experiment cell of the experiment space.

Given a set of directory names indicated by a name pattern, we load the 
summary results .csv files (for topN = 25, 50, 100) found in the multiple 
work directories used by the job runs for a given experiment cell.

We extract the MAX recall@N-based metric scores that have already been 
marked in these source .csv files by preceding step of the PPNN pipeline.

We consolidate all of these granular MAX epochs and MAX scores scatter across
multiple .csv files in multiple directories into a single, higher-level,
summary .csv file designed to contain all of the MAX recall@N-based metric 
epochs and scores generated for a given experiment cell, no matter how many 
runs of the experiment cell might be executed over time.

For example, at time t we might do 4 runs for a given experiment cell and
run this script to consolidate the MAX results for those 4 job runs. The mean
epochs and scores we can calculate and report will be from samples of size 4.
Later, however, we may find time to do another 4 runs, giving a total of 8
job run directories containing top25, top50 and top100 summary .csv results
files. We can then run this script again (step 9 of the PPNN pipeline) to
consolidate the results from the 8 directories, meaning we can generate and
report results from samples of size 8, but without having to re-run the first 
4 job runs.  We can re-consolidate the first 4 runs with the 2nd 4 runs.

The higher-level summary .csv file for the experiment cell is written to 
a central results directory designated for a given experiment family.
'''

#%%

import os
import sys
import glob
import json
import pandas as pd
from datetime import date

import vrd_utils14 as vrdu14


#%% gather arguments supplied to this script

# set the experiment space training region cell id
if len(sys.argv) > 1:
    trc_id = sys.argv[1]
else:
    trc_id = 'trc0460'  # 'trc0460'
if not trc_id.startswith('trc'):
    raise ValueError(f'training region cell id {trc_id} not recognised')

# set the experiment space prediction region cell id
if len(sys.argv) > 2:
    prc_id = sys.argv[2]
else:
    prc_id = 'prc018'
if not prc_id.startswith('prc'):
    raise ValueError(f'prediction region cell id {prc_id} not recognised')

# set the experiment space results region cell id
if len(sys.argv) > 3:
    rrc_id = sys.argv[3]
else:
    rrc_id = 'rrc01'
if not rrc_id.startswith('rrc'):
    raise ValueError(f'results region cell id {rrc_id} not recognised')

# set the platform on which the training script is running
if len(sys.argv) > 4:
    platform = sys.argv[4]
else:
    platform = 'macstudio'

if not platform in vrdu14.supported_platforms:
    raise ValueError(f'platform {platform} not recognised')

# set the pattern of directory names to process
if len(sys.argv) > 5:
    dirname_pattern = sys.argv[5]
else:
    dirname_pattern = 'ppnn-cell-run*'   # 'ppnn-cell-run*'

# ensure the dirname_pattern ends with a directory separator character
# so that the glob.glob(path) function will match directory names rather
# than file names
if not dirname_pattern.endswith(os.sep):
    dirname_pattern = dirname_pattern + os.sep

# nb: dirname_pattern specifies the set of directories containing the 
# results of pipeline job runs for of a given experiment cell of the
# experiment space; these directories contain the summary topN results
# .csv files from which to extract and consolidate the MAX recall@N epochs
# and MAX recall@N scores


#%% lookup config of the experiment space prediction region cell

# specify the experiment family
experiment_family = 'nnkgs0'      # NN+KG_S0

#
# get the training region cell configuration (dimension levels)
#

cfg = vrdu14.get_training_region_cell_config(experiment_family, trc_id)

# set the training region dimension levels
tr_d_model1 = cfg['D_model1_level']
tr_d_model2 = cfg['D_model2_level']      # new - add extra output neuron or not
tr_d_dataCat = cfg['D_dataCat_level']
tr_d_dataFeat = cfg['D_dataFeat_level']
tr_d_kgS0 = cfg['D_kgS0_level']                  # was tr_d_target
tr_d_kgS1 = cfg['D_kgS1_level']
tr_d_kgS2 = cfg['D_kgS2_level']
tr_d_kgS3 = cfg['D_kgS3_level']
tr_d_nopredTarget = cfg['D_nopredTarget_level']  # was tr_d_loss
tr_d_onto = cfg['D_onto_level']

#
# get the prediction region cell configuration (dimension levels)
#

# get the training region cell configuration (dimension levels)
cfg = vrdu14.get_prediction_region_cell_config(experiment_family, prc_id)

# assign the dimension levels to more familiar variable names
pr_d_predConf = cfg['D_predConf_level']        
pr_d_predMax = cfg['D_predMax_level']
pr_d_predKG = cfg['D_predKG_level']
pr_d_predNoPred = cfg['D_predNoPred_level'] 

#
# get the results region cell configuration (dimension levels)
#

cfg = vrdu14.get_results_region_cell_config(experiment_family, rrc_id)

# the composition of the test set VR annotations to use as targets
# for performance evaluation purposes
rr_d_perfTarget = cfg['D_perfTarget_level']


#%% build the name of the output directory (central results directory)

#root_dir = '~'   # local hard drive
root_dir = os.path.join(os.sep, 'Volumes', 'My Passport for Mac') # external hard drive

if platform == 'hyperion':
    central_results_dir = os.path.join('~', 'sharedscratch', 'research', 
                                       'results', experiment_family)
else:
    central_results_dir = os.path.join(root_dir, 'research', 'results',
                                       experiment_family)

central_results_dir = os.path.expanduser(central_results_dir)

print(f'central results dir: {central_results_dir}')


#%% declare (record) experiment meta data

print()
print(f'# experiment family: {experiment_family}')
print(f'# training region cell id: {trc_id}')
print(f'# prediction region cell id: {prc_id}')
print(f'# results region cell id: {rrc_id}')
print()
print(f'Date: {date.today()}')

# the platform on which we're running
print(f'Platform: {platform}')

# the name of this Python script
#scriptName = sys.argv[0]
scriptName = os.path.basename(__file__)
print(f'Script name: {scriptName}')

# the name of the conda environment in which we're running
print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")


#%% set the filename pattern of the performance results files to be processed

# Each target directory contains multiple results .csv files relating to a
# given job run for a given experiment space cell. These are the input files
# we wish to process. For example, we might have:
#
#     ppnn-cell-run1/
#         vrd_ppnn_trc0460_prc018_rrc01_results_topN_025_summary.csv
#         vrd_ppnn_trc0460_prc018_rrc01_results_topN_050_summary.csv
#         vrd_ppnn_trc0460_prc018_rrc01_results_topN_100_summary.csv 
#         vrd_ppnn_trc0460_prc018_rrc01_results_topN_999_summary.csv
    
# set the input filename pattern
filename_pattern = 'vrd_ppnn_' + trc_id + '_' + prc_id + '_' + rrc_id
filename_pattern = filename_pattern + '_results_topN_*_summary.csv'

print(f"Results filename pattern: {filename_pattern}")


#%% get the names of the directories to be processed

# build the path to the parent directory that contains the individual
# experiment cell run directories we wish to process
if platform == 'hyperion':
    parent_dir = os.path.join('~', 'sharedscratch', 'research')
else:
    #parent_dir = os.path.join(root_dir, 'research')
    
    # nb: the following setting is for reprocessing archived cells, if required
    parent_dir = os.path.join(root_dir, 'research', 'results-archive', 'nnkgs0')
    parent_dir = os.path.join(parent_dir, 'setID-01', 'trial-02') 
    parent_dir = os.path.join(parent_dir, 'cell-trc3003-prc018-rrc01')

parent_dir = os.path.expanduser(parent_dir)

print(f'parent directory: {parent_dir}')

# build the path pattern
path_pattern = os.path.join(parent_dir, dirname_pattern)

# get the names of the directories to be processed
dirnames_to_process = glob.glob(path_pattern)
dirnames_to_process = sorted(dirnames_to_process)

if len(dirnames_to_process) == 0:
    raise ValueError(f'no directories found for pattern {dirname_pattern}')

print(f'directories to process: {len(dirnames_to_process)}')
for idx in range(len(dirnames_to_process)):
    print(idx, dirnames_to_process[idx])


#%%

def load_performance_results(path):

    with open(path, 'r') as fp:
        results_data = json.load(fp)

    return results_data


#%% function to get the input files to be processed for one directory

def get_results_filepaths(directory, filename_pattern):

    path = os.path.join(directory, filename_pattern)

    # gather the names of all of the performance results files to be processed
    perf_results_paths = glob.glob(path)
    perf_results_paths = sorted(perf_results_paths)
    
    #print(f'Number of performance results files to process: {len(perf_results_paths)}')
    
    return perf_results_paths


#%% initialise lists to hold results for eventual dataframe columns

experiment_cell_run_folders = []
topN_values = []

max_recallN_m1_epochs = []
max_recallN_m1_predicates = []
max_recallN_m1_vrs_avail = []
max_recallN_m1_vrs = []
max_recallN_m1_hits = []
max_recallN_m1_scores = []

max_recallN_m2_epochs = []
max_recallN_m2_predicates = []
max_recallN_m2_vrs_avail = []
max_recallN_m2_vrs = []
max_recallN_m2_hits = []
max_recallN_m2_scores = []

max_recallN_m3_epochs = []
max_recallN_m3_predicates = []
max_recallN_m3_vrs_avail = []
max_recallN_m3_vrs = []
max_recallN_m3_hits = []
max_recallN_m3_scores = []

# mAP (mean average precision)
max_mean_ap_epochs = []
max_mean_ap_scores = []


#%% main processing loop

for dirname in dirnames_to_process:
    
    # get the filenames to process in the current directory
    filepaths_to_process = get_results_filepaths(dirname, filename_pattern)
    if len(filepaths_to_process) == 0:
        continue
        #raise ValueError(f'no target .csv files found in directory {dirname}')
    
    # get the folder name from the end of the dirname path
    tokens = dirname.split(sep=os.sep)
    foldername = tokens[len(tokens)-2]
    
    print(f'foldername: {foldername}, results files: {len(filepaths_to_process)}')
    
    # iterate over the results summary .csv files for different values of
    # topN for the current directory (job run of the experiment cell)
    for filepath in filepaths_to_process:
        
        filename = filepath.removeprefix(dirname)
        
        print(f'processing filename: {filename}')
        
        # extract trc_id, prc_id, rrc_id, and topN from the filename
        tokens = filename.split(sep='_')
        if len(tokens) != 9:
            raise ValueError('unexpected number of components in filename')
        file_trc_id = tokens[2]
        file_prc_id = tokens[3]
        file_rrc_id = tokens[4]
        file_topN = int(tokens[7])
        
        # ensure the results file belongs to the expected experiment space cell
        if file_trc_id != trc_id:
            raise ValueError(f'training region cell mismatch: {trc_id}, {file_trc_id}')
        if file_prc_id != prc_id:
            raise ValueError(f'prediction region cell mismatch:" {prc_id}, {file_prc_id}')        
        if file_rrc_id != rrc_id:
            raise ValueError(f'results region cell mismatch:" {rrc_id}, {file_rrc_id}')
        if not file_topN in [25, 50, 100, 999]:
            raise ValueError(f'unexpected topN value: {file_topN}')
        
        # load the next *_results_topN_*_summary.csv file
        dfres = pd.read_csv(filepath)        

        # get the max score for metric recall@N_m1, and associated epoch
        mask = dfres['rm1_dir'] == 'MAX'
        if sum(mask) != 1:
            raise ValueError(f'unexpected number of MAX instances in column rm1_dir: {sum(mask)}')        
        max_recall_m1_epoch = dfres[mask]['epoch'].item()
        max_recall_m1_predicates = dfres[mask]['mean_pred_preds'].item()
        max_recall_m1_vrs_avail = dfres[mask]['mean_pred_vrs_avail'].item()
        max_recall_m1_vrs = dfres[mask]['mean_pred_vrs'].item()
        max_recall_m1_hits = dfres[mask]['mean_hits'].item()
        max_recall_m1_score = dfres[mask]['recall@N_m1'].item()
        
        # get the max score for metric recall@N_m2, and associated epoch
        mask = dfres['rm2_dir'] == 'MAX'
        if sum(mask) != 1:
            raise ValueError(f'unexpected number of MAX instances in column rm2_dir: {sum(mask)}')        
        max_recall_m2_epoch = dfres[mask]['epoch'].item()
        max_recall_m2_predicates = dfres[mask]['mean_pred_preds'].item()
        max_recall_m2_vrs_avail = dfres[mask]['mean_pred_vrs_avail'].item()
        max_recall_m2_vrs = dfres[mask]['mean_pred_vrs'].item()
        max_recall_m2_hits = dfres[mask]['mean_hits'].item()
        max_recall_m2_score = dfres[mask]['recall@N_m2'].item()

        # get the max score for metric recall@N_m3, and associated epoch
        mask = dfres['rm3_dir'] == 'MAX'
        if sum(mask) != 1:
            raise ValueError(f'unexpected number of MAX instances in column rm3_dir: {sum(mask)}')        
        max_recall_m3_epoch = dfres[mask]['epoch'].item()
        max_recall_m3_predicates = dfres[mask]['mean_pred_preds'].item()
        max_recall_m3_vrs_avail = dfres[mask]['mean_pred_vrs_avail'].item()
        max_recall_m3_vrs = dfres[mask]['mean_pred_vrs'].item()
        max_recall_m3_hits = dfres[mask]['mean_hits'].item()
        max_recall_m3_score = dfres[mask]['recall@N_m3'].item()
        
        # get the max score for metric mean Avg Precision (mAP), and associated epoch
        mask = dfres['map_dir'] == 'MAX'
        if sum(mask) != 1:
            raise ValueError(f'unexpected number of MAX instances in column map_dir: {sum(mask)}')    
        max_mean_ap_epoch = dfres[mask]['epoch'].item()
        max_mean_ap_score = dfres[mask]['mean_ap'].item()

        # save what has just been extracted
        experiment_cell_run_folders.append(foldername)
        topN_values.append(file_topN)
        
        max_recallN_m1_epochs.append(max_recall_m1_epoch)
        max_recallN_m1_predicates.append(max_recall_m1_predicates)
        max_recallN_m1_vrs_avail.append(max_recall_m1_vrs_avail)
        max_recallN_m1_vrs.append(max_recall_m1_vrs)
        max_recallN_m1_hits.append(max_recall_m1_hits)
        max_recallN_m1_scores.append(max_recall_m1_score)
        
        max_recallN_m2_epochs.append(max_recall_m2_epoch)
        max_recallN_m2_predicates.append(max_recall_m2_predicates)
        max_recallN_m2_vrs_avail.append(max_recall_m2_vrs_avail)
        max_recallN_m2_vrs.append(max_recall_m2_vrs)
        max_recallN_m2_hits.append(max_recall_m2_hits)
        max_recallN_m2_scores.append(max_recall_m2_score)
        
        max_recallN_m3_epochs.append(max_recall_m3_epoch)
        max_recallN_m3_predicates.append(max_recall_m3_predicates)
        max_recallN_m3_vrs_avail.append(max_recall_m3_vrs_avail)
        max_recallN_m3_vrs.append(max_recall_m3_vrs)
        max_recallN_m3_hits.append(max_recall_m3_hits)
        max_recallN_m3_scores.append(max_recall_m3_score) 
        
        max_mean_ap_epochs.append(max_mean_ap_epoch)
        max_mean_ap_scores.append(max_mean_ap_score)    
        


n_rows = len(max_recallN_m1_scores)

# assemble all of the data of interest into a dataframe
df = pd.DataFrame({'family': [experiment_family] * n_rows, 
                   'trc_id': [trc_id] * n_rows,      
                   'd_model1': [tr_d_model1] * n_rows,      
                   'd_model2': [tr_d_model2] * n_rows,      
                   'd_dataCat': [tr_d_dataCat] * n_rows,      
                   'd_dataFeat': [tr_d_dataFeat] * n_rows,      
                   'd_kgS0': [tr_d_kgS0] * n_rows,      
                   'd_kgS1': [tr_d_kgS1] * n_rows,      
                   'd_kgS2': [tr_d_kgS2] * n_rows,      
                   'd_kgS3': [tr_d_kgS3] * n_rows,   
                   'd_nopredTarget': [tr_d_nopredTarget] * n_rows,
                   'd_onto': [tr_d_onto] * n_rows,
                   'prc_id': [prc_id] * n_rows,
                   'd_predConf': [pr_d_predConf] * n_rows,
                   'd_predMax': [pr_d_predMax] * n_rows,
                   'd_predKG': [pr_d_predKG] * n_rows,
                   'd_predNoPred': [pr_d_predNoPred] * n_rows,
                   'rrc_id': [rrc_id] * n_rows,
                   'd_perfTarget': [rr_d_perfTarget] * n_rows,
                   'expt_cell_run_folder': experiment_cell_run_folders,
                   'topN': topN_values,
                   'max_recall_m1_epoch': max_recallN_m1_epochs,
                   'max_recall_m1_predicates': max_recallN_m1_predicates,
                   'max_recall_m1_vrs_avail': max_recallN_m1_vrs_avail, 
                   'max_recall_m1_vrs': max_recallN_m1_vrs,
                   'max_recall_m1_hits': max_recallN_m1_hits,
                   'max_recall_m1_score': max_recallN_m1_scores,
                   'max_recall_m2_epoch': max_recallN_m2_epochs,
                   'max_recall_m2_predicates': max_recallN_m2_predicates,
                   'max_recall_m2_vrs_avail': max_recallN_m2_vrs_avail, 
                   'max_recall_m2_vrs': max_recallN_m2_vrs,
                   'max_recall_m2_hits': max_recallN_m2_hits,
                   'max_recall_m2_score': max_recallN_m2_scores,
                   'max_recall_m3_epoch': max_recallN_m3_epochs,
                   'max_recall_m3_predicates': max_recallN_m3_predicates,
                   'max_recall_m3_vrs_avail': max_recallN_m3_vrs_avail, 
                   'max_recall_m3_vrs': max_recallN_m3_vrs,
                   'max_recall_m3_hits': max_recallN_m3_hits,
                   'max_recall_m3_score': max_recallN_m3_scores,
                   'max_mean_ap_epoch': max_mean_ap_epochs,
                   'max_mean_ap_score': max_mean_ap_scores                   
                  })

      
# build output filename
out_filename = 'vrd_ppnn_' + trc_id + '_' + prc_id + '_' + rrc_id
out_filename = out_filename + '_consolidated_cell_results.csv'

# build output filepath
outfilepath = os.path.join(central_results_dir, out_filename)

# save dataframe to .csv file
df.to_csv(outfilepath, index=False)

print(f'consolidated cell results: {out_filename}')

print('processing complete')

#%%















