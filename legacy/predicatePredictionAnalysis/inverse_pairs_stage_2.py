#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script drives the computation of scores for metrics that relate
to the analysis of predicted VRs, where the predicted predicate is an
owl:inverseOf some other property (ie where the relation has an inverse).

The script processes 1 cell at a time. Each cell has results for 12 models:
3 topN values * 4 job runs. The output .csv file has summary inverse pair
metric scores for these same 12 models.

Example input .csv file:
vrd_ppnn_trc0460_prc018_rrc01_consolidated_cell_results.csv

Example output .csv file:
vrd_ppnn_trc0460_prc018_rrc01_consolidated_cell_results_inverse_pairs.csv    

The analysis investigates the extent to which KG reasoning associated with
the inference semantics of owl:inverseOf induces a PPNN to
predict VRs in inverse pairs.
'''


#%%

import os
import pandas as pd
import numpy as np


#%%

#root_dir = '~'   # local hard drive
root_dir = os.path.join(os.sep, 'Volumes', 'My Passport for Mac') # external hard drive


#%%

# specify the experiment family
experiment_family = 'nnkgs0'

central_results_dir = os.path.join(root_dir, 'research', 'results',
                                   experiment_family)
central_results_dir = os.path.expanduser(central_results_dir)

print(f'central results dir: {central_results_dir}')


#%% specify the base directory for all the symmetric pairs .csv files

predictions_dir_base = os.path.join(root_dir, 'research', 'results-archive', 'nnkgs0')
predictions_dir_base = os.path.join(predictions_dir_base, 'setID-01', 'trial-02')

predictions_dir_base = os.path.expanduser(predictions_dir_base)

print(f'predictions directory base: {predictions_dir_base}')


#%% specify the experiment space cell to be processed

# specify the cell components
trc_id = 'trc8471'
prc_id = 'prc018'
rrc_id = 'rrc01'

# build the filename for the cell's consolidated results .csv file
filename = 'vrd_ppnn_' + trc_id + '_' + prc_id + '_' + rrc_id + '_'
filename = filename + 'consolidated_cell_results.csv'
cell_cons_results_filename = filename

# build the name of the cell's archive folder
cell_folder = 'cell-' + trc_id + '-' + prc_id + '-' + rrc_id

print('consolidated cell results file to be processed:')
print(cell_cons_results_filename)

cell_cons_results_path = os.path.join(central_results_dir, cell_cons_results_filename)


#%%

def load_consolidated_cell_results(filepath):
    
    results_df = pd.read_csv(filepath) 

    return results_df


#%%

cell_con_res_df = load_consolidated_cell_results(cell_cons_results_path)

print('consolidated cell results loaded')
print(cell_con_res_df.shape)


#%% verify all the files we need actually exist before we try to process them

nrows = cell_con_res_df.shape[0]

for idx in range(nrows):

    cell_run_folder = cell_con_res_df.iloc[idx]['expt_cell_run_folder']
    cell_run_topN = cell_con_res_df.iloc[idx]['topN']
    cell_run_epoch_int = cell_con_res_df.iloc[idx]['max_recall_m2_epoch']
    cell_run_epoch_str = str(cell_run_epoch_int).zfill(3)
    
    # build the name of the target inverse pairs predictions .csv file
    # to be summarised;
    # example: vrd_ppnn_trc0460_ckpt_034_prc018_preds_inverse_pairs.csv
    # (nb: this file is cell specific, job run specific and epoch specific)
    inv_pairs_filename = 'vrd_ppnn_' + trc_id + '_ckpt_' + cell_run_epoch_str + \
                           '_' + prc_id + '_preds_inverse_pairs_' + rrc_id + '.csv'
    
    # assemble the file path for the target inverse pairs .csv file
    filepath = os.path.join(predictions_dir_base, cell_folder, 
                            cell_run_folder, inv_pairs_filename)
    
    print(filepath)
    
    # get the image-level scores of our inverse pairs metrics
    res_df = pd.read_csv(filepath)

print()
print('all 12 inverse pairs .csv files can be found and loaded')


#%% initialise lists to hold results for eventual dataframe columns

# context features
experiment_cell_run_folders = []
topN_values = []
max_recall_m2_epoch = []

# inverse pair metric scores (means)
n_ip_pvrs = []
n_ip_pvr_pairs = []
n_ip_pvr_singletons = []
proportion_ip_pvrs_in_pairs = []
n_pairs_subm_together_topN_025 = []
n_pairs_subm_together_topN_050 = []
n_pairs_subm_together_topN_100 = []
n_pairs_subm_together_topN_999 = []
mean_pairs_confidence_absolute_differences = []
n_ip_pvrs_hits_topN_025 = []
n_ip_pvrs_hits_topN_050 = []
n_ip_pvrs_hits_topN_100 = []
n_ip_pvrs_hits_topN_999 = []
proportion_ip_pvrs_hits_topN_025 = []
proportion_ip_pvrs_hits_topN_050 = []
proportion_ip_pvrs_hits_topN_100 = []
proportion_ip_pvrs_hits_topN_999 = []
mean_ip_pvr_confidence = []


#%%

n_decimal = 7

nrows = cell_con_res_df.shape[0]

for idx in range(nrows):
    
    cell_run_folder = cell_con_res_df.iloc[idx]['expt_cell_run_folder']
    cell_run_topN = cell_con_res_df.iloc[idx]['topN']
    cell_run_epoch_int = cell_con_res_df.iloc[idx]['max_recall_m2_epoch']
    cell_run_epoch_str = str(cell_run_epoch_int).zfill(3)
    
    # save the context information for the mean metric scores for the 
    # current cell/run/epoch combination
    experiment_cell_run_folders.append(cell_run_folder)
    topN_values.append(cell_run_topN)
    max_recall_m2_epoch.append(cell_run_epoch_int)
    
    # build the name of the target inverse pairs predictions .csv file
    # to be summarised;
    # example: vrd_ppnn_trc0460_ckpt_034_prc018_preds_inverse_pairs.csv
    # (nb: this file is cell specific, job run specific and epoch specific)
    inv_pairs_filename = 'vrd_ppnn_' + trc_id + '_ckpt_' + cell_run_epoch_str + \
                           '_' + prc_id + '_preds_inverse_pairs_' + rrc_id + '.csv'
    
    print(f"Inverse pairs .csv filename: {inv_pairs_filename}")
    
    # assemble the file path for the target inverse pairs .csv file
    filepath = os.path.join(predictions_dir_base, cell_folder, 
                            cell_run_folder, inv_pairs_filename)
    
    # get the image-level scores of our inverse pairs metrics
    res_df = pd.read_csv(filepath)
    
    # calculate the means of all of the image-level metric scores
    # (nb: by default, Pandas Series.mean() has 'skipna=True', so any 
    #  missing values will be skipped over)
    mean_n_ip_pvrs = np.round(res_df['n_ip_pvrs'].mean(), n_decimal)
    mean_n_ip_pvr_pairs = np.round(res_df['n_ip_pvr_pairs'].mean(), n_decimal)
    mean_n_ip_pvr_singletons = np.round(res_df['n_ip_pvr_singletons'].mean(), n_decimal)
    mean_prop_ip_pvrs_in_pairs = np.round(res_df['proportion_ip_pvrs_in_pairs'].mean(), n_decimal)
    mean_n_pairs_subm_together_topN_025 = np.round(res_df['n_pairs_subm_together_topN_025'].mean(), n_decimal)
    mean_n_pairs_subm_together_topN_050 = np.round(res_df['n_pairs_subm_together_topN_050'].mean(), n_decimal)
    mean_n_pairs_subm_together_topN_100 = np.round(res_df['n_pairs_subm_together_topN_100'].mean(), n_decimal)
    mean_n_pairs_subm_together_topN_999 = np.round(res_df['n_pairs_subm_together_topN_999'].mean(), n_decimal)
    mean_mean_pairs_confidence_absolute_differences = np.round(res_df['mean_pairs_confidence_absolute_differences'].mean(), n_decimal)
    mean_n_ip_pvrs_hits_topN_025 = np.round(res_df['n_ip_pvrs_hits_topN_025'].mean(), n_decimal)
    mean_n_ip_pvrs_hits_topN_050 = np.round(res_df['n_ip_pvrs_hits_topN_050'].mean(), n_decimal)
    mean_n_ip_pvrs_hits_topN_100 = np.round(res_df['n_ip_pvrs_hits_topN_100'].mean(), n_decimal)
    mean_n_ip_pvrs_hits_topN_999 = np.round(res_df['n_ip_pvrs_hits_topN_999'].mean(), n_decimal)
    mean_prop_ip_pvrs_hits_topN_025 = np.round(res_df['proportion_ip_pvrs_hits_topN_025'].mean(), n_decimal)
    mean_prop_ip_pvrs_hits_topN_050 = np.round(res_df['proportion_ip_pvrs_hits_topN_050'].mean(), n_decimal)
    mean_prop_ip_pvrs_hits_topN_100 = np.round(res_df['proportion_ip_pvrs_hits_topN_100'].mean(), n_decimal)
    mean_prop_ip_pvrs_hits_topN_999 = np.round(res_df['proportion_ip_pvrs_hits_topN_999'].mean(), n_decimal)
    mean_mean_ip_pvr_confidence = np.round(res_df['mean_ip_pvr_confidence'].mean(), n_decimal)

    # save the mean metric scores for the current cell/run/epoch combination
    n_ip_pvrs.append(mean_n_ip_pvrs)
    n_ip_pvr_pairs.append(mean_n_ip_pvr_pairs)
    n_ip_pvr_singletons.append(mean_n_ip_pvr_singletons)
    proportion_ip_pvrs_in_pairs.append(mean_prop_ip_pvrs_in_pairs)
    n_pairs_subm_together_topN_025.append(mean_n_pairs_subm_together_topN_025)
    n_pairs_subm_together_topN_050.append(mean_n_pairs_subm_together_topN_050)
    n_pairs_subm_together_topN_100.append(mean_n_pairs_subm_together_topN_100)
    n_pairs_subm_together_topN_999.append(mean_n_pairs_subm_together_topN_999)
    mean_pairs_confidence_absolute_differences.append(mean_mean_pairs_confidence_absolute_differences)
    n_ip_pvrs_hits_topN_025.append(mean_n_ip_pvrs_hits_topN_025)
    n_ip_pvrs_hits_topN_050.append(mean_n_ip_pvrs_hits_topN_050)
    n_ip_pvrs_hits_topN_100.append(mean_n_ip_pvrs_hits_topN_100)
    n_ip_pvrs_hits_topN_999.append(mean_n_ip_pvrs_hits_topN_999)
    proportion_ip_pvrs_hits_topN_025.append(mean_prop_ip_pvrs_hits_topN_025)
    proportion_ip_pvrs_hits_topN_050.append(mean_prop_ip_pvrs_hits_topN_050)
    proportion_ip_pvrs_hits_topN_100.append(mean_prop_ip_pvrs_hits_topN_100)
    proportion_ip_pvrs_hits_topN_999.append(mean_prop_ip_pvrs_hits_topN_999)
    mean_ip_pvr_confidence.append(mean_mean_ip_pvr_confidence)   


#%%

# gather the summary (mean) inverse pair metric scores into a dataframe
df = pd.DataFrame({'expt_cell_run_folder': experiment_cell_run_folders,
                   'topN': topN_values,
                   'max_recall_m2_epoch': max_recall_m2_epoch,
                   'n_ip_pvrs': n_ip_pvrs,
                   'n_ip_pvr_pairs': n_ip_pvr_pairs,
                   'n_ip_pvr_singletons': n_ip_pvr_singletons, 
                   'proportion_ip_pvrs_in_pairs': proportion_ip_pvrs_in_pairs,
                   'n_pairs_subm_together_topN_025': n_pairs_subm_together_topN_025, 
                   'n_pairs_subm_together_topN_050': n_pairs_subm_together_topN_050,
                   'n_pairs_subm_together_topN_100': n_pairs_subm_together_topN_100,
                   'n_pairs_subm_together_topN_999': n_pairs_subm_together_topN_999,
                   'mean_pairs_confidence_absolute_differences': mean_pairs_confidence_absolute_differences, 
                   'n_ip_pvrs_hits_topN_025': n_ip_pvrs_hits_topN_025,
                   'n_ip_pvrs_hits_topN_050': n_ip_pvrs_hits_topN_050,
                   'n_ip_pvrs_hits_topN_100': n_ip_pvrs_hits_topN_100,
                   'n_ip_pvrs_hits_topN_999': n_ip_pvrs_hits_topN_999,
                   'proportion_ip_pvrs_hits_topN_025': proportion_ip_pvrs_hits_topN_025,
                   'proportion_ip_pvrs_hits_topN_050': proportion_ip_pvrs_hits_topN_050, 
                   'proportion_ip_pvrs_hits_topN_100': proportion_ip_pvrs_hits_topN_100,
                   'proportion_ip_pvrs_hits_topN_999': proportion_ip_pvrs_hits_topN_999,
                   'mean_ip_pvr_confidence': mean_ip_pvr_confidence
                  })


#%%

# build output filename 
output_filename_base = cell_cons_results_filename.removesuffix('.csv')
output_filename = output_filename_base + '_inverse_pairs_' + rrc_id + '.csv'

# assemble the file path for the symmetric pairs consolidated results .csv file
filepath = os.path.join(central_results_dir, output_filename)
    
# save dataframe to .csv file
df.to_csv(filepath, index=True)
    
print()
print("Inverse pairs consolidated results .csv file saved")
print(filepath)


#%%

print()
print('Processing complete')











