#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script summarises consolidated cell results for scores of metrics 
associated with measuring symmetric pairs amongst predicted VRs.

The script processes multiple consolidated cell results for symmetric pairs 
files at once, and saves an output .csv file, formatted like a table,
for rendering as a table in LaTeX.  The table includes reports metric scores
associated with 3 values of topN: 25, 50, 100

Example input .csv files:
vrd_ppnn_trc0460_prc018_rrc01_consolidated_cell_results_symmetric_pairs.csv 
vrd_ppnn_trc2470_prc018_rrc01_consolidated_cell_results_symmetric_pairs.csv 

Output is written to a .csv file in the central results directory.

The analysis investigates the extent to which KG reasoning associated with
the inference semantics of owl:SymmetricProperty induces a PPNN to
predict VRs in symmetric pairs.
'''


#%%

import os
import pandas as pd

import vrd_utils16 as vrdu16


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


#%% specify the cells to compare, and to include in the table

# note: specify the cells in the order (top-down) that you want them to
# appear (left-right) in the table (.csv file) 

cell_IDs = [ ['trc0460', 'prc018', 'rrc01'], 
             ['trc3471', 'prc018', 'rrc01'], 
             ['trc2470', 'prc018', 'rrc01'],
             ['trc0471', 'prc018', 'rrc01'],
             ['trc6470', 'prc018', 'rrc01']]

# example input files:
#   'vrd_ppnn_trc0460_prc018_rrc01_consolidated_cell_results_symmetric_pairs_rrc01.csv',
#   'vrd_ppnn_trc2470_prc018_rrc01_consolidated_cell_results_symmetric_pairs_rrc01.csv'


#%% initialise lists to hold results for eventual dataframe columns

# context features
topN_025 = [25] * 9
topN_050 = [50] * 9
topN_100 = [100] * 9
topN_999 = [999] * 9
metric_names = ['n_sp_pvrs', 'n_sp_pvr_pairs', 'n_sp_pvr_singletons', 
                'prop_sp_pvrs_in_pairs', 'mean_sp_pvr_conf', 
                'mean_pairs_conf_abs_diff', 'n_pairs_subm_together_topN', 
                'n_sp_pvrs_hits_topN', 'prop_sp_pvrs_hits_topN']

# lists for storing column data
n_sp_pvrs = []
n_sp_pvr_pairs = []
n_sp_pvr_singletons = []
prop_sp_pvrs_in_pairs = []
mean_sp_pvr_conf = []
mean_pairs_conf_abs_diff = []
n_pairs_subm_together_topN = []
n_sp_pvrs_hits_topN = []
prop_sp_pvrs_hits_topN = []

cell_columns = {}


#%%

for cell in cell_IDs:
    
    trc_id = cell[0]
    prc_id = cell[1]
    rrc_id = cell[2]
    
    cell_name = 'cell-' + trc_id + '-' + prc_id + '-' + rrc_id
    
    # build the filename for the cell's consolidated results 
    # symmetric pairs .csv file
    filename = 'vrd_ppnn_' + trc_id + '_' + prc_id + '_' + rrc_id + '_'
    filename = filename + 'consolidated_cell_results_symmetric_pairs_' + rrc_id + '.csv'
    
    filepath = os.path.join(central_results_dir, filename)
    
    # load the .csv for the current cell
    sym_pairs_res_df = pd.read_csv(filepath) 
    
    # summarise the results and prepare them for presentation
    res = vrdu16.summarise_consolidated_sym_pairs_results(sym_pairs_res_df)
    
    # store the summarised results
    scores = []
    for topN_entry, metric_scores in res.items():
        scores += list(metric_scores.values())
  
    cell_columns[cell_name] = scores


#%% assemble the data into a dictionary for creating a dataframe

results = {}
results['topN'] = topN_025 + topN_050 + topN_100 + topN_999
results['metric'] = metric_names + metric_names + metric_names + metric_names

for cell_name, cell_scores in cell_columns.items():
    results[cell_name] = cell_scores


#%% transform the dictionary into a dataframe

df = pd.DataFrame(results)


#%% set output filename and filepath

output_filename = 'symmetric_pairs_stage_4_table_output_' + rrc_id + '.csv'

filepath = os.path.join(central_results_dir, output_filename)


#%% save dataframe to a .csv file

df.to_csv(filepath, index=True)

print('symmetric pairs results table saved')
print(filepath)

# NOTE: if the results are interesting and need to be preserved, the
# name of the output file must be changed manually; otherwise, the next
# time this script is run it may over-write the previous output file


#%%

print()
print('Processing complete')











