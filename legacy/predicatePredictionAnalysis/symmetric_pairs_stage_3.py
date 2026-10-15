#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script summarises consolidated cell results for scores of metrics 
associated with measuring symmetric pairs amongst predicted VRs.


The script processes 1 consolidated cell results for symmetric pairs file 
at a time.

Example input .csv file:
vrd_ppnn_trc0460_prc018_rrc01_consolidated_cell_results_symmetric_pairs.csv 

Output is written to the console, for transcribing into a table.    

The analysis investigates the extent to which KG reasoning associated with
the inference semantics of owl:SymmetricProperty induces a PPNN to
predict VRs in symmetric pairs.
'''


#%%

import os
import pandas as pd

import vrd_utils16 as vrdu16


#%%

root_dir = '~'   # local hard drive
#root_dir = os.path.join(os.sep, 'Volumes', 'My Passport for Mac') # external hard drive


#%%

# specify the experiment family
experiment_family = 'nnkgs0'

central_results_dir = os.path.join(root_dir, 'research', 'results',
                                   experiment_family)
central_results_dir = os.path.expanduser(central_results_dir)

print(f'central results dir: {central_results_dir}')


#%% specify the experiment space cell to be processed

# specify the cell components
trc_id = 'trc0460'
prc_id = 'prc018'
rrc_id = 'rrc01'

# build the filename for the cell's consolidated results symmetric pairs .csv file
filename = 'vrd_ppnn_' + trc_id + '_' + prc_id + '_' + rrc_id + '_'
filename = filename + 'consolidated_cell_results_symmetric_pairs.csv'
sym_pairs_results_filename = filename

print('consolidated cell results symmetric pairs file to be processed:')
print(sym_pairs_results_filename)

sym_pairs_results_path = os.path.join(central_results_dir, sym_pairs_results_filename)


#%%
 
sym_pairs_res_df = pd.read_csv(sym_pairs_results_path) 

print('consolidated cell symmetric pairs results loaded')
print(sym_pairs_res_df.shape)


#%% compute the average of the mean scores for the 4 job runs

results = vrdu16.summarise_consolidated_sym_pairs_results(sym_pairs_res_df)


#%% display the results to the console

for key, val in results.items():
    print(key)
    for key2, val2 in val.items():
        print(key2, val2)
    print()


#%%

print()
print('Processing complete')











