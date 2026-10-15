#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script drives the computation of scores for metrics that relate
to the analysis of predicted VRs, where the predicted predicate is an
owl:SymmetricProperty (ie where the relation is symmetric).

For a given cell, it finds all of the epoch-specific predictions .json files
mentioned in the cell's consolidated results .csv file and, for each one,
generates a .csv file of symmetric pairs metrics scores.  The number of 
output (.csv) files is thus always 12 (4 job runs per cell, with MAX
recall epochs for each of the 3 topN values).

The analysis investigates the extent to which KG reasoning associated with
the inference semantics of owl:SymmetricProperty induces a PPNN to
predict VRs in symmetric pairs.
'''


#%%

import os
import pandas as pd
import json 

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


#%% specify the base directory for all the predictions .json files

predictions_dir_base = os.path.join(root_dir, 'research', 'results-archive', 'nnkgs0')
predictions_dir_base = os.path.join(predictions_dir_base, 'setID-01', 'trial-02')

predictions_dir_base = os.path.expanduser(predictions_dir_base)

print(f'predictions directory base: {predictions_dir_base}')


#%% specify the experiment space cell to be processed

# specify the cell components
trc_id = 'trc6470'
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


#%%

# set path to directory in which the NeSy4VRD annotations files reside
anno_dir = os.path.join('..', 'data', 'annotations')

# get the NeSy4VRD predicate names
path = os.path.join(anno_dir, 'nesy4vrd_predicates.json')

with open(path, 'r') as fp:
    vrd_predicates = json.load(fp)


#%%

# specify the set of NeSy4VRD predicates whose object property counterparts 
# in the VRD-World ontology have been declared to be symmetric 
# (with owl:SymmetricProperty axioms)
nesy4vrd_symmetric_property_names = ['near', 'by', 'next to', 'beside', 
                                     'adjacent to', 'across from', 'with']

# convert the symmetric property names to integer indices (integer labels)
nesy4vrd_symmetric_property_indices = [ vrd_predicates.index(p) for p in nesy4vrd_symmetric_property_names ]


#%%

nrows = cell_con_res_df.shape[0]

for idx in range(nrows):
    
    cell_run_folder = cell_con_res_df.iloc[idx]['expt_cell_run_folder']
    cell_run_topN = cell_con_res_df.iloc[idx]['topN']
    cell_run_epoch = cell_con_res_df.iloc[idx]['max_recall_m2_epoch']
    cell_run_epoch = str(cell_run_epoch).zfill(3)
    
    print(f'Cell job run: {cell_run_folder}, topN: {cell_run_topN}')
       
    # build the name of the target predictions .json file
    # example: vrd_ppnn_trc0460_ckpt_034_prc018_preds.json
    # (nb: this file is cell specific, job run specific and epoch specific)
    predictions_filename = 'vrd_ppnn_' + trc_id + '_ckpt_' + cell_run_epoch + \
                           '_' + prc_id + '_preds.json'
    
    print(f"Predicted VRs filename: {predictions_filename}")
    
    # assemble the file path for the target predictions .json file
    filepath = os.path.join(predictions_dir_base, cell_folder, 
                            cell_run_folder, predictions_filename)
    
    # load the predictions .json file to be processed wrt symmetric pairs
    with open(filepath, 'r') as fp:
        predicted_vrs_per_image = json.load(fp)
    
    # compute image-level scores wrt our 'symmetric pairs' metrics and 
    # receive the results in the form of a dataframe
    res_df = vrdu16.process_predicted_vrs_for_symmetric_pairs(predicted_vrs_per_image,
                                                              nesy4vrd_symmetric_property_indices,
                                                              rrc_id)

    # build output filename 
    output_filename_base = predictions_filename.removesuffix('.json')
    output_filename = output_filename_base + '_symmetric_pairs_' + rrc_id + '.csv'

    # assemble the file path for the predictions symmetric pairs .csv file
    filepath = os.path.join(predictions_dir_base, cell_folder, 
                            cell_run_folder, output_filename)
    
    # save dataframe to .csv file
    res_df.to_csv(filepath, index=True)
    
    print("VR predictions symmetric pairs .csv file saved")
    print(filepath)
    print()


#%%

print()
print('Processing complete')











