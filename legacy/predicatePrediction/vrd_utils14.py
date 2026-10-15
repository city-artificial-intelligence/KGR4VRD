#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
Utility functions that provide lookup services for experiment space
configurations.  Each experiment family can have a unique experiment space
and a unique experiment space configuration.

As new experiment families are implemented, this module needs to be updated
to provide specialised lookup services tailored for that experiment family.
'''

#%%

import os
import numpy as np
import pandas as pd


#%% declare the currently supported experiment families

supported_experiment_families = ['nnkgs0'       # NN+KG_S0
                                ]


#%% declare the currently supported platforms on which we can run

supported_platforms = ['mbp16', 'macstudio', 'kratos', 'hyperion', 'aws']


#%%

def get_training_region_cell_config(experiment_family, cell_id_str):
    
    if not experiment_family in supported_experiment_families:
        raise ValueError(f'experiment family {experiment_family} not supported')
    
    if experiment_family == 'nnkgs0':
        results = get_training_region_cell_config_nnkgs0(experiment_family, 
                                                         cell_id_str)
    
    return results


#%%

def get_prediction_region_cell_config(experiment_family, cell_id_str):
    
    if not experiment_family in supported_experiment_families:
        raise ValueError(f'experiment family {experiment_family} not supported')
    
    if experiment_family == 'nnkgs0':
        results = get_prediction_region_cell_config_nnkgs0(experiment_family, 
                                                           cell_id_str)
    
    return results


#%%

def get_results_region_cell_config(experiment_family, cell_id_str):
    
    if not experiment_family in supported_experiment_families:
        raise ValueError(f'experiment family {experiment_family} not supported')
    
    if experiment_family == 'nnkgs0':
        results = get_results_region_cell_config_nnkgs0(experiment_family, 
                                                        cell_id_str)
    
    return results


#%%

def get_training_region_cell_config_nnkgs0(experiment_family,
                                           cell_id_str):

    if not cell_id_str.startswith('trc'):
        raise ValueError(f'training region cell ID {cell_id_str} not recognised')
    
    experiment_space_model_dir = os.path.join('..', 'experimentSpaceModels', 
                                               experiment_family)
    
    experiment_space_model_file = 'training_region.csv'  

    filepath = os.path.join(experiment_space_model_dir, 
                            experiment_space_model_file)

    #dtypes = {'invalid_cell': str}

    #df = pd.read_csv(filepath, dtype=dtypes)
    df = pd.read_csv(filepath)
    
    cell_id = int(cell_id_str.removeprefix('trc'))
    #print(f'dh cell_id: {cell_id}')
    
    #print(f"dh Training_Region_Cell_ID: {df['Training_Region_Cell_ID'].values}")
    
    mask = df['Training_Region_Cell_ID'].values == cell_id

    nhits = np.sum(mask)
    #print((f'dh nhits: {nhits}'))
    
    if nhits < 1:
        raise ValueError(f'training region cell ID {cell_id} not found')
    
    if nhits > 1:
        raise ValueError(f'training region cell ID {cell_id} has multiple instances')
    
    cell_id_config = df[mask]
    
    # ensure the cell is valid (ie represents a viable experiment)
    val = cell_id_config['invalid_cell'].item()
    if pd.isna(val):
        pass   # the cell is NOT marked as being invalid
    else:
        raise ValueError(f'training region cell ID {cell_id} is marked as invalid')
        
    d_model1_level = int(cell_id_config['D_model1'].item()) 
    d_model2_level = int(cell_id_config['D_model2'].item())
    d_datacat_level = int(cell_id_config['D_dataCat'].item())   
    d_datafeat_level = int(cell_id_config['D_dataFeat'].item())    
    d_kgS0_level = int(cell_id_config['D_kgS0'].item())
    d_kgS1_level = int(cell_id_config['D_kgS1'].item())
    d_kgS2_level = int(cell_id_config['D_kgS2'].item())
    d_kgS3_level = int(cell_id_config['D_kgS3'].item())
    d_nopredTarget_level = int(cell_id_config['D_nopredTarget'].item())
    d_onto_level = int(cell_id_config['D_onto'].item())
   
    results = {}
    results['training_region_cell_id'] = cell_id
    results['D_model1_level'] = d_model1_level
    results['D_model2_level'] = d_model2_level
    results['D_dataCat_level'] = d_datacat_level
    results['D_dataFeat_level'] = d_datafeat_level
    results['D_kgS0_level'] = d_kgS0_level
    results['D_kgS1_level'] = d_kgS1_level
    results['D_kgS2_level'] = d_kgS2_level
    results['D_kgS3_level'] = d_kgS3_level
    results['D_nopredTarget_level'] = d_nopredTarget_level
    results['D_onto_level'] = d_onto_level
    
    return results


#%%

def get_prediction_region_cell_config_nnkgs0(experiment_family,
                                             cell_id_str):

    if not cell_id_str.startswith('prc'):
        raise ValueError(f'prediction region cell ID {cell_id_str} not recognised')
    
    experiment_space_model_dir = os.path.join('..', 'experimentSpaceModels', 
                                               experiment_family)
    
    experiment_space_model_file = 'prediction_region.csv'  

    filepath = os.path.join(experiment_space_model_dir, 
                            experiment_space_model_file)

    df = pd.read_csv(filepath)
    
    cell_id = int(cell_id_str.removeprefix('prc'))
    
    mask = df['Prediction_Region_Cell_ID'].values == cell_id

    nhits = np.sum(mask)
    
    if nhits < 1:
        raise ValueError(f'prediction region cell ID {cell_id} not found')
    
    if nhits > 1:
        raise ValueError(f'prediction region cell ID {cell_id} has multiple instances')
    
    cell_id_config = df[mask]
    
    d_predConf_level = cell_id_config['D_predConf'].item() 
    d_predMax_level = int(cell_id_config['D_predMax'].item())
    d_predKG_level = int(cell_id_config['D_predKG'].item())   
    d_predNoPred_level = int(cell_id_config['D_predNoPred'].item())    

    results = {}
    results['prediction_region_cell_id'] = cell_id
    results['D_predConf_level'] = d_predConf_level
    results['D_predMax_level'] = d_predMax_level
    results['D_predKG_level'] = d_predKG_level
    results['D_predNoPred_level'] = d_predNoPred_level

    return results


#%%

def get_results_region_cell_config_nnkgs0(experiment_family,
                                          cell_id_str):

    if not cell_id_str.startswith('rrc'):
        raise ValueError(f'results region cell ID {cell_id_str} not recognised')
    
    experiment_space_model_dir = os.path.join('..', 'experimentSpaceModels', 
                                               experiment_family)
    
    experiment_space_model_file = 'results_region.csv'  

    filepath = os.path.join(experiment_space_model_dir, 
                            experiment_space_model_file)

    df = pd.read_csv(filepath)
        
    cell_id = int(cell_id_str.removeprefix('rrc'))
    
    mask = df['Results_Region_Cell_ID'].values == cell_id

    nhits = np.sum(mask)
    
    if nhits < 1:
        raise ValueError(f'results region cell ID {cell_id} not found')
    
    if nhits > 1:
        raise ValueError(f'results region cell ID {cell_id} has multiple instances')
    
    cell_id_config = df[mask]
    
    d_perfTarget_level = int(cell_id_config['D_perfTarget'].item()) 
  
    results = {}
    results['results_region_cell_id'] = cell_id
    results['D_perfTarget_level'] = d_perfTarget_level

    return results


