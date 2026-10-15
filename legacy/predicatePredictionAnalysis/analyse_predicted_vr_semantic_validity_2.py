#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script processes predicted VRs in a preds.json file for nnkgs1.
The preds.json file in question MUST have been preprocessed to have all
its predicted VRs assigned their 'vr_type_class' by the KG. That is, 
the predicted VRs must already have been extended with their 'vr_type_class', 
as determined by the KG.

The necessary preprocessing can happen in one of two places:
    1. the companion script to this one suffixed with '_1.py'. 
    2. Step 7 (performancee evaluation) of the PPNN pipeline when KG
       filtering is ACTIVE (ie D_predKG == 2)

This script uses the vr_type_class of each predicted VR to calculate
various metrics relating to the frequency with which invalid VRs are
predicted.

This script performs no KG interaction and does not leverage the
master 3D VR type tensor in the central results directory in any way.

Inputs:
* preds.json file 

Outputs:
* log file with metric scores for the current preds.json file  
'''

#%%

import torch 
import numpy as np 
import os
import json
import time 
import sys  

import nesy4vrd_utils4 as vrdu4


#%% load object classes and predicates and convert to ontology names 

# set the path to the directory where the NeSy4VRD visual relationship
# annotations files reside
anno_dir = os.path.join('..', 'data', 'annotations')

# get the master list of NeSy4VRD object class names
vrd_objects_path = os.path.join(anno_dir, 'nesy4vrd_objects.json')

# get the VRD object class names
with open(vrd_objects_path, 'r') as file:
    vrd_object_class_names = json.load(file)

nr_object_classes = len(vrd_object_class_names)

# get the master list of NeSy4VRD predicate names
vrd_predicates_path = os.path.join(anno_dir, 'nesy4vrd_predicates.json')

# get the VRD predicates names
with open(vrd_predicates_path, 'r') as file:
    vrd_predicate_names = json.load(file)

nr_predicates = len(vrd_predicate_names)

ontoClassNames = vrdu4.convert_NeSy4VRD_classNames_to_ontology_classNames(vrd_object_class_names)
ontoPropNames = vrdu4.convert_NeSy4VRD_predicateNames_to_ontology_propertyNames(vrd_predicate_names)


#%% specify the name of the predicted VR (preds.json) file to be processed

pvr_filename = 'vrd_ppnn_trc3003_ckpt_042_prc018_preds.json' 


# ALSO: specify an rrc_id
rrc_id = 'rrc01'


#%% specify the directory where the preds.json file sits

workdir = 'ppnn-workdir2'

root_dir = '~'   # local hard drive
#root_dir = os.path.join(os.sep, 'Volumes', 'My Passport for Mac') # external hard drive

# use this for local hard drive
vr_predictions_dir = os.path.join(root_dir, 'research', workdir)
    
# use this for external hard drive 
#vr_predictions_dir = os.path.join(root_dir, 'research', 'results-archive', 'nnkgs0')
#vr_predictions_dir = os.path.join(vr_predictions_dir, 'setID-01', 'trial-02')
#vr_predictions_dir = os.path.join(vr_predictions_dir, 'cell-trc2470-prc018-rrc01')
#vr_predictions_dir = os.path.join(vr_predictions_dir, workdir)

vr_predictions_dir = os.path.expanduser(vr_predictions_dir)

pvr_filepath = os.path.join(vr_predictions_dir, pvr_filename)


#%% choose whether or not to redirect stdout to a log file

redirect_stdout = True 


#%% build the log file name 

if redirect_stdout:
    logfile_name = pvr_filename.removesuffix('.json')
    logfile_name = logfile_name + '_vr_type_semantic_validity_analysis_log.txt'
else:
    logfile_name = ''

# NOTE: the log file is stored in the same directory as the preds.json file
# and under a similar name


#%% redirect stdout

if redirect_stdout:
    stdout_file_saved = sys.stdout
    logfile_path = os.path.join(vr_predictions_dir, logfile_name)
    print(f'redirecting stdout to log file: {logfile_path}')
    sys.stdout = open(logfile_path, 'w')


#%% record script run info to log file 

scriptName = os.path.basename(__file__)

print('*** analyse predicted vr semantic validity 2 ***')
print()
print(f'script name: {scriptName}')
print()
print(f'predicted VRs file: {pvr_filename}')
print() 
print(f"predicted VRs file directory: {vr_predictions_dir}")
print()
print(f'log file name: {logfile_name}')
print() 


#%% function to load a VR predictions file

def load_predicted_vrs_per_image(filepath):
    
    with open(filepath, 'r') as fp:
        vr_predictions_per_image = json.load(fp)

    return vr_predictions_per_image 


#%% load the predicted VRs file 

predicted_vrs_per_image = load_predicted_vrs_per_image(pvr_filepath)

print() 
print(f'predicted VR file loaded: {pvr_filepath}')


#%% initialise two 3D tensors for storing VR type information 

print()
print('initialising 3D tensors')
print() 

vr_type_classifications = torch.zeros((nr_predicates, nr_object_classes, nr_object_classes))
print()
print(f'vr_type_classifications shape: {vr_type_classifications.shape}')
n_cells_vr_type_classifications = vr_type_classifications.shape[0] * vr_type_classifications.shape[1] * vr_type_classifications.shape[2]
print(f'vr_type_classifications total cells: {n_cells_vr_type_classifications}')

vr_type_counts = torch.zeros((nr_predicates, nr_object_classes, nr_object_classes))
print()
print(f'vr_type_counts shape: {vr_type_counts.shape}')
n_cells_vr_type_counts = vr_type_counts.shape[0] * vr_type_counts.shape[1] * vr_type_counts.shape[2]
print(f'vr_type_counts total cells: {n_cells_vr_type_counts}')
print()


#%%

verbose = True

img_count = 0
pvr_count = 0

n_pvr_topN_025 = 0
n_pvr_topN_025_valid = 0
n_pvr_topN_025_invalid = 0

n_pvr_topN_050 = 0
n_pvr_topN_050_valid = 0
n_pvr_topN_050_invalid = 0

n_pvr_topN_100 = 0
n_pvr_topN_100_valid = 0
n_pvr_topN_100_invalid = 0

n_pvr_topN_999 = 0
n_pvr_topN_999_valid = 0
n_pvr_topN_999_invalid = 0


#%% assemble data and compute some pvr-level metric scores along the way 

start_time = time.time()

for img_idx, batch in enumerate(predicted_vrs_per_image.items()):

    img_count += 1
    
    imname, predicted_vr_dict = batch
    
    predicted_vrs = predicted_vr_dict['predicted_vrs']

    for pvr in predicted_vrs:
        
        # get the VR's property and object class labels (integer indices)   
        prop_idx = pvr['predicate']
        obj1_cls_idx = pvr['subject']['category']
        obj2_cls_idx = pvr['object']['category']
        
        # get the VR type classification: semantically valid (1), invalid (2)
        vr_type_class = pvr['vr_type_class']
        
        # validate the VR type class
        if not vr_type_class in [1.0, 2.0]:
            raise ValueError(f'PROBLEM: vr_type_class not recognised: {vr_type_class}')
        
        #
        # count the predicted VRs
        #
        
        pvr_count += 1
        
        #
        # populate the 3D VR type classifications tensor for the current VR type
        #
        
        if vr_type_classifications[prop_idx, obj1_cls_idx, obj2_cls_idx] == 0.0:
            vr_type_classifications[prop_idx, obj1_cls_idx, obj2_cls_idx] = vr_type_class
        
        #
        # count the instances of predicted VRs by VR type in the VR type counts 3D tensor 
        #
        
        vr_type_counts[prop_idx, obj1_cls_idx, obj2_cls_idx] += 1
        
        #
        # compute most of the pvr-level metrics by counting
        # the predicted VRs by VR type class per topN value
        #
        
        if pvr[rrc_id]['submitted_topN_025'] == 1:
            n_pvr_topN_025  += 1
            if vr_type_class == 1.0:
                n_pvr_topN_025_valid += 1
            else:
                n_pvr_topN_025_invalid += 1
        
        if pvr[rrc_id]['submitted_topN_050'] == 1:
            n_pvr_topN_050  += 1
            if vr_type_class == 1.0:
                n_pvr_topN_050_valid += 1
            else:
                n_pvr_topN_050_invalid += 1 
        
        if pvr[rrc_id]['submitted_topN_100'] == 1:
            n_pvr_topN_100  += 1
            if vr_type_class == 1.0:
                n_pvr_topN_100_valid += 1
            else:
                n_pvr_topN_100_invalid += 1         

        if pvr[rrc_id]['submitted_topN_999'] == 1:
            n_pvr_topN_999  += 1
            if vr_type_class == 1.0:
                n_pvr_topN_999_valid += 1
            else:
                n_pvr_topN_999_invalid += 1 


#%% finish computing the pvr-level metric scores 

pvr_semantic_invalidity_rate_topN_025 = n_pvr_topN_025_invalid / n_pvr_topN_025
pvr_semantic_invalidity_rate_topN_025 = round(pvr_semantic_invalidity_rate_topN_025, 6)

pvr_semantic_invalidity_rate_topN_050 = n_pvr_topN_050_invalid / n_pvr_topN_050
pvr_semantic_invalidity_rate_topN_050 = round(pvr_semantic_invalidity_rate_topN_050, 6)

pvr_semantic_invalidity_rate_topN_100 = n_pvr_topN_100_invalid / n_pvr_topN_100
pvr_semantic_invalidity_rate_topN_100 = round(pvr_semantic_invalidity_rate_topN_100, 6)

pvr_semantic_invalidity_rate_topN_999 = n_pvr_topN_999_invalid / n_pvr_topN_999
pvr_semantic_invalidity_rate_topN_999 = round(pvr_semantic_invalidity_rate_topN_999, 6)


#%% report the pvr-level metric scores 

print()
print() 
print('---------- pvr-level metric scores ----------')
print()
print(f'number of images processed: {img_count}')
print()
print(f'number of predicted VRs: {pvr_count}')
print()

print(f'n_pvr_topN_025: {n_pvr_topN_025}')
print(f'n_pvr_topN_025_valid: {n_pvr_topN_025_valid}')
print(f'n_pvr_topN_025_invalid: {n_pvr_topN_025_invalid}')
print(f'pvr_semantic_invalidity_rate_topN_025: {pvr_semantic_invalidity_rate_topN_025}')

print()
print(f'n_pvr_topN_050: {n_pvr_topN_050}')
print(f'n_pvr_topN_050_valid: {n_pvr_topN_050_valid}')
print(f'n_pvr_topN_050_invalid: {n_pvr_topN_050_invalid}')
print(f'pvr_semantic_invalidity_rate_topN_050: {pvr_semantic_invalidity_rate_topN_050}')

print()
print(f'n_pvr_topN_100: {n_pvr_topN_100}')
print(f'n_pvr_topN_100_valid: {n_pvr_topN_100_valid}')
print(f'n_pvr_topN_100_invalid: {n_pvr_topN_100_invalid}')
print(f'pvr_semantic_invalidity_rate_topN_100: {pvr_semantic_invalidity_rate_topN_100}')

print()
print(f'n_pvr_topN_999: {n_pvr_topN_999}')
print(f'n_pvr_topN_999_valid: {n_pvr_topN_999_valid}')
print(f'n_pvr_topN_999_invalid: {n_pvr_topN_999_invalid}')
print(f'pvr_semantic_invalidity_rate_topN_999: {pvr_semantic_invalidity_rate_topN_999}')


#%% compute VR type classification metric scores

n_vr_types_classified = torch.count_nonzero(vr_type_classifications)

mask = vr_type_classifications == 1.0
n_vr_types_classified_valid = torch.sum(mask)

mask = vr_type_classifications == 2.0
n_vr_types_classified_invalid = torch.sum(mask)

vr_type_semantic_invalidity_rate = n_vr_types_classified_invalid / n_vr_types_classified
vr_type_semantic_invalidity_rate = vr_type_semantic_invalidity_rate.item()
vr_type_semantic_invalidity_rate = round(vr_type_semantic_invalidity_rate, 6)


#%% report the VR type classification metric scores 

print()
print()
print('---------- VR type classification metric scores ----------')
print()
print(f'number of VR types classified: {n_vr_types_classified}')
print()
print(f'number of VR types classified valid: {n_vr_types_classified_valid}')
print()
print(f'number of VR types classified invalid: {n_vr_types_classified_invalid}')
print() 
print(f'VR type semantic invalidity rate: {vr_type_semantic_invalidity_rate}')


#%% compute VR type count metric scores

# get the VR type space cells associated with invalid VR types
mask = vr_type_classifications == 2.0
invalid_vr_type_cells = mask.nonzero()

invalid_vr_types_friendly = []
invalid_vr_types_counts = []

for cell in invalid_vr_type_cells:
    prop_idx, obj1_cls_idx, obj2_cls_idx = cell[0], cell[1], cell[2]
    vr_type_cnt = vr_type_counts[prop_idx, obj1_cls_idx, obj2_cls_idx]
    obj1_cls_name = ontoClassNames[obj1_cls_idx]
    prop_name = ontoPropNames[prop_idx]
    obj2_cls_name = ontoClassNames[obj2_cls_idx]
    vr_type_friendly = '(' + obj1_cls_name + ', ' + prop_name + ', ' + obj2_cls_name + ')'
    invalid_vr_types_friendly.append(vr_type_friendly)
    invalid_vr_types_counts.append(vr_type_cnt)


#%% report the VR type count metric scores

# sort the indices of the invalid VR types in descending order by instance count
ascending_sorted_indices = np.argsort(invalid_vr_types_counts)
descending_sorted_indices = ascending_sorted_indices[::-1]  

print()
print()
print('---------- VR type count metric scores ----------')
print()

print('Invalid VR type | Instance count')
print()
for idx in range(len(descending_sorted_indices)):
    vrt_idx = descending_sorted_indices[idx]
    vr_type_friendly = invalid_vr_types_friendly[vrt_idx]
    vr_type_cnt = invalid_vr_types_counts[vrt_idx]
    print(f'{idx+1}, {vr_type_friendly}, {vr_type_cnt}')


#%%

end_time = time.time()

processing_time_min = (end_time - start_time) / 60

print()
print('---------------------------------------------')
print() 
print(f'processing time: {processing_time_min:.2f} min')
print()

#%%

if redirect_stdout:
    # flush stdout buffer
    sys.stdout.flush()
    # close redirected output file
    sys.stdout.close()
    # restore sys.stdout to original file handler
    sys.stdout = stdout_file_saved

print('Processing completed')




