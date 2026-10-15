#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script processes predicted VRs in a preds.json file for nnkgs1.
It extends each pvr with the binary class (semantically valid or 
semantically invalid) of its VR type.  So it extends a preds.json file, 
giving a new attribute to every pvr for every image: vr_type_class

In the process of doing this, the script builds and extends the 
**master 3D VR type tensor** that models the entire VR type space. This
master 3D tensor is a global, shared resource that we maintain in order to
minimise the need for KG interactions for binary classification.
Ultimately, only the KG can determine the class (valid or invalid) of a
VR type. But by remembering its decisions, in the master 3D VR type tensor,
and accumulating new class decisions in it each time a releveant script runs,
we can gradually reduce the need for KG interaction and hence speed-up any
nnkgs1 processing we do that involves KG interaction.

The vr_type_class attribute stored with each pvr can then be used in
subsequent scripts to analyse the pvrs and calculate metrics and drive
visualisations (plots or tables).

Requirements:
* GraphDB must be ready: running but empty except for ontology 
  vrd_world_v1_2_disjoint.owl 

Inputs:
* master 3D VR type tensor
* preds.json file 

Outputs:
* master 3D VR type tensor
* preds.json file
* log file (containing computed stats) 
'''

#%%

import torch 
import os
import json
import time 
import sys  

import nesy4vrd_utils4 as vrdu4
import vrd_utils17 as vrdu17


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


#%% specify the name of the master 3D VR type tensor file 

#master_vr_type_tensor_filename = 'master-vr-type-tensor-kg-filtering.json'
master_vr_type_tensor_filename = 'master-vr-type-tensor-nnkgs1.json'


#%% specify the directory where the master 3D VR type tensor sits 

central_results_dir = os.path.join(root_dir, 'research', 'results', 'nnkgs0')

central_results_dir = os.path.expanduser(central_results_dir)

master_vr_type_tensor_filepath = os.path.join(central_results_dir, master_vr_type_tensor_filename)


#%% choose whether or not to redirect stdout to a log file

redirect_stdout = True


#%% build the log file name 

if redirect_stdout:
    logfile_name = pvr_filename.removesuffix('.json')
    logfile_name = logfile_name + '_vr_type_classification_log.txt'
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

print('*** analyse predicted vr semantic validity 1 ***')
print()
print(f'script name: {scriptName}')
print()
print(f'predicted VRs file: {pvr_filename}')
print() 
print(f"predicted VRs file directory: {vr_predictions_dir}")
print()
print(f'master 3D VR type tensor file: {master_vr_type_tensor_filename}')
print()
print(f'master 3D VR type tensor directory: {central_results_dir}')
print()
print(f'log file name: {logfile_name}')
print() 


#%% function to load a VR predictions file

def load_predicted_vrs_per_image(filepath):
    
    with open(filepath, 'r') as fp:
        vr_predictions_per_image = json.load(fp)

    return vr_predictions_per_image 


#%% function to save VR predictions to a file

def save_predictions_to_file(filepath, results):
    
    with open(filepath, 'w') as fp:
        json.dump(results, fp)

    return None


#%% function to load the master 3D VR type tensor

def load_master_vr_type_3d_tensor(filepath):
    
    with open(filepath, 'r') as fp:
        master_vr_type_tensor = json.load(fp)

    # convert the JSON serialisable rendering of the master 3D VR type
    # tensor to an actual tensor
    master_vr_type_tensor = torch.tensor(master_vr_type_tensor)

    return master_vr_type_tensor 


#%% function to save the master 3D VR type tensor

def save_master_vr_type_3d_tensor(filepath, master_vr_type_tensor):
    
    # convert the real tensor to a JSON serialisable rendering
    master_vr_type_tensor = master_vr_type_tensor.numpy().tolist()
    
    with open(filepath, 'w') as fp:
        json.dump(master_vr_type_tensor, fp)

    return None


#%% load the predicted VRs file 

predicted_vrs_per_image = load_predicted_vrs_per_image(pvr_filepath)

print() 
print(f'predicted VR file loaded: {pvr_filepath}')


#%% load the master 3D VR type tensor 

# On the very first run of this script there is nothing to load.
# Instead, we initialise the master 3D VR type tensor to begin its life 
#master_vr_type_tensor = torch.zeros((nr_predicates, nr_object_classes, nr_object_classes))
#print() 
#print('master 3D VR type tensor initialised')


# On all subsequent runs of this script, load the master 3D VR type tensor
master_vr_type_tensor = load_master_vr_type_3d_tensor(master_vr_type_tensor_filepath)
print() 
print(f'master 3D VR type tensor loaded: {master_vr_type_tensor_filepath}')

print(f'master VR type tensor shape: {master_vr_type_tensor.shape}')
n_cells_master_vr_type_tensor = master_vr_type_tensor.shape[0] * master_vr_type_tensor.shape[1] * master_vr_type_tensor.shape[2]
print(f'master VR type tensor size (in cells): {n_cells_master_vr_type_tensor}')

#
# capture starting stats for the master 3D VR type tensor
#

n_nonzero_master_start = torch.count_nonzero(master_vr_type_tensor)

mask = master_vr_type_tensor == 1.0
n_vr_types_valid_start = torch.sum(mask)

mask = master_vr_type_tensor == 2.0
n_vr_types_invalid_start = torch.sum(mask)


#%% global variables for KG interaction 
  
#graphdb_address_port = 'http://192.168.0.151:7200'
graphdb_address_port = 'http://localhost:7200'
repository_name = 'test'
base_graphdb_url = graphdb_address_port + '/repositories/' + repository_name

print()
print(f'GraphDB url: {base_graphdb_url}')

# specify directory for storing curl tool output files
curl_dir = os.path.join('~', 'research', 'curl')
curl_dir = os.path.expanduser(curl_dir)

print() 
print(f'curl tool output dir: {curl_dir}')


#%% 

start_seq_num = 1000000
entity_seq_num = {'entity_seq_num' : start_seq_num}

print()
print(f"starting sequence number for KG entities: {entity_seq_num['entity_seq_num']}")


#%% final config 

verbose = True

# recall: the test set has 962 images; so a value of 1000 will cover all of them
max_images_to_process = 1000

img_count = 0
pvr_count = 0

n_kg_calls_required = 0
n_kg_calls_not_required = 0

n_kg_calls_actually_made = 0

n_pvrs_with_valid_vr_types = 0
n_pvrs_with_invalid_vr_types = 0

n_master_vr_type_tensor_updates = 0

# this must be set to a negative number!
img_idx_of_first_image_needing_kg_calls = -1

print()
print(f'number of images to process: {max_images_to_process}')


#%% 

start_time = time.time()

for img_idx, batch in enumerate(predicted_vrs_per_image.items()):

    if img_idx+1 > max_images_to_process:
        print() 
        print('processing stopping: max images to process exceeded')
        print() 
        break
      
    img_count += 1
    
    imname, predicted_vr_dict = batch
    
    if verbose:
        print() 
        print(f'processing image: idx {img_idx}, {imname}')
        print()
        if redirect_stdout:
            sys.stdout.flush()
    
    predicted_vrs = predicted_vr_dict['predicted_vrs']

    for pvr in predicted_vrs:
        
        # get the VR's property and object class labels (integer indices)   
        prop_idx = pvr['predicate']
        obj1_cls_idx = pvr['subject']['category']
        obj2_cls_idx = pvr['object']['category']
        
        if verbose: 
            vr_friendly = '(' + ontoClassNames[obj1_cls_idx] + ', ' + ontoPropNames[prop_idx] + ', ' + ontoClassNames[obj2_cls_idx] + ')'
            print()
            print(f'processing vr: {vr_friendly}')
            print(f'prop_idx {prop_idx}, obj1_cls_idx {obj1_cls_idx}, obj2_cls_idx {obj2_cls_idx}')
        
        # get the VR type class of the current predicted VR from the 
        # master 3D VR type tensor
        vr_type_class = master_vr_type_tensor[prop_idx, obj1_cls_idx, obj2_cls_idx]
        
        # convert from tensor to float (so it's JSON serialisable)
        vr_type_class = vr_type_class.item()
        
        # check if the VR type has already been classified
        if vr_type_class == 0.0:
            vr_type_classification_required = True
            n_kg_calls_required += 1
        else:
            vr_type_classification_required = False 
            n_kg_calls_not_required += 1
        
        if not vr_type_class in[0.0, 1.0, 2.0]:
            raise ValueError(f'PROBLEM: vr_type_class not recognised: {vr_type_class}')
        
        if vr_type_classification_required:
            
            # call the KG
            vr_type_class = vrdu17.call_kg_to_classify_vr_type(prop_idx, 
                                                               obj1_cls_idx, 
                                                               obj2_cls_idx, 
                                                               entity_seq_num, 
                                                               ontoClassNames, 
                                                               ontoPropNames)
            
            n_kg_calls_actually_made += 1
            
            # record the first image to require a KG call 
            if img_idx_of_first_image_needing_kg_calls < 0:
                img_idx_of_first_image_needing_kg_calls = img_idx 
            
            if vr_type_class == 1.0:  # valid VR type, but a valid response
                
                pass
            
            elif vr_type_class == 2.0:  # invalid VR type, but a valid response
                
                pass

            elif vr_type_class == 0.0:
                
                print('PROBLEM: KG interaction failed unexpectedly')
                print('VR type for cell unknown')
                raise ValueError('PROBLEM: KG failed to classify cell VR type')
                
            else:
                    
                raise ValueError('PROBLEM: cell_vr_type_class not recognised')
            
            # store the class (semantically valid or invalid) of the VR type 
            # of the predicted VR; store it in the master 3D VR type tensor
            master_vr_type_tensor[prop_idx, obj1_cls_idx, obj2_cls_idx] = vr_type_class 
            n_master_vr_type_tensor_updates += 1
                      
            if verbose:
                print(f'KG classifed VR type as: {vr_type_class}')
        
         
        # extend the predicted VR with its VR type class (in JSON serialisable format)
        pvr['vr_type_class'] = vr_type_class
        
        pvr_count += 1
        
        if vr_type_class == 1.0:
            n_pvrs_with_valid_vr_types += 1
        else:
            n_pvrs_with_invalid_vr_types += 1

        if redirect_stdout:
            sys.stdout.flush()        


#%% store the updates that have been made 

save_predictions_to_file(pvr_filepath, predicted_vrs_per_image)
print()
print(f'Extended predicted VRs file saved: {pvr_filename}')

save_master_vr_type_3d_tensor(master_vr_type_tensor_filepath, master_vr_type_tensor)
print()
print(f'Updated master 3D VR type tensor saved: {master_vr_type_tensor_filename}')


#%% capture ending stats for the master 3D VR type tensor

n_nonzero_master_end = torch.count_nonzero(master_vr_type_tensor)
n_nonzero_master_new = n_nonzero_master_end - n_nonzero_master_start

mask = master_vr_type_tensor == 1.0
n_vr_types_valid_end = torch.sum(mask)
n_vr_types_valid_new = n_vr_types_valid_end - n_vr_types_valid_start

mask = master_vr_type_tensor == 2.0
n_vr_types_invalid_end = torch.sum(mask)
n_vr_types_invalid_new = n_vr_types_invalid_end - n_vr_types_invalid_start


#%% report summary info

end_time = time.time()

processing_time_min = (end_time - start_time) / 60

print()
print('--- summary statistics for script run ---')
print()
print(f'number of images processed: {img_count}')
print()
print(f'img_idx of first image to need KG calls: {img_idx_of_first_image_needing_kg_calls}')
print()
print(f'number of predicted VRs processed: {pvr_count}')
print()
print(f'number of predicted VRs with valid VR types: {n_pvrs_with_valid_vr_types}')
print()
print(f'number of predicted VRs with invalid VR types: {n_pvrs_with_invalid_vr_types}')
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - -')
print()
print(f'number of KG calls required: {n_kg_calls_required}')
print()
print(f'number of KG calls not required: {n_kg_calls_not_required}')
print()
print(f'number of KG calls actually made: {n_kg_calls_actually_made}')
print() 
print(f'number of master VR type tensor updates: {n_master_vr_type_tensor_updates}')
print()
print(f'number of nonzero cells in master VR type tensor at start: {n_nonzero_master_start}')
print(f'number of nonzero cells in master VR type tensor at end: {n_nonzero_master_end}')
print(f'number of nonzero cells in master VR type tensor new: {n_nonzero_master_new}')
print()
print(f'number of vr types valid at start: {n_vr_types_valid_start}')
print(f'number of vr types valid at end: {n_vr_types_valid_end}')
print(f'number of vr types valid new: {n_vr_types_valid_new}')
print()
print(f'number of vr types invalid at start: {n_vr_types_invalid_start}')
print(f'number of vr types invalid at end: {n_vr_types_invalid_end}')
print(f'number of vr types invalid new: {n_vr_types_invalid_new}')

print()
print(f'processing time: {processing_time_min:.2f} min')
print()
print('------------------------------------------')


#%% health checks

if n_kg_calls_actually_made != n_kg_calls_required:
    print()
    print('PROBLEM: n_kg_calls_actually_made != n_kg_calls_required')
        
if n_kg_calls_required + n_kg_calls_not_required != pvr_count:
    print()
    print('PROBLEM: tracking of kg calls vs predicted VRs is wonky')

if n_nonzero_master_new != n_kg_calls_actually_made:
    print()
    print('PROBLEM: n_nonzero_master_new != n_kg_calls_actually_made')
    

#%%

if redirect_stdout:
    # flush stdout buffer
    sys.stdout.flush()
    # close redirected output file
    sys.stdout.close()
    # restore sys.stdout to original file handler
    sys.stdout = stdout_file_saved

print('Processing completed')


