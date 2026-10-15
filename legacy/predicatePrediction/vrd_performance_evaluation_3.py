#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script drives the evaluation of the predictive performance of our
hybrid (NN+KG) neurosymbolic systems at the task of visual relationship
detection.
'''

#%%

import os
import json
import glob
import time
import sys
from datetime import date
import torch 

import nesy4vrd_utils4 as vrdu4
import vrd_utils12 as vrdu12
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
    topN = 50

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

#
# get the training region cell configuration (dimension levels)
#

cfg = vrdu14.get_training_region_cell_config(experiment_family, trc_id)

tr_d_onto = cfg['D_onto_level']

#
# get the prediction region cell configuration (dimension levels)
#

cfg = vrdu14.get_prediction_region_cell_config(experiment_family, prc_id)

# assign the dimension levels to more familiar variable names
if cfg['D_predKG_level'] == 1: 
    kg_filtering = False                          
else:
    kg_filtering = True
       
# repackage the prediction region cell configuration
prediction_region_config = {}
prediction_region_config['pred_region_id'] = prc_id
prediction_region_config['kg_filtering'] = kg_filtering


#
# get the results region cell configuration (dimension levels)
#

cfg = vrdu14.get_results_region_cell_config(experiment_family, rrc_id)

# the composition of the test set VR annotations to use as targets
# for performance evaluation purposes
rr_d_perfTarget = cfg['D_perfTarget_level']
        

#%% build the name of the work directory (where the predicted VRs are)

root_dir = '~'   # local hard drive
#root_dir = os.path.join(os.sep, 'Volumes', 'My Passport for Mac') # external hard drive

if platform == 'hyperion':
    vr_predictions_dir = os.path.join('~', 'sharedscratch', 'research', workdir)
else:
    vr_predictions_dir = os.path.join(root_dir, 'research', workdir)
    
    # nb: the following setting is for reprocessing archived cells, if required
    #vr_predictions_dir = os.path.join(root_dir, 'research', 'results-archive', 'nnkgs0')
    #vr_predictions_dir = os.path.join(vr_predictions_dir, 'setID-01', 'trial-02')
    #vr_predictions_dir = os.path.join(vr_predictions_dir, 'cell-trc5500-prc018-rrc01')
    #vr_predictions_dir = os.path.join(vr_predictions_dir, workdir)

scriptName = os.path.basename(__file__)
print(f'script name: {scriptName}')

vr_predictions_dir = os.path.expanduser(vr_predictions_dir)
print(f'work dir   : {vr_predictions_dir}')


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


#%% choose whether or not to redirect stdout to a file

# redirecting stdout allows one to retain a results log text file
# for documentation

redirect_stdout = True


#%% set the name of the log file to which stdout will be redirected

if redirect_stdout:
    results_log_filename = 'vrd_ppnn_' + trc_id + '_' + prc_id + '_' + rrc_id
    results_log_filename = results_log_filename + '_results_topN_' 
    results_log_filename = results_log_filename + str(topN).zfill(3) + '_log.txt'
else:
    results_log_filename = ''


#%% redirect stdout

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(vr_predictions_dir, results_log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%% declare (record) experiment meta data

print()
print(f'# experiment family: {experiment_family}')
print(f'# training region cell id: {trc_id}')
print(f'# prediction region cell id: {prc_id}')
print(f'# results region cell id: {rrc_id}')
print(f'# topN for recall@N: {topN}')
print()
print(f'Date: {date.today()}')


#%% record key info in log file

# the platform on which we're running
print(f'Platform: {platform}')

# the name of this Python script
#scriptName = sys.argv[0]
scriptName = os.path.basename(__file__)
print(f'Script: {scriptName}')

# the name of the conda environment in which we're running
print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")


#%% function to load a VR predictions file

def load_vr_predictions(filepath):
    
    with open(filepath, 'r') as fp:
        vr_predictions = json.load(fp)

    return vr_predictions    


#%% function to save measures of predictive performance to a file

def save_extended_vr_predictions_to_file(filepath, vr_predictions):
    
    with open(filepath, 'w') as fp:
        json.dump(vr_predictions, fp)

    return None


#%% function to save measures of predictive performance to a file

def save_performance_to_file(filepath, results):
    
    #print(results)
    
    with open(filepath, 'w') as fp:
        json.dump(results, fp)

    return None


#%% function to load GT annotations file

def load_gt_annotations(filepath):
    
    with open(filepath, 'r') as fp:
        gt_annotations = json.load(fp)

    return gt_annotations   


#%% get the paths of the VR prediction files to be processed

# set the filename pattern
vr_predictions_pattern = 'vrd_ppnn_' + trc_id + '_ckpt_*_' 
vr_predictions_pattern = vr_predictions_pattern + prc_id + '_preds.json'
print(f"VR predictions filename pattern: {vr_predictions_pattern}")

path = os.path.join(vr_predictions_dir, vr_predictions_pattern)

# gather the names of all of the files to be processed
vr_predictions_paths = glob.glob(path)
vr_predictions_paths = sorted(vr_predictions_paths)
print(f'Number of VR predictions files to be processed: {len(vr_predictions_paths)}')


#%% configure and load the test set annotated VRs to be used as targets

# The levels of 1) dimension D_targetPerf of the experiment space results 
# region and 2) dimension D_onto of the experiment space training region
# together determine the particular file of VR annotations that we use
# here as targets for predictive performance evaluation purposes.

if rr_d_perfTarget == 1:
    # use the initial (sparse & arbitrary) test set VR annotations as the
    # targets for predictive performance evaluation purposes
    anno_dir = os.path.join('..', 'data', 'annotations')
    filename = 'nesy4vrd_annotations_test.json'
    
    # Note: in this setting, dimension D_onto, represented by tr_d_onto,
    # can take any level and the experiment configuration may still be valid.
    # In other words, it's not necessarily the case that we should enforce
    # that tr_d_onto==1. This is because dimension D_onto might specify
    # that an ontology is used with the experiment, and this use could be
    # only in relation to the training of the PPNN model (eg dimension
    # D_target). For example, it would be valid to have tr_d_target==2,
    # tr_d_onto==2 while rr_d_targetperf==1.
    
elif rr_d_perfTarget == 2:
    # use KG-augmented test set VR annotations as the targets for predictive 
    # performance evaluation purposes
    #
    # the file to be loaded depends on the version of the VRD-World ontology 
    # that was used to drive the KG-augmentation (KG materialisation), and
    # that's what tr_d_onto, the level of dimension D_onto, gives us
    anno_dir = os.path.join('..', 'data', 'annotations_augmented')
    if tr_d_onto == 1:
        raise ValueError(f'level {tr_d_onto} of dimension D_onto not valid here')
    elif tr_d_onto == 2: # symmetry, transitivity, and inverses
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1b.json'
    elif tr_d_onto == 3: # full version
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1.json'
    elif tr_d_onto == 4: # rdfs:subPropertyOf and owl:equivalentProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1c.json'
    elif tr_d_onto == 5: # owl:SymmetricProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1g.json'
    elif tr_d_onto == 6: # owl:TransitiveProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1h.json'
    elif tr_d_onto == 7: # owl:inverseOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1i.json'
    elif tr_d_onto == 8: # rdfs:subPropertyOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1j.json'
    elif tr_d_onto == 9: # owl:equivalentProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1k.json'
    elif tr_d_onto == 10: # owl:SymmetricProperty and owl:inverseOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1e.json'
    elif tr_d_onto == 11: # owl:TransitiveProperty, rdfs:subPropertyOf and owl:equivalentProperty    
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1n.json'
    elif tr_d_onto == 12:  # sym, trans, subProp, equivProp
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1p.json'
    elif tr_d_onto == 13:  # trans, inverseOf, subProp, equivProp
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1r.json'      
    elif tr_d_onto == 14:  # trans, inverseOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1f.json' 
    elif tr_d_onto == 15:  # sym, trans
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1d.json'
    elif tr_d_onto == 16:  # sym, equivProp
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1s.json'
    else:
        raise ValueError(f'level {tr_d_onto} of dimension D_onto not recognised')

elif rr_d_perfTarget == 3:
    # baseline test set VR annos filtered wrt filter scheme 01 
    anno_dir = os.path.join('..', 'data', 'annotations')
    filename = 'nesy4vrd_annotations_test_filtered_per_scheme_01.json'

elif rr_d_perfTarget == 4: 
    # KG-augmented test set VR annos filtered wrt filter scheme 01 
    
    anno_dir = os.path.join('..', 'data', 'annotations_augmented')
    if tr_d_onto == 1:
        raise ValueError(f'level {tr_d_onto} of dimension D_onto not valid here')
    elif tr_d_onto == 2: # symmetry, transitivity, and inverses
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1b_filtered_per_scheme_01.json'
    elif tr_d_onto == 3: # full version
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1_filtered_per_scheme_01.json'
    elif tr_d_onto == 4: # rdfs:subPropertyOf and owl:equivalentProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1c_filtered_per_scheme_01.json'
    elif tr_d_onto == 5: # owl:SymmetricProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1g_filtered_per_scheme_01.json'
    elif tr_d_onto == 6: # owl:TransitiveProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1h_filtered_per_scheme_01.json'
    elif tr_d_onto == 7: # owl:inverseOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1i_filtered_per_scheme_01.json'
    elif tr_d_onto == 8: # rdfs:subPropertyOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1j_filtered_per_scheme_01.json'
    elif tr_d_onto == 9: # owl:equivalentProperty
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1k_filtered_per_scheme_01.json'
    elif tr_d_onto == 10: # owl:SymmetricProperty and owl:inverseOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1e_filtered_per_scheme_01.json'
    elif tr_d_onto == 11: # owl:TransitiveProperty, rdfs:subPropertyOf and owl:equivalentProperty    
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1n_filtered_per_scheme_01.json'
    elif tr_d_onto == 12:  # sym, trans, subProp, equivProp
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1p_filtered_per_scheme_01.json'
    elif tr_d_onto == 13:  # trans, inverseOf, subProp, equivProp
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1r_filtered_per_scheme_01.json'      
    elif tr_d_onto == 14:  # trans, inverseOf
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1f_filtered_per_scheme_01.json' 
    elif tr_d_onto == 15:  # sym, trans
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1d_filtered_per_scheme_01.json'
    elif tr_d_onto == 16:  # sym, equivProp
        filename = 'nesy4vrd_annotations_test_augmented_per_onto_v1_1s_filtered_per_scheme_01.json'
    else:
        raise ValueError(f'level {tr_d_onto} of dimension D_onto not recognised')

else:
    raise ValueError(f'level {rr_d_perfTarget} of dimension D_targetPerf not recognised')
    

anno_filepath = os.path.join(anno_dir, filename)

print(f"Annotated VR targets: {anno_filepath}")

# load the ground-truth visual relationships (targets); we only need to do
# this once, upfront, because we can reuse them for evaluating the
# predictive performance of multiple different sets of predicted VRs  
gt_vr_dict = load_gt_annotations(anno_filepath)

gt_vr_nr_entries = len(gt_vr_dict)

print(f"The gt_vr_dict has entries for {gt_vr_nr_entries} images")


#%% set flag for conversion of bbox format

# All predicted VRs use the FRCNN convention for representing bboxes.
# Under normal circumstances, the annotated VRs used as targets for
# calculating performance will always be VRs using the VRD convention for
# representing bboxes. Thus, under normal circumstances, we want to
# the bboxes of the target (ground-truth) VRs to be converted to FRCNN
# format. Otherwise, few predicted VRs will ever match with target VRs.
convert_gt_bbox_format = True

print(f'Converting GT bboxes to FRCNN format: {convert_gt_bbox_format}')


#%% set mode: production or testing

# set the number of images to process per VR predictions file: just a few (for 
# test purposes) or all of them (for a production run)
# N = 0 is for production --- it means process all images
# N > 0 is for testing --- it means process the first N images only
n_images_to_process = 0

if n_images_to_process == 0:
    print('Number of images to process per predictions file: all')
else:
    print(f'Number of images to process per predictions file: {n_images_to_process}')


#%% for KG filtering only (which may or may not be active)

start_seq_num = 1000000
entity_seq_num = {'entity_seq_num' : start_seq_num}

print()
print(f"starting sequence number for KG entities: {entity_seq_num['entity_seq_num']}")


save_master_vr_type_tensor_required = False 


#%% if KG filtering is active, load the master 3D VR type tensor

if kg_filtering:
    
    print()
    print('KG filtering is active:')
    print('* semantically invalid predicted VRs will be filtered out')
    print('* KG interaction may be required')
    print('* GraphDB may be called') 
    print('* GraphDB expected to be available and ready (empty except for ontology)')
    print('* master 3D VR type tensor needed and may be updated')
    
    master_vr_type_tensor_filename = 'master-vr-type-tensor-kg-filtering.json'
    
    # NOTE: for now, we store the master 3D vr type tensor on local disk, not external drive
    central_results_dir = os.path.join('~', 'research', 'results', 'nnkgs0')
    central_results_dir = os.path.expanduser(central_results_dir)
    master_vr_type_tensor_filepath = os.path.join(central_results_dir, master_vr_type_tensor_filename)
    
    with open(master_vr_type_tensor_filepath, 'r') as fp:
        master_vr_type_tensor = json.load(fp)
    
    # convert the JSON serialisable rendering of the master 3D VR type
    # tensor to an actual tensor
    master_vr_type_tensor = torch.tensor(master_vr_type_tensor)
    
    print()
    print('master 3D VR type tensor loaded:')
    print(master_vr_type_tensor_filepath)
    print() 

else: 

    master_vr_type_tensor = None 


n_kg_calls_total = 0


#%% main processing loop

#cnt = 0

start_time_total = time.time()

# iterate over the VR predictions files
for item in vr_predictions_paths:

    #cnt += 1
    #if cnt > 3:
    #    break

    # capture start time
    start_time_model = time.time()
    
    # get the filename of the current VR predictions file being processed
    prefix = vr_predictions_dir + os.sep
    vr_pred_filename = item.removeprefix(prefix)
    print(f"\nProcessing VR file: {vr_pred_filename}")   

    # load the next set of predicted VRs
    predicted_vrs_per_image = load_vr_predictions(item)
    
    # evaluate performance for the current set of predicted VRs
    performance_results = vrdu12.evaluate_performance(predicted_vrs_per_image,
                                                      gt_vr_dict,
                                                      topN,
                                                      convert_gt_bbox_format,
                                                      n_images_to_process,
                                                      prediction_region_config,
                                                      rrc_id,
                                                      entity_seq_num,
                                                      ontoClassNames,
                                                      ontoPropNames,
                                                      master_vr_type_tensor)
    
    # nb: the dictionary predicted_vrs_per_image is updated in-place 
    # by function evaluate_performance(); extended info is stored for
    # each image and its predicted VRs; we want to preserve this extended
    # info by saving the predicted_vrs_per_image dictionary back to disk
    # under the very same file name from which it was loaded
    save_extended_vr_predictions_to_file(item, predicted_vrs_per_image)
    
    # if KG filtering was active, check if any KG calls were made
    if kg_filtering:
        n_kg_calls = performance_results['n_kg_calls']
        if n_kg_calls > 0:
            save_master_vr_type_tensor_required = True
        n_kg_calls_total += n_kg_calls
    
    # save the predictive performance results to a file
    path_filename = item.removesuffix('preds.json')
    path_filename = path_filename + rrc_id + '_results_topN_' 
    path_filename = path_filename + str(topN).zfill(3) + '.json'
    results_path_filename = path_filename
    save_performance_to_file(results_path_filename, performance_results)
    prefix = vr_predictions_dir + os.sep
    results_filename = results_path_filename.removeprefix(prefix)
    print() 
    print(f'Performance scores: {results_filename}')

    # measure elapsed time
    end_time_model = time.time()
    model_time = (end_time_model - start_time_model) / 60
    print(f"Processing time   : {model_time:.2f} minutes")
    
    if redirect_stdout:
        sys.stdout.flush()

# if we're doing KG filtering and the master 3D VR type tensor has been
# updated (due to KG interaction and the classification of new VR type not
# already encountered amongst other predicted VRs), then save the updated
# master 3D VR type tensor 
if kg_filtering:
    print()
    print('KG filtering was active')  
    if save_master_vr_type_tensor_required:
        # convert the real tensor to a JSON serialisable rendering
        master_vr_type_tensor = master_vr_type_tensor.numpy().tolist()
        with open(master_vr_type_tensor_filepath, 'w') as fp:
            json.dump(master_vr_type_tensor, fp)
        print(f'number of new KG classified VR types: {n_kg_calls_total}')
        print('master 3D VR type tensor updated and saved:')
        print(master_vr_type_tensor_filepath) 
    else:
        print('number of new KG classified VR types: 0')
        print('the master 3D VR type tensor was not updated')
        print('so no need to save it back to disk')

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


