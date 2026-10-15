#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script prepares input data for Predicate Prediction neural networks (PPNNs).
It consumes a source data file and produces a file of PPNN input data derived
from that source data.

There are two categories of source data:
1) Outputs from an object detection NN (ODNN); ie the set of objects detected (by
   an ODNN) in each image of an image dataset, where detected objects are 
   represented by their a) predicted bboxes and b) predicted object class labels.
2) Ground-truth VR annotations (whether initial or KG-augmented, per NN+KG_S0
   and NN+KG_S2)

This script processes both categories of source data. 

Category (1) source data:
* can be loaded and processed straight away, without any preprocessing
* supports the 'relationship prediction' regime of experiments, where the
  inaccuracies of object detection are allowed to effect the VR prediction results

Category (2) source data:
* must be loaded and preprocessed before it can be processed
* the preprocessing involves extracting the annotated bboxes and object classes from
  the annotated VRs
* supports the 'predicate prediction' regime of experiments, where the
  inaccuracies of object detection are eliminated by using the ground-truth annotated
  objects as proxies for predicted objects detected by an ODNN; another way to see this
  regime of experiments is to see it as supposing we had a perfect ODNN that was able
  to detect exactly what had been annotated, in every respect; this has a dramatic
  impact on the VR prediction results

With respect to the training set, for both categories of source data we
ignore the distinction between (sub)training and validation sets. That is,
the script prepares PPNN input data for the full training set, which includes
entries for the images allocated to the NeSy4VRD (sub)training and validation 
sets.  It's the responsibility of PPNN training scripts to ensure they
process only the entries pertaining to the (sub)training and validation sets,
as appropriate.

The script computes and stores the maximum possible number of bbox geometric
features with respect to each ordered pair of objects, for each image.  The
particular subset of this maximal set of features that are actually used at
PPNN training time is determined by the configuration of the PPNN training 
script. This is a key aspect of experiment design.

THIS SCRIPT IS DESIGNED TO BE RUN INTERACTIVELY, CELL BY CELL, IN AN IDE.

See document '/predicate prediction/Predicate prediction notes.md' 
for the original thinking behind this script.

NOTE: there's no point creating PPNN input data from category (2) source
data such as VR annotations files that have been augmented by OWL 
reasoning.  These will give identical results to PPNN input data created
from the NeSy4VRD baseline VR annotations files. This is because the VR
augmentation done by OWL reasoning (using a version of VRD-World) does NOT
change or increase the objects per image.  The augmentation affects only
relations between existing objects.  But PPNN input data is derived purely
from objects (detected by an ODNN, or annotated by an annotator). Since the
objects per image won't change with VR augmentation, the PPNN input data
derived from augmented annotations won't change.  So there's no point
creating such PPNN input data files in the first place.
'''

#%%

import os
import json

import vrd_utils8 as vrdu8


#%%

filepath_root = '..'


#%% discussion of source data categories

# There are several different potential files of source data.

# Category (1) - JSON files containing the objects detected in VRD images
# by a trained FRCNN object detection neural network (ODNN). These source 
# data are used to derive PPNN input data that is used for the 
# 'relationship detection' regime of experiments, where object detection
# plays an active role.
#
# File options:
# - vrd_frcnn_v2_1_1_checkpoint_250_ihc_1_testset_detected_objects.json
# - vrd_frcnn_v2_1_1_checkpoint_250_ihc_2_testset_detected_objects.json
# - vrd_frcnn_v2_1_1_checkpoint_250_ihc_3_testset_detected_objects.json
# - vrd_frcnn_v2_1_1_checkpoint_250_ihc_4_testset_detected_objects.json

# Category (2) - JSON files containing ground-truth annotated visual 
# relationships (VRs). These data are the NeSy4VRD VR annotations files.
# These source data are used to derive PPNN input data that is used for the
# 'predicate prediction' regime of experiments, where object detection
# does not play a role.
#
# File options:
# - nesy4vrd_annotations_train.json              (initial annotated VRs)
# - nesy4vrd_annotations_test.json               (initial annotated VRs)
# - nesy4vrd_annotations_train_augmented_per_onto_v1_1.json  
# - nesy4vrd_annotations_test_augmented_per_onto_v1_1.json
# - nesy4vrd_annotations_train_augmented_per_onto_v1_1b.json
# - nesy4vrd_annotations_test_augmented_per_onto_v1_1b.json


#%% specify the category of the source data to be processed

source_data_category = 2


#%% D1: category (1) source data - objects detected by an ODNN

# Category (1) source data are objects-per-image that have been detected
# by a trained FRCNN object detector. These data are already in the correct
# format --- the D1 format: a dictionary with objects per image. 
#
# In the 'relationship detection' regime of experiments, we use the outputs
# of an ODNN to derive PPNN input data.  This exposes our VR detection
# system to the vagaries and noise introduced by object detection.
#
# Since category (1) source data is already in the correct D1 format, it
# does not require any preprocessing. We can simply load the data.

if source_data_category == 1:

    # set the directory where the ODNN output source data are located
    odnn_output_dir = os.path.join(filepath_root, 'data', 'odnn_output_data')
    
    # set the ODNN output file to be processed
    odnn_output_filename = 'vrd_frcnn_v2_1_1_checkpoint_250_ihc_4_trainset_detected_objects.json'
    #odnn_output_filename = 'vrd_frcnn_v2_1_1_checkpoint_250_ihc_4_testset_detected_objects.json'
    
    # assemble the full filepath
    odnn_output_filepath = os.path.join(odnn_output_dir, odnn_output_filename)
    
    # load the ODNN output data
    with open(odnn_output_filepath, 'r') as fp:
        d1_objects_per_image = json.load(fp)
    
    print(f'Entries in D1 dictionary: {len(d1_objects_per_image)}')


#%% D2: category (2) source data - VR annotations

# Category (2) source data are VR annotations.  We need to preprocess these 
# to extract the set of unique objects (bboxes and class labels) that have
# been annotated for each image.  
#
# In the 'predicate prediction' regime of experiments, we pretend that the 
# ground-truth annotated objects within the VR annotations are the outputs
# of an imaginary ODNN that detects precisely what had been annotated, as if 
# we had an imaginary perfect object detector.

if source_data_category == 2:

    # set the path to the image directory
    #image_dir = os.path.join(filepath_root, 'data', 'train_images')
    image_dir = os.path.join(filepath_root, 'data', 'test_images')
    
    # set the path to the NeSy4VRD annotations directory
    anno_dir = os.path.join(filepath_root, 'data', 'annotations')
    
    # set the name of the NeSy4VRD annotations file to be processed
    #vrd_anno_file = os.path.join(anno_dir, 'nesy4vrd_annotations_train.json')
    vrd_anno_file = os.path.join(anno_dir, 'nesy4vrd_annotations_test.json')
 
    # get the VR annotations
    with open(vrd_anno_file, 'r') as fp:
        vrd_img_annos = json.load(fp)
    
    # get the names of the images for the full train or test set
    vrd_img_names = list(vrd_img_annos.keys())
    
    # From the annotated VRs for each image, extract the set of unique objects
    # referred to within those VRs.
    # (nb: we need the image_dir to get the image dimensions HxW, which are
    #  used downstream to calculate certain bbox geometric features)
    # (nb: this also converts the bbox format from VRD format to FRCNN format)
    results = vrdu8.avrs_2_objects_per_image(vrd_img_names, 
                                             vrd_img_annos,
                                             image_dir,
                                             include_image_dims=True)
    
    d1_objects_per_image = results
    
    print(f'Entries in D1 dictionary: {len(d1_objects_per_image)}')


#%% test the D1 data, regardless of source data category

cnt = 0
target = 2
    
for k, v in d1_objects_per_image.items():
    cnt += 1
    if cnt < target:
        pass
    elif cnt == target:
        print(k)
        print()
        print(v)
    else:
        break


#%% get the number of object classes

# set the VR annotations directory
anno_dir = os.path.join(filepath_root, 'data', 'annotations')

# get the NeSy4VRD object class names
path = os.path.join(anno_dir, 'nesy4vrd_objects.json')
with open(path, 'r') as fp:
    nesy4vrd_object_classes = json.load(fp)
   
n_object_classes = len(nesy4vrd_object_classes)


#%% D2: convert D1 data to D2 data

# convert the 'objects' per image into 'ordered pairs of objects' per image
results = vrdu8.create_object_ordered_pairs(d1_objects_per_image,
                                            include_image_dims=True)

d2_object_ordered_pairs_per_image = results

print(f'Entries in D2 dictionary: {len(d2_object_ordered_pairs_per_image)}')


#%% test the D2 data

cnt = 0
target = 1

for k, v in d2_object_ordered_pairs_per_image.items():
    cnt += 1
    if cnt < target:
        pass
    elif cnt == target:
        print(k)
        print()
        print(v)
    else:
        break


#%% D3: extend the features associated with the object ordered pairs

results = vrdu8.extend_image_and_object_ord_pair_features(d2_object_ordered_pairs_per_image,
                                                          n_object_classes)

d3_object_ordered_pairs_per_image_extended = results

print(f'Entries in D3 dictionary: {len(d3_object_ordered_pairs_per_image_extended)}')


#%% test the D3 data

cnt = 0
target = 1

for k, v in d3_object_ordered_pairs_per_image_extended.items():
    cnt += 1
    if cnt < target:
        pass
    elif cnt == target:
        print(k)
        print()
        print(v)
    else:
        break


#%% test conversion of D3 data into a tensor of feature vectors

# The functionality tested here is used downstream during PPNN model 
# training, where data (features) for ordered pairs of objects from a D3 
# dictionary are converted into feature vector tensors for feeding as input
# to PPNN models. This is a convenient place to check that the conversion
# process works as expected.

cnt = 0
target = 5

for imname, imdict in d3_object_ordered_pairs_per_image_extended.items():
    cnt += 1
    if cnt == target:
        break

obj_ord_pairs = imdict['obj_ord_pairs']


print(f'imname: {imname}')
print(f'nr of ordered pairs of objects: {len(obj_ord_pairs)}')
print()

featureset = 1

aten = vrdu8.ppnn_image_dict_2_tensor(imdict, n_object_classes, featureset)

print(f'aten shape: {aten.shape}')


#%% inspect the contents of the tensor of feature vectors

print(aten[0:3,0:5])

print()

print(aten[5:10,226:231])


#%% save PPNN input data to JSON file

## ***** WARNING *****
##
## ppnn input data ppnn_input_pp_train_1.json
##   - consumes over 335 MB !!!
##   - take caution where you save such a large file !!!
##   - on the training set data, this cell takes about 1 minute to complete !!!
##
## ppnn input data ppnn_input_pp_test_v1.json
##   - consumes 76.7 MB !!!

# 'rd' = 'relationship detection'  (category 1 data)
# 'pp' = 'predicate prediction'    (category 2 data)

#
# set name of PPNN input data file (be descriptive)
#

if source_data_category == 1:
    filename = 'dummy'
    #filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_4_trainset.json'
    #filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_4_testset.json'
elif source_data_category == 2:
    #filename = 'dummy'
    #filename = 'ppnn_input_pp_baseline_annotations_train.json'
    filename = 'ppnn_input_pp_baseline_annotations_test.json'
else:
    raise ValueError('source data category not recognised')


# set directory in which to store the PPNN input data file initially
# (nb: the JSON file may later be moved to `/data/ppnn_input_data/`)
filepath = os.path.join('~', 'research', 'ppnn', filename)
filepath = os.path.expanduser(filepath)

# save the PPNN input data to disk in JSON format
vrdu8.write_ppnn_input_data_to_file(d3_object_ordered_pairs_per_image_extended,
                                    path=filepath)

print()
print(f'PPNN input data saved to file: {filepath}')


#%%

print('Processing complete')



