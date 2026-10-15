#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
During PPNN training, we capture the values of the logits of the most
important PPNN model output neurons for each ordered pair of objects in 
designated images. These logits are written to image-specific and 
epoch-specific .json files.

This script consolidates these multiple .json files containing PPNN model
output logits for designated images into summary .csv files.  A summary
.csv file shows, for a given image, the values of the logits for the most 
important PPNN model output neurons as the training epochs advance.
These most important output neurons are the ones whose logits, if 
positive, will correspond to 'hits' --- ie to predicted visual relationships
that will (ultimately) match with (or hit) ground-truth, target VRs.

The consolidated .csv files provide an easy, visual way to assess the
efficiency and effectiveness of the PPNN learning that goes on during a 
particular training run.  And it gives this view early in the pipeline, just 
after the training step. We don't need to wait for a full pipeline to complete
to get a rough sense of how well a PPNN model will ultimately perform at
VR detection.
'''

#%%

import os
import glob
import json
import sys
import pandas as pd

import vrd_utils14 as vrdu14


#%% gather arguments supplied to this script

# get or set the experiment space training region cell id
if len(sys.argv) > 1:
    trc_id = sys.argv[1]
else:
    trc_id = 'trc0460'
if not trc_id.startswith('trc'):
    raise ValueError(f'training region cell id {trc_id} not recognised')

# get or set the platform on which the training script is running
if len(sys.argv) > 2:
    platform = sys.argv[2]
else:
    platform = 'macstudio'
if not platform in vrdu14.supported_platforms:
    raise ValueError(f'platform {platform} not recognised')

# # get or set the work directory (the folder in which to store output files)
if len(sys.argv) > 3:
    workdir = sys.argv[3]
else:
    workdir = 'ppnn'


#%% build the path to the model checkpoint directory for storing outputs

if platform == 'hyperion':
    monitor_dir = os.path.join('~', 'sharedscratch', 'research', workdir)
else:
    monitor_dir = os.path.join('~', 'research', workdir)

scriptName = os.path.basename(__file__)
print(f'script name: {scriptName}')

monitor_dir = os.path.expanduser(monitor_dir)
print(f"work dir   : {monitor_dir}")


#%% get the names of the (sub)training set images

anno_dir = os.path.join('..', 'data', 'annotations')

filename = 'nesy4vrd_image_names_train_training.json'

filepath = os.path.join(anno_dir, filename)

with open(filepath, 'r') as fp:
    vrd_img_names_training = json.load(fp)

print(f'Image names for (sub)training set loaded: {len(vrd_img_names_training)}')


#%%

def build_input_filename_pattern(trc_id, mb_num):

    filename_pattern = 'vrd_ppnn_' + trc_id + '_monitor_img_'
    filename_pattern = filename_pattern + str(mb_num).zfill(4) + '_epoch_*.json'

    return filename_pattern


#%%

def build_output_filename(trc_id, mb_num):

    filename = 'vrd_ppnn_' + trc_id + '_logits_img_'
    filename = filename + str(mb_num).zfill(4) + '_consolidated.csv'

    return filename


#%%

def load_logits_for_epoch(path):

    with open(path, 'r') as fp:
        logit_data = json.load(fp)

    return logit_data


#%%

def initialise_data_dictionary():

    dd = {}

    dd['key'] = []
    dd['bbx_ord_pair'] = []

    return dd


#%%

def consolidate_logits_for_image(img_name, img_idx, monitor_dir, 
                                 logit_filenames, trc_id):

    dd = initialise_data_dictionary()

    for idx1, filepath in enumerate(logit_filenames):

        lfe = load_logits_for_epoch(filepath)

        if lfe['img_name'] != img_name:
            raise ValueError(f"Problem; unexpected img_name: {lfe['img_name']}")

        if lfe['img_idx'] != img_idx:
            raise ValueError(f"Problem; unexpected img_idx: {lfe['img_idx']}")

        epoch_num = lfe['epoch_num']

        logits = []

        target_logits = lfe['target_logits']

        for idx2, keyval in enumerate(target_logits.items()):
            key, val = keyval
            bbx_ord_pair = val['bbx_ord_pair']
            logit_val = val['logit_val']

            if idx1 == 0:
                dd['key'] = dd['key'] + [key]
                dd['bbx_ord_pair'] = dd['bbx_ord_pair'] + [bbx_ord_pair]
            else:
                if key != dd['key'][idx2]:
                    raise ValueError("Problem; target element mismatch")
                if bbx_ord_pair != dd['bbx_ord_pair'][idx2]:
                    raise ValueError("Problem; bbx_ord_pair mismatch")

            logits.append(logit_val)

        # build new dataframe column name for current epoch
        colname = 'e' + str(epoch_num).zfill(4)

        dd[colname] = logits


    # convert the data dictionary to a Pandas DataFrame
    df = pd.DataFrame(dd)

    # build output filename and path
    out_filename = build_output_filename(trc_id, img_idx)
    outfilepath = os.path.join(monitor_dir, out_filename)

    # save dataframe to .csv file
    df.to_csv(outfilepath, index=False)

    return None


#%% main processing loop NEW

# gather the logit data filenames present in the target directory
filename_pattern = 'vrd_ppnn_' + trc_id + '_logits_img_*.json'
search_pattern = os.path.join(monitor_dir, filename_pattern)
logit_filenames = glob.glob(search_pattern)
logit_filenames = sorted(logit_filenames)

index_prefix  = monitor_dir + os.sep + 'vrd_ppnn_' + trc_id + '_logits_img_'

active_img_idx = None

filenames_for_image = []

img_count = 0


# iterate over all of the (sorted) captured logit .json files; gather the
# (per epoch) filenames that pertain to unique images; consolidate those
# per epoch logits into a summary .csv file for each image whose logits
# are being monitored

for filename in logit_filenames:
    
    # extract the image index number from the current filename
    img_idx_str = filename.removeprefix(index_prefix)
    img_idx = int(img_idx_str[0:4])
       
    if img_idx == active_img_idx:
        # we continue gathering filenames for the active image
        pass
    else:
        # we have encountered a filename for a new image
        if len(filenames_for_image) > 0:
            # finish processing the active image by consolidating its
            # logit .json files into a summary .csv file
            img_name = vrd_img_names_training[active_img_idx]
            consolidate_logits_for_image(img_name, active_img_idx, monitor_dir, 
                                         filenames_for_image, trc_id)
            print(f"logits consolidated for image index: {active_img_idx}")
        
        # prepare for the new active image
        active_img_idx = img_idx
        filenames_for_image = []
        img_count += 1
    
    # gather filenames for the currently active image
    filenames_for_image.append(filename)


if len(filenames_for_image) > 0:
    # finish processing the final active image by consolidating its
    # logit .json files into a summary .csv file
    img_name = vrd_img_names_training[active_img_idx]
    consolidate_logits_for_image(img_name, active_img_idx, monitor_dir, 
                                 filenames_for_image, trc_id)    
    print(f"logits consolidated for image index: {active_img_idx}")

print(f'number of images for which logits were consolidated: {img_count}')

print('processing complete')



