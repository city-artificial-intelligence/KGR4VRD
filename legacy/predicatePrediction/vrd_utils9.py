#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module contains utility functions that enable introspective analysis
of how efficiently and effectively a PPNN model is learning what we want 
it to learn during training.

For a designated image, the logit values of particular PPNN output neurons 
are captured during training and written to a file. The file is specific
to the image and to a particular training epoch.

The function capture_logits_for_image() calls all the others.
'''


#%% imports

import torch
import os
import json


#%%

def build_filename(trc_id, img_idx, epoch_num):
    '''
    Build a filename for captured learning data (PPNN output logits).  

    Parameters
    ----------
    trc_id : string
        Identifies the training region cell of the experiment space.
    mb_num : integer
        The number (enumerative index) of the mini-batch being processed.
        Since our batch-size is always 1, this number maps to a unique
        image_name, the name of the image whose visual relationship data 
        is being processed.  It's main purpose here is to uniquely identify
        an image (ie it's a lot shorter than a string image name).
    epoch_num : integer
        The number of the training epoch being processed.

    Returns
    -------
    filename : string
        A filename for captured learning data (logits).

    '''
    
    filename = 'vrd_ppnn_' + trc_id + '_logits_img_'
    filename = filename + str(img_idx).zfill(4) + '_epoch_'
    filename = filename + str(epoch_num).zfill(3) + '.json'

    return filename


#%%

def write_logits_to_file(data, path):
    '''
    Write data to a JSON file.

    Parameters
    ----------
    data : dictionary
        A dictionary of image filenames and their associated features, in
        particular object ordered pairs and their associated sets of
        features.
    path : string
        A path and filename indicating where to store the file on disk and
        what name is to be assigned to that file.

    Returns
    -------
    None
    '''

    with open(path, 'w') as fp:
        json.dump(data, fp)

    return None


#%%

def get_bbox_ordered_pair(pair_dict, entry_idx):
    '''
    Get the bbox ordered pair that corresponds to the value in entry_idx
    '''

    bbx_ord_pair = None

    for idx, key in enumerate(pair_dict):
        if idx == entry_idx:
            bbx_ord_pair = key
            break

    return bbx_ord_pair    # '((x1, y1, x2, y2), (x1, y1, x2, y2))'


#%%

def capture_logits_for_image(trc_id, img_idx, epoch_num, img_name,
                             ppnn_img_dict, output, targets, device, 
                             monitor_dir):
    '''
    Filter the components associated with a single mini-batch of PPNN
    training activity for key indicators of how the neural learning
    process is proceeding. Save these key indicators to a text file for
    visual analysis.

    Parameters
    ----------
    trc_id : string
        The ID of the training region cell of the experiment space that is
        being processed.
    mb_num : integer
        The mini-batch number of the training loop. This corresponds to a 
        unique image name since each mini-batch has data for one image.
    epoch_num : integer
        The epoch number within the training loop of the PPNN model.
    img_name : string
        The name of the image associated with the mini-batch under analysis.
    ppnn_img_dict : dictionary
        The dictionary holding the PPNN input data associated with the
        current image and mini-batch.
    output : tensor
        The tensor of outputs for the current mini-batch, as produced by the
        PPNN model undergoing training. This is the result of the PPNN model's
        forward pass over the PPNN input data associated with the image that
        is the subject of the current mini-batch.
    targets : tensor
        The tensor of binary, multi-hot target vectors associated with the
        current mini-batch.
    monitor_dir : string
        A path to a directory in which to store monitor data files.

    Returns
    -------
    message : string
        The path/name of the file written to disk containing the information
        to facilitate introspective analysis of the PPNN learning process.
    '''

    # Things to monitor
    # 1) the logit values for the elements with nonzero targets
    # 2) the logit values for the 'nopredicate' elements
    # 3) any positive logit value
    # 4) the proportion of nonzero target elements with positive logit values
    # 5) the rank of the logit values for the elements with nonzero targets
    #    in terms of their magnitude relative to all the others; ie are the
    #    logit values for the elements with nonzero targets the largest
    #    values for that row, even if they aren't yet positive


    # NOTE: if we change back to processing tensors with 3 dimensions 
    # (ie that have a leading 'batch' dimension), when we simply comment-out
    # this check for 2 dimensions and things should work fine
    if targets.dim() != 2:
        raise ValueError('targets tensor has unexpected number of dimensions')
    
    if targets.dim() == 2:
        targets2 = targets
        output2 = output
    else:
        targets2 = targets.squeeze()
        output2 = output.squeeze()
        
    
    # NOTE: all the code in the function below this point assumes that we
    # are working with 2D tensors (ie no leading batch dimension)

    # find all of the nonzero elements in the targets tensor and get
    # their coordinates; these nonzero elements are the 'hot' predicates, the
    # ones that have been annotated in ground-truth visual relationships
    # for ordered pairs of objects; their coordinates correspond to the
    # logits in the PPNN model output tensor that we wish to monitor as
    # training proceeds; as these logits become large and positive, it means 
    # a PPNN model is learning to predict the target (correct) predicates and 
    # this is a good sign that correlates with good 'recall' predictive
    # performance; so this is what we wish to see happening as we monitor
    # these data --- we wish to see the logits we track becoming large and
    # positive as the training epochs grow in number
    # 
    # nb: we ignore the last column in the targets because those last elements
    # in each target vector correspond to the 'no predicate' predicate; if
    # this element is nonzero (hot), it signals that no ground-truth VRs
    # were annotated for the corresponding ordered pair of objects

    # nb: in PyTorch 1.13.1 the torch.nonzero() function is not supported 
    # on the MPS backend and will fail back to run on the CPU; so we use .cpu()
    # to put a copy of the tensor in CPU memory ourselves, explicitly, rather
    # than have it happen invisibly, plus get a UserWarning msg written to
    # the ipython console; so if we're running on an MPS device and using
    # PyTorch 1.13.1, we need to copy the tensor to CPU memory
    # 
    # nb: in PyTorch 2.0 the torch.nonzero() function is supported
    # on the MPS backend, so we don't need this computation overhead of
    # copying the tensor into CPU memory in order for the nonzero() function
    # to work; so if we're running on PyTorch 2, we can ignore this issue

    if device == torch.device('mps'):
        if torch.__version__ < '2.0.1':
            targets2 = targets2.cpu()

    watch_elems = targets2[:,0:-1].nonzero()

    watch_elems_logits = []
    for idx in range(watch_elems.shape[0]):
        logit = output2[watch_elems[idx][0], watch_elems[idx][1]].item()
        watch_elems_logits.append(round(logit,5))

    # initialise a mother dictionary for holding everything we wish to monitor
    logit_dict = {}
    logit_dict['img_name'] = img_name
    logit_dict['epoch_num'] = epoch_num
    logit_dict['img_idx'] = img_idx

    # extract dictionary of object ordered pairs from ppnn image dict
    pair_dict = ppnn_img_dict['obj_ord_pairs']

    # build a dictionary containing the current logit values for the
    # target elements (predicates per ordered pairs of objects) that we
    # wish to monitor
    target_logit_dict = {}
    for idx in range(watch_elems.shape[0]):
        key = watch_elems[idx].tolist()   #  [row, col]
        entry_idx = key[0]
        key = str(key)                    # '[row, col]'
        val = {}
        val['bbx_ord_pair'] = get_bbox_ordered_pair(pair_dict, entry_idx)
        val['logit_val'] = watch_elems_logits[idx]     #  n.nnn
        target_logit_dict[key] = val

    # add the logit values for the target elements to the mother dictionary
    logit_dict['target_logits'] = target_logit_dict
    
    filename = build_filename(trc_id, img_idx, epoch_num)

    path_and_filename = os.path.join(monitor_dir, filename)
    
    write_logits_to_file(logit_dict, path_and_filename)

    return None


