#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module defines a subclass of PyTorch's torch.utils.data.Dataset class
called PPNNDataset.  The PPNNDataset class is used in the training of
predicate prediction neural networks (PPNNs).
'''

import torch
import os
import sys

if not os.getcwd().endswith('predicatePrediction'):
    raise RuntimeError('unexpected current working directory')
filepath_root = '..'
vrd_utils_dir_path = os.path.join(filepath_root, 'extensibility', 'analysis')
sys.path.insert(0, vrd_utils_dir_path)
import nesy4vrd_utils as vrdu

import vrd_utils8 as vrdu8
import vrd_ppnn_matching as vrdm


class PPNNDataset(torch.utils.data.Dataset):

    def __init__(self, data_filepath=None, featureset=None,
                 anno_filepath=None, nr_object_classes=None,
                 nr_predicates=None, targets_required=True):
        super(PPNNDataset, self).__init__()

        # get the ppnn input data
        self.ppnn_data = vrdu8.load_ppnn_input_data(data_filepath)

        # the feature set specifies the set of bbox geometric and spatial
        # relation features included in the ppnn input data for training
        # or inference
        if featureset not in [1,2,3,4]:
            raise ValueError(f'Feature set {featureset} not recognised')
        self.featureset = featureset

        self.img_anno = vrdu.load_VRD_image_annotations(anno_filepath)

        self.img_names = list(self.img_anno.keys())

        self.nr_object_classes = nr_object_classes

        self.nr_predicates = nr_predicates
        
        # targets are required during training but not during inference
        self.targets_required = targets_required

    def get_ppnn_data_for_image(self, idx):
        
        img_name = self.img_names[idx]

        ppnn_img_dict = self.ppnn_data[img_name]
        
        # sometimes the PPNN input data for an image will be empty; this
        # happens if an ODNN detected 0 or 1 objects for an image; we need
        # an image to have at least 2 objects before ordered pairs of objects
        # can be formed; we skip these special cases and return a dummy
        # data_tensor with value None; it's up to the caller to check this
        # data_tensor to see it's None
        if len(ppnn_img_dict['obj_ord_pairs']) == 0:
            data_tensor = None
        else:  
            # convert the dictionary of PPNN input data to a tensor
            data_tensor = vrdu8.ppnn_image_dict_2_tensor(ppnn_img_dict,
                                                         self.nr_object_classes,
                                                         self.featureset)
        
        # nb: we return all 3 elements: the image name, the source ppnn input
        # data for the image, and the corresponding ppnn data tensor;
        # all 3 have their uses; crucially, the ppnn_img_dict contains a
        # dictionary of the 'object ordered pairs' for the image; this plays
        # a key role allowing us to construct and interpret tensors by
        # telling us the ordered pair of objects associated with each row of
        # these tensors (eg PPNN output tensors, loss tensors, etc.); the
        # order of the entries in this dictionary corresponds 1-to-1 to the
        # rows of these other tensors
        return img_name, ppnn_img_dict, data_tensor

    def get_targets_for_image(self, idx, ppnn_img_dict):
        
        # BITT : build initial target tensor

        #
        # BITT step 1: create target vector set A
        #

        img_name = self.img_names[idx]
        img_anno = self.img_anno[img_name]
        
        # from the image's annotated VRs, extract the set of unique objects
        # that have been annotated, as represented by their bboxes and 
        # class labels; also convert the format of these bboxes from
        # the standard VRD format to the FRCNN format
        results = vrdu8.avrs_2_objects_per_image([img_name], 
                                                 {img_name: img_anno},
                                                 image_dir=None,
                                                 include_image_dims=False)
        objects_per_img = results
        
        # test-mode
        #for k, v in objects_per_img.items():
        #    print(f"dataset 1: object_per_img: {v['boxes']}")
        
        
        # from the set of unique objects create all possible ordered pairs
        # of objects (excluding all pairs of like objects); note that the
        # bboxes in the ordered pairs are in FRCNN format
        results = vrdu8.create_object_ordered_pairs(objects_per_img,
                                                    include_image_dims=False)
        trgt_obj_ord_pairs = results
        
        # test-mode
        #for k, v in trgt_obj_ord_pairs.items():
        #    print(f"dataset 2: obj_ord_pairs: {v['obj_ord_pairs']}")        
           
        # set the dimension (length) of the target vectors 
        #
        # we set it to be 1 greater than the number of predicates in NeSy4VRD;
        # we do this regardless of the level of dimension D_model2 (per
        # variable tr_d_model2), which is not even passed into this module
        # because we don't need to know its value
        #
        # if D_model2==2, the PPNN output tensor will have an extra column
        # for a 'no predicate' predicate; so the target tensor will need an
        # extra 'no predicate' column as well, so the PPNN output and target
        # tensors match in size
        #
        # if D_model2==1, the PPNN output tensor will not have an extra
        # column; but even so, we still want to introduce an extra
        # 'no predicate' column because the 'no predicate' column plays an
        # important role in loss computation where we use it to distinguish 
        # between *matched* and *unmatched* target vectors
        #
        # a direct consequence of adding an extra column to the target
        # vectors here, regardless of the level of D_model2, is that, when
        # D_model2==1, the sizes of the PPNN output and the target tensors
        # won't match --- because the target tensor will have 1 extra
        # column; but this state of affairs is only temporary; when
        # D_model2==1, just prior to the calculation of loss, this extra
        # column is removed from the target tensor so that the sizes of the 
        # output and target tensors match
        target_dim = self.nr_predicates + 1
        
        # transform the ordered pairs of objects in the image into a
        # preliminary set of target vectors, called set A.
        targets_A = vrdu8.create_target_vectors_A(trgt_obj_ord_pairs, 
                                                  img_anno,
                                                  target_dim)
        
        # test-mode
        #n_rows = int(targets_A.shape[0])
        #n_nopred = int(targets_A[:,-1].sum())
        #if n_nopred == n_rows:
        #    print('dataset 3: all targets_A are nopred targets')       
        
        #
        # BITT steps 2 & 3: matching and creation of target vector set B
        # 

        targets_B = vrdm.create_target_vectors_B(ppnn_img_dict,
                                                 trgt_obj_ord_pairs,
                                                 targets_A)

        testing = False
        if testing:
            print('vrd_ppnn_dataset.py get_targets_for_image()')
            print(f'targets_B.shape: {targets_B.shape}')
            n_matched_nopredpred = 0
            n_unmatched_nopredpred = 0
            n_rows = int(targets_B.shape[0])
            for idxt in range(n_rows):
                n_preds = targets_B[idxt].sum()           
                if targets_B[idxt,-1] == 1.0:
                    n_unmatched_nopredpred += 1
                if targets_B[idxt,-1] == 2.0:
                    n_matched_nopredpred += 1
                    n_preds -= 1
                print(f'idxt: {idxt}, n_preds: {n_preds}, nopredpred: {targets_B[idxt,-1]}')
            print(f'n_matched_nopredpred: {n_matched_nopredpred}, n_unmatched_nopredpred: {n_unmatched_nopredpred}')
            #n_nopredpred = int(targets_B[:,-1].sum())
            #if n_nopredpred == n_rows:
            #    print('dataset 4: all targets_B are nopred targets')      

        # convert the set B target vectors (in a 2D numpy array) to a 2D tensor
        targets = torch.tensor(targets_B, dtype=torch.float32)
        return targets

    def __len__(self):
        # return the number of image entries in the set of image annotations
        # specified by the caller in the anno_filepath parameter
        return len(self.img_names)

    def __getitem__(self, idx):
        # the amount of ppnn input data (number of feature vectors) varies
        # for different images; so the sizes of our mini-batches will vary;
        # they will always pertain to a single image, but the number of
        # input feature vectors will vary greatly (as a function of the
        # number of objects detected/annotated in each image)
        img_name, ppnn_img_dict, inputdata = self.get_ppnn_data_for_image(idx)
        if self.targets_required:  # training mode; targets needed for loss calc
            targets = self.get_targets_for_image(idx, ppnn_img_dict)
        else:                      # inference mode; targets not needed (yet)
            targets = None
        # assemble everthing we wish to return into a dictionary
        results = {'img_name': img_name,
                   'ppnn_img_dict': ppnn_img_dict,
                   'inputdata': inputdata,
                   'targets': targets}

        return results
