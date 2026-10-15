#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module defines utility functions that pertain to the preparation of 
PPNN input data.

The preparation occurs in two stages and hence PPNN input data preparation
tasks span different steps of the PPNN pipeline.
* step 1 - first stage preparation (batch preparation of PPNN input data files)
* step 2 - PPNN training (real-time preparation of Dataset mini-batches)
* step 5 - PPNN inference (real-time preparation of Dataset mini-batches)
'''

#%%

import os
import sys
import itertools
import copy
import json
import numpy as np
import pandas as pd
import torch

if not os.getcwd().endswith('predicatePrediction'):
    raise RuntimeError('unexpected current working directory')
filepath_root = '..'
vrd_utils_dir_path = os.path.join(filepath_root, 'extensibility', 'analysis')
sys.path.insert(0, vrd_utils_dir_path)
import nesy4vrd_utils as vrdu

import vrd_utils5 as vrdu5


#%%

def load_ppnn_input_data(path):
    '''
    Load PPNN input data, converting it from JSON to a dictionary.

    Parameters:
        path : string
            Path to PPNN input data to be loaded.

    Returns:
        ppnn_input_data : dictionary
            The 'keys' are image names; the 'values' are dictionaries whose
            entries (keys) are ordered pairs of bboxes and whose values are
            sets of features.
    '''

    with open(path, 'r') as fp:
        ppnn_input_data = json.load(fp)

    return ppnn_input_data

#%% restructure bbox specifications

def restructure_bboxes(boxes, toFormat):
    '''
    Restructure bbox specifications.
    
    Transform bbox specifications between VRD format, 
    [ymin, ymax, xmin, xmax], and FRCNN format, [xmin, ymin, xmax, ymax]. 
    (FRCNN is the Faster RCNN object detector.)
    
    Parameters:
        boxes : list
            A list of bbox specifications, where a bbox specification is
            represented by 4 numbers.
        toFormat : string
            Indicates the format to which the incoming bbox specifications
            are to be transformed.

    Returns:
        boxes2 : list
            A list of transformed bbox specifications, and where the bboxes
            are represented as tuples rather than lists.
    '''

    if toFormat not in ['vrd', 'frcnn']:
        raise ValueError('toFormat not recognised')

    boxes2 = []
    for box in boxes:
        if toFormat == 'frcnn':
            box2 = (box[2], box[0], box[3], box[1])
        else:
            box2 = (box[1], box[3], box[0], box[2])
        boxes2.append(box2)

    return boxes2


#%%

def avrs_2_objects_per_image(img_names, img_annos, image_dir, include_image_dims):
    '''
    From the annotated visual relationships (avrs) for VRD images, extract
    the unique set of objects referred to within those visual relationships.

    This conversion is part of enabling the 'predicate prediction' regime
    of experiments, where PPNN input data is constructed not from the 
    outputs of an object detector (ODNN) but from the ground-truth
    annotated visual relationships (VRs).

    This function extracts from the annotated VRs the unique set of objects
    referenced in those VRs.  These objects are represented by their bboxes 
    and object class labels.

    For each image, the functions builds a dictionary entry keyed by image
    name, whose value is dictionary containing  
     - the height/width dimensions of the image
     - the objects in the image, as a dictionary, where the keys are the 
     bboxes (as tuples) and the values are the integer object class labels.

    Parameters
    ----------
    img_names : list
        Names of VRD image files
    img_annos : dictionary
        Visual relationship annotations for VRD images (in the standard
        format used by the VRD dataset)
    imagedir : string
        Indicates the directory in which to find the target images:
        the 'train_images' or the 'test_images' directory.
    include_image_dims : boolean
        Indicates whether or not the caller wishes the image dimensions to
        be included in the dictionary that is constructed and returned.       

    Returns
    -------
    objects_per_image : dictionary of dictionaries
        A dictionary where each key is an 'image name' and each value is
        an 'image dictionary' with the following keys:values
            'imwidth': width (integer)
            'imheight': height (integer)
            'boxes': [list of bbox tuples]
            'labels': [list of integer object class labels]
    
    Details
    -------
    Annotated visual relationships (avrs) always use the native VRD format
    for specifying bounding boxes: [ymin, ymax, xmin, xmax]. One of the
    responsibilities of this function is to convert the VRD format to the
    FRCNN format, [xmin, ymin, xmax, ymax].  So any downstream processing
    of the outputs of this function should assume the bboxes are in
    FRCNN format!!
    '''

    if include_image_dims not in [True, False]:
        raise ValueError('include_image_dims must be True or False')
    
    objects_per_image = {}
    
    for imname in img_names:
        imanno = img_annos[imname]

        # extract the object bboxes and categories (classes) from the
        # annotated visual relationships for the current image
        objects = vrdu.get_bboxes_and_object_classes(imname, imanno)
        boxes = list(objects.keys())
        labels = list(objects.values())

        # Restructure the bbox specifications: convert them from the
        # protocol used by VRD visual relationship annotations,
        # (ymin, ymax, xmin, xmax), to the protocol used for bboxes
        # output by a Faster R-CNN object detection model,
        # (xmin, ymin, xmax, ymax). And express the bboxes as tuples
        # rather than lists.
        boxes2 = restructure_bboxes(boxes, toFormat='frcnn')

        if include_image_dims:
            # get image dimensions
            imsize = vrdu.get_image_size(imname, imagedir=image_dir)
            # save the results
            objects_per_image[imname] = { 'imwidth': imsize[0],
                                          'imheight': imsize[1],
                                          'boxes': boxes2,
                                          'labels': labels }
        else:
            # save the results
            objects_per_image[imname] = { 'boxes': boxes2,
                                          'labels': labels }

    return objects_per_image


#%%

def create_object_ordered_pairs(objects_per_image, include_image_dims):
    '''
    Convert dictionaries of 'objects per image' to dictionaries of
    'ordered pairs of objects per image', and store features
    associated with each ordered pair of objects.

    For a given image, take all its objects and create all ordered pairs 
    of bboxes. Remove bbox pairs where sub_bbox == obj_bbox. Retain the 
    object class label of each bbox in the pair, storing these as basic 
    features of the bbox pair.

    Parameters
    ----------
    objects_per_image : dictionary of dictionaries
        A dictionary keyed by 'image name' where the values are 'image
        dictionaries' with keys:values :
            'imwidth': width (integer)
            'imheight': height (integer)
            'boxes': [list of bboxes, as tuples]
            'labels': [list of integer object class labels]
    include_image_dims : boolean
        Indicates whether or not the caller wishes the image dimensions to
        be included in the feature dictionary that is constructed.

    Returns
    -------
    object_ordered_pairs_per_image : dictionary
        A dictionary keyed by 'image name' where the values are 'image
        dictionaries' with keys:values:
            'im_w_orig': width (integer)
            'im_h_orig': height (integer)
            'obj_ord_pairs': { entry }
        and where each entry in the obj_ord_pairs dictionary is keyed by
        an ordered pair of bboxes (as a string), and has structure:
            '(obj1_bbox, obj2_bbox)': { 'b1_lab': obj1_label
                                        'b2_lab': obj2_label }
    '''

    results = {}

    for imname, imdict in objects_per_image.items():

        bboxes = imdict['boxes']
        labels = imdict['labels']

        # combine corresponding bboxes and object class labels into tuples
        bb_cls_tuples_zip = zip(bboxes, labels)

        # convert the output of zip() to a list of tuples
        bb_cls_tuples = [item for item in bb_cls_tuples_zip]

        # form all possible ordered pairs of our (bbox, label) 2-tuples
        ordered_pairs1 = itertools.product(bb_cls_tuples, repeat=2)

        # remove those ordered pairs where the two elements are identical
        ordered_pairs2 = [item for item in ordered_pairs1 if item[0] != item[1]]

        # create a dictionary of object ordered pair feature dictionaries,
        # where each object ordered pair feature dictionary is keyed by
        # a 2-tuple representing an ordered pair of object bboxes, and
        # the values are feature dictionaries containing (for now) the
        # object class labels (categories) as the only features
        #
        # [The object ordered pair feature dictionaries will have the set
        #  of features they contain expanded, elsewhere. Here we are just
        #  establishing the basic data structure for storing sets of
        #  features associated with ordered pairs of object bboxes.]
        #
        object_ordered_pair_dict = {}
        for pair in ordered_pairs2:
            obj1 = pair[0]
            obj2 = pair[1]
            obj1_bbox = obj1[0]
            obj1_label = obj1[1]
            obj2_bbox = obj2[0]
            obj2_label = obj2[1]
            bbox_ord_pair = str((obj1_bbox, obj2_bbox))
            object_ordered_pair_dict[bbox_ord_pair] = { 'b1_lab': obj1_label,
                                                        'b2_lab': obj2_label }

        # build a dictionary for the current image that includes the
        # list of object ordered pair dictionaries
        if include_image_dims:
            results[imname] = { 'im_w_orig': imdict['imwidth'],
                                'im_h_orig': imdict['imheight'],
                                'obj_ord_pairs': object_ordered_pair_dict }
        else:
            results[imname] = { 'obj_ord_pairs': object_ordered_pair_dict }

    object_ordered_pairs_per_image = results

    return object_ordered_pairs_per_image


#%%

def create_target_vectors_A(obj_ord_pairs, img_anno, target_dim):
    '''
    Create a set of binary, multi-hot target vectors for one particular image.

    We refer to the target vectors created here as 'set A' target vectors.
    This is because they represent a preliminary starting point rather than
    a final set of target vectors. The set A target vectors produced here
    will undergo subsequent processing. Some may be rejected and replaced
    with other target vectors. Even the set A target vectors that survive
    may ultimately be modified just prior to loss computation based on
    KG reasoning.
    
    Each target vector in set A corresponds to a particular 
    ordered pair of objects annotated for a given VRD image.
    The elements of target vectors correspond to VRD predicates. The
    last element is special and corresponds to the 'no predicate' predicate.

    Parameters
    ----------
    obj_ord_pairs : dictionary
        A dictionary keyed by 'image name' where the values are 'image
        dictionaries' with keys:values:
            'obj_ord_pairs': { entry }
        and where each entry in the obj_ord_pairs dictionary is keyed by
        an ordered pair of bboxes (as a string), and has structure:
            '(obj1_bbox, obj2_bbox)': { 'b1_lab': obj1_label
                                        'b2_lab': obj2_label }
    img_anno : list of dictionaries
        The annotated visual relationships (VRs) for one particular image.
        The format of the dictionaries in the list is as per the default
        format used for the VR annotations of the VRD dataset. And the
        format of the bboxes is the standard VRD format.
    target_dim : integer
        The dimension for the target vectors. This will be the number of
        predicates + 1, where the extra element will be treated as a
        'no predicate' predicate or indicator --- indicating that there 
        was no annotated VR (hence no predicate) for the associated 
        ordered pair of objects.

    Returns
    -------
    targets : 2D numpy array
        A set of binary, multi-hot target vectors.
        The target vectors correspond positionally to the object ordered
        pair entries in dictionary obj_ord_pairs.
    '''

    if len(obj_ord_pairs) > 1:
        raise ValueError('More than one entry in obj_ord_pairs dict')

    targets = []

    for imname, imdict in obj_ord_pairs.items():

        # get the object ordered pairs dictionary for the current image
        pair_dict = imdict['obj_ord_pairs']

        # iterate over the object ordered pairs for the current image
        for key, feature_dict in pair_dict.items():

            # in the current key, the ordered pair of bboxes is a string
            # with format '((xmin,ymin,xmax,ymax), (xmin,ymin,xmax,ymax))',
            # that is a 2-tuple of 4-tuples; we break this string down and
            # convert it into two lists, one for 'subject' bbox (bb1), and
            # one for the 'object' bbox (bb2); note that the bboxes are
            # already in FRCNN format
            bb_ord_pair_tuple = eval(key)
            bb1 = list(bb_ord_pair_tuple[0])
            bb2 = list(bb_ord_pair_tuple[1])

            # get the bbox (integer) class labels
            bb1_label = feature_dict['b1_lab']
            bb2_label = feature_dict['b2_lab']

            # initialise target vector for current ordered pair of objects
            target = [0.0] * target_dim

            # iterate thru the annotated VRs for the image and find the ones
            # that match with the current ordered pair of objects, if any;
            # (nb: we convert the bboxes in img_anno from VRD format to 
            # FRCNN format before doing the matching with the ppnn input
            # ordered pairs of objects (whose bboxes are in FRCNN format)
            for vr in img_anno:
                bb = vr['subject']['bbox']
                # convert bbox from VRD format to FRCNN format
                sub_bbox = [bb[2], bb[0], bb[3], bb[1]]
                sub_idx = vr['subject']['category']
                
                prd_idx = vr['predicate']
                
                bb = vr['object']['bbox']
                # convert bbox from VRD format to FRCNN format                
                obj_bbox = [bb[2], bb[0], bb[3], bb[1]]
                obj_idx = vr['object']['category']

                if sub_bbox == bb1 and obj_bbox == bb2:
                    if sub_idx != bb1_label or obj_idx != bb2_label:
                        raise ValueError(f'bbox matching problem: {imname}')
                    target[prd_idx] = 1.0

            # if no annotated VRs in img_anno for the current ordered 
            # pair of objects in pair_dict were found, 
            # set the 'no predicate' element of the target vector to 1
            # to indicate that no predicate was annotated for the current
            # ordered pair of objects
            if sum(target) == 0.0:
                target[-1] = 1.0

            # save target vector
            targets.append(target)

    # convert target vectors from a list of lists to a 2D numpy array
    targets = np.array(targets)

    return targets


#%%

def get_new_image_dims_and_scaling_factors(orig_width, orig_height, max_dim_val):
    '''
    Calculate new dimensions for an image given a target maximum dimension
    value, and calculate the associated horizontal and vertical scaling
    factors for the notionally resized image that will permit the
    bboxes of the image to be appropriately resized whilst preserving all
    of the original (relative) geometric features of the image and its
    bboxes.

    Parameters
    ----------
    orig_width : integer
        The original width (in pixels) of a given image.
    orig_height : integer
        The original height (in pixels) of a given image.
    max_dim_val : integer
        The desired magnitude of the largest dimension of the notionally
        resized image.  The notional image resizing and the calculation of
        the horizontal and vertical scaling factors are determined by the
        value of this argument.

    Returns
    -------
    new_width : real
        The width of the notionally resized image.
    hsf : real
        The horizontal scaling factor to be used for rescaling the xmin and
        xmax elements specified in bboxes for the notionally resized image.
    new_height : real
        The height of the notionaly resized image.
    vsf : real
        The vertical scaling factor to be used for rescaling the ymin and
        ymax elements specified in bboxes for the notionally resized image.
    '''

    max_dim_val = float(max_dim_val)

    if orig_width > orig_height:
        new_width = max_dim_val
        new_height = (orig_height / orig_width) * max_dim_val
    else:
        new_height = max_dim_val
        new_width = (orig_width / orig_height) * max_dim_val

    new_width = round(new_width, 2)
    new_height = round(new_height, 2)

    # horizontal scaling factor
    hsf = round(new_width / orig_width, 5)

    # vertical scaling factor
    vsf = round(new_height / orig_height, 5)

    return new_width, hsf, new_height, vsf


#%%

def resize_bbox(orig_bbox, hsf, vsf):
    '''
    Resize a bounding box given horizontal and vertical scaling factors.

    Parameters
    ----------
    orig_bbox : tuple
        The original bbox, with format: (xmin, ymin, xmax, ymax)
    hsf : real
        A horizontal scaling factor.
    vsf : real
        A vertical scaling factor

    Returns
    -------
    new_bbox : list
        Specification of a resized (down-scaled) bbox: [xmin, ymin, xmax, ymax]

    '''

    new_bbox = [0,0,0,0]

    new_bbox[0] = round(float(orig_bbox[0] * hsf), 5)  # xmin
    new_bbox[1] = round(float(orig_bbox[1] * vsf), 5)  # ymin
    new_bbox[2] = round(float(orig_bbox[2] * hsf), 5)  # xmax
    new_bbox[3] = round(float(orig_bbox[3] * vsf), 5)  # ymax

    return new_bbox


#%%

def class_label_2_binary_onehot_vector(nr_object_types, label):
    '''
    Convert an integer object class label into a binary, one-hot vector
    representation.

    We use lists to represent the binary one-hot vectors because the data
    will be stored in JSON format, which does not support numpy arrays.

    Parameters
    ----------
    nr_object_types : integer
        The number of distinct object classes associated with the VRD dataset.

    label : integer
        An integer object class label (category).

    Returns
    -------
    bohv : list
        A binary one-hot vector representation of an object class label.

    '''

    bohv = [0.0] * nr_object_types
    bohv[label] = 1.0

    return bohv


#%%

def extend_image_and_object_ord_pair_features(object_ord_pairs_per_img,
                                              nr_object_types):
    '''
    Extend the set of features associated with each image and with the
    object ordered pairs belonging to each image.

    This function calculates and introduces:
    * new image-level features, including new (smaller) image dimensions,
      per a notional resizing (down-scaling) of the images
    * binary, one-hot vector representations of the object class labels
      for the bboxes in each object ordered pair
    * supplementary bbox specifications that reflect notional bbox
      resizing (down-scaling), per the notional image resizing (down-scaling)
    * geometric features calculated for each bbox and for the geometric
      (spatial) relationships between the ordered pairs of bboxes

    Parameters
    ----------
    object_ordered_pairs_per_image : dictionary
        A dictionary keyed by 'image name' where the values are 'image
        dictionaries' with keys:values:
            'im_w_orig': width (integer)
            'im_h_orig': height (integer)
            'obj_ord_pairs': { entry }
        and where each entry in the obj_ord_pairs dictionary is keyed by
        an ordered pair of bboxes (as a string), and has structure:
            '(obj1_bbox, obj2_bbox)': { 'b1_lab': obj1_label
                                        'b2_lab': obj2_label }

    nr_object_types : integer
        The number of distinct object classes associated with the VRD dataset.

    Returns
    -------
    object_ord_pairs_per_img_2 : dictionary
        An extended version of the dictionary received as input, where
        extra features have been introduced in the feature dictionary
        associated with each object ordered pair.
    '''

    # set the size of the maximum dimension to be used in the notional
    # resizing of all of the images and, in turn, of all of the object
    # bboxes associated with each image
    max_dim_val = 8

    # set number of decimals for rounding
    ndec = 3

    # the incoming dictionary is mutable, so we make a deep copy so we can
    # create an extended version of it without disturbing the original
    object_ord_pairs_per_img_2 = copy.deepcopy(object_ord_pairs_per_img)

    # extend the features associated with each ordered pair of objects
    for imname, imdict in object_ord_pairs_per_img_2.items():

        # get the original dimensions of the image
        im_w = imdict['im_w_orig']
        im_h = imdict['im_h_orig']

        # calculate the new dimensions of the current image per notional
        # resizing relative to the target magnitude of the largest dimension
        res = get_new_image_dims_and_scaling_factors(im_w, im_h, max_dim_val)
        im_w_new, hsf, im_h_new, vsf = res

        # capture the new image dimensions per the notionally
        # re-sized (down-scaled) image
        imdict['im_w'] = im_w_new
        imdict['im_h'] = im_h_new

        # get the object ordered pairs dictionary for the current image
        pair_dict = imdict['obj_ord_pairs']

        # iterate over the object ordered pairs for the current image
        for key, feature_dict in pair_dict.items():

            # introduce binary, one-hot vector representations of the
            # object class labels of the two bboxes in the ordered pair
            label = feature_dict['b1_lab']
            bohv = class_label_2_binary_onehot_vector(nr_object_types, label)
            feature_dict['b1_lab_bohv'] = bohv
            label = feature_dict['b2_lab']
            bohv = class_label_2_binary_onehot_vector(nr_object_types, label)
            feature_dict['b2_lab_bohv'] = bohv

            # in the current key, the ordered pair of bboxes has format
            # 'string': '((x1,y1,x2,y2),(x1,y1,x2,y2))';
            # convert this string back to a 2-tuple of bbox tuples and then
            # isolate the individual bbox tuples
            bb_ord_pair_tuple = eval(key)
            b1 = bb_ord_pair_tuple[0]
            b2 = bb_ord_pair_tuple[1]

            # resize the bboxes per the notional resizing of the image
            # (nb: the bbox coordinates have been rounded to 5 decimal points)
            b1_new = resize_bbox(b1, hsf, vsf)
            b2_new = resize_bbox(b2, hsf, vsf)           
            feature_dict['b1'] = b1_new
            feature_dict['b2'] = b2_new

            # get the centroids of the resized (down-scaled) bboxes
            b1_new_centroid, b2_new_centroid = vrdu5.bb_centroids(b1_new, b2_new)
            b1_new_centroid = list(b1_new_centroid)
            b1_new_centroid[0] = round(b1_new_centroid[0], ndec)
            b1_new_centroid[1] = round(b1_new_centroid[1], ndec)
            b2_new_centroid = list(b2_new_centroid)
            b2_new_centroid[0] = round(b2_new_centroid[0], ndec)
            b2_new_centroid[1] = round(b2_new_centroid[1], ndec)
            feature_dict['b1c'] = b1_new_centroid
            feature_dict['b2c'] = b2_new_centroid

            # get the widths and heights of the resized (down-scaled bboxes)
            # (nb: the magnitudes of the values of these features will be
            #  at most equal to the resized image dimensions, and will
            #  typically be much less; so if we're comfortable using the
            #  resized image dimensions as PPNN input features, there is no
            #  reason to not use the resized bbox widths and heights as
            #  PPNN input features too.)
            width, height = vrdu5.bb_width_height(b1_new)
            feature_dict['b1w'] = round(width, ndec)
            feature_dict['b1h'] = round(height, ndec)
            width, height = vrdu5.bb_width_height(b2_new)
            feature_dict['b2w'] = round(width, ndec)
            feature_dict['b2h'] = round(height, ndec)

            # get Euclidean distance between the centroids of the resized
            # (down-scaled) bboxes
            # (nb: we use the raw distance because of the notional resizing
            #  of the images and bboxes; so the magnitude of the values of
            #  this feature will, in practice, be in the interval
            #  [0, max_dim_val] or there abouts)
            eucl_dist_cent = vrdu5.bb_euclidean_distance_centroids(b1_new, b2_new)
            feature_dict['edc_b1b2'] = round(eucl_dist_cent, ndec)

            #
            # calculate geometric features of the individual bboxes and of
            # the relationships between the pair of bboxes, and introduce
            # these features into the feature dictionary
            #
            # nb: we can safely work with the original image dimensions and
            # original bbox sizes because the values for the features we
            # calculate will be the same either way
            #

            try:
                res = vrdu5.calc_bbox_geometric_features(b1, b2, im_w, im_h)
            except:
                print()
                print(sys.exc_info())
                print()
                print(f'Image: {imname}')
                print(f'bbox1: {b1}')
                print(f'bbox2: {b2}')
                print()
                raise Exception("Problem calculating bbox geometric features")

            # aspect ratios (h/w) of bboxes
            feature_dict['b1_ar'] = res['b1_ar']
            feature_dict['b2_ar'] = res['b2_ar']

            # bbox area to image area ratios
            feature_dict['b1a2iar'] = res['b1a2ia']
            feature_dict['b2a2iar'] = res['b2a2ia']

            # sine and cosine of angle between bbox centroids
            feature_dict['sine'] = res['sine']
            feature_dict['cosine'] = res['cosine']

            # intersection over union (IoU)
            feature_dict['iou'] = res['iou']

            # inclusion ratio of b1 within b2
            # [degree (extent) to which b1 is enclosed within b2]
            feature_dict['ir_b1b2'] = res['ir_b1b2']
            # inclusion ratio of b2 within b1
            # [degree (extent) to which b2 is enclosed within b1]
            feature_dict['ir_b2b1'] = res['ir_b2b1']

            # ratio of horiz distance between right & left edges of two bboxes
            # relative to image width
            feature_dict['hde2iwr'] = res['hde2iwr']
            # ratio of vert distance between top & bottom edges of two bboxes
            # relative to image height
            feature_dict['vde2ihr'] = res['vde2ihr']

            # ratio of the horizontal distance between the centroids of two
            # bboxes relative to the image width
            feature_dict['hdc2iwr'] = res['hdc2iwr']
            # ratio of the vertical distance between the centroids of two bboxes
            # relative to the image height
            feature_dict['vdc2ihr'] = res['vdc2ihr']

            # ratios of Euclidean distance between bbox centroids relative
            # to image width and image height
            feature_dict['edc2wr'] = res['eucl_dist_to_width_ratio']
            feature_dict['edc2hr'] = res['eucl_dist_to_height_ratio']

            # horiz dist between bbox centroids and nearest edges, as
            # ratios of img width
            feature_dict['hd_b1c_2_b2ne_r'] = res['hd_b1c_2_b2ne_r']
            feature_dict['hd_b2c_2_b1ne_r'] = res['hd_b2c_2_b1ne_r']

            # vert dist between bbox centroids and nearest edges, as
            # ratios of img height
            feature_dict['vd_b1c_2_b2ne_r'] = res['vd_b1c_2_b2ne_r']
            feature_dict['vd_b2c_2_b1ne_r'] = res['vd_b2c_2_b1ne_r']

    return object_ord_pairs_per_img_2


#%%

def write_ppnn_input_data_to_file(data, path):
    '''
    Write PPNN input data to a json file on disk.

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

def initialise_ppnn_data_dictionary(nr_object_types, featureset):
    '''
    Initialise a dictionary for holding PPNN input data that is suited to
    being converted directly into a Pandas DataFrame which will then
    be converted into a PyTorch Tensor ready to feed as input into a PPNN
    model for training or inference.

    The dictionary design does NOT reference an image name. The dictionary
    is designed to hold data for a single image under the assumption that
    the caller will manage the association of the data in the dictionary
    with a particular VRD dataset image, as required.

    Each 'key' in the dictionary will become a column name in a Pandas
    DataFrame. The 'value' of each 'key' in the constructed dictionary is
    initialised as an empty list. The lists are assigned values by the
    caller. Each list will become the data values in the column of a Pandas
    DataFrame associated with the 'key'.  The DataFrame will subsequently
    be converted into a 2D PyTorch Tensor (minus the column headings).

    Python dictionary order is guaranteed to be the insertion order.
    And Pandas DataFrames constructed from a dictionary have a column
    order that follows the insertion order of that dictionary.  The
    dictionary constructed in this function has been designed with
    these factors in mind.

    This function supports the construction of multiple different
    'feature sets' of the PPNN input data dictionary.  Different feature
    sets correspond to different subsets of the available features.  The
    features contained in different sets grows cumulatively.
    More concretely:
      set 1: the minimal set of features (default features only)
             - object bboxes and object class binary one-hot vectors
               only; ie ODNN output (or VR annotation info) only;
               no supplementary derived geometric features of any kind
      set 2: all features of v1, plus some additional ones
             - similar to what Donadello used in his paper; except where he
               used ratios of bbox areas relative to each other, we use the
               ratios of bbox areas relative to image areas and leave it to
               the NNs to learn relationships between those two ratios
      set 3: all features of v2, plus some additional ones
      set 4: all features of v3, plus some additional ones
             - the maximal set of features

    WARNING: The order of the key insertions in the dictionary constructed
    here should NOT BE CHANGED except in extreme circumstances and in the
    full knowledge of the potential consequences of doing so!!!

    Parameters
    ----------
    nr_object_types : integer
        The number of distinct object classes used in the VRD dataset.
    featureset : integer
        A value identifying the set of features to be included in the PPNN
        input data dictionary to be constructed.

    Returns
    -------
    dd : dictionary
        A PPNN input data dictionary.
    '''

    if featureset not in [1,2,3,4]:
        raise ValueError(f"Feature set identifier not recognised: {featureset}")

    dd = {}

    # create dictionary keys for the elements of the binary, one-hot
    # vector representation of the object class of bbox 1
    keybase = 'b1cv_'
    for idx in range(nr_object_types):
        key = keybase + str(idx).zfill(3)
        dd[key] = []

    # keys for the components of (resized / down-scaled) bbox 1
    dd['b1_xmin'] = []
    dd['b1_ymin'] = []
    dd['b1_xmax'] = []
    dd['b1_ymax'] = []

    # keys for the components of the centroid of (resized) bbox 1
    if featureset > 2:
        dd['b1c_x'] = []
        dd['b1c_y'] = []

    # keys for the width and height of (resized) bbox 1
    if featureset > 2:
        dd['b1w'] = []
        dd['b1h'] = []

    # create dictionary keys for the elements of the binary, one-hot vector
    # representation of the object class of bbox 2
    keybase = 'b2cv_'
    for idx in range(nr_object_types):
        key = keybase + str(idx).zfill(3)
        dd[key] = []

    # keys for the components of (resized / down-scaled) bbox 2
    dd['b2_xmin'] = []
    dd['b2_ymin'] = []
    dd['b2_xmax'] = []
    dd['b2_ymax'] = []

    # keys for the components of the centroid of (resized) bbox 2
    if featureset > 2:
        dd['b2c_x'] = []
        dd['b2c_y'] = []

    # keys for the width and height of (resized) bbox 1
    if featureset > 2:
        dd['b2w'] = []
        dd['b2h'] = []

    # image width & height (of notionally resized image)
    if featureset > 2:
        dd['im_w'] = []
        dd['im_h'] = []

    # euclidean distance between the centroids of (resized) bboxes 1 and 2
    if featureset > 1:
        dd['edc_b1b2'] = []

    # aspect ratios (h/w) of bboxes 1 and 2
    if featureset > 2:
        dd['b1_ar'] = []
        dd['b2_ar'] = []

    # bbox area to image area ratios
    if featureset > 1:
        dd['b1a2iar'] = []
        dd['b2a2iar'] = []

    # sine and cosine of the angle between the centroids of bboxes 1 and 2
    if featureset > 1:
        dd['sin'] = []
        dd['cos'] = []

    # Intersection over Union (IoU) for bboxes 1 and 2
    if featureset > 2:
        dd['iou'] = []

    # inclusion ratios of b1 within b2, and b2 within b1
    if featureset > 1:
        dd['ir_b1b2'] = []
        dd['ir_b2b1'] = []

    # horizontal distance between the right & left edges of bboxes 1 and 2
    # relative to image width (as a ratio)
    if featureset > 3:
        dd['hde2iwr'] = []
    # vertical distance between the top & bottom edges of bboxes 1 and 2
    # relative to image height (as a ratio)
    if featureset > 3:
        dd['vde2ihr'] = []

    # horizontal distance between the centroids of bboxes 1 and 2 relative
    # to image width (as a ratio)
    if featureset > 3:
        dd['hdc2iwr'] = []
    # vertical distance between the centroids of bboxes 1 and 2 relative
    # to image height (as a ratio)
    if featureset > 3:
        dd['vdc2ihr'] = []

    # euclidean distance between the centroids of bboxes 1 and 2 relative
    # to image width (as a ratio)
    if featureset > 3:
        dd['edc2wr'] = []
    # euclidean distance between the centroids of bboxes 1 and 2 relative
    # to image height (as a ratio)
    if featureset > 3:
        dd['edc2hr'] = []

    # horizontal distance between the centroid of bbox 1 and the nearest edge
    # of bbox 2 relative to image width (as a ratio)
    if featureset > 3:
        dd['hd_b1c_2_b2ne_r'] = []
    # horizontal distance between the centroid of bbox 2 and the nearest edge
    # of bbox 1 relative to image width (as a ratio)
    if featureset > 3:
        dd['hd_b2c_2_b1ne_r'] = []

    # vertical distance between the centroid of bbox 1 and the nearest edge
    # of bbox 2 relative to image height (as a ratio)
    if featureset > 3:
        dd['vd_b1c_2_b2ne_r'] = []
    # vertical distance between the centroid of bbox 2 and the nearest edge
    # of bbox 1 relative to image height (as a ratio)
    if featureset > 3:
        dd['vd_b2c_2_b1ne_r'] = []

    return dd


#%%

def ppnn_image_dict_2_tensor(ppnn_image_dict, nr_object_types, featureset):
    '''
    Convert a dictionary of image-related data into a PyTorch tensor ready for
    feeding as input to a PPNN model for training or inference.

    Parameters
    ----------
    ppnn_image_dict : dictionary
        A dictionary of ppnn data associated with a particular VRD dataset
        image. The structure of the dictionary is not yet appropriate for
        conversion into a PyTorch tensor and must first undergo significant
        restructuring.
    nr_object_types : integer
        The number of distinct object classes associated with the VRD dataset.
    featureset : integer
        A value identifying the set of features to be drawn from the PPNN
        input data dictionary and rendered within the returned data tensor.

    Returns
    -------
    data_tensor : Tensor
        A 2D tensor of PPNN input data corresponding to a particular VRD
        dataset image.  The number of rows varies per image.  The tensor
        corresponds to a mini-batch of training/inference data.
    '''

    if featureset not in [1,2,3,4]:
        raise ValueError(f"Feature set identifier not recognised: {featureset}")

    # get the dimensions of the (notionally resized) image with which
    # the ppnn data are associated
    im_w = ppnn_image_dict['im_w']
    im_h = ppnn_image_dict['im_h']

    # get the object ordered pairs dictionary for the current image
    pair_dict = ppnn_image_dict['obj_ord_pairs']

    #
    # initialise lists to temporarily hold features whose values are not
    # scalars and, hence, which need to be restructured in a column-wise
    # manner before the lists of the ppnn input data dictionary keys can
    # be populated correctly
    #

    # object class label binary one-hot vectors
    obj1_label_bohvs = []
    obj2_label_bohvs = []

    # object bboxes (resized / down-scaled)
    obj1_bboxes = []
    obj2_bboxes = []

    # centroids of the resized bboxes
    b1_centroids = []
    b2_centroids = []

    # initialise a ppnn input data dictionary designed for conversion
    # into a DataFrame which can then be converted into a PyTorch tensor
    dd = initialise_ppnn_data_dictionary(nr_object_types, featureset)

    #
    # iterate over the object ordered pairs for the current image
    # and 1) if the features have scalar values, load them directly
    # into the data dictionary, and 2) otherwise load the features
    # into temporary lists for subsequent restructuring
    #
    for key, fd in pair_dict.items():

        obj1_label_bohvs.append(fd['b1_lab_bohv'])
        obj1_bboxes.append(fd['b1'])
        b1_centroids.append(fd['b1c'])

        obj2_label_bohvs.append(fd['b2_lab_bohv'])
        obj2_bboxes.append(fd['b2'])
        b2_centroids.append(fd['b2c'])

        if featureset > 1:
            dd['edc_b1b2'] = dd['edc_b1b2'] + [fd['edc_b1b2']]
            dd['b1a2iar'] = dd['b1a2iar'] + [fd['b1a2iar']]
            dd['b2a2iar'] = dd['b2a2iar'] + [fd['b2a2iar']]
            dd['sin'] = dd['sin'] + [fd['sine']]
            dd['cos'] = dd['cos'] + [fd['cosine']]
            dd['ir_b1b2'] = dd['ir_b1b2'] + [fd['ir_b1b2']]
            dd['ir_b2b1'] = dd['ir_b2b1'] + [fd['ir_b2b1']]

        if featureset > 2:
            dd['b1w'] = dd['b1w'] + [fd['b1w']]
            dd['b1h'] = dd['b1h'] + [fd['b1h']]
            dd['b2w'] = dd['b2w'] + [fd['b2w']]
            dd['b2h'] = dd['b2h'] + [fd['b2h']]
            dd['im_w'] = dd['im_w'] + [im_w]
            dd['im_h'] = dd['im_h'] + [im_h]
            dd['b1_ar'] = dd['b1_ar'] + [fd['b1_ar']]
            dd['b2_ar'] = dd['b2_ar'] + [fd['b2_ar']]
            dd['iou'] = dd['iou'] + [fd['iou']]

        if featureset > 3:
            dd['hde2iwr'] = dd['hde2iwr'] + [fd['hde2iwr']]
            dd['vde2ihr'] = dd['vde2ihr'] + [fd['vde2ihr']]
            dd['hdc2iwr'] = dd['hdc2iwr'] + [fd['hdc2iwr']]
            dd['vdc2ihr'] = dd['vdc2ihr'] + [fd['vdc2ihr']]
            dd['edc2wr'] = dd['edc2wr'] + [fd['edc2wr']]
            dd['edc2hr'] = dd['edc2hr'] + [fd['edc2hr']]
            dd['hd_b1c_2_b2ne_r'] = dd['hd_b1c_2_b2ne_r'] + [fd['hd_b1c_2_b2ne_r']]
            dd['hd_b2c_2_b1ne_r'] = dd['hd_b2c_2_b1ne_r'] + [fd['hd_b2c_2_b1ne_r']]
            dd['vd_b1c_2_b2ne_r'] = dd['vd_b1c_2_b2ne_r'] + [fd['vd_b1c_2_b2ne_r']]
            dd['vd_b2c_2_b1ne_r'] = dd['vd_b2c_2_b1ne_r'] + [fd['vd_b2c_2_b1ne_r']]


    #
    # for the features whose values were not scalar, restructure the
    # temporary lists into matrices and load the columns of these
    # matrices into the data dictionary for the appropriate keys
    #

    matrix = np.array(obj1_label_bohvs)
    keybase = 'b1cv_'
    for idx in range(nr_object_types):
        key = keybase + str(idx).zfill(3)
        dd[key] = matrix[:,idx]

    # nb: we presume the bboxes are already in FRCNN format
    matrix = np.array(obj1_bboxes)
    dd['b1_xmin'] = matrix[:,0]
    dd['b1_ymin'] = matrix[:,1]
    dd['b1_xmax'] = matrix[:,2]
    dd['b1_ymax'] = matrix[:,3]

    if featureset > 2:
        matrix = np.array(b1_centroids)
        dd['b1c_x'] = matrix[:,0]
        dd['b1c_y'] = matrix[:,0]

    matrix = np.array(obj2_label_bohvs)
    keybase = 'b2cv_'
    for idx in range(nr_object_types):
        key = keybase + str(idx).zfill(3)
        dd[key] = matrix[:,idx]

    # nb: we presume the bboxes are already in FRCNN format
    matrix = np.array(obj2_bboxes)
    dd['b2_xmin'] = matrix[:,0]
    dd['b2_ymin'] = matrix[:,1]
    dd['b2_xmax'] = matrix[:,2]
    dd['b2_ymax'] = matrix[:,3]

    if featureset > 2:
        matrix = np.array(b2_centroids)
        dd['b2c_x'] = matrix[:,0]
        dd['b2c_y'] = matrix[:,0]

    # convert the ppnn input data dictionary to a Pandas DataFrame
    df = pd.DataFrame(dd)

    # convert the DataFrame to a PyTorch Tensor
    # (nb: something in this operation, either the df.values or the 
    #  dtype=float32, undoes the rounding that had previously been done
    #  when the ppnn input data was first prepared and stored in JSON)
    data_tensor = torch.tensor(df.values, dtype=torch.float32)

    #return df, data_tensor
    return data_tensor
