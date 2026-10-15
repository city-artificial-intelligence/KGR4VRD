#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module defines functions pertaining to the task of matching two sets
of ordered object pairs associated with a given image.

One set contains the ordered object pairs for a given image found in PPNN 
input data. The other set contains ordered object pairs for the same 
image but where the objects are those referenced in the ground-truth VR 
annotations for the image.


'''


#%%

import numpy as np
import pandas as pd

#%%

def build_ppnn_dataframe(ppnn_img_dict):


    # initialise lists to hold the ppnn input data for the given image
    sx1 = []
    sy1 = []
    sx2 = []
    sy2 = []
    ox1 = []
    oy1 = []
    ox2 = []
    oy2 = []
    slabel = []
    olabel = []

    # get the object ordered pairs dictionary for the current image
    pair_dict = ppnn_img_dict['obj_ord_pairs']

    # iterate over the object ordered pairs for the given image and
    # transfer the bbox and class label features to the lists
    # (nb: the bbox format is assumed to be FRCNN)
    for key, feature_dict in pair_dict.items():

        # in the current key, the ordered pair of bboxes has format
        # 'string': '((x1,y1,x2,y2),(x1,y1,x2,y2))';
        # convert this string back to a 2-tuple of bbox tuples and then
        # isolate the individual bbox tuples
        bb_ord_pair_tuple = eval(key)
        bb1 = bb_ord_pair_tuple[0]
        bb2 = bb_ord_pair_tuple[1]

        # transfer subject bbox to lists
        sx1.append(bb1[0])
        sy1.append(bb1[1])
        sx2.append(bb1[2])
        sy2.append(bb1[3])

        # transfer object bbox to lists
        ox1.append(bb2[0])
        oy1.append(bb2[1])
        ox2.append(bb2[2])
        oy2.append(bb2[3])

        # transfer bbox class labels to lists
        label = feature_dict['b1_lab']
        slabel.append(label)
        label = feature_dict['b2_lab']
        olabel.append(label)

    # store the lists of features in a dataframe
    df_ppnn = pd.DataFrame({'sx1': sx1,      # subject bbox xmin
                            'sy1': sy1,      # subject bbox ymin
                            'sx2': sx2,      # subject bbox xmax
                            'sy2': sy2,      # subject bbox ymax
                            'ox1': ox1,      #  object bbox xmin
                            'oy1': oy1,      #  object bbox ymin
                            'ox2': ox2,      #  object bbox xmax
                            'oy2': oy2,      #  object bbox ymax
                            'slb': slabel,   # 'subject' object class label
                            'olb': olabel})  # 'object' object class label

    # add the column 'hit' initialised to all (integer) zeros; it will be
    # used to record which ppnn input ordered pairs of objects are found
    # to match with (ie to 'hit') target ordered pairs of objects
    df_ppnn['hit'] = 0

    # add the column 'iou' initialised to all (real) zeros; it will be used
    # to record the IoU that ppnn input ordered pairs of objects marked as
    # 'hits' were found to have with the target ordered pairs of objects
    # with which they were deemed to match
    df_ppnn['iou'] = 0.0

    # add the column 'idx' initialised to all -1; it will be used to
    # record the index of the target ordered pair of objects with which
    # the ppnn input ordered pair of objects has been matched (if it has in
    # fact been matched); a -1 value will indicate it has NOT been matched
    df_ppnn['idx'] = -1

    return df_ppnn


#%%

def build_trgt_dataframe(trgt_obj_ord_pairs):

    # Indicate the bbox format used in the target data being processed:
    # True - indicates that the bbox format in the target data is the
    #        standard VRD format [ymin, ymax, xmin, xmax]
    # False - indicates that the bbox format in the target data is the
    #         alternate FRCNN format, [xmin, ymin, xmax, ymax]
    vrd_bbox_format = False


    # initialise lists to hold the ppnn input data for the given image
    sx1 = []
    sy1 = []
    sx2 = []
    sy2 = []
    ox1 = []
    oy1 = []
    ox2 = []
    oy2 = []
    slabel = []
    olabel = []

    if len(trgt_obj_ord_pairs) != 1:
        raise ValueError('Dictionary should have just 1 entry')

    # extract the single key:value pair
    # (we don't know the key, so we do it crudely)
    for imname, imdict in trgt_obj_ord_pairs.items():
        break

    # get the object ordered pairs dictionary for the image
    pair_dict = imdict['obj_ord_pairs']

    # iterate over the object ordered pairs for the given image and
    # transfer the bbox and class label features to the lists
    for key, feature_dict in pair_dict.items():

        # in the current key, the ordered pair of bboxes has format
        # 'string': '((x1,y1,x2,y2),(x1,y1,x2,y2))';
        # convert this string back to a 2-tuple of bbox tuples and then
        # isolate the individual bbox tuples
        bb_ord_pair_tuple = eval(key)
        bb1 = bb_ord_pair_tuple[0]
        bb2 = bb_ord_pair_tuple[1]

        # transfer bbox features to lists
        if vrd_bbox_format:
            sx1.append(bb1[2])
            sy1.append(bb1[0])
            sx2.append(bb1[3])
            sy2.append(bb1[1])
            ox1.append(bb2[2])
            oy1.append(bb2[0])
            ox2.append(bb2[3])
            oy2.append(bb2[1])
        else:
            sx1.append(bb1[0])
            sy1.append(bb1[1])
            sx2.append(bb1[2])
            sy2.append(bb1[3])
            ox1.append(bb2[0])
            oy1.append(bb2[1])
            ox2.append(bb2[2])
            oy2.append(bb2[3])

        # transfer bbox class labels to lists
        label = feature_dict['b1_lab']
        slabel.append(label)
        label = feature_dict['b2_lab']
        olabel.append(label)


    # store the lists of features in a dataframe
    df_trgt = pd.DataFrame({'sx1': sx1,      # subject bbox xmin
                            'sy1': sy1,      # subject bbox ymin
                            'sx2': sx2,      # subject bbox xmax
                            'sy2': sy2,      # subject bbox ymax
                            'ox1': ox1,      #  object bbox xmin
                            'oy1': oy1,      #  object bbox ymin
                            'ox2': ox2,      #  object bbox xmax
                            'oy2': oy2,      #  object bbox ymax
                            'slb': slabel,   # 'subject' object class label
                            'olb': olabel})  # 'object' object class label


    # add the column 'hit' initialised to all (integer) zeros; it will be
    # used to record which ppnn input ordered pairs of objects are found
    # to match with (ie to 'hit') target ordered pairs of objects
    df_trgt['hit'] = 0

    # add the column 'idx' initialised to all -1; it will be used to
    # record the index of the ppnn input ordered pair of objects with which
    # the target ordered pair of objects has been matched (if it has in
    # fact been matched); a -1 value will indicate it has NOT been matched
    df_trgt['idx'] = -1

    return df_trgt


#%%

def match_ppnn_pairs_with_target_pairs(df_ppnn, df_trgt):


    # set the bbox IoU minimum threshold that must be exceeded for
    # an ppnn input ordered pair of objects to be deemed to match with
    # a target ordered pair of objects
    iou_thresh = 0.5

    n_ppnn = df_ppnn.shape[0]
    ppnn_hits = [0] * n_ppnn
    ppnn_hits_trgt_idx = [-1] * n_ppnn
    ppnn_iou = [0.0] * n_ppnn

    n_trgt = df_trgt.shape[0]
    trgt_hits = [0] * n_trgt
    trgt_hits_ppnn_idx = [-1] * n_trgt

    # iterate over the ppnn input ordered pairs of objects
    for idxp in range(n_ppnn):

        # get the subject bbox for the current ppnn ordered pair
        p_sx1 = df_ppnn.iloc[idxp]['sx1']
        p_sy1 = df_ppnn.iloc[idxp]['sy1']
        p_sx2 = df_ppnn.iloc[idxp]['sx2']
        p_sy2 = df_ppnn.iloc[idxp]['sy2']

        # get the object bbox for the current ppnn ordered pair
        p_ox1 = df_ppnn.iloc[idxp]['ox1']
        p_oy1 = df_ppnn.iloc[idxp]['oy1']
        p_ox2 = df_ppnn.iloc[idxp]['ox2']
        p_oy2 = df_ppnn.iloc[idxp]['oy2']

        # get the subject, predicate, object) class labels for the
        p_slabel = df_ppnn.iloc[idxp]['slb']
        p_olabel = df_ppnn.iloc[idxp]['olb']

        # initialise our variable for tracking the index of the target
        # ordered pair of objects found to have the best overall bbox IoU
        # (Intersection-over-Union) with the current ppnn ordered pair
        best_trgt_idx = -1

        # initialise our variable for tracking the value of the best bbox IoU
        best_iou = -1

        # iterate over the target ordered pairs of objects and look for the
        # 'best hit' with respect to the current ppnn ordered pair
        for idxt in range(n_trgt):

            # if the class labels don't match, the current target
            # ordered pair is not a candidate hit; so we're done with it
            if p_slabel == df_trgt.iloc[idxt]['slb'] and \
               p_olabel == df_trgt.iloc[idxt]['olb']:
                pass
            else:
                continue

            # if the current target pair has already been 'hit' (matched),
            # we're done with it; it can't be matched more than once
            if trgt_hits[idxt] > 0:
                continue

            # calculate the width & height of the intersection of
            # the corresponding 'subject' bboxes
            s_max_x1 = np.max([p_sx1, df_trgt.iloc[idxt]['sx1']])
            s_max_y1 = np.max([p_sy1, df_trgt.iloc[idxt]['sy1']])
            s_min_x2 = np.min([p_sx2, df_trgt.iloc[idxt]['sx2']])
            s_min_y2 = np.min([p_sy2, df_trgt.iloc[idxt]['sy2']])
            s_intersect_width = s_min_x2 - s_max_x1 + 1
            s_intersect_height = s_min_y2 - s_max_y1 + 1

            # calculate the width & height of the intersection of
            # the corresponding 'object' bboxes
            o_max_x1 = np.max([p_ox1, df_trgt.iloc[idxt]['ox1']])
            o_max_y1 = np.max([p_oy1, df_trgt.iloc[idxt]['oy1']])
            o_min_x2 = np.min([p_ox2, df_trgt.iloc[idxt]['ox2']])
            o_min_y2 = np.min([p_oy2, df_trgt.iloc[idxt]['oy2']])
            o_intersect_width = o_min_x2 - o_max_x1 + 1
            o_intersect_height = o_min_y2 - o_max_y1 + 1

            # if the two 'subject' bboxes intersect, and the two 'object'
            # bboxes also intersect, then we potentially have a hit;
            # otherwise, we're done with the current target pair
            if s_intersect_width > 0 and s_intersect_height > 0 and \
               o_intersect_width > 0 and o_intersect_height > 0:
                pass
            else:
                continue

            # calculate the intersection area of the 'subject' bboxes
            s_intersect_area = s_intersect_width * s_intersect_height

            # calculate the union area of the 'subject' bboxes
            p_s_width = p_sx2 - p_sx1 + 1
            p_s_height = p_sy2 - p_sy1 + 1
            p_s_area = p_s_width * p_s_height
            g_s_width = df_trgt.iloc[idxt]['sx2'] - df_trgt.iloc[idxt]['sx1'] + 1
            g_s_height = df_trgt.iloc[idxt]['sy2'] - df_trgt.iloc[idxt]['sy1'] + 1
            g_s_area = g_s_width * g_s_height
            s_union_area = p_s_area + g_s_area - s_intersect_area

            # calculate the IoU of the 'subject' bboxes
            s_iou = s_intersect_area / s_union_area

            # calculate the intersection area of the 'object' bboxes
            o_intersect_area = o_intersect_width * o_intersect_height

            # calculate the union area of 'object' bboxes
            p_o_width = p_ox2 - p_ox1 + 1
            p_o_height = p_oy2 - p_oy1 + 1
            p_o_area = p_o_width * p_o_height
            g_o_width = df_trgt.iloc[idxt]['ox2'] - df_trgt.iloc[idxt]['ox1'] + 1
            g_o_height = df_trgt.iloc[idxt]['oy2'] - df_trgt.iloc[idxt]['oy1'] + 1
            g_o_area = g_o_width * g_o_height
            o_union_area = p_o_area + g_o_area - o_intersect_area

            # calculate the IoU of the 'object' bboxes
            o_iou = o_intersect_area / o_union_area

            # get the smaller of the 'subject' bbox and 'object' bbox IoUs
            iou = np.min([s_iou, o_iou])

            # if the smaller of the two IoUs exceeds the IoU threshold,
            # then both of them do; so the current target ordered pair of
            # objects is a 'candidate hit'
            if iou > iou_thresh:
                # if the smaller of the two IoUs exceeds the best IoU found
                # so far, then the current target ordered pair of objects
                # is the 'best candidate hit' found so far; so keep track
                # of it
                if iou > best_iou:
                    best_iou = iou
                    best_trgt_idx = idxt

        # if we found a match (best hit) amongst the target ordered pairs
        # for the current ppnn ordered pair of objects, then record this fact;
        # mark the ppnn ordered pair as 'being a hit', and mark the target
        # ordered pair as 'having been hit' (so it can't be hit again)
        if best_trgt_idx >= 0:
            ppnn_hits[idxp] = 1
            ppnn_hits_trgt_idx[idxp] = best_trgt_idx
            ppnn_iou[idxp] = np.round(best_iou,4)
            trgt_hits[best_trgt_idx] = 1
            trgt_hits_ppnn_idx[best_trgt_idx] = idxp

    # transfer the info on the matching back into the ppnn dataframe
    df_ppnn['hit'] = ppnn_hits
    df_ppnn['iou'] = ppnn_iou
    df_ppnn['idx'] = ppnn_hits_trgt_idx

    # transfer the info on the matching back into the trgt dataframe
    df_trgt['hit'] = trgt_hits
    df_trgt['idx'] = trgt_hits_ppnn_idx

    # NOTE: the dataframes are mutable, so they will be updated in-place and
    # the results of the matching communicated to the caller that way

    return None


#%%

# note: if zero ppnn input ordered pairs of objects have been matched
# with target ordered pairs of objects, the resulting set of target
# vectors (set B) will contain only 'no predicate' target vectors

def assemble_target_vector_set_B(df_ppnn, df_trgt, targets_A):

    # initialise a set of all-zeros target vectors, the shape of which
    # has the same number of rows as the ppnn input data
    n_rows = df_ppnn.shape[0]
    n_cols = targets_A.shape[1]
    targets_B = np.zeros((n_rows, n_cols))

    # We initialise the target vectors in set B to be 'no predicate' 
    # target vectors. This is because most of the possible ordered pairs
    # of objects for an image will NOT have associated annotated visual 
    # relationships, hence the majority of target vectors will usually be
    # 'no predicate' target vectors. 
    targets_B[:, -1] = 1.0

    # In set B, align (positionally) *matched* target vectors with their
    # matching ppnn input ordered pair of objects (and, by extension, with
    # the corresponding row of the output tensor); the rows of the
    # output and target tensors must always correspond to the same
    # ordered pair of objects
    #
    # Iterate over the ppnn input ordered pairs of objects and, for each one
    # that was *matched* with a target ordered pair of objects, copy the
    # matching target vector from set A into the index position (row) in
    # target vector set B that is aligned with the matching ppnn input
    # ordered pair of objects.

    # Note: targets_B is initialised with all 'no predicate' target vectors.
    # Some initial targets_B target vectors will be overwritten by a matched
    # target vector from targets_A. We call these *matched* target vectors.
    # These target vectors may be 'proper predicate' or 'no predicate'
    # target vectors. The initial targets_B target vectors that are NOT
    # overwritten by matched target vectors from targets_A are *unmatched*
    # target vectors. These target vectors are all 'no predicate' target
    # vectors.
    #
    # When considering 'no predicate' target vectors, we wish to distinguish
    # between *matched* 'no predicate' target vectors and *unmatched*
    # 'no predicate' target vectors.  Our method for doing so involves use
    # of the 'no predicate' element of the 'no predicate' target vectors 
    # in targets_B.
    # Normally, the value of the 'no predicate' element in a 'no predicate'
    # target vector would be 1.0 only. But we introduce a 2nd special value
    # of 2.0.  Thus, a value of 1.0 identifies an *unmatched* 'no predicate'
    # target vector; a value of 2.0 identifies a *matched* 'no predicate'
    # target vector.
    # (Note: as part of preparing for the calculation of loss, the values of
    #  2.0 are changed back to 1.0; so the special values of 2.0 are
    #  temporary only; once the information they convey has been utilised,
    #  the values are changed to 1.0)

    # Note: a *matched* target vector may be either a 'proper predicate'
    # target vector or a 'no predicate' target vector.

    # CAUTION: testing code involving variable 'hits' is kept around but commented-out
    
    #hits = False
    for idxp in range(n_rows):
        if df_ppnn.loc[idxp, 'hit'] == 1:     # ppnn ordered pair of objects was matched.
            idxt = df_ppnn.loc[idxp, 'idx']   # get idx of matched target vector.
            targets_B[idxp] = targets_A[idxt] # copy matched target vector into place.
            if targets_B[idxp, -1] == 1.0: # a *matched* 'no predicate' target vector,
                targets_B[idxp, -1] = 2.0  # so mark it with special value 2.0
    #       hits = True

    # no 'hits' could indicate a problem if it happens consistently
    #if not hits:   
    #    print('vrd_ppnn_matching: no hits from matching')

    return targets_B


#%%

def create_target_vectors_B(ppnn_img_dict, trgt_obj_ord_pairs, targets_A):

    df_ppnn = build_ppnn_dataframe(ppnn_img_dict)

    df_trgt = build_trgt_dataframe(trgt_obj_ord_pairs)


    # perform matching
    # (the results of the matching are stored in-place, within the
    #  respective dataframes)
    match_ppnn_pairs_with_target_pairs(df_ppnn, df_trgt)

    targets_B = assemble_target_vector_set_B(df_ppnn, df_trgt, targets_A)
    
    # test-mode
    #n_rows = int(targets_B.shape[0])
    #n_nopred = int(targets_B[:,-1].sum())
    #if n_nopred == n_rows:
    #    print('vrd_ppnn_matching: all targets are nopred targets')

    return targets_B
