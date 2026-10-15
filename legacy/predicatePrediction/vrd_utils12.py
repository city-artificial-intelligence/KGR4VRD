#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
Utility functions for evaluating VR detection predictive performance.

Performance is calculated for a given value of topN (eg 25, 50, 100).

Scores are computed for 3 metrics:
* global recall@N
* mean per image recall@N
* mean avg_recall@K_topN
'''

#%%

import numpy as np
import pandas as pd

import vrd_utils17 as vrdu17


#%%

def build_predicted_vrs_dataframe(predicted_vrs, maxN, indices_of_invalid_pvrs):
    
    # initialise lists to hold the predicted VR data for an image
    slabel = []
    plabel = []
    olabel = []
    sx1 = []
    sy1 = []
    sx2 = []
    sy2 = []
    ox1 = []
    oy1 = []
    ox2 = []
    oy2 = []
    score = [] 
    pvrIdx = []   # the indices of predicted VRs wrt the list of predicted VRs

    # transfer the predicted VR data for the current image to the lists
    # (note how this transfer presumes that the bbox format is FRCNN)  
    #
    # nb: the list of indices of invalid predicted VRs is empty unless
    # we are doing KG filtering to ensure invalid predicted VRs are 
    # filtered out and never submitted for performance evaluation
    
    for pvr_idx, pvr in enumerate(predicted_vrs):
        if not pvr_idx in indices_of_invalid_pvrs: 
            slabel.append(pvr['subject']['category'])
            plabel.append(pvr['predicate'])
            olabel.append(pvr['object']['category'])
            sx1.append(pvr['subject']['bbox'][0])
            sy1.append(pvr['subject']['bbox'][1])
            sx2.append(pvr['subject']['bbox'][2])
            sy2.append(pvr['subject']['bbox'][3])
            ox1.append(pvr['object']['bbox'][0])
            oy1.append(pvr['object']['bbox'][1])
            ox2.append(pvr['object']['bbox'][2])
            oy2.append(pvr['object']['bbox'][3])
            score.append(pvr['confidence'])
            pvrIdx.append(pvr_idx)

    # put the predicted VR data for the current image into a dataframe 
    dfPI = pd.DataFrame({'slb': slabel,       
                         'plb': plabel,       
                         'olb': olabel,       
                         'sx1': sx1,          
                         'sy1': sy1,          
                         'sx2': sx2,          
                         'sy2': sy2,          
                         'ox1': ox1,          
                         'oy1': oy1,          
                         'ox2': ox2,          
                         'oy2': oy2,          
                         'score': score,
                         'vrIdx': pvrIdx})     
    
    # sort the predicted VRs for the current image by the confidence
    # scores of the predictions, in descending order
    dfPI = dfPI.sort_values(by='score', ascending=False)

    # keep (up to) the topN predicted VRs for the current image, and
    # discard the rest; many images will have fewer than topN predicted VRs,
    # in which case all of them will be retained
    #
    # (note: chopping the size of the dfPI dataframe has implications for
    #  determining the nr of predicted VRs submitted per image; the new size  
    #  (number of rows) of the dfPI dataframe gives the number of predicted 
    #  VRs actually submitted for performance evaluation; this number may well
    #  be less than the number of predicted VRs available
    dfPI = dfPI.iloc[0:maxN]

    # add the column 'hit' initialised to all (integer) zeros; this will be
    # used to record which predicted VRs were found to be 'hits' (True 
    # Positives) that successfully predicted (matched with) a particular 
    # ground-truth VR
    dfPI['hit'] = 0

    # add the column 'iou' initialised to all (real) zeros; this will be used 
    # to record the IoU that the bboxes of a predicted VR marked as a 'hit'
    # was found to have achieved with the bboxes of the gt VR with which it
    # was matched (ie which it was deemed to have hit)
    dfPI['iou'] = 0.0

    
    return dfPI


#%% 

def build_groundtruth_vrs_dataframe(gt_vrs, convert_gt_bbox_format):
    
    # initialise lists to hold the predicted VR data for an image
    slabel = []
    plabel = []
    olabel = []
    sx1 = []
    sy1 = []
    sx2 = []
    sy2 = []
    ox1 = []
    oy1 = []
    ox2 = []
    oy2 = []   

    # transfer the predicted VR data for the current image to the lists
    # (note how this transfer presumes that the bbox format is FRCNN)    
    for vr in gt_vrs:
        slabel.append(vr['subject']['category'])
        plabel.append(vr['predicate'])
        olabel.append(vr['object']['category'])
        if convert_gt_bbox_format:  
            # convert from VRD bbox format to FRCNN bbox format
            sx1.append(vr['subject']['bbox'][2])
            sy1.append(vr['subject']['bbox'][0])
            sx2.append(vr['subject']['bbox'][3])
            sy2.append(vr['subject']['bbox'][1])
            ox1.append(vr['object']['bbox'][2])
            oy1.append(vr['object']['bbox'][0])
            ox2.append(vr['object']['bbox'][3])
            oy2.append(vr['object']['bbox'][1])            
        else: 
            # bbox format is already FRCNN, so no conversion required
            sx1.append(vr['subject']['bbox'][0])
            sy1.append(vr['subject']['bbox'][1])
            sx2.append(vr['subject']['bbox'][2])
            sy2.append(vr['subject']['bbox'][3])
            ox1.append(vr['object']['bbox'][0])
            oy1.append(vr['object']['bbox'][1])
            ox2.append(vr['object']['bbox'][2])
            oy2.append(vr['object']['bbox'][3])

    # put the gt VR data for the current image into a dataframe 
    dfGI = pd.DataFrame({'slb': slabel,       
                         'plb': plabel,       
                         'olb': olabel,       
                         'sx1': sx1,          
                         'sy1': sy1,          
                         'sx2': sx2,          
                         'sy2': sy2,          
                         'ox1': ox1,          
                         'oy1': oy1,          
                         'ox2': ox2,          
                         'oy2': oy2})             
    
    # add the column 'hit' initialised to all (integer) zeros; this will be
    # used to record which ground-truth VRs were found to have been 'hit by'
    # (matched with) a predicted VR for the same image
    dfGI['hit'] = 0
    
    
    return dfGI


#%% 

def perform_matching(dfPI, dfGI):
    '''
    Match predicted VRs with annotated (ground-truth) VRs, and track the hits.
    
    This VR matching algorithm is our Python implementation of the matching
    algorithm coded in MATLAB by the originators of the VRD dataset, per
    Lu et al. (2016).

    Parameters
    ----------
    dfPI : Pandas dataframe
        Predicted VRs for a given image
    dfGI : Pandas dataframe
        Ground-truth VRs for a given image 

    Returns
    -------
    dfPI : Pandas dataframe
        An updated dataframe of predicted VRs
    dfGI : Pandas dataframe
        An updated dataframe of ground-truth VRs
    
    Note: The dataframes are mutable and hence are updated in-place, but
    we explicitly return them nonetheless to make what's happening clear.
    '''
    
    # the IoU threshold for determining matching bboxes
    # (this is default setting that should stay fixed)
    iou_thresh = 0.5
    
    # initialise lists for tracking predicted VR hits and IoU measures
    n_pred_vrs = dfPI.shape[0]
    pred_hits = [0] * n_pred_vrs
    pred_iou = [0.0] * n_pred_vrs

    # initialise list for tracking gt VR hits
    n_gt_vrs = dfGI.shape[0]
    gt_hits = [0] * n_gt_vrs

    # iterate over the predicted VRs
    for idxp in range(n_pred_vrs):
        
        # get the (subject, predicate, object) class labels for the
        # current predicted VR
        p_slabel = dfPI.iloc[idxp]['slb']
        p_plabel = dfPI.iloc[idxp]['plb']
        p_olabel = dfPI.iloc[idxp]['olb']
        
        # get the subject bbox for the current predicted VR
        p_sx1 = dfPI.iloc[idxp]['sx1']
        p_sy1 = dfPI.iloc[idxp]['sy1']      
        p_sx2 = dfPI.iloc[idxp]['sx2']        
        p_sy2 = dfPI.iloc[idxp]['sy2']      
        
        # get the object bbox for the current predicted VR
        p_ox1 = dfPI.iloc[idxp]['ox1']
        p_oy1 = dfPI.iloc[idxp]['oy1']      
        p_ox2 = dfPI.iloc[idxp]['ox2']        
        p_oy2 = dfPI.iloc[idxp]['oy2']        

        # initialise our variable for tracking the index of the gt VR 
        # found to have the best overall bbox IoU (Intersection-over-Union)
        # with the current predicted VR
        best_idx = -1
        
        # initialise our variable for tracking the value of the best bbox IoU
        best_iou = -1
             
        # iterate over the gt VRs for the current image and look for the
        # 'best hit' with respect to the current predicted VR
        for idxg in range(n_gt_vrs):

            # ----- check if the class labels match -----

            # if the (s, p, o) class labels of the current predicted VR
            # don't match those of the current gt VR, the current gt VR
            # cannot be a hit, so skip it and consider the next gt VR
            if p_slabel == dfGI.iloc[idxg]['slb'] and \
               p_plabel == dfGI.iloc[idxg]['plb'] and \
               p_olabel == dfGI.iloc[idxg]['olb']:
                pass
            else:
                continue

            # ----- check if the gt VR has already been hit -----

            # if the current gt VR has already been hit, skip it and
            # consider the next gt VR; i.e. a gt VR can be hit at 
            # most once
            if gt_hits[idxg] > 0:
                continue
            
            # ----- check if the subject and object bboxes match -----

            # calculate the width & height of the intersection of
            # the subject bboxes of the two VRs
            s_max_x1 = np.max([p_sx1, dfGI.iloc[idxg]['sx1']])
            s_max_y1 = np.max([p_sy1, dfGI.iloc[idxg]['sy1']])            
            s_min_x2 = np.min([p_sx2, dfGI.iloc[idxg]['sx2']])
            s_min_y2 = np.min([p_sy2, dfGI.iloc[idxg]['sy2']])
            s_intersect_width = s_min_x2 - s_max_x1 + 1
            s_intersect_height = s_min_y2 - s_max_y1 + 1

            # calculate the width & height of the intersection of
            # the object bboxes of the two VRs
            o_max_x1 = np.max([p_ox1, dfGI.iloc[idxg]['ox1']])
            o_max_y1 = np.max([p_oy1, dfGI.iloc[idxg]['oy1']])            
            o_min_x2 = np.min([p_ox2, dfGI.iloc[idxg]['ox2']])
            o_min_y2 = np.min([p_oy2, dfGI.iloc[idxg]['oy2']])
            o_intersect_width = o_min_x2 - o_max_x1 + 1
            o_intersect_height = o_min_y2 - o_max_y1 + 1

            # if the subject bboxes of the two VRs intersect, and the object 
            # bboxes of the two VRs also intersect, then we potentially
            # have a hit; otherwise, we're done with the current gt VR
            if s_intersect_width > 0 and s_intersect_height > 0 and \
               o_intersect_width > 0 and o_intersect_height > 0:
                pass
            else:
                continue
            
            # calculate the intersection area of the subject bbox pair
            s_intersect_area = s_intersect_width * s_intersect_height
            
            # calculate the union area of the subject bbox pair
            p_s_width = p_sx2 - p_sx1 + 1
            p_s_height = p_sy2 - p_sy1 + 1
            p_s_area = p_s_width * p_s_height
            g_s_width = dfGI.iloc[idxg]['sx2'] - dfGI.iloc[idxg]['sx1'] + 1
            g_s_height = dfGI.iloc[idxg]['sy2'] - dfGI.iloc[idxg]['sy1'] + 1
            g_s_area = g_s_width * g_s_height
            s_union_area = p_s_area + g_s_area - s_intersect_area
            
            # calculate the IoU of the subject bbox pair
            s_iou = s_intersect_area / s_union_area
 
            # calculate the intersection area of the object bbox pair
            o_intersect_area = o_intersect_width * o_intersect_height           
 
            # calculate the union area of object bbox pair
            p_o_width = p_ox2 - p_ox1 + 1
            p_o_height = p_oy2 - p_oy1 + 1
            p_o_area = p_o_width * p_o_height
            g_o_width = dfGI.iloc[idxg]['ox2'] - dfGI.iloc[idxg]['ox1'] + 1
            g_o_height = dfGI.iloc[idxg]['oy2'] - dfGI.iloc[idxg]['oy1'] + 1
            g_o_area = g_o_width * g_o_height
            o_union_area = p_o_area + g_o_area - o_intersect_area            
            
            # calculate the IoU of the object bbox pair                
            o_iou = o_intersect_area / o_union_area
            
            # get the smaller of the subject bbox IoU and object bbox IoU
            iou = np.min([s_iou, o_iou])

            # if the smaller of the two IoUs exceeds the IoU threshold,
            # then both of them do; so the current gt VR is a potential hit
            if iou > iou_thresh:
                # if the smaller of the two IoUs exceeds the best IoU found
                # so far (amongst the gt VRs for the current image), then
                # the current gt VR is the 'best candidate hit' found so far,
                # so keep track of it
                if iou > best_iou: 
                    best_iou = iou 
                    best_idx = idxg
                    
        # if we found a gt VR 'best hit' for the current predicted VR, then
        # record this fact; mark the predicted VR as being a hit (a True
        # Positive), and mark the gt VR as having been hit (so it can't
        # be hit again)             
        if best_idx >= 0:
            pred_hits[idxp] = 1
            pred_iou[idxp] = np.round(best_iou,4)
            gt_hits[best_idx] = 1

    # transfer the information about which predicted VRs for the current
    # image are 'hits' (True Positives) back to the predicted VR global 
    # dataframe, dfPG; (this information will be further processed later)
    dfPI['hit'] = pred_hits
    
    # transfer the information about the bbox IoU achieved between the
    # bboxes of the predicted VRs that were 'hits' and the bboxes of 
    # the gt VRs they hit (ie with which they were matched); (this 
    # information may be of interest during qualitative predictive
    # performance evaluation)
    dfPI['iou'] = pred_iou
        
    # transfer the information about which gt VRs for the current image
    # were 'hit' back to the gt VR global dataframe, dfGG; (this 
    # information isn't required for computing measures of our metrics,
    # but it will have great value in terms of facilitating qualitative
    # predictive performance evaluation) 
    dfGI['hit'] = gt_hits
    
    # return the updated predicted VR and ground-truth VR dataframes
    return dfPI, dfGI


#%%

def capture_predicted_vr_outcomes(predicted_vrs, dfPI, topN, rrc_id):
    '''
    Extend the predicted VRs with outcomes relating to performance
    evaluation at the given value of topN.

    Parameters
    ----------
    predicted_vrs : list of dictionaries
        Each dictionary in the list is predicted VR. We call these the
        'available' predicated VRs.
    dfPI : Pandas dataframe
        The subset of available predicted VRs that were submitted for performance
        evaluation. Each predicted VR in this dataframe is identified by the
        index of its position in the list of available predicted VRs, in
        list 'predicted_vrs'.
    topN : integer
        The value of N in recall@N.

    Returns
    -------
    We don't return anything explicitly, but we do extend (update) the
    parameter 'predicted_vrs' (a list of dictionaries) in-place. These
    lists are in turn part of enclosing dictionaries. The implicit update
    is deliberate and essential --- it's what we're here to do. But it's
    applied subtly, so we point it out.
    '''
    
    # prepare the dictionary keys we'll use for storing stats
    rrc_attr = rrc_id  # eg rrc01, rrc02, ...
    submitted_attr = 'submitted_topN_' + str(topN).zfill(3)     
    hit_attr = 'hit_topN_' + str(topN).zfill(3)
        
    for idx, vr in enumerate(predicted_vrs):
        
        mask = dfPI['vrIdx'] == idx 
        if sum(mask) > 1:
            raise ValueError('unexpected number of VR idx instances')
            
        # was the predicted VR submitted for performance evaluation ?
        if sum(mask) == 1:
            submitted = 1
            # was the predicted VR a hit ?
            if dfPI['hit'][mask].item() == 1:
                hit = 1
            else:
                hit = 0
        else:
            submitted = 0
            hit = 0
        
        if rrc_attr in vr:
            pass
        else:
            vr[rrc_attr] = {}
        
        vr[rrc_attr][submitted_attr] = submitted 

        vr[rrc_attr][hit_attr] = hit 
    
    return None


#%%

def calculate_image_level_metrics(dfPI, dfGI):
    
    nr_gt_vrs = dfGI.shape[0]      

    #
    # compute per image recall@N
    #
    
    if nr_gt_vrs > 0:
        nr_hits = dfGI['hit'].sum()    
        per_image_recallN = nr_hits / nr_gt_vrs
        per_image_recallN = np.round(per_image_recallN, 5)       
    else:
        # return NaNs so the values will be excluded when means are calculated
        nr_hits = np.nan
        per_image_recallN = np.nan
       
    #
    # compute per image average recall@k top-N
    #
    
    nr_pred_vrs = dfPI.shape[0] # nr of predicted VRs submitted for perf. eval.
    
    if nr_pred_vrs > 0:      
        if nr_gt_vrs > 0:           
            # calculate recall@k for each rank k
            hit_cumsum = dfPI['hit'].cumsum()
            recall_at_k = hit_cumsum / nr_gt_vrs
   
            # get the subrange of values of recall_at_k over which we wish
            # to take the average
            if nr_gt_vrs <= nr_pred_vrs:
                idx1 = nr_gt_vrs - 1
                idx2 = nr_pred_vrs
            else:
                idx1 = nr_pred_vrs - 1
                idx2 = nr_pred_vrs
            
            recall_at_k_subrange = recall_at_k[idx1:idx2]
            avg_recall_at_k = np.mean(recall_at_k_subrange)
            avg_recall_at_k = np.round(avg_recall_at_k, 5) 
        else:
            # return NaN so the value will be excluded when means are calculated
            avg_recall_at_k = np.nan
    else:
        avg_recall_at_k = 0.0 
    
    
    #
    # compute AP (Average Precision)
    #
    
    if nr_pred_vrs > 0:      
        if nr_gt_vrs > 0:
            
            # calculate recall@k for each rank k
            hit_cumsum = dfPI['hit'].cumsum().to_numpy()
            #print(f'hit_cumsum shape: {hit_cumsum.shape}')
            
            recall_at_k = hit_cumsum / nr_gt_vrs
            #print(f'recall_at_k shape: {recall_at_k.shape}')
            
            # calculate the discrete first difference of the consecutive elements
            recall_at_k_delta = np.diff(recall_at_k, n=1)
            recall_at_k_delta = np.insert(recall_at_k_delta, 0, recall_at_k[0])
            #print(f'recall_at_k_delta shape: {recall_at_k_delta.shape}')
            
            # calculate precision@k for each rank k
            seq_of_k_values = np.array(range(1,hit_cumsum.size+1))
            #print(f'seq_of_k_values shape: {seq_of_k_values.shape}')
        
            # compute precision at each rank k
            precision_at_k = hit_cumsum / seq_of_k_values
            #print(f'precision_at_k shape: {precision_at_k.shape}')
            
            # calculate average precision
            avg_precision = np.dot(precision_at_k, recall_at_k_delta)
            avg_precision = np.round(avg_precision, 5)
    
        else:
            # return NaN so the value will be excluded when means are calculated
            avg_precision = np.nan
    else:
        avg_precision = 0.0 
    
    
    # if nr_gt_vrs is 0, return NaN so that when the mean nr_gt_vrs is 
    # calculated the NaN values will be excluded (whereas 0s would be included)
    if nr_gt_vrs == 0:
        nr_gt_vrs = np.nan
    
    #
    # package the results
    #
    
    results = {}   
    results['nr_gt_vrs'] = nr_gt_vrs 
    results['nr_pred_vrs'] = nr_pred_vrs 
    results['nr_hits'] = nr_hits
    results['per_image_recall@N'] = per_image_recallN
    results['avg recall@k'] = avg_recall_at_k
    
    results['avg_precision'] = avg_precision

    return results


#%%

def evaluate_performance_original(predicted_vrs_per_image,
                         vrd_img_anno,
                         topN,
                         convert_gt_bbox_format,
                         n_images_per_epoch):
    
    raise Exception('Function not in use')
    
    # initialise lists for holding measures for image-level metrics
    # so we can later compute overall means
    nr_gt_vrs_per_image = []
    nr_pred_preds_per_image = []
    nr_pred_vrs_per_image = []
    nr_hits_per_image = []
    recallN_per_image = []
    avg_recall_at_k_per_image = []
    
    # set the number of decimals for rounding the results scores
    ndecimal = 1
    
    # iterate over the entries: the predicted VRs for each image
    for idx, entry in enumerate(predicted_vrs_per_image.items()):
        
        # if we're in testing mode, check when to stop processing
        if n_images_per_epoch > 0:
            if idx+1 > n_images_per_epoch:
                break
        
        imname = entry[0]
        predicted_vr_dict = entry[1]
        
        n_predicted_predicates = predicted_vr_dict['n_predicted_predicates']
        predicted_vrs = predicted_vr_dict['predicted_vrs']
                
        # get the GT VR annotations for the current image
        gt_vrs = vrd_img_anno[imname]
        
        # put the predicted VRs for the current image into a dataframe
        dfPI = build_predicted_vrs_dataframe(predicted_vrs, topN)
        
        # put the ground-truth VRs for the current image into a dataframe
        dfGI = build_groundtruth_vrs_dataframe(gt_vrs, convert_gt_bbox_format)        
        
        # match the predicted VRs with the ground-truth VRs and mark the hits
        dfPI, dfGI = perform_matching(dfPI, dfGI)
        
        # calculate measures for the image-level recall@N-based metrics
        results = calculate_image_level_metrics(dfPI, dfGI)
        
        # preserve the measures of our image_level metrics
        nr_gt_vrs_per_image.append(results['nr_gt_vrs'])
        nr_pred_preds_per_image.append(n_predicted_predicates)
        nr_pred_vrs_per_image.append(results['nr_pred_vrs'])
        nr_hits_per_image.append(results['nr_hits'])
        recallN_per_image.append(results['per_image_recall@N'])
        avg_recall_at_k_per_image.append(results['avg recall@k']) 


    # compute the overall measure of metric 'global recall@N'
    global_recallN = sum(nr_hits_per_image) / sum(nr_gt_vrs_per_image)
    global_recallN = np.round(100 * global_recallN, ndecimal)
    
    # compute the overall measure of metric 'mean per image recall@N'
    mean_per_image_recallN = np.round(100 * np.mean(recallN_per_image), ndecimal)
    
    # compute the overall measure of metric 'mean avg recall@k top-N'
    mean_avg_recallK_topN = np.round(100 * np.mean(avg_recall_at_k_per_image), ndecimal)
    
    # compute the mean number of ground-truth VRs per image
    mean_gt_vrs_per_img = np.round(np.mean(nr_gt_vrs_per_image), ndecimal)
    
    # compute the mean number of predicted predicates per image
    mean_pred_preds_per_img = np.round(np.mean(nr_pred_preds_per_image), ndecimal)
    
    # compute the mean number of predicted VRs per image
    mean_pred_vrs_per_img = np.round(np.mean(nr_pred_vrs_per_image), ndecimal)
    
    # assemble the overall measures of our metrics into a dictionary
    performance_results = {}
    performance_results['topN'] = topN
    performance_results['mean_gt_vrs_per_img'] = mean_gt_vrs_per_img
    performance_results['mean_pred_preds_per_img'] = mean_pred_preds_per_img
    performance_results['mean_pred_vrs_per_img'] = mean_pred_vrs_per_img
    performance_results['global_recallN'] = global_recallN
    performance_results['mean_per_image_recallN'] = mean_per_image_recallN
    performance_results['mean_avg_recallK_topN'] = mean_avg_recallK_topN
    
    return performance_results


#%%

def evaluate_performance(predicted_vrs_per_image,
                         vrd_img_anno,
                         topN,
                         convert_gt_bbox_format,
                         n_images_per_epoch,
                         prediction_region_config,
                         rrc_id,
                         entity_seq_num,
                         ontoClassNames,
                         ontoPropNames,
                         master_vr_type_tensor):
    '''
    Evaluate performance wrt 3 recall@N-based metrics for a given topN.

    Parameters
    ----------
    predicted_vrs_per_image : TYPE
        DESCRIPTION.
    vrd_img_anno : TYPE
        DESCRIPTION.
    topN : TYPE
        DESCRIPTION.
    convert_gt_bbox_format : TYPE
        DESCRIPTION.
    n_images_per_epoch : TYPE
        DESCRIPTION.

    Returns
    -------
    performance_results : dictionary
        Summary (mean) per-image statistics and recall@N scores.
    
    NOTE: Parameter predicted_vrs_per_image is a dictionary of dictionaries 
    and lists, all of which are mutable; we extend the predicted VRs
    available per image to capture new statistics to enable later analyses
    and visualisations. These updates are applied in-place. We don't return
    the updated predicted_vrs_per_image explicitly, but we expect the 
    caller to save the extended data to disk during test set inference,
    but not during validation inference.  Thus, when this function is called
    during validation inference, the extended data is still added to the
    predicted VRs dictionary, but since these data are never saved to disk,
    the in-place updates have no effect, which is what we want. Thus, in
    effect, tduring validation inference, the value of the rrc_id parameter
    does not really matter. It could be a dummy value and it wouldn't matter.
    During validation inference, we just need some value for rrc_id to be
    passed in, even if it's a hardcoded value.
    '''
    
    # get the level of the experiment space dimension (D_predKG) that tells
    # us whether or not to apply KG filtering to the predicted VRs in order
    # to filter out semantically invalid VRs from submission from the
    # NeSy system (ie and hence from submission for performance evaluation)
    #
    # nb: the level of D_predKG is rendered here as a boolean: True or False
    kg_filtering = prediction_region_config['kg_filtering']
    
    n_kg_calls = 0
    
    
    # initialise lists for holding measures for image-level metrics
    # so we can later compute overall means
    nr_gt_vrs_per_image = []
    nr_pred_preds_per_image = []
    nr_pred_vrs_per_image = []    # nr of pred VRs submitted for perf. eval. per image
    nr_pred_vrs_avail_per_image = []   # nr of predicted VRs available per image
    nr_hits_per_image = []
    recallN_per_image = []
    avg_recall_at_k_per_image = []
    avg_precision_per_image = []

    # prepare the dictionary keys we'll use for storing recall stats    
    recall_rrc_attr = 'recall_' + rrc_id
    recallN_attr = 'recallN_topN_' + str(topN).zfill(3) 
    avg_recallK_attr = 'avg_recall_at_k_topN_' + str(topN).zfill(3)            
     
    # prepare the dictionary keys we'll use for storing precision stats    
    precision_rrc_attr = 'avg_precision_' + rrc_id
    avg_precision_attr = 'avg_precision_topN_' + str(topN).zfill(3) 

    # set the number of decimals for rounding the results scores
    ndecimal = 1
    
    # iterate over the entries in the image annotations dictionary
    for idx, entry in enumerate(vrd_img_anno.items()):
        
        # if we're in testing mode, check when to stop processing
        if n_images_per_epoch > 0:
            if idx+1 > n_images_per_epoch:
                break
        
        # unpack the dictionary entry
        imname = entry[0]  # image name
        gt_vrs = entry[1]  # annotated (ground-truth) VRs
        
        if imname in predicted_vrs_per_image:
            
            predicted_vr_dict = predicted_vrs_per_image[imname]

            nr_predicted_predicates = predicted_vr_dict['n_predicted_predicates']
            predicted_vrs = predicted_vr_dict['predicted_vrs']
            
            # if KG filtering has been requested (per the prc_id config),
            # then perform it as part of filtering out semantically invalid VRs 
            if kg_filtering:
                res = vrdu17.get_indices_of_invalid_pvrs(predicted_vrs,
                                                         entity_seq_num,
                                                         ontoClassNames,
                                                         ontoPropNames,
                                                         master_vr_type_tensor)
                indices_of_invalid_pvrs, n_kg_calls_for_image = res
                n_kg_calls += n_kg_calls_for_image
            else:
                indices_of_invalid_pvrs = []
            
            # put topN predicted VRs for current image into dataframe
            dfPI = build_predicted_vrs_dataframe(predicted_vrs, topN,
                                                 indices_of_invalid_pvrs)
            
            # put the ground-truth VRs for the current image into a dataframe
            dfGI = build_groundtruth_vrs_dataframe(gt_vrs, convert_gt_bbox_format)        
            
            # match the predicted VRs with the ground-truth VRs and mark the hits
            dfPI, dfGI = perform_matching(dfPI, dfGI)
            
            # capture what happens to the predicted VRs available for the
            # current image; for the current topN, record which ones were 
            # submitted for performance evaluation, and which of those 
            # were hits
            #
            # (nb: the predicted_vrs is a list of dictionaries, so it is
            #  mutable; we extend it (update it) in-place; the predicted_vrs 
            #  are also part of an enclosing dictionary, which is implicitly
            #  updated as a whole; this is all deliberate and intended, but
            #  it's a subtle side-effect that must be appreciated)
            capture_predicted_vr_outcomes(predicted_vrs, dfPI, topN, rrc_id)
            
            # calculate measures for the image-level recall@N-based metrics
            results = calculate_image_level_metrics(dfPI, dfGI)
                       
            nr_gt_vrs = results['nr_gt_vrs']
            nr_pred_vrs = results['nr_pred_vrs']  # nr pred VRs submitted for perf. eval.
            nr_pred_vrs_avail = len(predicted_vrs)     # nr pred VRs available
            nr_hits = results['nr_hits']
            per_image_recallN = results['per_image_recall@N']
            avg_recallK = results['avg recall@k']
            avg_precision = results['avg_precision']
            
            # preserve the image-level recall metric scores with the image
            # entry (to be saved back to disk, elsewhere); these metrics
            # are recall@N_m2 (recall per image) and recall@N_m3 (avg 
            # recall@K)
            
            if recall_rrc_attr in predicted_vr_dict:
                pass
            else:
                predicted_vr_dict[recall_rrc_attr] = {}
            
            predicted_vr_dict[recall_rrc_attr][recallN_attr] = per_image_recallN  # recall@N_m2

            predicted_vr_dict[recall_rrc_attr][avg_recallK_attr] = avg_recallK    # recall@N_m3
            
            # preserve the image-level precision metric scores with the
            # image entry (to be saved back to disk, elsewhere)

            if precision_rrc_attr in predicted_vr_dict:
                pass
            else:
                predicted_vr_dict[precision_rrc_attr] = {}
            
            predicted_vr_dict[precision_rrc_attr][avg_precision_attr] = avg_precision
            
            
        else:
            # sometimes an image from the annotations (gt) dictionary will not
            # have a corresponding entry in the predicted VRs dictionary; this
            # happens if there was no PPNN input data for the image, a 
            # scenario that arises when an ODNN detects fewer than 2 objects
            # in an image; a VR needs 2 objects, so if there are fewer than
            # 2 objects in an image, there can be no PPNN input data for
            # that image, and so no predicted VRs for that image; 
            # note that this scenario can only arise in the 'relationship
            # detection' regime of experiments, where PPNN input data is
            # derived from ODNN outputs rather than VR annotations
            
            # note: we set these variables to 0, not np.nan; so when averages
            # are calculated, these 0s will affect the means that are 
            # calculated

            nr_predicted_predicates = 0
            nr_gt_vrs = len(gt_vrs)
            nr_pred_vrs = 0
            nr_pred_vrs_avail = 0
            nr_hits = 0
            per_image_recallN = 0
            avg_recallK = 0
            avg_precision = 0
        
        # gather the scores of our image-level metrics
        nr_gt_vrs_per_image.append(nr_gt_vrs)
        nr_pred_preds_per_image.append(nr_predicted_predicates)
        nr_pred_vrs_per_image.append(nr_pred_vrs)
        nr_pred_vrs_avail_per_image.append(nr_pred_vrs_avail)
        nr_hits_per_image.append(nr_hits)
        recallN_per_image.append(per_image_recallN)
        avg_recall_at_k_per_image.append(avg_recallK)
        avg_precision_per_image.append(avg_precision)
               


    # compute the overall measure of metric 'global recall@N'
    total_gt_vrs = np.nansum(nr_gt_vrs_per_image)
    if total_gt_vrs > 0:
        global_recallN = np.nansum(nr_hits_per_image) / total_gt_vrs
        global_recallN = np.round(100 * global_recallN, ndecimal)
    else:
        global_recallN = 0.0
    
    # compute the overall measure of metric 'mean per image recall@N'
    mean_per_image_recallN = np.round(100 * np.nanmean(recallN_per_image), ndecimal)
    
    # compute the overall measure of metric 'mean avg recall@k top-N'
    mean_avg_recallK_topN = np.round(100 * np.nanmean(avg_recall_at_k_per_image), ndecimal)
    
    # compute mean Average Precision (mAP)
    mean_avg_precision_topN = np.round(100 * np.nanmean(avg_precision_per_image), ndecimal)
    
    # compute the mean number of ground-truth VRs per image
    # (nb: if images have no gt VRs due to filtering, the number is Nan 
    #  rather than 0; and NaNs are excluded from the calculation of the mean;
    #  ie when gt VRs have been filtered, the mean nr_gt_vrs is for those
    #  images that have some gt VRs)
    mean_gt_vrs_per_img = np.round(np.nanmean(nr_gt_vrs_per_image), ndecimal)
    
    # compute the mean number of predicted predicates per image
    mean_pred_preds_per_img = np.round(np.mean(nr_pred_preds_per_image), ndecimal)
    
    # compute the mean number of predicted VRs submitted for perf. eval. per image
    mean_pred_vrs_per_img = np.round(np.mean(nr_pred_vrs_per_image), ndecimal)
    
    # compute the mean number of predicted VRs available per image
    mean_pred_vrs_avail_per_img = np.round(np.mean(nr_pred_vrs_avail_per_image), ndecimal)
    
    # compute the mean number of hits per image
    mean_hits_per_img = np.round(np.nanmean(nr_hits_per_image), ndecimal)
    
    # assemble all the results into a dictionary
    performance_results = {}
    performance_results['topN'] = topN
    performance_results['mean_gt_vrs_per_img'] = mean_gt_vrs_per_img
    performance_results['mean_pred_preds_per_img'] = mean_pred_preds_per_img
    performance_results['mean_pred_vrs_avail_per_img'] = mean_pred_vrs_avail_per_img
    performance_results['mean_pred_vrs_per_img'] = mean_pred_vrs_per_img
    performance_results['mean_hits_per_img'] = mean_hits_per_img
    performance_results['global_recallN'] = global_recallN
    performance_results['mean_per_image_recallN'] = mean_per_image_recallN
    performance_results['mean_avg_recallK_topN'] = mean_avg_recallK_topN
    performance_results['n_kg_calls'] = n_kg_calls 
    
    performance_results['mean_avg_precision'] = mean_avg_precision_topN
    
    
    return performance_results
