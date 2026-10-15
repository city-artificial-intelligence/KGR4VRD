#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
Utility functions for PPNN prediction selection.
'''

#%%

import numpy as np


#%% function to convert predicted predicates into predicted VRs

def predicates_2_vrs(predicted_predicates, confidence_scores, 
                     bb1, bb2, bb1_label, bb2_label, 
                     nopred_prediction_policy, no_predicate_idx):
    
    '''
    Convert the predicates predicted for a given ordered pair of objects
    into a corresponding set of predicted visual relationships (VRs).
    
    In the process, apply the designated policy for handling predictions of
    'no predicate', per dimension D_predNoPred.
      
    Parameters
    ----------
    predicted_predicates : list of integers
        A list of predicted and selected predicates (represented by their 
        integer indices (aka labels)). The indices (predicate labels) are
        sorted in descending order of confidence of prediction; so the
        1st element of the list is the most confidently predicted predicate, 
        the 2nd element is the next-most confidently predicted predicate, etc.
    
    confidence_scores : list
        A list of confidence scores (probabilities) predicted for the 
        predicates in the list of predicted_predicates. The list is sorted
        in descending order of confidence score.
    
    bb1, bb2 : lists
        Bounding box specifications for an ordered pair of objects
    
    bb1_label, bb2_label : integers
        Integer object class labels for an ordered pair of objects
    
    nopred_prediction_policy : integer
        Specifies the policy for handling 'no predicate' predictions. This
        corresponds directly to experiment space prediction region 
        dimension D_predNoPred
    
    no_predicate_idx : integer or None
        The index (label) of the PPNN output neuron that corresponds to a 
        'no predicate' predicate prediction.  (If nopred_prediction_policy==1,
        this parameter is ignored, and should have value None.)
    
    Returns
    -------
    predicted_vrs_for_object_pair : list of dictionaries
    
    
    Policies for handling predictions of the 'no predicate' predicate
    -----------------------------------------------------------------
    
    we specify here the policies that have been identified and implemented
    for handling predictions of the 'no predicate' predicate for a given
    ordered pair of objects in an image; we describe here the 'what' of each
    policy, and the code (below) provides the 'how'
    
    first we specify a feature that is shared by all of our 'no predicate'
    handling policies; we state it here, once, rather than repeat it in each
    policy definition:
    * we never produce a predicted VR for the 'no predicate' predicate; doing
      so would be a nonsense and a waste, as such a predicted VR could never 
      be a 'hit' (ie never match a ground-truth VR); worse, such a predicted
      VR could potentially crowd-out a viable predicted VR that has a real 
      chance of becoming a 'hit'    
    
    nopred_prediction_policy == 1:
    * predictions of 'no predicate' are made implicitly only, by no predicates
      having been predicted;
    
    nopred_prediction_policy == 2:
    * if proper predicates have been predicted along with the 'no predicate'
      predicate, accept those that have been predicted with greater confidence 
      than the 'no predicate' predicate and convert them into predicted VRs; 
      but ignore the other proper predicates that were predicted with less 
      confidence than the 'no predicate' predicate by not converting them
      into predicted VRs
    * note: with this policy, a prediction of the 'no predicate' predicate
      has a medium effect: some predicted proper predicates graduate to
      becoming predicted VRs but others do not
    
    nopred_prediction_policy == 3:
    * if proper predicates have been predicted along with the 'no predicate'
      predicate, accept all of them and convert all of them into predicted 
      VRs, regardless of how the confidence of their predictions compares with 
      the confidence with which the 'no predicate' predicate was predicted
    * note: with this policy, a prediction of the 'no predicate' predicate 
      has the weakest effect (virtually none); the only potential effect is 
      to (sometimes, perhaps) crowd-out the selection of some other predicted 
      proper predicate; (this is an underlying side-effect common to all
      of our 'no predicate' policies)

    nopred_prediction_policy == 4:
    * if proper predicates have been predicted along with the 'no predicate'
      predicate, ignore all of them and do not convert any into predicted VRs
    * note: with this policy, a prediction of the 'no predicate' predicate 
      has the strongest effect: we do not create ANY predicted VRs for the
      associated ordered pair of objects
          
    ------
    
    additional notes relevant to 'how' these policies can/should be
    implemented in code ...
 
    in the list of selected, predicted predicates 'predicted_predicates',
    the 'no predicate' predicate may or may not appear; further, if it
    does appear, it can appear anywhere in the list: at the front, in 
    the middle, or at the end; we can't make any assumptions about whether
    or where the 'no predicate' predicate is present in the list
    
    the predicates in the list 'predicted_predicates' are ordered in 
    descending order of prediction confidence; that is, the first predicate
    has the largest predicted probability; the second predicate has the
    next largest predicted probability; etc.; we can exploit this or allow 
    for this when implementing our 'no predicate' prediction handling 
    policies in code
    '''
    
    predicted_vrs_for_object_pair = []
    
    if nopred_prediction_policy == 4:
        if no_predicate_idx in predicted_predicates:
            return predicted_vrs_for_object_pair
    
    for idx, predicate in enumerate(predicted_predicates):
        
        if nopred_prediction_policy == 2:
            if int(predicate) == int(no_predicate_idx):
                # we 'break' so that predicates predicted with less confidence
                # than the 'no predicate' predicate (ie those that appear in
                # the list 'after' the 'no predicate' predicate) are ignored 
                # and are not converted into predicted VRs
                break

        if nopred_prediction_policy == 3:
            if int(predicate) == int(no_predicate_idx):
                # we 'continue' so that predicates predicted with less 
                # confidence than the 'no predicate' predicate (ie those that 
                # appear in the list 'after' the 'no predicate' predicate) 
                # are not ignored and are converted into predicted VRs
                continue
        
        # assemble a predicted visual relationship (VR) in required format
        vr = {}
        vr['predicate'] = int(predicate)
        subj = {}
        subj['category'] = bb1_label
        subj['bbox'] = bb1
        vr['subject'] = subj
        obj = {}
        obj['category'] = bb2_label
        obj['bbox'] = bb2
        vr['object'] = obj
        conf_score = float(confidence_scores[idx])
        vr['confidence'] = round(conf_score, 12)
        
        predicted_vrs_for_object_pair.append(vr)    
    
    return predicted_vrs_for_object_pair


#%% function for selecting predicate predictions from amongst PPNN output logits

def select_predictions(ppnn_output_per_image, 
                       tr_d_model2, 
                       prediction_region_config,
                       n_images_per_epoch):
    
    '''
    From the predicate probabilities output by a PPNN during inference,
    select those predicates that we deem to have been 'predicted'; then 
    transform these into a set of 'predicted visual relationships'.
    Do this for all the images in a given dataset.
    
    Parameters
    ----------
    ppnn_output_per_image : dictionary
        A dictionary holding, for each image (usually of the test set)
        the ppnn output generated by inference for that image, and the
        set of object ordered pairs for that image (to which the ppnn
        output relates).
            
        'img_name' : { 'ppnn_output_probs': output_probs,
                       'inference_loss': mb_loss,
                       'ppnn_object_pairs': object_pairs }
            
        The predicate probabilities are the tensor created from the
        tensor of logits output by a PPNN after applying a Sigmoid
        function to transform the logits into probabilities.

        Each probability represents a PPNN's opinion as to the 
        likelihood that a particular NeSy4VRD predicate should be used
        to describe a visual relationship between a particular
        ordered pair of objects in an image.
            
        The value object_pairs is also a dictionary, with structure
            
        '(obj1_bbox, obj2_bbox)': { 'b1_lab': obj1_label
                                    'b2_lab': obj2_label }

        Crucially, there is 1-to-1 positional correspondence between the 
        object ordered pair entries in dictionary object_pairs and
        the rows of the predicate probabilities tensor output_probs.  
        This positional correspondence is what enables us to interpret a row
        of predicate probabilities in tensor output_probs: it links the 
        row of predicate probabilities to a specific ordered pair of objects.
        This 1-to-1 positional correspondence is thus what enables us to 
        build our 'predicted VRs': it tells us what predicates have been
        predicted for which ordered pair of objects.
    
    prediction_region_config : dictionary
        Contains the specification of a particular prediction region of the
        experiment space. A prediction region is described by values (levels)
        for 4 different dimensions of the experiment space.  These settings
        control how logits output by a PPNN model are converted into
        'predicted predicates', and how 'predicted predicates' become
        'selected predicted predicates', and how 'selected predicted 
        predicates' are converted into 'predicted VRs'.
    
    Returns
    -------
    results : dictionary
        A dictionary mapping image names to sets of predicted visual
        relationships for the image.
            
        The structure of the dictionary is:
            
        results[img_name] = { 'n_predicted_predicates': nppfi,
                              'predicted_vrs': predicted_vrs_for_image } 
        
        The value of the 'n_predicted_predicates' key, nppfi, is a scalar
        integer representing the number of predicted predicates for the
        image.  (This is generally different from the number predicted VRs, 
        as not all predicted predicates are 'selected' for conversion into
        predicted VRs.)
        
        The value of the 'predicted_vrs' key, predicted_vrs_for_image, is
        a list of dictionaries. Each dictionary describes a predicted visual
        relationship for the image. The structure of the dictionary is very
        like the native VRD dictionary format for storing annotated visual
        relationships. But there are two key differences to be aware of:
        1) the format used for bboxes is the FRCNN format, 
        [xmin, ymin, xmax, ymax], rather than the VRD format,
        [ymin, ymax, xmin, xmax], and
        2) each dictionary describing a visual relationship contains an
        additional key:value pair recording the confidence of the predicted
        visual relationship (a real number in [0, 1] representing the 
        predicted likelihood or probability of the VR).
        
        More specifically, the structure of the predicted_vrs_for_image
        dictionary is as follows:
        
        {"img_1": [{"predicate": 1, 
                    "object": {"category": 58, "bbox": [347, 482, 571, 595]}, 
                    "subject": {"category": 0, "bbox": [213, 313, 263, 415]}, 
                    "confidence": 0.843}, 
                   {"predicate": 5, 
                    "object": {"category": 58, "bbox": [337, 460, 551, 592]}, 
                    "subject": {"category": 0, "bbox": [272, 213, 562, 890]}, 
                    "confidence": 0.927}, 
                   ... ],
         "img_2": [{"predicate": 1,
                    "object": {"category": 58, "bbox": [347, 482, 571, 595]},
                    "subject": {"category": 0, "bbox": [213, 313, 263, 415]},
                    "confidence": 0.988},
                   ... ],
          ...
         }
    
    '''
    
    # unpack the prediction region configuration
    pred_conf_thresh = prediction_region_config['pred_conf_thresh']
    max_preds_per_obj_pair = prediction_region_config['max_preds_per_obj_pair']
    nopred_prediction_policy = prediction_region_config['nopred_prediction_policy']    
    
    # verify the specified policy for handling predictions of 'no predicate'
    if not nopred_prediction_policy in [1, 2, 3, 4]:
        raise ValueError('nopred_prediction_policy {nopred_prediction_policy} not recognised')
    if tr_d_model2 == 1 and nopred_prediction_policy != 1:
        raise ValueError('level combination for dimensions D_model2 and D_predNoPred is invalid (1)')
    if tr_d_model2 == 2 and nopred_prediction_policy == 1:
        raise ValueError('level combination for dimensions D_model2 and D_predNoPred is invalid (2)')
    
    # initialise counter
    cnt = 0

    # initialise a dictionary to store all of the predicted visual
    # relationships for each image in the dataset
    results = {}

    for img_name, ppnn_data in ppnn_output_per_image.items():
    
        # get the predicate probabilities for the current image
        predicate_probs = ppnn_data['ppnn_output_probs']
        
        # get the ordered pairs of objects for the current image
        object_pairs = ppnn_data['ppnn_object_pairs']
        
        # ensure the sizes are compatible; the number of ordered pairs of
        # objects must equal the number of rows in the tensor of
        # predicate probabilities since we must have 1-to-1 positional
        # correspondence between these two
        predicate_probs_np = np.array(predicate_probs)
        if not len(object_pairs) == predicate_probs_np.shape[0]:
            raise ValueError('Problem: size mismatch')
        
        # initialise list for storing the number of predicates predicted
        # for each ordered pair of objects associated with the current image
        n_predicted_predicates_for_image = []
        
        # initialise list for storing the predicted visual relationships
        # for the current image
        predicted_vrs_for_image = []
        
        # iterate over the ordered pairs of objects for the current image;
        # for each ordered pair of objects decide which predicates have
        # been predicted, select the ones we want, and transform them 
        # into predicted visual relationships
        for idx, kvpair in enumerate(object_pairs.items()):
            
            key = kvpair[0]
            value = kvpair[1]
            
            # get the bboxes for the current ordered pair of objects
            #
            # in the current key, the ordered pair of bboxes is a string
            # with format: '((xmin,ymin,xmax,ymax), (xmin,ymin,xmax,ymax))';
            # convert this string into two lists of integers, one for
            # the 'subject' bbox (bb1), one for the 'object' bbox (bb2)
            #
            # (note that the bboxes here are already in the FRCNN format,
            #  which is (xmin, ymin, xmax, ymax), so no bbox format
            #  conversion is required)
            bb_ord_pair_tuple = eval(key)
            bb1 = list(bb_ord_pair_tuple[0])
            bb2 = list(bb_ord_pair_tuple[1])            
            
            # get the bbox (integer) class labels for the current ordered
            # pair of objects
            bb1_label = value['b1_lab']
            bb2_label = value['b2_lab']
            
            # get the corresponding predicate probabilities that have been
            # predicted for the current ordered pair of objects
            pred_probs = np.array(predicate_probs[idx])
            
            # sort the predicate probabilities for the current object pair
            # in ascending order and get their indices sorted correspondingly; 
            # these indices correspond to (represent) predicate labels; the
            # predicates with the highest predicted probabilities will be at
            # the end of this sorted vector; so we can pick them off by
            # starting from the end of the vector and working backwards
            pred_probs_indices_sorted = np.argsort(pred_probs)

            # identify the predicates whose predicted probabilities exceed
            # the confidence threshold set in the current prediction
            # region configuration specified for our experiment space; 
            # for the current prediction region configuration, these are
            # the predicates deemed to have been 'predicted'
            #
            # (NOTE: this is dimension D_predConf in action)
            predicted_mask = pred_probs > pred_conf_thresh
           
            # get and save the number of predicates deemed to have been 
            # predicted for the current ordered pair of objects
            # (nb: preserving and reporting this info may be useful,
            # eg by helping to tune experiment space hyperparameters)
            n_predicted_predicates = np.sum(predicted_mask)
            n_predicted_predicates_for_image.append(n_predicted_predicates)
            
            #
            # from amongst the predicates deemed to have been 'predicted'
            # by the trained PPNN (per the current prediction region config), 
            # we now select those that have been most confidently
            # predicated, up to the maximum number allowed by the level 
            # (value) of dimension D_predMax, which is what variable
            # 'max_preds_per_obj_pair' contains
            #            

            # nb: the end of the array of sorted indices points to the predicate
            # with the largest predicted probability; starting at the end
            # of this array and working backwards, we select the predicates
            # with the largest predicted probabilities (the most confident
            # predictions), up to the maximum number of predicted predicates 
            # allowed by the experiment space config parameter 
            # 'max_preds_per_obj_pair' (ie we select up to the top-N predicted
            # predicates)
            #
            # (nb: the predicates that are selected and stored in the list 
            #  selected_predicted_predicates have an important characteristic
            #  that we can exploit when converting selected predicates to
            #  selected VRs: the predicates are listed in descending order
            #  of prediction confidence; that is, the first predicate in
            #  the list has the largest predicted probability; the second
            #  predicate in the list has the next highest predicted
            #  probability; etc.;)
            #  
            # (nb: one of these top-N predicted predicates could be the
            #  'no predicate' predicate; the index of the 'no predicate'
            #  predicate can appear anywhere in the array of sorted predicate
            #  indices 'pred_probs_indices_sorted', depending on the
            #  confidence with which it was predicted relative to the other
            #  predicates; therefore, if it has been predicted, it may be 
            #  selected, and may appear anywhere in the list of selected
            #  predicted predicates, 'selected_predicted_predicates': at the
            #  front, in the middle, or at the end; we can't make any
            #  assumptions about whether or where it may appear in the list
            #  selected_predicted_predicates; this is important to be aware
            #  of when implementing our 'no predicate' prediction handling
            #  policies, in function 'predicates_2_vrs()')
            #

            if n_predicted_predicates > 0:
                
                # initialise lists to hold selections
                selected_predicted_predicates = []
                selected_predicted_predicates_conf_scores = []
                
                # select the predicates with the strongest prediction
                # confidence, up to the maximum number allowed
                for idx2 in range(n_predicted_predicates):
                    if idx2 > max_preds_per_obj_pair - 1:
                        break
                    idx3 = -1 - idx2
                    predicate_idx = pred_probs_indices_sorted[idx3]
                    if predicted_mask[predicate_idx]:
                        selected_predicted_predicates.append(predicate_idx)
                        selected_predicted_predicates_conf_scores.append(pred_probs[predicate_idx])                    
                
                #for idx2 in range(max_preds_per_obj_pair):
                #    idx3 = -1 - idx2
                #    predicate_idx = pred_probs_indices_sorted[idx3]
                #    if predicted_mask[predicate_idx]:
                #        selected_predicted_predicates.append(predicate_idx)
                #        selected_predicted_predicates_conf_scores.append(pred_probs[predicate_idx])
                
                # set the index of the 'no predicate' predicate, if there is one    
                if tr_d_model2 == 1:
                    no_predicate_idx = None
                else:
                    no_predicate_idx = len(pred_probs) - 1

                # convert the selected predicted predicates for the current 
                # ordered pair of objects into a set of predicted visual 
                # relationships (predicted VRs)              
                res = predicates_2_vrs(selected_predicted_predicates,
                                       selected_predicted_predicates_conf_scores,
                                       bb1, bb2, bb1_label, bb2_label,
                                       nopred_prediction_policy, no_predicate_idx)
                
                predicted_vrs_for_object_pair = res
                
            else:
                
                # if no predicates have been predicted for the current
                # ordered pair of objects, there is nothing to select or to
                # convert into predicted VRs.  So we assign an outcome of
                # an empty list.
                #
                # If tr_d_model2 == 1, this scenario corresponds to an 
                # *implicit* prediction of 'no predicate'. In this case,
                # the only valid value for nopred_prediction_policy is 1. And
                # policy 1 is straight forward: when no predicate is 
                # predicted, we don't create any predicted VRs. So our 
                # simple assignment of an empty list also doubles as our
                # implementation of policy nopred_prediction_policy==1,
                # which corresponds to level D_predNoPred_1. This
                # implementation of policy is so subtle we take care here
                # to point it out.
                #
                # If tr_d_model2 == 2, this scenario corresponds to an
                # *implicit* prediction of 'no predicate' (because no proper
                # predicate was predicted) coupled with the absence of an
                # *explicit* prediction of 'no predicate' (which might have
                # been made, but wasn't). This is a somewhat contradictory
                # scenario. It would be ideal if the two means for
                # predicting 'no predicate' reinforced one another. But
                # this scenario is a curiosity only, not a problem. 
                # Regardless of whether nopred_prediction_policy==2, 3 or 4,
                # this assignment of an empty list is the correct (only)
                # course of action in this scenario.
        
                predicted_vrs_for_object_pair = []

                 
            # concatenate the list of predicted VRs for the current ordered
            # pair of objects with the accumulating list of predicted VRs
            # for the current image as a whole
            predicted_vrs_for_image = predicted_vrs_for_image + \
                                      predicted_vrs_for_object_pair

        
       
        # store the predicted visual relationships for the current image
        # in the master results dictionary
        # (nb: the number of predicted predicates for an image may be 
        #  different (ie larger) than the number of predicted VRs, due to
        #  the parameter controlled prediction selection process)
        nppfi = np.sum(n_predicted_predicates_for_image)
        nppfi = int(nppfi)
        results[img_name] = { 'n_predicted_predicates': nppfi,
                              'predicted_vrs': predicted_vrs_for_image }
               
        # count the images processed
        cnt += 1
        #if cnt % 50 == 0:
        #    print(f'processed image {cnt}: {imname}')
        
        # if we're in test mode and processing a fixed number of image
        # entries, then check if it's time to stop
        if n_images_per_epoch > 0:
            if cnt >= n_images_per_epoch:
                break 
    
    return results


