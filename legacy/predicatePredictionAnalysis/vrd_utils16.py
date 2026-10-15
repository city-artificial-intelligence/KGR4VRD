#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module defines utility functions that provide services relating to 
the analysis of predicted visual relationships (predicted VRs).
'''

#%%

import numpy as np
import itertools
import pandas as pd


#%%

def analyse_predicted_vr_symmetric_pairs(predicted_vrs, 
                                         nesy4vrd_symmetric_property_indices,
                                         rrc_id):
    '''
    Measure the extent to which the predicted VRs for a given image occur
    in symmetric pairs for those predicates deemed to be symmetric.
    '''
    
    # get the predicted VRs that use a symmetric predicate
    symmetric_prop_pvrs = []
    for pvr_idx, pvr in enumerate(predicted_vrs):
        if pvr['predicate'] in nesy4vrd_symmetric_property_indices:
            symmetric_prop_pvrs.append((pvr_idx, pvr['predicate']))

    # the number of symmetric property predicted VRs
    n_sp_pvrs = len(symmetric_prop_pvrs)

    # extract symmetric property predicted VR indices
    symmetric_prop_pvrs_indices = [ entry[0] for entry in symmetric_prop_pvrs ]
    
    # form all possible (unordered) pairs of the predicted VRs that have 
    # symmetric predicates
    possible_symmetric_pairs = itertools.combinations(symmetric_prop_pvrs, r=2)
    possible_symmetric_pairs = list(possible_symmetric_pairs)
    
    # within the set of possible pairs, identify the set of symmetric pairs
    symmetric_pairs = []
    for pair in possible_symmetric_pairs:        
        entry1 = pair[0]  # eg (3,18)  (pvr_idx, predicate_idx)
        entry1_pvr_idx = entry1[0]
        entry1_pred_idx = entry1[1]
        entry2 = pair[1]  # eg (5,18)  (pvr_idx, predicate_idx)
        entry2_pvr_idx = entry2[0]
        entry2_pred_idx = entry2[1]        
        if entry1_pred_idx == entry2_pred_idx: 
            # the two pvrs share the same predicate; so they may be a symmetric pair
            pvr1 = predicted_vrs[entry1_pvr_idx]
            pvr2 = predicted_vrs[entry2_pvr_idx]
            if pvr1['subject'] == pvr2['object'] and pvr1['object'] == pvr2['subject']:
                # the current pair of pvrs are a symmetric pair
                symmetric_pairs.append((entry1_pvr_idx, entry2_pvr_idx))
    
    # the number of symmetric property predicted VR symmetric pairs
    n_sp_pvr_pairs = len(symmetric_pairs)
    
    # the number of symmetric property predicted VR singletons
    n_sp_pvr_singletons = n_sp_pvrs - (n_sp_pvr_pairs * 2)
    
    if n_sp_pvrs > 0:
        proportion_sp_pvrs_in_pairs = (n_sp_pvr_pairs * 2) / n_sp_pvrs        
    else:
        proportion_sp_pvrs_in_pairs = np.nan
    
    # (1) count the number of instances where both of the predicted VRs in a 
    # symmetric pair are submitted together for performance evaluation, for
    # topN = 25, 50, 100
    # (2) measure how similar are the confidence scores of the predicted VRs
    # involved in each symmetric pair
        
    symmetric_pairs_submitted_together_topN_025 = [False] * n_sp_pvr_pairs
    symmetric_pairs_submitted_together_topN_050 = [False] * n_sp_pvr_pairs
    symmetric_pairs_submitted_together_topN_100 = [False] * n_sp_pvr_pairs
    symmetric_pairs_submitted_together_topN_999 = [False] * n_sp_pvr_pairs
    confidence_absolute_differences = []
    for idx, pair in enumerate(symmetric_pairs):
        pvr1_idx = pair[0]
        pvr2_idx = pair[1]
        pvr1 = predicted_vrs[pvr1_idx]
        pvr2 = predicted_vrs[pvr2_idx]
        # (1)
        if pvr1[rrc_id]['submitted_topN_025'] == 1 and pvr2[rrc_id]['submitted_topN_025'] == 1:
            symmetric_pairs_submitted_together_topN_025[idx] = True
        if pvr1[rrc_id]['submitted_topN_050'] == 1 and pvr2[rrc_id]['submitted_topN_050'] == 1:
            symmetric_pairs_submitted_together_topN_050[idx] = True
        if pvr1[rrc_id]['submitted_topN_100'] == 1 and pvr2[rrc_id]['submitted_topN_100'] == 1:
            symmetric_pairs_submitted_together_topN_100[idx] = True 
        if pvr1[rrc_id]['submitted_topN_999'] == 1 and pvr2[rrc_id]['submitted_topN_999'] == 1:
            symmetric_pairs_submitted_together_topN_999[idx] = True
        # (2)
        confidence_absolute_diff = abs(pvr1['confidence'] - pvr2['confidence'])
        confidence_absolute_differences.append(confidence_absolute_diff)
    
    n_pairs_submitted_together_topN_025 = sum(symmetric_pairs_submitted_together_topN_025)
    n_pairs_submitted_together_topN_050 = sum(symmetric_pairs_submitted_together_topN_050)    
    n_pairs_submitted_together_topN_100 = sum(symmetric_pairs_submitted_together_topN_100)
    n_pairs_submitted_together_topN_999 = sum(symmetric_pairs_submitted_together_topN_999)
       
    if len(confidence_absolute_differences) > 0:
        mean_pairs_confidence_absolute_diffences = np.mean(confidence_absolute_differences)
    else:
        # we definitely do not want to return 0 in this case; we want 
        # to return NaN; np.mean() returns NaN automatically whenever
        # the list of confidences is empty, but we want to make this
        # explicit by settin Nan explicitly; in the saved .csv file, the NaN 
        # values become missing values (no value); these can be treated as
        # NA when calculating a summary such as a mean, say; we want to 
        # exclude the NAs from the mean, not return 0 and have 0s included
        # in the mean
        mean_pairs_confidence_absolute_diffences = np.nan  # value undefined
        
    # (1) count the number of sp_pvrs that are hits
    # (2) calc the proportion of sp_pvrs that are hits
    # (3) calc the mean confidence of the sp_pvrs
    n_sp_pvrs_hits_topN_025 = 0
    n_sp_pvrs_hits_topN_050 = 0
    n_sp_pvrs_hits_topN_100 = 0
    n_sp_pvrs_hits_topN_999 = 0
    sp_pvr_confidences = []
    for pvr_idx in symmetric_prop_pvrs_indices:
        pvr = predicted_vrs[pvr_idx]
        # (1)
        if pvr[rrc_id]['hit_topN_025'] == 1:
            n_sp_pvrs_hits_topN_025 += 1
        if pvr[rrc_id]['hit_topN_050'] == 1:
            n_sp_pvrs_hits_topN_050 += 1    
        if pvr[rrc_id]['hit_topN_100'] == 1:
            n_sp_pvrs_hits_topN_100 += 1 
        if pvr[rrc_id]['hit_topN_999'] == 1:
            n_sp_pvrs_hits_topN_999 += 1
        # (3)
        sp_pvr_confidences.append(pvr['confidence'])
        
    # (2)
    if n_sp_pvrs > 0:
        proportion_sp_pvrs_hits_topN_025 = n_sp_pvrs_hits_topN_025 / n_sp_pvrs
        proportion_sp_pvrs_hits_topN_050 = n_sp_pvrs_hits_topN_050 / n_sp_pvrs    
        proportion_sp_pvrs_hits_topN_100 = n_sp_pvrs_hits_topN_100 / n_sp_pvrs
        proportion_sp_pvrs_hits_topN_999 = n_sp_pvrs_hits_topN_999 / n_sp_pvrs 
    else:
        proportion_sp_pvrs_hits_topN_025 = np.nan
        proportion_sp_pvrs_hits_topN_050 = np.nan
        proportion_sp_pvrs_hits_topN_100 = np.nan 
        proportion_sp_pvrs_hits_topN_999 = np.nan

    # (3)
    if len(sp_pvr_confidences) > 0:
        mean_sp_pvr_confidence = np.mean(sp_pvr_confidences)
    else:
        mean_sp_pvr_confidence = np.nan
    
    # assemble the results
    results = {}
    results['symmetric_property_predicted_vrs'] = symmetric_prop_pvrs_indices
    results['symmetric_pairs'] = symmetric_pairs
    results['n_sp_pvrs'] = n_sp_pvrs
    results['n_sp_pvr_pairs'] = n_sp_pvr_pairs
    results['n_sp_pvr_singletons'] = n_sp_pvr_singletons
    results['proportion_sp_pvrs_in_pairs'] = proportion_sp_pvrs_in_pairs
    results['n_pairs_subm_together_topN_025'] = n_pairs_submitted_together_topN_025
    results['n_pairs_subm_together_topN_050'] = n_pairs_submitted_together_topN_050
    results['n_pairs_subm_together_topN_100'] = n_pairs_submitted_together_topN_100
    results['n_pairs_subm_together_topN_999'] = n_pairs_submitted_together_topN_999
    results['mean_pairs_confidence_absolute_differences'] = mean_pairs_confidence_absolute_diffences
    results['n_sp_pvrs_hits_topN_025'] = n_sp_pvrs_hits_topN_025
    results['n_sp_pvrs_hits_topN_050'] = n_sp_pvrs_hits_topN_050
    results['n_sp_pvrs_hits_topN_100'] = n_sp_pvrs_hits_topN_100
    results['n_sp_pvrs_hits_topN_999'] = n_sp_pvrs_hits_topN_999
    results['proportion_sp_pvrs_hits_topN_025'] = proportion_sp_pvrs_hits_topN_025
    results['proportion_sp_pvrs_hits_topN_050'] = proportion_sp_pvrs_hits_topN_050
    results['proportion_sp_pvrs_hits_topN_100'] = proportion_sp_pvrs_hits_topN_100
    results['proportion_sp_pvrs_hits_topN_999'] = proportion_sp_pvrs_hits_topN_999
    results['mean_sp_pvr_confidence'] = mean_sp_pvr_confidence
    
    return results 


#%%

def process_predicted_vrs_for_symmetric_pairs(predicted_vrs_per_image,
                                              nesy4vrd_symmetric_property_indices,
                                              rrc_id):

    # initialise lists to hold the image-level metric scores that will become the 
    # columns of our .csv file
    name_per_img = []
    n_sp_pvrs_per_image = []
    n_sp_pvr_pairs_per_image = []
    n_sp_pvr_singletons_per_image = []   
    proportion_sp_pvrs_in_pairs_per_image = []   
    n_pairs_subm_together_topN_025_per_image = []
    n_pairs_subm_together_topN_050_per_image = []
    n_pairs_subm_together_topN_100_per_image = []
    n_pairs_subm_together_topN_999_per_image = []
    mean_pairs_confidence_absolute_differences_per_image = []
    n_sp_pvrs_hits_topN_025_per_image = []
    n_sp_pvrs_hits_topN_050_per_image = []
    n_sp_pvrs_hits_topN_100_per_image = []
    n_sp_pvrs_hits_topN_999_per_image = []
    proportion_sp_pvrs_hits_topN_025_per_image = []
    proportion_sp_pvrs_hits_topN_050_per_image = []
    proportion_sp_pvrs_hits_topN_100_per_image = []
    proportion_sp_pvrs_hits_topN_999_per_image = []
    mean_sp_pvr_confidence_per_image = []
    
    n_decimal = 7

    for img_name, img_pred_vr_info in predicted_vrs_per_image.items():
        
        predicted_vrs = img_pred_vr_info['predicted_vrs']
    
        name_per_img.append(img_name)
    
        results = analyse_predicted_vr_symmetric_pairs(predicted_vrs, 
                                                       nesy4vrd_symmetric_property_indices,
                                                       rrc_id)
    
        n_sp_pvrs_per_image.append(results['n_sp_pvrs'])    
        n_sp_pvr_pairs_per_image.append(results['n_sp_pvr_pairs']) 
        n_sp_pvr_singletons_per_image.append(results['n_sp_pvr_singletons']) 
        prop_sp_pvrs_in_pairs = np.round(results['proportion_sp_pvrs_in_pairs'], n_decimal)
        proportion_sp_pvrs_in_pairs_per_image.append(prop_sp_pvrs_in_pairs)
        n_pairs_subm_together_topN_025_per_image.append(results['n_pairs_subm_together_topN_025']) 
        n_pairs_subm_together_topN_050_per_image.append(results['n_pairs_subm_together_topN_050']) 
        n_pairs_subm_together_topN_100_per_image.append(results['n_pairs_subm_together_topN_100'])
        n_pairs_subm_together_topN_999_per_image.append(results['n_pairs_subm_together_topN_999'])
        mean_pairs_conf_abs_diff = np.round(results['mean_pairs_confidence_absolute_differences'], n_decimal)
        mean_pairs_confidence_absolute_differences_per_image.append(mean_pairs_conf_abs_diff) 
        n_sp_pvrs_hits_topN_025_per_image.append(results['n_sp_pvrs_hits_topN_025']) 
        n_sp_pvrs_hits_topN_050_per_image.append(results['n_sp_pvrs_hits_topN_050']) 
        n_sp_pvrs_hits_topN_100_per_image.append(results['n_sp_pvrs_hits_topN_100']) 
        n_sp_pvrs_hits_topN_999_per_image.append(results['n_sp_pvrs_hits_topN_999'])
        prop_sp_pvrs_hits_topN_025 = np.round(results['proportion_sp_pvrs_hits_topN_025'], n_decimal)
        proportion_sp_pvrs_hits_topN_025_per_image.append(prop_sp_pvrs_hits_topN_025) 
        prop_sp_pvrs_hits_topN_050 = np.round(results['proportion_sp_pvrs_hits_topN_050'], n_decimal)
        proportion_sp_pvrs_hits_topN_050_per_image.append(prop_sp_pvrs_hits_topN_050) 
        prop_sp_pvrs_hits_topN_100 = np.round(results['proportion_sp_pvrs_hits_topN_100'], n_decimal)
        proportion_sp_pvrs_hits_topN_100_per_image.append(prop_sp_pvrs_hits_topN_100) 
        prop_sp_pvrs_hits_topN_999 = np.round(results['proportion_sp_pvrs_hits_topN_999'], n_decimal)
        proportion_sp_pvrs_hits_topN_999_per_image.append(prop_sp_pvrs_hits_topN_999)         
        mean_sp_pvr_conf = np.round(results['mean_sp_pvr_confidence'], n_decimal)
        mean_sp_pvr_confidence_per_image.append(mean_sp_pvr_conf) 

    # assemble the metric scores for each image into a dataframe
    df = pd.DataFrame({'img_name': name_per_img,
                       'n_sp_pvrs': n_sp_pvrs_per_image,
                       'n_sp_pvr_pairs': n_sp_pvr_pairs_per_image,
                       'n_sp_pvr_singletons': n_sp_pvr_singletons_per_image, 
                       'proportion_sp_pvrs_in_pairs': proportion_sp_pvrs_in_pairs_per_image,
                       'n_pairs_subm_together_topN_025': n_pairs_subm_together_topN_025_per_image, 
                       'n_pairs_subm_together_topN_050': n_pairs_subm_together_topN_050_per_image,
                       'n_pairs_subm_together_topN_100': n_pairs_subm_together_topN_100_per_image,
                       'n_pairs_subm_together_topN_999': n_pairs_subm_together_topN_999_per_image,
                       'mean_pairs_confidence_absolute_differences': mean_pairs_confidence_absolute_differences_per_image, 
                       'n_sp_pvrs_hits_topN_025': n_sp_pvrs_hits_topN_025_per_image,
                       'n_sp_pvrs_hits_topN_050': n_sp_pvrs_hits_topN_050_per_image,
                       'n_sp_pvrs_hits_topN_100': n_sp_pvrs_hits_topN_100_per_image,
                       'n_sp_pvrs_hits_topN_999': n_sp_pvrs_hits_topN_999_per_image,
                       'proportion_sp_pvrs_hits_topN_025': proportion_sp_pvrs_hits_topN_025_per_image,
                       'proportion_sp_pvrs_hits_topN_050': proportion_sp_pvrs_hits_topN_050_per_image, 
                       'proportion_sp_pvrs_hits_topN_100': proportion_sp_pvrs_hits_topN_100_per_image,
                       'proportion_sp_pvrs_hits_topN_999': proportion_sp_pvrs_hits_topN_999_per_image,
                       'mean_sp_pvr_confidence': mean_sp_pvr_confidence_per_image
                      })

    return df


#%%

# NOTE: some of the symmetric pair predicted VR metrics are independent of
# topN; for these metrics, it's tempting to take the average of all 12 
# samples in the consolidated results symmetric pairs .cvs file; but we do
# NOT do this; instead, we find the samples that pertain to specific values
# of topN and average only those (4) sample scores; this is consistent with
# what we do elsewhere in R plot visualisations

def summarise_consolidated_sym_pairs_results(df):
    
    topN_values = [25, 50, 100, 999]
    n_decimal = 5
    results = {}
    
    for topN in topN_values:
          
        mask = df['topN'] == topN

        stats = df['n_sp_pvrs'][mask]
        n_sp_pvrs = np.round(np.mean(stats), n_decimal)
    
        stats = df['n_sp_pvr_pairs'][mask]
        n_sp_pvr_pairs = np.round(np.mean(stats), n_decimal)
    
        stats = df['n_sp_pvr_singletons'][mask]
        n_sp_pvr_singletons = np.round(np.mean(stats), n_decimal)
    
        stats = df['proportion_sp_pvrs_in_pairs'][mask]
        proportion_sp_pvrs_in_pairs = np.round(np.mean(stats), n_decimal)
    
        stats = df['mean_sp_pvr_confidence'][mask]
        mean_sp_pvr_confidence = np.round(np.mean(stats), n_decimal)
    
        stats = df['mean_pairs_confidence_absolute_differences'][mask]
        mean_pairs_confidence_absolute_differences = np.round(np.mean(stats), n_decimal)    

        if topN == 25:
            stats = df['n_pairs_subm_together_topN_025'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_sp_pvrs_hits_topN_025'][mask]
            n_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_sp_pvrs_hits_topN_025'][mask]
            proportion_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
        elif topN == 50:
            stats = df['n_pairs_subm_together_topN_050'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_sp_pvrs_hits_topN_050'][mask]
            n_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_sp_pvrs_hits_topN_050'][mask]
            proportion_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
        elif topN == 100:
            stats = df['n_pairs_subm_together_topN_100'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_sp_pvrs_hits_topN_100'][mask]
            n_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_sp_pvrs_hits_topN_100'][mask]
            proportion_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
        elif topN == 999:
            stats = df['n_pairs_subm_together_topN_999'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_sp_pvrs_hits_topN_999'][mask]
            n_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_sp_pvrs_hits_topN_999'][mask]
            proportion_sp_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
        else:
            raise ValueError('topN not recognised')

        topN_str = str(topN).zfill(3)
        result_set = 'topN_' + topN_str
    
        results[result_set] = {
            'n_sp_pvrs': n_sp_pvrs,
            'n_sp_pvr_pairs': n_sp_pvr_pairs,
            'n_sp_pvr_singletons': n_sp_pvr_singletons,
            'proportion_sp_pvrs_in_pairs': proportion_sp_pvrs_in_pairs,
            'mean_sp_pvr_confidence': mean_sp_pvr_confidence,
            'mean_pairs_confidence_absolute_differences': mean_pairs_confidence_absolute_differences,
            'n_pairs_subm_together_topN': n_pairs_subm_together_topN,
            'n_sp_pvrs_hits_topN': n_sp_pvrs_hits_topN,
            'proportion_sp_pvrs_hits_topN': proportion_sp_pvrs_hits_topN,
            }
    
    return results


#%%

def analyse_predicted_vr_inverse_pairs(predicted_vrs, 
                                       nesy4vrd_inverse_properties_1_indices,
                                       nesy4vrd_inverse_properties_2_indices,
                                       rrc_id):
    '''
    Measure the extent to which the predicted VRs for a given image occur
    in symmetric pairs for those predicates deemed to be symmetric.
    '''
    
    # combine the two sets of inverse-pair property (predicate) indices 
    # and eliminate any duplicates
    inverse_properties_indices = nesy4vrd_inverse_properties_1_indices + \
                                 nesy4vrd_inverse_properties_2_indices
    inverse_properties_indices = list(set(inverse_properties_indices)) 
  
    # get the predicted VRs that use a predicate that has an inverse
    inverse_prop_pvrs = []
    for pvr_idx, pvr in enumerate(predicted_vrs):
        if pvr['predicate'] in inverse_properties_indices:
            inverse_prop_pvrs.append((pvr_idx, pvr['predicate']))

    # the number of inverse property predicted VRs
    n_ip_pvrs = len(inverse_prop_pvrs)

    # extract inverse property predicted VR indices
    inverse_prop_pvrs_indices = [ entry[0] for entry in inverse_prop_pvrs ]
    
    # form all possible (unordered) pairs of the predicted VRs that have 
    # a predicate that has an inverse
    possible_inverse_pairs = itertools.combinations(inverse_prop_pvrs, r=2)
    possible_inverse_pairs = list(possible_inverse_pairs)
    
    # within the set of possible pairs, identify the set of inverse pairs
    inverse_pairs = []
    for pair in possible_inverse_pairs:
        
        entry1 = pair[0]  # eg (3,18)  (pvr_idx, predicate_idx)
        entry1_pvr_idx = entry1[0]
        entry1_pred_idx = entry1[1]
        entry2 = pair[1]  # eg (5,18)  (pvr_idx, predicate_idx)
        entry2_pvr_idx = entry2[0]
        entry2_pred_idx = entry2[1] 
        
        inverse_pair = False
        
        if entry1_pred_idx in nesy4vrd_inverse_properties_1_indices:
            pair_idx = nesy4vrd_inverse_properties_1_indices.index(entry1_pred_idx)
            pair_pred_idx = nesy4vrd_inverse_properties_2_indices[pair_idx]
            if entry2_pred_idx == pair_pred_idx:
                # the predicates of the two predicted VRs (pvrs) are inverses of
                # one another; so this pair of pvrs may be an inverse pair; 
                # next we need to check that the two objects are the same
                pvr1 = predicted_vrs[entry1_pvr_idx]
                pvr2 = predicted_vrs[entry2_pvr_idx]
                if pvr1['subject'] == pvr2['object'] and pvr1['object'] == pvr2['subject']:
                    # the current pair of pvrs is an inverse pair
                    inverse_pairs.append((entry1_pvr_idx, entry2_pvr_idx))
                    inverse_pair = True

        if entry1_pred_idx in nesy4vrd_inverse_properties_2_indices and not inverse_pair:
            pair_idx = nesy4vrd_inverse_properties_2_indices.index(entry1_pred_idx)
            pair_pred_idx = nesy4vrd_inverse_properties_1_indices[pair_idx]
            if entry2_pred_idx == pair_pred_idx:
                # the predicates of the two predicted VRs (pvrs) are inverses of
                # one another; so this pair of pvrs may be an inverse pair; 
                # next we need to check that the two objects are the same
                pvr1 = predicted_vrs[entry1_pvr_idx]
                pvr2 = predicted_vrs[entry2_pvr_idx]
                if pvr1['subject'] == pvr2['object'] and pvr1['object'] == pvr2['subject']:
                    # the current pair of pvrs is an inverse pair
                    inverse_pairs.append((entry1_pvr_idx, entry2_pvr_idx))

    # the number of predicted VR inverse pairs
    n_ip_pvr_pairs = len(inverse_pairs)
    
    # the number of predicted VRs with a predicate that has an inverse, but
    # which occur as singletons (outside of a pair)
    n_ip_pvr_singletons = n_ip_pvrs - (n_ip_pvr_pairs * 2)
    
    if n_ip_pvrs > 0:
        proportion_ip_pvrs_in_pairs = (n_ip_pvr_pairs * 2) / n_ip_pvrs        
    else:
        proportion_ip_pvrs_in_pairs = np.nan
    
    # (1) count the number of instances where both of the predicted VRs in an 
    # inverse pair are submitted together for performance evaluation, for
    # topN = 25, 50, 100
    # (2) measure how similar are the confidence scores of the predicted VRs
    # involved in each inverse pair
        
    inverse_pairs_submitted_together_topN_025 = [False] * n_ip_pvr_pairs
    inverse_pairs_submitted_together_topN_050 = [False] * n_ip_pvr_pairs
    inverse_pairs_submitted_together_topN_100 = [False] * n_ip_pvr_pairs
    inverse_pairs_submitted_together_topN_999 = [False] * n_ip_pvr_pairs
    confidence_absolute_differences = []
    for idx, pair in enumerate(inverse_pairs):
        pvr1_idx = pair[0]
        pvr2_idx = pair[1]
        pvr1 = predicted_vrs[pvr1_idx]
        pvr2 = predicted_vrs[pvr2_idx]
        # (1)
        if pvr1[rrc_id]['submitted_topN_025'] == 1 and pvr2[rrc_id]['submitted_topN_025'] == 1:
            inverse_pairs_submitted_together_topN_025[idx] = True
        if pvr1[rrc_id]['submitted_topN_050'] == 1 and pvr2[rrc_id]['submitted_topN_050'] == 1:
            inverse_pairs_submitted_together_topN_050[idx] = True
        if pvr1[rrc_id]['submitted_topN_100'] == 1 and pvr2[rrc_id]['submitted_topN_100'] == 1:
            inverse_pairs_submitted_together_topN_100[idx] = True 
        if pvr1[rrc_id]['submitted_topN_999'] == 1 and pvr2[rrc_id]['submitted_topN_999'] == 1:
            inverse_pairs_submitted_together_topN_999[idx] = True
        # (2)
        confidence_absolute_diff = abs(pvr1['confidence'] - pvr2['confidence'])
        confidence_absolute_differences.append(confidence_absolute_diff)
    
    n_pairs_submitted_together_topN_025 = sum(inverse_pairs_submitted_together_topN_025)
    n_pairs_submitted_together_topN_050 = sum(inverse_pairs_submitted_together_topN_050)    
    n_pairs_submitted_together_topN_100 = sum(inverse_pairs_submitted_together_topN_100)
    n_pairs_submitted_together_topN_999 = sum(inverse_pairs_submitted_together_topN_999)
       
    if len(confidence_absolute_differences) > 0:
        mean_pairs_confidence_absolute_diffences = np.mean(confidence_absolute_differences)
    else:
        # we definitely do not want to return 0 in this case; we want 
        # to return NaN; np.mean() returns NaN automatically whenever
        # the list of confidences is empty, but we want to make this
        # explicit by settin Nan explicitly; in the saved .csv file, the NaN 
        # values become missing values (no value); these can be treated as
        # NA when calculating a summary such as a mean, say; we want to 
        # exclude the NAs from the mean, not return 0 and have 0s included
        # in the mean
        mean_pairs_confidence_absolute_diffences = np.nan  # value undefined
        
    # (1) count the number of ip_pvrs that are hits
    # (2) calc the proportion of ip_pvrs that are hits
    # (3) calc the mean confidence of the ip_pvrs
    n_ip_pvrs_hits_topN_025 = 0
    n_ip_pvrs_hits_topN_050 = 0
    n_ip_pvrs_hits_topN_100 = 0
    n_ip_pvrs_hits_topN_999 = 0
    ip_pvr_confidences = []
    for pvr_idx in inverse_prop_pvrs_indices:
        pvr = predicted_vrs[pvr_idx]
        # (1)
        if pvr[rrc_id]['hit_topN_025'] == 1:
            n_ip_pvrs_hits_topN_025 += 1
        if pvr[rrc_id]['hit_topN_050'] == 1:
            n_ip_pvrs_hits_topN_050 += 1    
        if pvr[rrc_id]['hit_topN_100'] == 1:
            n_ip_pvrs_hits_topN_100 += 1 
        if pvr[rrc_id]['hit_topN_999'] == 1:
            n_ip_pvrs_hits_topN_999 += 1
        # (3)
        ip_pvr_confidences.append(pvr['confidence'])
        
    # (2)
    if n_ip_pvrs > 0:
        proportion_ip_pvrs_hits_topN_025 = n_ip_pvrs_hits_topN_025 / n_ip_pvrs
        proportion_ip_pvrs_hits_topN_050 = n_ip_pvrs_hits_topN_050 / n_ip_pvrs    
        proportion_ip_pvrs_hits_topN_100 = n_ip_pvrs_hits_topN_100 / n_ip_pvrs 
        proportion_ip_pvrs_hits_topN_999 = n_ip_pvrs_hits_topN_999 / n_ip_pvrs 
    else:
        proportion_ip_pvrs_hits_topN_025 = np.nan
        proportion_ip_pvrs_hits_topN_050 = np.nan
        proportion_ip_pvrs_hits_topN_100 = np.nan 
        proportion_ip_pvrs_hits_topN_999 = np.nan

    # (3)
    if len(ip_pvr_confidences) > 0:
        mean_ip_pvr_confidence = np.mean(ip_pvr_confidences)
    else:
        mean_ip_pvr_confidence = np.nan
    
    # assemble the results
    results = {}
    results['inverse_property_predicted_vrs'] = inverse_prop_pvrs_indices
    results['inverse_pairs'] = inverse_pairs
    results['n_ip_pvrs'] = n_ip_pvrs
    results['n_ip_pvr_pairs'] = n_ip_pvr_pairs
    results['n_ip_pvr_singletons'] = n_ip_pvr_singletons
    results['proportion_ip_pvrs_in_pairs'] = proportion_ip_pvrs_in_pairs
    results['n_pairs_subm_together_topN_025'] = n_pairs_submitted_together_topN_025
    results['n_pairs_subm_together_topN_050'] = n_pairs_submitted_together_topN_050
    results['n_pairs_subm_together_topN_100'] = n_pairs_submitted_together_topN_100
    results['n_pairs_subm_together_topN_999'] = n_pairs_submitted_together_topN_999
    results['mean_pairs_confidence_absolute_differences'] = mean_pairs_confidence_absolute_diffences
    results['n_ip_pvrs_hits_topN_025'] = n_ip_pvrs_hits_topN_025
    results['n_ip_pvrs_hits_topN_050'] = n_ip_pvrs_hits_topN_050
    results['n_ip_pvrs_hits_topN_100'] = n_ip_pvrs_hits_topN_100
    results['n_ip_pvrs_hits_topN_999'] = n_ip_pvrs_hits_topN_999
    results['proportion_ip_pvrs_hits_topN_025'] = proportion_ip_pvrs_hits_topN_025
    results['proportion_ip_pvrs_hits_topN_050'] = proportion_ip_pvrs_hits_topN_050
    results['proportion_ip_pvrs_hits_topN_100'] = proportion_ip_pvrs_hits_topN_100
    results['proportion_ip_pvrs_hits_topN_999'] = proportion_ip_pvrs_hits_topN_999
    results['mean_ip_pvr_confidence'] = mean_ip_pvr_confidence
    
    return results 


#%%

def process_predicted_vrs_for_inverse_pairs(predicted_vrs_per_image,
                                            nesy4vrd_inverse_properties_1_indices,
                                            nesy4vrd_inverse_properties_2_indices,
                                            rrc_id):

    # initialise lists to hold the image-level metric scores that will become the 
    # columns of our .csv file
    name_per_img = []
    n_ip_pvrs_per_image = []
    n_ip_pvr_pairs_per_image = []
    n_ip_pvr_singletons_per_image = []   
    proportion_ip_pvrs_in_pairs_per_image = []   
    n_pairs_subm_together_topN_025_per_image = []
    n_pairs_subm_together_topN_050_per_image = []
    n_pairs_subm_together_topN_100_per_image = []
    n_pairs_subm_together_topN_999_per_image = []
    mean_pairs_confidence_absolute_differences_per_image = []
    n_ip_pvrs_hits_topN_025_per_image = []
    n_ip_pvrs_hits_topN_050_per_image = []
    n_ip_pvrs_hits_topN_100_per_image = []
    n_ip_pvrs_hits_topN_999_per_image = []
    proportion_ip_pvrs_hits_topN_025_per_image = []
    proportion_ip_pvrs_hits_topN_050_per_image = []
    proportion_ip_pvrs_hits_topN_100_per_image = []
    proportion_ip_pvrs_hits_topN_999_per_image = []
    mean_ip_pvr_confidence_per_image = []
    
    n_decimal = 7

    for img_name, img_pred_vr_info in predicted_vrs_per_image.items():
        
        predicted_vrs = img_pred_vr_info['predicted_vrs']
    
        name_per_img.append(img_name)
    
        results = analyse_predicted_vr_inverse_pairs(predicted_vrs, 
                                                     nesy4vrd_inverse_properties_1_indices,
                                                     nesy4vrd_inverse_properties_2_indices,
                                                     rrc_id)
    
        n_ip_pvrs_per_image.append(results['n_ip_pvrs'])    
        n_ip_pvr_pairs_per_image.append(results['n_ip_pvr_pairs']) 
        n_ip_pvr_singletons_per_image.append(results['n_ip_pvr_singletons']) 
        prop_ip_pvrs_in_pairs = np.round(results['proportion_ip_pvrs_in_pairs'], n_decimal)
        proportion_ip_pvrs_in_pairs_per_image.append(prop_ip_pvrs_in_pairs)
        n_pairs_subm_together_topN_025_per_image.append(results['n_pairs_subm_together_topN_025']) 
        n_pairs_subm_together_topN_050_per_image.append(results['n_pairs_subm_together_topN_050']) 
        n_pairs_subm_together_topN_100_per_image.append(results['n_pairs_subm_together_topN_100'])
        n_pairs_subm_together_topN_999_per_image.append(results['n_pairs_subm_together_topN_999'])
        mean_pairs_conf_abs_diff = np.round(results['mean_pairs_confidence_absolute_differences'], n_decimal)
        mean_pairs_confidence_absolute_differences_per_image.append(mean_pairs_conf_abs_diff) 
        n_ip_pvrs_hits_topN_025_per_image.append(results['n_ip_pvrs_hits_topN_025']) 
        n_ip_pvrs_hits_topN_050_per_image.append(results['n_ip_pvrs_hits_topN_050']) 
        n_ip_pvrs_hits_topN_100_per_image.append(results['n_ip_pvrs_hits_topN_100']) 
        n_ip_pvrs_hits_topN_999_per_image.append(results['n_ip_pvrs_hits_topN_999']) 
        prop_ip_pvrs_hits_topN_025 = np.round(results['proportion_ip_pvrs_hits_topN_025'], n_decimal)
        proportion_ip_pvrs_hits_topN_025_per_image.append(prop_ip_pvrs_hits_topN_025) 
        prop_ip_pvrs_hits_topN_050 = np.round(results['proportion_ip_pvrs_hits_topN_050'], n_decimal)
        proportion_ip_pvrs_hits_topN_050_per_image.append(prop_ip_pvrs_hits_topN_050) 
        prop_ip_pvrs_hits_topN_100 = np.round(results['proportion_ip_pvrs_hits_topN_100'], n_decimal)
        proportion_ip_pvrs_hits_topN_100_per_image.append(prop_ip_pvrs_hits_topN_100) 
        prop_ip_pvrs_hits_topN_999 = np.round(results['proportion_ip_pvrs_hits_topN_999'], n_decimal)
        proportion_ip_pvrs_hits_topN_999_per_image.append(prop_ip_pvrs_hits_topN_999)
        mean_ip_pvr_conf = np.round(results['mean_ip_pvr_confidence'], n_decimal)
        mean_ip_pvr_confidence_per_image.append(mean_ip_pvr_conf) 

    # assemble the metric scores for each image into a dataframe
    df = pd.DataFrame({'img_name': name_per_img,
                       'n_ip_pvrs': n_ip_pvrs_per_image,
                       'n_ip_pvr_pairs': n_ip_pvr_pairs_per_image,
                       'n_ip_pvr_singletons': n_ip_pvr_singletons_per_image, 
                       'proportion_ip_pvrs_in_pairs': proportion_ip_pvrs_in_pairs_per_image,
                       'n_pairs_subm_together_topN_025': n_pairs_subm_together_topN_025_per_image, 
                       'n_pairs_subm_together_topN_050': n_pairs_subm_together_topN_050_per_image,
                       'n_pairs_subm_together_topN_100': n_pairs_subm_together_topN_100_per_image,
                       'n_pairs_subm_together_topN_999': n_pairs_subm_together_topN_999_per_image,
                       'mean_pairs_confidence_absolute_differences': mean_pairs_confidence_absolute_differences_per_image, 
                       'n_ip_pvrs_hits_topN_025': n_ip_pvrs_hits_topN_025_per_image,
                       'n_ip_pvrs_hits_topN_050': n_ip_pvrs_hits_topN_050_per_image,
                       'n_ip_pvrs_hits_topN_100': n_ip_pvrs_hits_topN_100_per_image,
                       'n_ip_pvrs_hits_topN_999': n_ip_pvrs_hits_topN_999_per_image,
                       'proportion_ip_pvrs_hits_topN_025': proportion_ip_pvrs_hits_topN_025_per_image,
                       'proportion_ip_pvrs_hits_topN_050': proportion_ip_pvrs_hits_topN_050_per_image, 
                       'proportion_ip_pvrs_hits_topN_100': proportion_ip_pvrs_hits_topN_100_per_image,
                       'proportion_ip_pvrs_hits_topN_999': proportion_ip_pvrs_hits_topN_999_per_image,
                       'mean_ip_pvr_confidence': mean_ip_pvr_confidence_per_image
                      })

    return df


#%%

# NOTE: some of the inverse pair predicted VR metrics are independent of
# topN; for these metrics, it's tempting to take the average of all 12 
# samples in the consolidated results inverse pairs .cvs file; but we do
# NOT do this; instead, we find the samples that pertain to specific values
# of topN and average only those (4) sample scores; this is consistent with
# what we do elsewhere in R plot visualisations

def summarise_consolidated_inv_pairs_results(df):
    
    topN_values = [25, 50, 100, 999]
    n_decimal = 5
    results = {}
    
    for topN in topN_values:
          
        mask = df['topN'] == topN

        stats = df['n_ip_pvrs'][mask]
        n_ip_pvrs = np.round(np.mean(stats), n_decimal)
    
        stats = df['n_ip_pvr_pairs'][mask]
        n_ip_pvr_pairs = np.round(np.mean(stats), n_decimal)
    
        stats = df['n_ip_pvr_singletons'][mask]
        n_ip_pvr_singletons = np.round(np.mean(stats), n_decimal)
    
        stats = df['proportion_ip_pvrs_in_pairs'][mask]
        proportion_ip_pvrs_in_pairs = np.round(np.mean(stats), n_decimal)
    
        stats = df['mean_ip_pvr_confidence'][mask]
        mean_ip_pvr_confidence = np.round(np.mean(stats), n_decimal)
    
        stats = df['mean_pairs_confidence_absolute_differences'][mask]
        mean_pairs_confidence_absolute_differences = np.round(np.mean(stats), n_decimal)    

        if topN == 25:
            stats = df['n_pairs_subm_together_topN_025'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_ip_pvrs_hits_topN_025'][mask]
            n_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_ip_pvrs_hits_topN_025'][mask]
            proportion_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
        elif topN == 50:
            stats = df['n_pairs_subm_together_topN_050'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_ip_pvrs_hits_topN_050'][mask]
            n_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_ip_pvrs_hits_topN_050'][mask]
            proportion_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
        elif topN == 100:
            stats = df['n_pairs_subm_together_topN_100'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_ip_pvrs_hits_topN_100'][mask]
            n_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_ip_pvrs_hits_topN_100'][mask]
            proportion_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
        elif topN == 999:
            stats = df['n_pairs_subm_together_topN_999'][mask]
            n_pairs_subm_together_topN = np.round(np.mean(stats), n_decimal)
            stats = df['n_ip_pvrs_hits_topN_999'][mask]
            n_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)
            stats = df['proportion_ip_pvrs_hits_topN_999'][mask]
            proportion_ip_pvrs_hits_topN = np.round(np.mean(stats), n_decimal)        
        else:
            raise ValueError('topN not recognised')

        topN_str = str(topN).zfill(3)
        result_set = 'topN_' + topN_str
    
        results[result_set] = {
            'n_ip_pvrs': n_ip_pvrs,
            'n_ip_pvr_pairs': n_ip_pvr_pairs,
            'n_ip_pvr_singletons': n_ip_pvr_singletons,
            'proportion_ip_pvrs_in_pairs': proportion_ip_pvrs_in_pairs,
            'mean_ip_pvr_confidence': mean_ip_pvr_confidence,
            'mean_pairs_confidence_absolute_differences': mean_pairs_confidence_absolute_differences,
            'n_pairs_subm_together_topN': n_pairs_subm_together_topN,
            'n_ip_pvrs_hits_topN': n_ip_pvrs_hits_topN,
            'proportion_ip_pvrs_hits_topN': proportion_ip_pvrs_hits_topN,
            }
    
    return results

