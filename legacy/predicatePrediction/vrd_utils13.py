#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
Utility functions and classes for doing inference and performance evaluation on a
PPNN validation set. The purpose is to enable 'early stopping' to be
integrated into the training of PPNN models.

The main function here is a function to drive to overall validation process.
'''

#%%

import torch

import vrd_utils10 as vrdu10
import vrd_utils11 as vrdu11
import vrd_utils12 as vrdu12


#%% function to drive inference and performance eval on PPNN validation set

def perform_validation(model, dataItem, 
                       vrd_img_names, vrd_img_names_val, 
                       device, n_images_per_epoch,
                       vrd_img_anno, prediction_region_config,
                       calculate_val_loss, loss_reduction,
                       tr_d_model2, nopred_target_policy,
                       compile_model):
    
    '''
    This function drives the overall processing of the validation set,
    whether for the calculation of validation loss and/or the calculation
    of validation predictive performance.
    
    Parameters
    ----------
    model : nn.Module
        A PPNN model undergoing training
    dataItem : a PPNNDataset object
        We use it to get an image-specific mini-batch worth of PPNN 
        training/inference data
    vrd_img_names : list of strings
        The full list of image names for the original (full) training set
        (before we carved a validation set out of it).
    vrd_img_names_val : list of strings
        The list of VRD image names that correspond to (define) the
        validation set of images that was carved out of the original (full)
        training set of image names.
    device : pytorch device
    n_images_per_epoch : integer
        The max number of image entries to process per epoch. Values of 0
        mean 'all images'. Non-zero values are used for testing only.
    vrd_img_anno : dictionary
        A VR annotations dictionary
    prediction_region_config : dictionary
        A dictionary containing the parameter settings for a particular
        prediction region cell of the experiment space.
    nopred_target_policy : integer
        The policy to use for handling 'no predicate' target vectors in
        preparation for the calculation of loss. Corresponds directly to
        dimension D_loss of the experiment space training region.
    compile_model : boolean
        Indicates whether the model has been compiled using torch.compile().
    
    
    
    Returns
    -------
    loss_results
    
    performance_results
    
    
    '''
    
    #
    # using the current model that's undergoing training, perform inference
    # on the PPNN input data that pertains to the images of the validation set
    #
    
    inference_mode = 'validation'
    
    # note: under PyTorch 2.0.1, doing inference on a model compiled with
    # torch.compile() is broken and fails if we use torch.inference_mode();
    # the work-around is to use torch.no_grad() for that case; since, under
    # PyTorch 2.0.1, torch.compile() only works if PyTorch was compiled with
    # CUDA, and CUDA only applies to Linux environments, the situation where
    # we need to use torch.no_grad() can only arise on Linux platforms (eg
    # my Kratos laptop or Hyperion or AWS)
    if device == torch.device('cuda') and compile_model == True:
        with torch.no_grad():
            ppnn_output_per_img = vrdu10.perform_inference(model, dataItem,
                                                           vrd_img_names,
                                                           vrd_img_names_val, 
                                                           device, 
                                                           n_images_per_epoch,
                                                           inference_mode,
                                                           calculate_val_loss,
                                                           loss_reduction,
                                                           tr_d_model2,
                                                           nopred_target_policy)
    else:
        with torch.inference_mode():
            ppnn_output_per_img = vrdu10.perform_inference(model, dataItem,
                                                           vrd_img_names,
                                                           vrd_img_names_val, 
                                                           device, 
                                                           n_images_per_epoch,
                                                           inference_mode,
                                                           calculate_val_loss,
                                                           loss_reduction,
                                                           tr_d_model2,
                                                           nopred_target_policy)
        #
        #ppnn_output_per_img[img_name] = {'ppnn_output_probs': output_probs,
        #                                 'inference_loss': mb_loss,
        #                                 'ppnn_object_pairs': obj_pair_info}
        #
        # if calculate_val_loss==False, mb_loss will always be 0.0

    # summarise the image-level measures of loss (mb_loss) into a summary
    # measure of 'avg loss per validation image'
    #
    # nb: if we're running on an MPS device, we must copy the tensor holding
    # the summary measure of loss into CPU memory before trying to round the
    # results to 5 decimals; whether in PyTorch 1.13.1 or PyTorch 2.0.1, 
    # trying to do torch.round() with the decimals feature on a tensor on an 
    # MPS device causes PyTorch to throw a NotImplementedError exception
    avg_loss_per_img_val = 0.0
    if calculate_val_loss:
        total_loss = 0.0
        for value in ppnn_output_per_img.values():
            total_loss += value['inference_loss']
        avg_loss_per_img_val = total_loss / len(ppnn_output_per_img)
        if device == torch.device('mps'):
            avg_loss_per_img_val = avg_loss_per_img_val.cpu()
        avg_loss_per_img_val = torch.round(avg_loss_per_img_val, decimals=5)

    # package the validation loss results
    loss_results = {'avg_loss_per_img_val': avg_loss_per_img_val}
    
    # from the PPNN inference outputs, select the outputs that represent
    # predicted predicates and transform these into predicted visual
    # relationships (VRs) for each image
    predicted_vrs_per_image = vrdu11.select_predictions(ppnn_output_per_img,
                                                        tr_d_model2, 
                                                        prediction_region_config,
                                                        n_images_per_epoch)
    
    # using the VRs predicted for the validation set images, compute
    # our recall@N-based performance evaluation metrics
    # (nb: we measure and track topN=50 performance only; the GT VR 
    #  annotations have bboxes specified in VRD format, so these need to
    #  be converted to FRCNN format to match with the bboxes used in the
    #  predicted VRs)
    topN = 50
    convert_gt_bbox_format=True
    
    # NOTE: we hardcode a value for the rrc_id parmeter here because the 
    # vrdu12.evaluate_performance() function requires one so that 
    # extended data added to the predicted VRs dictionary is always
    # associated with an rrc_id. But those updates to the predicted VRs
    # dictionary are only saved to disk during proper, test set 
    # performance evaluation. Here, in this setting, we are doing
    # validation inference and computing validation performance. So we don't
    # save the predicted VR dictionary back to disk when the call to
    # vrdu12.evaluate_performance() returns. Thus, the value of the rrc_id
    # parameter does NOT MATTER in this setting.  We could use anything,
    # even a dummy value. We use a dummy value to better highlight that the
    # value does not matter.  We use 'valPro' for 'validation processing'.
    rrc_id = 'valPro'  
    
    # NOTE: several parameters are needed wrt KG filtering during proper
    # performance evaluation on the test set images; no such KG filtering
    # is applied wrt validation processing, so we set all these parameters
    # to None; they are never used unless KG filtering is active
    entity_seq_num = None 
    ontoClassNames = None 
    ontoPropNames = None
    master_vr_type_tensor = None
    
    # ensure that KG filtering is NOT activated
    # (nb: this may override the natural setting for the current prc_id)
    prediction_region_config['kg_filtering'] = False
    
    performance_results = vrdu12.evaluate_performance(predicted_vrs_per_image,
                                                      vrd_img_anno,
                                                      topN,
                                                      convert_gt_bbox_format,
                                                      n_images_per_epoch,
                                                      prediction_region_config,
                                                      rrc_id,
                                                      entity_seq_num,
                                                      ontoClassNames,
                                                      ontoPropNames,
                                                      master_vr_type_tensor)
    
    #performance_results['topN'] 
    #performance_results['mean_gt_vrs_per_img'] 
    #performance_results['mean_pred_vrs_per_img']
    #performance_results['global_recallN']
    #performance_results['mean_per_image_recallN'] 
    #performance_results['mean_avg_recallK_topN']
    
    return loss_results, performance_results


#%% class for handling early stopping

# TODO: I MUST CITE the code upon which my code is based;
# see 'explore_early_stopping.py' for web page references

class EarlyStoppingMonitor:
    
    '''
    Monitor the behaviour of a score stream and determine whether and when
    a stopping criterion has been satisfied (so we can stop training early,
    before overfitting takes hold).
    
    The scores in the score stream may represent measures of validation loss 
    or validation performance. The single stoppping criterion algorithm is
    designed to handle both, identically.
    '''
      
    def __init__(self, patience=3, min_delta=0, verbose=False, 
                 score_type='performance'):
        
        '''
        Parameters
        ----------
        patience : int
            The number of epochs with no improvement in score to wait before
            signalling it's time to stop training. (The number of epochs to
            wait after the best score is observed before stopping training.)

        min_delta : float
            The minimum change in the score for that change to be considered
            an improvement.
            
        verbose : boolean
            If True, print diagnostic info.
        
        score_type : string
            Indicates whether the score stream we are monitoring consists of
            scores that represent some form of validation loss (ie where a 
            decrease is an improvement) or scores that represent some form of
            validation performance (ie where an increase is an
            improvement).
        
        Returns
        -------
        stop_training : boolean
            A instruction to stop training or let it continue.
        
        '''
            
        if patience < 1:
            raise ValueError('Patience must be a positive integer')
        
        if min_delta < 0.0:
            raise ValueError('Min_delta must be a non-negative float')
            
        if verbose not in [True, False]:
            raise ValueError('Verbose boolean flag not recognised')
        
        if not score_type in ['loss', 'performance']:
            raise ValueError('Score type not recognised')
        
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.score_type = score_type
        self.counter = 0
        self.best_score = float('-Inf')
        self.stop_training = False

    def monitor_score(self, score):
        
        '''
        Our algorithm assumes that larger scores are improvements and
        smaller scores are not. But the algorithm handles the tracking
        of both validation loss and validation performance.
        '''
        
        if score < 0.0:
            raise ValueError('Score should be non-negative')
        
        # if we are tracking validation loss, then flip the sign of the score
        # to make it negative so that increases can be treated as improvements
        if self.score_type == 'loss':
            score = -score

        if score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'ES counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.stop_training = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'ES counter reset; best score: {self.best_score}')

        return None


# After training stops, evaluate the models for the most recent 30 epochs 
# and take the best scores we find for the 3 metrics


#%% class for handling early stopping

# Here we use an additional stopping condition based on the 'amount' the score
# degrades from the best score observed, regardless of the number of epochs
# it takes to get to that amount of degradation
# - eg, for me, it could be something like 1.5% degrade
# if best_score is 47.5 %, we stop when the score drops to 46% or less

# This is a dual-aspect stopping condition that utilises both
# patience in terms of number of epochs, plus the idea in (A); whichever
# of the two conditions is triggered first ends up stopping the training;
# this way we could claim that (A) is our main stopping condition, but we
# also have the default, epoch patience (set at a fairly large number of
# around 20 epochs or more) for assurance that stopping will happen

# consider: epoch-patience = 20 or 25 epochs, and amount-patience = 1.5%   
# Early testing showed that the stopping epochs generated by the two
# stopping conditions are VERY similar 

# TODO: I MUST CITE the code upon which my code is based
# see 'explore_early_stopping.py' for web page references

class EarlyStoppingMonitor2:
    
    '''
    Monitor the behaviour of a score stream and determine whether and when
    a stopping criterion has been satisfied (so we can stop training early,
    before overfitting takes hold).
    
    The scores in the score stream may represent measures of validation loss 
    or validation performance. The single stoppping criterion algorithm is
    designed to handle both, identically.
    
    This implementation supports two stopping criterion, one based on 
    epoch_patience and one based on amount_patience. Whichever stopping
    criterion triggers first will stop the training.
    '''
      
    def __init__(self, epoch_patience=10, amount_patience=1.5, 
                 min_delta=0.0, score_type='performance', verbose=False):
        
        '''
        Parameters
        ----------
        epoch_patience : int
            The number of epochs to wait with no improvement in the score
            (beyond the 'best score' observed so far) 
            before signalling it's time to stop training. (That is, the 
            number of epochs to wait after the 'best score' is observed 
            before stopping training.) (That is, the number of epochs with no 
            improvement after which training will be stopped.)
    
        amount_patience : float
            The amount by which the score can degrade relative to the 'best
            score' observed so far before signalling it's time to stop
            training.

        min_delta : float
            The minimum change in the score for that change to be considered
            an improvement.
            
        verbose : boolean
            If True, print diagnostic info.
        
        score_type : string
            Indicates whether the score stream we are monitoring consists of
            scores that represent some form of validation loss (ie where a 
            decrease is an improvement) or scores that represent some form of
            validation performance (ie where an increase is an
            improvement).
        
        Returns
        -------
        stop_training : boolean
            A instruction to stop training or let it continue.
        
        '''
            
        if epoch_patience < 1:
            raise ValueError('Epoch patience must be a positive integer')

        if amount_patience < 0:
            raise ValueError('Amount patience must be a positive float')
        
        if min_delta < 0.0:
            raise ValueError('Min_delta must be a non-negative float')
            
        if verbose not in [True, False]:
            raise ValueError('Verbose boolean flag not recognised')
        
        if not score_type in ['loss', 'performance']:
            raise ValueError('Score type not recognised')
        
        self.epoch_patience = epoch_patience
        self.amount_patience = amount_patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.score_type = score_type
        self.counter = 0
        self.best_score = float('-Inf')
        self.stop_training = False

    def monitor_score(self, score):
        
        '''
        Our algorithm assumes that larger scores are improvements and
        smaller scores are not. But the algorithm handles the tracking
        of both validation loss and validation performance.
        '''
        
        if score <= 0.0:
            raise ValueError('Score must be positive')
        
        # if we are tracking validation loss, then flip the sign of the score
        # to make it negative so that increases can be treated as improvements
        if self.score_type == 'loss':
            score = -score
        
        # check amount_patience stopping condition
        if self.score_type == 'performance':
            if self.best_score - score > self.amount_patience:
                self.stop_training = True
                if self.verbose:
                    print('ES triggered: by amount_patience')
        
        # check epoch_patience stopping condition
        if score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'ES counter: {self.counter} of {self.epoch_patience}')
            if self.counter >= self.epoch_patience:
                self.stop_training = True
                if self.verbose:
                    print('ES triggered: by epoch_patience')
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'ES counter reset; best score: {self.best_score}')

        return None

