#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
Utility functions for PPNN inference.
'''

#%%

import torch
import vrd_ppnn_loss as vls


#%% function for performing inference using a particular model

def perform_inference(model, dataItem, 
                      vrd_img_names, vrd_img_names_proc,
                      device, n_images_per_epoch,
                      inference_mode,
                      calculate_loss, loss_reduction, 
                      tr_d_model2,
                      nopred_target_policy):
    '''
    Parameters
    ----------
    model : nn.Module
        A PPNN model.
    dataItem : PPNNDataset object
        A PPNNDataset object for retrieving PPNN data related to images.
    vrd_img_names : list of strings
        The list of image names for a full dataset, either the (original)
        full training set or the full test set.
    vrd_img_names_proc : list of strings
        A list of image names representing the set of images to be processed.
        The image names in this list will represent (define) either the
        validation set (carved from the original full training set) or
        the test set.  (NOTE: when doing inference on the test set, the two
        lists of image names will be identical; but when doing inference on
        the validation set during training, the two lists will be different, 
        with this 'proc' list being a small subset of the other list)
    device : torch.device
        The device type on which we are running (cpu, gpu, mps)
    n_images_per_epoch : integer
        The maximum number of images-worth of PPNN data to process. A value
        of zero (0) means process every name in vrd_img_names; a positive
        value of N means process only the first N names.
    inference_mode : string
        Specifies whether this function has been called within the context
        of 1) performing 'validation set inference' (during PPNN training,
        where validation loss is calculated, which implies the need for
        targets), or 2) performing 'test set inference', where no loss is
        ever calculated, and hence targets are not needed
    calculate_loss : boolean
        A logical flag indicating whether or not to calculate loss as part 
        of doing inference. When doing 'validation set inference', this may
        be True or False. When doing 'test set inference', we expect it to
        be False; but even if it was True it wouldn't cause loss to be
        calculated because "inference_mode=='test'" would overrule it.
    loss_reduction : string
        The reduction (summarisation) requested in the calculation of loss.
        The possible values are: 'none', 'sum', 'mean'. If calculate_loss
        is False, this value is ignored.
    tr_d_model2 : integer
        Indicates an aspect of the PPNN model architecture: whether or not
        the output layer contains a special neuron for a notional
        'no predicate' predicate
    nopred_target_policy : integer
        The policy to use for handling 'no predicate' target vectors in
        preparation for the calculation of loss. Corresponds directly to
        dimension D_loss of the experiment space training region.

    Returns
    -------
    ppnn_output_per_img : dictionary
        A dictionary containing the results of the inference operation. Each
        entry in the dictionary pertains to a particular image name.

    '''
    
    # initialise an image counter
    cnt = 0

    # initialise a dictionary to store all of the outputs of the PPNN model
    # associated with each image for which inference is performed
    ppnn_output_per_img = {}

    # iterate over the names of the images whose associated data we
    # we wish to process; this will be either the image names of the
    # validation set or the image names of the test set
    for imname in vrd_img_names_proc:
    
        # get the PPNN data (inputs and targets) for the current image name
        img_idx = vrd_img_names.index(imname)
        batch = dataItem[img_idx]  
        
        # unpack the mini-batch of data
        img_name = batch['img_name']
        if img_name != imname:
            raise ValueError('problem with image indexing')
        ppnn_img_dict = batch['ppnn_img_dict']
        inputdata = batch['inputdata']
        if inference_mode == 'validation':
            targets = batch['targets']
        
        # sometimes there may be no ppnn input data for an image; this can
        # arise if an ODNN fails to detect at least 2 objects in an image;
        # when this happens, skip the image and start processing the next        
        if inputdata == None:
            continue

        # extract dictionary of object ordered pairs from ppnn image dict;
        # the keys of this dictionary are the bbox ordered pairs to which
        # the PPNN outputs correspond; the values of this dictionary include
        # the object class labels of the two objects in the ordered pair; all
        # this info is needed to convert the PPNN outputs (predicate
        # predictions) into predicted VRs: (bbx1, cls1, pred, bbx2, cls2);
        # we need to store this info along with the PPNN outputs because
        # this info is what allows us to interpret the PPNN outputs (ie
        # to understand to what they correspond); see vrd_utils8.py for
        # detailed descriptions of the structure and content of this
        # dictionary
        obj_pair_dict = ppnn_img_dict['obj_ord_pairs']

        # but we only need a subset of the info in this dictionary; IE
        # we only need the keys (ordered pairs of bboxes) and their
        # object class labels; we extract just what we need so as to avoid
        # storing masses of unnecessary data with the inference outputs
        #
        #      '(obj1_bbox, obj2_bbox)': { 'b1_lab': obj1_label
        #                                  'b2_lab': obj2_label }
        #
        obj_pair_info = {}
        for key, feature_dict in obj_pair_dict.items():
            obj_pair_info[key] = {'b1_lab': feature_dict['b1_lab'],
                                  'b2_lab': feature_dict['b2_lab']}

        # if a non-CPU device is available, push tensors to it
        if device != torch.device('cpu'):
            inputdata = inputdata.to(device)
            if inference_mode == 'validation':
                targets = targets.to(device)

        # perform inference on the PPNN input data
        # (for each ordered pair of objects, predict the predicates that
        # that describe their visual relationships)
        output = model(inputdata)
        
        # If we're doing inference on the validation set for the purpose
        # of enabling 'early stopping' during training, then calculate_loss
        # may well be True, in which case the loss that we calculate will
        # be a validation loss. Otherwise, if we're doing inference on 
        # the test set, then calculate_loss == False and we bypass loss
        # calculation altogether
        mb_loss = 0.0
        if inference_mode == 'validation' and calculate_loss:

            # NOTE 1: in NN+KG_S1, during training we use KG reasoning to 
            # help compute a training loss penalty; strictly speaking, we 
            # should do the same here when calculating validation loss ---
            # so that training loss and validation loss are computed in
            # exactly the same way. But we base our early-stopping on
            # validation performance (recall@N), not validation loss. So we
            # never compute validation loss. So there's no point in 
            # writing the code to perform KG interaction if we're never
            # planning to ever use it.
            #
            # NOTE 2: another reason for not enabling the KG interaction here
            # is that we don't have a parameter to tell us when to perform
            # the KG interaction and when not; we'd need to introduce a 
            # new parameter to perform_inference() to give us the control
            # we're missing; what we need is the level of training region
            # dimension D_kgS1, which we get in variable like tr_d_kgS1.
            # If tr_d_kgS1 == 2 we would want to perform the KG interaction.
            #
            # mask_matrix = vrdukg.interact_with_kg_for_nnkgs1(output, ppnn_img_dict)
            # kg_results = {'kg_results_type': 'nnkgs1',
            #               'kg_results_data': mask_matrix}            
            
            kg_results = None
        
            # compute loss on the PPNN output
            mb_loss_results = vls.compute_loss(tr_d_model2, 
                                               nopred_target_policy, 
                                               kg_results,
                                               output,
                                               targets,
                                               loss_reduction,
                                               device)

            # clarify the loss results returned by compute_loss()
            mb_loss_core_matrix, mb_loss_penalty_matrix = mb_loss_results

            # reduce (summarise) the loss contributions conveyed in the
            # cells of the core loss matrix
            mb_loss_core = torch.sum(mb_loss_core_matrix)
            
            if mb_loss_penalty_matrix == None:
                mb_loss_penalty = 0.0
            else:
                mb_loss_penalty = torch.sum(mb_loss_penalty_matrix)
                
            # combine the two components of loss
            mb_loss = mb_loss_core + mb_loss_penalty
        
                
        # convert the logits output by the PPNN model to probabilities
        # using an element-wise sigmoid (logistic) function
        output_probs = torch.sigmoid(output)

        # reformat the output_probs so they can be written to JSON
        #
        # nb: with PyTorch 1.13.1, converting the output_probs tensor to
        # numpy using .numpy() works on CPU and GPU devices but not on MPS
        # devices; but with PyTorch 2.0.1, using .numpy() works only on CPU
        # devices and not on GPU or MPS devices; since we expect to use
        # PyTorch 2.0.1 and to not use CPU devices for any real training,
        # we have opted to get a copy of the tensor into CPU memory using
        # the PyTorch tensor.cpu() function by default for all device types
        output_probs = output_probs.detach().cpu()
        output_probs = output_probs.numpy().tolist()

        # store outputs (inference results) for subsequent processing
        #
        # note 1: there is a 1-to-1 positional correspondence between an
        # entry in the ppnn_img_dict (and hence in the object_pair_info 
        # dict) and a row in the PPNN output and output_probs tensors; eg 
        # entry N in the obj_pair_info dict gives the ordered pair of objects
        # to which the predicted predicate probabilities in row N of the 
        # output_probs tensor relate.
        ppnn_output_per_img[img_name] = {'ppnn_output_probs': output_probs,
                                         'inference_loss': mb_loss,
                                         'ppnn_object_pairs': obj_pair_info}
        
        # count the images processed
        cnt += 1
        #if cnt % 50 == 0:
        #    print(f'processed image {cnt}: {imname}')
        
        # check if it's time to stop processing input data
        if n_images_per_epoch > 0:
            if cnt >= n_images_per_epoch:
                break 
    
    return ppnn_output_per_img


