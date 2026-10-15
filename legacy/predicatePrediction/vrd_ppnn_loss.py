#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module contains definitions of custom loss functions.
'''

#%%

import torch
import torch.nn as nn
import numpy as np


#%%

def custom_bce_with_logits(logit, target, reduction='mean',
                           weight=None, pos_weight=None, neg_weight=None):
    '''
    See document 'pytorch-binary-cross-entropy-with-logits (bcewl) - notes.txt'
    and 'pytorch-bcewl.py' for background on loss computation algorithms
    defined here. The computations here are based on PyTorch's
    implementation of the BCEWithLogitsLoss loss function.
    
    We have implemented here our own custom version of that implementation ---
    one that accepts a new custom tensor parameter 'neg_weight'.
    If provided, these weights can be applied against the 2nd (negative) term 
    of the conventional binary cross-entropy loss function.
    
    We implemented this customisation in anticipation of wanting to have
    granular control over the loss values computed by the 2nd (negative)
    term of the BCE loss function. The PyTorch BCE loss function already
    accepts a 'pos_weight' tensor, for optionally modifying loss values 
    computed by the 1st (positive) term of the BCE loss function. We
    introduced the 'neg_weight' tensor in order to have similar control  
    over the loss values computed by the 2nd (negative) term of the BCE
    loss function.
    
    In addition to accepting the new neg_weight tensor as a parameter, the
    implementations of the actual loss computation functions had to be 
    re-thought and re-designed in order to admit the 'neg_weight' tensor
    in the mathematics of things correctly. This was a non-trivial task.
    
    Note: The authors of the PyTorch implementation of the BCE loss function
    (upon which our code is based) claim, in the documentation, to have
    applied the 'log-exp trick' in their code. This is a tactic for
    avoiding explosion during exponentiation. 
        They employ the PyTorch clamp() function as part of this tactic.
    We follow their lead in that respect.
        But my investigations of the 'log-exp trick' convince me that this
    claim by the PyTorch authors is partly mistaken.  At most we can say
    they use a tactic derived from or related to the 'log-exp trick'. The
    tactic avoids explosion during exponentiation, which is the main thing.
    But to call it the 'log-exp trick' is, I think, a claim too far. 

    A weight matrix of all 1s, if provided, has no effect on the calculation 
    of loss.  
    
    A neg-weight matrix of all 1s has no effect on the calculation of loss.
    '''
    if not (target.size() == logit.size()):
        raise ValueError('Target size must match logit size')

    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError('Reduction not recognised')

    if pos_weight is not None:
        if not (pos_weight.size() == logit.size()):
            raise ValueError('Pos_weight size must match logit size')

    if neg_weight is not None:
        if not (neg_weight.size() == logit.size()):
            raise ValueError('Neg_weight size must match logit size')

    if weight is not None:
        if not (weight.size() == logit.size()):
            raise ValueError('Weight size must match logit size')

    adjust = torch.clamp(-logit, min=0)

    if pos_weight is not None and neg_weight is not None:
        log_weight = (pos_weight - neg_weight) * target + neg_weight
        loss = (1 - target) * neg_weight * logit + log_weight * (
               ((-adjust).exp() + (-logit - adjust).exp()).log() + adjust
               )
    elif pos_weight is not None:
        log_weight = (pos_weight - 1) * target + 1
        loss = (1 - target) * logit + log_weight * (
               ((-adjust).exp() + (-logit - adjust).exp()).log() + adjust
               )
    elif neg_weight is not None:
        log_weight = (1 - neg_weight) * target + neg_weight
        loss = (1 - target) * neg_weight * logit + log_weight * (
               ((-adjust).exp() + (-logit - adjust).exp()).log() + adjust
               )
    elif pos_weight is None and neg_weight is None:
        loss = ( (1 - target) * logit + adjust + 
                 ((-adjust).exp() + (-logit - adjust).exp()).log()
               )
    else:
        raise ValueError('Problem with loss calculation')

    # lastly, we do an element-wise multiplication with the 'weight' matrix;
    # this may, for example, zero-out certain rows of the loss tensor in 
    # order to nullify the influence of certain types of target vectors
    if weight is not None:
        loss = loss * weight

    # apply reduction
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        pass

    return loss


#%%

class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean',
                 pos_weight=None, neg_weight=None):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logit, target):
        res = custom_bce_with_logits(logit, target,
                                     weight=self.weight,
                                     reduction=self.reduction,
                                     pos_weight=self.pos_weight,
                                     neg_weight=self.neg_weight)
        return res


#%%

def get_adjusting_value(x):
    '''
    Set the optimal value for adjusting the exponent in an exponentiation
    expression so as to avoid risk of exponentiation causing exploding
    results and numerical overflow.  This value is used whenever the
    'log-exp trick' is employed to ensure numerical stability when
    exponentiation operations are performed.

    Parameters
    ----------
    x : real (scalar)
        A single, scalar real number (a logit) in interval [-\infty, \infty]

    Returns
    -------
    val : real (scalar)
        A value to be used in an exponentiation operation along with the
        input value x.

    '''
    if -x > 0.0:
        val = -x
    else:
        val = 0.0
    return val


#%%

def calc_special_value_A(x):
    '''
    Calculate a special value we call 'A' as a function of the input parameter.

    Parameters
    ----------
    x : real (scalar)
        A single, scalar real number (a logit) in interval [-\infty, \infty]

    Returns
    -------
    val : real (scalar)
        A real value that is a function of the input x

    '''

    adj = get_adjusting_value(x)

    numer = (-adj) - np.log(np.exp(-adj) + np.exp(-x - adj))

    denom = (-x) - adj - np.log(np.exp(-adj) + np.exp(-x - adj))

    val = numer / denom

    return val


#%%

def apply_policy_d_nopredTarget_1(targets):
    '''
    This function implements the policy for handling 'no predicate' target
    vectors that corresponds to level D_loss_1 of dimension D_loss of the
    experiment space training region.
    
    The levels of dimension D_loss correspond to different policies for
    handling 'no predicate' target vectors that appear in the targets
    tensor. 'no predicate' target vectors are notional (dummy) target
    vectors that represent the absence of underlying VR annotations 
    (predicates) for a given ordered pair of objects in an image.
    
    The policies for handling these 'no predicate' target vectors 
    control whether and when and why and how they are allowed to influence 
    the calculation of loss.
    
    Level D_loss_1 corresponds to the following policy:
    
    what: let no 'no predicate' target vectors influence the calculation of 
    loss in any way
    
    how: we build a 'weight' matrix that lets us zero-out the loss elements
    we don't want to allow to participate in the calculation of loss; 
    for all 'no predicate' target vectors in the target tensor, we 
    set the corresponding rows of this 'weight' matrix to all zeros; 
    this way, when the loss function multiplies the 'weight' matrix and
    the 'loss' matrix element-wise, we can zero-out any loss that has been
    calculated in relation to 'no predicate' target vectors. Rows of the
    'weight' matrix that do not correspond to 'no predicate' target vectors
    will be all 1s --- so the matrix multiplication won't change and
    associated loss elements.
      
    That is, policy D_loss_1 is designed to completely nullify the presence
    of the dummy 'no predicate' target vectors in the target tensor. We
    introduced them to enable loss calculation via binary cross-entropy
    where the shapes of the output and target tensors must be identical.
    But having introduced these dummy target vectors for structural
    (tensor shape) reasons, policy D_loss_1 seeks to nullify their effect on
    on the supervision of training. Policy D_loss_1 nullifies their
    influence on loss calculation completely.
    
    Parameters
    ----------
    targets : 2D tensor (no batch dimension)
        The tensor of binary multi-hot target vectors to be used in the
        calculation of (a binary cross-entropy-based) loss

    Returns
    -------
    targets : 2D tensor (no batch dimension)
        A copy of the targets tensor parameter with potential modifications
        to the values stored in the 'no predicate' element of some of the
        target vectors.
    
    weight : 2D tensor, with same size as targets tensor
        A tensor of weight parameters to be passed to a loss function along
        with the targets tensor.  (loss weights)
    '''

    size = targets.size()  

    # initialise the weight matrix to all 1s; if the weight matrix stays
    # all 1s, it will have no effect on the calculation of loss
    weight = torch.ones(size)

    rows = size[0]
    cols = size[1]
     
    zeros = torch.zeros(cols)

    # WARNING start ========================================================
    #
    # the code block between the delimiters 'WARNING start' and 
    # 'WARNING end' behaves incorrectly if we're running on an MPS device
    # with PyTorch 1.13.1; if we're computing training loss, the code block
    # behaves correctly, but if we're computing validation loss things go
    # wonky; the code line 'if targets[idx, -1] == 1.0' becomes oddly 
    # erratic; the condition gets satisfied when it shouldn't, and when it
    # is satisfied, the assignment of 'zeros' fails to happen; the effect
    # is that the weight tensor is constructed improperly and so the 
    # calculation of validation loss on my MBP16 laptop goes badly wrong
    # and is radically different from the validation loss computed on my
    # iMac (CPU) and on my Linux laptop Kratos (GPU), which are identical
    #
    # happily, the cause of the problem, whatever it is, has been identified
    # by the PyTorch people and the problem is fixed in PyTorch 2.0.1; if
    # we run on an MPS device under PyTorch 2.0.1, the weight matrix is
    # constructed properly when we calculating validation loss and the 
    # computed validation loss is identical to that computed on iMac (CPU) 
    # and Kratos (GPU)
    #
    # So: the net effect is that we have an unwanted DEPENDENCY; for this code
    # block to execute correctly on an MPS device, we MUST be running with
    # PyTorch 2, not PyTorch 1.13
    
    for idx in range(rows):
        # nullify the influence of ALL 'no predicate' target vectors,
        # *matched* and *unmatched*, on the calculation of loss
        if targets[idx, -1] != 0.0: # != 0.0 indicates 'no predicate' target
            weight[idx,:] = zeros
        # remove the distinction between *matched* and *unmatched*
        # 'no predicate' target vectors by changing any instances of the
        # special value 2.0 to 1.0
        if targets[idx, -1] == 2.0:
            targets[idx, -1] = 1.0

    # WARNING end ==========================================================
    
    return targets, weight


#%%

def apply_policy_d_nopredTarget_2(targets):
    '''
    This function implements the policy for handling 'no predicate' target
    vectors that corresponds to level D_loss_2 of dimension D_loss of the
    experiment space training region.
    
    The levels of dimension D_loss correspond to different policies for
    handling 'no predicate' target vectors that appear in the targets
    tensor. 'no predicate' target vectors are notional (dummy) target
    vectors that represent the absence of underlying VR annotations 
    (predicates) for a given ordered pair of objects in an image.
    
    The policies for handling these 'no predicate' target vectors 
    control whether and when and why and how they are allowed to influence 
    the calculation of loss.
    
    Level D_loss_2 corresponds to the following policy:
    
    what: let *matched* 'no predicate' target vectors influence the 
    calculation of loss, but prevent *unmatched* 'no predicate' target 
    vectors from doing so
    
    how: we build a 'weight' matrix that lets us zero-out the loss elements
    we don't want to allow to participate in the calculation of loss; 
    for all *unmatched* 'no predicate' target vectors in the target tensor, we 
    set the corresponding rows of this 'weight' matrix to all zeros; 
    this way, when the loss function multiplies the 'weight' matrix and
    the 'loss' matrix element-wise, we can zero-out any loss that has been
    calculated in relation to *unmatched* 'no predicate' target vectors. 
    Rows of the 'weight' matrix that do not correspond to *unmatched* 
    'no predicate' target vectors will be all 1s --- so the matrix 
    multiplication won't change and associated loss elements.
      
    That is, policy D_loss_2 is designed to only partially nullify the presence
    of the dummy 'no predicate' target vectors in the target tensor. It
    nullifies the influence of *unmatched* 'no predicate' target vectors but
    allows *matched* 'no predicate' target vectors to participate in the
    calculation of loss and, thereby, to provide supervision to the training
    of our PPNN models.
    
    Note: in experiments that are part of the 'predicate detection' regime,
    all 'no predicate' target vectors will be *matched*; so, for
    'predicate detection', ALL 'no predicate' target vectors will be 
    allowed to influence the calculation of loss (which is very similar to
    policy D_loss_3)
    
    Parameters
    ----------
    targets : 2D tensor (no batch dimension)
        The tensor of binary multi-hot target vectors to be used in the
        calculation of (a binary cross-entropy-based) loss

    Returns
    -------
    targets : 2D tensor (no batch dimension)
        A copy of the targets tensor parameter with potential modifications
        to the values stored in the 'no predicate' element of some of the
        target vectors.
    
    weight : 2D tensor, with same size as targets tensor
        A tensor of weight parameters to be passed to a loss function along
        with the targets tensor.  (loss weights)
    '''

    size = targets.size()  

    weight = torch.ones(size)

    rows = size[0]
    cols = size[1]
    
    zeros = torch.zeros(cols)
   
    for idx in range(rows):
        # nullify the influence of *unmatched* 'no predicate' target vectors
        # upon the calculation of loss
        if targets[idx, -1] == 1.0:  # 1.0 indicates *unmatched* 'no predicate' target
            weight[idx,:] = zeros
        # remove the distinction between *matched* and *unmatched*
        # 'no predicate' target vectors by changing any instances of the
        # special value 2.0 to 1.0
        if targets[idx, -1] == 2.0:
            targets[idx, -1] = 1.0

    return targets, weight


#%%

def apply_policy_d_nopredTarget_3(targets):
    '''
    This function implements the policy for handling 'no predicate' target
    vectors that corresponds to level D_loss_3 of dimension D_loss of the
    experiment space training region.
    
    The levels of dimension D_loss correspond to different policies for
    handling 'no predicate' target vectors that appear in the targets
    tensor. 'no predicate' target vectors are notional (dummy) target
    vectors that represent the absence of underlying VR annotations 
    (predicates) for a given ordered pair of objects in an image.
    
    The policies for handling these 'no predicate' target vectors 
    control whether and when and why and how they are allowed to influence 
    the calculation of loss.
    
    Level D_loss_3 corresponds to the following policy:
    
    what: let *all* 'no predicate' target vectors (matched and unmatched) 
    influence the calculation of loss
    
    how: we could build a 'weight' matrix of all 1s; this way, when the
    loss tensor is multiplied by the 'weight' tensor (element-wise), 
    the loss values that were computed won't be changed in any way; but
    it's simpler to simply not use any 'weight' tensor in this case; it
    is an optional parameter after all, so we can signal that it is not
    used simply by setting it to None
          
    Note: in experiments that are part of the 'predicate detection' regime,
    ALL 'no predicate' target vectors will be *matched* 'no predicate'
    target vectors; so, in 'predicate detection' experiments, policy D_loss_3 
    ends up having an effect identical to that of policy D_loss_3

    In 'relationship detection' regime experiments, however, 'no predicate' 
    target vectors may be either *matched* or *unmatched*. In this case,
    policy D_loss_3 has a different effect to policy D_loss_2. In this case,
    policy D_loss_3 allows *unmatched* 'no predicate' target vectors to 
    influence the calculation of loss along with *matched* ones; whereas
    in policy D_loss_2, only *matched* 'no predicate' target vectors are
    alloed to influence the calculation of loss.
    
    Parameters
    ----------
    targets : 2D tensor (no batch dimension)
        The tensor of binary multi-hot target vectors to be used in the
        calculation of (a binary cross-entropy-based) loss

    Returns
    -------
    targets : 2D tensor (no batch dimension)
        A copy of the targets tensor parameter with potential modifications
        to the values stored in the 'no predicate' element of some of the
        target vectors.
    
    weight : None
        For this policy the weight tensor is not needed in the calculation
        of loss, so we return None to signal that the tensor is not to be used.
    '''

    size = targets.size()  
    rows = size[0]
   
    for idx in range(rows):
        # remove the distinction between *matched* and *unmatched*
        # 'no predicate' target vectors by changing any instances of the
        # special value 2.0 to 1.0
        if targets[idx, -1] == 2.0:
            targets[idx, -1] = 1.0

    weight = None

    return targets, weight


#%%

def prepare_for_loss_calc(nopred_target_policy, targets):
    '''
    Parameters
    ----------
    nopred_target_policy : integer
        The policy to use for handling 'no predicate' target vectors in
        preparation for the calculation of loss. Corresponds directly to
        dimension D_nopredTarget of the experiment space training region.    
    
    
    '''
    # The nopred_target_policy specifies the level of dimension D_loss
    # that is in effect for the current experiment being run. Each level
    # corresponds to a different policy for handling 'no predicate' target
    # vectors and controlling what influence they have over the calcuation
    # of loss.    

    # NOTE: if we need to use the results of KG interaction to refine the 
    # target vectors in the target tensor and/or the weights in either
    # the 'weight' or 'neg_weight' tensors, this may well be the place 
    # to do it   

    if nopred_target_policy == 1:    # level D_nopredTarget_1 
        targets2, weight = apply_policy_d_nopredTarget_1(targets)  
        neg_weight = None            
    elif nopred_target_policy == 2:  # level D_nopredTarget_2 
        targets2, weight = apply_policy_d_nopredTarget_2(targets)  
        neg_weight = None            
    elif nopred_target_policy == 3:  # level D_nopredTarget_3 
        targets2, weight = apply_policy_d_nopredTarget_3(targets)  
        neg_weight = None            
    else:
        raise ValueError(f'nopred_target_policy {nopred_target_policy} not recognised')

    results = {'targets': targets2,
               'weight': weight,
               'neg_weight': neg_weight}

    return results


#%%

def compute_penalty_loss_for_nnkgs1(output, kg_results):
    '''
    Compute loss penalties based on KG reasoning for experiment family
    NN+KG_S1.
    '''
    
    penalty_matrix = torch.zeros(output.size())
    
    output_probs = torch.sigmoid(output)
    
    cells_for_loss_penalty = kg_results['kg_results_data']
    
    cells_to_process = cells_for_loss_penalty.nonzero()
    
    for cell in cells_to_process:
        row, col = cell[0], cell[1]
        conf = output_probs[row, col]
        loss_penalty = - torch.log(1 - conf)
        penalty_matrix[row, col] = loss_penalty
    
    return penalty_matrix


#%%

def compute_loss(tr_d_model2, nopred_target_policy, kg_results,
                 output, targets,
                 loss_reduction, device):
    
    '''
    Parameters
    ----------
    tr_d_model2 : integer
        Indicates whether the output layer of the PPNN model includes a 
        special 'no predicate' neuron to permit explicit prediction of
        'no predicate' for a given ordered pair of objects. Corresponds
        directly to dimension D_model2 of the experiment space training
        region.
    nopred_target_policy : integer
        The policy to use for handling 'no predicate' target vectors in
        preparation for the calculation of loss. Corresponds directly to
        dimension D_nopredTarget of the experiment space training region.
    kg_results : dictionary
        A dictionary conveying information derived from KG reasoning to
        be incorporated into the overall calculation of loss. The keys
        of the dictionary are constant, but the contents can vary with the
        experiment family.  For example, for NN+KG_S1 we have
            kg_results = {'kg_results_type': 'nnkgs1',
                          'kg_results_data': mask_matrix}
        where mask_matrix is a binary matrix.
    output : tensor
        A tensor of PPNN model outputs from a PPNN forward pass for a single
        mini-batch (ie image-worth of PPNN data). The values in the tensor
        are logits in the interval [-Inf, Inf], not probabilities.
    targets : tensor
        A tensor of example targets for a single mini-batch (ie for an 
        image-worth of PPNN data). Each target vector (row) in the tensor
        is a binary multi-hot vector.
    loss_reduction : string
        The reduction (summarisation) requested in the calculation of loss.
        The possible values are: 'none', 'sum', 'mean'
    device : torch.device
        The device (CPU or GPU or MPS) on which we are running.
        
    Returns
    -------
    mb_loss : tensor
        The loss calculated for the current mini-batch.
    
    '''

    testing = False

    # --------------------------------------------------------------------   

    # testing code

    if testing:
        print('vrd_ppnn_loss.py compute_loss() - incoming targets')
        print(f'targets.shape: {targets.shape}')
        n_matched_nopredpred = 0
        n_unmatched_nopredpred = 0
        n_rows = int(targets.shape[0])
        for idxt in range(n_rows):
            n_preds = targets[idxt].sum()           
            if targets[idxt,-1] == 1.0:
                n_unmatched_nopredpred += 1
            if targets[idxt,-1] == 2.0:
                n_matched_nopredpred += 1
                n_preds -= 1
            print(f'idxt: {idxt}, n_preds: {n_preds}, nopredpred: {targets[idxt,-1]}')
        print(f'n_matched_nopredpred: {n_matched_nopredpred}, n_unmatched_nopredpred: {n_unmatched_nopredpred}')

    # --------------------------------------------------------------------   


    # prepare for the calculation of loss
    results = prepare_for_loss_calc(nopred_target_policy, targets)
    targets = results['targets']
    weight = results['weight']
    neg_weight = results['neg_weight']

    # ensure all tensors are on the same device (CPU, GPU, MPS)
    if device != torch.device('cpu'):
        if weight != None:
            weight = weight.to(device)
        if neg_weight != None:
            neg_weight = neg_weight.to(device)


    # --------------------------------------------------------------------   

    # testing code

    #print(f'targets.device: {targets.device}')
    #print(f'weight.device: {weight.device}')
    #if neg_weight == None:
    #    print('neg_weight: None')
    #else:
    #    print(f'neg_weight.device: {neg_weight.device}')

    #print('targets portion')
    #print(targets[0:10,0:15])
    #print('weight portion')
    #print(weight[0:10,0:15])

    #print(f'targets size: {targets.size()}')
    #print(f'weight size: {weight.size()}')
    #print(f'neg_weight: {neg_weight}')
    #force_stop = True
    #break
    
    if testing:
        print('vrd_ppnn_loss.py compute_loss() - final targets')
        print(f'targets.shape: {targets.shape}')
        n_matched_nopredpred = 0
        n_unmatched_nopredpred = 0
        n_rows = int(targets.shape[0])
        for idxt in range(n_rows):
            n_preds = targets[idxt].sum()           
            if targets[idxt,-1] == 1.0:
                n_unmatched_nopredpred += 1
            if targets[idxt,-1] == 2.0:
                n_matched_nopredpred += 1
                n_preds -= 1
            print(f'idxt: {idxt}, n_preds: {n_preds}, nopredpred: {targets[idxt,-1]}')
        print(f'n_matched_nopredpred: {n_matched_nopredpred}, n_unmatched_nopredpred: {n_unmatched_nopredpred}')

    if testing:
        if weight == None:
            print('weight tensor: None')
        else:
            print('vrd_ppnn_loss.py compute_loss() - weight tensor')
            print(f'weight.shape: {weight.shape}')
            n_rows = int(weight.shape[0])
            for idxt in range(n_rows):
                rowsum = weight[idxt].sum()
                print(f'idxt: {idxt}, rowsum: {rowsum}')

    # --------------------------------------------------------------------

    # if the PPNN model does not have a special neuron in its output layer
    # to permit explicit predictions of 'no predicate' for a given ordered
    # pair of objects, then the target vectors used for loss calculation
    # must not have a special element for the 'no predicate' predicate; in
    # which case, we need to remove that last column of the targets tensor;
    # its purpose has now been served and we can safely remove it, and must
    # remove it --- otherwise, the targets tensor won't be the same size as 
    # the output tensor and BCE loss computation will throw an exception
    if tr_d_model2 == 1:
        targets = targets[:,:-1]
        if weight != None:
            weight = weight[:,:-1]

    # compute the core (base) loss for a given mini-batch (ie for a given
    # image-worth of PPNN data)
    core_loss = custom_bce_with_logits(output, targets,
                                       reduction=loss_reduction,
                                       weight=weight,
                                       neg_weight=neg_weight)

    # --------------------------------------------------------------------
    
    #
    # If data has been provided derived from KG reasoning based on interaction
    # with a KG, incorporate it into the calculation of loss.
    #
    
    if kg_results == None:
        penalty_loss = None
    else:
        if kg_results['kg_results_type'] == 'nnkgs1':
            penalty_loss = compute_penalty_loss_for_nnkgs1(output, kg_results)
        else:
            raise ValueError('kg_results not recognised')

    # --------------------------------------------------------------------

    return core_loss, penalty_loss




