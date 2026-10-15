#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module defines Predicate Prediction neural network (PPNN) models.
The PPNN models are multi-class, multi-label classification models.

The job of these models is to learn to predict visual relationships between
ordered pairs of objects detected in images. The objects are represented by 
bounding box (bbox) specifications [xmin, ymin, xmax, ymax] and binary 
one-hot vectors indicating the object class of the bbox. 

The size/complexity (representation capacity) of the PPNN models varies. We
wish to have a collection of PPNN models possessing a spectrum of learning
capability: from weak learners to strong learners.

The hypothesis motivating the desire for models exhibiting a range of 
learning capability is that weak learners will, in general, provide more
space for KGs, and KG reasoning, to add value by guiding the learning so
the PPNN becomes a stronger predictor of visual relationships. Conversely,
we hypothesise that, in general, strong learners will provide less space for
KGs to add value.

The forward() methods of the PPNN models do NOT apply a Sigmoid activation
function on the logits produced by the output layer. The raw logits ---
real numbers in the interval [-\infty, \infty] --- are returned. This is 
because the Sigmoid activation functions are applied as part of the 
calculation of loss within our custom BCEWithLogitsLoss loss function. This
tactic is employed in order to realise enhanced numerical stability by
leveraging the 'log-exp trick'.                
'''

#%%

#import torch
import torch.nn as nn
import torch.nn.functional as F

#%%

class PPNN_1(nn.Module):
    def __init__(self, in_features, out_size):
        super(PPNN_1, self).__init__()
        self.in_features = in_features
        self.h1 = nn.Linear(self.in_features, self.in_features * 2)
        self.h2 = nn.Linear(self.in_features * 2, self.in_features * 2)
        self.h3 = nn.Linear(self.in_features * 2, self.in_features * 2)
        self.h4 = nn.Linear(self.in_features * 2, self.in_features)
        self.out = nn.Linear(self.in_features, out_size)
    
    def forward(self, x):
        x = F.elu(self.h1(x))
        x = F.elu(self.h2(x))
        x = F.elu(self.h3(x))
        x = F.elu(self.h4(x))
        out = self.out(x)
        return out


#%%

if __name__ == '__main__':
    in_features = 232
    model = PPNN_1(in_features, nr_predicates=72,
                   include_a_noPrediction_neuron=False)
    print(model)

#%%


