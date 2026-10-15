#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 18:50:21 2022

@author: dave
"""

'''
This is a module of utility functions for deriving geometric features
of object bounding boxes (bboxes). These utility functions derive geometric 
features for individual bounding boxes as well as for relationships between 
ordered pairs of bounding boxes.  The bbox geometric features will be fed
as input to a neural network to help it predict visual relationships
between ordered pairs of objects detected in images.

These functions are used on pairs of bboxes for objects detected
in VRD dataset images by a Faster R-CNN object detector model. The objects
detected within a given image will have had their bboxes pre-arranged into 
(subject, object) ordered pairs. These bbox pairs (2-tuples of bboxes) are
the inputs to the bbox geometric feature functions defined in this
module.

NOTE: All of the functions defined in this module assume that the bbox 
coordinates are in the format [xmin, ymin, xmax, ymax]. This is the format 
of the bounding boxes output by a Faster R-CNN object detector model. It is 
NOT the format used in the visual relationship annotations of the VRD
dataset images, which is [ymin, ymax, xmin, xmax]. 

NOTE: Consider revising the calculations of widths and heights, etc, to use
width = b[2] - b[0] + 1
height = b[3] - b[1] + 1
as was done in an online example we found, and as is done in the MATLAB code
for the Lu & Fei-Fei (2016) paper. Is this a safety tactic, to avoid values
of zero (0)?  It's not at all clear that this is the 'mathematically
correct' way to do it.  Further, recall that in City
module INM705 (Image Analysis), Alex Ter-Sarkisov did NOT add the 1. And
I have found other examples online
(https://medium.com/analytics-vidhya/basics-of-bounding-boxes-94e583b5e16c)
where they do NOT add the 1 to calculate the width and height. 
'''

#%%

import math

#%% bbox width and height

def bb_width_height(b):
    width = float(b[2] - b[0])
    height = float(b[3] - b[1])
    return width, height

#%% bbox areas

def bb_areas(b1, b2):
    b1_width, b1_height = bb_width_height(b1)
    b1_area = b1_width * b1_height
    b2_width, b2_height = bb_width_height(b2)
    b2_area = b2_width * b2_height
    return b1_area, b2_area

#%% bbox centroids

def bb_centroids(b1, b2):
    b1_width, b1_height = bb_width_height(b1)
    b1_c_x = b1[0] + (b1_width / 2)
    b1_c_y = b1[1] + (b1_height / 2)
    b2_width, b2_height = bb_width_height(b2)
    b2_c_x = b2[0] + (b2_width / 2)
    b2_c_y = b2[1] + (b2_height / 2)   
    return (b1_c_x, b1_c_y), (b2_c_x, b2_c_y)

#%% Euclidean distance between bbox centroids

def bb_euclidean_distance_centroids(b1, b2):
    '''
    Calculate the Euclidean distance between the centroids of two bboxes.
    '''
    b1_c, b2_c = bb_centroids(b1,b2)
    eucl_dist = math.sqrt( (b1_c[0] - b2_c[0])**2 + (b1_c[1] - b2_c[1])**2 )
    return eucl_dist

#%% Ratios of Euclidean distance between centroids to image width & height

def bb_euclidean_distance_centroids_ratios(b1, b2, im_w, im_h):
    '''
    Calculate the ratios of the Euclidean distance between the centroids
    of the two bboxes relative to the width and height of the image.
    In practice, values will be in [0, 1] almost always, but it's 
    possible that the distance between centroids could be larger than 
    either the width or height of an image, but in practice it won't ever
    be by much. So we expect values to always be in [0, 2], say.
    '''
    eucl_dist = bb_euclidean_distance_centroids(b1,b2)
    eucl_dist_to_width_ratio = eucl_dist / im_w
    eucl_dist_to_height_ratio = eucl_dist / im_h 
    return eucl_dist_to_width_ratio, eucl_dist_to_height_ratio

#%% Sine and Cosine of angle between bbox centroids

def bb_sine_and_cosine_of_angle_between_centroids(b1, b2):
    '''
    Treat the centroid of bbox b1 as though it were the origin. 
    Calculate the sine and cosine of the angle between the two centroids
    by moving counter-clockwise relative to the centroid of bbox b1.
    '''
    hypotenuse_length = bb_euclidean_distance_centroids(b1,b2)
    if hypotenuse_length == 0:
        hypotenuse_length = 1   # avoid division by zero
    b1_c, b2_c = bb_centroids(b1,b2)
    vert_length = b2_c[1] - b1_c[1]
    sine_theta = vert_length / hypotenuse_length
    horiz_length = b2_c[0] - b1_c[0]
    cosine_theta = horiz_length / hypotenuse_length
    return sine_theta, cosine_theta

#%% bbox aspect ratios

def bb_aspect_ratios(b1, b2):
    b1_width, b1_height = bb_width_height(b1)
    b1_aspect_ratio = b1_height / b1_width
    b2_width, b2_height = bb_width_height(b2)
    b2_aspect_ratio = b2_height / b2_width
    return b1_aspect_ratio, b2_aspect_ratio

#%% bbox intersection area

def bb_intersection_area(b1, b2):
    x1 = max(b1[0], b2[0])  # max xmin
    y1 = max(b1[1], b2[1])  # max ymin
    x2 = min(b1[2], b2[2])  # min xmax
    y2 = min(b1[3], b2[3])  # min ymax
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    return float(intersection_area)

#%% bbox union area

def bb_union_area(b1, b2):
    intersection_area = bb_intersection_area(b1,b2)   
    b1_area, b2_area = bb_areas(b1,b2)
    union_area = b1_area + b2_area - intersection_area
    return union_area

#%% bbox IoU (Intersection over Union)

def bb_intersection_over_union(b1, b2):
    '''
    IoU is always in the unit interval [0,1].
    '''
    ia = bb_intersection_area(b1,b2)
    ua = bb_union_area(b1,b2)
    iou = ia / ua
    return iou

#%% bbox inclusion ratios

def bb_inclusion_ratios(b1, b2):
    '''
    Calculate the inclusion ratios for a pair of bounding boxes.
    An inclusion ratio measures the degree to which one bbox is 
    enclosed within another bbox. Inclusion ratios always lie in
    the unit interval [0,1]. If two bboxes are identical, both
    inclusion ratios will be exactly 1, which corresponds to an
    IoU of 1.
    '''
    b1_area, b2_area = bb_areas(b1,b2)
    intersection_area = bb_intersection_area(b1,b2)
    
    # inclusion ratio of b1 within b2
    # degree (extent) to which b1 is enclosed within b2
    ir_b1b2 = intersection_area / b1_area
    
    # inclusion ratio of b2 within b1
    # degree (extent) to which b2 is enclosed within b1
    ir_b2b1 = intersection_area / b2_area
    
    return ir_b1b2, ir_b2b1

#%% bbox area ratios

def bb_area_ratios(b1, b2):
    '''
    Calculate the area ratios for a pair of bounding boxes. An area ratio
    measures the ratio of the area of one bbox relative to the area of the
    other bbox. Area ratios always lie in the interval [0, infty].  
    '''
    b1_area, b2_area = bb_areas(b1,b2)  
    ar_b1b2 = b1_area / b2_area 
    ar_b2b1 = b2_area / b1_area 
    return ar_b1b2, ar_b2b1

#%% ratios of bbox area to image area

def bb_area_to_image_area_ratios(b1, b2, im_w, im_h):
    '''
    Calculate the ratios of the areas of the bboxes relative to
    the area of the entire image with which the bboxes are
    associated. Such ratios always lie in the unit interval, [0, 1].
    '''
    b1_area, b2_area = bb_areas(b1,b2)
    im_area = im_w * im_h
    b1_to_im_area_ratio = b1_area / im_area
    b2_to_im_area_ratio = b2_area / im_area
    return b1_to_im_area_ratio, b2_to_im_area_ratio

#%%

def bb_horiz_dist_edges_ratio(b1, b2, im_w):
    '''
    Ratio of the horizontal distance between the right and left edges
    of two bboxes relative to the image width.  Values in [0, 1].
    '''
    
    # begin by assuming that there is no horizontal free space between
    # the two bboxes; ie assume that the two bboxes overlap in horizontal
    # space (even though they may not actually intersect with one another
    # due to vertical displacement)
    horiz_dist = 0
    
    if b1[2] < b2[0]:  # b1_xmax < b2_xmin
        # b1 is fully to the left of b2, with free space between
        horiz_dist = b2[0] - b1[2]
        
    if b2[2] < b1[0]:  # b2_xmax < b1_xmin 
        # b2 is fully to the left of b1, with free space between
        horiz_dist = b1[0] - b2[2]
    
    horiz_dist_to_im_width_ratio = horiz_dist / im_w
    
    return horiz_dist_to_im_width_ratio

#%%

def bb_vert_dist_edges_ratio(b1, b2, im_h):
    '''
    Ratio of the vertical distance between the bottom and top edges
    of two bboxes relative to the image height.  Values in [0, 1].
    '''
    vert_dist = 0
    
    if b1[3] < b2[1]:
        # b1 is fully above b2, with free space between
        # (nb: we say b1 'above' b2 here because bbox specifications are
        #  defined assuming the top-left of the image is the 'origin', (0,0),
        #  but we observe an image with the bottom-left as our origin)
        vert_dist = b2[1] - b1[3]
        
    if b2[3] < b1[1]:
        # b2 is fully above b1, with free space between
        vert_dist = b1[1] - b2[3]
        
    vert_dist_to_im_height_ratio = vert_dist / im_h
    
    return vert_dist_to_im_height_ratio

#%% 

def bb_horiz_dist_centroids_ratio(b1, b2, im_w):
    '''
    Ratio of the horizontal distance between the centroids of two bboxes 
    relative to the image width.  Values in [0,1].
    '''
    b1_c, b2_c = bb_centroids(b1,b2)
    horiz_dist = abs(b1_c[0] - b2_c[0])
    horiz_dist_to_im_width_ratio = horiz_dist / im_w
    return horiz_dist_to_im_width_ratio

#%%

def bb_vert_dist_centroids_ratio(b1, b2, im_h):
    '''
    Ratio of the vertical distance between the centroids of two bboxes 
    relative to the image height. Values in [0,1].
    '''
    b1_c, b2_c = bb_centroids(b1,b2)
    vert_dist = abs(b1_c[1] - b2_c[1])
    vert_dist_to_im_height_ratio = vert_dist / im_h
    return vert_dist_to_im_height_ratio

#%%

def bb_horiz_dist_centroids_to_nearest_edges(b1, b2, im_w):
    '''
    The horizontal distances between the centroid of one bbox and the
    nearest edge of the other bbox, both as ratios of image width.
    '''
    
    b1_c, b2_c = bb_centroids(b1,b2)  # (b1_c_x, b1_c_y), (b2_c_x, b2_c_y)
    
    # calc horizontal distance between centroid of bbox b1 and the
    # nearest edge of bbox b2
    hd_b1c_b2_left_edge = abs(b2[0] - b1_c[0])   # abs(b2_xmin - b1_c_x)
    hd_b1c_b2_right_edge = abs(b2[2] - b1_c[0])  # abs(b2_xmax - b1_c_x)
    hd_b1c_b2_nearest_edge = min(hd_b1c_b2_left_edge, hd_b1c_b2_right_edge)
    hd_b1c_b2_nearest_edge_ratio = hd_b1c_b2_nearest_edge / im_w

    # calc horizontal distance between centroid of bbox b2 and the
    # nearest edge of bbox b1
    hd_b2c_b1_left_edge = abs(b1[0] - b2_c[0])   # abs(b1_xmin - b2_c_x)
    hd_b2c_b1_right_edge = abs(b1[2] - b2_c[0])  # abs(b1_xmax - b2_c_x)
    hd_b2c_b1_nearest_edge = min(hd_b2c_b1_left_edge, hd_b2c_b1_right_edge)
    hd_b2c_b1_nearest_edge_ratio = hd_b2c_b1_nearest_edge / im_w
    
    return hd_b1c_b2_nearest_edge_ratio, hd_b2c_b1_nearest_edge_ratio

#%% 

def bb_vert_dist_centroids_to_nearest_edges(b1, b2, im_h):
    '''
    The vertical distances between the centroid of one bbox and the
    nearest edge of the other bbox, both as ratios of image height.
    '''
    
    b1_c, b2_c = bb_centroids(b1,b2)  # (b1_c_x, b1_c_y), (b2_c_x, b2_c_y)
    
    # calc vertical distance between centroid of bbox b1 and the
    # nearest edge of bbox b2
    vd_b1c_b2_bottom_edge = abs(b2[1] - b1_c[1])  # abs(b2_ymin - b1_c_y)
    vd_b1c_b2_top_edge = abs(b2[3] - b1_c[1])     # abs(b2_ymax - b1_c_y)
    vd_b1c_b2_nearest_edge = min(vd_b1c_b2_bottom_edge, vd_b1c_b2_top_edge)
    vd_b1c_b2_nearest_edge_ratio = vd_b1c_b2_nearest_edge / im_h

    # calc horizontal distance between centroid of bbox b2 and the
    # nearest edge of bbox b1
    vd_b2c_b1_bottom_edge = abs(b1[1] - b2_c[1])    # abs(b1_ymin - b2_c_y)
    vd_b2c_b1_top_edge = abs(b1[3] - b2_c[1])   # abs(b1_ymax - b2_c_y)
    vd_b2c_b1_nearest_edge = min(vd_b2c_b1_bottom_edge, vd_b2c_b1_top_edge)
    vd_b2c_b1_nearest_edge_ratio = vd_b2c_b1_nearest_edge / im_h
    
    return vd_b1c_b2_nearest_edge_ratio, vd_b2c_b1_nearest_edge_ratio

#%%

def calc_bbox_geometric_features(b1, b2, im_w, im_h):
    '''
    For a given ordered pair of bboxes, calculate and return the full set
    of bbox geometric features that we might wish to use as training data
    features with which to train our predicate prediction neural networks
    (PPNNs).
    '''
    
    results = {}
    
    # number of decimals for rounding
    ndec = 3
    
    
    #
    # geometric features of individual bboxes
    #
    
    
    # bbox aspect ratios (height / width)
    # (Values are in the range [0, infty] in theory, but in practice they
    #  will mostly be in a range of, say, [0, 5], with objects like 'posts'
    #  and 'towers' having larger ranges up to, say, [0, 10], and objects
    #  like 'poles' being outliers with the largest range of say [0, 20].)
    # (Given that extreme values will be rare, we are opting, initially,
    #  to work with the raw aspect ratio values and to not worry about
    #  scaling/normalising them into narrower intervals in some fashion.
    #  And the basis for any such scaling/normalising is not obvious.)
    b1_ar, b2_ar = bb_aspect_ratios(b1, b2)
    # aspect ratio (h/w) of bbox b1
    results['b1_ar'] = round(b1_ar, ndec)
    # aspect ratio (h/w) of bbox b2
    results['b2_ar'] = round(b2_ar, ndec)
    
    # ratios of bbox area to image area
    # (values are always in the unit interval [0,1])
    ba2ia_b1, ba2ia_b2 = bb_area_to_image_area_ratios(b1, b2, im_w, im_h)
    # ratio of area of box b1 to image area
    results['b1a2ia'] = round(ba2ia_b1, ndec)
    # ratio of area of box b2 to image area
    results['b2a2ia'] = round(ba2ia_b2, ndec)
    
    
    #
    # features of spatial relationships between an ordered pair of bboxes
    #
    
    
    # Ratios of Euclidean distance between bbox centroids to the 
    # image width & height
    # (values are always in the unit interval [0,1])
    res = bb_euclidean_distance_centroids_ratios(b1, b2, im_w, im_h)
    eucl_dist_to_width_ratio = res[0]
    eucl_dist_to_height_ratio = res[1]
    # ratio of Euclidean distance between bbox centroids relative 
    # to image width
    results['eucl_dist_to_width_ratio'] = round(eucl_dist_to_width_ratio, ndec)
    # ratio of Euclidean distance between bbox centroids relative 
    # to image height  
    results['eucl_dist_to_height_ratio'] = round(eucl_dist_to_height_ratio, ndec)
    
    # Sine and Cosine of angle between bbox centroids
    # (values are always in the interval [-1,1])
    res = bb_sine_and_cosine_of_angle_between_centroids(b1, b2)
    sine_theta = res[0]
    cosine_theta = res[1]
    # sine of angle between bbox centroids
    results['sine'] = round(sine_theta, ndec)
    # cosine of angle between bbox centroids
    results['cosine'] = round(cosine_theta, ndec)
    
    # Intersection of Union (IoU)
    # (values are always in the unit interval [0,1])
    iou = bb_intersection_over_union(b1, b2)
    results['iou'] = round(iou, ndec)
    
    # Inclusion ratios
    # (values are always in the unit interval [0,1])
    ir_b1b2, ir_b2b1 = bb_inclusion_ratios(b1, b2)
    # inclusion ratio of b1 within b2
    # degree (extent) to which b1 is enclosed within b2
    results['ir_b1b2'] = round(ir_b1b2, ndec)
    # inclusion ratio of b2 within b1
    # degree (extent) to which b2 is enclosed within b1
    results['ir_b2b1'] = round(ir_b2b1, ndec)  
    
    # ratio of horiz distance between right & left edges of two bboxes
    # relative to image width
    # (values are always in the unit interval [0,1])
    hde2iwr = bb_horiz_dist_edges_ratio(b1, b2, im_w)
    results['hde2iwr'] = round(hde2iwr, ndec)
    
    # ratio of vert distance between top & bottom edges of two bboxes
    # relative to image height
    # (values are always in the unit interval [0,1])    
    vde2ihr = bb_vert_dist_edges_ratio(b1, b2, im_h)
    results['vde2ihr'] = round(vde2ihr, ndec)
    
    # ratio of the horizontal distance between the centroids of two bboxes 
    # relative to the image width. Values in [0,1].
    hdc2iwr = bb_horiz_dist_centroids_ratio(b1, b2, im_w)
    results['hdc2iwr'] = round(hdc2iwr, ndec)
    
    # ratio of the vertical distance between the centroids of two bboxes 
    # relative to the image height. Values in [0,1].   
    vdc2ihr = bb_vert_dist_centroids_ratio(b1, b2, im_h)
    results['vdc2ihr'] = round(vdc2ihr, ndec)    
    
    # horiz dist between bbox centroids and nearest edges, as ratios of img width 
    res = bb_horiz_dist_centroids_to_nearest_edges(b1, b2, im_w)
    results['hd_b1c_2_b2ne_r'] = res[0]
    results['hd_b2c_2_b1ne_r'] = res[1]
    
    # vert dist between bbox centroids and nearest edges, as ratios of img height
    res = bb_vert_dist_centroids_to_nearest_edges(b1, b2, im_h)
    results['vd_b1c_2_b2ne_r'] = res[0]
    results['vd_b2c_2_b1ne_r'] = res[1]
    
    return results


   
    

    









  



















