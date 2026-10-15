#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: David Herron
"""

'''
This module defines a PyTorch Dataset class for our VRD object detection.

The VRDDataset class is designed to serve image data for a Faster-RCNN
object detection model, in particular the Torchvision implementation
of a faster-rcnn-resnet50-fpn model. It loads (returns) one example
from the NeSy4VRD dataset that corresponds to a specific key (integer  
index). The example returned consists of 1 image (minimally transformed,
because the faster-rcnn-resnet50-fpn model performs its own 
transformations internally) and the image's corresponding targets (a 
dictionary of bounding boxes and corresponding class labels).
'''

import torch
import torchvision.transforms as transforms
import numpy as np
import os
import PIL
import sys

vrd_utils_dir_path = os.path.join('~','archive', 'repo-vrd-kg', 'extensibility', 'analysis')
vrd_utils_dir_path = os.path.expanduser(vrd_utils_dir_path)
sys.path.insert(0, vrd_utils_dir_path)
import nesy4vrd_utils as vrdu


#class VRDDataset(torch.utils.data.Dataset):
class NeSy4VRD_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, vrd_img_dir, nesy4vrd_anno_file, img_names):
        super(NeSy4VRD_Dataset, self).__init__()

        self.img_dir = vrd_img_dir

        self.img_anno = vrdu.load_VRD_image_annotations(nesy4vrd_anno_file)
        
        #self.img_names = list(self.img_anno.keys())
        
        # a List of the image names available for processing
        self.img_names = img_names

    def transform_img(self, img):
        # convert PIL image of mode RGB (shape HxWxC with pixels in range [0,255])
        # to torch.FloatTensor image (shape CxHxW with pixels in range [0,1])
        img = transforms.ToTensor()(img)
        return img

    def get_img(self, idx):
        imgfilepath = os.path.join(self.img_dir, self.img_names[idx])
        img = PIL.Image.open(imgfilepath)
        img = self.transform_img(img)
        return img

    def restructure_bboxes(self, bbox_coords):
        # IMPORTANT:
        # Restructure the VRD bbox coordinates, format [ymin, ymax, xmin, xmax],
        # to the format expected by a PyTorch Faster R-CNN model,
        # format [xmin, ymin, xmax, ymax].
        bboxes = []
        for bbox in bbox_coords:
            bbox2 = (bbox[2], bbox[0], bbox[3], bbox[1])
            bboxes.append(bbox2)
        return bboxes            

    def increment_class_indices(self, bbox_classes):
        # IMPORTANT:
        # The VRD object classes do NOT include a class for 'background'
        # (aka 'unlabelled'). But a PyTorch Faster R-CNN model expects class
        # index 0 to correspond to a class of 'background'. So we need to 
        # increment every VRD class index by 1 so as to make room for a
        # new object class of 'background' with class index 0.
        classes = list(np.array(bbox_classes) + 1)
        return classes

    def get_targets(self, idx):    
        img_name = self.img_names[idx]
        img_anno = self.img_anno[img_name]
        bboxes = vrdu.get_bboxes_and_object_classes(img_name, img_anno)
        bbox_coords = list(bboxes.keys())
        bbox_coords = self.restructure_bboxes(bbox_coords) 
        bbox_classes = list(bboxes.values())
        bbox_classes = self.increment_class_indices(bbox_classes)
        targets = {}
        # nb: the model expects bboxes to have dtype=float
        targets['boxes'] = torch.as_tensor(bbox_coords, dtype=torch.float)
        # nb: the model expects labels to have dtype=int64
        targets['labels'] = torch.as_tensor(bbox_classes, dtype=torch.int64)
        return targets

    def __len__(self):
        # return size of dataset
        return len(self.img_names)

    def __getitem__(self, idx):
        # return an image and its annotated bboxes and class labels
        img = self.get_img(idx)        
        targets = self.get_targets(idx) 
        return idx, img, targets

