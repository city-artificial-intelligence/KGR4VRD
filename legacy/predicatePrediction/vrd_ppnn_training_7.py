#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script trains Predicate Prediction neural network (PPNN) models.

This script corresponds to Step 2 of the PPNN pipeline v4.

This version of the training script incorporates early stopping using a
validation set carved out of the baseline NeSy4VRD training set.
'''

#%%

import torch

import os
import json
import time
import sys
from datetime import date

import vrd_ppnn_models as vpm
import vrd_ppnn_loss as vls
from vrd_ppnn_dataset import PPNNDataset
import vrd_utils9 as vrdu9
import vrd_utils13 as vrdu13
import vrd_utils14 as vrdu14

import nesy4vrd_utils4 as vrdu4
import vrd_utils17 as vrdukg 


#%% gather arguments supplied to this script

# set the experiment space training region cell id
if len(sys.argv) > 1:
    trc_id = sys.argv[1]
else:
    trc_id = 'trc0460'  # 'trc0460'
if not trc_id.startswith('trc'):
    raise ValueError(f'training region cell id {trc_id} not recognised')

# set the platform on which the training script is running
if len(sys.argv) > 2:
    platform = sys.argv[2]
else:
    platform = 'macstudio'
if not platform in vrdu14.supported_platforms:
    raise ValueError(f'platform {platform} not recognised')

# set the work directory (the folder in which to store output files)
if len(sys.argv) > 3:
    workdir = sys.argv[3]
else:
    workdir = 'ppnn'


#%% lookup config of the experiment space training region cell

# specify the experiment family
experiment_family = 'nnkgs0'      # NN+KG_S0

#
# get the training region cell configuration (dimension levels)
#

cfg = vrdu14.get_training_region_cell_config(experiment_family, trc_id)

# set the training region dimension levels
tr_d_model1 = cfg['D_model1_level']
tr_d_model2 = cfg['D_model2_level']      # new - add extra output neuron or not
tr_d_dataCat = cfg['D_dataCat_level']
tr_d_dataFeat = cfg['D_dataFeat_level']
tr_d_kgS0 = cfg['D_kgS0_level']                  # was tr_d_target
tr_d_kgS1 = cfg['D_kgS1_level']
tr_d_kgS2 = cfg['D_kgS2_level']
tr_d_kgS3 = cfg['D_kgS3_level']
tr_d_nopredTarget = cfg['D_nopredTarget_level']  # was tr_d_loss
tr_d_onto = cfg['D_onto_level']


#%% lookup config of the experiment space prediction region cell

# NOTE: We use the configuration of a fixed experiment space prediction 
# region cell for doing prediction selection as part of doing recall@N-based 
# performance evaluation on the validation set. We do validation set
# processing as part of enabling 'early stopping' on PPNN model training.

# The experiment space prediction region cell we use for validation set
# processing is 'prc017' or 'prc018'
if tr_d_model2 == 1:
    prc_id = 'prc017'   # because D_predNoPred=1
else:
    prc_id = 'prc018'   # because D_predNoPred=2 (ie != 1)

# get the prediction region cell configuration (dimension levels)
cfg = vrdu14.get_prediction_region_cell_config(experiment_family, prc_id)

# assign the dimension levels to more familiar variable names
pred_conf_thresh = cfg['D_predConf_level']        
max_preds_per_obj_pair = cfg['D_predMax_level']   
if cfg['D_predKG_level'] == 1: 
    kg_filtering = False                          
else:
    kg_filtering = True
nopred_prediction_policy = cfg['D_predNoPred_level']        

# repackage the prediction region cell configuration
prediction_region_config = {}
prediction_region_config['pred_region_id'] = prc_id
prediction_region_config['pred_conf_thresh'] = pred_conf_thresh
prediction_region_config['max_preds_per_obj_pair'] = max_preds_per_obj_pair
prediction_region_config['kg_filtering'] = kg_filtering
prediction_region_config['nopred_prediction_policy'] = nopred_prediction_policy


#%% build the path to the model checkpoint directory for storing outputs

if platform == 'hyperion':
    model_checkpoint_dir = os.path.join('~', 'sharedscratch', 'research', workdir)
else:
    model_checkpoint_dir = os.path.join('~', 'research', workdir)

scriptName = os.path.basename(__file__)
print(f'script name: {scriptName}')

model_checkpoint_dir = os.path.expanduser(model_checkpoint_dir)
print(f"work dir   : {model_checkpoint_dir}")


#%% choose whether or not to redirect stdout to a file

# redirecting to a file allows one to retain a log of the training run

redirect_stdout = True


#%% build the name of the training log file (if redirecting stdout)

if redirect_stdout:
    training_log_filename = 'vrd_ppnn_' + trc_id + '_ckpt_training_log.txt'
else:
    training_log_filename = ''


#%% redirect stdout

if redirect_stdout:
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(model_checkpoint_dir, training_log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')


#%% declare (record) experiment meta data

print()
print(f'# experiment family: {experiment_family}')
print(f'# training region cell id: {trc_id}')
print(f'# prediction region cell id: {prc_id} (per validation set)')
print()
print(f'Date: {date.today()}')


#%% establish the device (CPU, GPU, MPS) on which we're training

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# force use of cpu    
#device = torch.device('cpu')


#%% record key info in the training log file

# the platform on which we're running
print(f'Platform: {platform}')

# the name of this script
#scriptName = sys.argv[0]
scriptName = os.path.basename(__file__)
print(f'Script: {scriptName}')

# the device (CPU, GPU, MPS) upon which we're running
print(f'Device: {device}')

# the name of the conda environment in which we're running
print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")

# the version of PyTorch we're using
print(f'PyTorch version: {torch.__version__}')


#%% get and count the VRD object class names

# set path to VRD object class names
vrd_obj_file = os.path.join('..', 'data', 'annotations',
                            'nesy4vrd_objects.json')

# get the VRD object class names
with open(vrd_obj_file, 'r') as file:
    vrd_object_class_names = json.load(file)

# count the object class names
nr_object_classes = len(vrd_object_class_names)

print(f"Nr of object classes: {nr_object_classes}")


#%% convert the NeSy4VRD object class names to ontology VRD-World class names

ontoClassNames = vrdu4.convert_NeSy4VRD_classNames_to_ontology_classNames(vrd_object_class_names)


#%% get and count the VRD predicate names

# set path to VRD predicate names
vrd_pred_filepath = os.path.join('..', 'data', 'annotations',
                                 'nesy4vrd_predicates.json')

# get the VRD predicates names
with open(vrd_pred_filepath, 'r') as file:
    vrd_predicate_names = json.load(file)

# count the predicate names
nr_predicates = len(vrd_predicate_names)

print(f"Nr of predicates: {nr_predicates}")


#%% convert the NeSy4VRD predicate names to ontology VRD-World object property names

ontoPropNames = vrdu4.convert_NeSy4VRD_predicateNames_to_ontology_propertyNames(vrd_predicate_names)


#%% configure the training set annotated VRs to be used as training targets

# The levels of dimensions D_kgS0 and D_onto of the experiment space 
# training region together determine the particular file of VR annotations
# that we load.  From these are constructed the target vectors used for
# the calculation of loss and, hence, for the supervising of PPNN training.

if tr_d_kgS0 == 1:
    # use the initial (sparse & arbitrary) training set VR annotations as the
    # source from which to derive target vectors
    anno_dir = os.path.join('..', 'data', 'annotations')
    filename = 'nesy4vrd_annotations_train.json'
    
    # Note: in this setting, dimension D_onto, represented by tr_d_onto,
    # can take any level and the experiment configuration may still be valid.
    # In other words, it's not necessarily the case that we should enforce
    # that tr_d_onto==1. This is because dimension D_onto might specify
    # that an ontology is used with the experiment, and this use could be
    # only in relation to the source of the PPNN input data (dimension
    # D_dataCat). For example, it would be valid to have tr_d_S0==1,
    # tr_d_onto==2 as long as tr_d_dataCat==3.
    
elif tr_d_kgS0 == 2:
    # use KG-augmented training set VR annotations as the source from which to
    # derive target vectors
    #
    # the file to be loaded depends on the version of the VRD-World ontology 
    # that was used to drive the KG-augmentation (KG materialisation), and
    # that's what tr_d_onto, the level of dimension D_onto, gives us
    anno_dir = os.path.join('..', 'data', 'annotations_augmented')
    if tr_d_onto == 1:
        raise ValueError(f'level {tr_d_onto} of dimension D_onto not valid here')
    elif tr_d_onto == 2:  # symmetry, transitivity, and inverses
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1b.json'
    elif tr_d_onto == 3:  # full version
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1.json'
    elif tr_d_onto == 4:  # rdfs:subPropertyOf and owl:equivalentProperty
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1c.json'
    elif tr_d_onto == 5:  # owl:SymmetricProperty
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1g.json'
    elif tr_d_onto == 6:  # owl:TransitiveProperty
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1h.json'
    elif tr_d_onto == 7:  # owl:inverseOf
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1i.json'
    elif tr_d_onto == 8:  # rdfs:subPropertyOf
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1j.json'
    elif tr_d_onto == 9:  # owl:equivalentProperty
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1k.json'
    elif tr_d_onto == 10:  # owl:SymmetricProperty and owl:inverseOf
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1e.json'
    elif tr_d_onto == 11:  # owl:TransitiveProperty, rdfs:subPropertyOf and owl:equivalentProperty    
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1n.json'
    elif tr_d_onto == 12:  # sym, trans, subProp, equivProp
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1p.json'
    elif tr_d_onto == 13:  # trans, inverseOf, subProp, equivProp
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1r.json'    
    elif tr_d_onto == 14:  # trans, inverseOf
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1f.json' 
    elif tr_d_onto == 15:  # sym, trans
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1d.json'
    elif tr_d_onto == 16:  # sym, equivProp
        filename = 'nesy4vrd_annotations_train_augmented_per_onto_v1_1s.json'
    else:
        raise ValueError(f'level {tr_d_onto} of dimension D_onto not recognised')

else:
    raise ValueError(f'level {tr_d_kgS0} of dimension D_kgS0 not recognised')
    
  
anno_filepath = os.path.join(anno_dir, filename)

print(f'Annotations file path: {anno_filepath}')


#%% get the original (full) set of training set image names

# We get the image names from the VR annotations file configured in the
# previous cell. The original (full) set of training set image names never 
# changes; the names are consistent across all training set VR annotations 
# files that might be utilised for an experiment.
# 
# We load the VR annotations here only so we can extract the image names.
# The PPNNDataset object loads the VR annotations as well, using the same
# anno_filepath, so it can construct target vectors from the VR
# annotations for each mini-batch (image-worth) of PPNN input data.

# load the VR annotations for the training set images
with open(anno_filepath, 'r') as file:
    vrd_img_anno = json.load(file)

# extract the full set of training set image names from the annotations
# dictionary; this is the full set of training set image names prior to
# having split the training set into (sub)training and validation sets
vrd_img_names = list(vrd_img_anno.keys())

print(f'Number of training set image names: {len(vrd_img_names)}')


#%% load the NeSy4VRD image names for the (sub)training and validation sets

# directory with files defining the (sub)training and validation sets
validation_dir = os.path.join('..', 'data', 'annotations')

#
# load the image names that represent the (sub)training set
#

filename = 'nesy4vrd_image_names_train_training.json'

filepath = os.path.join(validation_dir, filename)

# load the image names defining the (sub)training set that remains from the
# original (full) training set after having carved out a validation set
with open(filepath, 'r') as fp:
    vrd_img_names_training = json.load(fp)

print(f'Image names for (sub)training set loaded: {len(vrd_img_names_training)}')


#
# load the image names that represent the validation set
#

filename = 'nesy4vrd_image_names_train_validation.json'

filepath = os.path.join(validation_dir, filename)

# load the image names defining the validation set carved out of the
# original (full) training set
with open(filepath, 'r') as fp:
    vrd_img_names_validation = json.load(fp)

print(f'Image names for validation set loaded: {len(vrd_img_names_validation)}')


#%% configure the correct PPNN input data file for training

# The level of dimension D_dataCat of the experiment space training region
# identifies the particular PPNN input data file to be used for the 
# current experiment.

if tr_d_dataCat == 1:
    # use ppnn input data derived from the objects detected in the 
    # training set images by our FRCNN object detector; this level of
    # dimension D_dataCat implies that the current experiment belongs to
    # the 'visual relationship detection' regime of experiments
    #data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_1_trainset.json'
    #data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_2_trainset.json'
    #data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_3_trainset.json'
    data_filename = 'ppnn_input_rd_v2_1_1_checkpoint_250_ihc_4_trainset.json'
elif tr_d_dataCat == 2:
    # use ppnn input data derived from the initial (sparse, arbitrary)
    # VR annotations of the training set images; this level of dimension 
    # D_dataCat implies that the current experiment belongs to the 
    # 'predicate detection' regime of experiments
    data_filename = 'ppnn_input_pp_per_nesy4vrd_annotations_train.json'
#elif tr_d_dataCat == 3:
    # use ppnn input data derived from the KG-augmented VR annotations of 
    # the training set images; this level of dimension D_dataCat implies
    # that the current experiment belongs to the 'predicate detection'
    # regime of experiments    
#    data_filename = 'to be determined'
else:
    raise ValueError(f'level {tr_d_dataCat} of dimension D_dataCat not recognised')

data_filepath = os.path.join('..', 'data', 'ppnn_input_data', data_filename)

print(f'PPNN input data file: {data_filepath}')


#%% instantiate a PPNNDataset object

# The level of dimension D_dataFeat of the experiment space training region
# identifies the particular PPNN input data feature set that is to be used
# for the current experiment.

if not tr_d_dataFeat in [1,2,3,4]:
    raise ValueError(f'PPNN input data feature set {tr_d_dataFeat} not recognised')

# the feature set of PPNN input data to be used for training
featureset = tr_d_dataFeat

print(f'PPNN input data feature set: {featureset}')

# instantiate our VRD Dataset class
dataItem = PPNNDataset(data_filepath=data_filepath,
                       featureset=featureset,
                       anno_filepath=anno_filepath,
                       nr_object_classes=nr_object_classes,
                       nr_predicates=nr_predicates,
                       targets_required=True)

print(f'PPNNDataset object has data for this many images: {len(dataItem)}')


#%% get the size of the PPNN input data feature vectors

# specify an arbitrary training image in terms of its index position
# img_idx=2   30 feature vectors
# img_idx=6    6 feature vectors
# img_idx=10  30 feature vectors
# img_idx=12   6 feature vectors
# img_idx=13  12 feature vectors
img_idx = 6    # 6

# access the ppnn input data for an arbitrary image
results = dataItem[img_idx]
inputdata = results['inputdata']

in_features = inputdata.shape[1]

print(f'Size of PPNN input data feature vectors: {in_features}')


#%% instantiate the appropriate PPNN model architecture

# Dimensions D_model1 and D_model2 together specify the precise PPNN
# model architecture to be used.

# apply the policy indicated by the level of dimension D_model2
# (nb: this is the primary purpose of dimension D_model2 --- to indicate
# whether or not our PPNN model is to have a special neuron in its output
# layer that represents a notional 'no predicate' predicate)
if tr_d_model2 == 1:
    out_size = nr_predicates
elif tr_d_model2 == 2:
    out_size = nr_predicates + 1
else:
    raise ValueError(f'level {tr_d_model2} of dimension D_model2 not recognised')

# apply the policy indicated by the level of dimension D_model1
if tr_d_model1 == 1:
    model = vpm.PPNN_1(in_features=in_features, out_size=out_size)
elif tr_d_model1 == 2:
    raise ValueError(f'level {tr_d_model1} of dimension D_model1 not yet implemented')
    #model = vpm.PPNN_2(in_features=in_features, out_size=out_size)
elif tr_d_model1 == 3:
    raise ValueError(f'level {tr_d_model1} of dimension D_model1 not yet implemented')
    #model = vpm.PPNN_3(in_features=in_features, out_size=out_size)
else:
    raise ValueError(f'level {tr_d_model1} of dimension D_model1 not recognised')

model_name = model.__class__.__name__

print(f'Model instantiated: {model_name}')
print(f'Model output layer size: {out_size}')


#%% optionally try compiling the model (per PyTorch 2)

# specify whether or not we are compiling the model using torch.compile()
# (nb: under PyTorch 2.0.1, using torch.compile() only works if PyTorch was
#  compiled with CUDA, not MPS or CPU-only versions of PyTorch)
compile_model = False

# set torch._dynamo.config.verbose=True to get more info if there's a 
# problem using torch.compile() or the model we compile using it
#td.config.verbose = True

mode = 'default'
#mode = 'reduce-overhead'
#mode = 'max-autotune'

#fullgraph = True
fullgraph = False

if compile_model:
    print(f'Compiling model; mode: {mode}, fullgraph: {fullgraph}')
    start_time = time.time()
    model = torch.compile(model, mode=mode, fullgraph=fullgraph)
    end_time = time.time()
    compile_time = (end_time - start_time) / 60
    print(f"Compile time: {compile_time:.2f} minutes\n")

# RuntimeError: Python 3.11+ not yet supported for torch.compile
# 
# but it works ok with Python 3.10


#%% set model mode and place model on correct device

# put model in training mode
model.train()

# if we are using a device other than the CPU, push the model to it
if device != torch.device('cpu'):
    model = model.to(device)


#%% establish optimiser

# (NOTE: we instantiate our optimiser AFTER having set the location of our
#  model (GPU or CPU). This is a recommended convention because it's required
#  to do this 'after' shifting a model to a GPU in cases where the selected
#  optimiser maintains internal state, such as the Adagrad optimiser)

learning_rate = 1e-4
weight_decay_rate = 1e-3

opt_args = {'lr': learning_rate, 'weight_decay': weight_decay_rate}
optimiser = torch.optim.Adam(model.parameters(), **opt_args)

print('Adam optimiser instantiated')
print(f'learning_rate = {learning_rate}, weight_decay = {weight_decay_rate}')

# NOTE: we MUST instantiate the optimiser here regardless of whether we
# are starting a new training run (with training_mode = 'start') or 
# continuing a training run (with training_mode = 'continue'), (see below).
# But, when training_mode = 'continue', some or all of the arguments used in
# the instantiation will be over-written when the saved model checkpoint file
# is loaded and the saved state of the optimiser in the previous training run
# is restored to our optimiser object. It's simplest to just let this
# over-writing happen rather than trying to avoid it.  There's no real need
# to avoid it, as long as users aren't confused into thinking that a certain
# learning rate is in effect when, in fact, a different learning rate is
# actually active.


#%% configure validation set processing

# set the training epoch number at which we want validation set processing
# to begin; we can avoid a lot of unnecessary validation computation by 
# learning to set this value optimally, based on experience;
# nb: to fully deactivate validation set processing, set the starting
# epoch to a large number (eg 500)
start_epoch_for_validation = 1

# specify whether or not we want validation loss to be calculated as part
# of validation set processing;
# note: this setting does NOT control whether we perform validation set
# processing or not; it merely determines, when we do perform validation
# set processing, whether or not we calculate validation loss along with
# validation predictive performance (recall@50)
calculate_val_loss = True

# set 'loss reduction' to control a particular aspect of loss calculation;
# this setting applies to both loss calculated on the training set and loss
# calculated on validation set (if that is requested)
# (options: 'none', 'mean', 'sum')
loss_reduction = 'none'

print(f'Calculate validation loss: {calculate_val_loss}')


#%% configure parameters affecting the calculation of loss

# set the policy for handling 'no predicate' target vectors during the 
# calculation of loss
#
# note: the choice of policy is specified by the level of dimension D_loss
# of the experiment space training region
nopred_target_policy = tr_d_nopredTarget


# ALERT: The policy specified here currently applies to the handling of
# 'no predicate' target vectors during the calculation of validation loss
# as well as training loss.  The time may come when this arrangement becomes
# a constraint and we find we need two separate variables so that the
# 'no predicate' target vector handling policy can be set independently
# for training loss and validation loss calculation.
#
# This issue, however, should not affect the 'early stopping' behaviour
# of this training script because we've settled upon using validation
# predictive performance (recall@N on the validation set) as the score stream
# to be monitored for 'early stopping' purposes, not validation loss as the
# score stream.


#%% specify the base pattern for model checkpoint file names

# specify the filename pattern to be shared by all model checkpoint files
# saved by the current run of this training script; the choice of filename
# pattern must include the trc_id so we can link all the checkpoint
# files to a particular experiment (experiment space region)

model_checkpoint_filename_base = 'vrd_ppnn_' + trc_id + '_ckpt_'

print(f"Model checkpoint filename base: {model_checkpoint_filename_base}")


#%%

def save_model_checkpoint(model_checkpoint_filename_base, epoch, model, 
                          optimiser, avg_loss_per_mb):
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimiser_state_dict': optimiser.state_dict(),
                  'avg_loss_per_mb': avg_loss_per_mb}

    filename = model_checkpoint_filename_base + str(epoch).zfill(3) + '_model.pth'
    path = os.path.join(model_checkpoint_dir, filename)
    torch.save(checkpoint, path)
    print(f"\nModel checkpoint file saved: {filename}")

    return None


#%%

def load_model_checkpoint(filepath, model, optimiser, device):
    # load checkpoint file; note that we do NOT set parameter map_location=device,
    # which indicates the location where ALL tensors should be loaded;
    # this is because we want our model on the GPU but our optimiser on the CPU,
    # which is where they are both saved from when torch.save() is executed
    #checkpoint = torch.load(filepath)
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # initialise the model and optimiser state (in-place) to the saved state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    # since we load the model onto a CPU, if we're on a machine with a GPU
    # we need to push the model to the GPU
    if device == torch.device('cuda') or device == torch.device('mps'):
        model = model.to(device)

    # get other saved variables
    epoch = checkpoint['epoch']
    avg_loss_per_mb = checkpoint['avg_loss_per_mb']

    print(f"Model checkpoint file loaded: {filepath}")
    
    print(f"Optimiser learning rate restored to: {optimiser.param_groups[0]['lr']}")
    print(f"Optimiser weight decay restored to: {optimiser.param_groups[0]['weight_decay']}")

    return epoch, avg_loss_per_mb

    # NOTE: the 'model' and 'optimiser' are mutable and are updated in-place
    # by this function

    # NOTE: we may save a model checkpoint when running on a machine with a
    # GPU and then later copy the model checkpoint to a Mac (that has only a
    # CPU), and then later try to load that model checkpoint on a Mac. 
    # if we use 'torch.load(filepath)' we get a RuntimeError on a Mac, saying:
    #
    # RuntimeError: Attempting to deserialize object on a CUDA device but 
    # torch.cuda.is_available() is False. If you are running on a CPU-only 
    # machine, please use torch.load with map_location=torch.device('cpu') 
    # to map your storages to the CPU.
    #
    # But if we do what the error message recommends, we then need to push
    # the model over the GPU again.


#%% define functions for loading/saving KG interaction results per image  

def load_kg_interaction_results(filepath):
    
    with open(filepath, 'r') as fp:
        kg_interaction_results_per_image = json.load(fp)
    
    return kg_interaction_results_per_image


def save_kg_interaction_results(filepath, kg_interaction_results_per_image):
    
    with open(filepath, 'w') as fp:
        json.dump(kg_interaction_results_per_image, fp)

    return None 

# if we're doing KG interaction for nnkgs1, specify the path to the
# KG interaction results .json file 
if tr_d_kgS1 == 2:
    # specify path to KG interaction results file 
    kg_interaction_filename = 'vrd_ppnn_' + trc_id + '_ckpt_training_kg_interaction_results.json'
    kg_interaction_filepath = os.path.join(model_checkpoint_dir, kg_interaction_filename) 
    print(f'KG interaction results file: {kg_interaction_filename}')

# TODO: remove once we done with QA testing
if tr_d_kgS1 == 2:
    # specify path to KG interaction results test file 
    kg_interaction_filename_test = 'vrd_ppnn_' + trc_id + '_ckpt_training_kg_interaction_results_test.json'
    kg_interaction_filepath_test = os.path.join(model_checkpoint_dir, kg_interaction_filename_test) 
    print(f'KG interaction results test file: {kg_interaction_filename_test}')


#%% define functions for loading/saving KG VR type classifications  

def load_kg_vr_type_classifications(filepath):
    
    with open(filepath, 'r') as fp:
        kg_vr_type_classifications = json.load(fp)
    
    # convert the serialised JSON data (list of lists of lists of reals)
    # to a tensor
    kg_vr_type_classifications = torch.tensor(kg_vr_type_classifications)
    
    return kg_vr_type_classifications


def save_kg_vr_type_classifications(filepath, kg_vr_type_classifications):
    
    # convert the 3D tensor to JSON serialisable data (a list of lists of lists of reals)
    kg_vr_type_classifications = kg_vr_type_classifications.numpy().tolist()    
    
    with open(filepath, 'w') as fp:
        json.dump(kg_vr_type_classifications, fp)

    return None 

# if we're doing KG interaction for nnkgs1, specify the path to the
# master 3D VR type tensor for nnkgs1 (ie for training)
if tr_d_kgS1 == 2:
    # specify path to master 3D VR type tensor for nnkgs1 (training)
    kg_vr_type_classifications_filename = 'master-vr-type-tensor-nnkgs1.json'
    central_results_dir = os.path.join('~', 'research', 'results', 'nnkgs0')
    central_results_dir = os.path.expanduser(central_results_dir)
    kg_vr_type_classification_filepath = os.path.join(central_results_dir, kg_vr_type_classifications_filename) 
    print('master 3D VR type tensor file to be used:')
    print(f'{kg_vr_type_classification_filepath}')


#%% configure training loop parameters

# set training mode: 'start' or 'continue'
# - 'start' : no checkpoint file to load; begin training from epoch 0
# - 'continue' : load checkpoint file and resume training from last epoch
training_mode = 'continue'

# set model checkpoint epoch from which to resume training
# (nb: if training_mode = 'start', continue_from_epoch is ignored)
continue_from_epoch = 5

# set number of epochs for which to train the model
n_epochs_train = 75
print(f'Number of epochs to train: {n_epochs_train}')

# set the training epoch number at which to start saving model checkpoint
# files; models for early epochs likely won't get evaluated on the test set,
# so saving them will likely be redundant; but saving a model takes no time,
# and sometimes it will be useful to have checkpoints for every epoch
# available; so, in practice, we'll tend to set this to 1 most of the time
# so that we save a checkpoint file for every epoch, even the very first ones
start_epoch_for_saving_model_checkpoints = 1

# set frequency (in number of epochs) for saving model checkpoint files
n_epochs_checkpoint = 1

# set frequency (in number of epochs) for saving PPNN logit (output) data, 
# for specified images, with which to monitor and visually analyse the 
# quality of the learning as the number of training epochs grows
n_epochs_monitor = 1

# set frequency (in number of mini-batches) for writing the average loss 
# per mini-batch to the training log file; ie report loss every N mini-batches
#n_mb_report_loss = 1  # convention for test runs is 2 or even 1
n_mb_report_loss = 16  # convention for production runs


#%% set up the images whose logits we wish to monitor (if any)

# specify the images (if any) for which we wish to capture logits output by
# the PPNN model during training so we can see and review how they change 
# as the training epochs advance. these images are specified in terms of their 
# index (position) within the list of image names for the (sub)training set.
#
# 9, 10, 11 are worth monitoring

#image_idxs_to_monitor = []
image_idxs_to_monitor = [9,10]  

print(f'Capturing logits for (sub)training set image indices: {image_idxs_to_monitor}')

# record the image indices and names in the training log file 
for idx, img_idx in enumerate(image_idxs_to_monitor):
    img_name = vrd_img_names_training[img_idx]
    print(f'Image idx: {img_idx}, image name: {img_name}')


#%% configure early stopping 

# activate / deactivate 'early stopping' 
# (nb: 'early stopping' and 'validation set processing' are independently
#  controlled; but activating early stopping without doing validation set
#  processing won't achieve anything)
early_stopping_active = True

# specify which score stream is to be used for early stopping:
# validation loss ('loss') or validation recall@50 predictive performance
# ('performance')
#
# (nb: we have settled on 'performance'; so the early stopping monitor
# monitors a stream of recall@50 performance scores and bases its early
# stopping decisions on the behaviour of the scores in that stream)
#
# (nb: this setting is ignored unless early stopping is activated)
#early_stopping_score_type = 'loss' 
early_stopping_score_type = 'performance'

# early stopping policy: patience=25, evaluate=35

# if early stopping is activated, instantiate an early stopping monitor
# to track a score stream derived from validation set processing
if early_stopping_active:
    early_stopper = vrdu13.EarlyStoppingMonitor(patience=25, 
                                                min_delta=0.0, 
                                                verbose=False, 
                                                score_type=early_stopping_score_type)

print(f'Start epoch for validation: {start_epoch_for_validation}')
print(f'Early stopping active: {early_stopping_active}')


#%%  more setup for the training loop

if training_mode == 'start':
    
    last_epoch = 0
    print('New training run starting ...')
    
    # if we're doing KG interaction for nnkgs1, initialise the tools we use
    # to manage it
    if tr_d_kgS1 == 2:
        # initialise a fresh dictionary for storing image-specific KG
        # interaction (KG VR instance classification) results (decisions)
        kg_interaction_results_per_image = {}
        
        # TODO: remove this when we're done doing QA testing
        kg_interaction_results_per_image_test = {}
        
        # load the master 3D VR type tensor 
        kg_vr_type_classifications = load_kg_vr_type_classifications(kg_vr_type_classification_filepath) 
    
elif training_mode == 'continue':
    
    # load model checkpoint file and continue training from there
    filename = model_checkpoint_filename_base + str(continue_from_epoch).zfill(3) 
    filename = filename + '_model.pth'
    filepath = os.path.join(model_checkpoint_dir, filename)
    args = {'filepath': filepath, 'model': model, 
            'optimiser': optimiser, 'device': device}
    last_epoch, last_loss = load_model_checkpoint(**args)
    print(f"last epoch: {last_epoch}; last avg loss per mb: {last_loss:.4f}")
    print('Existing training run continuing ...')
    
    # if we're doing KG interaction for nnkgs1, load the tools we use to manage it
    if tr_d_kgS1 == 2:
        kg_interaction_results_per_image = load_kg_interaction_results(kg_interaction_filepath)
        kg_interaction_results_per_image_test = load_kg_interaction_results(kg_interaction_filepath_test)
        kg_vr_type_classifications = load_kg_vr_type_classifications(kg_vr_type_classification_filepath)    
    
else:
    
    raise ValueError('training mode not recognised')

# initialise range of epoch numbers for training loop; if we are continuing
# a training run, this range will begin at 'continue_from_epoch + 1'
first_epoch = last_epoch + 1
final_epoch = first_epoch + n_epochs_train


#%% decide whether to adjust the optimiser learning rate

set_new_learning_rate = False
new_learning_rate = 1e-4

# note: if we want to adjust the learning rate, we have to do it here 'after'
# having loaded the model checkpoint, because loading the checkpoint 
# restores the optimiser state (and learning rate) saved from the previous
# training run

if set_new_learning_rate:
    optimiser.param_groups[0]['lr'] = new_learning_rate
    print(f"Optimiser learning rate changed to: {optimiser.param_groups[0]['lr']}")

# (note: we can use first_epoch and final_epoch to define a schedule
#  for adjusting the learning rate downwards; that way, we could define
#  a schedule upfront and then let the learning rate change automatically,
#  with no need to intervene manually here
#
# for example, a schedule could be:
#if first_epoch < 100:
#    new_learning_rate = 1e-3
#elif first_epoch < 150:
#    new_learning_rate = 1e-4
#else:
#    new_learning_rate = 1e-5
#
# note: defining a schedule here, in our code, may work better than 
# defining a schedule in the optimiser because of the fact that we
# plan to use multiple runs of the training script, with each run doing
# only a small sequence of training epochs; the optimiser scheduling
# facility may well not be sophisticated enough to cope with our needs


#%% configure parameters for performing short tests

# for test mode, we may wish to force the script to stop at a particular
# point; if so, activate that point in the main training loop and set
# this flag to True; otherwise, in normal production mode, this flag should
# always be set to False
force_stop = False

# set max number of images to process per epoch per training set
# (nb: a value of 0 means process all the image entries; non-zero
#  values are used for testing only)
n_images_per_epoch_train = 0

# set max number of images to process per epoch per validation set
# (nb: a value of 0 means process all the image entries; non-zero
#  values are used for testing only)
n_images_per_epoch_val = 0


if n_images_per_epoch_train == 0:
    print('Number of images to process per training epoch: all')
else:
    print(f'Number of images to process per training epoch: {n_images_per_epoch_train}')

if n_images_per_epoch_val == 0:
    print('Number of images to process per validation epoch: all')
else:
    print(f'Number of images to process per validation epoch: {n_images_per_epoch_val}')


#%% specify a value for the nnkgs1 loss penalty regularisation hyperparameter

# note: 
# if we're NOT using nnkgs1, this should be normally be 0.0;
# if we are using nnkgs1, this should normally be 1.0 or more (5, 10, 25, 100, ...)

nnkgs1_lambda = 1.0

if tr_d_kgS1 == 1 and not nnkgs1_lambda == 0.0:
    raise ValueError('unexpected value for nnkgs1_lambda: {nnkgs1_lambda}')

if tr_d_kgS1 == 2 and nnkgs1_lambda < 1.0:
    raise ValueError('unexpected value for nnkgs1_lambda: {nnkgs1_lambda}')
    
# NOTE: we put tight controls here to prevent mistakes; if we want to try
# special case trials, we'll have to deliberately disable these safety checks

if tr_d_kgS1 == 2: 
    print(f'nnkgs1 loss penalty regularisation param: {nnkgs1_lambda}')


#%% specify the KG interaction style to use for experiment family nnkgs1

# valid values:
# 1 - HTTP PUT rdf statements to a named graph
# 2 - HTTP POST rdf statements to the default graph 

kg_interaction_style = 2

if not kg_interaction_style in [1, 2]:
    raise ValueError(f'kg interaction style not recognised: {kg_interaction_style}')

# NOTE: if tr_d_kgS1 == 1, we are not doing nnkgs1 and so not doing any 
# KG interaction; so the value of kg_interaction_style doesn't matter 
# because it's ignored and doesn't effect anything

# specify the epoch when KG interaction is to start
kg_interaction_start_epoch = 3

## NOTE: if tr_d_kgS1 == 1, this variable is ignored and plays no role 

# initialise counter to prevent wasteful KG cleardowns
# (relevant per KG interaction style 2 only)
n_kg_interactions_since_last_kg_cleardown = 0

# specify threshold of KG activity required before a KG cleardown is warranted
# (relevant per KG interaction style 2 only)
kg_cleardown_activity_threshold = 2000

# initialise counter for KG cleardowns
# (relevant per KG interaction style 2 only) 
n_kg_cleardowns = 0

# initialise counter for total number of calls to the KG for 
# binary classification of a VR type
# (relevant per KG interaction style 2 only)  
n_kg_calls_total = 0

# initialise counter for special-purpose diagnostics
# (relevant per KG interaction style 2 only) 
#n_kg_calls_cumulative_total = 0

# initialise a seconds counter for time spent clearing the KG and
# loading the ontology
# (relevant per KG interaction style 2 only) 
total_KG_cleardown_and_ontology_load_time_sec = 0

# initialise the sequence number we use to make entity names unique when
# we insert RDF triple trios into the KG
# (relevant per KG interaction style 2 only)
start_seq_num = 1000000
entity_seq_num = {'entity_seq_num' : start_seq_num}

if tr_d_kgS1 == 2: 
    print(f'KG interaction style: {kg_interaction_style}')
    print(f'KG interaction start epoch: {kg_interaction_start_epoch}')
    print(f'KG cleardown activity threshold (per style 2 only): {kg_cleardown_activity_threshold}')


#%% main training loop

start_time = time.time()
start_time_checkpoint = start_time

for epoch in range(first_epoch, final_epoch):

    if force_stop:
        print('forced stop to training !!!')
        break    
    
    print(f'\nepoch {epoch:4d} starting ...')
    if redirect_stdout:
        sys.stdout.flush()
    
    # if we're doing KG interaction for nnkgs1, prepare for it
    if tr_d_kgS1 == 2: 
        
        # if we're using KG interaction style 2, then we manage the size of 
        # the KG in GraphDB ourselves, by clearing it and reloading the ontology
        if kg_interaction_style == 2 and epoch >= kg_interaction_start_epoch:
            
            if n_kg_interactions_since_last_kg_cleardown > kg_cleardown_activity_threshold:
                
                print(f'\nKG cleardown at top of epoch: {epoch}')
                n_kg_cleardowns += 1
                
                # empty the KG
                # (nb: nothing is returned; this either works or an exception is thrown)
                clear_kg_start_time = time.time()
                vrdukg.clear_graphdb_kg()  
                clear_kg_end_time = time.time()
                clear_kg_time_sec = clear_kg_end_time - clear_kg_start_time
                clear_kg_time_min = clear_kg_time_sec / 60
                print(f"KG cleardown time 1: {clear_kg_time_sec:.2f} seconds")
                print(f"KG cleardown time 2: {clear_kg_time_min:.2f} minutes")
                # reload the ontology into the empty graph
                # (nb: nothing is returned; this either works or an exception is thrown)
                load_onto_start_time = time.time()
                vrdukg.load_ontology_into_graphdb_kg()
                load_onto_end_time = time.time()
                load_onto_time_sec = load_onto_end_time - load_onto_start_time
                load_onto_time_min = load_onto_time_sec / 60
                print(f"Load ontology time 1: {load_onto_time_sec:.2f} seconds")
                print(f"Load ontology time 2: {load_onto_time_min:.2f} minutes")
                print()
                # accumulate KG cleardown and ontology load timings
                total_KG_cleardown_and_ontology_load_time_sec += (clear_kg_time_sec + load_onto_time_sec)
                # reset counter
                n_kg_interactions_since_last_kg_cleardown = 0
                # reset the sequence number we use to make entity names unique in the KG
                # (nb: we use a dictionary because it's mutable)
                entity_seq_num = {'entity_seq_num' : start_seq_num}
    
    epoch_loss = 0
    epoch_loss_penalty = 0
    mb_group_loss = 0
    mb_group_loss_penalty = 0
    n_kg_calls_epoch = 0

    # iterate over the names of the images in the (sub)training set
    for mb_idx, imname in enumerate(vrd_img_names_training):
        
        #print() 
        #print(f'processing img_idx: {mb_idx}')
    
        # get the PPNN data for the next image to be processed;
        # (nb: the index (position number) of the image must derive from 
        # the full list of training set image names, not the subtrain set 
        # of names; so we get the img_idx from the list vrd_img_names, not
        # from the list vrd_img_names_training)
        img_idx = vrd_img_names.index(imname)
        batch = dataItem[img_idx]
        
        # unpack the batch of data returned by the PPNNDataset object
        # (nb: each mini-batch of data pertains to a single image, but
        #  the amount of data (the size) varies per image; there can be
        #  as few as 2 feature vectors per image, or many dozen 
        img_name = batch['img_name']
        if img_name != imname:
            raise ValueError('problem with image indexing')
        ppnn_img_dict = batch['ppnn_img_dict']
        inputdata = batch['inputdata']    # eg torch.Size([42, 239])
        targets = batch['targets']        # eg torch.Size([42, 72])
        
        # sometimes there may be no ppnn input data for an image; this can
        # arise if an ODNN fails to detect at least 2 objects in an image;
        # when this happens, skip the image and start processing the next
        if inputdata == None:
            continue
     
        # test-mode: check targets
        #n_rows = int(targets.shape[0])
        #n_nopred = int(targets[:,-1].sum())
        #print(f'training: bidx: {bidx+1}, nr of nopreds: {n_nopred}')
        #if n_nopred == n_rows:
        #    print('training: all targets are nopred targets')

        #force_stop = True
        #if force_stop: 
        #    break

        # if a non-CPU device is available, push tensors to it
        if device != torch.device('cpu'):
            inputdata = inputdata.to(device)
            targets = targets.to(device)

        # forward pass through model
        output = model(inputdata)          # eg torch.Size([42, 72])
        
        # CAUTION: if we use torch.compile to compile the model under PyTorch
        # 2.0.1, the model forward pass throws the following exception on 
        # Mac machines:
        # "AssertionError: Torch not compiled with CUDA enabled".
        # So torch.compile seems to work only on CUDA devices, at least
        # under PyTorch 2.0.1

        # NOTE: If we wish to interact with an OWL-based KG within the 
        # training loop, this is the place to do it --- after we have the 
        # model outputs and before we compute training loss.
        #
        # In NN+KG_S0, if we were using KG reasoning to do VR annotation
        # augmentation in real-time rather than upfront in batch mode with
        # KG materialisation, we would interact with the KG here.
        # We would need to build our target tensor here, based on the 
        # KG-augmented set of VRs for the current image, rather than 
        # in the PPNNDataset object at the top of the loop.
        #
        # In NN+KG_S1, we want KG reasoning feedback as to the semantic 
        # validity (or lack of it) of the emerging VR predictions implied 
        # by the logits in the PPNN output vectors. We would do that KG 
        # interaction here, once we have the PPNN output logits and before 
        # we calculate loss.
        #
        # In NN+KG_S2, we want KG reasoning feedback as to the plausibility
        # or implausibility of emerging VR predictions implied by the logits
        # in the PPNN model output vectors. We would do that KG
        # interaction here.
        

        # this condition has been used for special trials
        # if tr_d_kgS1 in [1, 2]:
            
        # TODO: vrdukg.interact_with_kg_for_nnkgs1() has 'print()' 
        # statements that need to be commented-out or removed once we
        # have refined nnkgs1 sufficiently
        
        # TODO: remove parameter tr_d_kgS1 from interact_with_kg_for_nnkgs1()
        # once we have refined nnkgs1 sufficiently
        
        # if we're doing KG interaction for nnkgs1, this is where it happens
        if tr_d_kgS1 == 2:
            # we delay KG interaction until after the first 5 epochs of
            # training to give the PPNN a chance to learn predictions of its
            # own, not fake predictions induced by the random weight 
            # initialisation of the PPNN model
            if epoch >= kg_interaction_start_epoch:
                
                if imname in kg_interaction_results_per_image:
                    kg_interaction_results = kg_interaction_results_per_image[imname]
                    # convert JSON serialised results (list of lists of reals) to tensor
                    kg_interaction_results = torch.tensor(kg_interaction_results)
                else:
                    kg_interaction_results = torch.zeros(output.size())                
                
                
                # TODO: remove all this diagnostics when we're done fixing the bug
                #if mb_idx in [13, 25, 26, 33, 37, 60, 64, 67]:
                if mb_idx in []:
                    print() 
                    n_vr_type_classifications_before = torch.count_nonzero(kg_vr_type_classifications)
                    print(f'n_vr_type_classifications_before: {n_vr_type_classifications_before}')
                    mask = kg_vr_type_classifications == 1.0
                    vr_types_classified_valid_before = mask.nonzero()
                    print(f'n_vr_types_classified_valid_before: {vr_types_classified_valid_before.shape[0]}')
                    print(f'vr_types_classified_valid_before: {vr_types_classified_valid_before}')
                    mask = kg_vr_type_classifications == 2.0
                    vr_types_classified_invalid_before = mask.nonzero()
                    print(f'n_vr_types_classified_invalid_before: {vr_types_classified_invalid_before.shape[0]}')
                    print(f'vr_types_classified_invalid_before: {vr_types_classified_invalid_before}')                
                
                    print()
                    print('kg_interaction_results for image, BEFORE calling kg.interact()')
                    print(kg_interaction_results)
                
                
                res = vrdukg.interact_with_kg_for_nnkgs1(output, 
                                                         ppnn_img_dict,
                                                         ontoClassNames,
                                                         ontoPropNames,
                                                         entity_seq_num,
                                                         kg_vr_type_classifications,
                                                         kg_interaction_results,
                                                         kg_interaction_style,
                                                         tr_d_kgS1)

                #if mb_idx in [13, 25, 26, 33, 37, 60, 64, 67]:
                if mb_idx in []:
                    # TODO: remove all this diagnostics when we're done fixing the bug
                    print()
                    n_vr_type_classifications_after = torch.count_nonzero(kg_vr_type_classifications)
                    print(f'n_vr_type_classifications_after: {n_vr_type_classifications_after}')
                    mask = kg_vr_type_classifications == 1.0
                    vr_types_classified_valid_after = mask.nonzero()
                    print(f'n_vr_types_classified_valid_after: {vr_types_classified_valid_after.shape[0]}')
                    print(f'vr_types_classified_valid_after: {vr_types_classified_valid_after}')
                    mask = kg_vr_type_classifications == 2.0
                    vr_types_classified_invalid_after = mask.nonzero()
                    print(f'n_vr_types_classified_invalid_after: {vr_types_classified_invalid_after.shape[0]}')
                    print(f'vr_types_classified_invalid_after: {vr_types_classified_invalid_after}')                
                    print()
                    
                    if imname in kg_interaction_results_per_image_test:
                        img_epoch_dict = kg_interaction_results_per_image_test[imname]
                    else:
                        img_epoch_dict = {}
                    
                    # store the kg_interaction_results 2D matrix for the current image
                    # and the current epoch for later analysis
                    epoch_key = 'epoch_' + str(epoch).zfill(3)
                    kg_interaction_results_epoch = kg_interaction_results.numpy().tolist()
                    img_epoch_dict[epoch_key] = kg_interaction_results_epoch
                    kg_interaction_results_per_image_test[imname] = img_epoch_dict
                    print('saving kg_interaction_results for image for epoch after kg.interact()')
                    print()

                
                # note: parameter kg_vr_type_classifications is a tensor and
                # hence mutable; it is updated in-place within the function;
                # this is deliberate and essential to eliminating 
                # redundant KG interactions and speeding training times
                
                # unpack the results from the function call
                cells_for_loss_penalty, kg_interaction_results, n_kg_calls = res 

                # package the loss penalty matrix for loss penalty computation
                kg_results = {'kg_results_type': 'nnkgs1',
                              'kg_results_data': cells_for_loss_penalty}

                # save the accumulating KG interaction history for the current image
                kg_interaction_results = kg_interaction_results.numpy().tolist()
                kg_interaction_results_per_image[imname] = kg_interaction_results
                
                # accumulate the count of kg interactions for the current epoch
                n_kg_calls_epoch += n_kg_calls
                
                # count the number of KG VR type classifications performed so far 
                #n_vr_type_classifications_so_far = torch.count_nonzero(kg_vr_type_classifications)
                #n_kg_calls_cumulative_total += n_kg_calls
                #print(f'n_kg_calls_cumulative_total: {n_kg_calls_cumulative_total}')
                #print(f'n_vr_type_classifications_so_far: {n_vr_type_classifications_so_far}')
                #if n_kg_calls_cumulative_total != n_vr_type_classifications_so_far:
                #    print('PROBLEM')
                #    print('KG call count not equal VR type classification count') 
                #    print(f'mb_idx {mb_idx}, imname {imname}')
                #    if redirect_stdout:
                #        sys.stdout.flush()
                #    raise Exception('PROBLEM: discrepancy 1 first appearance')
                              
                # TODO: remove this flush once we've refined nnkgs1 sufficiently
                #if redirect_stdout:
                #    sys.stdout.flush()                

            else:
                kg_results = None
        else:
            kg_results = None
        
        
        # compute training loss on the PPNN output
        mb_loss_results = vls.compute_loss(tr_d_model2, 
                                           nopred_target_policy, 
                                           kg_results,
                                           output,
                                           targets,
                                           loss_reduction,
                                           device)
        
        # clarify the loss results returned by compute_loss()
        mb_loss_core_matrix, mb_loss_penalty_matrix = mb_loss_results
        
        # example sizes of these matrices: torch.Size([42, 72])

        #force_stop = True
        #if force_stop: 
        #    break
        
        # reduce (summarise) the loss contributions conveyed in the
        # cells of the core loss matrix
        mb_loss_core = torch.sum(mb_loss_core_matrix)
        
        if mb_loss_penalty_matrix == None:
            mb_loss_penalty_raw = 0.0
        else:
            mb_loss_penalty_raw = torch.sum(mb_loss_penalty_matrix)
        mb_loss_penalty = nnkgs1_lambda * mb_loss_penalty_raw
            
        # combine the two components of loss
        mb_loss = mb_loss_core + mb_loss_penalty
        
        # backpropagate and update model parameters
        optimiser.zero_grad()
        mb_loss.backward()
        optimiser.step()
        
        # accumulate combined loss components
        mb_group_loss += mb_loss
        epoch_loss += mb_loss
        
        # accumulate loss penalty
        mb_group_loss_penalty += mb_loss_penalty
        epoch_loss_penalty += mb_loss_penalty

        # write loss to log file so we can monitor its behaviour 
        if (mb_idx+1) % n_mb_report_loss == 0:
            avg_loss_per_mb_group = mb_group_loss / n_mb_report_loss
            print(f"mb {mb_idx+1:4d}; avg loss per mb group: {avg_loss_per_mb_group:.5f}")
            if tr_d_kgS1 == 2:
                avg_loss_penalty_per_mb_group = mb_group_loss_penalty / n_mb_report_loss
                print(f"mb {mb_idx+1:4d}; avg loss penalty per mb group: {avg_loss_penalty_per_mb_group:.5f}")
            if redirect_stdout:
                sys.stdout.flush()
            mb_group_loss = 0
            mb_group_loss_penalty = 0

        # capture PPNN logit output for designated (sub)training set 
        # images so we can monitor how well the model is learning to predict
        # logits that correspond to ground-truth target VRs as training
        # advances epoch by epoch
        if epoch % n_epochs_monitor == 0:
            if mb_idx in image_idxs_to_monitor:
                vrdu9.capture_logits_for_image(trc_id, mb_idx, epoch,
                                               img_name, ppnn_img_dict,
                                               output, targets, device,
                                               model_checkpoint_dir)

        # if we're in test mode (training for a fixed number of
        # image entries per epoch), then check if it's time to stop
        if n_images_per_epoch_train > 0:
            if (mb_idx+1) >= n_images_per_epoch_train:
                break

    #if force_stop:
    #    continue
    
    #
    # ------------------------------------
    # end-of-epoch processing starts here
    # ------------------------------------
    #

    # compute average training loss per mini-batch (image) over epoch
    if n_images_per_epoch_train > 0:
        avg_training_loss_per_mb = epoch_loss / n_images_per_epoch_train
    else:
        avg_training_loss_per_mb = epoch_loss / len(vrd_img_names_training)
    
    # compute average training loss penalty per mini-batch (image) over epoch
    if tr_d_kgS1 == 2:
        if n_images_per_epoch_train > 0:
            avg_training_loss_penalty_per_mb = epoch_loss_penalty / n_images_per_epoch_train
        else:
            avg_training_loss_penalty_per_mb = epoch_loss_penalty / len(vrd_img_names_training)    
    
    
    # format the epoch number for recording in log file
    #(nb: this permits reliable sorting of key log file lines on all platforms)
    epoch_f = str(epoch).zfill(3)

    # report training loss to log file
    print(f"epoch {epoch_f}: all done; statistics ...")
    print(f"epoch {epoch_f}: avg training loss per mb: {avg_training_loss_per_mb:.5f}")

    if tr_d_kgS1 == 2:
        # count the KG interactions since the last KG cleardown
        n_kg_interactions_since_last_kg_cleardown += n_kg_calls_epoch
        print(f"epoch {epoch_f}: avg training loss penalty per mb: {avg_training_loss_penalty_per_mb:.5f}")
        print(f'epoch {epoch_f}: total number of KG interactions: {n_kg_calls_epoch}')
        # count the total number of KG calls for VR type classification
        n_kg_calls_total += n_kg_calls_epoch 

    #force_stop = True
    #if force_stop:
    #    break

    #
    # if instructed to do so, process the validation set and measure
    # both validation loss and validation predictive performance
    #
    
    avg_validation_loss_per_mb = 0.0
    validation_global_recallN = 0.0
    
    if epoch >= start_epoch_for_validation:

        # process the validation set with the current model; measure both
        # validation loss and validation predictive performance using our
        # recall@N-based metrics
        vlr, vpr = vrdu13.perform_validation(model, 
                                             dataItem,
                                             vrd_img_names,
                                             vrd_img_names_validation,
                                             device,
                                             n_images_per_epoch_val,
                                             vrd_img_anno, 
                                             prediction_region_config,
                                             calculate_val_loss, 
                                             loss_reduction,
                                             tr_d_model2,
                                             nopred_target_policy,
                                             compile_model)
        
        val_loss_results = vlr
        avg_validation_loss_per_mb = val_loss_results['avg_loss_per_img_val']
        
        val_perf_results = vpr
        validation_global_recallN = val_perf_results['global_recallN']
        
        #val_perf_results['topN'] 
        #val_perf_results['mean_gt_vrs_per_img'] 
        #val_perf_results['mean_pred_vrs_per_img'] 
        #val_perf_results['global_recallN'] 
        #val_perf_results['mean_per_image_recallN'] 
        #val_perf_results['mean_avg_recallK_topN'] 
        
        # report the validation loss and validation performance to the log
        # file so we can post-process this info into a .csv file for
        # viewing and analysis, along with training loss
        print(f"epoch {epoch_f}: avg validation loss per mb: {avg_validation_loss_per_mb:.5f}")
        print(f"epoch {epoch_f}: validation global_recall@N: {validation_global_recallN:.1f}")


    # periodically save of model checkpoint file
    if epoch % n_epochs_checkpoint == 0:
        end_time = time.time()
        train_time = (end_time - start_time_checkpoint) / 60
        print(f"epoch {epoch_f}: checkpoint training time: {train_time:.2f} minutes")
        if epoch >= start_epoch_for_saving_model_checkpoints:
            # save checkpoint files only after a certain number of epochs,
            # when loss convergence and maximal performance begin to be
            # realistic possibilities
            start_time_save_ckpt = time.time()
            save_model_checkpoint(model_checkpoint_filename_base, epoch, 
                                  model, optimiser, avg_training_loss_per_mb)
            end_time_save_ckpt = time.time()
            ckpt_save_time = (end_time_save_ckpt - start_time_save_ckpt) / 60
            print(f"epoch {epoch_f}: checkpoint saving time: {ckpt_save_time:.2f} minutes\n")

        start_time_checkpoint = time.time()


    # if requested, monitor the stream of validation loss/performance
    # scores as training proceeds and stop the training when and if the
    # stopping criteria configured for the EarlyStopper are satisfied
    if early_stopping_active:
        if early_stopping_score_type == 'loss':
            score = avg_validation_loss_per_mb
        elif early_stopping_score_type == 'performance':
            score = validation_global_recallN
        else:
            raise ValueError('early stopping score type not recognised')
        early_stopper.monitor_score(score)
        if early_stopper.stop_training:
            print(f'early stopping triggered at epoch: {epoch}')
            force_stop = True

    
    # flush print statements to the log file so we can monitor the action
    # without waiting to see what's been happening
    if redirect_stdout:
        sys.stdout.flush()



# print total training time (in minutes and hours)
end_time = time.time()
train_time = (end_time - start_time) / 60
print(f"\nTotal training time 1: {train_time:.2f} minutes")
train_time = train_time / 60
print(f"Total training time 2: {train_time:.2f} hours")


# save a checkpoint file if training epochs have been processed since
# the last time a checkpoint file was saved
if epoch % n_epochs_checkpoint != 0:
    if not force_stop:
        save_model_checkpoint(epoch, model, optimiser, avg_training_loss_per_mb)


# if we're doing KG interaction for nnkgs1, print summary info and 
# save the two tools we use to manage the KG interaction to JSON files 
if tr_d_kgS1 == 2:
    print()
    print('------ nnkgs1 summary info ------')
    print(f'total number of KG interactions for VR type classification: {n_kg_calls_total}')
    
    if kg_interaction_style == 2:
        print('------ nnkgs1 - KG interaction style 2 summary info ------')
        print(f'number of KG cleardowns: {n_kg_cleardowns}')
        print(f'total KG cleardown and ontology load time 1: {total_KG_cleardown_and_ontology_load_time_sec:.2f} seconds')
        total_KG_cleardown_and_ontology_load_time_min = total_KG_cleardown_and_ontology_load_time_sec / 60
        print(f'total KG cleardown and ontology load time 2: {total_KG_cleardown_and_ontology_load_time_min:.2f} minutes')  
    
    save_kg_interaction_results(kg_interaction_filepath, kg_interaction_results_per_image)
    print(f'KG interaction results per image saved: {kg_interaction_filename}')
    
    if n_kg_calls_total > 0:
        save_kg_vr_type_classifications(kg_vr_type_classification_filepath, kg_vr_type_classifications)
        print(f'KG VR type classifications saved: {kg_vr_type_classification_filepath}')
    else:
        print('KG VR type classifications tensor not saved because no KG interactions occurred')
    
    # TODO: remove this when we're done QA testing
    save_kg_interaction_results(kg_interaction_filepath_test, kg_interaction_results_per_image_test)
    print() 
    print(f'KG interaction results per image saved: {kg_interaction_filename}')
    

if redirect_stdout:
    # flush stdout buffer
    sys.stdout.flush()
    # close redirected output file
    sys.stdout.close()
    # restore sys.stdout to original file handler
    sys.stdout = stdout_file_saved

print('Processing completed')






















