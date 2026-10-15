#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script converts consolidated training log statistics text files 
into .csv files.

Example input files: training loss scores, validation loss scores, 
validation performance scores
* vrd_ppnn_trc0460_ckpt_training_loss_per_epoch.txt
* vrd_ppnn_trc0460_ckpt_training_loss_per_epoch_val_loss.txt
* vrd_ppnn_trc0460_ckpt_training_loss_per_epoch_val_perf.txt
  
Example output file:
* vrd_ppnn_trc0460_ckpt_training_loss_summary.csv
'''


#%%

import pandas as pd
import os
import sys
from datetime import date

import vrd_utils14 as vrdu14


#%% gather arguments supplied to this script

# get or set the experiment space training region cell id
if len(sys.argv) > 1:
    trc_id = sys.argv[1]
else:
    trc_id = 'trc0460'
if not trc_id.startswith('trc'):
    raise ValueError(f'training region cell id {trc_id} not recognised')

# get or set the platform on which the training script is running
if len(sys.argv) > 2:
    platform = sys.argv[2]
else:
    platform = 'macstudio'
if not platform in vrdu14.supported_platforms:
    raise ValueError(f'platform {platform} not recognised')

# # get or set the work directory (the folder in which to store output files)
if len(sys.argv) > 3:
    workdir = sys.argv[3]
else:
    workdir = 'ppnn'


#%% lookup config of the experiment space training region cell

# specify the experiment family
experiment_family = 'nnkgs0'      # NN+KG_S0

# no lookups required


#%% build the name of the work directory

if platform == 'hyperion':
    workdirpath = os.path.join('~', 'sharedscratch', 'research', workdir)
else:
    workdirpath = os.path.join('~', 'research', workdir)

scriptName = os.path.basename(__file__)
print(f'script name: {scriptName}')

workdirpath = os.path.expanduser(workdirpath)
print(f"work dir   : {workdirpath}")


#%% declare (record) experiment meta data

print()
print(f'# experiment family: {experiment_family}')
print(f'# training region cell id: {trc_id}')
print()
print(f'Date: {date.today()}')


#%% set the names of the input files we wish to process

# note: these input files are created using shell command 'grep' to
# extract per epoch training loss, validation loss and validation 
# performance scores from the training log file of a PPNN model training run

train_loss_filename = 'vrd_ppnn_ckpt_training_loss_per_epoch.txt'

val_loss_filename = 'vrd_ppnn_ckpt_training_loss_per_epoch_val_loss.txt'

val_perf_filename = 'vrd_ppnn_ckpt_training_loss_per_epoch_val_perf.txt'

train_loss_filepath = os.path.join(workdirpath, train_loss_filename)
val_loss_filepath = os.path.join(workdirpath, val_loss_filename)
val_perf_filepath = os.path.join(workdirpath, val_perf_filename)

print('processing these input files:')
print(f'train loss scores: {train_loss_filename}')
print(f'val   loss scores: {val_loss_filename}')
print(f'val   perf scores: {val_perf_filename}')


#%% build the name of the output file we wish to create

# specify the name of the output .csv file
output_filename = 'vrd_ppnn_' + trc_id + '_ckpt_training_loss_summary.csv'

output_filepath = os.path.join(workdirpath, output_filename)


#%% function to process the consolidated training loss text file

# example text file line to be processed:
# "./vrd_ppnn_cr001_ckpt_training_log_run_01.txt:epoch   82: avg training loss per mb: 16.80421"
#
# nb: 4 parts delimited by 3 colons ':'


def process_training_loss(train_loss_filepath):

    with open(train_loss_filepath) as fp:
        training_loss_per_epoch_lines = fp.readlines()    

    training_epochs = []
    training_losses = []
    training_loss_direction_changes = []

    previous_loss = float('inf')

    for line_num, line in enumerate(training_loss_per_epoch_lines):
    
        # split the line into parts delimited by colons, ':'
        line = line.strip()
        line_parts = line.split(':')
        #print(f'line_parts: {line_parts}')
        
        # check that we can expect to interpret the text line correctly
        if not len(line_parts) == 4:
            raise ValueError(f'wrong number of parts on line {line_num+1}')
        if not (line_parts[0].startswith('./vrd_ppnn_') or 
                line_parts[0].startswith('vrd_ppnn_')):
            raise ValueError(f'unrecognised prefix on line {line_num+1}')
        if not line_parts[2] == ' avg training loss per mb':
            raise ValueError(f'line {line_num+1} not recognised')

        # get the training epoch number
        try:
            tokens = line_parts[1].split(' ', maxsplit=1)
            tokens[1] = tokens[1].strip()
            epoch = int(tokens[1])
        except:
            raise ValueError(f'problem with epoch, line {line_num+1}')
        
        # get the training loss for the current epoch
        try:
            loss_str = line_parts[3].strip()
            loss = float(loss_str)
        except:
            raise ValueError(f'problem with loss, line {line_num+1}')
        
        if not loss > 0:
            raise ValueError(f'invalid loss detected, line {line_num+1}')
    
        # check the direction of change
        if loss > previous_loss:
            direction = 'up'
        else:
            direction = ' '
        
        # save the data extracted from the text file for the current epoch
        training_epochs.append(epoch)
        training_losses.append(loss_str)
        training_loss_direction_changes.append(direction)
    
        previous_loss = loss


    # package the results
    results = { 'train_epoch': training_epochs, 
                'train_loss': training_losses, 
                'train_loss_dir_chg': training_loss_direction_changes }
    
    return results


#%% function to process the consolidated validation loss text file

# example text file line to be processed:
# "./vrd_ppnn_cr001_ckpt_training_log_run_01.txt:epoch   82: avg validation loss per mb: 18.12345"
#
# nb: 4 parts delimited by 3 colons ':'


def process_validation_loss(validation_loss_filepath):

    with open(validation_loss_filepath) as fp:
        validation_loss_per_epoch_lines = fp.readlines()   
    
    training_epochs = []
    validation_losses = []
    validation_loss_direction_changes = []
    
    previous_loss = float('inf')

    for line_num, line in enumerate(validation_loss_per_epoch_lines):
    
        # split the line into parts delimited by colons, ':'
        line = line.strip()
        line_parts = line.split(':')
        
        # check that we can expect to interpret the text line correctly
        if not len(line_parts) == 4:
            raise ValueError(f'wrong number of parts on line {line_num+1}')
        if not (line_parts[0].startswith('./vrd_ppnn_') or 
                line_parts[0].startswith('vrd_ppnn_')):
            raise ValueError(f'unrecognised prefix on line {line_num+1}')
        if not line_parts[2] == ' avg validation loss per mb':
            raise ValueError(f'line {line_num+1} not recognised')

        # get the training epoch number
        try:
            tokens = line_parts[1].split(' ', maxsplit=1)
            tokens[1] = tokens[1].strip()
            epoch = int(tokens[1])
        except:
            raise ValueError(f'problem with epoch, line {line_num+1}')
        
        # get the training loss for the current epoch
        try:
            loss_str = line_parts[3].strip()
            loss = float(loss_str)
        except:
            raise ValueError(f'problem with loss, line {line_num+1}')
        
        if not loss > 0:
            raise ValueError(f'invalid loss detected, line {line_num+1}')
    
        # check the direction of change
        if loss > previous_loss:
            direction = 'up'
        else:
            direction = ' '
        
        # If the first epoch number is greater than 1, then we have a void
        # in the range of epoch numbers for which scores have been 
        # calculated. We need to fill this void with dummy data so that
        # the set of epoch numbers matches exactly with those of the
        # training loss scores.
        if line_num == 0 and epoch > 1:
            n_epochs = epoch - 1
            for idx in range(n_epochs):
                dummy_epoch = idx + 1
                training_epochs.append(dummy_epoch)
                validation_losses.append("")
                validation_loss_direction_changes.append("")
        
        # save the data extracted from the text file for the current epoch
        training_epochs.append(epoch)
        validation_losses.append(loss_str)
        validation_loss_direction_changes.append(direction)
    
        previous_loss = loss


    # package the results
    results = { 'train_epoch': training_epochs, 
                'val_loss': validation_losses, 
                'val_loss_dir_chg': validation_loss_direction_changes }
    
    return results


#%% function to process the consolidated validation performance text file

# example text file line to be processed:
# "./vrd_ppnn_cr001_ckpt_training_log_run_01.txt:epoch   82: avg validation loss per mb: 18.12345"
#
# nb: 4 parts delimited by 3 colons ':'


def process_validation_performance(validation_perf_filepath):

    with open(validation_perf_filepath) as fp:
        validation_perf_per_epoch_lines = fp.readlines()   
    
    training_epochs = []
    validation_performances = []
    validation_performance_direction_changes = []
    
    previous_perf = float('-inf')

    for line_num, line in enumerate(validation_perf_per_epoch_lines):
    
        # split the line into parts delimited by colons, ':'
        line = line.strip()
        line_parts = line.split(':')
        
        # check that we can expect to interpret the text line correctly
        if not len(line_parts) == 4:
            raise ValueError(f'wrong number of parts on line {line_num+1}')
        if not (line_parts[0].startswith('./vrd_ppnn_') or 
                line_parts[0].startswith('vrd_ppnn_')):
            raise ValueError(f'unrecognised prefix on line {line_num+1}')
        if not line_parts[2] == ' validation global_recall@N':
            raise ValueError(f'line {line_num+1} not recognised')

        # get the training epoch number
        try:
            tokens = line_parts[1].split(' ', maxsplit=1)
            tokens[1] = tokens[1].strip()
            epoch = int(tokens[1])
        except:
            raise ValueError(f'problem with epoch, line {line_num+1}')
        
        # get the validation performance for the current epoch
        try:
            perf_str = line_parts[3].strip()
            perf = float(perf_str)
        except:
            raise ValueError(f'problem with performance, line {line_num+1}')
        
        if not perf > 0:
            raise ValueError(f'invalid performance detected, line {line_num+1}')
    
        # check the direction of change
        if perf < previous_perf:
            direction = 'down'
        else:
            direction = ' '

        # If the first epoch number is greater than 1, then we have a void
        # in the range of epoch numbers for which scores have been 
        # calculated. We need to fill this void with dummy data so that
        # the set of epoch numbers matches exactly with those of the
        # training loss scores.
        if line_num == 0 and epoch > 1:
            n_epochs = epoch - 1
            for idx in range(n_epochs):
                dummy_epoch = idx + 1
                training_epochs.append(dummy_epoch)
                validation_performances.append("")
                validation_performance_direction_changes.append("")
        
        # save the data extracted from the text file for the current epoch
        training_epochs.append(epoch)
        validation_performances.append(perf_str)
        validation_performance_direction_changes.append(direction)
    
        previous_perf = perf


    # package the results
    results = { 'train_epoch': training_epochs, 
                'val_perf': validation_performances, 
                'val_perf_dir_chg': validation_performance_direction_changes }
    
    return results


#%% function to merge multiple dictionaries of results

def merge_results(training_loss_results, validation_loss_results,
                  validation_perf_results):
    
    # ensure that the dictionaries of results all refer to the same 
    # sequence of training epoch numbers
    train_epochs_1 = training_loss_results['train_epoch']
    train_epochs_2 = validation_loss_results['train_epoch']
    train_epochs_3 = validation_perf_results['train_epoch']
    
    epoch_problem = False
    if not train_epochs_1 == train_epochs_2:
        epoch_problem = True
    if not train_epochs_1 == train_epochs_3:        
        epoch_problem = True
    if epoch_problem:
        raise ValueError('epoch numbers not aligned across results files')
    
    # merge the multiple dictionaries into one
    results = {}
    results['epoch'] = train_epochs_1
    results['train_loss'] = training_loss_results['train_loss']
    results['tl_dir'] = training_loss_results['train_loss_dir_chg'] 
    results['val_loss'] = validation_loss_results['val_loss']
    results['vl_dir'] = validation_loss_results['val_loss_dir_chg']       
    results['val_perf'] = validation_perf_results['val_perf']
    results['vp_dir'] = validation_perf_results['val_perf_dir_chg'] 
    
    return results


#%% main processing

# process the consolidated training loss text file
training_loss_results = process_training_loss(train_loss_filepath)
print('training loss scores processed successfully')
    
# process the consolidated validation loss text file
validation_loss_results = process_validation_loss(val_loss_filepath)
print('validation loss scores processed successfully')    

# process the consolidated training loss text file
validation_perf_results = process_validation_performance(val_perf_filepath)
print('validation performance scores processed successfully') 
   
# merge results into a single dictionary
combined_results = merge_results(training_loss_results,
                                 validation_loss_results,
                                 validation_perf_results)
print('scores merged successfully')
    

#%% finish up

# save the combined results to a .csv file
df = pd.DataFrame(combined_results)
df.to_csv(output_filepath, index=False)
print(f'output file: {output_filename}')
print('processing complete')


