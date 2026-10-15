#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This script logically expands scene graphs of NeSy4VRD images, according
to the NeSy4VRD OWL ontology, VRD-World, using GraphDB.

This script does the following:
* assumes an empty default graph of a GraphDB repository is ready and waiting
* it loads OWL ontology VRD-World into the KG
* it converts NeSy4VRD visual relationship annotations (image scene graphs)
  into RDF triples and loads them into the KG
  - inference happens automatically as triples are loaded into a GraphDB KG,
    so there is no need to invoke OWL reasoning to materialise the scene graphs;
    they are materialised as they are loaded
* it uses a SPARQL query to extract the logically expanded scene graph for
  each image from the KG
* it converts the extracted scene graph for each image back into its native 
  NeSy4VRD format
* it saves the reconstituted (now logically expanded) image scene graphs 
  to a disk file in JSON format

This script was designed to be executable interactively, cell by cell, in and IDE,
and to be runable in batch mode, writing to an external log file.

DEPENDENCIES:

Python packages:
    SPARQLWrapper

GraphDB
* a running GraphDB knowledge graph accessible by URL (ip address & port number)
'''

#%% imports

import vrd_utils15 as vrdu15

import os
import json
import time 
import sys 

import nesy4vrd_utils4 as vrdu4


#%% get the NeSy4VRD object class names and predicate names

# set the path to the directory where the NeSy4VRD visual relationship
# annotations files reside
anno_dir = os.path.join('..', 'data', 'annotations')

# get the master list of NeSy4VRD object class names
vrd_objects_path = os.path.join(anno_dir, 'nesy4vrd_objects.json')
vrd_objects = vrdu4.load_NeSy4VRD_object_class_names(vrd_objects_path)

# get the master list of NeSy4VRD predicate names
vrd_predicates_path = os.path.join(anno_dir, 'nesy4vrd_predicates.json')
vrd_predicates = vrdu4.load_NeSy4VRD_predicate_names(vrd_predicates_path)


#%% set work directory

work_dir = os.path.join('~', 'research', 'loadAndAugment')
work_dir = os.path.expanduser(work_dir)


#%% load the image scene graph(s) to be KG-augmented

scene_graph_filename = 'nesy4vrd_formatted_image_scene_graphs.json'

scene_graph_filepath = os.path.join(work_dir, scene_graph_filename)

vrd_anno = vrdu4.load_NeSy4VRD_image_annotations(scene_graph_filepath)

# get a list of the VRD image names from the NeSy4VRD annotations dictionary
vrd_img_names = list(vrd_anno.keys())

print(f'scene graph file loaded: {scene_graph_filepath}')


#%% specify directory for storing curl tool output files

curl_dir = os.path.join('~', 'research', 'curl')
curl_dir = os.path.expanduser(curl_dir)


#%% convert NeSy4VRD object class and predicate names to VRD-World ontology names

ontoClassNames = vrdu4.convert_NeSy4VRD_classNames_to_ontology_classNames(vrd_objects)

ontoPropNames = vrdu4.convert_NeSy4VRD_predicateNames_to_ontology_propertyNames(vrd_predicates)


#%% initialise sequence numbers for individual objects to be loaded into KG

number_of_vrd_object_classes = len(vrd_objects)
vrdu4.initialise_object_sequence_numbers(number_of_vrd_object_classes)


#%%

redirect_stdout = True

# build name of log file
if redirect_stdout:
    log_filename = 'expand_scene_graphs_in_graphdb_log.txt'
    stdout_file_saved = sys.stdout
    file_path_name = os.path.join(work_dir, log_filename)
    print(f'redirecting stdout to log file: {file_path_name}')
    sys.stdout = open(file_path_name, 'w')

print()
print('*** expand_scene_graphs_in_graphdb.py log file ***')
print()
print(f'work directory: {work_dir}')


#%%

# ip address and port number of GraphDB instance
#graphdb_address_port = 'http://192.168.0.151:7200'
graphdb_address_port = 'http://localhost:7200'

repository_name = 'test'

base_graphdb_url = graphdb_address_port + '/repositories/' + repository_name

print()
print('GraphDB base url:')
print(base_graphdb_url)


#%% verify connectivity and check that the default graph is empty

graph_name = 'default'

print()
print('Checking size of GraphDB default graph ...')

outcome, n_triples = vrdu15.get_graph_size(base_graphdb_url, graph_name)

print(f'GraphDB interaction outcome: {outcome}')
print(f"Number of triples in graph '{graph_name}': {n_triples}")

# OWL2-RL ruleset has 1020 triples
# OWL2-RL (Optimized) ruleset has 786 triples
if n_triples != 1020 and n_triples != 786:
    print()
    print('STOPPING: GraphDB default graph not ready')
    raise ValueError('GraphDB default graph not ready')

## NOTE:
## 1) the ruleset configured for a GraphDB repository has some number of
##    RDF/RDFS/OWL axioms and statements associated with it; these are
##    stored away invisibly, but they contribute to the count of 'inferred'
##    statements in the repository; for the OWL2-RL ruleset there are 1020
##    associated inferred statements; so a GraphDB repository configured
##    to use the OWL2-RL ruleset is empty when it contains 0 explicit triples
##    and 1020 inferred triples;


#%% specify a version of the VRD-World OWL ontology with which to work

target_ontology = 'vrd_world_v1_2c.owl'

ontology_filepath = os.path.join(work_dir, target_ontology)

print()
print(f'Ontology filepath: {ontology_filepath}')


#%% load the ontology into the default graph of the GraphDB repository

print()
print('-------------------------------------------------------')
print() 
print('Loading ontology into GraphDB default graph ...')
print() 

if redirect_stdout:
    sys.stdout.flush()

start_time = time.time()

vrdu15.post_rdf_file_to_default_graph_via_http(base_graphdb_url, 
                                               ontology_filepath,
                                               curl_dir,
                                               verbose=False)

end_time = time.time()
ontology_load_and_infer_time = end_time - start_time

outcome, resp_code, msg, msg_info = vrdu15.get_http_interaction_outcome(curl_dir, 
                                                                        verbose=False)

show_details = False
if show_details:
    print()
    print(f'interaction outcome: {outcome}')
    print(f'http_response code: {resp_code}')
    print(f'error msg: {msg}')
    print('error info:')
    print(msg_info)


#%% check the KG size after loading the ontology

graph_name = 'default'

print()
print('Checking size of GraphDB default graph ...')

outcome, n_triples = vrdu15.get_graph_size(base_graphdb_url, graph_name)

print(f'GraphDB interaction outcome: {outcome}')
print(f"Number of triples in graph '{graph_name}': {n_triples}")


#%%

print()
print(f'We will be augmenting scene graphs for {len(vrd_img_names)} images')


#%%

# control whether or not to print detailed info to the console during
# processing; if you are processing the VRs for ALL images (or a large
# number, you will want to turn this off; otherwise, if you are
# processing just a small number of images, you find the information
# helpful)
verbose_mode = True

# show individual RDF triples inserted in KG
verbose2_mode = True

print() 
print(f'verbose mode is: {verbose_mode}')


#%% convert VR annotations into RDF triples and load into GraphDB

print()
print('------------------------------------------------------------------')
print()
print('Converting image scene graphs to RDF triples and inserting into KG ...')
print()

if redirect_stdout:
    sys.stdout.flush()

image_cnt = 0
total_object_cnt = 0
total_triple_cnt = 0

# maintain a list of the unique ids created to represent each VRD image
# when it is loaded into the KG; there will be positional correspondence
# between image_id entries in this list and image filename entries in
# the list (variable) named 'vrd_img_names'
kg_vrd_img_ids = []

# maintain a list of the counts of the number of human-annotated VRs
# per VRD dataset image; there will be positional correspondence
# between vr count entries in this list and image filename entries in
# the list (variable) named 'vrd_img_names'
vrd_img_vr_cnts = []

# maintain a list of dictionaries, one dictionary per image; the dictionary 
# for a particular image contains one entry for each object annotated 
# as appearing in the image; the 'key' for an entry is the object's bbox
# (in the form of a tuple); the 'value' is a list of 3 elements:
# [kg_object_id, kg_object_uriref, object_idx], where object_idx is the
# VRD object class label (integer index) identifying the object class of the 
# corresponding bbox
objects_per_image = []


start_time = time.time()


# iterate over the image names whose scene graphs are to be augmented
for img_idx, imname in enumerate(vrd_img_names): 
    
    image_cnt += 1
    
    # show progress
    if image_cnt % 5 == 0:
        time_now = time.time()
        time_so_far = time_now - start_time
        time_so_far_in_minutes = time_so_far / 60
        print(f'processing image idx {img_idx}; time so far is {time_so_far_in_minutes:.2f} minutes')
        if redirect_stdout:
            sys.stdout.flush()
   
    imanno = vrd_anno[imname]
    image_objects = {}
    image_triple_cnt = 0
    
    if verbose_mode:
        print(f'\nprocessing image: {imname}')
        
    if verbose2_mode:
        print()
        print('... RDF triples inserted into KG ...')
        print()
    
    # create rdf triples describing the current VRD image object 
    results = vrdu4.build_triples_for_image(imname, graphStore='graphdb')
    kg_image_id, kg_image_uriref, triples = results
    kg_vrd_img_ids.append(kg_image_id)
    image_triple_cnt += len(triples)
    
    # concatenate the triples into one string
    rdf_statements = ''
    for triple in triples:
        rdf_statements = rdf_statements + ' ' + triple
        if verbose2_mode:
            print(triple)
      
    # insert the RDF triples into the default graph of a GraphDB repository
    vrdu15.post_rdf_statements_to_default_graph_via_http(base_graphdb_url, 
                                                         rdf_statements,
                                                         curl_dir,
                                                         verbose=False)

    outcome, resp_code, msg, msg_info = vrdu15.get_http_interaction_outcome(curl_dir, 
                                                                            verbose=False)
    
    if outcome != 'success':
        print()
        print('insert of triples for Image entity did not succeed')
        print()
        print(f'img_idx {img_idx}, img_name {imname}')
        print()
        print(f'interaction outcome: {outcome}')
        print(f'http_response code: {resp_code}')
        print(f'error msg: {msg}')
        print('error info:')
        print(msg_info)
        raise RuntimeError('problem interacting with GraphDB')

    if verbose2_mode:
        print()

    #
    # iterate over the VR annotations for the current image; convert
    # them into RDF triples and add them to the KG
    #

    vr_cnt = 0 

    for vr in imanno:
        
        vr_cnt += 1
        vr_triple_cnt = 0
        
        # get the elements of the current annotated VR 
        sub_idx = vr['subject']['category']
        sub_bbox = tuple(vr['subject']['bbox'])
        prd_idx = vr['predicate']
        obj_idx = vr['object']['category']
        obj_bbox = tuple(vr['object']['bbox'])
        
        # process the 'subject' object of the VR
        if sub_bbox in image_objects:
            # the triples defining this object have already been added to
            # the KG; just retrieve its URI for reuse below
            _, kg_subject_uriref, _ = image_objects[sub_bbox]
            triples_A = []
            triple_cnt = 0
        else:
            # this is the first time we've encountered this particular object,
            # so build triples for defining it and add them to the KG
            results = vrdu4.build_triples_for_object(sub_idx, sub_bbox,
                                                     ontoClassNames,
                                                     kg_image_uriref,
                                                     graphStore='graphdb')
            kg_subject_id, kg_subject_uriref, triples_A = results
            triple_cnt = len(triples_A)
            image_objects[sub_bbox] = [kg_subject_id, kg_subject_uriref, sub_idx]
                 
        vr_triple_cnt += triple_cnt
        
        # process the 'object' object of the VR
        if obj_bbox in image_objects:
            # the triples defining this object have already been added to
            # the KG; just retrieve its URI for reuse below
            _, kg_object_uriref, _ = image_objects[obj_bbox]
            triples_B = []
            triple_cnt = 0
        else:
            # this is the first time we've encountered this particular object,
            # so build triples for defining it and add them to the KG            
            results = vrdu4.build_triples_for_object(obj_idx, obj_bbox,
                                                     ontoClassNames,
                                                     kg_image_uriref,
                                                     graphStore='graphdb')
            kg_object_id, kg_object_uriref, triples_B = results
            triple_cnt = len(triples_B)
            image_objects[obj_bbox] = [kg_object_id, kg_object_uriref, obj_idx]
            
        vr_triple_cnt += triple_cnt
        
        # link the 'subject' object to the 'object' object of the VR;
        # use the ontology object property that corresponds to the VRD
        # predicate in the annotated VR; the resulting single triple is the 
        # key triple that represents (expresses) the visual relationship 
        # between the particular ordered pair of objects referenced in the
        # current annotated VR
        triples_C = vrdu4.build_triple_linking_subject_to_object(kg_subject_uriref,
                                                                 prd_idx,
                                                                 kg_object_uriref,
                                                                 ontoPropNames,
                                                                 graphStore='graphdb')
        vr_triple_cnt += len(triples_C)
        
        # concatenate all of the triples associated with the current 
        # annotated VR into one string
        rdf_statements = ''
        
        if len(triples_A) > 0:
            for triple in triples_A:
                rdf_statements = rdf_statements + ' ' + triple
                if verbose2_mode:
                    print(triple)
            if verbose2_mode:
                print()
        else:
            if verbose2_mode:
                print(f'{kg_subject_uriref} as subject')
                print()
        
        if len(triples_B) > 0:
            for triple in triples_B:
                rdf_statements = rdf_statements + ' ' + triple
                if verbose2_mode:
                    print(triple)
            if verbose2_mode:
                print()
        else:
            if verbose2_mode:
                print(f'{kg_object_uriref} as object')
                print()
        
        for triple in triples_C:
            rdf_statements = rdf_statements + ' ' + triple
            if verbose2_mode:
                print(triple)
        if verbose2_mode:
            print()
            
        # insert the rdf triples for the current annotated VR into the 
        # default graph of a GraphDB repository
        vrdu15.post_rdf_statements_to_default_graph_via_http(base_graphdb_url, 
                                                             rdf_statements,
                                                             curl_dir,
                                                             verbose=False)
        
        outcome, resp_code, msg, msg_info = vrdu15.get_http_interaction_outcome(curl_dir, 
                                                                                verbose=False)
        
        if outcome != 'success':
            print()
            print('insert of triples for annotated VRs did not succeed')
            print()
            print(f'img_idx {img_idx}, img_name {imname}')
            print()
            print(f'interaction outcome: {outcome}')
            print(f'http_response code: {resp_code}')
            print('error msg:')
            print(msg)
            print()
            print('error info:')
            print(msg_info)
            raise RuntimeError('problem interacting with GraphDB')         
              
        image_triple_cnt += vr_triple_cnt
    
     
    # save the 'image_objects' dictionary for the current image; we'll reuse 
    # this info later when we extract triples from the expanded (materialised)
    # KG for this image and convert them back into VR annotations; 
    # the reason we use this tactic is simply for processing efficiency;
    # the reasoning (materialisation) of the KG won't invent new objects
    # or affect the bounding boxes of existing objects in any way;
    # so it's more efficient to keep track the bounding boxes for each 
    # object in a given image externally from the KG, and then reuse them
    # as appropriate when reconstituting the augmented set of visual
    # relationship annotations for each image.
    objects_per_image.append(image_objects)
    
    # save the number of VRs associated with the current image (prior to
    # KG materialisation)
    vrd_img_vr_cnts.append(vr_cnt)
    
    if verbose_mode:
        print()
        print(f'number of annotated VRs for image: {vr_cnt}')
        print(f'triples loaded to KG for image   : {image_triple_cnt}')
    
    total_triple_cnt += image_triple_cnt


end_time = time.time()
annotated_vr_load_and_infer_time = end_time - start_time

print()
print(f'Number of images processed: {image_cnt}')
print()
print(f'Total VR-related triples loaded to KG: {total_triple_cnt}')


#%% check the KG size after inserting triples for all the annotated VRs

graph_name = 'default'

print()
print('Checking size of GraphDB default graph ...')

outcome, n_triples = vrdu15.get_graph_size(base_graphdb_url, graph_name)

print(f'GraphDB interaction outcome: {outcome}')
print(f"Number of triples in graph '{graph_name}': {n_triples}")


#%% extract the augmented triples and convert back to VR annotations

print() 
print('----------------------------------------------------------')
print()
print('Extracting augmented triples from KG and converting to annotated VRs ...')
print()


# maintain a dictionary to hold the KG-augmented list of visual relationships
# (VRs) for each image in the VRD dataset
vrd_anno_augmented = {}


# iterate over the image names whose scene graphs were augmented
for idx, imname in enumerate(vrd_img_names): 
    
    # get the original VRs for the current image
    imanno = vrd_anno[imname]     

    # get the unique id used in the KG to represent the current VRD image
    kg_image_id = kg_vrd_img_ids[idx]
 
    # build a SPARQL query designed to retrieve the triples in the KG 
    # associated with the visual relationships for the current image
    query = vrdu4.assemble_SPARQL_query(imname, kg_image_id, graphStore='graphdb')

    if verbose_mode:    
        print(f"\nextracting triples from KG for image '{imname}'")
    else:
        if idx % 100 == 0:
            print(f"extracting VRs from KG for image idx {idx}")
            if redirect_stdout:
                sys.stdout.flush()
    
    # execute the SPARQL query   
    results, exception_info = vrdu15.execute_sparql_query(base_graphdb_url, query)

    # evaluate the outcome
    outcome = vrdu15.get_sparql_interaction_outcome(results, exception_info)
    
    if outcome == 'failure':
        print(exception_info)
        raise RuntimeError('Interaction via Sparql failed')
    elif outcome == 'problem':
        print('problem')
        raise RuntimeError('Interaction via Sparql problem')
    else:
        pass
    
    # report the original number of VRs for the current image and the
    # new number of VRs after KG materialisation for comparison
    if verbose_mode:
        print(f"number of original VRs: {vrd_img_vr_cnts[idx]}")
        print(f"number of expanded VRs: {len(results['results']['bindings'])}")
        print()
    
    # get the objects and their bboxes for current image that were saved earlier
    image_objects = objects_per_image[idx]
    
    # create an inverted dictionary keyed by the kg_object_id value
    # (nb: we can ignore the middle 'value' element, v[1],
    #      so each entry will be: 'kg_object_id : [bbox, obj_idx]')
    inverted_image_objects = { v[0]: [k, v[2]] for k, v in image_objects.items() }

    # establish a list to hold the soon-to-be reconstructed set of VRs 
    # for the current image
    imanno_aug = []

    # iterate over the triples extracted from the KG for the current image;
    # each one effectively corresponds to a distinct VR;
    # validate them and convert them from their KG representations back
    # into standard VR format
    
    for triple in results['results']['bindings']:
        
        # ensure the two objects in the current triple are DIFFERENT
        # (nb: if they're not different, something in the SPARQL query
        # used to extract triples from the KG is broken)
        if triple['subObj']['value'] == triple['objObj']['value']:
            raise ValueError(f'subObj same as objObj on image f{imname}')
                
        # convert the KG URI for the property linking the two objects to its 
        # corresponding VRD predicate integer index representation
        # - nb: since the ontoPropNames have positional correspondence
        #   with the VRD predicate names, the index position of the 
        #   ontoProperty within the ontoPropNames list will be the
        #   correct VRD predicate integer label
        # - nb: if an ontoProperty is not recognised as valid, an Exception
        #   will automatically be thrown which stops processing
        ontoProperty = triple['property']['value'].split('#')[1]
        prd_idx = ontoPropNames.index(ontoProperty)
        
        # convert the KG URI for the 'subject' object to its corresponding
        # representation as a pair: bbox and class label index
        kg_object_id = triple['subObj']['value'].split('#')[1]
        sub_bbox, sub_idx = inverted_image_objects[kg_object_id]
        sub_bbox = list(sub_bbox)

        # convert the KG URI for the 'object' object to its corresponding
        # representation as a pair: bbox and class label index
        kg_object_id = triple['objObj']['value'].split('#')[1]
        obj_bbox, obj_idx = inverted_image_objects[kg_object_id]
        obj_bbox = list(obj_bbox)

        # assemble the elements into the dictionary format used for 
        # representing visual relationship annotations
        vr = {'predicate': prd_idx,
              'object': {'category': obj_idx, 'bbox': obj_bbox},
              'subject': {'category': sub_idx, 'bbox': sub_bbox}}

        # add the VR to the list of VRs for the current image
        imanno_aug.append(vr)
    
    # save the list of reconstituted VRs for the current image in the master
    # dictionary of VRs for all images being processed
    vrd_anno_augmented[imname] = imanno_aug


print() 
print('Extraction of triples from KG and conversion to annotated VRs complete')

#%% save augmented set of VR annotations to file on disk 

scene_graph_aug_filename = 'nesy4vrd_formatted_image_scene_graphs_aug_per_onto_v1_2c.json'

scene_graph_aug_filepath = os.path.join(work_dir, scene_graph_aug_filename)

with open(scene_graph_aug_filepath, 'w') as fp:
    json.dump(vrd_anno_augmented, fp) 

print()
print(f'KG-augmented scene graphs(s) saved: {scene_graph_aug_filepath}')


#%%

if redirect_stdout:
    # flush stdout buffer
    sys.stdout.flush()
    # close redirected output file
    sys.stdout.close()
    # restore sys.stdout to original file handler
    sys.stdout = stdout_file_saved


#%%

print()
print('Processing completed successfully!')


#%%

def get_visual_relationships(imanno, vrd_objects, vrd_predicates):
 
    vrs = []
    for vr in imanno:
        sub_cls_idx = vr['subject']['category']
        prd_cls_idx = vr['predicate']
        obj_cls_idx = vr['object']['category']
        sub_name = vrd_objects[sub_cls_idx]
        prd_name = vrd_predicates[prd_cls_idx]
        obj_name = vrd_objects[obj_cls_idx]
        vrs.append((sub_name, prd_name, obj_name))

    return vrs


#%% display initial scene graph

imname = 'image_hypothetical_01.jpg'

imanno = vrd_anno[imname]
vrs = get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% display KG-augmented scene graph

imanno = vrd_anno_augmented[imname]
vrs = get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)





