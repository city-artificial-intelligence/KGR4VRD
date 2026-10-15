#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module contains utility functions that support interaction with a KG
in some KG triple store.

The KG tools (triple stores) with which interaction is supported are 
currently limited to:
    * GraphDB

The low-level mechanics of interacting with GraphDB (eg whether by SPARQL 
or by REST API) are defined in vrd_utils15.py

The functionality defined here, in this module, supports particular 
application tasks for which (or contexts in which) we are wanting to interact 
with a KG. 
'''

#%%

import os 
import torch 

import vrd_utils15 as vrdu15


#%% global variables
  
#graphdb_address_port = 'http://192.168.0.151:7200'

graphdb_address_port = 'http://localhost:7200'
repository_name = 'test'
base_graphdb_url = graphdb_address_port + '/repositories/' + repository_name

# specify the name of the 'named graph' within GraphDB into which to 
# insert our data (RDF triple trios)
graph_name = 'http://data'
    
# specify directory for storing curl tool output files
curl_dir = os.path.join('~', 'research', 'curl')
curl_dir = os.path.expanduser(curl_dir)

# specify the VRD-World OWL ontology file to work with
ontology_name = 'vrd_world_v1_2_disjoint.owl'

#ontology_dir = os.path.join('~', 'Downloads')
ontology_dir = os.path.join('..', 'ontology')
ontology_dir = os.path.expanduser(ontology_dir)
ontology_path = os.path.join(ontology_dir, ontology_name)


#%%

def clear_graphdb_kg():
    
    clear_named_graph()
    
    clear_default_graph()
    
    return None 
    

#%% 

def clear_named_graph():
    
    vrdu15.clear_data_from_named_graph_via_http(base_graphdb_url,
                                                graph_name,
                                                curl_dir,
                                                verbose=False)
    
    # determine the outcome of the interaction with the KG 
    res = vrdu15.get_http_interaction_outcome(curl_dir, verbose=False)
    outcome, resp_code, msg, msg_info = res  
    if outcome != 'success':
        print('KG interaction failure when clearing the named graph of the KG')
        print() 
        print(f'HTTP response code: {resp_code}')
        print()
        print(f'msg: {msg}')
        print()
        print(f'msg_info: {msg_info}')
        print('------------------------------------')
        raise Exception('KG interaction failed clearing named graph')

    return None 


#%%

def clear_default_graph():
    
    vrdu15.clear_data_from_default_graph_via_http(base_graphdb_url,
                                                  curl_dir,
                                                  verbose=False)
    
    # determine the outcome of the interaction with the KG 
    res = vrdu15.get_http_interaction_outcome(curl_dir, verbose=False)
    outcome, resp_code, msg, msg_info = res  
    if outcome != 'success':
        print('KG interaction failure when clearing the default graph of the KG')
        print() 
        print(f'HTTP response code: {resp_code}')
        print()
        print(f'msg: {msg}')
        print()
        print(f'msg_info: {msg_info}')
        print('------------------------------------')
        raise Exception('KG interaction failed clearing default graph')
    
    return None 


#%% 

def load_ontology_into_graphdb_kg():
    
    
    vrdu15.post_rdf_file_to_default_graph_via_http(base_graphdb_url, 
                                                   ontology_path,
                                                   curl_dir,
                                                   verbose=False)
    
    # determine the outcome of the interaction with the KG 
    res = vrdu15.get_http_interaction_outcome(curl_dir, verbose=False) 
    outcome, resp_code, msg, msg_info = res     
    if outcome != 'success':
        print('KG interaction failure when loading ontology')
        print() 
        print(f'HTTP response code: {resp_code}')
        print()
        print(f'msg: {msg}')
        print()
        print(f'msg_info: {msg_info}')
        print('------------------------------------')
        raise Exception('KG interaction failed loading ontology')
    
    return None


#%%

def get_object_ordered_pair_object_classes(output_cell, ppnn_img_dict):
    '''
    Get the object classes for the object ordered pair associated with
    a given cell in a PPNN output matrix.
    '''
    
    obj_ord_pairs_dict = ppnn_img_dict['obj_ord_pairs']
    obj_ord_pairs_list = list(obj_ord_pairs_dict.keys())
    
    # the row of the output cell is an index (pointer) to a particular
    # object ordered pair, each of which has an associated object class
    row_idx = output_cell[0]
    
    # get the object ordered pair associated with the cell's row in the
    # PPNN output matrix
    obj_ord_pair = obj_ord_pairs_list[row_idx]
    
    # get the PPNN input training data features associated with the 
    # current object ordered pair 
    obj_ord_pair_features = obj_ord_pairs_dict[obj_ord_pair]
    
    # get the integer class labels (indices) of the object ordered pair
    obj1_class_idx = obj_ord_pair_features['b1_lab']
    obj2_class_idx = obj_ord_pair_features['b2_lab']
    
    return obj1_class_idx, obj2_class_idx


#%%

def output_cell_2_RDF_triple_trio(output_cell, 
                                  ppnn_img_dict,
                                  ontoClassNames,
                                  ontoPropNames,
                                  entity_seq_num):
    '''
    Convert an ordered pair of integers representing a cell of the PPNN
    output matrix (and, by extension, a predicted VR) into an RDF triple trio 
    representation of that predicted VR.

    Parameters
    ----------
    output_cell : tensor
        A 1D tensor with two integer elements indicating a particular cell 
        (row, col) of the 2D PPNN output matrix.  The cell of the PPNN output 
        matrix maps to a particular predicted VR. The row of the cell maps to
        a particular ordered pair of objects in an image. The col of the cell
        maps to the predicate that is being predicted to relate the two
        objects in a visual relationship (VR).
    
    ppnn_img_dict : dictionary
        A dictionary contain the PPNN input training data for a given image.

    Returns
    -------
    rdf_triple_trio : List
        A list of strings, where each string is a formatted RDF triple
        ready for insertion into a GraphDB KG.
    
    
    A template for an RDF triple trio looks like this:
        
    1)    vrd:object1 vrd:propName? vrd:object2
    2)    vrd:object1 rdf:type vrd:className1?
    3)    vrd:object2 rdf:type vrd:className2?
    
    The only variables in the template are the elements with '?' at the end.
    We can use fixed names for the individuals (entities) in the triple trio 
    because of the way we've chosen to interact with our GraphDB KG.
    '''
    
    # set the prefix to use for entity, property and class names
    prefix = 'vrd:'
    
    # increment the sequence number we use to make entity names unique
    entity_seq_num['entity_seq_num'] += 1
    
    # get the incremented sequence number 
    seq_num = entity_seq_num['entity_seq_num']   
    
    generic_object_1_name = prefix + 'object1_' + str(seq_num)
    generic_object_2_name = prefix + 'object2_' + str(seq_num)
    
    res = get_object_ordered_pair_object_classes(output_cell, ppnn_img_dict)
    obj1_class_idx, obj2_class_idx = res
    
    obj1_class_name = ontoClassNames[obj1_class_idx]
    obj2_class_name = ontoClassNames[obj2_class_idx]
    
    col_idx = output_cell[1]
    
    predicate_property_name = ontoPropNames[col_idx]
    
    rdf_triple_trio = []
     
    # create triple (1): vrd:object1 rd:propName? vrd:object2
    property_name = prefix + predicate_property_name
    triple = generic_object_1_name + ' ' + property_name + ' ' + generic_object_2_name + ' .'
    rdf_triple_trio.append(triple)
    
    # create triple (2): vrd:object1 rdf:type vrd:className1?
    class_name = prefix + obj1_class_name
    triple = generic_object_1_name + ' rdf:type ' + class_name + ' .'
    rdf_triple_trio.append(triple)
    
    # create triple (3): vrd:object2 rdf:type vrd:className2?
    class_name = prefix + obj2_class_name
    triple = generic_object_2_name + ' rdf:type ' + class_name + ' .'
    rdf_triple_trio.append(triple)  
    
    return rdf_triple_trio


#%%

def call_kg_for_nnkgs1(cell, 
                       ppnn_img_dict,
                       ontoClassNames,
                       ontoPropNames,
                       entity_seq_num,
                       kg_interaction_style):
    
    verbose = False
    
    # convert cell in PPNN output matrix to RDF triple trio
    rdf_triple_trio = output_cell_2_RDF_triple_trio(cell, 
                                                    ppnn_img_dict,
                                                    ontoClassNames,
                                                    ontoPropNames,
                                                    entity_seq_num)
        
    # concatenate the triples in the trio into one string
    rdf_statements = ''
    for triple in rdf_triple_trio:
        rdf_statements = rdf_statements + ' ' + triple
        
    # attempt to insert the RDF triple trio into the KG
    # (IE: use the KG and KG reasoning as a symbolic binary classifier to 
    #  classify the predicted VR represented by the RDF triple trio as 
    #  being semantically valid or semantically invalid)
    if kg_interaction_style == 1:
            
        vrdu15.put_rdf_statements_to_named_graph_via_http(base_graphdb_url,
                                                          rdf_statements,
                                                          graph_name,
                                                          curl_dir,
                                                          verbose=False)
            
    elif kg_interaction_style == 2:

        vrdu15.post_rdf_statements_to_default_graph_via_http(base_graphdb_url,
                                                             rdf_statements,
                                                             curl_dir,
                                                             verbose=False)
        
    else:
            
        raise ValueError(f'KG interaction style not recognised: {kg_interaction_style}')
    
    # determine the outcome of the interaction with the KG 
    outcome, resp_code, msg, msg_info = vrdu15.get_http_interaction_outcome(curl_dir, 
                                                                            verbose=False)
    
    # initialise our return data
    # (nb: a value of 0 corresponds to 'unclassified'; in this case, 
    #  a decision of 'unclassified' only arises if the KG interaction 
    #  fails in some unexpected way 
    kg_vr_type_classification_decision = 0.0
    
    if outcome == 'success':
        
        # the KG classified the cell's VR type as 'valid'
        kg_vr_type_classification_decision = 1.0 
         
        if verbose:
            print()
            print('insertion of this triple trio succeeded:')
            for triple in rdf_triple_trio:
                print(triple)                
    
    elif outcome == 'failure':
                   
        # the KG classified the cell's VR type as 'invalid'
        kg_vr_type_classification_decision = 2.0 
        
        if verbose:
            print()
            print('insertion of this triple trio failed:')
            for triple in rdf_triple_trio:
                print(triple)
        
        # note: we always get an HTTP response code of 500 for an RDF
        # triple trio insertion failure that arises from a GraphDB
        # consistency check rule being violated; but we'd like to know
        # if we ever get something back other than response code 500
        if not resp_code == 500:
            print('INFO: KG interaction anomaly') 
            print(f'unexpected HTTP response code: {resp_code}')
            print(f'msg: {msg}')
            print(f'msg_info: {msg_info}')
            
    elif outcome == 'problem':
        
        print('PROBLEM: KG interaction problem')
        print(f'HTTP response code: {resp_code}')
        print(f'msg: {msg}')
        print(f'msg_info: {msg_info}')

    else:
        
        print('PROBLEM: KG interaction problem: outcome not recognised')
            
      
    return kg_vr_type_classification_decision 


#%%

def interact_with_kg_for_nnkgs1(output, 
                                ppnn_img_dict, 
                                ontoClassNames, 
                                ontoPropNames,
                                entity_seq_num,
                                kg_vr_type_classifications,
                                kg_interaction_results,
                                kg_interaction_style,
                                tr_d_kgS1):
    '''
    Evaluate the VRs that the PPNN is learning to predict to determine
    whether or not they represent semantically valid VRs with respect to
    the VRD-World ontology. 
    
    Build and return a binary mask matrix that identifies which, if any,
    of the predicted VRs implied by the PPNN output matrix are, in fact,
    semantically invalid (illegal) VRs.

    Parameters
    ----------
    output : tensor
        The output matrix returned by the forward pass of the PPNN model.
        It contains raw logit values (real numbers between negative and 
        positive infinity). A 2D matrix.
    ppnn_img_dict : dictionary
        A dictionary containing the PPNN input training data for a given
        image.

    Returns
    -------
    cells_for_loss_penalty : tensor 
        A 2D tensor (matrix) the same size as the 'output' matrix accepted
        as an input parameter. A 1 indicates that the corresponding 
        emerging prediction for a VR is a semantically invalid VR
        according to KG reasoning wrt the VRD-World ontology.
    
    NOTE: parameter kg_vr_type_classifications is mutable and is updated
    in-place within this function; this is deliberate and essential and an
    important side-effect that readers of the code must appreciate
    '''
    
    verbose = True 

    #
    # -----------------
    #
    
    if output.shape[1] > 71:  # 71 = number of NeSy4VRD predicates
        # if the output matrix has a special column for a 'no predicate'
        # predicate, ignore this column because it can never correspond
        # to a KG property; it will cause in 'index out of range' error
        output_probs = torch.sigmoid(output[:,0:-1])
    else:
        output_probs = torch.sigmoid(output)

    mask = output_probs > 0.5
    
    cells_to_evaluate = mask.nonzero()
    
    n_cells_to_evaluate = cells_to_evaluate.shape[0]
    
    if verbose:
        print(f'KG interaction - n_cells to evaluate: {n_cells_to_evaluate}')
    

    # ------------
    
    #
    # within the set of PPNN output cells identified for evaluation, isolate
    # the ones for which the corresponding VR type has not yet been classified 
    # by the KG; these are the only ones for which we need to interact with
    # the KG; for the others, we already know the VR type classification
    # (semantically valid or semantically invalid)
    #
    
    cells_needing_classification = []
    
    for cell in cells_to_evaluate:
        property_idx = cell[1] 
        res = get_object_ordered_pair_object_classes(cell, ppnn_img_dict)
        obj1_class_idx, obj2_class_idx = res 
        if kg_vr_type_classifications[property_idx, obj1_class_idx, obj2_class_idx] == 0.0:
            cells_needing_classification.append(cell)
       
    if verbose:
        print(f'KG interaction - n_cells needing classification: {len(cells_needing_classification)}')
        print(f'PPNN output cells needing classification: {cells_needing_classification}')
    
    # ------------------
    
    #
    # process the PPNN output matrix cells for which the class (valid or invalid)
    # of the corresponding VR type is not yet been determined
    #
       
    kg_call_count = 0
    n_cells_newly_classified_as_valid_vrs = 0
    n_cells_newly_classified_as_invalid_vrs = 0
    
    # iterate over the PPNN output cells whose VR types need classification by the KG
    for cell in cells_needing_classification:
        
        #
        # if possible, get the class (valid or invalid) of the cell's VR type 
        # from the 3D tensor of VR type classes; (it's possible that an
        # earlier cell in the list mapped to the same VR type and that its
        # class is now known, and we don't need to call the KG after all)
        #
        
        prop_idx = cell[1] 
        res = get_object_ordered_pair_object_classes(cell, ppnn_img_dict)
        obj1_class_idx, obj2_class_idx = res
        
        cell_vr_type_class = kg_vr_type_classifications[prop_idx, obj1_class_idx, obj2_class_idx]
        
        if cell_vr_type_class == 0.0:
            kg_interaction_required = True
        else:
            kg_interaction_required = False 
            if not cell_vr_type_class in [1.0, 2.0]:
                raise ValueError(f'cell VR type class not recognised (a): {cell_vr_type_class}')
                
        #
        # when required, call the KG to classify the cell's VR type 
        #
        
        if kg_interaction_required:
            
            cell_vr_type_class = call_kg_for_nnkgs1(cell, ppnn_img_dict,
                                                    ontoClassNames, ontoPropNames,
                                                    entity_seq_num, kg_interaction_style)

            if not cell_vr_type_class in [0.0, 1.0, 2.0]:
                raise ValueError(f'cell VR type class not recognised (b): {cell_vr_type_class}')
            
            kg_call_count += 1
            
            #
            # record the VR type classification decision in 3D tensor straight away;
            # this way, if a subsequent cell in the list maps to the same VR type,
            # we can avoid calling the KG again, redundantly
            #
                
            if cell_vr_type_class == 1.0:  # valid VR type
                
                # in the 3D tensor, record that the VR type is 'valid' 
                kg_vr_type_classifications[prop_idx, obj1_class_idx, obj2_class_idx] = 1.0
                if verbose:
                    print('cell newly classified as valid VR:')
                    print(f'prop_idx {prop_idx}, obj1_class_idx {obj1_class_idx}, obj2_class_idx {obj2_class_idx}')
            
            elif cell_vr_type_class == 2.0:  # invalid VR type
                
                # in the 3D tensor, record that the VR type is 'invalid' 
                kg_vr_type_classifications[prop_idx, obj1_class_idx, obj2_class_idx] = 2.0
                if verbose:
                    print('cell newly classified as invalid VR:')
                    print(f'prop_idx {prop_idx}, obj1_class_idx {obj1_class_idx}, obj2_class_idx {obj2_class_idx}')

            elif cell_vr_type_class == 0.0:
                
                print('PROBLEM: KG interaction failed unexpectedly')
                print('VR type for cell unknown')
                raise ValueError('PROBLEM: KG failed to classify cell VR type')
                
            else:
                    
                raise ValueError('cell_vr_type_class not recognised')

        
        # count the new classifications
        if cell_vr_type_class == 1.0:
            n_cells_newly_classified_as_valid_vrs += 1
        elif cell_vr_type_class == 2.0:
            n_cells_newly_classified_as_invalid_vrs += 1 
        
        #
        # record the class (valid or invalid) of the cell's VR type in the 
        # image-specific 2D matrix we use for holding these results
        #    

        row, col = cell[0], cell[1]         
        if cell_vr_type_class == 1.0:
            kg_interaction_results[row, col] = 1.0        
        elif cell_vr_type_class == 2.0:
            kg_interaction_results[row, col] = 2.0   
        
        
    # -------------------------------
    
    if verbose:
        print() 
        print(f'KG interaction - n_cells newly classified as valid VRs: {n_cells_newly_classified_as_valid_vrs}') 
        print(f'KG interaction - n_cells newly classified as invalid VRs: {n_cells_newly_classified_as_invalid_vrs}')

    # --------------------------------   

    #
    # build a binary matrix to communicate which cells of the PPNN
    # output matrix are associated with VRs that the KG has classified
    # as being semantically invalid
    #
    # this set of cells increases in size as training proceeds; we
    # consider all of them, cumulatively, not just the ones that may
    # have been newly classified as invalid in the current function call
    #
    # all cells representing VRs classified as semantically invalid by the 
    # KG attract a loss penalty during loss computation, not just those
    # newly classified as invalid in the current function call
    #
    
    mask = kg_interaction_results == 2.0
    cells_for_loss_penalty = torch.zeros(output.size()) 
    cells_for_loss_penalty[mask] = 1.0
    
    if verbose:
        print(f'KG interaction - n_cells for invalid VRs for loss penalty: {torch.sum(cells_for_loss_penalty)}')
    
    
    return cells_for_loss_penalty, kg_interaction_results, kg_call_count


#%%

def vr_2_RDF_triple_trio(prop_idx, obj1_cls_idx, obj2_cls_idx, 
                         entity_seq_num, ontoClassNames, ontoPropNames):

    '''
    Create an RDF triple trio to represent a visual relationship (VR)
    '''
    
    # set the prefix to use for entity, property and class names
    prefix = 'vrd:'
    
    # increment the sequence number we use to make entity names unique
    entity_seq_num['entity_seq_num'] += 1
    
    # get the incremented sequence number 
    seq_num = entity_seq_num['entity_seq_num']   
    
    generic_object_1_name = prefix + 'object1_' + str(seq_num)
    generic_object_2_name = prefix + 'object2_' + str(seq_num)
    
    obj1_cls_name = ontoClassNames[obj1_cls_idx]
    obj2_cls_name = ontoClassNames[obj2_cls_idx]
    
    predicate_property_name = ontoPropNames[prop_idx]
    
    rdf_triple_trio = []
     
    # create triple (1): vrd:object1 rd:propName? vrd:object2
    property_name = prefix + predicate_property_name
    triple = generic_object_1_name + ' ' + property_name + ' ' + generic_object_2_name + ' .'
    rdf_triple_trio.append(triple)
    
    # create triple (2): vrd:object1 rdf:type vrd:className1?
    class_name = prefix + obj1_cls_name
    triple = generic_object_1_name + ' rdf:type ' + class_name + ' .'
    rdf_triple_trio.append(triple)
    
    # create triple (3): vrd:object2 rdf:type vrd:className2?
    class_name = prefix + obj2_cls_name
    triple = generic_object_2_name + ' rdf:type ' + class_name + ' .'
    rdf_triple_trio.append(triple)  
    
    return rdf_triple_trio


#%%

def call_kg_to_classify_vr_type(prop_idx, 
                                obj1_cls_idx, 
                                obj2_cls_idx, 
                                entity_seq_num, 
                                ontoClassNames, 
                                ontoPropNames):
    
    verbose = False
        
    rdf_triple_trio = vr_2_RDF_triple_trio(prop_idx, 
                                           obj1_cls_idx, 
                                           obj2_cls_idx, 
                                           entity_seq_num, 
                                           ontoClassNames, 
                                           ontoPropNames)
        
    # concatenate the triples in the trio into one string
    rdf_statements = ''
    for triple in rdf_triple_trio:
        rdf_statements = rdf_statements + ' ' + triple
        
    vrdu15.post_rdf_statements_to_default_graph_via_http(base_graphdb_url,
                                                         rdf_statements,
                                                         curl_dir,
                                                         verbose=False)        
        
    # determine the outcome of the interaction with the KG 
    outcome, resp_code, msg, msg_info = vrdu15.get_http_interaction_outcome(curl_dir, 
                                                                            verbose=False)
    
    vr_type_class = 0.0
    
    if outcome == 'success':  # valid VR type
            
        # record that the KG classified the VR type of the VR instance
        # as semantically 'valid'
        vr_type_class = 1.0 
                                 
        if verbose:
            print()
            print('insertion of this triple trio succeeded:')
            for triple in rdf_triple_trio:
                print(triple)                
        
    elif outcome == 'failure':  # invalid VR type 
                       
        # record that the KG classified the VR type of the VR instance
        # as semantically 'invalid'
        vr_type_class = 2.0 
            
        if verbose:
            print()
            print('insertion of this triple trio failed:')
            for triple in rdf_triple_trio:
                print(triple)
            
        # note: we always get an HTTP response code of 500 for an RDF
        # triple trio insertion failure that arises from a GraphDB
        # consistency check rule being violated; but we'd like to know
        # if we ever get something back other than response code 500
        if not resp_code == 500:
            print('INFO: KG interaction anomaly') 
            print(f'unexpected HTTP response code: {resp_code}')
            print(f'msg: {msg}')
            print(f'msg_info: {msg_info}')
                
    elif outcome == 'problem':
            
        print('PROBLEM: KG interaction problem')
        print(f'HTTP response code: {resp_code}')
        print(f'msg: {msg}')
        print(f'msg_info: {msg_info}')

    else:
            
        print('PROBLEM: KG interaction problem: outcome not recognised')
              
    
    return vr_type_class 


#%%

def get_pvr_vr_type_class(pvr, 
                          entity_seq_num, 
                          ontoClassNames, 
                          ontoPropNames,
                          master_vr_type_tensor):
    
    verbose = True
    
    vr_type_class = 0.0
    
    kg_call_made = False 
    
    if 'vr_type_class' in pvr:
        
        vr_type_class = pvr['vr_type_class']
    
    else:
        
        # check the master 3D VR type tensor for the class of the VR type
        
        prop_idx = pvr['predicate']
        obj1_cls_idx = pvr['subject']['category']
        obj2_cls_idx = pvr['object']['category']
        
        vr_type_class = master_vr_type_tensor[prop_idx, obj1_cls_idx, obj2_cls_idx] 
        vr_type_class = vr_type_class.item()  # convert from tensor to float 
        
        if vr_type_class == 0.0:  # VR type not yet classified by KG 
            
            #
            # call the KG to classify the VR type of the pvr 
            #
            
            vr_type_class = call_kg_to_classify_vr_type(prop_idx, 
                                                        obj1_cls_idx, 
                                                        obj2_cls_idx, 
                                                        entity_seq_num, 
                                                        ontoClassNames, 
                                                        ontoPropNames)
            kg_call_made = True
            
            if verbose:
                obj1_cls_name = ontoClassNames[obj1_cls_idx]
                prop_name = ontoPropNames[prop_idx]
                obj2_cls_name = ontoClassNames[obj2_cls_idx]
                print()
                print(f'kg called to classify VR type: {obj1_cls_idx}, {prop_idx}, {obj2_cls_idx}')
                print(f'({obj1_cls_name}, {prop_name}, {obj2_cls_name})')
                        
            if vr_type_class == 1.0:  # valid VR type, but a valid response
                
                pass
            
            elif vr_type_class == 2.0:  # invalid VR type, but a valid response
                
                pass

            elif vr_type_class == 0.0:
                
                print('PROBLEM: KG interaction failed unexpectedly')
                print('VR type for cell unknown')
                raise ValueError('PROBLEM: KG failed to classify cell VR type')
                
            else:
                 
                raise ValueError('PROBLEM: cell_vr_type_class not recognised')
            
            # store the class (semantically valid or invalid) of the VR type 
            # in the master 3D VR type tensor so the info about VR type classes
            # accumulates and minimises KG interaction in future 
            #
            # NOTE: the caller has a responsibility to SAVE the master 3D
            # VR type tensor if it is updated; the kg_call_made boolean
            # return flag allows the caller to do this when required;
            # if a KG call is made, the master is updated (unless something
            # goes wrong, of course)
            master_vr_type_tensor[prop_idx, obj1_cls_idx, obj2_cls_idx] = vr_type_class 
        
        # extend the pvr with its VR type class now that we know what it is 
        pvr['vr_type_class'] = vr_type_class     



    return vr_type_class, kg_call_made 


#%%

def get_indices_of_invalid_pvrs(predicted_vrs,
                                entity_seq_num,
                                ontoClassNames,
                                ontoPropNames,
                                master_vr_type_tensor):
    
    indices_of_invalid_pvrs = []
    
    n_kg_calls = 0
    
    for pvr_idx, pvr in enumerate(predicted_vrs):
        
        vr_type_class, kg_call_made = get_pvr_vr_type_class(pvr, 
                                                            entity_seq_num,
                                                            ontoClassNames, 
                                                            ontoPropNames,
                                                            master_vr_type_tensor)
                
        if vr_type_class == 2.0:  # semantically invalid
            indices_of_invalid_pvrs.append(pvr_idx)
        
        if kg_call_made:
            n_kg_calls += 1 
    
    return indices_of_invalid_pvrs, n_kg_calls 







