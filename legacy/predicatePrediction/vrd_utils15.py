#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Herron
"""

'''
This module defines functions for interacting with graphs in a GraphDB
repository.

Some functions support interaction via HTTP (ie via GraphDB's RDF4J REST API).
For this we use the 'curl' tool. We tell curl how to construct a particular
HTTP request and we os.system() to execute the curl command.

Some functions support interaction via SPARQL. For this we use the Python
package SPARQLWrapper.

For HTTP, functions exist to support:
    * POST (insert) an RDF file into the default graph
    * POST (insert) RDF statements (in a string) into the default graph
    * POST (insert) RDF statements (in a string) into a named graph
    * PUT (replace) RDF statements (in a string) in a named graph
    * clear data from the default graph
    * clear data from a named graph 

For SPARQL, this module provides functions for:
    * execute a SPARQL 1.1 Update update instruction (for default or named graph)
    * execute a SPARQL query (for default or named graph)
    * build a SPARQL 1.1 Update INSERT DATA instruction (for default or named graph)
    * build a SPARQL 1.1 Update CLEAR graph instruction (for default or named graph)
'''

#%%

from SPARQLWrapper import SPARQLWrapper, JSON, POST

import os


#%% 

def post_rdf_file_to_default_graph_via_http(base_url,
                                            rdf_filename,
                                            curl_dir,
                                            verbose=False):
    '''
    Load RDF statements contained in a file into the default graph of
    a GraphDB repository via HTTP (ie via the GraphDB REST API).
    
    This function creates and sends an HTTP POST request using the tool 'curl'.
    The data (RDF statements) are read from a file and packaged as the 
    payload before the HTTP POST request is sent to GraphDB via its REST API
    called the RDF4J API. (See http://localhost:7200/webapi on a running 
    GraphDB Workbench browser page for details.)
    
    The POST method of HTTP leads to a data payload being *merged* with any 
    existing data within the target resource at the server end. Thus, this
    function can be called multiple times to successively load multiple
    files of RDF statements into a given GraphDB graph. All the RDF statements
    will accumulate and co-exist in the graph.
    
    A curl command is constructed for creating the HTTP POST request.
    The use of the '--data' option is what tells curl that we want to use 
    the POST method of HTTP. When the '--data' option is used and the data
    payload is to be loaded from a file, prepending the filename with '@'
    instructs curl to find that file and load its contents as the payload
    of the HTTP POST request.
    
    As explained here, https://everything.curl.dev/http/post/simple.html,
    whereas normally curl sends exactly the bytes you give it, there is an
    exception when option '--data' is used. With option '--data', curl
    skips over the carriage return and newline characters. If you want the
    carriage return and newline characters included, you need to use the
    option variant '--data-binary', as we do here.
    
    The curl command is constructed to tell curl to send any output 
    associated with the HTTP response from the server (GraphDB) to one of 
    two files: 
        a) output that curl would otherwise send to stdout is redirected to 
           file curl_out.txt
        b) output that curl would otherwise send to stderr is redirected to 
           file curl_err.txt
    
    The '--write-out' option uses a tiny template to extract from the
    HTTP response from GraphDB a variable of interest: the HTTP response code.
    This information is sent to stderr, which has itself been redirected to
    the file curl_err.txt.  
    
    The main reason for writing curl's output to files is so that callers
    of this function can inspect it and act on it.
    
    It's the job of the caller of this function to inspect the contents of 
    the two output files, curl_out.txt and curl_err.txt, to determine whether 
    the data transfer operation was successful or failed for some reason. 
    
    The HTTP response code is extracted and written to curl_err.txt. This
    happens regardless of the value of the response code (ie regardless of
    whether the operation succeeded or failed). Inspecting the HTTP
    response code in curl_err.txt is the primary means for callers of this
    function to determine whether a data transfer operation succeeded or
    failed.
    
    If an operation succeeds, a fresh curl_out.txt is created but nothing
    is written to it, so it remains empty.  An empty curl_out.txt file is 
    thus another indicator of a successful operation. The condition of an
    empty curl_out.txt file should always occur with an HTTP response code
    (in the curl_err.txt file) of 2xx (failure), in particular, of 204.  If 
    these two indicators of success ever do not co-occur, investigation is 
    required urgently because it's likely a problem exists.
    
    If an operation fails, a fresh curl_out.txt is created and the 
    message of the ClientHTTPException thrown within GraphDB is returned 
    in the HTTP response and is written to it as its only contents.  Thus,
    a non-empty curl_out.txt file is another indicator of a failed operation. 
    The condition of a non-empty curl_out.txt file should always occur with 
    an HTTP response code of 4xx (success). If these two indicators of 
    failure ever do not co-occur, investigation is required urgently because
    it's likely a problem exists.
    
    If verbose=True, various headers for the HTTP request and response are
    written to stderr (ie to curl_err.txt). This includes the HTTP response
    code. So if verbose=True, there are two instances of the HTTP response
    code in curl_err.txt, so extracting it via the '--write-out' option
    becomes a bit redundant.
    
    The status returned by the os.system() command is not useful. It can be
    zero regardless of whether a data transfer operation succeeds or fails,
    and it bears no relation to the HTTP response code returned by GraphDB.
    
    NOTE: an alternate way to achieve the same outcome is to use the '-T' 
    option instead of '--data' or '--data-binary', and to explicitly set
    the method to POST with '-X POST'. Option '-T' reads from a file, and  
    does not skip over carriage returns and newlines. But the '-T' option
    sets the HTTP method to PUT. That's why we need to override that by
    explicitly setting the method to POST with '-X POST'.
    '''
    
    curl_out_path = os.path.join(curl_dir, 'curl_out.txt')
    curl_err_path = os.path.join(curl_dir, 'curl_err.txt')
    
    if verbose:
        print()
        print(f'Data to merge into graph will be loaded from file: {rdf_filename}')
        print()
    
    if verbose: 
        cmd = "curl" + \
              " -H 'Content-Type: text/turtle'" + \
              " --data-binary @" + rdf_filename + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              " --verbose " + \
              base_url + "/rdf-graphs/service?default"        
    else: 
        cmd = "curl" + \
              " -H 'Content-Type: text/turtle'" + \
              " --data-binary @" + rdf_filename + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              base_url + "/rdf-graphs/service?default"  
    
    #if verbose:
    #    print(cmd)
    #    print()

    #status = os.system(cmd)
    #print(f'status: {status}')
    
    _ = os.system(cmd)
    
    return None


#%%

def post_rdf_statements_to_default_graph_via_http(base_url,
                                                  rdf_statements,
                                                  curl_dir,
                                                  verbose=False):
    '''
    Load RDF statements into the default graph of a GraphDB repository via
    HTTP (ie via the RDF4J REST API).
    
    This function is designed for loading small sets of prepared RDF
    statements formatted as one Python string.
    
    This function creates and sends an HTTP POST request using the tool 'curl'.
    The data (RDF statements) are packaged directly into the payload of the 
    HTTP POST request, as is.
    
    The POST method of HTTP leads to a data payload being *merged* with any 
    existing data within the target resource at the server end. Thus, this
    function can be called multiple times to successively load different
    sets of RDF statements into the default graph of a GraphDB repository.
     
    GraphDB manages the data payload as a single transaction. Thus, for
    example, if one or more statements in the payload cause a consistency
    check rule to fail and a ConsistencyCheck exception to be thrown, the 
    whole PUT operation fails and the whole transaction is rolled back. So
    the state of the GraphDB repository is not altered in any way. Nothing
    explicit or implicit is added to any graph in the repository. So there
    is nothing to remove (eg to clean up, or to manage data accumulation,
    or to ensure lingering inferred triples don't affect reasoning wrt 
    subsequent updates').
    '''

    curl_out_path = os.path.join(curl_dir, 'curl_out.txt')
    curl_err_path = os.path.join(curl_dir, 'curl_err.txt')
        
    if verbose:
        print()
        print('Data payload:')
        print(rdf_statements)
        print()
    
    if verbose: 
        cmd = "curl" + \
              " -H 'Content-Type: text/turtle'" + \
              " --data '" + rdf_statements + "'" + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              " --verbose " + \
              base_url + "/rdf-graphs/service?default"        
    else: 
        cmd = "curl" + \
              " -H 'Content-Type: text/turtle'" + \
              " --data '" + rdf_statements + "'" + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              base_url + "/rdf-graphs/service?default"  
    
    if verbose:
        print(cmd)
        print()

    #status = os.system(cmd)
    #print(f'status: {status}')
    
    _ = os.system(cmd)
    
    return None


#%%

def post_rdf_statements_to_named_graph_via_http(base_url,
                                                rdf_statements,
                                                graph_name,
                                                curl_dir,
                                                verbose=False):
    '''
    Load RDF statements into a named graph of a GraphDB repository via
    HTTP (ie via the RDF4J REST API).
    
    This function is designed for loading small sets of prepared RDF
    statements formatted as one Python string.
    
    This function creates and sends an HTTP POST request using the tool 'curl'.
    The data (RDF statements) are packaged directly into the payload of the 
    HTTP POST request, as is.
    
    The POST method of HTTP leads to a data payload being *merged* with any 
    existing data within the target resource at the server end. Thus, this
    function can be called multiple times to successively load different
    sets of RDF statements into a named graph.
    
    If the named graph does not already exist, it is created.
    
    GraphDB manages the data payload as a single transaction. Thus, for
    example, if one or more statements in the payload cause a consistency
    check rule to fail and a ConsistencyCheck exception to be thrown, the 
    whole PUT operation fails and the whole transaction is rolled back. So
    the state of the GraphDB repository is not altered in any way. Nothing
    explicit or implicit is added to any graph in the repository. So there
    is nothing to remove (eg to clean up, or to manage data accumulation,
    or to ensure lingering inferred triples don't affect reasoning wrt 
    subsequent updates').
    '''

    curl_out_path = os.path.join(curl_dir, 'curl_out.txt')
    curl_err_path = os.path.join(curl_dir, 'curl_err.txt')

    if verbose:
        print()
        print(f'Data to be merged into named graph: {rdf_statements}')
        print()
    
    if verbose: 
        cmd = "curl" + \
              " -H 'Content-Type: text/turtle'" + \
              " --data '" + rdf_statements + "'" + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              " --verbose " + \
              base_url + "/rdf-graphs/service?graph=" + graph_name       
    else: 
        cmd = "curl" + \
              " -H 'Content-Type: text/turtle'" + \
              " --data '" + rdf_statements + "'" + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              base_url + "/rdf-graphs/service?graph=" + graph_name  
    
    if verbose:
        print(cmd)
        print()

    #status = os.system(cmd)
    #print(f'status: {status}')
    
    _ = os.system(cmd)
    
    return None


#%%

def put_rdf_statements_to_named_graph_via_http(base_url,
                                               rdf_statements,
                                               graph_name,
                                               curl_dir,
                                               verbose=False):
    '''
    Put RDF statements into a named graph of a GraphDB repository via HTTP
    (ie via the RDF4J REST API).
    
    This function puts a set of prepared RDF statements (formatted as one 
    string) into a named graph in a GraphDB repository. If the named graph
    does not exist, it is created.  If the named graph already exists, its
    contents are replaced with the new RDF statements.
    
    This function creates and sends an HTTP PUT request using the tool 'curl'.
    The prepared RDF statements are packaged directly into the payload of the 
    HTTP PUT request, as is.
    
    The PUT method of HTTP leads to a data payload *replacing* the target
    resource, in this case a named graph in GraphDB. Multiple, successive
    calls will establish (or not) multiple different instantiations of the
    named graph.  Thus, this function is logically equivalent to doing a
    CLEAR operation (to empty the named graph) followed by an HTTP POST 
    request to add (insert) new rdf data into the empty named graph.
    
    If the named graph does not already exist, it is created. Otherwise, it
    is replaced.
    
    GraphDB manages the data payload as a single transaction. Thus, for
    example, if one or more statements in the payload cause a consistency
    check rule to fail and a ConsistencyCheck exception to be thrown, the 
    whole PUT operation fails and the whole transaction is rolled back. So
    the state of the GraphDB repository is not altered in any way. Nothing
    explicit or implicit is added to any graph in the repository. So there
    is nothing to remove (eg to clean up, or to manage data accumulation,
    or to ensure lingering inferred triples don't affect reasoning wrt 
    subsequent updates').
    '''

    curl_out_path = os.path.join(curl_dir, 'curl_out.txt')
    curl_err_path = os.path.join(curl_dir, 'curl_err.txt')

    if verbose:
        print()
        print(f'RDF statements to PUT into graph: {rdf_statements}')
        print()
    
    if verbose: 
        cmd = "curl" + \
              " -X PUT " + \
              " -H 'Content-Type: text/turtle'" + \
              " --data '" + rdf_statements + "'" + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              " --verbose " + \
              base_url + "/rdf-graphs/service?graph=" + graph_name    
    else: 
        cmd = "curl" + \
              " -X PUT " + \
              " -H 'Content-Type: text/turtle'" + \
              " --data '" + rdf_statements + "'" + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              base_url + "/rdf-graphs/service?graph=" + graph_name   
    
    if verbose:
        print(cmd)
        print()

    #status = os.system(cmd)
    #print(f'status: {status}')
    
    _ = os.system(cmd)
    
    return None


#%%

def clear_data_from_default_graph_via_http(base_url,
                                           curl_dir,
                                           verbose=False):
    '''
    Clear all RDF statements from the default graph of a GraphDB repository
    via HTTP (ie via the GraphDB REST API).
    
    NOTE: In GraphDB, clearing the default graph does not remove it; it
    remains intact but empty.
    '''

    curl_out_path = os.path.join(curl_dir, 'curl_out.txt')
    curl_err_path = os.path.join(curl_dir, 'curl_err.txt')    

    if verbose: 
        cmd = "curl" + \
              " -X DELETE " + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              " --verbose " + \
              base_url + "/rdf-graphs/service?default"        
    else: 
        cmd = "curl" + \
              " -X DELETE " + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              base_url + "/rdf-graphs/service?default"  
    
    if verbose:
        print(cmd)
        print()

    #status = os.system(cmd)
    #print(f'status: {status}')
    
    _ = os.system(cmd)

    return None


#%%

def clear_data_from_named_graph_via_http(base_url,
                                         graph_name,
                                         curl_dir,
                                         verbose=False):
    '''
    Clear all RDF statements from a named graph of a GraphDB repository
    via HTTP (ie via the RDF4J REST API).
    
    NOTE: In GraphDB, clearing a named graph removes the named graph. 
    (This is valid. The Sparql 1.1 Update specification permits 
     implementations to decide whether to remove empty graphs or not.)
    '''

    curl_out_path = os.path.join(curl_dir, 'curl_out.txt')
    curl_err_path = os.path.join(curl_dir, 'curl_err.txt')     

    if verbose: 
        cmd = "curl" + \
              " -X DELETE " + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              " --verbose " + \
              base_url + "/rdf-graphs/service?graph=" + graph_name        
    else: 
        cmd = "curl" + \
              " -X DELETE " + \
              " --output " + curl_out_path + " " + \
              " --stderr " + curl_err_path + " " + \
              " --write-out '%{stderr} http_response_code: %{response_code}' " + \
              " --silent " + \
              base_url + "/rdf-graphs/service?graph=" + graph_name  
    
    if verbose:
        print(cmd)
        print()

    #status = os.system(cmd)
    #print(f'status: {status}')
    
    _ = os.system(cmd)

    return None


#%%

def extract_response_code(line):
    '''
    Extract an HTTP response code form a string.
    
    The string should be formatted as ' http_response_code: nnn'
    '''
    
    http_response_code = 0
    
    tokens = line.split(':')
    if len(tokens) != 2:
        print(f'problem: unexpected number of tokens in line; {line}')
    else:
        rc_key = tokens[0].strip()
        if rc_key != 'http_response_code':
            print("problem: token 'http_response_code' not found")
        else:
            http_response_code = int(tokens[1].strip())
    
    return http_response_code


#%%

def get_http_interaction_outcome(curl_dir, verbose=False):
    '''
    Determine the outcome of an HTTP (REST API) interaction with GraphDB.
    
    Information regarding the latest HTTP (REST API) interaction with GraphDB 
    is stored in two text files: curl_out.txt and curl_err.txt
    
    HTTP response codes in range 4xx are client errors and in range 5xx are
    server errors. There are no HTTP response codes above 5xx.  So we treat
    any response code >= 400 as indicating a failure.
    
    If a GraphDB consistency check rule fails, GraphDB throws a 
    ConsistencyException and returns HTTP response code 500. The
    message for the ConsistencyException names the particular consistency
    check rule that failed.
    
    With GraphDB, if an error has occurred, the error message is on the
    first line of curl_out.txt.  In the case of consistency check rule 
    failures (exceptions), curl_out.txt also contains the triples involved
    in the logical contradiction that caused the consistency check to fail.
    These can serve as an 'explanation' for the consistency check failure.
    
    Parse and analyse the contents of the two files and determine the
    outcome category of the interaction:
        - success
        - failure
        - problem
    '''
   
    curl_out_path = os.path.join(curl_dir, 'curl_out.txt')
    curl_err_path = os.path.join(curl_dir, 'curl_err.txt') 
   
    #
    # get the http_response_code from the curl_err.txt file 
    #
    
    with open(curl_err_path) as fp:
        err_lines = fp.readlines()
    
    http_response_code = 0
    
    if len(err_lines) > 1:
        resp_code_found = False
        for idx, line in enumerate(err_lines):
            line = line.strip()
            if line.startswith('http_response_code'):
                http_response_code = extract_response_code(line)
                resp_code_found = True
        if not resp_code_found:
            raise ValueError('http_response_code not found')
    elif len(err_lines) == 1:
        http_response_code = extract_response_code(err_lines[0])
    else:
        raise ValueError('curl_err.txt file is empty')
    
    if verbose:
        print(f'curl_err.txt has http response code {http_response_code}')

    #
    # check the curl_out.txt file for a possible error message, and
    # possibly more as well
    #
    
    with open(curl_out_path) as fp:
        out_lines = fp.readlines()
    
    if len(out_lines) > 0:
        error_msg_returned = True
        error_msg = out_lines[0]
        if verbose:
            print('curl_out.txt is non-empty (error msg returned)')
            print(f'error msg: {error_msg}')
        error_info = ''
        if len(out_lines) > 1:
            error_info = out_lines[1:]
            
    else:
        error_msg_returned = False
        error_msg = ''
        error_info = ''
        if verbose:
            print('curl_out.txt is empty (no error msg returned)')
    
    #
    # evaluate the two response conditions and ensure they tally
    #
    
    if http_response_code == 204 and not error_msg_returned:
        outcome = 'success'
        if verbose:
            print('success response code and no error msg')
    elif http_response_code >= 400 and not error_msg_returned:
        outcome = 'problem'
        if verbose:
            print('problem: contradictory response conditions (a)')
    elif http_response_code == 204 and error_msg_returned:
        outcome = 'problem'
        if verbose:
            print('problem: contradictory response conditions (b)')
    elif http_response_code >= 400 and error_msg_returned:
        outcome = 'failure'
        if verbose:
            print('failure response code and error msg')
    else:
        outcome = 'problem'
        if verbose:
            print('problem: unrecognised response conditions')

    return outcome, http_response_code, error_msg, error_info


#%%

def get_sparql_interaction_outcome(results, exception_info):
    '''
    Determine the outcome of SPARQL interaction with GraphDB.
    '''
   
    if results == None and exception_info == None:
        outcome = 'problem'
    elif results == None and exception_info != None:
        outcome = 'failure'
    elif results != None and exception_info == None:
        outcome = 'success'
    elif results != None and exception_info != None:
        outcome = 'problem'
    else:
        outcome = 'problem'
    
    return outcome


#%%

def execute_sparql_update(sparql_query_endpoint, sparql_update_instruction):
    '''
    Execute a SPARQL update instruction per the Sparql 1.1 Update 
    specification.
    
    A common use case is inserting rdf statements into a graph with a
    Sparql 1.1 Update 'INSERT DATA' instruction.
    
    The incoming, prepared SPARQL update instruction controls whether the 
    instruction targets the default graph or a named graph.
    
    If the update instruction succeeds, some (limited) results are 
    returned.  If the update instruction fails for any reason, an 
    Exception is thrown and caught and returned.
    '''
    
    sparql_update_endpoint = sparql_query_endpoint + '/statements'

    results = None
    exception_info = None

    try:
        
        sparql = SPARQLWrapper(sparql_update_endpoint)
        
        sparql.setMethod(POST)

        sparql.addParameter(name='update', value=sparql_update_instruction)
        
        sparql.setReturnFormat(JSON)
        
        # execute the update instruction
        results = sparql.query()
                
    except Exception as e:

        exception_info = e 
    
    return results, exception_info


#%%

def build_sparql_insert_data_instruction(prefixes, graph_name, rdf_statements):
    '''
    Build a Sparql INSERT DATA instruction per the Sparql 1.1 Update
    specification.
    
    The insert can be targetted at either the default graph or at a 
    named graph.
    '''
    
    if len(graph_name) == 0:
        raise ValueError('graph name required')
    
    if graph_name == 'default':
        
        # insert into the default graph
        instruction = prefixes + " " + \
            """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            INSERT DATA { """ + \
                rdf_statements + \
            " }" 
            
    else:
    
        # insert into a named graph
        instruction = prefixes + " " + \
            """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            INSERT DATA { """ + \
                "GRAPH <" + graph_name + "> { " + \
                       rdf_statements + \
                " }" + \
            " }"    
    
    return instruction


#%%

def build_sparql_clear_graph_instruction(graph_name, silent=False):
    '''
    Build a Sparql CLEAR graph instruction per the Sparql 1.1 Update
    specification.
    
    The CLEAR can be targetted at either the default graph or at a 
    named graph.
    
    With GraphDB, a CLEAR operation on a named graph empties the graph and 
    removes the empty graph. A CLEAR on the default graph just empties the 
    graph.
    
    With GraphDB, a CLEAR operation on a named graph that does not exist
    succeeds.  The default graph always exists.
    '''
    
    if len(graph_name) == 0:
        raise ValueError('graph name required')
    
    if graph_name == 'default':
        
        # clear the default graph
        instruction = "CLEAR DEFAULT"
            
    else:
    
        # clear a named graph
        instruction = "CLEAR GRAPH <" + graph_name + ">"
    
    return instruction


#%%

def execute_sparql_query(sparql_query_endpoint, sparql_query):
    '''
    Execute a SPARQL query.
     
    The incoming, prepared SPARQL query controls whether the query
    targets the default graph or a named graph.
    
    If the query succeeds, results are returned. If the query fails for
    any reason, an Exception is thrown and caught and returned.
    '''
    
    results = None
    exception_info = None

    try:
        
        sparql = SPARQLWrapper(sparql_query_endpoint)
        
        sparql.setQuery(sparql_query)
        
        sparql.setReturnFormat(JSON)
        
        # execute the query
        results = sparql.query()
        
        results = results.convert() 
                
    except Exception as e:

        exception_info = e 
    
    return results, exception_info


#%% 

def get_graph_size(base_graphdb_url, graph_name):
    '''
    A utility function to get the size of a graph, as measured by the
    number of triples in the graph (explicit and/or inferred).

    The graph can be the default graph or a named graph.
    '''
    
    if graph_name == 'default':
        # set a query to return all triples in the default graph
        query = """
            SELECT * 
            WHERE {
                ?s ?p ?o .
                }
            """ 
    else:
        # set a query to return all triples in the default graph
        query = """
            SELECT * 
            WHERE { """ + \
                "GRAPH <" + graph_name + "> { " + \
                    "?s ?p ?o . " + \
                "} " + \
            "}"
    
    results, exception_info = execute_sparql_query(base_graphdb_url, query)

    outcome = get_sparql_interaction_outcome(results, exception_info)
    
    if outcome == 'success':
        n_triples = len(results["results"]["bindings"])
    else:
        n_triples = None
    
    return outcome, n_triples
    
















