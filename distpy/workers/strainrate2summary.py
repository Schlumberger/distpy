# (C) 2020, Schlumberger. Refer to LICENSE

import numpy
import datetime
import scipy.signal
import os
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.directory_services as directory_services
import distpy.calc.pub_command_set as pub_command_set

import distpy.calc.extra_numpy as extra_numpy
import distpy.calc.processing_commands as processing_commands

'''
 strainrate2summary : process one file of strainrate data using the provided
                      command tree in commandJSON.

                      This executes in one thread.
'''
def strainrate2summary(filename, xaxis, prf, dirout, commandJson, extended_list):
    # try to make a datestamp that WITSML could use...
    tokens = filename.split(os.sep)
    # the final token without its .npy is the unix timestamp
    unixtime = int(tokens[-1][:-4])
    datestring = datetime.datetime.utcfromtimestamp(unixtime).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    datedir = str(unixtime)

    ltxJson = []
    ltxJson = io_helpers.latexJson({"command_list" : commandJson['command_list']},ltxJson)

    # Configure the hardware
    boxsize = commandJson.get('BOXSIZE', 500)
    extra_numpy.set_boxsize(boxsize)
    data = numpy.load(filename)
    #print(data.shape)
    nx = data.shape[0]
    nt = data.shape[1]

    # Is this a request to document a workflow?
    isDocs = commandJson.get('document',0)

    # initialize the command list that will eventually be executed.
    # This first one is anomalous - need a good way to initiate the top of the tree...
    # possibly this is a root...
    command_list=[]
    # { "name" : "data",           "uid" :  0},
    command_list.append(pub_command_set.DataLoadCommand(data,{}))

    
    # command_list is a list of dictionaries
    for command in commandJson['command_list']:
        # we add the basics of nx and prf, which belong to the data
        dir_suffix = command.get('directory_out','NONE')
        if not dir_suffix=='NONE':
            dirval = os.path.join(dirout,dir_suffix)
            if not os.path.exists(dirval):
                os.makedirs(dirval)
            command['directory_out']=dirval
        dir_suffix = command.get('directory_in','NONE')
        if not dir_suffix=='NONE':
            dirval = os.path.join(dirout,dir_suffix)
            if not os.path.exists(dirval):
                os.makedirs(dirval)
            command['directory_in']=dirval
        command['date_dir']=datedir
        command['datestring']=datestring
        command['nx']=nx
        command['prf']=prf
        command['xaxis']=xaxis
        command['nt']=nt
        # prf/nt is the rescale factor
        command['f_rescale']=float(nt)/float(prf)
        index = command.get('band00',-1)
        if index>=0:
            command['command']=command_list[index]
        #s(len(command_list))
        command_list.append(processing_commands.CommandFactory(command_list,command,extended_list=extended_list))

    # if documenting - that is one path...execution is the other
    if isDocs==1:
        lines =[]
        lines = io_helpers.latexTop(lines)
        lines = io_helpers.command2latex(command_list,lines, commandJson.get('name','Command Set'), commandJson.get('description','Commands used'))
        for line in ltxJson:
            lines.append(line)
        lines = io_helpers.latexPng(lines)
        lines = io_helpers.latexTail(lines)
        with open(os.path.join(dirout,'config.tex'),'w') as f:
            for line in lines:
                f.write(line+'\n')
                
        # dot graphviz graph
        lines = io_helpers.dot_graph(commandJson['command_list'])
        with open(os.path.join(dirout,'config.gv'),'w') as f:
            for line in lines:
                f.write(line+'\n')
        
    else:
        # if postconditions are all already met, we can skip this workflow...
        post_cond = True
        for command in command_list:
            for filename in command.postcond():
                if not directory_services.exists(filename):
                    post_cond = False

        if post_cond==True:
            print(filename, ' post-conditions are all satisfied, skipping processing.')
        else:
            for command in command_list:
                command.execute()
