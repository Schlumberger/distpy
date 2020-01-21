# (C) 2020, Schlumberger. Refer to package LICENSE.

'''
 CommandFactory - governs the use of commands in distpy
                  for processing and plotting
# There is a special command at the head-node - where data is loaded
'''
import numpy
import distpy.calc.extra_numpy as extra_numpy
import distpy.calc.pub_command_set as pub_command_set
# - example of extension
# import distpy.calc.my_command_set as my_command_set
import distpy.calc.plt_command_set as plt_command_set


def CommandFactory(commandList, commandJson):
    # currently we are just extending our own list...
    knownList = {}
    # Add your own command sets below. The order indicates
    # precidence. For example my_command_set is overwriting
    # commands  in the public command set.
    knownList = pub_command_set.KnownCommands(knownList)
    # plotting subset
    knownList = plt_command_set.KnownCommands(knownList)
    # - example of extension
    # knownList = my_command_set.KnownCommands(knownList)
    
    name = commandJson.get('name','NONE')
    plot_type = commandJson.get('plot_type','NONE')
    print(name)
    # multiple entries for the previous command
    # to support the n-to-1 paths
    previous = commandJson.get('in_uid',-1)
    prevlist = commandJson.get('gather_uids',[-1])
    prev_stack = []
    for prev in prevlist:
        if prev>=0:
            prev_stack.append(commandList[prev])
    commandJson['commands']=prev_stack
    if previous>=0:
        prev_result = commandList[previous]
        return (knownList[name](prev_result,commandJson))
    return None
