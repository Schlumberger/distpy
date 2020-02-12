# (C) 2020, Schlumberger. Refer to LICENSE.
'''
 unit_test - performs a unit test on every command in pub_command_set

           - generates the command dictionary for the wiki on the github
             repository.
             This consists of 2 markdown pages, one is the links for the sidebar
             and the other is the command dictionary itself.
            
'''
import distpy.calc.pub_command_set as pub_command_set
import distpy.calc.plt_command_set as plt_command_set
import distpy.io_help.io_helpers as io_helpers

import numpy


def unit_test():
    knownList = {}
    knownList = pub_command_set.KnownCommands(knownList)
    # PLOT COMMANDS NOT YET SUPPORTED...
    #knownList = plt_command_set.KnownCommands(knownList)

    # lines in the reference
    lines = []
    # header...
    header = '''This is an alphabetical list of the commands in [pub_command_set.py](https://github.com/Schlumberger/distpy/blob/master/distpy/calc/pub_command_set.py), the example JSON  is
used only for generating the documentation and is not a viable
processing flow.'''
    lines.append(header)
    xaxis = numpy.arange(100)
    data = numpy.random.random((100,100)) + 1
    base_command = pub_command_set.DataLoadCommand(data,{})


    # links
    links = []
    links.append('### [Command Dictionary](https://github.com/Schlumberger/distpy/wiki/Command-Dictionary)')

    # alphabetical sorting
    for k,v in knownList.items():
        if k!='NONE':
            links.append('['+k+'](https://github.com/Schlumberger/distpy/wiki/Command-Dictionary#'+k+')')
            command = knownList[k](base_command,{'name':k, 'prf' : 10000, 'xaxis' : xaxis})
            lines = io_helpers.command2md(command,lines)
            # unit test...
            print(k)
            command.execute()

    with open('sidebar.md','w') as f:
        for line in links:
            f.write(line+'\n')

    with open('reference.md','w') as f:
        for line in lines:
            f.write(line+'\n\n')
        
if __name__ == "__main__":
    unit_test()
