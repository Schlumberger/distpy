# (C) 2020, Schlumberger. Refer to LICENSE

import numpy
import matplotlib.pyplot as plt
import datetime
import scipy.signal
import os
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.directory_services as directory_services
import distpy.calc.extra_numpy as extra_numpy
import distpy.calc.extra_pyplot as extra_pyplot
import distpy.calc.unit_handler as unit_handler
import distpy.calc.pub_command_set as pub_command_set
import distpy.calc.processing_commands as processing_commands

'''
 read_zones : read a CSV file containing measured_depth zone information
              for plot annotation
'''
def read_zones(csvFile):
    lines = directory_services.csv_reader(csvFile)
    #conversion = 1.0
    #if (lines[1][0]=="ft"):
    #    conversion = FT_TO_M
    zones=[]
    for a in range(2, len(lines),2):
        one_zone={}
        tokens = lines[a]
        one_zone['start'] = float(tokens[0])
        one_zone['name'] = tokens[-1]
        tokens = lines[a+1]
        one_zone['end'] = float(tokens[0])
        zones.append(one_zone)
    print(len(zones))
    return zones

'''
 plotgenerator : generates a command_list which is then executed to generate the
                 plots. 
'''
def plotgenerator(dirin, dirout, plotData):
    
    # Configure the hardware
    boxsize = plotData.get('BOXSIZE', 500)
    extra_numpy.set_boxsize(boxsize)

    depth_display_unit = plotData['depth_display_unit']
    start_of_fibre = plotData['start_of_fibre']
    end_of_fibre = plotData['end_of_fibre']
    
    figure_size = plotData['figure_size']
    dots_per_inch = plotData['dpi']
    event_list = plotData['label_list']
    
    ## clusters and stages
    segs_blob = plotData['well_segments']
    
    # blob locations
    TIME_REF_BLOB = directory_services.path_join(dirin,plotData['time_reference'])
    DEPTH_REF_BLOB = directory_services.path_join(dirin,plotData['depth_reference'])
    

    time_ref = directory_services.load(TIME_REF_BLOB)
    time_list = extra_pyplot.time_stamps(time_ref)
    nt = time_ref.shape[0]
    
    depth_ref = directory_services.load(DEPTH_REF_BLOB)
    if (depth_display_unit=="ft"):
        depth_ref = unit_handler.M_TO_FT(depth_ref)
    nx = depth_ref.shape[0]

    # well segmentation for flow allocations
    well_segs = read_zones(segs_blob)

    # same command factory as for strain-rate processing - giving maths functions...
    #dir_suffix = plotData.get('directory_out',dirout)
    #if not dir_suffix=='NONE':
    #   dirval = os.path.join(dirout,dir_suffix)
    dirval = dirout
    directory_services.makedir(dirval)
    
    plt.switch_backend('Agg')
    command_list = []
    # small 2D array for a command zero
    data=numpy.zeros((10,10),dtype=numpy.double)
    command_list.append(pub_command_set.DataLoadCommand(data,{})) 

    for plot in plotData['plots']:
        # Locally package the global information for the generation of this particular plot
        plot['nx']=nx
        plot['nt']=nt
        plot['label_list']= plotData['label_list']
        plot['time_ref']=time_ref
        plot['depth_ref']=depth_ref
        plot['directory_in']=dirin
        
        plot['depth_display_unit'] = plotData['depth_display_unit']
        plot['start_of_fibre'] = plotData['start_of_fibre']
        plot['end_of_fibre'] = plotData['end_of_fibre']
        plot['figure_size'] = plotData['figure_size']
        plot['dpi'] = plotData['dpi']
        plot['well_segments'] = well_segs
        # internal mapping to the commands
        # 1. commands have names
        # 2. in_uid from a previous command (here always command zero)
        plot['name'] = plot['plot_type']
        plot['in_uid']=0

        plot['directory_out']=dirval
        command_list.append(processing_commands.CommandFactory(command_list,plot))


    # Actual plot generation occurs here...
    for command in command_list:
        print(command)
        command.execute()
