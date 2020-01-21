# (C) 2020 Schlumberger. Refer to LICENSE

'''
    Extra functions needed for matplotlib.pyplot output

    NOTE: 2D signal processing belongs in extra_numpy
          only actual matplotlib output stuff should be in here
          
extra_pyplot
'''
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.patches as patches
import time
import numpy

'''
 Color-scales - in general prefer the python visually uniform color scales
                since there is much less chance of misinterpretation

 The following approximation to the UAVK color scale from Techlog is made
 available here. Use with caution!
'''
def UAVK():
    UAVK_list = ["white","navy","blue","royalblue","deepskyblue","cyan",
                 "palegreen","greenyellow","yellow","orange",
                 "darkorange","orangered","tomato","red","firebrick",
                 "brown"]
    return UAVK_list

'''
 All the python colorscales such as magnma and viridis, plus the UAVK
'''
def supported_colorscales(request):
    cscale=request
    if (request == 'UAVK'):
        cscale = matplotlib.colors.LinearSegmentedColormap.from_list("", UAVK())
    return cscale
    
'''
 add_user_events : labelled time-stamps to annotate plots
'''
def add_user_events(event_list,plt, time_ref):
    # Add user events
    index1 = 0
    for label in event_list:
        event_start = label['event_label_start']
        event_mark  = label['event_mark']
        event_text  = label['event_text']
        
        event_time=time.mktime((time.strptime(event_mark,'%Y-%m-%d %H:%M:%S')))
        event_line = numpy.searchsorted(time_ref, event_time)
        plt.axvline(event_line,color='r')
        plt.text(event_line,event_start,event_text,rotation='vertical',fontsize=7, bbox=dict(facecolor='yellow', alpha=0.85))
        index1 += 1

'''
 add_user_zones : labelled depth extents to annotate plots, for example
                  with perforation cluster locations
'''
def add_user_zones(event_list, plt, depth_ref, line_color):
    # Add user events
    index1 = 0
    for label in event_list:
        start = label['start']
        end   = label['end']
        name  = label['name']
        
        depth_line = numpy.searchsorted(depth_ref, start)
        plt.axhline(depth_line,color=line_color)
        depth_line = numpy.searchsorted(depth_ref, end)
        plt.axhline(depth_line,color=line_color)
        # patches.Rectangle((location), width, height, ...)
        #rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
        plt.text(0,depth_line,name,rotation='horizontal',fontsize=7, bbox=dict(facecolor='yellow', alpha=0.85))

'''
 time_stamps : turn a list of unixtimestamps into a lits of string-based
               timestamps for writing on plots
'''
def time_stamps(time_ref):
    time_list = []
    for seconds in time_ref:
        time_list.append(time.asctime(time.gmtime(seconds)))
    return time_list

'''
 add_fibre_ends : draw lines corresponding to the start and end of the fibre
                  Typically DAS data will have data before the wellhead, and
                  noise corresponding to backscatter that occurred beyond the
                  two-way travel time.
'''
def add_fibre_ends(start_of_fibre,end_of_fibre,plt):
    # Start and end of fibre
    plt.axhline(start_of_fibre,color='w')
    plt.axhline(end_of_fibre,color='w')

