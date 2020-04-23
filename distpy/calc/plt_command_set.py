# (C) 2020, Schlumberger. Refer to LICENSE
'''
 plt_command_set.py

 Collecting all the plot commands in one place

 The BasicCommand can be found in distpy.calc.pub_command_set
 All commands in here must inherit from BasicCommand
 Most commands in here should inherit from BasicPlotCommand, which
     extends BasicCommand in a way that makes matplotlib plots easy
     to generate with consistent annotation
'''
import numpy
import os
import matplotlib.pyplot as plt
from numba import vectorize, float32

import distpy.calc.extra_numpy as extra_numpy
import distpy.calc.extra_pyplot as extra_pyplot
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.directory_services as directory_services
from distpy.calc.pub_command_set import BasicCommand
from distpy.calc.pub_command_set import universal_arglist

'''
 ThumbnailCommand : turn the current 2D matrix into an image, which can be aliased.
                    This is commonly called in the middle of a signal processing flow to
                    generate interim result plots.
'''
class ThumbnailCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._dirname = jsonArgs.get('directory_out','NONE')
        self._fname = jsonArgs.get('date_dir','NONE')
        if "xaxis" in jsonArgs:
            self._xaxis = [jsonArgs['xaxis'][0],jsonArgs['xaxis'][-1]]
            self._taxis = [0,jsonArgs['nt']/jsonArgs['prf']]
        else:
            self._xaxis = [0,1]
            self._taxis = [0,1]
        if "taxis" in jsonArgs:
            self._taxis = [jsonArgs['taxis'][0],jsonArgs['taxis'][-1]]
        self._format = jsonArgs.get('format','png')
        self._std_val_mult = jsonArgs.get('clip_level',1.0)
        self._jsonArgs = jsonArgs
        
            

    def docs(self):
        docs={}
        docs['one_liner']="Create a thumbnail image of the current 2D processed data in the specified format."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['directory_out','format','clip_level'] }
        return docs

    def postcond(self):
        return [directory_services.path_join(self._dirname,self._fname)]

    def execute(self):
        super().execute()
        dirname = self._dirname
        if not dirname=='NONE':
            io_helpers.thumbnail_plot(dirname,self._fname,self._previous.result(),
                                      xscale=self._xaxis,tscale=self._taxis,
                                      plt_format=self._format, std_val_mult=self._std_val_mult, jsonArgs=self._jsonArgs)

'''
 BasicPlotCommand - in particular extends BasicCommand with annotations() and write_plot()
                    to make it easy to generate consistent plot annotations (e.g. depth and time ranges overlain on the pictures)
'''
class BasicPlotCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._indir = jsonArgs.get('directory_in','NONE')
        self._dirname = jsonArgs.get('directory_out','NONE')
        self._out_plot = jsonArgs.get('out_plot','NONE')

        # For all plots...
        self._depth_display_unit = jsonArgs['depth_display_unit']
        self._start_of_fibre = jsonArgs['start_of_fibre']
        self._end_of_fibre = jsonArgs['end_of_fibre']
        self._figure_size = jsonArgs['figure_size']
        self._dots_per_inch = jsonArgs['dpi']
        self._event_list = jsonArgs['label_list']
        ## clusters and stages - file location
        self._well_segs = jsonArgs['well_segments']
        self._depth_ref = jsonArgs['depth_ref']
        self._time_ref = jsonArgs['time_ref']
        self._nt = self._time_ref.shape[0]
        self._nx = self._depth_ref.shape[0]

        # For a specific plot...
        self._plot_list = jsonArgs['plot_list']
        self._plot_type = jsonArgs['plot_type']
        self._out_plot = jsonArgs['out_plot']

    def annotations(self, plt, title_text, t_interval):
        # how are plots and figures passed around?
        # here we are assuming that plt = matplotlib.pyplot can handle it
        # by assumption...not a good strategy
        if (t_interval<1):
            t_interval=1
        time_list = extra_pyplot.time_stamps(self._time_ref)
        nt = self._nt
        plt.title(title_text, fontsize=10)
        plt.xticks(numpy.arange(0, (nt-t_interval), t_interval), time_list[0:nt:t_interval],rotation='vertical',fontsize=7)      ## plot x ticks and labels
        plt.xlim(0,nt)
        plt.gca().axes.invert_xaxis()
        plt.axvline(0,color='k',linewidth=5)
        plt.ylim(numpy.max(self._depth_ref),numpy.min(self._depth_ref))

        extra_pyplot.add_user_events(self._event_list, plt, self._time_ref)
        extra_pyplot.add_fibre_ends(self._start_of_fibre, self._end_of_fibre, plt)
        extra_pyplot.add_user_zones(self._well_segs, plt, self._depth_ref, 'r')

    def write_plot(self,plt):
        super().execute()
        dirname = self._dirname
        if not dirname=='NONE':
            io_helpers.write_plot(dirname,self._out_plot, plt)

    def postcond(self):
        return [directory_services.path_join(self._dirname,self._out_plot)]

    def execute(self):
        super.execute()

'''
 WellLogPlotCommand : plot the data as a well-log, the basic "waterfall"
'''
class WellLogPlotCommand(BasicPlotCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)

    def execute(self):
        plot_list = self._plot_list
        nt = self._nt
        depth_ref = self._depth_ref
        n_bands = len(plot_list)
        # Equal plot widths for multiple panels (e.g. all the bands on one plot)
        h_len = 0.966/(n_bands+2) 
        for a in range(n_bands):
            local_plot_list = plot_list[a]
            fname = os.path.join(self._indir,local_plot_list['data'])
            traces = numpy.load(fname)
            
            title_text = local_plot_list['title_text']
            cscale = extra_pyplot.supported_colorscales(local_plot_list.get('colormap','magma'))
            
            vscale     = numpy.max(extra_numpy.despike(traces,3))               ##despike and get max value for scaling 2 *sdev
            vscalemax = vscale*10
            vscalemin=0
            t_interval = int(nt/8)
            # data is read, so now create the plot
            # PLOT...
            # NOTE: fontsize is hard coded!!! 
            plt.axes([0.15 + (a * h_len), 0.1, h_len, .8])
            plt.imshow(traces, extent=[nt,0, numpy.max(depth_ref), numpy.min(depth_ref)], cmap=cscale, aspect='auto', vmin=vscalemin,vmax=vscalemax)
            # standard finishing...
            self.annotations(plt, title_text, t_interval)
        self.write_plot(plt)

'''
  StackPlotCommand : multiple datasets (e.g. FBE bands) overlain at different transparency levels - which is a useful false color plotting technique
'''
class StackPlotCommand(BasicPlotCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)

    def execute(self):
        plot_list = self._plot_list
        nt = self._nt
        depth_ref = self._depth_ref

        n_bands = len(plot_list)
        # Equal plot widths for multiple panels (e.g. all the bands on one plot)
        title_text = 'Stack of '
        t_interval = int(nt/8)
        for a in range(n_bands):
            local_plot_list = plot_list[a]
            fname = os.path.join(self._indir,local_plot_list['data'])
            traces = numpy.load(fname)
            
            title_text = title_text + local_plot_list['title_text'] + ' '
            cscale = local_plot_list['colormap']
            alpha_val = float(local_plot_list['alpha'])
                        
            vscale     = numpy.max(extra_numpy.despike(traces,3))               ##despike and get max value for scaling 2 *sdev
            vscalemax = vscale*10
            vscalemin=0
            # data is read, so now create the plot
            # PLOT...
            plt.imshow(traces, extent=[nt,0, numpy.max(depth_ref), numpy.min(depth_ref)], cmap=cscale, alpha=alpha_val, aspect='auto', vmin=vscalemin,vmax=vscalemax)
        self.annotations(plt, title_text, t_interval)
        self.write_plot(plt)

'''
 RGBPlotCommand : plot 3 results (e.g. FBE bands) as the R, G and B intensities of an image, another false-color plotting technique
'''
class RGBPlotCommand(BasicPlotCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)

    def execute(self):
        plot_list = self._plot_list
        nt = self._nt
        nx = self._nx
        depth_ref = self._depth_ref
        t_interval = int(nt/8)
        red = plot_list[0]
        green = plot_list[1]
        blue = plot_list[2]
        title_text = 'R = ' + red['title_text'] + ', G = ' + green['title_text'] + ', B = ' + blue['title_text']
        
        img = numpy.zeros((nx, nt, 3), dtype=float)
        fname = os.path.join(self._indir,red['data'])
        img[:,:,0] = numpy.load(fname)
        fname = os.path.join(self._indir,green['data'])
        img[:,:,1] = numpy.load(fname)
        fname = os.path.join(self._indir,blue['data'])
        img[:,:,2] = numpy.load(fname)
        
        img[:,:,0]   = extra_numpy.intensity(img[:,:,0]) 
        img[:,:,1]   = extra_numpy.intensity(img[:,:,1])
        img[:,:,2]   = extra_numpy.intensity(img[:,:,2])
        
        # support inverted (so that (0,0,0) maps to white instead of black)
        # this is an optional argument so we are using the get() method instead of []
        if (red.get('inverted','no')=='yes'):
            img[:,:,0] = 1 - img[:,:,0]
        if (green.get('inverted','no')=='yes'):
            img[:,:,1] = 1 - img[:,:,1]
        if (blue.get('inverted','no')=='yes'):
            img[:,:,2] = 1 - img[:,:,2]
        
        img = img*255
                    
        plt.imshow((img.astype(numpy.uint8)), extent=[nt,0, numpy.max(depth_ref), numpy.min(depth_ref)], aspect='auto')
        self.annotations(plt,title_text, t_interval)
        self.write_plot(plt)


'''
  KnownCommands : the list of commands in this file. If a command is not listed here, it means that it is not accessible via distpy
'''
def KnownCommands(knownList):
    knownList['thumbnail']      = ThumbnailCommand
    knownList['well_log']       = WellLogPlotCommand
    knownList['stack']          = StackPlotCommand
    knownList['rgb']            = RGBPlotCommand
    return knownList
