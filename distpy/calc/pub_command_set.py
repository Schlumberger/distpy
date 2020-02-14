'''
pub_command_set.py

This module contains the list of public commands. These will generally wrapper some
numpy or extra_numpy function.

(C) 2020, Schlumberger. Refer to LICENSE
'''
import numpy
import os
from scipy import signal
from scipy import ndimage
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter, freqz
from scipy.stats import kurtosis, hmean, gmean, skew, moment
import distpy.calc.extra_numpy as extra_numpy
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.directory_services as directory_services


'''
  The universal_arglist is used in automatic documentation generation.
  By holding a universal_arglist we are gently encouraging re-use where
  possible and gently avoiding the same argument keyword having very
  different meanings in different contexts.
'''
NONE = "NONE"
DEFAULT = "default"
DESC = "description"
universal_arglist = {
    "args" :          {DEFAULT : None, DESC : "Arguments a passed through to the underlying math library"},
    "xaxis" :         {DEFAULT : None, DESC : "A numpy vector of distances along the fibre"},
    "taxis" :         {DEFAULT : None, DESC : "A numpy vector of times associated with columns of data"},
    "directory_in" :  {DEFAULT : NONE, DESC : "The subdirectory from which to read data"},
    "directory_out" : {DEFAULT : NONE, DESC : "The subdirectory where results will be written"},
    "command_list" :  {                DESC : "A list of sub-commands collected in a single macro"},
    "train" :         {                DESC : "A dictionary containing training parameters for kera, e.g. {'epochs':150,'batch_size':10"},
    "func" :          {                DESC : "Either rms_from_fft or te_from_fft"}, 
    "labels" :        {                DESC : "A list of column headers"},
    "order" :         {DEFAULT : 5,    DESC : "The order for a filter calculation such as the Butterworth filter"},
    "type" :          {DEFAULT : "lowpass",DESC : "The type of a filter which can be lowpass, highpass, bandpass, or bandstop"},
    "padtype" :       {DEFAULT : "even",DESC : "The type of end-effect control on a filter, see scipy.signal.filtfilt"},
    "prf" :           {DEFAULT : 10000, DESC : "The pulse repetition frequency in Hz (one over the time sample rate)"},
    "freq" :          {DEFAULT : 200.0, DESC : "A frequency in Hz"},
    "filename" :      {DEFAULT : NONE,  DESC : "A filename for read or write operations"},
    "axis" :          {DEFAULT : -1,    DESC : "The axis to apply an operation to, typically in distpy axis=0 for depth and axis=1 for time"},
    "xsigma" :        {DEFAULT : 1.0,   DESC : "A standard deviation in the x-direction"},
    "tsigma" :        {DEFAULT : 1.0,   DESC : "A standard deviation in the time direction"},
    "xorder" :        {DEFAULT : 5,     DESC : "In a 2D filter this is the order in the x-direction"},
    "torder" :        {DEFAULT : 5,     DESC : "In a 2D filter, this is the order in the t-direction"},
    "distance" :      {DEFAULT : 3,     DESC : "The number of samples in a median filter"},
    "window_length" : {DEFAULT : 5,     DESC : "The length of a running mean window"},
    "offset" :        {DEFAULT : 0,     DESC : "An offset from the start of the data, in samples"},
    "xmin" :          {DEFAULT : None,  DESC : "A minimum value on the x-axis"},
    "xmax" :          {DEFAULT : None,  DESC : "A maximum value on the x-axis"},
    "tmin" :          {DEFAULT : None,  DESC : "A minimum value on the time-axis"},
    "tmax" :          {DEFAULT : None,  DESC : "A maximum value on the time-axis"},
    "commands" :      {DEFAULT : [None],DESC : "TODO"},
    "data_style" :    {DEFAULT : NONE,  DESC : "A string identifier for the data inside the WITSML file"},
    "method" :        {DEFAULT : "lin_fit", DESC : "The method for curve fitting, see scipy.optimize.curve_fit"},
    "moment" :        {DEFAULT : 1,     DESC : "The order of central moment, see scipy.stats.moment"},
    "max_velocity":   {DEFAULT : 1600,  DESC : "The maximum phase velocity"},
    "min_velocity":   {DEFAULT : 1400,  DESC : "The minimum phase velocity"},
    "smooth" :        {DEFAULT : 0, DESC : "The smoothing factor for the filter"},
    "bounds" :        {DEFAULT : ([-5,0],[0,400]), DESC : "The bounds on curve fitting, see scipy.optimize.curve_fit"},
    "initSlopes" :    {DEFAULT : [-0.5], DESC : "An initial slope estimate for the curve fitting, see scipy.optimize.curve_fit"},
    "xsample" :       {DEFAULT : 1,     DESC : "The level of downsampling in the x-directioin"},
    "tsample" :       {DEFAULT : 1,     DESC : "The level of downsampling in the time-direction"},
    "sta":            {DEFAULT : 50,    DESC : "The short-term average window-length in samples"},
    "lta":            {DEFAULT : 200,   DESC : "The long-term average window-length in samples"},
    "xdir" :          {DEFAULT : 5,     DESC : "Wiener filter length in the x-direction"},
    "tdir" :          {DEFAULT : 5,     DESC : "Wiener filter length in the t-direction"},
    "low_freq" :      {DEFAULT : None,  DESC : "A list of low frequency values for band-pass windows in Hz"},
    "high_freq" :     {DEFAULT : None,  DESC : "A list of high frequency values for band-pass windows in Hz"},
    "noisePower" :    {DEFAULT : None,  DESC : "The noise power in the Wiener filter"},      
    "format" :        {DEFAULT : "png", DESC : "The format of the picture output"},
    "clip_level" :    {DEFAULT : 1.0,   DESC : "The number of standard devaitions about the mean for plotting thumbnails"}
    }

    

'''
 DataLoadCommand : the special command, which is number 0 in any workflow
                   requires that some sort of file (in our case a
                   {timestamp}.npy) exists and is loaded so that its data
                   can be processed
'''
class DataLoadCommand(object):
    def __init__(self, data, jsonArgs):
        self._data = data
        self._name = 'load_data'

    def get_name(self):
        return self._name

    # allow for a post condiction that is a
    # list. We use this to return the names of any files
    # that would be written - so that we can test whether or not
    # ateps can be skipped in the RESTART case.
    def postcond(self):
        return []

    def docs(self):
        return {}

    def isGPU(self):
        return False
    
    def execute(self):
        pass

    def result(self):
        return self._data


'''
 All commands that are not the DataLoadCommand must
 inherit from the BasicCommand.

 Override postcond() if you want Restarts to work
 Override docs() if you want automatic self-documentation to work
 Override execute() if you want to do any calculations on the data
'''
# A command that does nothing...
class BasicCommand(object):
    def __init__(self, command, jsonArgs):
        self._previous = command
        self._name = jsonArgs['name']
        self._result = self._previous.result()
        
    def get_name(self):
        return self._name

    # allow for a post condiction that is a
    # list. We use this to return the names of any files
    # that would be written - so that we can test whether or not
    # ateps can be skipped in the RESTART case.
    def postcond(self):
        return []

    def docs(self):
        return {}

    def isGPU(self):
        return False
        
    def execute(self):
        self._result = self._previous.result()
        
        
    def result(self):
        return self._result


'''
 ToGPUCommand : Go to GPU
'''
class ToGPUCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs

    def docs(self):
        docs={}
        docs['one_liner']="Put the current result on the GPU and perform subsequent steps there. If no GPU is available, this has no effect."
        return docs

    def isGPU(self):
        return True
    
    def execute(self):
        self._result = extra_numpy.to_gpu(self._previous.result())

'''
 FromGPUCommand : Return from GPU
'''
class FromGPUCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs

    def docs(self):
        docs={}
        docs['one_liner']="Put the current result on the CPU and perform subsequent steps there. If no GPU is available, this has no effect."
        return docs

    def isGPU(self):
        return True
    
    def execute(self):
        self._result = extra_numpy.from_gpu(self._previous.result())



'''
  MacroCommand : This implements macros - providing a lightweight system for
                 more complex commands - see extra_numpy.macro_factory for the
                 list of available commands.
'''
class MacroCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._macro = jsonArgs

    def docs(self):
        docs={}
        docs['one_liner']="Create a macro containing sub-commands"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['command_list'] }
        return docs

    def execute(self):
        traces = self._previous.result()
        # NOTE: the unit-test order matters. 'percentage_on' or 'sum' will result in a single trace, no good for 2D filters...
        command_list = self._macro.get('command_list',[{'name':'abs'}, {'name':'normalize'},
                                                       {'name':'gaussian_filter'},{'name':'gradient'},
                                                       {'name':'threshold'},{'name':'median_filter'},
                                                       {'name':'percentage_on'},{'name':'sum'}])
        for commands in command_list:
            #print(commands,traces.shape)
            traces = extra_numpy.macro_factory(commands, traces)
        self._result = traces

'''
 AbsCommand : wrappers the numpy.abs() function
'''
class AbsCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)

    def docs(self):
        docs={}
        docs['one_liner']="Take the absolute value of the input"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_abs(self._previous.result())
'''
 KerasCommand : wrappers the extra_numpy.kera_model() function
'''
class KerasCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs
        self._prevstack = jsonArgs.get('commands',[None])

    def docs(self):
        docs={}
        docs['one_liner']="Take the kurtosis of the input using scipy.stats.kurtosis(). Use k statistics to eliminate bias and omit any NaNs."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['filename','train'] }
        return docs

    def execute(self):
        train = self._args.get('train',None)
        filename = self._args.get('filename',None)
        Y = None
        if self._prevstack[0] is not None:
            Y = self._prevstack[0].result()
        self._result = extra_numpy.keras_model(self._previous.result(),filename,Y=Y,train=train)



'''
 KurtosisCommand : wrappers the scipy.stats.kurtosis() function
'''
class KurtosisCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Take the kurtosis of the input using scipy.stats.kurtosis(). Use k statistics to eliminate bias and omit any NaNs."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def execute(self):
        self._result = stats.kurtosis(self._previous.result(),axis=self._axis, bias=False, nan_policy='omit')

'''
 SkewnessCommand : wrappers the scipy.stats.skew() function
'''
class SkewnessCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Take the skewness of the input using scipy.stats.skewn(). Eliminate bias and omit any NaNs."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def execute(self):
        self._result = stats.skew(self._previous.result(),axis=self._axis, bias=False, nan_policy='omit')

'''
 MomentCommand : wrappers the scipy.stats.moment() function
'''
class MomentCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)
        self._moment = jsonArgs.get('moment',1)

    def docs(self):
        docs={}
        docs['one_liner']="Take the nth moment of the input using scipy.stats.moment(). Omit any NaNs."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis','moment'] }
        return docs

    def execute(self):
        self._result = stats.moment(self._previous.result(),moment=self._moment, axis=self._axis, nan_policy='omit')
'''
 MeanCommand : wrappers the numpy.mean() function
'''
class MeanCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Take the mean of the input using numpy.mean()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_mean(self._previous.result(),axis=self._axis, keepdims=True)
'''
 StdCommand : wrappers the numpy.std() function
'''
class StdDevCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Take the standard deviation of the input using numpy.std()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = numpy.std(self._previous.result(),axis=self._axis, keepdims=True)

'''
 HarmonicMeanCommand : wrappers the scipy.stats.hmean() function
'''
class HarmonicMeanCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Take the harmonic mean of the input using scipy.stats.hmean()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def execute(self):
        self._result = stats.hmean(self._previous.result(),axis=self._axis)
'''
 GeometricMeanCommand : wrappers the scipy.stats.gmean() function
'''
class GeometricMeanCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Take the geometric mean of the input using scipy.stats.gmean()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def execute(self):
        self._result = stats.gmean(self._previous.result(),axis=self._axis)
'''
 RealCommand : wrappers the numpy.real() function
'''
class RealCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)

    def docs(self):
        docs={}
        docs['one_liner']="Take the real value of the input"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_real(self._previous.result())

'''
 HashCommand : creates a 64-bit integer hash using extra_numpy.hash(), reducing the data from 2D to 1D
'''
class HashCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._ioffset = int(jsonArgs.get('offset',0))

    def docs(self):
        docs={}
        docs['one_liner']="Creates a 64-bit integer hash at each depth using extra_numpy.hash(), reducing the data from 2D to 1D"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['offset'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.hash(self._previous.result(),ioffset=self._ioffset)
'''
 ButterCommand : wrappers the signal.butter() function and applies it
                 using signal.filtfilt()
'''
class ButterCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._order = jsonArgs.get('order',5)
        self._btype = jsonArgs.get('type','lowpass')
        self._ptype = jsonArgs.get('padtype','even')
        self._prf = jsonArgs.get('prf',10000)
        self._freq = jsonArgs.get('freq',200.0)

    def docs(self):
        docs={}
        docs['one_liner']="Setup a Butterworth filter using scipy.signal.butter() and apply it using scipy.signal.filtfilt()"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['order','type','padtype','prf','freq'] }
        return docs

    def execute(self):
        Wn = self._freq/(self._prf*0.5)
        b, a = signal.butter(self._order,Wn,self._btype,output='ba')
        self._result = signal.filtfilt(b,a,self._previous.result(),axis=1, padtype=self._ptype)

'''
 ArgmaxCommand : Used to extract the first index of the maximum value in each
                 row (or column). This is a first order way to extract a
                 profile versus depth from a processed DTS gradient
'''
class ArgmaxCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',0)

    def docs(self):
        docs={}
        docs['one_liner']="Index of the maximum value in each row or column."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_argmax(self._previous.result(), axis=self._axis)

'''
 GaussianCommand : Applies a 2D Gaussian blurring filter using signal.ndarray.gaussian_filter()
'''
class GaussianCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._xdir = jsonArgs.get('xsigma',1.0)
        self._tdir = jsonArgs.get('tsigma',1.0)
        self._xorder = jsonArgs.get('xorder',5)
        self._torder = jsonArgs.get('torder',5)

    def docs(self):
        docs={}
        docs['one_liner']="Applies a 2D Gaussian blurring filter using signal.ndarray.gaussian_filter()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xsigma','tsigma','xorder','torder'] }
        return docs

    def execute(self):
        self._result = ndimage.gaussian_filter(self._previous.result(), [self._xdir, self._tdir], order=[self._xorder, self._torder],mode='constant')

'''
  MedianFilterCommand : applies a 2D square median filter using ndimage.median_filter()
'''
class MedianFilterCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._distance = jsonArgs.get('distance',3)

    def docs(self):
        docs={}
        docs['one_liner']="Applies a 2D square median filter using ndimage.median_filter()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['distance'] }
        return docs

    def execute(self):
        self._result = ndimage.median_filter(self._previous.result(), self._distance)


'''
 SumCommand : Sums data along the specified axis, reducing from 2D to 1D using numpy.sum()
'''
class SumCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Sums the data along the specified axis, reducing from 2D to 1D using numpy.sum()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        docs['args']['axis']['default']=1
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_sum(self._previous.result(), axis=self._axis, keepdims=True)


'''
 RunningMeanCommand : A 1D running mean averaging filter, using the extra_numpy.running_mean()
'''
class RunningMeanCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        # previously was 'length'
        self._N = jsonArgs.get('window_length',5)
        self._axis = jsonArgs.get('axis',0)

    def docs(self):
        docs={}
        docs['one_liner']="A 1D running mean averaging filter, using the extra_numpy.running_mean()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['window_length','axis'] }
        docs['args']['axis']['default']=0
        return docs

    def execute(self):
        self._result = extra_numpy.running_mean(self._previous.result(), self._N, axis=self._axis)


'''
 InterpCommand : interpolation for downsampling using extra_numpy.interp()
'''
class InterpCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._xp = jsonArgs.get('xaxis', None)
        self._fp = jsonArgs.get('fp', None)
        self._axis = jsonArgs.get('axis',0)

    def docs(self):
        docs={}
        docs['one_liner']="Interplation for downsampling."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xaxis','fp','axis'] }
        docs['args']['axis']['default']=0
        return docs

    def execute(self):
        self._result = extra_numpy.interp(self._previous.result(), self._xp, self._fp, axis=self._axis)

'''
 CurveFitCommand : Flexible curve fitting using scipy.optimize.curvefit()
'''
class CurveFitCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._func = jsonArgs.get('method','lin_fit')
        self._bound = jsonArgs.get('bounds',([-5,0],[0,400]))
        self._initslope = jsonArgs.get('initSlopes',[-0.5])
        self._xaxis = jsonArgs.get('xaxis',None)

    def docs(self):
        docs={}
        docs['one_liner']="Curve fitting using scipy.optimize.curvefit()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['method','bounds','initSlopes','xaxis'] }
        return docs

    def execute(self):
        fit_func, coeffs0 = extra_numpy.select_fit(self._func, self._previous.result(),self._initslope)
        coeffs,pcov = curve_fit(fit_func,self._previous.result(),self._xaxis,p0=coeffs0,
                                      bounds=self._bounds)
        self._result = fit_func(self._previous.result(), coeffs)
    
'''
 ClipCommand : clip the current matrix so that further operations operate on a smaller window
'''
class ClipCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._xaxis = jsonArgs.get('xaxis',None)
        self._taxis = jsonArgs.get('taxis',None)            
        self._xmin = jsonArgs.get('xmin',numpy.min(self._xaxis))
        self._xmax = jsonArgs.get('xmax',numpy.max(self._xaxis))
        self._tmin = jsonArgs.get('tmin',numpy.min(self._taxis))
        self._tmax = jsonArgs.get('tmax',numpy.max(self._taxis))

    def docs(self):
        docs={}
        docs['one_liner']="Clip the data so that all subsequent operations operate on a small window."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xaxis','taxis','xmin','xmax','tmin','tmax'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        ixmin = 0
        ixmax = 0
        itmin = 0
        itmax = 0
        for x in self._xaxis:
            if self._xmin>x:
                ixmin+=1
            if self._xmax>x:
                ixmax+=1
        if self._taxis is not None:
            for t in self._taxis:
                if self._tmin>t:
                    itmin+=1
                if self._tmax>t:
                    itmax+=1
        else:
            itmin=0
            itmax=self._previous.result().shape[1]
        self._result = (self._previous.result())[ixmin:ixmax,itmin:itmax]

'''
 DiffCommand : numerical differencing with a window_length offset
'''
class DiffCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',-1)
        # previously was 'n'
        self._n = jsonArgs.get('window_length',1)

    def docs(self):
        docs={}
        docs['one_liner']="Simple differencing using the window_length as the offset."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['window_length','axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_diff(self._previous.result(),n=self._n, axis=self._axis)

'''
 DownsampleCommand : index-based downsampling. Note there is no filtering, so for anti-alias
                     downsampling apply an appropriate low-pass filter before calling this.
'''
class DownsampleCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._xdir = jsonArgs.get('xsample',1)
        self._tdir = jsonArgs.get('tsample',1)

    def docs(self):
        docs={}
        docs['one_liner']="Downsampling to reduce the data size."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xsample','tsample'] }
        return docs

    def isGPU(self):
        return True


    def execute(self):
        self._result = (self._previous.result())[::self._xdir,::self._tdir]

'''
 Peak2PeakCommand : The maximum peak-to-peak difference with the maximum and minimum separated
                    by less than the defined window_length. This reduces the data from 2D (x,t)
                    to a trace (x) using extra_numpy.peak_to_peak()
'''
class Peak2PeakCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._window_length = jsonArgs.get('window_length',200)

    def docs(self):
        docs={}
        docs['one_liner']="The maximum peak-to-peak difference with the maximum and minimum separated by less than the defined window_length. This reduces the data from 2D (x,t) to a trace (x)."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['window_length'] }
        return docs


    def execute(self):
        self._result = extra_numpy.peak_to_peak(self._previous.result(), self._window_length)

'''
 CountPeaksCommand : Counts peaks using extra_numpy.count_peaks(), which uses scipy.signal.find_peaks_cwt()
'''
class CountPeaksCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._sta = jsonArgs.get('sta',50)
        self._lta = jsonArgs.get('lta',200)

    def docs(self):
        docs={}
        docs['one_liner']="Couting peaks in a signal using scipy.signal.find_peaks_cwt()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['sta','lta'] }
        return docs


    def execute(self):
        self._result = extra_numpy.count_peaks(self._previous.result(), self._sta, self._lta)


'''
 MultiplyCommand : elementwise multiply
'''
class MultiplyCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])

    def docs(self):
        docs={}
        docs['one_liner']="Elementwise multiply, the output data-type will be the same as thaat of the data entering in the in_uid. This data is multiplied by data provided in the gather_uids"
        #docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['window_length','axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        if self._prevstack[0]!=None:
            self._result = self._previous.result()*((self._prevstack[0]).result())

'''
 GatherCommand : Gather results together
'''
class GatherCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])

    def docs(self):
        docs={}
        docs['one_liner']="Gathers the data with all the data provided in the gather_uids to make one big matrix"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        if self._prevstack[0]!=None:
            datalist = []
            for dataset in self._prevstack:
                datalist.append(dataset.result())
            self._result = extra_numpy.gather(self._previous.result(),datalist)

'''
 StaLtaCommand : short-term average to long-term-average onset-picker transform
'''
class StaLtaCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._sta = jsonArgs.get('sta',50)
        self._lta = jsonArgs.get('lta',200)

    def docs(self):
        docs={}
        docs['one_liner']="Short-term average (STA) divided by long-term average (LTA). This transform highlights onset and so often forms part of an automated pick or edge-detection."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['sta','lta'] }
        return docs


    def execute(self):
        self._result = extra_numpy.sta_lta(self._previous.result(), self._sta, self._lta)

'''
 WienerCommand : Apply a 2D Wiener filter using scipy.signal.wiener()
'''
class WienerCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._xdir = jsonArgs.get('xdir',5)
        self._tdir = jsonArgs.get('tdir',5)
        self._noisePower = jsonArgs.get('noisePower',None)

    def docs(self):
        docs={}
        docs['one_liner']="2D Wiener filter. See scipy.signal.wiener."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xdir','tdir','noisePower'] }
        return docs


    def execute(self):
        self._result = signal.wiener(self._previous.result(), [self._xdir, self._tdir], self._noisePower)
        

'''
 WriteWITSMLCommand : writes the results to the WITSML/FBE format.
'''
class WriteWITSMLCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._outdir = jsonArgs.get('directory_out','NONE')
        self._xaxis = jsonArgs.get('xaxis',None)
        self._prf = jsonArgs.get('prf',10000)
        self._prevstack = jsonArgs.get('commands',[None])
        self._data_style = jsonArgs.get('data_style','NONE')
        self._datedir = jsonArgs.get('date_dir','NONE')
        self._datestring = jsonArgs.get('datestring','NONE')
        self._lowf = jsonArgs.get('low_freq',[0])
        self._highf = jsonArgs.get('high_freq',[1])
        self._labellist = jsonArgs.get('labels',[])
        print(jsonArgs)

    def postcond(self):
        return [directory_services.path_join(self._outdir,self._datedir+'.fbe')]


    def docs(self):
        docs={}
        docs['one_liner']="Write out to the WITSML/FBE format, suitable for loading into viewers such as Techlog or Petrel."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['directory_out','xaxis','data_style','low_freq','high_freq','labels'] }
        return docs


    def execute(self):
        if not self._outdir=='NONE':
            # Resolve band00 here - because if you do it in __init__ you
            # may not have executed the Command that generates the result
            # and so you will get the input a
            self._band00 = None
            # Here we extract the result...because we never pass commands outside
            # the command_set..only numpy or similar in the calculations themselves
            if self._prevstack[0]!=None:
                self._band00= (self._prevstack[0]).result()
            io_helpers.write2witsml(self._outdir,self._datedir,self._datestring,self._xaxis, self._band00, self._previous.result(),
                                    self._lowf, self._highf, self._prf, data_style=self._data_style, label_list=self._labellist)

'''
 RMSfromFFTCommand : the calculation of RMS noise within a frequency band, using the reduced memory functions
'''
class RMSfromFFTCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        f_rescale = jsonArgs.get('f_rescale',1.0)
        self._i1 = int(jsonArgs.get('low_freq',0)*f_rescale)
        highf = jsonArgs.get('high_freq',-1)
        if highf>0:
            highf = highf*f_rescale
        if (int(highf)==self._i1):
            highf=-1
        self._i2 = int(highf)

    def docs(self):
        docs={}
        docs['one_liner']="Calculate the RMS energy between two frequencies."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['low_freq','high_freq'] }
        return docs

    def isGPU(self):
        return True
    

    def execute(self):
        i1 = self._i1
        i2 = self._i2
        if i2<1:
            i2 = int(self._previous.result().shape[1]/2)
        self._result = extra_numpy.agnostic_zeros(self._previous.result(),(self._previous.result().shape[0],1),dtype=numpy.double)
        self._result = extra_numpy.reduced_mem((self._previous.result()[:,i1:i2]), self.result(), extra_numpy.rms_from_fft, {})


'''
 FFTCommand : applies and FFT along the requested axis using the reduced memory functions
'''
class FFTCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Compute the Fast Fourier Transform (FFT) of the data along the requested axis."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def isGPU(self):
        return True
    

    def execute(self):
        if (self._axis==1):
            self._result = extra_numpy.agnostic_zeros(self._previous.result(),(self._previous.result().shape), dtype=numpy.complex)
            self._result = extra_numpy.reduced_mem(self._previous.result(), self.result(), extra_numpy.agnostic_fft, {'axis':self._axis})
        else:
            # currently the low memory version is only developed for the time orientation....
            self._result = extra_numpy.agnostic_fft(self._previous.result(),axis=self._axis)

'''
 IFFTCommand : the inverse FFT along the requested axis
'''
class IFFTCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    def docs(self):
        docs={}
        docs['one_liner']="Compute the Inverse Fast Fourier Transform (IFFT) of the data along the requested axis."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        if (self._axis==1):
            self._result = extra_numpy.agnostic_zeros(self._previous.result(),(self._previous.result().shape), dtype=numpy.complex)
            self._result = extra_numpy.reduced_mem(self._previous.result(), self.result(), extra_numpy.agnostic_ifft, {'axis':self._axis})
        else:
            # currently the low memory version is only developed for the time orientation....
            self._result = extra_numpy.agnostic_ifft(self._previous.result(),axis=self._axis)

'''
 DesripeCommand : Remove vertical stripes from the data using extra_numpy.destripe()
'''
class DestripeCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)

    def docs(self):
        docs={}
        docs['one_liner']="Remove vertical stripes from the data using extra_numpy.destripe()."
        return docs

    def isGPU(self):
        return True
    
    def execute(self):
        self._result = extra_numpy.destripe(self._previous.result())

'''
 UpCommand : Calculate the up-going waves using extra_numpy.up_wave()
'''
class UpCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)

    def docs(self):
        docs={}
        docs['one_liner']="Calculate the up-going waves using extra_numpy.up_wave(). Note that the data should be 2D FFTd before this command and are returned as complex values."
        return docs

    def isGPU(self):
        return True    

    def execute(self):
        self._result = extra_numpy.up_wave(self._previous.result())

'''
 DownCommand : Calculate the down-going waves using extra_numpy.down_wave()
'''
class DownCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)

    def docs(self):
        docs={}
        docs['one_liner']="Calculate the down-going waves using extra_numpy.down_wave(). Note that the data should be 2D FFTd before this command and are returned as complex values."
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.down_wave(self._previous.result())

'''
 VelocityMapCommand : Calculate the phase-velocity at each pixel in a 2D FFT space
'''
class VelocityMapCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs

    def docs(self):
        docs={}
        docs['one_liner']="Calculate the phase velocity at each pixel in a 2D FFT space."
        return docs


    def execute(self):
        localshape = self._previous.result().shape
        nx = localshape[0]
        nt = localshape[1]
        dx = numpy.abs(self._args['xaxis'][1]-self._args['xaxis'][0])
        dt = 1.0/self._args['prf']
        self._result = extra_numpy.vel_map(nx,nt,dx,dt)

'''
 VelocityMaskCommand : Construct a phase-velocity filter in 2D space. The input should be from the velocity_map command
'''
class VelocityMaskCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs

    def docs(self):
        docs={}
        docs['one_liner']="Construct a phase-velocity filter in 2D space. The input should be from the velocity_map command."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['min_velocity','max_velocity'] }
        return docs

    def execute(self):
        velmap = self._previous.result()
        minv = self._args.get('min_velocity',1400.0)
        maxv = self._args.get('max_velocity', 1600.0)
        smoothval = self._args.get('smooth',0)
        self._result = extra_numpy.vel_mask(velmap,minv,maxv,smoothval)

'''
 MultipleCalcsCommand : allows multiple calculations to be made using the reduced_mem form.  
'''
class MultipleCalcsCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        f_rescale = jsonArgs.get('f_rescale',1.0)
        flist = jsonArgs.get('low_freq',[0])
        self._lowf = []
        for freq in flist:
            self._lowf.append(int(freq*f_rescale))
        flist = jsonArgs.get('high_freq',[0])
        self._highf = []
        icount =0
        for freq in flist:
            highf=freq
            if highf>0:
                highf=highf*f_rescale
            if int(highf)==self._lowf[icount]:
                highf=-1
                print("RMS band too narrow - setting to Nyquist")
            self._highf.append(int(highf))
            icount=icount+1
        self._funcName = jsonArgs.get('func','NONE')

    def docs(self):
        docs={}
        docs['one_liner']="Perform multiple calculations using the extra_numpy.reduced_mem() system."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['low_freq','high_freq','func'] }
        return docs

    def isGPU(self):
        return True
        
    def execute(self):
        if not self._funcName=='NONE':
            nyq = int(self._previous.result().shape[1]/2)
            for a in range(len(self._highf)):                
                if (self._highf[a]<0):
                    self._highf[a] = nyq
            self._result = extra_numpy.multiple_calcs(self._previous.result(),self._lowf, self._highf, extra_numpy.available_funcs(self._funcName))

'''
 The KnownCommands. If a command does not appear here it can't be accessed from distpy
'''
def KnownCommands(knownList):
    knownList['NONE']           = BasicCommand
    knownList['data']           = DataLoadCommand
    knownList['abs']            = AbsCommand
    knownList['argmax']         = ArgmaxCommand
    knownList['butter']         = ButterCommand
    knownList['clip']           = ClipCommand
    knownList['count_peaks']    = CountPeaksCommand
    knownList['destripe']       = DestripeCommand
    knownList['diff']           = DiffCommand
    knownList['downsample']     = DownsampleCommand
    knownList['down_wave']      = DownCommand
    knownList['fft']            = FFTCommand
    knownList['from_gpu']       = FromGPUCommand
    knownList['gather']         = GatherCommand
    knownList['gaussian']       = GaussianCommand
    knownList['geometric_mean'] = GeometricMeanCommand
    knownList['harmonic_mean']  = HarmonicMeanCommand
    knownList['ifft']           = IFFTCommand
    knownList['keras']          = KerasCommand
    knownList['kurtosis']       = KurtosisCommand
    knownList['macro']          = MacroCommand
    knownList['median_filter']  = MedianFilterCommand
    knownList['mean']           = MeanCommand
    knownList['multiply']       = MultiplyCommand
    knownList['multiple_calcs'] = MultipleCalcsCommand
    knownList['peak_to_peak']   = Peak2PeakCommand
    knownList['rms_from_fft']   = RMSfromFFTCommand
    knownList['real']           = RealCommand
    knownList['running_mean']   = RunningMeanCommand
    knownList['sum']            = SumCommand
    knownList['skewness']       = SkewnessCommand
    knownList['sta_lta']        = StaLtaCommand
    knownList['std_dev']        = StdDevCommand
    knownList['to_gpu']         = ToGPUCommand
    knownList['up_wave']        = UpCommand
    knownList['velocity_map']   = VelocityMapCommand
    knownList['velocity_mask']  = VelocityMaskCommand
    knownList['wiener']         = WienerCommand
    knownList['write_witsml']   = WriteWITSMLCommand
    return knownList    

'''
 tesxt_for_users : A header and introduction for autogenerated documentation.
'''
def text_for_users():
    text_for_users = {}
    text_for_users['heading']="Public Command Set"
    text_for_users['text'] = [("A set of commands that can be collected together to form a tree-structure ",
                              "for signal processing. The general form of a command is to have a unique identifier ",
                              " \"uid\" and to inherit from a parent via that parent's uid \"uid_in\".\n",
                              "\n",
                              "Commands may have additional attributes as outlined below:\n")]
    
    
