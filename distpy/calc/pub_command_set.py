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
import copy
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
    "coeffs" :        {DEFAULT : None, DESC : "Filter coefficients"},
    "xaxis" :         {DEFAULT : None, DESC : "A numpy vector of distances along the fibre"},
    "taxis" :         {DEFAULT : None, DESC : "A numpy vector of times associated with columns of data"},
    "directory_in" :  {DEFAULT : NONE, DESC : "The subdirectory from which to read data"},
    "directory_out" : {DEFAULT : NONE, DESC : "The subdirectory where results will be written"},
    "direction"     : {DEFAULT : ">", DESC : "The direction for applying the threshold, > or <"},
    "double_ended"  : {DEFAULT : -1,   DESC : "For handling double-ended fibre [-1=single-ended, 0=start-of-fibre half, 1=end-of-fibre half"},
    "edge_order"    : {DEFAULT :  1,   DESC : "Gradient is calculatd using N-th order accurate differences at the boundaries"},
    "filename"      : {DEFAULT : NONE, DESC : "A filename, the complete path will be defined by directory_out and project configurations"},
    "command_list" :  {                DESC : "A list of sub-commands collected in a single macro"},
    "train" :         {                DESC : "A dictionary containing training parameters for keras, e.g. { 'epochs' : 150, 'batch_size' : 10 }"},
    "func" :          {                DESC : "Either rms_from_fft or te_from_fft"}, 
    "labels" :        {                DESC : "A list of column headers"},
    "order" :         {DEFAULT : 5,    DESC : "The order for a filter calculation such as the Butterworth filter"},
    "type" :          {DEFAULT : "lowpass",DESC : "The type of a filter which can be lowpass, highpass, bandpass, or bandstop"},
    "padtype" :       {DEFAULT : "even",DESC : "The type of end-effect control on a filter, see scipy.signal.filtfilt"},
    "prf" :           {DEFAULT : 10000, DESC : "The pulse repetition frequency in Hz (one over the time sample rate)"},
    "freq" :          {DEFAULT : 200.0, DESC : "A frequency in Hz or a wavenumber in 1/m"},
    "filename" :      {DEFAULT : NONE,  DESC : "A filename for read or write operations"},
    "index" :         {DEFAULT : 0,     DESC : "The index of a row or column in sanmples"},
    "axis" :          {DEFAULT : -1,    DESC : "The axis to apply an operation to, typically in distpy axis=0 for depth and axis=1 for time"},
    "xsigma" :        {DEFAULT : 1.0,   DESC : "A standard deviation in the x-direction"},
    "tsigma" :        {DEFAULT : 1.0,   DESC : "A standard deviation in the time direction"},
    "xorder" :        {DEFAULT : 5,     DESC : "In a 2D filter this is the order in the x-direction"},
    "torder" :        {DEFAULT : 5,     DESC : "In a 2D filter, this is the order in the t-direction"},
    "distance" :      {DEFAULT : 3,     DESC : "The number of samples in a median filter"},
    "window_length" : {DEFAULT : 5,     DESC : "The length of a filter window in samples"},
    "n_clusters" :    {DEFAULT : 10,    DESC : "The number of clusters to use when classifying the data"},
    "offset" :        {DEFAULT : 0,     DESC : "An offset from the start of the data, in samples"},
    "xmin" :          {DEFAULT : None,  DESC : "A minimum value on the x-axis"},
    "xmax" :          {DEFAULT : None,  DESC : "A maximum value on the x-axis"},
    "tmin" :          {DEFAULT : None,  DESC : "A minimum value on the time-axis"},
    "tmax" :          {DEFAULT : None,  DESC : "A maximum value on the time-axis"},
    "commands" :      {DEFAULT : [None],DESC : "TODO"},
    "m" :             {DEFAULT : 1.0,     DESC : "The slope of a linear transform (y = m*x + c)"},
    "c" :             {DEFAULT : 0.0,     DESC : "The intercept of a linear transform (y = m*x + c)"},
    "data_style" :    {DEFAULT : NONE,  DESC : "A string identifier for the data inside the WITSML file"},
    "method" :        {DEFAULT : "lin_fit", DESC : "The method for curve fitting, see scipy.optimize.curve_fit"},
    "mode" :          {DEFAULT : "constant", DESC : "Filter edge handling, see the scipy.ndimage documentation."},
    "moment" :        {DEFAULT : 1,     DESC : "The order of central moment, see scipy.stats.moment"},
    "max_velocity":   {DEFAULT : 1600,  DESC : "The maximum phase velocity"},
    "min_velocity":   {DEFAULT : 1400,  DESC : "The minimum phase velocity"},
    "velocity":       {DEFAULT : 1400,  DESC : "Phase velocity"},
    "bandwidth":      {DEFAULT : 0.1,   DESC : "Width of the Gaussian ray in pixels"},
    "shape":          {DEFAULT : [9,9], DESC : "Shape of the filter in pixels e.g. [9,9], both values must be even"},
    "max_val":        {DEFAULT : 1.0,   DESC : "An upper bound"},
    "min_val":        {DEFAULT : 0.0,   DESC : "A lower bound"},
    "smooth" :        {DEFAULT : 0, DESC : "The smoothing factor for the filter"},
    "threshold" :     {DEFAULT : 0, DESC : "An upper or lower limit"},
    "value" :         {DEFAULT : 0, DESC : "A single floating point number"},
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
    "clip_level" :    {DEFAULT : 1.0,   DESC : "The number of standard devaitions about the mean for plotting thumbnails"},
    "tolerance" :     {DEFAULT : 1.0e-6, DESC : "A small tolerance to avoid numerical divide-by-zero"}
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

    @classmethod
    def docs(cls):
        return {}

    # If the pattern for numpy/cupy agnostic coding has been used
    # see https://github.com/Schlumberger/distpy/wiki/Dev-Notes-:-GPU
    def isGPU(self):
        return False

    # If this operation is like a linear filter so that in principle the
    # ordering can be swapped - NOTE this does not check for large/small
    # rounding error considerations.
    def isCommutative(self):
        return False
    
    def execute(self):
        pass

    # This can be overridden with a specific execution that
    # ASSERTS the particular behaviour expected
    def unit_test(self):
        self.execute()
        assert(True),"Default execution failure"

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

    @classmethod
    def docs(cls):
        return {}

    def isGPU(self):
        return False

    def isCommutative(self):
        return False
        
    def execute(self):
        self._result = self._previous.result()
        
    # This can be overridden with a specific execution that
    # ASSERTS the particular behaviour expected
    def unit_test(self):
        self.execute()
        assert(True),"Default execution failure"
      
    def result(self):
        return self._result


'''
 ToGPUCommand : Go to GPU
'''
class ToGPUCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Take the absolute value of the input"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_abs(self._previous.result())

'''
 AngleCommand : wrappers the numpy.angle() function
'''
class AngleCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Take the angle of the complex input"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_angle(self._previous.result())

'''
 CopyCommand : passes on a copy of the current data
               To minimize memory footprint we inplace overwrite
               by default as often as possible.
               Having an explicity copy step allows data to be duplicated
               when inplace overwrite doesn't make sense.
'''
class CopyCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Copy the current data block.To minimize memory footprint we inplace overwrite by default as often as possible. Having an explicity copy step allows data to be duplicated when inplace overwrite doesn't make sense. "
        return docs

    def isGPU(self):
        return False

    def execute(self):
        self._result = copy.deepcopy(self._previous.result())


'''
 RescaleCommand : rescales the data from 0 to 1
'''
class RescaleCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Rescale the data from 0 to 1. If an additional data set it suppied, the rescale uses that data for min() and max()"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        Y = None
        if len(self._prevstack)>0:
            if self._prevstack[0] is not None:
                Y = self._prevstack[0].result()
        self._result = extra_numpy.agnostic_rescale(self._previous.result(), external_scaling=Y)
'''
 SoftThresholdCommand : applies a threshold greater or less
'''
class SoftThresholdCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._direction = jsonArgs.get('direction','>')
        self._threshold = jsonArgs.get('threshold',0.0)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Applies a soft threshold, clipping the values at the given threshold."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['direction','threshold'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.soft_threshold(self._previous.result(),self._threshold,direction=self._direction)

'''
 PeakBroadeningCommand : broadens any peaks by duplicating the maximum value
'''
class PeakBroadeningCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._window_length = jsonArgs.get('window_length',50)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Broadening of the local maxima, by extending them in time."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['window_length'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.broaden(self._previous.result(),window_length=self._window_length)


'''
 HardThresholdCommand : applies a threshold greater or less
'''
class HardThresholdCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._direction = jsonArgs.get('direction','>')
        self._threshold = jsonArgs.get('threshold',0.0)
        self._value     = jsonArgs.get('value',0.0)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Applies a hard threshold, all values beyond the threshold are replaced with the supplied value."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['direction','threshold','value'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.hard_threshold(self._previous.result(),self._threshold, self._value,direction=self._direction)



'''
 UnwrapCommand : unwraps using the numpy.unwrap() function
'''
class UnwrapCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',-1)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Unwrap angles in the selected axis direction using numpy.unwrap"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        docs['args']['axis']['default']=-1
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_unwrap(self._previous.result(), axis=self._axis)



'''
 ConjCommand : wrappers the numpy.conj() function
'''
class ConjCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Take the complex conjugate value of the input"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_conj(self._previous.result())

'''
 ApproxVlfCommand: weighted average of a trace, gives a reasonable approximation to Very Low Frequency response
'''
class ApproxVlfCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="A weighted average focussed on the central trace, equivalent to a very low pass filter. The results approximate the very low frequency response in a robust way."
        return docs

    def isGPU(self):
        return False


    def execute(self):
        super().execute()
        self._result = extra_numpy.approx_vlf(self._previous.result())
        

'''
 KerasCommand : wrappers the extra_numpy.kera_model() function
'''
class KerasCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Load an existing keras model, and either use it for prediction or train-then-predict."
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
 KMeansCommand : wrappers the extra_numpy.kmeans_clustering() function
'''
class KMeansCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Perform a KMeans clustering into a fixed number of clusters. Return the cluster number versus depth."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['n_clusters'] }
        return docs

    def execute(self):
        n_clusters = self._args.get('n_clusters',10)
        if len(self._prevstack) < 1:
            self._prevstack = [None]
        if self._prevstack[0] is not None:
            shape_out = self._previous.result().shape
            for a in range(len(self._prevstack)):
                test_shape = self._prevstack[a].result().shape
                if (test_shape[0]*test_shape[1])<(shape_out[0]*shape_out[1]):
                    shape_out=test_shape
            xrows = shape_out[0]*shape_out[1]
            xcols = len(self._prevstack)+1
            data_big = numpy.zeros((xrows,xcols))
            for a in range(xcols-1):
                data_big[:,a]=self._prevstack[a].result().flatten()[:xrows]
            data_big[:,-1]=self._previous.result().flatten()[:xrows]
            result = extra_numpy.kmeans_clustering(data_big,n_clusters=n_clusters)
            self._result = numpy.reshape(result,shape_out)
        else:
            # Single input...
            self._result = extra_numpy.kmeans_clustering(self._previous.result(),n_clusters=n_clusters)



'''
 KurtosisCommand : wrappers the scipy.stats.kurtosis() function
'''
class KurtosisCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',1)

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Take the real value of the input"
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_real(self._previous.result())

'''
 AnalyticSignalCommand : wrappers the extra_numpy.agnostic_analytic_signal() function
'''
class AnalyticSignalCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._prf = jsonArgs.get('prf',10000)
        self._freq = jsonArgs.get('freq',200.0)
        self._axis = jsonArgs.get('axis',1)
        self._N = jsonArgs.get('window_length',64)
        self._prf = jsonArgs.get('prf',10000.0)
        self._wavn = numpy.abs(jsonArgs['xaxis'][1]-jsonArgs['xaxis'][0])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Estimate the analytic signal using a locally filtered maximum likelihood method"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis','freq','window_length'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        fs = self._prf
        axis = self._axis
        if axis==1:
            fs = self._wavn
        fc = self._freq
        window_length = self._N
        self._result = extra_numpy.agnostic_analytic_signal(self._previous.result(),fc,fs,window_length=window_length, axis=axis)

'''
 HashCommand : creates a 64-bit integer hash using extra_numpy.hash(), reducing the data from 2D to 1D
'''
class HashCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._ioffset = int(jsonArgs.get('offset',0))

    @classmethod
    def docs(cls):
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
        self._args = jsonArgs
        self._order = jsonArgs.get('order',5)
        self._btype = jsonArgs.get('type','lowpass')
        self._ptype = jsonArgs.get('padtype','even')
        self._prf = jsonArgs.get('prf',10000)
        self._freq = jsonArgs.get('freq',200.0)
        self._axis = jsonArgs.get('axis',1)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Setup a Butterworth filter using scipy.signal.butter() and apply it using scipy.signal.filtfilt()"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['order','type','padtype','prf','freq','axis'] }
        return docs

    # This is linear filtering...
    def isCommutative(self):
        return True

    def execute(self):
        Wn = self._freq/(self._prf*0.5)
        if self._axis == 0:
            dx = numpy.abs(self._args['xaxis'][1]-self._args['xaxis'][0])
            Wn = self._freq/(dx*0.5)
        b, a = signal.butter(self._order,Wn,self._btype,output='ba')
        self._result = signal.filtfilt(b,a,self._previous.result(),axis=self._axis, padtype=self._ptype)

'''
 ArgmaxCommand : Used to extract the first index of the maximum value in each
                 row (or column). This is a first order way to extract a
                 profile versus depth from a processed DTS gradient
'''
class ArgmaxCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',0)

    @classmethod
    def docs(cls):
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
        self._mode = jsonArgs.get('mode', 'constant')

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Applies a 2D Gaussian blurring filter using signal.ndarray.gaussian_filter()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xsigma','tsigma','xorder','torder'] }
        return docs

    # This is linear filtering...
    def isCommutative(self):
        return True

    def execute(self):
        self._result = ndimage.gaussian_filter(self._previous.result(), [self._xdir, self._tdir], order=[self._xorder, self._torder],mode=self._mode)

'''
 SobelCommand : Applies a Sobel edge detection filter using signal.ndarray.sobel()
'''
class SobelCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',-1)
        self._mode = jsonArgs.get('mode','constant')

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Applies a Sobel edge detection filter using signal.ndarray.sobel()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis','mode'] }
        return docs

    # This is linear filtering...
    def isCommutative(self):
        return True

    def execute(self):
        self._result = ndimage.sobel(self._previous.result(),axis=self._axis, mode=self._mode)

'''
 ConvolveCommand : Applies the supplied convolution filter
'''
class ConvolveCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._coeffs = jsonArgs.get('coeffs',[[1,0,-1],[2,0,-2],[1,0,-1]])
        self._mode = jsonArgs.get('mode','constant')

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Convolves the supplied filter with the data."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['coeffs','mode'] }
        docs['args']['coeffs']['default'] = [[1,0,-1],[2,0,-2],[1,0,-1]] 
        return docs

    # This is linear filtering...
    def isCommutative(self):
        return True

    def execute(self):
        self._result = extra_numpy.convolve2D(self._previous.result(),self._coeffs, mode=self._mode)

'''
 TXDipCommand : Gaussian-ray dip filter in the x-t domain
'''
class TXDipCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args=jsonArgs

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="time-space domain dip filter."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['velocity','bandwidth','shape'] }
        docs['args']['velocity']['default'] = 100 
        docs['args']['bandwidth']['default'] = 0.1 
        docs['args']['shape']['default'] = [9,9] 
        return docs

    # This is linear filtering...
    def isCommutative(self):
        return True

    def execute(self):
        dx = numpy.abs(self._args['xaxis'][1]-self._args['xaxis'][0])
        dt = 1.0/self._args['prf']
        speed = self._args.get('velocity',100)
        bandwidth = self._args.get('bandwidth',0.1)
        boxshape = self._args.get('shape',[9,9])
        
        self._result = extra_numpy.tx_dipfilter(self._previous.result(),dx,dt,speed,bandwidth=0.1,width=9,height=9)

'''
 CorrelateCommand : Applies the supplied correlation filter
'''
class CorrelateCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._coeffs = jsonArgs.get('coeffs',[[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1]])
        self._mode = jsonArgs.get('mode','constant')

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Correlates the supplied filter with the data."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['coeffs','mode'] }
        docs['args']['coeffs']['default'] = [[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]
        return docs

    # This is linear filtering...
    def isCommutative(self):
        return True

    def execute(self):
        self._result = ndimage.correlate(self._previous.result(),self._coeffs, mode=self._mode)



'''
  MedianFilterCommand : applies a 2D square median filter using ndimage.median_filter()
'''
class MedianFilterCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._distance = jsonArgs.get('distance',3)

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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
 RollCommand : rolls the data along the requested axis using extra_numpy.agnostic_roll
'''
class RollCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis  = jsonArgs.get('axis',1)
        self._shift = jsonArgs.get('window_length',1)
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Rolls the data along the specified axis, using numpy.roll(). An array can be passed in through gather_uid"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis','window_length'] }
        docs['args']['axis']['default']=1
        docs['args']['window_length']['default']=1
        return docs

    def isGPU(self):
        return True

    def execute(self):
        Y = None
        if len(self._prevstack)>0:
            if self._prevstack[0] is not None:
                Y = self._prevstack[0].result()
        if Y is not None:
            self._result = numpy.zeros(((self._previous.result()).shape))
            if self._axis==1:
                for a in range((self._previous.result()).shape[0]):
                    self._result[a,:]=extra_numpy.agnostic_roll((self._previous.result())[a,:], Y[a])
            else:
                for a in range((self._previous.result()).shape[1]):
                    self._result[:,a]=extra_numpy.agnostic_roll((self._previous.result())[:,a], Y[a])
        else:
            self._result = extra_numpy.agnostic_roll(self._previous.result(), self._shift, axis=self._axis)





'''
 RunningMeanCommand : A 1D running mean averaging filter, using the extra_numpy.running_mean()
'''
class RunningMeanCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        # previously was 'length'
        self._N = jsonArgs.get('window_length',5)
        self._axis = jsonArgs.get('axis',0)

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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
        self._dirname = jsonArgs.get('directory_out','NONE')
        self._xaxis = jsonArgs.get('xaxis',None)
        self._taxis = jsonArgs.get('taxis',None)            
        self._xmin = jsonArgs.get('xmin',numpy.min(self._xaxis))
        self._xmax = jsonArgs.get('xmax',numpy.max(self._xaxis))
        self._tmin = jsonArgs.get('tmin',numpy.min(self._taxis))
        self._tmax = jsonArgs.get('tmax',numpy.max(self._taxis))
        self._double_ended = jsonArgs.get('double_ended',-1)
        # for unit tests and documentation...
        if self._tmin is None:
            self._tmin=0
        self._fname = str(numpy.int(self._tmin))

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Clip the data so that all subsequent operations operate on a small window. If directory_out is specified the new axes will also be created in the storage directory"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xaxis','taxis','xmin','xmax','tmin','tmax','directory_out','double_ended'] }
        return docs

    def isGPU(self):
        return True

    def postcond(self):
        return [directory_services.path_join(self._dirname,self._fname+'.npy')]

    def execute(self):
        data = self._previous.result()
        xend = int(self._xaxis.shape[0]*0.5)
        print(xend)
        if self._double_ended==0:
            self._xaxis = self._xaxis[:xend]
            data = data[:xend,:]
        if self._double_ended==1:
            self._xaxis = self._xaxis[xend:]
            data = data[xend:,:]
            self._xaxis = self._xaxis[::-1]
            data = data[::-1,:]
        ixmin = (numpy.abs(self._xaxis-self._xmin)).argmin()
        ixmax = (numpy.abs(self._xaxis-self._xmax)).argmin()
        itmin = 0
        itmax = 0
        if self._taxis is not None:
            itmin = (numpy.abs(self._taxis-self._tmin)).argmin()
            itmax = (numpy.abs(self._taxis-self._tmax)).argmin()
        else:
            itmin=0
            itmax=data.shape[1]
        print('XMIN',self._xmin,self._xaxis[ixmin], ixmin)
        print('XMAX',self._xmax,self._xaxis[ixmax], ixmax)
        self._result = (data)[ixmin:ixmax,itmin:itmax]
        dirname = self._dirname
        if not dirname=='NONE':
            io_helpers.numpy_out(dirname,self._fname,self._result)
            io_helpers.numpy_out(dirname,'measured_depth',self._xaxis[ixmin:ixmax])
            io_helpers.numpy_out(dirname,'time',self._taxis[itmin:itmax])
'''
 DiffCommand : numerical differencing with a window_length offset
'''
class DiffCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._axis = jsonArgs.get('axis',-1)
        # previously was 'n'
        self._n = jsonArgs.get('window_length',1)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Simple differencing using the window_length as the offset."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['window_length','axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        self._result = extra_numpy.agnostic_diff(self._previous.result(),n=self._n, axis=self._axis)
        # boundary condition
        if self._axis==1:
            self._result[:,-1]=self._result[:,-2]
        if self._axis==0:
            self._result[-1,:]=self._result[-2,:]

'''
 GradientCommand : numerical central differencing
'''
class GradientCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._xaxis = jsonArgs['xaxis']
        self._taxis = jsonArgs.get('taxis',None)            
        self._edge_order = jsonArgs.get('edge_order',1)
        self._axis = jsonArgs.get('axis',0)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Numercal gradient of the data via central differencing."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['edge_order','axis'] }
        docs['args']['axis']['default']=0
        return docs

    def isGPU(self):
        return False

    def execute(self):
        if self._axis is not None:
            if self._axis==1:
                self._result = numpy.gradient(self._previous.result(),self._taxis,edge_order=self._edge_order, axis=1)
            else:
                self._result = numpy.gradient(self._previous.result(),self._xaxis,edge_order=self._edge_order, axis=0)
        else:
            self._result = numpy.gradient(self._previous.result(),[self._xaxis,self._taxis],edge_order=self._edge_order)

                

'''
 DownsampleCommand : index-based downsampling. Note there is no filtering, so for anti-alias
                     downsampling apply an appropriate low-pass filter before calling this.
'''
class DownsampleCommand(BasicCommand):
    def __init__(self,command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._xdir = jsonArgs.get('xsample',1)
        self._tdir = jsonArgs.get('tsample',1)

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Couting peaks in a signal using scipy.signal.find_peaks_cwt()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['sta','lta'] }
        return docs


    def execute(self):
        self._result = extra_numpy.count_peaks(self._previous.result(), self._sta, self._lta)


'''
 DeconvolveCommand : deconvolves the data from gather_uids from the data. assumes we are in the Fourier domain
'''
class DeconvolveCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._tolerance = jsonArgs.get('tolerance',1.0e-6)
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="deconvolves the data from gather_uids from the data in the in_uid. Assumes we are in the Fourier domain"
        #docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['tolerance'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):        
        if self._prevstack[0]!=None:
            data = (self._prevstack[0]).result()
            self._result = extra_numpy.deconvolve(self._previous.result(),data,tolerance=self._tolerance)
        else:
            self._result = self._previous.result() # null behaviour

'''
 MultiplyCommand : elementwise multiply
'''
class MultiplyCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])
        self._axis = jsonArgs.get('axis',None)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Elementwise multiply, the output data-type will be the same as that of the data entering in the in_uid. This data is multiplied by data provided in the gather_uids"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):        
        if self._prevstack[0]!=None:
            data = (self._prevstack[0]).result()
            if data.shape==self._previous.result().shape:
                self._result = self._previous.result()*data
            else:
                if not self._axis is None:
                    if self._axis == 1:
                        self._result = numpy.einsum('ij,j->ij',self._previous.result(),data.flatten())
                    else:
                        data = numpy.einsum('ij,i->ij',self._previous.result(),data.flatten())
                else:
                    self._result = self._previous.result() # null behaviour
                
'''
 AddCommand : elementwise sum
'''
class AddCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Elementwise sum, the output data-type will be the same as that of the data entering in the in_uid."
        #docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['window_length','axis'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        if self._prevstack[0]!=None:
            self._result = self._previous.result()+((self._prevstack[0]).result())
'''
 LinearScalarCommand : y = mx+c
'''
class LinearTransformCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._m = jsonArgs.get('m', 1.0)
        self._c = jsonArgs.get('c', 0.0)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Provide two scalars m and c, linearly transform the data y = m*data + c"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['m','c'] }
        return docs

    # This is linear filtering...
    def isCommutative(self):
        return True

    def isGPU(self):
        return True

    def execute(self):
        m = self._m
        c = self._c
        self._result = m*self._previous.result()+c


'''
 GatherCommand : Gather results together
'''
class GatherCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])

    @classmethod
    def docs(cls):
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
 ExtractCommand : Extract a named column or row
'''
class ExtractCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._prevstack = jsonArgs.get('commands',[None])
        self._idx = jsonArgs.get('index',0)
        self._axis = jsonArgs.get('axis',1)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Extracts a single row (axis=0) or column (axis=1) at the specified index, as a separate dataset"
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['index','axis'] }
        docs['args']['axis']['default'] = 1
        return docs

    def isGPU(self):
        return True

    def execute(self):
        if self._axis==0:
            self._result = (self._previous.result())[self._idx,:]
        else:
            self._result = (self._previous.result())[:,self._idx]
        self._result = numpy.reshape(self._result,(self._result.flatten().shape[0],1))


'''
 StaLtaCommand : short-term average to long-term-average onset-picker transform
'''
class StaLtaCommand(BasicCommand):
    def __init__(self,command,jsonArgs):
        super().__init__(command, jsonArgs)
        self._sta = jsonArgs.get('sta',50)
        self._lta = jsonArgs.get('lta',200)

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="2D Wiener filter. See scipy.signal.wiener."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['xdir','tdir','noisePower'] }
        return docs


    def execute(self):
        print('wiener ['+str(self._xdir) + ', ' + str(self._tdir) + ']')
        self._result = signal.wiener(self._previous.result(), [self._xdir, self._tdir], self._noisePower)


'''
 WriteNPYCommand : writes out the current datablock to NPY format
'''
class WriteNPYCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._dirname = jsonArgs.get('directory_out','NONE')
        self._fname = jsonArgs.get('date_dir','NONE')
        self._xaxis = jsonArgs['xaxis']
        self._taxis = jsonArgs.get('taxis',None)            


    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Write the current state of the processed data to the npy format."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['directory_out'] }
        return docs

    def postcond(self):
        return [directory_services.path_join(self._dirname,self._fname+'.npy')]

    def isGPU(self):
        return True

    def execute(self):
        super().execute()
        dirname = self._dirname
        if not dirname=='NONE':
            extra_numpy.agnostic_save(dirname,self._fname,self._previous.result())
            extra_numpy.agnostic_save(dirname,'measured_depth',self._xaxis)
            if self._taxis is not None:
                extra_numpy.agnostic_save(dirname,'time',self._taxis)

'''
 WriteNPYCommand : writes out the current datablock to NPY format
'''
class ReadNPYCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._dirname = jsonArgs.get('directory_out','NONE')
        self._fname = jsonArgs.get('filename','NONE')

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Load a npy format file."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['directory_out','filename'] }
        return docs

    def isGPU(self):
        return True

    def execute(self):
        super().execute()
        dirname = self._dirname
        if not dirname=='NONE':
            # we can use the previous result to tell us whether we are on GPU or CPU
            self.result = extra_numpy.agnostic_load(dirname,self._fname,self._previous.result())


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
        #print(jsonArgs)

    def postcond(self):
        return [directory_services.path_join(self._outdir,self._datedir+'.fbe')]


    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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
        self._axis = jsonArgs.get('axis',0)

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Remove vertical stripes from the data using extra_numpy.destripe()."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def isGPU(self):
        return True
    
    def execute(self):
        if self._axis==0:
            self._result = extra_numpy.destripe(self._previous.result())
        else:
            self._result = extra_numpy.destripe(self._previous.result().transpose()).transpose()
            

'''
 UpCommand : Calculate the up-going waves using extra_numpy.up_wave()
'''
class UpCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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

    @classmethod
    def docs(cls):
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
 BoundedSelectCommand : return a mask within a pair of bounds on the data
'''
class BoundedSelectCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Return a masks within a pair of bounds on the data (e.g. selecting a cluster from k-means results)."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['min_val','max_val'] }
        return docs

    def execute(self):
        velmap = self._previous.result()
        minv = self._args.get('min_val',0.0)
        maxv = self._args.get('max_val', 1.0)
        self._result = extra_numpy.bounded_select(velmap,minv,maxv)

'''
 CMPCommand : return a virtual source-based common midpoint gather
'''
class CMPCommand(BasicCommand):
    def __init__(self, command, jsonArgs):
        super().__init__(command, jsonArgs)
        self._args = jsonArgs

    @classmethod
    def docs(cls):
        docs={}
        docs['one_liner']="Returns a common midpoint gather based on virtual sources."
        docs['args'] = { a_key: universal_arglist[a_key] for a_key in ['axis'] }
        return docs

    def execute(self):
        axis = self._args.get('axis',0)
        self._result = extra_numpy.virtual_cmp(self._previous.result(),axis=axis)


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

    @classmethod
    def docs(cls):
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
    knownList['angle']          = AngleCommand
    knownList['add']            = AddCommand
    knownList['analytic_signal']= AnalyticSignalCommand
    knownList['argmax']         = ArgmaxCommand
    knownList['approx_vlf']     = ApproxVlfCommand
    knownList['bounded_select'] = BoundedSelectCommand
    knownList['broaden']        = PeakBroadeningCommand
    knownList['butter']         = ButterCommand
    knownList['clip']           = ClipCommand
    knownList['conj']           = ConjCommand
    knownList['convolve']       = ConvolveCommand
    knownList['copy']           = CopyCommand
    knownList['correlate']      = CorrelateCommand
    knownList['count_peaks']    = CountPeaksCommand
    knownList['data_load']      = ReadNPYCommand
    knownList['deconvolve']     = DeconvolveCommand
    knownList['destripe']       = DestripeCommand
    knownList['diff']           = DiffCommand
    knownList['downsample']     = DownsampleCommand
    knownList['dip_filter']     = TXDipCommand
    knownList['down_wave']      = DownCommand
    knownList['extract']        = ExtractCommand
    knownList['fft']            = FFTCommand
    knownList['from_gpu']       = FromGPUCommand
    knownList['gather']         = GatherCommand
    knownList['gaussian']       = GaussianCommand
    knownList['geometric_mean'] = GeometricMeanCommand
    knownList['gradient']       = GradientCommand
    knownList['harmonic_mean']  = HarmonicMeanCommand
    knownList['hard_threshold'] = HardThresholdCommand
    knownList['ifft']           = IFFTCommand
    knownList['keras']          = KerasCommand
    knownList['kmeans']         = KMeansCommand
    knownList['kurtosis']       = KurtosisCommand
    knownList['lin_transform']  = LinearTransformCommand
    knownList['macro']          = MacroCommand
    knownList['median_filter']  = MedianFilterCommand
    knownList['mean']           = MeanCommand
    knownList['multiply']       = MultiplyCommand
    knownList['multiple_calcs'] = MultipleCalcsCommand
    knownList['peak_to_peak']   = Peak2PeakCommand
    knownList['rms_from_fft']   = RMSfromFFTCommand
    knownList['real']           = RealCommand
    knownList['rescale']        = RescaleCommand
    knownList['roll']           = RollCommand
    knownList['running_mean']   = RunningMeanCommand
    knownList['sobel']          = SobelCommand
    knownList['soft_threshold'] = SoftThresholdCommand
    knownList['sum']            = SumCommand
    knownList['skewness']       = SkewnessCommand
    knownList['sta_lta']        = StaLtaCommand
    knownList['std_dev']        = StdDevCommand
    knownList['to_gpu']         = ToGPUCommand
    knownList['unwrap']         = UnwrapCommand
    knownList['up_wave']        = UpCommand
    knownList['velocity_map']   = VelocityMapCommand
    knownList['velocity_mask']  = VelocityMaskCommand
    knownList['virtual_cmp']    = CMPCommand
    knownList['wiener']         = WienerCommand
    knownList['write_npy']      = WriteNPYCommand
    knownList['write_witsml']   = WriteWITSMLCommand
    return knownList    

'''
 text_for_users : A header and introduction for autogenerated documentation.
'''
def text_for_users():
    text_for_users = {}
    text_for_users['heading']="Public Command Set"
    text_for_users['text'] = [("A set of commands that can be collected together to form a tree-structure ",
                              "for signal processing. The general form of a command is to have a unique identifier ",
                              " \"uid\" and to inherit from a parent via that parent's uid \"uid_in\".\n",
                              "\n",
                              "Commands may have additional attributes as outlined below:\n")]
    
    
