# (C) 2020, Schlumberger. Refer to LICENSE.

# For GPU support also see agnostic.py

import numpy
import datetime

import distpy.calc.agnostic as agnostic

import scipy.signal
import copy
from scipy import ndimage
from scipy.signal import butter, lfilter, freqz
from scipy.signal import hilbert


# To optimize runtimes - alter this boxsize
# This is used by the reduced memory enabled functions
# see reduced_mem() and available_funcs()
# The reduced memory option is particularly for low memory devices
BOXSIZE=400
GPU_CPU = agnostic.agnostic()


# extra functions for image cleanup (i.e. introduced for plotting, but
# all are 1 or 2D data signal processing so belong in here)
# https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
def normalized(a, axis=-1, order=2):
    xp = GPU_CPU.get_numpy(a)
    l2 = xp.atleast_1d(xp.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / xp.expand_dims(l2, axis)
'''
despike - despiking using a local median filter
'''
def despike(data, m = 2.):
    # Median means that this is not available for GPU
    d = numpy.abs(data - numpy.median(data))
    mdev = numpy.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]
'''
intensity - rescale data to lie in the 0-1 range
'''
def intensity(data):
    xp = GPU_CPU.get_numpy(data)
    # min goes to zero
    val1 = xp.min(data)
    data = data - xp.min(data)
    # max goes to nearly 1
    data = data / (xp.max(data)+1e-6)
    return data

'''
Curve fitting functions
'''
#various functions for curve fitting
def exp_fit(x, a, b, c):
    xp = GPU_CPU.get_numpy(x)
    return a * xp.exp(b *x) + c
def sum2exp_fit(x, a, b, c, d, e):
    xp = GPU_CPU.get_numpy(x)
    return a*xp.exp(b*x) + c*GPU_CPU.exp(d*x) + e
def lin_fit(x,a,b):
    return a*x + b
# A selection of a curve fit function
def select_fit(strFit, values, initslopes):
    # NOT GPU READY...
    func_fit = lin_fit
    coeffs0 = [initslopes[0],values[-1]]
    if strFit=="exp_fit":
        coeffs0 = [values[0]-values[-1],initslopes[0],values[-1]]
        func_fit = exp_fit
    if strFit=="sum2exp_fit":
        coeffs0 = [values[0]-values[-1],initslopes[0],values[0]-values[-1],initslopes[1],values[-1]]
        func_fit = sum2exp_fit
    return func_fit, coeffs0

'''
 Some functions that are GPU_CPU and used directly in
 pub_command_set.py. This means pub_command_set depends on extra_numpy
 and does not need to know about GPU/CPU issues.
'''
def agnostic_abs(x):
    xp = GPU_CPU.get_numpy(x)
    return xp.abs(x)

def agnostic_argmax(x, axis=None):
    xp = GPU_CPU.get_numpy(x)
    return xp.argmax(x, axis=axis)

def agnostic_diff(x, n=1, axis=-1):
    xp = GPU_CPU.get_numpy(x)
    return xp.diff(x,n=n,axis=axis)


def agnostic_fft(x,axis=-1):
    xp = GPU_CPU.get_numpy(x)
    return xp.fft.fft(x,axis=axis)


def agnostic_ifft(x,axis=-1):
    xp = GPU_CPU.get_numpy(x)
    return xp.fft.ifft(x,axis=axis)

def agnostic_mean(x, axis=None, keepdims=True ):
    xp = GPU_CPU.get_numpy(x)
    return xp.mean(x,axis=axis, keepdims=keepdims)

def agnostic_real(x):
    xp = GPU_CPU.get_numpy(x)
    return xp.real(x)

def agnostic_std(x, axis=None, keepdims= True):
    xp = GPU_CPU.get_numpy(x)
    return xp.std(x,axis=axis, keepdims=keepdims)

def agnostic_sum(x, axis=None, keepdims= True):
    xp = GPU_CPU.get_numpy(x)
    return xp.sum(x,axis=axis, keepdims=keepdims)

def agnostic_zeros(x,size_tuple,dtype=numpy.double):
    xp = GPU_CPU.get_numpy(x)
    return xp.zeros(size_tuple,dtype=dtype)

    

'''
Configure the calculation for different hardware sizes.
'''
def set_boxsize(newval):
    global BOXSIZE
    BOXSIZE = newval

def to_gpu(data):
    return GPU_CPU.asarray(data)

def from_gpu(data):
    return GPU_CPU.asnumpy(data)

'''
Running mean calculation
'''
# see https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean_1d(x, N):
    # NOT GPU because of numpy.insert()
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def running_mean(data, N, axis=0):
    # NOT GPU because of extra_numpy.running_mean_1d
    if axis==0:
        for a in range(data.shape[0]):
            data[a,:-N+1]=running_mean_1d(data[a,:],N)
    else:
        for a in range(data.shape[1]):
            data[:-N+1,a]=running_mean_1d(data[:,a],N)
'''
 gather - gather together a list of matrices in a single numpy matrix
'''
def gather(data,prevstack):
    xp = GPU_CPU.get_numpy(data)
    nx = data.shape[0]
    nt = data.shape[1]
    for dataset in prevstack:
        if dataset.ndim > 1:
            nt += dataset.shape[1]
        else:
            nt += 1
    result = xp.zeros((nx,nt),dtype=data.dtype)
    nt = data.shape[1]
    result[:,:nt] = data[:,:]
    ntold=nt
    for dataset in prevstack:
        if dataset.ndim > 1:
            nt += dataset.shape[1]
            result[:,ntold:nt]=dataset[:,:]
        else:
            result[:,ntold:nt]=xp.reshape(dataset,(-1,1))
            nt += 1
        ntold=nt
    return result

'''
 interp - a wrapper around numpy's 1d interpolation
'''
def interp(data, xp, fp, axis=0):
    # NOT GPU because of numpy.interp()
    if axis==0:
        for a in range(data.shape[0]):
            data[a,:]=numpy.interp(data[a,:],xp,fp)
    else:
        for a in range(data.shape[1]):
            data[:,a]=numpy.interp(data[:,a],xp,fp)


'''
 approx_vlf - returns a weighted average of the data
              This is a Gaussian window sum, we return the median
              a combination which seems robust to data with fading-lines
'''
def approx_vlf(data):
    # NOT GPU because of scipy.signal
    # A gaussian window to make a weighted average trace, (i.e. a low pass filter)
    win_len = data.shape[1]
    variance = 3000000
    win_gauss = numpy.arange(0,win_len)
    win_gauss = numpy.exp((-(win_gauss-(0.5*win_len))**2)/(2*variance))
    win_gauss = win_gauss/numpy.sum(win_gauss)
    
    # Applied with a median filter (some DAS have fading issues)
    return scipy.signal.medfilt(numpy.average(data,weights=win_gauss,axis=1),3)



'''
 te_from_fft - this is the generic Frequency Band Energy in total-energy form
                 Note that the data is already FFT'd and only the frequencies you
                 want to sum are sent in.
'''
def te_from_fft(localData):
    xp = GPU_CPU.get_numpy(localData)
    # calulate Total Energy (sum of squares)
    nx = localData.shape[0]
    squared = xp.real(localData*xp.conj(localData))
    nval = (len(squared[0])-1)*2
    # integration for RMS
    if nval%2:
        rmsval = (squared[:,0]+2*xp.sum(squared[:,1:],axis=1))
    else:
        rmsval = (squared[:,0]+2*xp.sum(squared[:,1:-1],axis=1) + squared[:,-1])
    return rmsval

'''
 rms_from_fft - this is the generic Frequency Band Energy in root-mean-square form
                 Note that the data is already FFT'd and only the frequencies you
                 want to sum are sent in.
'''
# The general FBE calculation - the sender must control which part of the FFT is sent in...
def rms_from_fft(localData):
    xp = GPU_CPU.get_numpy(localData)
    # calulate RMS
    nx = localData.shape[0]
    nval = (nx-1)*2
    # First get the total energy
    rmsval = te_from_fft(localData)
    # Convert to RMS for output
    return (xp.sqrt(rmsval/nval)/xp.sqrt(nval))


'''
 macro_factory - a mini command factory.
     This allows for the definition of macros. So more complex commands are
     encapsulated in a single command - see pub_command_set.Macro
'''
def macro_factory(args, traces):
    # NOTE GPU because of scipy.ndimage.median_filter
    # set up all the possible defaults...
    axis_val = args.get('axis',0)
    distance_val = args.get('distance',5)
    sigma_val = args.get('sigma',2)
    zone_index = args.get('zone_index',0)
    threshold = args.get('threshold',0)
    suffix = args.get('suffix','POST')

    commandList = {
        "abs"             : numpy.abs(traces),
        "gaussian_filter" : ndimage.gaussian_filter(traces,sigma=sigma_val),
        "gradient"        : numpy.gradient(traces.astype(numpy.double), distance_val, axis=axis_val),
        "normalize"       : normalized(traces, axis=axis_val),
        "percentage_on"   : 100*numpy.sum(traces,axis=axis_val)/traces.shape[axis_val],  
        "sum"             : numpy.sum(traces, axis=axis_val),
        "threshold"       : (traces > threshold),
        "median_filter"   : ndimage.median_filter(traces, distance_val)
    }
    # if the command is not found the data is returned untouched...
    return commandList.get(args['name'], traces)
    
'''
 reduced_mem - manages memory for processing on low spec Edge devices
             - Usage: reduced_mem(dataIn, dataout, te_from_fft, {})
             -        reduced_mem(dataIn, dataOut, numpy.fft.fft, {'axis' : 1}
'''
def reduced_mem(dataIn, dataOut, func, keyword_arguments):
    global BOXSIZE
    nx = dataIn.shape[0]
    istart = 0    
    while istart+BOXSIZE<nx:
        dataOut[istart:istart+BOXSIZE,:] = func(dataIn[istart:istart+BOXSIZE,:], **keyword_arguments).reshape(BOXSIZE,dataOut.shape[1])
        istart = istart + BOXSIZE
    if istart<nx:
        dataOut[istart:,:] = func(dataIn[istart:,:], **keyword_arguments).reshape(dataOut.shape[0]-istart,dataOut.shape[1])
        istart = istart + BOXSIZE
    return dataOut

'''
 up_wave - performs a 2D FFT, separates the up-going wave and returns the 2D iFFT
         - NOTE: the caller will end up with a complex matrix of numbers.
'''
def up_wave(data):
    # Assume 2D FFT
    SMALL=1.0e-6
    m = len(data)
    n = len(data[0])
    m_on_2 = m//2
    n_on_2 = n//2
    DATA = copy.deepcopy(data)

    DATA[m_on_2:,:n_on_2]*=SMALL
    DATA[:m_on_2,n_on_2:]*=SMALL
    DATA[:m_on_2,n_on_2:]*=(1-SMALL)
    DATA[m_on_2:,:n_on_2]*=(1-SMALL)
    xp = GPU_CPU.get_numpy(DATA)
    return (xp.fft.ifft2(DATA))

'''
 down_wave - performs a 2D FFT, separates the up-going wave and returns the 2D iFFT
           - NOTE: the caller will end up with a complex matrix of numbers.
           see also - up_wave
'''
def down_wave(data):
    # Assume 2D FFT 
    SMALL=1.0e-6
    m = len(data)
    n = len(data[0])
    m_on_2 = m//2
    n_on_2 = n//2

    DATA = copy.deepcopy(data)
    DATA[m_on_2:,:n_on_2]*=(1-SMALL)
    DATA[:m_on_2,n_on_2:]*=(1-SMALL)
    DATA[:m_on_2,n_on_2:]*=SMALL
    DATA[m_on_2:,:n_on_2]*=SMALL
    xp = GPU_CPU.get_numpy(DATA)
    return (xp.fft.ifft2(DATA))

'''
 source - Usage: pass in the results of up_wave and down_wave as up and down
        -        The returned matrix highlights sources
        -       (i.e. the cross-corrlation of the up and down wave is maximum at sources)
'''
def source(up,down):
    xp = GPU_CPU.get_numpy(up)
    return xp.abs((up*xp.conj(down)))

'''
 reflection - Usage: pass in the results of up_wave and down_wave as up and down
            -        The returned matrix highlights reflections
            -        (i.e. the down-data deconvolved from the up-data
'''
def reflection(up,down):
    xp = GPU_CPU.get_numpy(up)
    TOLERANCE = 1.0e-7
    return xp.abs((up*xp.conj(down))/((down*xp.conj(down))+TOLERANCE))

'''
 vel_map - Returns the phase-velocity map in the 2D FFT space
'''
def vel_map(nx,nt,dx,dt):
    # COULD BE GPU COMPATIBLE...BUT VEL_MASK IS NOT
    TOLERANCE=1.0e-7
    freq = numpy.linspace(0,1.0/(2*dt),nt//2)
    wavn = numpy.linspace(0,1.0/(2*dx),nx//2)
    frequency = numpy.zeros((1,nt),dtype=numpy.double)
    frequency[0,:nt//2]=freq
    frequency[0,-(freq.shape[0]):]=numpy.flipud(freq)
    halfLength = nx//2
    wavenumber = numpy.zeros((nx,1),dtype=numpy.double)
    wavenumber[0:halfLength]=numpy.reshape(wavn,(halfLength,1))
    lineTest = numpy.reshape(numpy.flipud(wavn),(halfLength,1))
    wavenumber[halfLength:]= lineTest[:len(wavenumber[halfLength:])]
    freqImage = numpy.reshape(numpy.tile(frequency,(nx,1)),(nx,nt))
    wavnImage = numpy.reshape(numpy.tile(wavenumber,(1,nt)),(nx,nt))
    return freqImage/(wavnImage+TOLERANCE)

'''
 vel_mask - use the results of vel_map to create a filter that band passes phase velocities
        - between vel_low and vel_high. The smoothing is applied so that the edges are not too
        - sharp. The padtype='even' prevents strong filter edge effects.
'''
def vel_mask(velmap,vel_low,vel_high,smooth=0):
    # NOT GPU COMPATIBLE DUE TO scipy.signal.filtfilt
    nx=velmap.shape[0]
    nt=velmap.shape[1]
    velm = numpy.zeros((nx,nt),dtype=numpy.double)
    velm[numpy.where(velmap > vel_low)]=1.0
    velm[numpy.where(velmap > vel_high)]=0.0
    if smooth>1:
        Wn = 1.0/smooth
        b, a = scipy.signal.butter(5,Wn,'low',output='ba')
        velm = numpy.abs(scipy.signal.filtfilt(b,a,velm,axis=1, padtype='even'))
    return velm
    
'''
  multiple_calcs - allows a range of FBEs to be calculated using the reduced_mem calculation system
'''
def multiple_calcs(localData, low_freq, high_freq, func):
    xp = GPU_CPU.get_numpy(localData)    
    bulkOut = xp.zeros((localData.shape[0],len(low_freq)),dtype=numpy.double)
    oneOut = xp.zeros((localData.shape[0],1))
    for a in range(len(low_freq)):
        oneOut = reduced_mem(localData[:,low_freq[a]:high_freq[a]],oneOut, func, {})
        bulkOut[:,a]=numpy.squeeze(oneOut)
    return bulkOut

'''
 running_max - return a running maximum value
'''
def running_max(data, running_window_length):
    nx = len(data)
    nt = len(data[0])
    xp = GPU_CPU.get_numpy(data)
    result = xp.zeros((nx,nt),dtype=xp.double)
    for a in range(0,nx,running_window_length):
        result[a,:]=xp.max(data[a:a+running_window_length],axis=0)
    return result

'''
 running_min : return a runningin minimum value
'''
def running_min(data, running_window_length):
    nx = len(data)
    nt = len(data[0])
    xp = GPU_CPU.get_numpy(data)
    result = xp.zeros((nx,nt),dtype=xp.double)
    for a in range(0,nx,running_window_length):
        result[a,:]=xp.min(data[a:a+running_window_length],axis=0)
    return result

'''
 sta_lta : short-term-average over long-term-average, a good signal onset detection transform
'''
def sta_lta(data, sta, lta):
    # Forward looking sta, backward looking lta...
    nx = len(data)
    nt = len(data[0])
    # NOT GPU because of ndimage.uniform_filter
    sta_result = ndimage.uniform_filter(data,size=(1,sta),mode='nearest')/sta
    lta_result = ndimage.uniform_filter(data,size=(1,lta),mode='nearest')/lta
    lta_result[numpy.where(numpy.abs(lta_result)<1e-6)]=1e-6
    return (sta_result/lta_result)

'''
 peak_to_peak : find the largest range occurring within the provided window
'''
def peak_to_peak(data,window_length):
    # NOT GPU because of ndimage.maximum_filter
    nx = len(data)
    nt = len(data[0])
    half_length=int(window_length*0.5)
    return numpy.max(ndimage.maximum_filter(data,size=(1,window_length),mode='nearest')-ndimage.minimum_filter(data,size=(1,window_length),mode='nearest'),axis=1)

'''
 count_peaks : count the number of peaks using scipy.signal.find_peaks_cwt
'''
def count_peaks(data,sta,lta):
    # NOT GPU because of scipy.signal.find_peaks_cwt
    nx = len(data)
    nt = len(data[0])
    result=numpy.zeros((nx,1),dtype=numpy.int)
    for a in range(nx):
        peak_kind = scipy.signal.find_peaks_cwt(data[a,:],numpy.arange(sta,lta,5))
        result[a]=len(peak_kind)
    return result

'''
 rescaled_freq : used to turn a frequency into the nearest index in the Fourier space
'''
def rescaled_freq(freq_list, freq_rescale):
    newlist=[]
    for freq in freq_list:
        newlist.append(int(freq*freq_rescale))
    return newlist


'''
virtual_source - assumes that the data is in the frequency domain

The returned result is still in the frequency domain
'''
def virtual_source(data,traceId):
    nx = data.shape[0]
    nt = data.shape[1]
    xp = GPU_CPU.get_numpy(data)
    sourceImage = xp.reshape(xp.tile(data[traceId,:],(nx,1)),(nx,nt))
    return data*xp.conj(sourceImage)

'''
the seismic sweetness attribute is the amplitude of the analytic signal
divided by the sqrt of the phase. It highlights high amplitude at low
frequency (in young clastic sedimentary basins this gives "sweet spots"
for potential hydrocarbons).
'''
def sweetness(data, dx):
    dk = 1/dx
    # NOT GPU because of scipy.signal.hilbert
    analytic_signal = hilbert(data,axis=0)
    instantaneous_phase = numpy.unwrap(numpy.angle(analytic_signal))
    instantaneous_frequency = (numpy.diff(instantaneous_phase) /(2.0*numpy.pi*dk)) 
    return numpy.abs(analytic_signal)/numpy.sqrt(instantaneous_frequency+1e-7)

'''
The scattering matrix - given Up and Down.
 | Rdown  Tup |
 | Tdown  Rup |

expects the fft(up) and fft(down) as inputs as
this is a frequency domain evaluation
'''
def scattering_matrix(up, down, ispan):
    Pup = up
    Pdown = down
    xp = GPU_CPU.get_numpy(up)
    Qup = xp.roll(up,ispan,axis=0)
    Qdown = xp.roll(down,ispan,axis=0)
    denominator = (Pdown*xp.conj(Qdown))-(xp.conj(Pup)*Qup)
    # threshold of 100 - seem high???
    denominator = xp.sign(denominator)/(xp.abs(denominator)+100.0)

    # scattering matrix
    Rdown=(Pup*xp.conj(Qdown))  -(xp.conj(Pdown)*Qup)
    Tup  =(Pdown*xp.conj(Pdown))-(Pup*xp.conj(Pup))
    Tdown=(Qdown*xp.conj(Qdown))-(Qup*xp.conj(Qup))
    Rup  =(Pdown*xp.conj(Qup))  -(xp.conj(Pup)*Qdown)
    # denominator correction
    Rdown = Rdown*denominator
    Tup   = Tup*denominator
    Tdown = Tdown*denominator
    Rup   = Rup*denominator

    return (Rdown, Tup, Tdown, Rup)

'''
 destripe : does a good job of removing "phase skips" that are vertical lines.
            See Bouali, (2010). A simple and robust destriping algorithm for
            imaging spectrometers: application to modis data. San Diego, California,
            ASPRS 2010 Annual Conference.
'''
def destripe(localData):
    # Vertical destriping...
    TOLERANCE=1.0e-6
    dt=1
    dx=1
    xp = GPU_CPU.get_numpy(localData)
    nx = len(localData)
    nt = len(localData[0])
    freq = xp.linspace(0,1.0/(2*dt),xp.int(nt/2))
    frequency = xp.zeros((1,nt),dtype=xp.double)
    frequency[0,0:xp.int(nt/2)]=freq
    frequency[0,xp.int(nt/2):]=xp.flipud(freq)
    wavn = xp.linspace(0,1.0/(2*dx),xp.int(nx/2))
    wavenumber = xp.zeros((nx,1),dtype=xp.double)
    wavenumber[0:xp.int(nx/2)]=xp.reshape(wavn,(xp.int(nx/2),1))
    wavenumber[xp.int(nx/2):]=numpy.reshape(xp.flipud(wavn),(xp.int(nx/2),1))
    freqImage = xp.reshape(xp.tile(frequency,(nx,1)),(nx,nt))
    wavnImage = xp.reshape(xp.tile(wavenumber,(1,nt)),(nx,nt))
    #Rx = (freqImage*freqImage)/((freqImage*freqImage)+(wavnImage*wavnImage)+TOLERANCE)
    #Rx[Rx<1e-6]=1e-6
    Ry = (wavnImage*wavnImage)/((freqImage*freqImage)+(wavnImage*wavnImage)+TOLERANCE)
    Ry[Ry<1e-6]=1e-6
    return xp.real(xp.fft.ifft2(xp.fft.fft2(localData)*Ry))

'''
 acoustic_properties : the eigenvalues of the scattering matrix
'''
def acoustic_properties(Rdown,Tup,Tdown,Rup):
    xp = GPU_CPU.get_numpy(Tup)
    tl=1-(xp.conj(Rdown)*Rdown+Tdown*xp.conj(Tdown))
    tr=(xp.conj(Rdown)*Tup)+(xp.conj(Tdown)*Rup)
    bl=(xp.conj(Tup)*Rdown)+(xp.conj(Rup)*Tdown)
    br=1-((xp.conj(Tup)*Tup)+(xp.conj(Rup)*Rup))
    detS=(tl*br)-(tr*bl)
    traceS=tl+br
    eval1=xp.real(0.5*(traceS+sqrt((traceS**2)-(4*detS))))
    eval2=xp.real(0.5*(traceS-sqrt((traceS**2)-(4*detS))))
    return (eval1, eval2)

'''
log_abs_sq : useful for computing the cepstrum if data is already in the frequency domain
             cepstrum is
                    abs(ifft(log_abs_sq(data)))
             so it can be calculated by a worklflow:
                    fft -> log_abs_sq -> ifft -> abs
'''
def log_abs_sq(data):
    xp = GPU_CPU.get_numpy(data)
    return xp.log(xp.abs(data)**2)

'''
 hash : take the next 64 values above and below 0.5 (assume the previous calculation
        has left the values as 0 and 1.
        Convert to a single 64-bit integer
'''
def hash(data,ioffset=0):
    xp = GPU_CPU.get_numpy(data)
    outData = xp.zeros((data.shape[0],1),dtype=xp.uint64)
    for a in range(data.shape[0]):
        outData[a] = sum(int(j)<<int(i) for i,j, in enumerate(reversed((data[a,ioffset:ioffset+64]).flatten().tolist())))
    return outData
    
'''
 available_funcs : these are the functions for which the reduced_mem options are available.
'''
def available_funcs(funcName):
    if funcName=='te_from_fft':
        return te_from_fft
    if funcName=='rms_from_fft':
        return rms_from_fft
    if funcName=='fft':
        return agnostic_fft

