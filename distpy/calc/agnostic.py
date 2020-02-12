# (C) 2020, Schlumberger. Refer to LICENSE. 
'''
 agnostic : to support code that is CPU and GPU compatible via cupy
            Unfortunately cupy cannot be installed on a machine that
            does not have GPU so we also have cupy_facade.py and do
            not include cupy in the requirements.txt

 When considering developing GPU algorithms refer to:
 https://docs-cupy.chainer.org/en/stable/reference/comparison.html

 The general policy is to have numpy/ndimage/scipy algorithms that are
 GPU-enabled by cupy. In this sense, GPU is secondary to CPU.
'''
HAS_GPU=False
try:
    import cupy as cp
    HAS_GPU=True
except:
    import distpy.calc.cupy_facade as cp

import numpy
import scipy
from scipy import ndimage


class agnostic(object):

    # Don't use this in distpy projects...
    #def get_array_module(self,a):
    #    return cp.get_array_module(a)

    # This is used in place of cp.get_array_module(a)
    # because in the future we might also want get_scipy_ndimage (see below)
    def get_numpy(self,a):        
        return cp.get_array_module(a)


    # Currently none of the extra_numpy algorithms using ndimage
    # are GPU compatible.
    def get_scipy_ndimage(self,a):
        global HAS_GPU
        if HAS_GPU==True:
            return cp.get_array_module(a)
        else:
            return ndimage

    def asarray(self,a):
        return cp.asarray(a)

    def asnumpy(self,a):
        return cp.asnumpy(a)
