# (C) 2020, Schlumberger. Refer to LICENSE.
'''
 cupy_facade - a do-nothing class for devices that don't have cupy installed

 This allows CPU/GPU agnostic code to be written. We engage with the GPU
 via the cuda_facade.
'''
import numpy

def get_array_module(x):
    return numpy


def asarray(x):
    return x

def asnumpy(x):
    return x
