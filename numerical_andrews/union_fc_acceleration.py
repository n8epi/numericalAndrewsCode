"""Validated search acceleration: FFT 2D filtering and reusable overlap factors.

Used by the median-congestion search. Selected best and worst candidates
are rescored with the unaccelerated reference implementation.
"""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
from contextlib import contextmanager
import numpy as np
from scipy.signal import fftconvolve
import visual_clutter.utils as u
import visual_clutter.clutter as c
from .fast_feature_congestion import accelerated_fc
_OVERLAPS={}

@contextmanager
def union_accelerated_fc():
    with accelerated_fc():
        saved=(u.conv2,c.conv2,u.RRoverlapconv,c.RRoverlapconv)
        previous=u.conv2
        def conv(x,y,mode=None):
            if x.size>4096 and y.ndim==2 and min(y.shape)>1:
                if mode=='same':
                    return np.rot90(fftconvolve(np.rot90(x,2),np.rot90(y,2),mode='same'),2)
                return fftconvolve(x,y,mode='full')
            return previous(x,y,mode)
        def overlap(kernel,arr):
            key=(arr.shape,arr.dtype.str,kernel.shape,kernel.tobytes())
            if key not in _OVERLAPS:
                _OVERLAPS[key]=conv(np.ones_like(arr),kernel,'same')
            return np.sum(kernel)*conv(arr,kernel,'same')/_OVERLAPS[key]
        u.conv2=c.conv2=conv
        u.RRoverlapconv=c.RRoverlapconv=overlap
        try:yield
        finally:u.conv2,c.conv2,u.RRoverlapconv,c.RRoverlapconv=saved
