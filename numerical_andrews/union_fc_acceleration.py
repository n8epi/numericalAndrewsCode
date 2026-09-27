"""Validated search acceleration: FFT 2D filtering and reusable overlap factors.

Only used by the union-fitted search. Final results and checkpoints are
rescored with the unaccelerated reference implementation.
"""
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

def validate():
    """Reproduce the four-dataset full-map check on fixed-range winners."""
    import json
    import time
    from PIL import Image
    from .paths import RESULTS
    from .experiments.analyze_feature_congestion_alpha import fc_profile
    rows=[]
    for key in ['iris','diabetes','bc','bc_standardized']:
        image=np.asarray(Image.open(RESULTS/'optimized_mmqv'/f'{key}_best.png'))
        start=time.perf_counter(); original,left=fc_profile(image)
        slow=time.perf_counter()-start
        start=time.perf_counter()
        with union_accelerated_fc(): accelerated,right=fc_profile(image)
        fast=time.perf_counter()-start
        np.testing.assert_allclose(left,right,rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(list(original.values()),list(accelerated.values()),rtol=1e-7,atol=1e-8)
        rows.append(dict(dataset=key,seconds_original=slow,seconds_accelerated=fast,
                         max_map_error=float(np.max(np.abs(left-right))),
                         original=original,accelerated=accelerated))
        print(rows[-1],flush=True)
    destination=RESULTS/'union_fitted'
    destination.mkdir(exist_ok=True)
    (destination/'acceleration_validation.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    validate()
