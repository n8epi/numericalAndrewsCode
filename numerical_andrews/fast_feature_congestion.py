"""Scoped acceleration of the same full-resolution FC calculation.

Odd one-dimensional convolutions use ndimage; overlap normalizations are
cached; finite scale maps use vectorized maxima. Original scoring is restored
on exit and is used to validate each search checkpoint and published winner.
"""
from numerical_andrews.paths import RESULTS
from contextlib import contextmanager
import numpy as np
from scipy.ndimage import convolve1d
import visual_clutter.utils as u
import visual_clutter.clutter as c

@contextmanager
def accelerated_fc():
    original_conv = u.conv2
    saved = (u.conv2,c.conv2,u.RRoverlapconv,c.RRoverlapconv,c.Vlc.collapse)
    cache = {}
    def conv(x,y,mode=None):
        if mode == 'same' and y.ndim == 2 and 1 in y.shape and y.size % 2:
            return convolve1d(x,y.ravel(),axis=1 if y.shape[0]==1 else 0,
                              mode='constant',cval=0.)
        return original_conv(x,y,mode)
    def overlap(kernel,arr):
        key=(arr.shape,arr.dtype.str,kernel.shape,kernel.tobytes())
        if key not in cache:
            cache[key]=conv(np.ones_like(arr),kernel,'same')
        return np.sum(kernel)*conv(arr,kernel,'same')/cache[key]
    def collapse(self,levels):
        kernel=np.array([[.05,.25,.4,.25,.05]])
        kernel=conv(kernel,kernel.T)
        out=levels[0].copy()
        for scale in range(1,len(levels)):
            here=levels[scale]
            for _ in range(scale):
                here=c.pt.upConv(image=here,filt=kernel,edge_type='reflect1',step=[2,2],start=[0,0])
            h,w=min(out.shape[0],here.shape[0]),min(out.shape[1],here.shape[1])
            out[:h,:w]=np.maximum(out[:h,:w],here[:h,:w])
        return out
    u.conv2=c.conv2=conv
    u.RRoverlapconv=c.RRoverlapconv=overlap
    c.Vlc.collapse=collapse
    try:
        yield
    finally:
        u.conv2,c.conv2,u.RRoverlapconv,c.RRoverlapconv,c.Vlc.collapse=saved

if __name__=='__main__':
    import json,time
    from pathlib import Path
    from PIL import Image
    from numerical_andrews.experiments.analyze_feature_congestion_alpha import fc_profile
    rows=[]
    root=RESULTS
    for key in ['iris','bc','diabetes']:
        img=np.asarray(Image.open(root/'optimized_mmqv_initial'/f'{key}_best.png'))
        t=time.perf_counter(); p,m=fc_profile(img); slow=time.perf_counter()-t
        t=time.perf_counter()
        with accelerated_fc(): q,n=fc_profile(img)
        fast=time.perf_counter()-t
        np.testing.assert_allclose(list(p.values()),list(q.values()),rtol=1e-7,atol=1e-9)
        rows.append(dict(dataset=key,original=p,accelerated=q,seconds_original=slow,
                         seconds_accelerated=fast,max_map_abs_difference=float(np.max(np.abs(m-n)))))
        print(rows[-1],flush=True)
    (root/'fc_acceleration_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
