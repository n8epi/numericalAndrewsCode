"""Per-alpha union-fitted MMQV searches and matched-scale MMSSQV comparisons.

The range is 1.05 times the largest sampled absolute ordinate of MMSSQV,
both named MMQV bases, and the candidate. This deterministic fitting rule is
part of the objective. Searches optimize baseline mean FC, never a ratio.
Identical objectives (MMSSQV peaks below the named-basis envelope) share runs.
"""
from __future__ import annotations
import argparse
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from numerical_andrews.core import (load_example_datasets, evaluate_mmqv_modes,
    evaluate_spatial_spectral_modes, spatial_spectral_modes)
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import ROOT, mmqv_transform
from numerical_andrews.rendering import render_curves, VERTICAL_PADDING
from numerical_andrews.experiments.analyze_feature_congestion_alpha import fc_profile
from numerical_andrews.union_fc_acceleration import union_accelerated_fc

OUT=ROOT/'union_fitted'
ALPHAS=sorted(set(np.logspace(-3,2,16).tolist()+[.2]))
SEEDS=[20260924,20260925,20260926]
METRICS=['mean_fc','upper_5pct_mean_fc','upper_1pct_mean_fc']

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(f'.tmp-{os.getpid()}');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def context(key):
    ds=next(x for x in load_example_datasets(include_standardized=True) if x.key==key)
    d=ds.values.shape[1];points=np.linspace(-.5,.5,4097)
    trig=evaluate_mmqv_modes(d+1,points)
    def curves(r):
        transform=mmqv_transform(d,r['angles'],r['reflections'],r['constant_sign'])
        return (ds.scores@transform.T)@trig.T
    p=d//2;h=(d-1)//2
    named=[dict(angles=np.zeros(p).tolist(),reflections=np.ones(h,dtype=int).tolist(),constant_sign=1),
           dict(angles=np.full(p,np.pi/2).tolist(),reflections=(-np.ones(h,dtype=int)).tolist(),constant_sign=1)]
    return ds,points,curves,named

def prepare():
    jobs=[];records=[]
    for key in ['iris','diabetes','bc','bc_standardized']:
        ds,points,curves,named=context(key)
        named_peak=max(float(np.abs(curves(r)).max()) for r in named)
        groups={}
        for index,alpha in enumerate(ALPHAS):
            modes,_=spatial_spectral_modes(alpha,256,ds.values.shape[1])
            yy=ds.scores@evaluate_spatial_spectral_modes(modes,points).T
            peak=float(np.abs(yy).max());floor=max(peak,named_peak)
            # Exact equal floors imply the same objective at every candidate.
            group=groups.setdefault(floor,f'g{len(groups):02d}')
            records.append(dict(dataset=key,alpha_index=index,alpha=alpha,group=group,
                mmssqv_peak=peak,named_peak=named_peak,range_floor=floor))
        for floor,group in groups.items():
            for seed in SEEDS: jobs.append((key,group,floor,seed))
    dump(OUT/'protocol.json',dict(protocol='per-alpha union of MMSSQV, cosine-first, sine-first, candidate',
        padding=VERTICAL_PADDING,grid_points=4097,cutoff=256,alphas=ALPHAS,seeds=SEEDS,
        objective='candidate mean FC under deterministic union fitting; not ratio',
        records=records,groups=len(jobs)//3,global_optimality_claim=False))
    return jobs

def search(key,group,floor,seed,budget_per_angle):
    path=OUT/'search'/key/group/f'{seed}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['status']=='complete' and saved['budget_per_angle']==budget_per_angle:
            return key,group,seed,saved['best']['mean_fc']
    ds,points,curves,named=context(key)
    d=ds.values.shape[1];p=d//2;h=(d-1)//2
    stages=[16*p,32*p,64*p];stages=[x for x in stages if x<budget_per_angle*p]+[budget_per_angle*p]
    rng=np.random.default_rng(seed);history=[];checkpoints=[];best=None;start=time.perf_counter()
    def image(r):
        yy=curves(r);limit=VERTICAL_PADDING*max(floor,float(np.abs(yy).max()))
        return render_curves(ds,yy,points,limit),limit
    def save(status):
        dump(path,dict(dataset=key,group=group,range_floor=floor,seed=seed,
             budget_per_angle=budget_per_angle,evaluations=len(history),status=status,
             best=best,history=history,checkpoints=checkpoints,stages=stages,
             elapsed_seconds=time.perf_counter()-start,global_optimality_claim=False,
             accelerated_scorer='union_fc_acceleration:FFT 2D filters and cached overlap factors',
             initialization='two named bases and previous fixed-scale best, followed by independent random population'))
    def evaluate(a,r,s,kind):
        nonlocal best
        rec=dict(evaluation=len(history)+1,stage=kind,angles=(np.asarray(a)%(2*np.pi)).tolist(),
                 reflections=np.asarray(r,dtype=int).tolist(),constant_sign=int(s))
        im,limit=image(rec)
        with union_accelerated_fc(): profile,_=fc_profile(im)
        rec.update(profile);rec['y_limit']=limit
        if best is None or rec['mean_fc']<best['mean_fc']: best=rec.copy()
        rec['best_mean_fc_so_far']=best['mean_fc'];history.append(rec)
        if len(history)%32==0:save('running')
        return rec
    pop=[evaluate(r['angles'],r['reflections'],r['constant_sign'],name)
         for r,name in zip(named,['cosine_first','sine_first'])]
    old=json.loads((ROOT/'optimized_mmqv'/f'{key}.json').read_text())['best']
    pop.append(evaluate(old['angles'],old['reflections'],old['constant_sign'],'fixed_scale_warm_start'))
    while len(pop)<max(12,2*p):
        pop.append(evaluate(rng.uniform(0,2*np.pi,p),rng.choice([-1,1],h),rng.choice([-1,1]),'random_start'))
    for stage,end in enumerate(stages):
        de_end=len(history)+int(.75*(end-len(history)));iteration=0
        while len(history)<de_end:
            target=iteration%len(pop);iteration+=1
            ids=rng.choice([i for i in range(len(pop)) if i!=target],3,replace=False)
            x,y,z=[pop[i] for i in ids]
            if rng.random()<.5:x=min(pop,key=lambda r:r['mean_fc'])
            delta=(np.array(y['angles'])-z['angles']+np.pi)%(2*np.pi)-np.pi
            donor=np.array(x['angles'])+rng.uniform(.5,1.)*delta
            mask=rng.random(p)<.8;mask[rng.integers(p)]=True
            a=np.where(mask,donor,pop[target]['angles'])
            r=np.where(rng.random(h)<.5,x['reflections'],pop[target]['reflections'])
            r=r*np.where(rng.random(h)<1/max(2,h),-1,1)
            s=x['constant_sign'] if rng.random()<.5 else pop[target]['constant_sign']
            if rng.random()<.1:s=-s
            if rng.random()<.1:a=rng.uniform(0,2*np.pi,p);r=rng.choice([-1,1],h);s=rng.choice([-1,1])
            trial=evaluate(a,r,s,f'stage_{stage+1}_population')
            if trial['mean_fc']<pop[target]['mean_fc']:pop[target]=trial
        local_start=len(history)
        while len(history)<end:
            a=np.array(best['angles']);r=np.array(best['reflections']);s=best['constant_sign']
            frac=(len(history)-local_start)/max(1,end-local_start);scale=(np.pi/2)*(.02**frac)
            choice=rng.random()
            if choice<.2:
                j=rng.integers(h+1)
                if j==h:s=-s
                else:r[j]*=-1
            elif choice<.6:a+=rng.normal(size=p)*scale/np.sqrt(p)
            else:a[rng.integers(p)]+=rng.choice([-1,1])*scale
            trial=evaluate(a,r,s,f'stage_{stage+1}_refinement')
            worst=max(range(len(pop)),key=lambda i:pop[i]['mean_fc'])
            if trial['mean_fc']<pop[worst]['mean_fc']:pop[worst]=trial
        im,_=image(best);original,_=fc_profile(im)
        errors={k:original[k]-best[k] for k in METRICS}
        np.testing.assert_allclose([original[k] for k in METRICS],[best[k] for k in METRICS],rtol=1e-7,atol=1e-8)
        best.update(original);checkpoints.append(dict(budget=end,best=best.copy(),original_minus_accelerated=errors))
        save('running')
    Image.fromarray(image(best)[0]).save(path.with_suffix('.png'))
    save('complete')
    return key,group,seed,best['mean_fc']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--budget-per-angle',type=int,default=64,choices=[16,32,64])
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    jobs=prepare();print(f'{len(jobs)} searches, {len(jobs)//3} distinct range objectives',flush=True)
    if args.prepare_only:return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(search,*job,args.budget_per_angle) for job in jobs]
        for f in as_completed(futures):print('COMPLETE',f.result(),flush=True)
if __name__=='__main__':main()
