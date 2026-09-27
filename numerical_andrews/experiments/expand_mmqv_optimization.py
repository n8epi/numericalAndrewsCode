"""Independent, dimension-scaled MMQV searches with joint toroidal DE moves.

Three seeds, checkpoints at 16/32/64 evaluations per phase angle. Each stage
uses population search followed by mixed joint/coordinate/discrete refinement.
All objective evaluations use the original full-resolution rendering protocol.
Accelerated arithmetic is checked against the original scorer at checkpoints.
"""
from __future__ import annotations
import argparse,json,time,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import ROOT,mmqv_transform,family_envelope
from numerical_andrews.core import load_example_datasets,evaluate_mmqv_modes
from numerical_andrews.rendering import render_curves
from numerical_andrews.experiments.analyze_feature_congestion_alpha import fc_profile
from numerical_andrews.fast_feature_congestion import accelerated_fc

SEEDS=[20260924,20260925,20260926]

def search(key,seed):
    ds=next(x for x in load_example_datasets(include_standardized=True) if x.key==key)
    old=json.loads((ROOT/'optimized_mmqv_initial'/f'{key}.json').read_text())
    d=ds.values.shape[1]; p=d//2; h=(d-1)//2
    stages=[16*p,32*p,64*p]; budget=stages[-1]
    points=np.linspace(-.5,.5,4097); trig=evaluate_mmqv_modes(d+1,points)
    limit=old['y_limit']; assert 1.05*family_envelope(ds.scores)<=limit
    rng=np.random.default_rng(seed); history=[]; checkpoints=[]; best=None
    directory=ROOT/'optimized_mmqv_expanded'/key; directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{seed}.json'; start=time.perf_counter()
    def raster(r):
        t=mmqv_transform(d,r['angles'],r['reflections'],r['constant_sign'])
        return render_curves(ds,(ds.scores@t.T)@trig.T,points,limit)
    def save(status):
        result=dict(dataset=key,dataset_name=ds.display_name,dimension=d,seed=seed,
          objective='mean_fc',evaluation_budget=budget,evaluations=len(history),
          status=status,global_optimality_claim=False,y_limit=limit,
          previous_y_limit=limit,all_mmqv_amplitude_bound=family_envelope(ds.scores),
          curve_grid_points=4097,canvas_pixels=[768,512],best=best,history=history,
          checkpoints=checkpoints,elapsed_seconds=time.perf_counter()-start,
          algorithm='Toroidal differential evolution plus joint/coordinate/discrete refinement',
          stages=stages,population_size=max(12,2*p),
          validation_evaluations=len(checkpoints),
          stopping_rule='Fixed independent budget of 64 evaluations per phase angle')
        path.write_text(json.dumps(result,indent=2)+'\n')
    def evaluate(a,r,s,kind):
        nonlocal best
        rec=dict(evaluation=len(history)+1,stage=kind,angles=(np.asarray(a)%(2*np.pi)).tolist(),
                 reflections=np.asarray(r,dtype=int).tolist(),constant_sign=int(s))
        # Original named scores are shared deterministic inputs, not inherited optimized solutions.
        if len(history)<2:
            profile={k:old['history'][len(history)][k] for k in ['mean_fc','upper_5pct_mean_fc','upper_1pct_mean_fc']}
        else:
            with accelerated_fc(): profile,_=fc_profile(raster(rec))
        rec.update(profile)
        if best is None or rec['mean_fc']<best['mean_fc']: best=rec.copy()
        rec['best_mean_fc_so_far']=best['mean_fc']; rec['elapsed_seconds']=time.perf_counter()-start
        history.append(rec)
        if len(history)%16==0: save('running')
        return rec
    pop=[evaluate(np.zeros(p),np.ones(h),1,'cosine_first_seed'),
         evaluate(np.full(p,np.pi/2),-np.ones(h),1,'sine_first_seed')]
    while len(pop)<max(12,2*p):
        pop.append(evaluate(rng.uniform(0,2*np.pi,p),rng.choice([-1,1],h),rng.choice([-1,1]),'independent_random_start'))
    for stage,end in enumerate(stages):
        de_end=len(history)+int(.75*(end-len(history)))
        iteration=0
        while len(history)<de_end:
            target=iteration%len(pop); iteration+=1
            ids=rng.choice([i for i in range(len(pop)) if i!=target],3,replace=False)
            x,y,z=[pop[i] for i in ids]
            if rng.random()<.5: x=min(pop,key=lambda r:r['mean_fc'])
            delta=(np.array(y['angles'])-z['angles']+np.pi)%(2*np.pi)-np.pi
            donor=np.array(x['angles'])+rng.uniform(.5,1.)*delta
            mask=rng.random(p)<.8; mask[rng.integers(p)]=True
            a=np.where(mask,donor,pop[target]['angles'])
            r=np.where(rng.random(h)<.5,x['reflections'],pop[target]['reflections'])
            r=r*np.where(rng.random(h)<1/max(2,h),-1,1)
            s=x['constant_sign'] if rng.random()<.5 else pop[target]['constant_sign']
            if rng.random()<.1: s=-s
            if rng.random()<.1:
                a=rng.uniform(0,2*np.pi,p); r=rng.choice([-1,1],h); s=rng.choice([-1,1])
            trial=evaluate(a,r,s,f'stage_{stage+1}_joint_population')
            if trial['mean_fc']<pop[target]['mean_fc']: pop[target]=trial
        local_start=len(history)
        while len(history)<end:
            a=np.array(best['angles']); r=np.array(best['reflections']); s=best['constant_sign']
            frac=(len(history)-local_start)/max(1,end-local_start)
            scale=(np.pi/2)*(.02**frac)
            choice=rng.random()
            if choice<.2:
                j=rng.integers(h+1)
                if j==h: s=-s
                else: r[j]*=-1
            elif choice<.6:
                a+=rng.normal(size=p)*scale/np.sqrt(p)
            else: a[rng.integers(p)]+=rng.choice([-1,1])*scale
            trial=evaluate(a,r,s,f'stage_{stage+1}_mixed_refinement')
            worst=max(range(len(pop)),key=lambda i:pop[i]['mean_fc'])
            if trial['mean_fc']<pop[worst]['mean_fc']: pop[worst]=trial
        original,_=fc_profile(raster(best))
        errors={k:original[k]-best[k] for k in original}
        np.testing.assert_allclose(list(original.values()),[best[k] for k in original],rtol=1e-7,atol=1e-8)
        checkpoints.append(dict(budget=end,best={**best,**original},original_minus_accelerated=errors))
        print(f'{key} seed {seed}: {end}/{budget}, mean {original["mean_fc"]:.9f}',flush=True)
        save('running')
    best=checkpoints[-1]['best']
    Image.fromarray(raster(best)).save(directory/f'{seed}_best.png')
    save('complete')
    return key,seed,best['mean_fc']

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=6)
    parser.add_argument("--datasets", nargs="+", default=["iris","diabetes","bc"],
                        choices=["iris","diabetes","bc","bc_standardized"])
    args=parser.parse_args()
    archive=ROOT/'optimized_mmqv_initial'
    if not archive.exists(): shutil.copytree(ROOT/'optimized_mmqv',archive)
    jobs=[]
    for key in args.datasets:
        for seed in SEEDS:
            path=ROOT/'optimized_mmqv_expanded'/key/f'{seed}.json'
            if not path.exists() or json.loads(path.read_text())['status']!='complete': jobs.append((key,seed))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(search,*job) for job in jobs]
        for future in futures: print('COMPLETE',future.result(),flush=True)

if __name__=='__main__': main()
