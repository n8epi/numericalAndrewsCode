"""Raw versus feature-standardized WDBC sensitivity, with separate MMQV searches.

Run --stage prepare, run and archive the 192-candidate preliminary search,
then expand_mmqv_optimization.py --datasets bc_standardized and --stage finalize.
See README.md for commands. Original raw-feature artifacts remain unchanged.
"""
import argparse,json,shutil
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_breast_cancer
from numerical_andrews.core import load_example_datasets,preprocessing_description,evaluate_mmqv_modes
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import ROOT,mmqv_transform,family_envelope
from numerical_andrews.experiments.expand_mmqv_optimization import SEEDS
from numerical_andrews.rendering import render_curves,VERTICAL_PADDING
from numerical_andrews.experiments.analyze_feature_congestion_alpha import (basis_cache_and_limit,fc_profile,build_landmarks,
    plot_profiles,build_fitted_display,plot_common_alpha_examples,truncation_validation)

KEY='bc_standardized'
OUT=ROOT/'standardized_breast_cancer'
METRICS=['mean_fc','upper_5pct_mean_fc','upper_1pct_mean_fc']

def dataset():
    return next(d for d in load_example_datasets(include_standardized=True) if d.key==KEY)

def prepare():
    OUT.mkdir(exist_ok=True)
    ds=dataset(); points=np.linspace(-.5,.5,4097); alphas=np.sort(np.append(np.logspace(-3,2,16),.2))
    _,bases,path_limit=basis_cache_and_limit(ds,points,alphas,256)
    limit=max(path_limit,VERTICAL_PADDING*family_envelope(ds.scores))
    trig=evaluate_mmqv_modes(31,points); history=[]
    for i,(angles,reflections,name) in enumerate([(np.zeros(15),np.ones(14),'cosine_first_seed'),(np.full(15,np.pi/2),-np.ones(14),'sine_first_seed')]):
        transform=mmqv_transform(30,angles,reflections,1)
        raster=render_curves(ds,(ds.scores@transform.T)@trig.T,points,limit)
        profile,_=fc_profile(raster)
        rec=dict(evaluation=i+1,stage=name,angles=angles.tolist(),reflections=reflections.astype(int).tolist(),constant_sign=1,**profile)
        history.append(rec); rec['best_mean_fc_so_far']=min(h['mean_fc'] for h in history)
    initial=dict(dataset=KEY,dataset_name=ds.display_name,dimension=30,status='complete',
         objective='mean_fc',seed=None,evaluations=2,evaluation_budget=2,history=history,
         best=min(history,key=lambda r:r['mean_fc']),y_limit=limit,
         all_mmqv_amplitude_bound=family_envelope(ds.scores),
         description='Two named starting bases only; no inherited optimized raw-feature candidate')
    (ROOT/'optimized_mmqv_initial'/f'{KEY}.json').write_text(json.dumps(initial,indent=2)+'\n')
    rows=[]
    for alpha in alphas:
        profile,_=fc_profile(render_curves(ds,ds.scores@bases[float(alpha)].T,points,limit))
        rows.append(dict(dataset=KEY,dataset_name=ds.display_name,observations=569,dimension=30,
                        alpha=float(alpha),cutoff_N=256,y_limit=limit,**profile))
        print('Scored standardized MMSSQV alpha',alpha,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'numerator_scores.csv',index=False)
    truncation_validation([ds],.001,256,512).to_csv(OUT/'truncation.csv',index=False)
    raw=load_breast_cancer(); centered=raw.data-raw.data.mean(axis=0)
    _,s,vt=np.linalg.svd(centered,full_matrices=False)
    metadata=dict(preprocessing=preprocessing_description(KEY),standard_deviation_ddof=0,
      observations=569,dimension=30,feature_names=raw.feature_names.tolist(),
      feature_means=raw.data.mean(axis=0).tolist(),feature_standard_deviations=raw.data.std(axis=0).tolist(),
      raw_pc1_variance_fraction=float(s[0]**2/s.dot(s)),
      standardized_pc1_variance_fraction=float(ds.singular_values[0]**2/ds.singular_values.dot(ds.singular_values)),
      raw_pc1_largest_loadings={raw.feature_names[i]:float(abs(vt[0,i])) for i in np.argsort(abs(vt[0]))[-3:][::-1]},
      alpha_values=alphas.tolist(),y_limit=limit,all_mmqv_amplitude_bound=family_envelope(ds.scores),
      rendering=json.loads((ROOT/'feature_congestion_alpha_metadata.json').read_text())['render'],
      seeds=SEEDS,budgets_per_seed=[240,480,960],full_resolution=True)
    metadata['rendering']['y_limit']=limit
    metadata['rendering']['coordinate_protocol']='fixed across all MMQV candidates and the MMSSQV path, bounded over the entire MMQV family'
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

def finalize():
    ds=dataset(); points=np.linspace(-.5,.5,4097)
    results=[json.loads((ROOT/'optimized_mmqv_expanded'/KEY/f'{seed}.json').read_text()) for seed in SEEDS]
    assert all(r['status']=='complete' and r['evaluations']==960 for r in results)
    initial=json.loads((ROOT/'optimized_mmqv_initial'/f'{KEY}.json').read_text())
    assert initial['evaluations']==192 and initial['status']=='complete'
    winner=min([initial]+results,key=lambda r:r['best']['mean_fc']); best=winner['best']
    assert best['mean_fc']<=initial['best']['mean_fc']
    raster=render_curves(ds,(ds.scores@mmqv_transform(30,best['angles'],best['reflections'],best['constant_sign']).T)@evaluate_mmqv_modes(31,points).T,points,winner['y_limit'])
    saved=(ROOT/'optimized_mmqv_initial'/f'{KEY}_best.png' if winner is initial else
           ROOT/'optimized_mmqv_expanded'/KEY/f'{winner["seed"]}_best.png')
    np.testing.assert_array_equal(raster,np.asarray(Image.open(saved)))
    profile,_=fc_profile(raster)
    np.testing.assert_allclose(list(profile.values()),[best[m] for m in METRICS],rtol=1e-13)
    winner.update(expanded_search_seeds=SEEDS,total_search_evaluations=3072,
                  total_original_checkpoint_validations=9,preprocessing=preprocessing_description(KEY))
    (ROOT/'optimized_mmqv'/f'{KEY}.json').write_text(json.dumps(winner,indent=2)+'\n')
    shutil.copyfile(saved,ROOT/'optimized_mmqv'/f'{KEY}_best.png')
    pd.DataFrame(winner['history']).to_csv(ROOT/'optimized_mmqv'/f'{KEY}_history.csv',index=False)
    standardized=pd.read_csv(OUT/'numerator_scores.csv'); standardized['baseline']='optimized'
    standardized['method']='MMSSQV'
    standardized['is_common_alpha']=np.isclose(standardized.alpha,.2,rtol=1e-12,atol=0)
    for m in METRICS:
        standardized['baseline_'+m]=profile[m]
        standardized[m+'_ratio']=standardized[m]/profile[m]
        standardized[m+'_percent_change']=100*(standardized[m+'_ratio']-1)
    raw=pd.read_csv(ROOT/'feature_congestion_alpha_profile.csv'); raw=raw[raw.dataset=='bc'].copy()
    raw['dataset_name']='Raw features'; standardized['dataset_name']='Standardized features'
    combined=pd.concat([raw,standardized],ignore_index=True)
    combined.to_csv(OUT/'profile.csv',index=False)
    build_landmarks(combined).to_csv(OUT/'landmarks.csv',index=False)
    pd.DataFrame([dict(dataset=KEY,y_limit=winner['y_limit'],**profile)]).to_csv(OUT/'baseline.csv',index=False)
    plot_profiles(combined,OUT/'comparison.png',OUT/'comparison.pdf')
    rawds=next(d for d in load_example_datasets() if d.key=='bc')
    displays=[build_fitted_display(replace(rawds,display_name='Raw features'),points,256),
              build_fitted_display(replace(ds,display_name='Standardized'),points,256)]
    plot_common_alpha_examples(displays,OUT/'examples.png',OUT/'examples.pdf')
    display_rows=[]
    for display in displays:
        for name,limits,values in [('MMQV',display.baseline_y_limit,display.baseline_profile),('MMSSQV',display.smoothed_y_limit,display.smoothed_profile)]:
            display_rows.append(dict(dataset=display.dataset.key,method=name,y_min=limits[0],y_max=limits[1],**values))
    pd.DataFrame(display_rows).to_csv(OUT/'display_profiles.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1)); checkpoint_rows=[]
    named=min(r['mean_fc'] for r in initial['history'][:2])
    axes[0].axhline(initial['best']['mean_fc']/named,color='black',ls='--',lw=.8,label='Initial search')
    for r in results:
        axes[0].plot(np.arange(1,961)/15,np.array([h['best_mean_fc_so_far'] for h in r['history']])/named,label=str(r['seed']))
        for cp in r['checkpoints']:
            checkpoint_rows.append(dict(seed=r['seed'],budget=cp['budget'],**{m:cp['best'][m] for m in METRICS}))
    axes[0].set(xlabel='Evaluations per phase angle',ylabel='Best mean FC / better named seed')
    axes[0].legend(fontsize=7)
    landmark=build_landmarks(standardized).set_index('landmark').loc['minimax_three_number_profile']
    numerator=standardized[np.isclose(standardized.alpha,landmark.alpha)].iloc[0]
    budget_rows=[]
    for stage,effort in enumerate([0,240,480,960]):
        choices=[initial['best']]
        if stage: choices += [r['checkpoints'][stage-1]['best'] for r in results]
        chosen=min(choices,key=lambda r:r['mean_fc'])
        budget_rows.append(dict(budget_per_run=effort,alpha=landmark.alpha,**{m+'_ratio':numerator[m]/chosen[m] for m in METRICS}))
    budget=pd.DataFrame(budget_rows)
    for m,label in zip(METRICS,['Mean','Upper 5%','Upper 1%']): axes[1].plot(budget.budget_per_run/15,budget[m+'_ratio'],marker='o',label=label)
    axes[1].axhline(1,color='gray',lw=.7); axes[1].legend(fontsize=7)
    axes[1].set(xlabel='Evaluations per angle, per run',ylabel='MMSSQV / retained MMQV',title=f'Fixed alpha = {landmark.alpha:g}')
    for ax in axes: ax.grid(alpha=.2); ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(OUT/'search.pdf'); fig.savefig(OUT/'search.png',dpi=300);plt.close(fig)
    pd.DataFrame(checkpoint_rows).to_csv(OUT/'independent_runs.csv',index=False);budget.to_csv(OUT/'budget_sensitivity.csv',index=False)
    metadata=json.loads((OUT/'metadata.json').read_text())
    vals=[r['best']['mean_fc'] for r in results]
    metadata.update(winning_seed=winner['seed'],best_scores=profile,independent_final_means=vals,
        independent_spread_percent=100*(max(vals)/min(vals)-1),
        improvement_over_named_percent=100*(1-profile['mean_fc']/named),
        initial_search_seed=initial["seed"],initial_search_budget=192,total_search_candidates=3072,
        improvement_over_initial_percent=100*(1-profile['mean_fc']/initial['best']['mean_fc']),
        max_checkpoint_score_error=max(abs(e) for r in results for cp in r['checkpoints'] for e in cp['original_minus_accelerated'].values()),
        alpha_improving_all_three=standardized.loc[(standardized[[m+'_ratio' for m in METRICS]]<1).all(axis=1),'alpha'].tolist(),
        status='complete',global_optimality_claim=False)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(build_landmarks(combined).to_string(index=False));print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','finalize'],required=True)
    args=parser.parse_args(); prepare() if args.stage=='prepare' else finalize()
