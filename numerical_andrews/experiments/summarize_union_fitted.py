"""Validate completed union-fitted searches and generate publication artifacts."""
import json
from numerical_andrews.tables import write_fitted_table
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import mmqv_transform
from numerical_andrews.experiments.compare_union_fitted import OUT, ROOT, METRICS, SEEDS, context, dump
from numerical_andrews.core import spatial_spectral_modes, evaluate_spatial_spectral_modes
from numerical_andrews.rendering import render_curves, VERTICAL_PADDING
from numerical_andrews.experiments.analyze_feature_congestion_alpha import fc_profile

NAMES={'iris':'Iris','diabetes':'Diabetes','bc':'Breast cancer (raw)',
       'bc_standardized':'Breast cancer (standardized)'}

def main():
    from numerical_andrews.experiments.audit_union_fitted import main as audit_candidates
    audit_candidates()
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=[];compression=[];validation=[];runs=[];displays=[];budget_rows=[]
    contexts={k:context(k) for k in NAMES}
    winners={}
    scorer_pool=ProcessPoolExecutor(max_workers=4)
    for rec in protocol['records']:
        key=rec['dataset'];group=rec['group'];ds,points,curves,named=contexts[key]
        if (key,group) not in winners:
            candidates=[]
            for seed in SEEDS:
                path=OUT/'search'/key/group/f'{seed}.json'
                obj=json.loads(path.read_text());assert obj['status']=='complete',path
                assert obj['evaluations']==64*(ds.values.shape[1]//2)
                candidates.append(obj)
                runs.append(dict(dataset=key,group=group,seed=seed,evaluations=obj['evaluations'],
                    mean_fc=obj['best']['mean_fc'],y_limit=obj['best']['y_limit']))
            for index,budget in enumerate(candidates[0]['stages']):
                values=[x['checkpoints'][index]['best']['mean_fc'] for x in candidates]
                budget_rows.append(dict(dataset=key,group=group,budget_per_run=budget,
                    best_mean_fc=min(values),worst_mean_fc=max(values),
                    relative_seed_spread=max(values)/min(values)-1))
            chosen=min(candidates,key=lambda x:x['best']['mean_fc'])
            winners[key,group]=chosen
        chosen=winners[key,group];best=chosen['best']
        modes,_=spatial_spectral_modes(rec['alpha'],256,ds.values.shape[1])
        mmssqv=ds.scores@evaluate_spatial_spectral_modes(modes,points).T
        old=json.loads((ROOT/'optimized_mmqv'/f'{key}.json').read_text())
        methods={'cosine_first':curves(named[0]),'sine_first':curves(named[1]),
                 'optimized':curves(best),'mmssqv':mmssqv}
        limit=VERTICAL_PADDING*max(float(np.abs(y).max()) for y in methods.values())
        assert np.isclose(limit,best['y_limit'],rtol=1e-14,atol=0)
        profiles={};maps={}
        images={method:render_curves(ds,y,points,limit) for method,y in methods.items()}
        pending={method:scorer_pool.submit(fc_profile,image) for method,image in images.items()}
        for method in methods:
            profile,local_map=pending[method].result()
            profiles[method]=profile;maps[method]=local_map
            if method=='optimized':
                np.testing.assert_allclose([profile[k] for k in METRICS],[best[k] for k in METRICS],rtol=1e-7,atol=1e-8)
                transform=mmqv_transform(
                    ds.values.shape[1],best['angles'],best['reflections'],best['constant_sign'])
                gram=float(np.max(np.abs(transform.T@transform-np.eye(ds.values.shape[1]))))
                assert gram<1e-12
                validation.append(dict(dataset=key,alpha=rec['alpha'],gram_error=gram,
                    mean_rescore_error=profile['mean_fc']-best['mean_fc']))
        fixed_y=curves(old['best']); fixed_limit=old['y_limit']
        if rec['alpha_index']==0:
            compression.append(dict(dataset=key,old_y_limit=fixed_limit,
               optimized_peak=float(np.abs(fixed_y).max()),
               optimized_fraction=float(np.abs(fixed_y).max())/fixed_limit,
               cosine_fraction=float(np.abs(methods['cosine_first']).max())/fixed_limit,
               sine_fraction=float(np.abs(methods['sine_first']).max())/fixed_limit))
        # Re-score the previous fixed-scale winner on this row's shared range,
        # enlarging only this diagnostic comparison if required would be invalid.
        # Instead record its own matched union range in a separate profile.
        old_limit=VERTICAL_PADDING*max(rec['range_floor'],float(np.abs(fixed_y).max()))
        old_profile,_=fc_profile(render_curves(ds,fixed_y,points,old_limit))
        row=dict(**rec,observations=ds.values.shape[0],dimension=ds.values.shape[1],y_limit=limit,selected_seed=chosen['seed'],
                 fixed_winner_refitted_y_limit=old_limit,
                 fixed_winner_refitted_mean_fc=old_profile['mean_fc'])
        for method,profile in profiles.items():
            for metric,value in profile.items(): row[f'{method}_{metric}']=value
        retained=min(['optimized','cosine_first','sine_first'],key=lambda name:profiles[name]['mean_fc'])
        row['retained_baseline']=retained
        assert profiles[retained]['mean_fc']<=min(profiles[n]['mean_fc'] for n in ['cosine_first','sine_first'])
        for metric in METRICS:
            row[f'retained_{metric}']=profiles[retained][metric]
            row[f'{metric}_ratio']=profiles['mmssqv'][metric]/profiles[retained][metric]
        for name in ['cosine_first','sine_first']:
            row[f'mean_ratio_to_{name}']=profiles['mmssqv']['mean_fc']/profiles[name]['mean_fc']
        row['optimized_to_better_named_mean']=profiles['optimized']['mean_fc']/min(profiles[n]['mean_fc'] for n in ['cosine_first','sine_first'])
        row['max_ratio']=max(row[k+'_ratio'] for k in METRICS)
        row['all_three_below_one']=row['max_ratio']<1
        row['is_common_alpha']=np.isclose(rec['alpha'],.2,rtol=1e-14,atol=0)
        rows.append(row)
        if row['is_common_alpha']:
            images['retained']=images[retained];maps['retained']=maps[retained];profiles['retained']=profiles[retained]
            displays.append((key,limit,images,maps,profiles))
            for name,im in images.items():Image.fromarray(im).save(OUT/f'{key}_{name}_alpha_0p2.png')
        print('Validated',key,rec['alpha'],tuple(round(row[k+'_ratio'],5) for k in METRICS),flush=True)
    scorer_pool.shutdown()
    df=pd.DataFrame(rows);df.to_csv(OUT/'profiles.csv',index=False)
    pd.DataFrame(compression).to_csv(OUT/'compression.csv',index=False)
    pd.DataFrame(runs).to_csv(OUT/'independent_runs.csv',index=False)
    pd.DataFrame(budget_rows).to_csv(OUT/'budget_sensitivity.csv',index=False)
    dump(OUT/'validation.json',validation)
    chosen_rows=[]
    for key in NAMES:
        part=df[df.dataset==key]
        for selection,row in [('common',part[part.is_common_alpha].iloc[0]),('minimax',part.loc[part.max_ratio.idxmin()])]:
            chosen_rows.append(dict(selection=selection,**row.to_dict()))
    landmarks=pd.DataFrame(chosen_rows);landmarks.to_csv(OUT/'landmarks.csv',index=False)
    plot_paths(df);plot_examples(displays);write_fitted_table(landmarks, OUT/'table.tex')
    summary={key:dict(named_better_at_final_range_alphas=df[(df.dataset==key)&(df.optimized_to_better_named_mean>1+1e-7)].alpha.tolist(),all_three_alphas=df[(df.dataset==key)&df.all_three_below_one].alpha.tolist(),
                    common=landmarks[(landmarks.dataset==key)&(landmarks.selection=='common')].to_dict('records')[0],
                    minimax=landmarks[(landmarks.dataset==key)&(landmarks.selection=='minimax')].to_dict('records')[0]) for key in NAMES}
    dump(OUT/'summary.json',summary)
    source_dir=Path(__file__).resolve().parents[1]
    drivers=['experiments/compare_union_fitted.py','experiments/summarize_union_fitted.py',
             'experiments/audit_union_fitted.py','experiments/analyze_feature_congestion_alpha.py',
             'experiments/optimize_mmqv_feature_congestion.py',
             'union_fc_acceleration.py','fast_feature_congestion.py','congestion.py',
             'rendering.py','core.py']
    dump(OUT/'final_metadata.json',dict(
        retained_baseline_rule='minimum mean FC among range-setting candidate, cosine-first, sine-first at final shared range',
        canvas_pixels=[768,512],dpi=100,line_width_points=.65,line_alpha=.52,
        parameter_interval=[-.5,.5],vertical_padding_factor=1.05,
        source_sha256={name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in drivers},
        data_sha256={key:hashlib.sha256(np.ascontiguousarray(ctx[0].values).tobytes()).hexdigest() for key,ctx in contexts.items()}))

def plot_paths(df):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1),sharex=True)
    for ax,key in zip(axes.flat,NAMES):
        part=df[df.dataset==key].sort_values('alpha')
        for metric,label,color in zip(METRICS,['Mean','Upper 5%','Upper 1%'],['#2468a0','#c57719','#973e75']):
            ax.semilogx(part.alpha,part[metric+'_ratio'],'.-',color=color,label=label,lw=1.2,ms=3)
        ax.semilogx(part.alpha,part.mean_ratio_to_cosine_first,':',color='#73814b',lw=1,label='Mean / cosine-first')
        ax.semilogx(part.alpha,part.mean_ratio_to_sine_first,'--',color='#757575',lw=1,label='Mean / sine-first')
        ax.axhline(1,color='black',lw=.65);ax.axvline(.2,color='#cccccc',lw=.8)
        ax.set_title(NAMES[key],fontsize=9);ax.grid(alpha=.15);ax.tick_params(labelsize=8)
        ax.set_ylabel('MMSSQV / MMQV',fontsize=8);ax.set_xlabel(r'$\alpha$',fontsize=9)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7,frameon=False)
    fig.tight_layout(rect=(0,.09,1,1))
    for suffix in ['pdf','png']:fig.savefig(OUT/f'ratios.{suffix}',dpi=200)
    plt.close(fig)

def plot_examples(displays):
    fig,axes=plt.subplots(4,4,figsize=(7.2,5.8),squeeze=False)
    for i,(key,limit,images,maps,profiles) in enumerate(displays):
        vmax=float(np.quantile(np.concatenate([maps['retained'].ravel(),maps['mmssqv'].ravel()]),.995))
        for j,(method,is_map) in enumerate([('retained',False),('mmssqv',False),('retained',True),('mmssqv',True)]):
            ax=axes[i,j]
            if is_map:ax.imshow(maps[method],cmap='magma',vmin=0,vmax=vmax,aspect='equal')
            else:ax.imshow(images[method],aspect='equal')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if i==0:ax.set_title(['Retained MMQV','MMSSQV','MMQV congestion','MMSSQV congestion'][j],fontsize=8)
            if j==0:ax.set_ylabel({'iris':'Iris','diabetes':'Diabetes','bc':'WDBC (raw)','bc_standardized':'WDBC (std.)'}[key]+f'\n±{limit:.3g}',fontsize=8)
            if is_map:ax.set_xlabel(' / '.join(f'{profiles[method][m]:.2f}' for m in METRICS),fontsize=7)
    fig.tight_layout(pad=.6)
    for suffix in ['pdf','png']:fig.savefig(OUT/f'examples.{suffix}',dpi=200)
    plt.close(fig)

if __name__=='__main__':main()
