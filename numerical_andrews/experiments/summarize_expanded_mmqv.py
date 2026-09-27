"""Promote validated best candidates and report independent-run/budget sensitivity."""
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import ROOT
from numerical_andrews.experiments.expand_mmqv_optimization import SEEDS

METRICS=['mean_fc','upper_5pct_mean_fc','upper_1pct_mean_fc']

def main():
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.5))
    summaries=[]; diagnostics=[]; runs=[]
    source=pd.read_csv(ROOT/'feature_congestion_two_baseline_profile.csv')
    for col,key in enumerate(['iris','bc','diabetes']):
        old=json.loads((ROOT/'optimized_mmqv_initial'/f'{key}.json').read_text())
        results=[json.loads((ROOT/'optimized_mmqv_expanded'/key/f'{seed}.json').read_text()) for seed in SEEDS]
        assert all(r['status']=='complete' for r in results)
        candidates=[(old,ROOT/'optimized_mmqv_initial'/f'{key}_best.png')]
        candidates.extend((r,ROOT/'optimized_mmqv_expanded'/key/f'{r["seed"]}_best.png') for r in results)
        winner,img=min(candidates,key=lambda item:item[0]['best']['mean_fc'])
        canonical={**winner,'expanded_search_seeds':SEEDS,'initial_search_archive':'optimized_mmqv_initial',
          'expanded_search_records':f'optimized_mmqv_expanded/{key}/',
          'total_search_evaluations':old['evaluations']+sum(r['evaluations'] for r in results),
          'total_original_checkpoint_validations':sum(r['validation_evaluations'] for r in results)}
        (ROOT/'optimized_mmqv'/f'{key}.json').write_text(json.dumps(canonical,indent=2)+'\n')
        shutil.copyfile(img,ROOT/'optimized_mmqv'/f'{key}_best.png')
        pd.DataFrame(canonical['history']).to_csv(ROOT/'optimized_mmqv'/f'{key}_history.csv',index=False)
        p=old['dimension']//2; named=min(x['mean_fc'] for x in old['history'][:2]); best=winner['best']
        seedvals=[r['best']['mean_fc'] for r in results]
        summaries.append(dict(dataset=key,dataset_name=old['dataset_name'],angles=p,
          evaluations_per_run=64*p,additional_search_evaluations=192*p,
          total_search_evaluations=canonical['total_search_evaluations'],winning_seed=winner['seed'],
          **{k:best[k] for k in METRICS},initial_mean_fc=old['best']['mean_fc'],
          improvement_over_initial_percent=100*(1-best['mean_fc']/old['best']['mean_fc']),
          improvement_over_best_named_percent=100*(1-best['mean_fc']/named),
          independent_best_mean=min(seedvals),independent_worst_mean=max(seedvals),
          independent_spread_percent=100*(max(seedvals)/min(seedvals)-1)))
        ax=axes[0,col]
        for r in results:
            vals=np.array([v['best_mean_fc_so_far'] for v in r['history']])
            assert np.all(np.diff(vals)<=0)
            ax.plot(np.arange(1,len(vals)+1)/p,vals/named,lw=1,label=str(r['seed'])[-2:])
            for cp in r['checkpoints']:
                runs.append(dict(dataset=key,seed=r['seed'],budget=cp['budget'],
                                 budget_per_angle=cp['budget']/p,**{k:cp['best'][k] for k in METRICS}))
        ax.axhline(old['best']['mean_fc']/named,color='black',ls='--',lw=.8,label='Initial search')
        ax.set_title(old['dataset_name'],fontsize=10)
        ax.set_xlabel('Evaluations per angle\n(per independent run)',fontsize=8)
        ax.set_ylabel('Best mean FC / better named seed' if col==0 else '',fontsize=8)
        ax.tick_params(labelsize=8); ax.grid(alpha=.2)
        # Fixed alpha isolates effects of baseline effort, rather than reselection of alpha.
        alpha=.001 if key=='diabetes' else 10.
        numerator=source[(source.dataset==key)&(source.baseline=='cosine_first')&np.isclose(source.alpha,alpha)].iloc[0]
        for stage,effort in enumerate([0,16,32,64]):
            available=[old['best']]
            if stage: available += [r['checkpoints'][stage-1]['best'] for r in results]
            retained=min(available,key=lambda r:r['mean_fc'])
            diagnostics.append(dict(dataset=key,budget_per_angle_per_run=effort,
              additional_search_evaluations=3*effort*p,alpha=alpha,
              **{f'baseline_{k}':retained[k] for k in METRICS},
              **{f'{k}_ratio':numerator[k]/retained[k] for k in METRICS}))
        subset=[r for r in diagnostics if r['dataset']==key]
        for metric,color,label in zip(METRICS,['#0072B2','#D55E00','#009E73'],['Mean','Upper 5%','Upper 1%']):
            axes[1,col].plot([r['budget_per_angle_per_run'] for r in subset],
                 [r[f'{metric}_ratio'] for r in subset],marker='o',ms=3,color=color,label=label)
        axes[1,col].axhline(1,color='gray',lw=.8)
        axes[1,col].set_title(rf'Fixed $\alpha={alpha:g}$',fontsize=9)
        axes[1,col].set_xlabel('Additional evaluations per angle\n(per independent run)',fontsize=8)
        axes[1,col].set_ylabel('MMSSQV / retained MMQV score' if col==0 else '',fontsize=8)
        axes[1,col].tick_params(labelsize=8); axes[1,col].grid(alpha=.2)
    axes[0,0].legend(title='Independent seed suffix',fontsize=7,title_fontsize=7,loc='best')
    axes[1,1].legend(fontsize=7,loc='upper right')
    fig.tight_layout()
    fig.savefig(ROOT/'optimized_mmqv_search.pdf'); fig.savefig(ROOT/'optimized_mmqv_search.png',dpi=300)
    for name,rows in [('optimized_mmqv_summary',summaries),('optimized_mmqv_independent_runs',runs),('optimized_mmqv_budget_sensitivity',diagnostics)]:
        frame=pd.DataFrame(rows); frame.to_csv(ROOT/f'{name}.csv',index=False); print(name,frame.to_string(index=False),sep='\n')

if __name__=='__main__': main()
