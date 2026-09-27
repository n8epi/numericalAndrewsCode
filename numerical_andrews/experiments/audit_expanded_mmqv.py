"""Audit saved independent searches, basis constraints and checkpoint scoring."""
import argparse,json
import numpy as np
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import ROOT,mmqv_transform
from numerical_andrews.experiments.expand_mmqv_optimization import SEEDS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",default=["iris","bc","diabetes"])
    args=parser.parse_args()
    rows=[]
    for key in args.datasets:
        starts=[]
        for seed in SEEDS:
            r=json.loads((ROOT/'optimized_mmqv_expanded'/key/f'{seed}.json').read_text())
            assert r['status']=='complete'
            d=r['dimension']; p=d//2; h=r['history']
            assert len(h)==r['evaluation_budget']==64*p
            assert [c['budget'] for c in r['checkpoints']]==[16*p,32*p,64*p]
            starts.append(h[2]['angles'])
            gram_error=energy_error=0.
            weights=np.array([0]+[j*j for j in range(1,p+1) for _ in range(2)])
            for i,v in enumerate(h):
                assert v['evaluation']==i+1
                assert len(v['angles'])==p and len(v['reflections'])==(d-1)//2
                assert set(v['reflections']).issubset({-1,1}) and v['constant_sign'] in (-1,1)
                t=mmqv_transform(d,v['angles'],v['reflections'],v['constant_sign'])
                gram_error=max(gram_error,float(np.max(np.abs(t.T@t-np.eye(d)))))
                energy_error=max(energy_error,float(np.max(np.abs(t.T@(weights[:,None]*t)-np.diag(weights[:d])))))
                assert np.isclose(v['best_mean_fc_so_far'],min(x['mean_fc'] for x in h[:i+1]),rtol=0,atol=1e-14)
            assert gram_error<1e-12 and energy_error<1e-10
            errors=[abs(e) for c in r['checkpoints'] for e in c['original_minus_accelerated'].values()]
            assert max(errors)<1e-7
            np.testing.assert_allclose(r['best']['mean_fc'],min(v['mean_fc'] for v in h),rtol=1e-7)
            rows.append(dict(dataset=key,seed=seed,candidates=len(h),max_gram_error=gram_error,
                             max_weighted_energy_error=energy_error,max_checkpoint_score_error=max(errors)))
        assert len({tuple(a) for a in starts})==len(SEEDS)
    filename = ('optimized_mmqv_audit.json' if args.datasets==['iris','bc','diabetes'] else
                'optimized_mmqv_audit_'+'_'.join(args.datasets)+'.json')
    (ROOT/filename).write_text(json.dumps(rows,indent=2)+'\n')
    print('Verified all',sum(r['candidates'] for r in rows),'candidates and',3*len(rows),'original-score checkpoints.')
    print('Maximum checkpoint score error:',max(r['max_checkpoint_score_error'] for r in rows))
if __name__=='__main__': main()
