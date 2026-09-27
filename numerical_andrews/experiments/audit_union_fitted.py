"""Audit every saved union-fitted candidate's MMQV invariants and checkpoints."""
import json
from pathlib import Path
import numpy as np
from numerical_andrews.experiments.compare_union_fitted import OUT, METRICS, dump
from numerical_andrews.experiments.optimize_mmqv_feature_congestion import mmqv_transform

def main(completed_only=False):
    protocol=json.loads((OUT/'protocol.json').read_text())
    jobs=sorted({(r['dataset'],r['group'],s) for r in protocol['records'] for s in protocol['seeds']})
    dimensions={'iris':4,'diabetes':10,'bc':30,'bc_standardized':30}
    max_gram=max_energy=max_checkpoint=0.;count=0;completed=0
    for key,group,seed in jobs:
        path=OUT/'search'/key/group/f'{seed}.json'
        if completed_only and not path.exists():continue
        record=json.loads(path.read_text())
        if completed_only and record['status']!='complete':continue
        completed+=1
        d=dimensions[key];assert record['status']=='complete'
        assert len(record['history'])==record['evaluations']==64*(d//2)
        target=(np.arange(d)+1)//2;ambient=(np.arange(d+1)+1)//2
        for i,candidate in enumerate(record['history'],1):
            assert candidate['evaluation']==i
            assert candidate['constant_sign'] in [-1,1]
            assert set(candidate['reflections']).issubset({-1,1})
            assert candidate['y_limit']>=1.05*record['range_floor']*(1-1e-14)
            transform=mmqv_transform(d,candidate['angles'],candidate['reflections'],candidate['constant_sign'])
            gram=np.max(np.abs(transform.T@transform-np.eye(d)))
            energy=np.max(np.abs(transform.T@(ambient[:,None]**2*transform)-np.diag(target**2)))
            max_gram=max(max_gram,float(gram));max_energy=max(max_energy,float(energy));count+=1
            assert gram<1e-12 and energy<1e-10
        for checkpoint in record['checkpoints']:
            for metric in METRICS:
                err=abs(checkpoint['original_minus_accelerated'][metric])
                max_checkpoint=max(max_checkpoint,err)
                assert err<=1e-8+1e-7*abs(checkpoint['best'][metric])
    result=dict(searches=completed,planned_searches=len(jobs),candidates=count,max_gram_error=max_gram,
        max_frequency_weighted_gram_error=max_energy,max_checkpoint_score_discrepancy=max_checkpoint,
        claim='Every candidate is an MMQV minimizer to roundoff; no congestion optimality certification')
    dump(OUT/('candidate_audit_partial.json' if completed_only else 'candidate_audit.json'),result);print(result)
if __name__=='__main__':main()
