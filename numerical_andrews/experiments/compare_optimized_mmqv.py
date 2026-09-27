"""Update comparisons from unchanged MMSSQV raster scores and optimized MMQV.

The archived two-baseline path supplies MMSSQV numerator scores. Reuse is
permitted only with identical saved coordinate ranges and rendering protocol.
Each winning MMQV raster is reconstructed and checked against the saved image
and score before forming ratios. For a full rerun, use
analyze_feature_congestion_alpha.py --baseline-mode optimized instead.
"""
from numerical_andrews.paths import RESULTS
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from numerical_andrews.core import load_example_datasets
from numerical_andrews.rendering import (render_curves, CANVAS_DPI, LINEWIDTH_POINTS, LINE_ALPHA, VERTICAL_PADDING)
from numerical_andrews.experiments.analyze_feature_congestion_alpha import (optimized_curves, fc_profile,
    build_landmarks, build_fitted_display, plot_profiles, plot_common_alpha_examples)

ROOT=RESULTS


def main():
    source=pd.read_csv(ROOT/'feature_congestion_two_baseline_profile.csv')
    metadata=json.loads((ROOT/'feature_congestion_two_baseline_metadata.json').read_text())
    assert metadata['render']['canvas_pixels']==[768,512]
    assert metadata['render']['curve_grid_points']==4097
    assert metadata['numerics']['fourier_cutoff_N']==256
    assert metadata['render']['dpi']==CANVAS_DPI
    assert metadata['render']['line_width_points']==LINEWIDTH_POINTS
    assert metadata['render']['line_alpha']==LINE_ALPHA
    assert metadata['render']['vertical_padding_factor']==VERTICAL_PADDING
    assert metadata['render']['parameter_interval']==[-0.5,0.5]
    old_baselines=pd.read_csv(ROOT/'feature_congestion_two_baseline_baseline.csv')
    points=np.linspace(-.5,.5,4097)
    profiles=[]; baselines=[]; displays=[]; display_rows=[]
    metrics=('mean_fc','upper_5pct_mean_fc','upper_1pct_mean_fc')
    for ds in load_example_datasets():
        curves,result=optimized_curves(ds,points)
        for i,name in enumerate(('cosine_first','sine_first')):
            old=old_baselines[(old_baselines.dataset==ds.key)&(old_baselines.baseline==name)].iloc[0]
            assert np.isclose(old.mean_fc,result['history'][i]['mean_fc'],rtol=1e-12,atol=0)
        previous=source[(source.dataset==ds.key)&(source.baseline=='cosine_first')].copy()
        assert len(previous)==17
        assert np.allclose(previous.y_limit,result['y_limit'],rtol=1e-15,atol=0)
        image=render_curves(ds,curves,points,result['y_limit'])
        assert np.array_equal(image,np.asarray(Image.open(ROOT/'optimized_mmqv'/f'{ds.key}_best.png')))
        profile,_=fc_profile(image)
        for metric in metrics:
            assert np.isclose(profile[metric],result['best'][metric],rtol=1e-13,atol=0)
            previous['baseline_'+metric]=profile[metric]
            previous[metric+'_ratio']=previous[metric]/profile[metric]
            previous[metric+'_percent_change']=100*(previous[metric+'_ratio']-1)
        previous['baseline']='optimized'
        profiles.append(previous)
        baselines.append(dict(dataset=ds.key,dataset_name=ds.display_name,
                              observations=ds.values.shape[0],dimension=ds.values.shape[1],
                              method='MMQV',baseline='optimized',y_limit=result['y_limit'],**profile))
        display=build_fitted_display(ds,points,256,'optimized')
        displays.append(display)
        for method,limits,values in [('MMQV',display.baseline_y_limit,display.baseline_profile),
                                    ('MMSSQV',display.smoothed_y_limit,display.smoothed_profile)]:
            display_rows.append(dict(dataset=ds.key,method=method,
                                     alpha=display.alpha if method=='MMSSQV' else None,
                                     y_min=limits[0],y_max=limits[1],
                                     coordinate_protocol='method_range_5pct_padding',**values))
        print('Verified winning raster and scores:',ds.key,flush=True)
    profile=pd.concat(profiles,ignore_index=True)
    profile.to_csv(ROOT/'feature_congestion_alpha_profile.csv',index=False)
    pd.DataFrame(baselines).to_csv(ROOT/'feature_congestion_alpha_baseline.csv',index=False)
    landmarks=build_landmarks(profile)
    landmarks.to_csv(ROOT/'feature_congestion_alpha_landmarks.csv',index=False)
    pd.DataFrame(display_rows).to_csv(ROOT/'feature_congestion_display_profiles.csv',index=False)
    plot_profiles(profile,ROOT/'feature_congestion_alpha_profile.png',ROOT/'feature_congestion_alpha_profile.pdf')
    plot_common_alpha_examples(displays,ROOT/'feature_congestion_alpha_examples.png',ROOT/'feature_congestion_alpha_examples.pdf','optimized')
    metadata['baseline_mode']='optimized'
    metadata['command']='compare_optimized_mmqv.py'
    metadata['reproduction_command']='.venv/bin/python compare_optimized_mmqv.py'
    metadata['full_reproduction_command']='.venv/bin/python analyze_feature_congestion_alpha.py --baseline-mode optimized'
    metadata['drivers']['optimization']='expand_mmqv_optimization.py'
    metadata['drivers']['optimization_summary']='summarize_expanded_mmqv.py'
    metadata['expanded_search_records']='optimized_mmqv_expanded/{dataset}/{seed}.json'
    metadata['drivers']['comparison_update']='compare_optimized_mmqv.py'
    orders=metadata['three_number_profile'].pop('baseline_orderings',metadata['three_number_profile'].get('named_seed_orderings'))
    metadata['three_number_profile']['named_seed_orderings']=orders
    metadata['optimization_records']='optimized_mmqv/{iris,bc,diabetes}.json'
    metadata['score_provenance']='MMSSQV scores reused from archived two-baseline run at unchanged ranges; winning MMQV rasters and scores independently reconstructed and verified'
    metadata['three_number_profile']['ratios']='MMSSQV divided by best mean-FC MMQV found'
    metadata['render']['coordinate_protocol']='fixed across all MMQV candidates and the MMSSQV path, bounded over the entire MMQV family'
    metadata['common_alpha_figure']['baseline']='optimized basis selected at fixed scale, displayed with fitted limits without reoptimization'
    (ROOT/'feature_congestion_alpha_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(landmarks.to_string(index=False))


if __name__=='__main__':
    main()
