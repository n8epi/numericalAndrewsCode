"""Rescore saved manuscript winners; optionally compare pre-refactor full maps."""
from numerical_andrews.paths import RESULTS
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from numerical_andrews.experiments.analyze_feature_congestion_alpha import fc_profile

ROOT=RESULTS
METRICS=['mean_fc','upper_5pct_mean_fc','upper_1pct_mean_fc']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--before-directory',type=Path)
    parser.add_argument('--previous-module',type=Path)
    args=parser.parse_args();rows=[];identity={}
    if args.previous_module:
        old=ast.parse(args.previous_module.read_text())
        new=ast.parse((Path(__file__).resolve().parents[1]/'congestion.py').read_text())
        for name in ['official_rgb_to_lab','official_gaussian_pyramid','feature_congestion']:
            a=ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name))
            b=ast.dump(next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name))
            assert a==b
            identity[name]=hashlib.sha256(a.encode()).hexdigest()
    for key in ['iris','bc','diabetes','bc_standardized']:
        saved=json.loads((ROOT/'optimized_mmqv'/f'{key}.json').read_text())
        raster=np.asarray(Image.open(ROOT/'optimized_mmqv'/f'{key}_best.png'))
        profile,local_map=fc_profile(raster)
        expected=np.array([saved['best'][m] for m in METRICS])
        values=np.array([profile[m] for m in METRICS])
        np.testing.assert_array_equal(values,expected)
        row=dict(dataset=key,profile=profile,saved_scores_bitwise_equal=True)
        if args.before_directory:
            before=np.load(args.before_directory/f'fc_before_{key}.npz')
            np.testing.assert_array_equal(raster,before['image'])
            np.testing.assert_array_equal(values,before['profile'])
            np.testing.assert_array_equal(local_map,before['local_map'])
            row['pre_refactor_map_bitwise_equal']=True
            row['max_pre_refactor_map_difference']=float(np.max(abs(local_map-before['local_map'])))
        rows.append(row); print(row,flush=True)
    result=dict(scope='Four retained full-resolution MMQV rasters; original scorer, acceleration disabled',
      unchanged_function_ast_sha256=identity,results=rows,
      compatibility_module_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'congestion.py').read_bytes()).hexdigest())
    (ROOT/'fc_compatibility_validation.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
