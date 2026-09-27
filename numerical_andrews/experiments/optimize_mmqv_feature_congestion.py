"""Budgeted multistart search over MMQV rotations, reflections and constant sign.

Every candidate is an MMQV minimizer. The objective is the published mean FC
of the full-resolution raster. This is a heuristic search, not a certificate
of global optimality. No rendering parameter is optimized.
"""
from __future__ import annotations

from numerical_andrews.paths import RESULTS
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from numerical_andrews.core import load_example_datasets, evaluate_mmqv_modes
from numerical_andrews.rendering import render_curves, VERTICAL_PADDING
from numerical_andrews.experiments.analyze_feature_congestion_alpha import fc_profile

ROOT = RESULTS


def mmqv_transform(dimension, angles, reflections, constant_sign):
    """Columns express an orthonormal MMQV basis in cosine-first coordinates.

    The even-dimensional unpaired last cosine is augmented by its sine so
    its phase can vary. Thus the ambient basis dimension is d+1 for even d.
    """
    pairs = (dimension-1)//2
    ambient = dimension + (dimension % 2 == 0)
    transform = np.zeros((ambient, dimension))
    transform[0, 0] = constant_sign
    for pair in range(pairs):
        j = 1+2*pair
        c, s = np.cos(angles[pair]), np.sin(angles[pair])
        h = reflections[pair]
        transform[j:j+2, j:j+2] = [[c, -h*s], [s, h*c]]
    if dimension % 2 == 0:
        c, s = np.cos(angles[-1]), np.sin(angles[-1])
        transform[-2:, -1] = [c, s]
    return transform


def family_envelope(scores):
    bound = np.abs(scores[:, 0]).copy()
    for j in range(1, scores.shape[1], 2):
        bound += np.sqrt(2)*np.linalg.norm(scores[:, j:j+2], axis=1)
    return float(bound.max())


def optimize(dataset_key, budget, seed):
    ds = next(d for d in load_example_datasets(include_standardized=True) if d.key == dataset_key)
    dimension = ds.values.shape[1]
    angles_count, pairs = dimension//2, (dimension-1)//2
    points = np.linspace(-0.5, 0.5, 4097)
    trig = evaluate_mmqv_modes(dimension+(dimension % 2 == 0), points)
    # These ranges cover the entire previous MMSSQV path. The analytic bound
    # covers every MMQV candidate, including phases not visited by this search.
    baselines = pd.read_csv(ROOT/'feature_congestion_alpha_baseline.csv')
    matching = baselines[baselines.dataset == ds.key]
    if matching.empty and ds.key == 'bc_standardized':
        previous_limit = json.loads((ROOT/'optimized_mmqv_initial'/f'{ds.key}.json').read_text())['y_limit']
    else:
        previous_limit = float(matching.y_limit.iloc[0])
    limit = max(previous_limit, VERTICAL_PADDING*family_envelope(ds.scores))
    rng = np.random.default_rng(seed)
    history, cache = [], {}
    best = None
    start = time.perf_counter()
    directory = ROOT/'optimized_mmqv'
    directory.mkdir(exist_ok=True)

    def evaluate(angles, reflections, sign, stage):
        nonlocal best
        angles = np.asarray(angles) % (2*np.pi)
        reflections = np.asarray(reflections, dtype=int)
        key = (tuple(angles), tuple(reflections), int(sign))
        if key in cache:
            return cache[key]
        if len(history) >= budget:
            return None
        transform = mmqv_transform(dimension, angles, reflections, sign)
        assert np.max(np.abs(transform.T@transform-np.eye(dimension))) < 1e-12
        curves = (ds.scores@transform.T)@trig.T
        assert np.max(np.abs(curves)) < limit
        raster = render_curves(ds, curves, points, limit)
        profile, _ = fc_profile(raster)
        record = dict(evaluation=len(history)+1, stage=stage,
                      angles=angles.tolist(), reflections=reflections.tolist(),
                      constant_sign=int(sign), **profile)
        cache[key] = record
        if best is None or record['mean_fc'] < best['mean_fc']:
            best = record
            Image.fromarray(raster).save(directory/f'{ds.key}_best.png')
        record['best_mean_fc_so_far'] = best['mean_fc']
        record['elapsed_seconds'] = time.perf_counter()-start
        history.append(record)
        # Checkpoint every evaluation so an interrupted run retains its evidence.
        result = dict(dataset=ds.key, dataset_name=ds.display_name,
                      dimension=dimension, objective='mean_fc', seed=seed,
                      evaluation_budget=budget, evaluations=len(history),
                      status='running', global_optimality_claim=False,
                      y_limit=limit, previous_y_limit=previous_limit,
                      all_mmqv_amplitude_bound=family_envelope(ds.scores),
                      curve_grid_points=len(points), canvas_pixels=[768,512],
                      starts=3, best=best, history=history)
        (directory/f'{ds.key}.json').write_text(json.dumps(result, indent=2)+'\n')
        if len(history) % 8 == 0:
            print(f'{ds.key}: {len(history)}/{budget}; best mean FC {best["mean_fc"]:.8f}',flush=True)
        return record

    evaluate(np.zeros(angles_count), np.ones(pairs), 1, 'cosine_first_seed')
    evaluate(np.full(angles_count, np.pi/2), -np.ones(pairs), 1, 'sine_first_seed')
    for _ in range(22):
        evaluate(rng.uniform(0,2*np.pi,angles_count), rng.choice([-1,1],pairs),
                 rng.choice([-1,1]), 'random_start')
    seeds = sorted(history, key=lambda r:r['mean_fc'])[:3]
    for run, initial in enumerate(seeds):
        current = initial
        remaining_runs = len(seeds)-run
        local_end = len(history)+(budget-len(history))//remaining_runs
        step = np.pi/4
        while len(history) < local_end and step >= np.pi/64:
            improved = False
            # Vary angles first, including the phase of the final unpaired mode.
            for j in range(angles_count):
                for direction in (1,-1):
                    if len(history) >= local_end:
                        break
                    candidate = np.array(current['angles'])
                    candidate[j] += direction*step
                    trial = evaluate(candidate,current['reflections'],current['constant_sign'],
                                     f'local_{run+1}_angle_step_{step:.8g}')
                    if trial is not None and trial['mean_fc'] < current['mean_fc']:
                        current, improved = trial, True
                        break
            # Search discrete components of O(2), and the sign of the constant.
            for j in range(pairs+1):
                if len(history) >= local_end:
                    break
                signs = np.array(current['reflections'])
                sign = current['constant_sign']
                if j < pairs:
                    signs[j] *= -1
                else:
                    sign *= -1
                trial = evaluate(current['angles'], signs, sign, f'local_{run+1}_reflection')
                if trial is not None and trial['mean_fc'] < current['mean_fc']:
                    current, improved = trial, True
            # Decrease the scale every sweep to explore finer phases within budget.
            step /= 2
    path = directory/f'{ds.key}.json'
    result = json.loads(path.read_text())
    result['status'] = 'complete'
    result['stopping_rule'] = 'Per-start evaluation budget or phase step below pi/64; no global convergence claim'
    result['elapsed_seconds'] = time.perf_counter()-start
    path.write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(history).to_csv(directory/f'{ds.key}_history.csv',index=False)
    print(f'FINISHED {ds.key}: {len(history)} evaluations, best={best["mean_fc"]:.9g}',flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['iris','bc','diabetes','bc_standardized'],required=True)
    parser.add_argument('--budget',type=int,required=True)
    parser.add_argument('--seed',type=int,default=20260923)
    args=parser.parse_args()
    if args.budget < 30:
        parser.error('budget must be at least 30')
    optimize(args.dataset,args.budget,args.seed)
