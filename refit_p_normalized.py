"""
P-Experiment Library Re-fit with Normalized Game Time
=====================================================

Re-fits all 196 P-experiment subjects using the v4 engine with
normalized_time=True. Parameters now mean "per unit of game progress"
rather than "per round," making them transfer across game lengths.

For 10-round PGG-P: dt = 1/9 ≈ 0.111 per round.
Rate parameters will be ~9x larger than v3 values but produce
identical dynamics on 10-round games, and correct dynamics on
100-round games.

Output: v4_library_fitted.json (same structure as v3_library_fitted.json)
"""

import os
import sys
import json
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

from pgg_p_loader import load_p_experiment
from vcms_engine_v4 import (
    VCMSParams, GameConfig, PARAM_NAMES, PARAM_BOUNDS_NORMALIZED,
    PGG_P_CONFIG, run_vcms_v4,
)


# PGG-P config with normalized time enabled
PGG_P_NORM_CONFIG = GameConfig(
    max_contrib=20, max_punish=30, has_punishment=True,
    n_signals=4,
    normalized_time=True,
)

# Parameters to fit (same 18 as v3)
FIT_PARAM_NAMES = [
    'c_base', 'alpha', 'v_rep', 'v_ref',
    'inertia',
    's_dir', 's_rate', 's_initial',
    's_frac', 'p_scale', 's_thresh',
    'b_initial', 'b_depletion_rate', 'b_replenish_rate', 'acute_threshold',
    'facilitation_rate',
    'h_strength', 'h_start',
]

FIT_BOUNDS = [PARAM_BOUNDS_NORMALIZED[n] for n in FIT_PARAM_NAMES]

# Defaults for v4-only params (fixed at 0)
FIXED_DEFAULTS = {
    'v_self_weight': 0.0,
    's_exploitation_rate': 0.0,
}


def make_params(x):
    """Create VCMSParams from optimizer vector."""
    d = dict(FIXED_DEFAULTS)
    for i, name in enumerate(FIT_PARAM_NAMES):
        d[name] = x[i]
    return VCMSParams(**d)


def objective(x, rounds):
    """Combined normalized RMSE via v4 engine with normalized time."""
    try:
        params = make_params(x)
        result = run_vcms_v4(params, rounds, PGG_P_NORM_CONFIG)
        return result['rmse_combined']
    except Exception:
        return 100.0


def fit_subject(rounds, maxiter=600, seed=42, verbose=False):
    """Fit 18 parameters via DE + Nelder-Mead."""
    t0 = time.time()

    # Stage 1: Differential Evolution
    de_result = differential_evolution(
        objective,
        bounds=FIT_BOUNDS,
        args=(rounds,),
        maxiter=maxiter,
        seed=seed,
        tol=1e-8,
        polish=False,
        mutation=(0.5, 1.5),
        recombination=0.8,
        popsize=20,
        workers=-1,
        updating='deferred',
    )

    if verbose:
        print(f"    DE: RMSE={de_result.fun:.5f} ({time.time()-t0:.1f}s)")

    # Stage 2: Nelder-Mead refinement
    t1 = time.time()
    nm_result = minimize(
        objective,
        x0=de_result.x,
        args=(rounds,),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10},
    )

    # Enforce bounds
    x_final = np.clip(nm_result.x,
                      [b[0] for b in FIT_BOUNDS],
                      [b[1] for b in FIT_BOUNDS])
    final_rmse = objective(x_final, rounds)

    if verbose:
        print(f"    NM: RMSE={final_rmse:.5f} ({time.time()-t1:.1f}s)")

    best_params = {}
    for i, name in enumerate(FIT_PARAM_NAMES):
        best_params[name] = float(x_final[i])

    return {
        'best_params': best_params,
        'best_rmse': float(final_rmse),
        'de_rmse': float(de_result.fun),
        'time_s': time.time() - t0,
    }


def build_normalized_library(library_path, data_dirs, output_path, maxiter=600):
    """Re-fit all P-experiment library subjects with normalized time."""

    with open(library_path) as f:
        library = json.load(f)

    # Load all city data
    all_data = {}
    for d in data_dirs:
        for fname in os.listdir(d):
            if fname.endswith('.csv') and 'P-EXPERIMENT' in fname:
                city_data = load_p_experiment(os.path.join(d, fname))
                for sid, rounds in city_data.items():
                    all_data[sid] = rounds
                print(f"  Loaded {len(city_data)} subjects from {fname}")

    print(f"\nTotal data subjects: {len(all_data)}")
    print(f"Library subjects to fit: {len(library)}")
    print(f"Config: normalized_time=True, dt=1/{10-1}={1/9:.4f} per round")

    missing = [sid for sid in library if sid not in all_data]
    if missing:
        print(f"WARNING: {len(missing)} library subjects not found: {missing}")

    results = {}
    total_t0 = time.time()

    for i, (sid, record) in enumerate(sorted(library.items())):
        if sid not in all_data:
            print(f"  [{i+1}/{len(library)}] {sid}: SKIPPED")
            continue

        rounds = all_data[sid]
        print(f"  [{i+1}/{len(library)}] {sid} ({record['behavioral_profile']})...",
              end=' ', flush=True)

        fit_result = fit_subject(rounds, maxiter=maxiter, seed=42)
        elapsed = fit_result['time_s']
        print(f"RMSE={fit_result['best_rmse']:.4f} ({elapsed:.1f}s)")

        # Run agent with fitted params for state trajectories
        params = make_params([fit_result['best_params'][n] for n in FIT_PARAM_NAMES])
        agent_result = run_vcms_v4(params, rounds, PGG_P_NORM_CONFIG)

        # Extract state trajectory
        state_trajectory = []
        for t_data in agent_result['trace']:
            state_trajectory.append({
                'B': t_data['budget']['b_post'],
                'S': t_data['state']['strain_end'],
                'A': t_data['routing']['affordability'],
                'm_eval': t_data['m_eval']['m_eval_acc'],
                'gate': t_data['routing']['gate'],
            })

        results[sid] = {
            # Original library data
            'behavioral_profile': record['behavioral_profile'],
            'population': record.get('population', ''),
            'session': record.get('session', ''),
            'contribution_trajectory': record['contribution_trajectory'],
            'punishment_sent_trajectory': record['punishment_sent_trajectory'],
            'punishment_received_trajectory': record['punishment_received_trajectory'],
            'others_mean_trajectory': record['others_mean_trajectory'],
            # V4 normalized-time fitted model
            'v3_params': fit_result['best_params'],  # key kept as v3_params for compat
            'v3_param_names': FIT_PARAM_NAMES,
            'v3_rmse': fit_result['best_rmse'],
            'v3_pred_c': agent_result['pred_contrib'],
            'v3_pred_p': agent_result['pred_punish'],
            'v3_state_trajectory': state_trajectory,
            # Metadata
            'engine': 'v4_normalized_time',
            'dt': 1.0 / (len(rounds) - 1),
        }

    total_elapsed = time.time() - total_t0
    print(f"\nTotal fitting time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Subjects fitted: {len(results)}/{len(library)}")

    rmses = [r['v3_rmse'] for r in results.values()]
    print(f"RMSE: mean={np.mean(rmses):.4f}, median={np.median(rmses):.4f}, "
          f"max={np.max(rmses):.4f}")

    # Compare with v3 library if available
    v3_path = os.path.join('.', 'v3_library_fitted.json')
    if os.path.exists(v3_path):
        with open(v3_path) as f:
            v3_lib = json.load(f)
        v3_rmses = [v3_lib[sid]['v3_rmse'] for sid in results if sid in v3_lib]
        v4_rmses = [results[sid]['v3_rmse'] for sid in results if sid in v3_lib]
        print(f"\nComparison with v3 library ({len(v3_rmses)} shared subjects):")
        print(f"  v3 mean RMSE: {np.mean(v3_rmses):.4f}")
        print(f"  v4 mean RMSE: {np.mean(v4_rmses):.4f}")
        print(f"  Delta:        {np.mean(v4_rmses) - np.mean(v3_rmses):+.4f}")

        # Per-subject comparison
        better = sum(1 for a, b in zip(v4_rmses, v3_rmses) if a < b)
        same = sum(1 for a, b in zip(v4_rmses, v3_rmses) if abs(a - b) < 0.001)
        worse = sum(1 for a, b in zip(v4_rmses, v3_rmses) if a > b + 0.001)
        print(f"  Better: {better}, Same (<0.001): {same}, Worse: {worse}")

        # Show biggest changes
        deltas = [(sid, results[sid]['v3_rmse'] - v3_lib[sid]['v3_rmse'])
                  for sid in results if sid in v3_lib]
        deltas.sort(key=lambda x: x[1])
        print(f"\n  Biggest improvements:")
        for sid, d in deltas[:5]:
            print(f"    {sid}: {d:+.4f}")
        print(f"  Biggest degradations:")
        for sid, d in deltas[-5:]:
            print(f"    {sid}: {d:+.4f}")

        # Parameter scale comparison
        print(f"\n  Parameter scale comparison (mean values):")
        scale_params = ['s_rate', 'b_depletion_rate', 'b_replenish_rate',
                        'facilitation_rate', 'h_start']
        shared = [sid for sid in results if sid in v3_lib]
        for pname in scale_params:
            v3_vals = [v3_lib[sid]['v3_params'][pname] for sid in shared]
            v4_vals = [results[sid]['v3_params'][pname] for sid in shared]
            ratio = np.mean(v4_vals) / max(np.mean(v3_vals), 0.001)
            print(f"    {pname:>20s}: v3={np.mean(v3_vals):.3f}, "
                  f"v4={np.mean(v4_vals):.3f}, ratio={ratio:.1f}x")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    return results


if __name__ == '__main__':
    library_path = 'p_experiment_canonical_library.json'
    data_dirs = ['.']
    output_path = 'v4_library_fitted.json'

    maxiter = 600
    if '--fast' in sys.argv:
        maxiter = 100
        print("FAST MODE: 100 iterations")

    build_normalized_library(library_path, data_dirs, output_path, maxiter=maxiter)
