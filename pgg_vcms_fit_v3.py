"""
PGG VCMS v3 Fitting Pipeline
=============================
Flat optimization: differential_evolution → Nelder-Mead refinement.
No staging (v2 showed staging hurts joint optima quality).

Usage:
  python pgg_vcms_fit_v3.py <data_dir> [--city boston|samara|istanbul] [--subject ID]
"""

import os
import sys
import json
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

from pgg_p_loader import load_p_experiment
from pgg_vcms_agent_v3 import (
    VCMSParams, PARAM_NAMES, PARAM_BOUNDS, DEFAULTS, KNOCKOUTS,
    run_vcms_agent, make_vcms_params, vcms_objective,
)


# =============================================================================
# Fitting
# =============================================================================

def fit_subject(rounds, maxiter=600, seed=42, verbose=False):
    """
    Fit all 16 parameters via flat differential_evolution + Nelder-Mead.
    
    Returns dict with best_params, best_rmse, result object.
    """
    free_names = PARAM_NAMES
    bounds = [PARAM_BOUNDS[n] for n in free_names]

    if verbose:
        print(f"  Fitting {len(free_names)} parameters (flat)...")

    t0 = time.time()

    # Stage 1: Differential Evolution (global)
    de_result = differential_evolution(
        vcms_objective,
        bounds=bounds,
        args=(rounds, free_names, None),
        maxiter=maxiter,
        seed=seed,
        tol=1e-8,
        polish=False,
        mutation=(0.5, 1.5),
        recombination=0.8,
        popsize=20,
        workers=-1,
        updating='deferred',  # required for workers>1
    )

    if verbose:
        print(f"    DE: RMSE={de_result.fun:.5f} ({time.time()-t0:.1f}s)")

    # Stage 2: Nelder-Mead refinement (local)
    t1 = time.time()
    nm_result = minimize(
        vcms_objective,
        x0=de_result.x,
        args=(rounds, free_names, None),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10},
    )

    # Enforce bounds on NM result
    x_final = np.clip(nm_result.x, [b[0] for b in bounds], [b[1] for b in bounds])
    final_rmse = vcms_objective(x_final, rounds, free_names, None)

    if verbose:
        print(f"    NM: RMSE={final_rmse:.5f} ({time.time()-t1:.1f}s)")

    # Build params dict
    best_params = {}
    for i, name in enumerate(free_names):
        best_params[name] = float(x_final[i])

    return {
        'best_params': best_params,
        'best_rmse': float(final_rmse),
        'de_rmse': float(de_result.fun),
        'time_s': time.time() - t0,
    }


def run_knockout(rounds, base_params, knockout_name):
    """
    Run a knockout: refit with specific params fixed, compare to base RMSE.
    """
    if knockout_name == 'no_routing':
        # Special case: no_routing is handled by flag, not param fixing
        # Run base params with routing disabled
        params = make_vcms_params(
            [base_params[n] for n in PARAM_NAMES],
            PARAM_NAMES, None
        )
        result = run_vcms_agent(params, rounds, knockout='no_routing')
        return {
            'knockout': knockout_name,
            'rmse': result['rmse_combined'],
            'method': 'flag',
        }

    ko_fixed = KNOCKOUTS[knockout_name]
    free_names = [n for n in PARAM_NAMES if n not in ko_fixed]
    bounds = [PARAM_BOUNDS[n] for n in free_names]

    # Seed from base params
    x0 = [base_params[n] for n in free_names]

    # Quick refit with knockout
    de_result = differential_evolution(
        vcms_objective,
        bounds=bounds,
        args=(rounds, free_names, ko_fixed),
        maxiter=300,
        seed=42,
        tol=1e-7,
        polish=False,
        mutation=(0.5, 1.5),
        recombination=0.8,
        popsize=15,
        workers=-1,
        updating='deferred',
    )

    nm_result = minimize(
        vcms_objective,
        x0=de_result.x,
        args=(rounds, free_names, ko_fixed),
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9},
    )

    x_final = np.clip(nm_result.x, [b[0] for b in bounds], [b[1] for b in bounds])
    ko_rmse = vcms_objective(x_final, rounds, free_names, ko_fixed)

    return {
        'knockout': knockout_name,
        'rmse': float(ko_rmse),
        'delta': float(ko_rmse - base_params.get('__base_rmse', 0)),
        'method': 'refit',
    }


# =============================================================================
# Main
# =============================================================================

def fit_city(data_dir, city_filter=None, subject_filter=None, verbose=True):
    """Fit all subjects, run knockouts, produce summary."""

    # Load all P-experiment CSVs from data directory
    subjects = {}
    csv_files = sorted(f for f in os.listdir(data_dir)
                       if f.endswith('.csv') and 'P-EXPERIMENT' in f)
    if city_filter:
        csv_files = [f for f in csv_files if city_filter.upper() in f.upper()]

    for csv_file in csv_files:
        city_tag = csv_file.split('_')[1]  # e.g., BOSTON
        session_tag = ''
        if 'SESSION' in csv_file.upper():
            for part in csv_file.upper().split('_'):
                if part.startswith('SESSION'):
                    session_tag = '_' + part.replace('.CSV', '')
        data = load_p_experiment(os.path.join(data_dir, csv_file))
        for sid, rounds in data.items():
            tag = f"{city_tag}_{sid}{session_tag}"
            subjects[tag] = rounds

    if subject_filter:
        subjects = {k: v for k, v in subjects.items() if subject_filter in k}

    if not subjects:
        print(f"No subjects found in {data_dir} with filters city={city_filter}, subject={subject_filter}")
        return

    print(f"Fitting {len(subjects)} subjects (v3 — 16 params, flat optimization)")
    print(f"{'='*70}")

    all_results = {}

    for tag, rounds in sorted(subjects.items()):
        print(f"\n--- {tag} ({len(rounds)} rounds) ---")

        # Fit
        fit = fit_subject(rounds, verbose=verbose)
        print(f"  Final RMSE: {fit['best_rmse']:.5f} ({fit['time_s']:.1f}s)")

        # Run agent with best params for predictions
        params = make_vcms_params(
            [fit['best_params'][n] for n in PARAM_NAMES],
            PARAM_NAMES, None
        )
        result = run_vcms_agent(params, rounds)

        # Print predictions vs actuals
        print(f"  C actual: {result['actual_contrib']}")
        print(f"  C pred:   {result['pred_contrib']}")
        print(f"  C RMSE:   {result['rmse_contrib']:.2f}")
        print(f"  P actual: {result['actual_punish']}")
        print(f"  P pred:   {result['pred_punish']}")
        print(f"  P RMSE:   {result['rmse_punish']:.2f}")

        # Show B trajectory
        b_traj = [t['budget']['b_post'] for t in result['trace']]
        a_traj = [t['routing']['affordability'] for t in result['trace']]
        print(f"  B traj:   {['%.2f' % b for b in b_traj]}")
        print(f"  A traj:   {['%.2f' % a for a in a_traj]}")

        # Key params
        p = fit['best_params']
        print(f"  Key params: c_base={p['c_base']:.3f}, b_initial={p['b_initial']:.3f}, "
              f"b_depl={p['b_depletion_rate']:.3f}, b_repl={p['b_replenish_rate']:.3f}, "
              f"s_dir={'+'if p['s_dir']>=0 else '-'}, s_rate={p['s_rate']:.3f}")

        # Knockouts
        ko_results = {}
        fit['best_params']['__base_rmse'] = fit['best_rmse']
        for ko_name in KNOCKOUTS:
            ko = run_knockout(rounds, fit['best_params'], ko_name)
            ko_results[ko_name] = ko
            delta = ko['rmse'] - fit['best_rmse']
            active = '***' if delta > 0.005 else ''
            print(f"    KO {ko_name:20s}: RMSE={ko['rmse']:.5f} (Δ={delta:+.5f}) {active}")

        all_results[tag] = {
            'fit': fit,
            'predictions': {
                'pred_c': result['pred_contrib'],
                'pred_p': result['pred_punish'],
                'actual_c': result['actual_contrib'],
                'actual_p': result['actual_punish'],
                'B_trajectory': b_traj,
                'A_trajectory': a_traj,
            },
            'knockouts': ko_results,
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    rmses = [r['fit']['best_rmse'] for r in all_results.values()]
    print(f"  Subjects: {len(rmses)}")
    print(f"  Mean RMSE: {np.mean(rmses):.5f}")
    print(f"  Median RMSE: {np.median(rmses):.5f}")
    print(f"  Std RMSE: {np.std(rmses):.5f}")
    print(f"  Min/Max: {np.min(rmses):.5f} / {np.max(rmses):.5f}")

    # Channel prevalence (active if knockout ΔRMSE > 0.005)
    print("\n  Channel prevalence:")
    for ko_name in KNOCKOUTS:
        active_count = sum(
            1 for r in all_results.values()
            if r['knockouts'][ko_name]['rmse'] - r['fit']['best_rmse'] > 0.005
        )
        print(f"    {ko_name:20s}: {active_count}/{len(rmses)} "
              f"({100*active_count/len(rmses):.0f}%)")

    # Save results
    out_path = os.path.join(data_dir, 'vcms_v3_results.json')
    # Convert numpy types for JSON
    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=to_json)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

    city = None
    subject = None
    for i, arg in enumerate(sys.argv):
        if arg == '--city' and i + 1 < len(sys.argv):
            city = sys.argv[i + 1]
        if arg == '--subject' and i + 1 < len(sys.argv):
            subject = sys.argv[i + 1]

    fit_city(data_dir, city_filter=city, subject_filter=subject, verbose=True)
