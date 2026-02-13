#!/usr/bin/env python3
"""
N-experiment Subject Fitting
=============================

Fit VCMS v4 parameters to 212 N-experiment (no punishment) subjects.
Adds cooperative profiles (especially Boston) to the library that
the P-experiment-only library lacks.

Reuses the fast forward pass from ipd_fit.py — identical dynamics
since both games have has_punishment=False (gate=0, no discharge).
Only difference: max_contrib=20, 10 rounds.
"""

import json
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

from pgg_p_loader import load_p_experiment
from vcms_engine_v4 import (
    PGG_N_CONFIG, VCMSParams, run_vcms_v4,
)
from ipd_fit import (
    predict_fast, FIT_PARAM_NAMES, FIT_BOUNDS, FIXED_PARAMS,
    params_array_to_dict,
)
from v3_n_experiment_test import N_EXPERIMENT_CSVS


# ================================================================
# OBJECTIVE
# ================================================================

def objective(x, rounds, actual, max_c):
    """RMSE of contribution predictions on 0-max_c scale."""
    preds = predict_fast(x, rounds, max_c)
    n = len(actual)
    sse = sum((actual[i] - preds[i]) ** 2 for i in range(n))
    return math.sqrt(sse / n)


def fit_subject(rounds, actual, max_c=20, seed=42):
    """Fit 15 VCMS parameters to one N-experiment subject."""
    result = differential_evolution(
        objective, FIT_BOUNDS,
        args=(rounds, actual, max_c),
        maxiter=100, popsize=10, tol=0.005,
        seed=seed, polish=True,
        disp=False,
    )
    return result.x, result.fun, result.nfev


def verify_fit(params_dict, rounds, actual):
    """Verify through official v4 engine."""
    v4_params = VCMSParams(**params_dict)
    result = run_vcms_v4(v4_params, rounds, PGG_N_CONFIG)
    preds = result['pred_contrib']
    n = len(actual)
    rmse = math.sqrt(sum((actual[i] - preds[i]) ** 2 for i in range(n)) / n)
    return rmse, preds


# ================================================================
# DATA LOADING
# ================================================================

def load_n_experiment_data():
    """Load all N-experiment subjects with city labels."""
    import os
    all_data = {}
    city_map = {}

    for city, csv_path in N_EXPERIMENT_CSVS:
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        data = load_p_experiment(csv_path)
        for sid, rounds in data.items():
            all_data[sid] = rounds
            city_map[sid] = city

    return all_data, city_map


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  N-EXPERIMENT SUBJECT FITTING")
    print("  Building VCMS library for PGG N-experiment subjects")
    print("=" * 72)

    # Load
    print("\nLoading N-experiment data...")
    n_data, city_map = load_n_experiment_data()
    print(f"  {len(n_data)} subjects across {len(set(city_map.values()))} cities")

    cities = sorted(set(city_map.values()))
    for city in cities:
        n = sum(1 for c in city_map.values() if c == city)
        print(f"    {city}: {n} subjects")

    total = len(n_data)
    print(f"\n  Parameters: {len(FIT_PARAM_NAMES)} free, "
          f"{len(FIXED_PARAMS)} fixed")

    # Fit all subjects
    library = {}
    all_rmse = []
    city_rmse = {c: [] for c in cities}

    t0 = time.time()
    sids = sorted(n_data.keys())

    for idx, sid in enumerate(sids):
        rounds = n_data[sid]
        actual = [r.contribution for r in rounds]
        coop_rate = np.mean(actual) / 20.0

        # Classify
        slope = np.polyfit(np.arange(len(actual)), actual, 1)[0]
        if slope < -0.3:
            stype = 'declining'
        elif slope > 0.3:
            stype = 'rising'
        else:
            if np.mean(actual) > 12:
                stype = 'stable-high'
            elif np.mean(actual) < 5:
                stype = 'stable-low'
            else:
                stype = 'stable-mid'

        # Fit
        best_x, rmse, nfev = fit_subject(rounds, actual, max_c=20, seed=42)
        params_dict = params_array_to_dict(best_x)

        # Verify
        verified_rmse, verified_preds = verify_fit(params_dict, rounds, actual)

        city = city_map[sid]

        library[sid] = {
            'treatment': 'N-experiment',
            'city': city,
            'subject_type': stype,
            'v4_params': params_dict,
            'contribution_trajectory': actual,
            'others_mean_trajectory': [r.others_mean for r in rounds],
            'punishment_sent_trajectory': [0.0] * len(rounds),
            'punishment_received_trajectory': [0.0] * len(rounds),
            'fit_rmse': float(verified_rmse),
            'fit_rmse_norm': float(verified_rmse / 20.0),
            'fit_pred': verified_preds,
            'cooperation_rate': float(coop_rate),
            'slope': float(slope),
        }

        all_rmse.append(verified_rmse)
        city_rmse[city].append(verified_rmse)

        if (idx + 1) % 20 == 0 or idx == total - 1:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"  [{idx + 1:>3d}/{total}] {sid:<12s} ({city:<10s}) "
                  f"rmse={verified_rmse:.2f} "
                  f"({elapsed:>5.0f}s, ~{eta:>4.0f}s rem)")

    total_time = time.time() - t0
    print(f"\n  Fitting complete: {total_time:.0f}s "
          f"({total_time / total:.1f}s per subject)")

    # ================================================================
    # QUALITY REPORT
    # ================================================================

    print(f"\n{'=' * 72}")
    print(f"  FITTING QUALITY")
    print(f"{'=' * 72}")

    print(f"\n  Overall: RMSE={np.mean(all_rmse):.2f} "
          f"(normalized={np.mean(all_rmse)/20:.4f})")

    # Per city
    print(f"\n  Per city:")
    print(f"    {'City':<12s} {'n':>4s} {'RMSE':>8s} {'Norm':>8s}")
    print(f"    {'-' * 36}")
    for city in cities:
        vals = city_rmse[city]
        print(f"    {city:<12s} {len(vals):>4d} {np.mean(vals):>8.2f} "
              f"{np.mean(vals)/20:>8.4f}")

    # Per type
    print(f"\n  Per type:")
    type_counts = {}
    type_rmse = {}
    for sid in library:
        stype = library[sid]['subject_type']
        if stype not in type_counts:
            type_counts[stype] = 0
            type_rmse[stype] = []
        type_counts[stype] += 1
        type_rmse[stype].append(library[sid]['fit_rmse'])

    print(f"    {'Type':<14s} {'n':>4s} {'RMSE':>8s} {'Norm':>8s}")
    print(f"    {'-' * 36}")
    for stype in sorted(type_counts.keys()):
        vals = type_rmse[stype]
        print(f"    {stype:<14s} {len(vals):>4d} {np.mean(vals):>8.2f} "
              f"{np.mean(vals)/20:>8.4f}")

    # ================================================================
    # PARAMETER PROFILES
    # ================================================================

    print(f"\n{'=' * 72}")
    print(f"  PARAMETER PROFILES")
    print(f"{'=' * 72}")

    # Compare with PGG P-experiment library
    pgg_lib = json.load(open('v3_library_fitted.json'))

    print(f"\n  {'Parameter':<20s} {'PGG-P':>10s} {'PGG-N':>10s} "
          f"{'N-decl':>10s} {'N-stab-hi':>10s} {'N-stab-lo':>10s}")
    print(f"  {'-' * 72}")

    for pname in FIT_PARAM_NAMES:
        pgg_vals = [rec['v3_params'].get(pname, 0)
                    for rec in pgg_lib.values()
                    if pname in rec['v3_params']]
        n_vals = [library[sid]['v4_params'][pname] for sid in library]
        decl_vals = [library[sid]['v4_params'][pname] for sid in library
                     if library[sid]['subject_type'] == 'declining']
        stab_hi = [library[sid]['v4_params'][pname] for sid in library
                   if library[sid]['subject_type'] == 'stable-high']
        stab_lo = [library[sid]['v4_params'][pname] for sid in library
                   if library[sid]['subject_type'] == 'stable-low']

        row = f"  {pname:<20s}"
        row += f" {np.mean(pgg_vals):>10.3f}" if pgg_vals else f" {'—':>10s}"
        row += f" {np.mean(n_vals):>10.3f}" if n_vals else f" {'—':>10s}"
        row += f" {np.mean(decl_vals):>10.3f}" if decl_vals else f" {'—':>10s}"
        row += f" {np.mean(stab_hi):>10.3f}" if stab_hi else f" {'—':>10s}"
        row += f" {np.mean(stab_lo):>10.3f}" if stab_lo else f" {'—':>10s}"
        print(row)

    # ================================================================
    # COMBINED LIBRARY SIZE
    # ================================================================

    ipd_lib = {}
    try:
        ipd_lib = json.load(open('ipd_library_fitted.json'))
    except FileNotFoundError:
        pass

    print(f"\n{'=' * 72}")
    print(f"  LIBRARY COVERAGE")
    print(f"{'=' * 72}")
    print(f"  PGG P-experiment: {len(pgg_lib):>4d} subjects")
    print(f"  PGG N-experiment: {len(library):>4d} subjects (NEW)")
    print(f"  IPD:              {len(ipd_lib):>4d} subjects")
    print(f"  Total:            {len(pgg_lib) + len(library) + len(ipd_lib):>4d} subjects")

    # ================================================================
    # SAVE
    # ================================================================

    out_path = 'n_library_fitted.json'
    with open(out_path, 'w') as f:
        json.dump(library, f, indent=2)
    print(f"\n  N-experiment library saved to {out_path} ({len(library)} subjects)")

    print(f"\n{'=' * 72}")
    print(f"  FITTING COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
