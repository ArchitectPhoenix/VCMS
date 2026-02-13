#!/usr/bin/env python3
"""
N-experiment Re-fit with Normalized Game Time
===============================================

Re-fits 212 N-experiment subjects using normalized_time=True.
10-round games: dt = 1/9 ≈ 0.111 per round.

Output: v4_n_library_fitted.json
"""

import json
import math
import time
import numpy as np

from pgg_p_loader import load_p_experiment
from vcms_engine_v4 import GameConfig, VCMSParams, run_vcms_v4
from normalized_fit import (
    predict_fast_normalized, objective, fit_subject_de,
    FIT_PARAM_NAMES, FIXED_PARAMS, params_array_to_dict,
)
from v3_n_experiment_test import N_EXPERIMENT_CSVS

import os


# N-experiment config with normalized time
PGG_N_NORM_CONFIG = GameConfig(
    max_contrib=20, max_punish=30, has_punishment=False,
    n_signals=2, normalized_time=True,
)


def load_n_experiment_data():
    """Load all N-experiment subjects with city labels."""
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


def verify_fit(params_dict, rounds, actual):
    """Verify through official v4 engine with normalized time."""
    v4_params = VCMSParams(**params_dict)
    result = run_vcms_v4(v4_params, rounds, PGG_N_NORM_CONFIG)
    preds = result['pred_contrib']
    n = len(actual)
    rmse = math.sqrt(sum((actual[i] - preds[i]) ** 2 for i in range(n)) / n)
    return rmse, preds


def main():
    print("=" * 72)
    print("  N-EXPERIMENT RE-FIT — NORMALIZED TIME")
    print("=" * 72)

    print("\nLoading N-experiment data...")
    n_data, city_map = load_n_experiment_data()
    print(f"  {len(n_data)} subjects across {len(set(city_map.values()))} cities")

    cities = sorted(set(city_map.values()))
    for city in cities:
        n = sum(1 for c in city_map.values() if c == city)
        print(f"    {city}: {n} subjects")

    total = len(n_data)
    n_rounds = 10
    print(f"\n  Config: normalized_time=True, dt=1/{n_rounds-1}={1/(n_rounds-1):.4f}")
    print(f"  Parameters: {len(FIT_PARAM_NAMES)} free, {len(FIXED_PARAMS)} fixed")

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
        best_x, rmse, nfev = fit_subject_de(rounds, actual, max_c=20, seed=42)
        params_dict = params_array_to_dict(best_x)

        # Verify
        verified_rmse, verified_preds = verify_fit(params_dict, rounds, actual)

        city = city_map[sid]

        library[sid] = {
            'treatment': 'N-experiment',
            'city': city,
            'subject_type': stype,
            'v3_params': params_dict,  # v3_params key for transfer test compat
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
            'engine': 'v4_normalized_time',
            'dt': 1.0 / (len(rounds) - 1),
        }

        all_rmse.append(verified_rmse)
        city_rmse[city].append(verified_rmse)

        if (idx + 1) % 20 == 0 or idx == total - 1:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"  [{idx + 1:>3d}/{total}] {sid:<12s} ({city:<10s} {stype:<12s}) "
                  f"rmse={verified_rmse:.2f} "
                  f"({elapsed:>5.0f}s, ~{eta:>4.0f}s rem)")

    total_time = time.time() - t0
    print(f"\n  Fitting complete: {total_time:.0f}s "
          f"({total_time / total:.1f}s per subject)")

    # Quality report
    print(f"\n{'=' * 72}")
    print(f"  FITTING QUALITY")
    print(f"{'=' * 72}")

    print(f"\n  Overall: RMSE={np.mean(all_rmse):.2f} "
          f"(normalized={np.mean(all_rmse)/20:.4f})")

    print(f"\n  Per city:")
    print(f"    {'City':<12s} {'n':>4s} {'RMSE':>8s} {'Norm':>8s}")
    print(f"    {'-' * 36}")
    for city in cities:
        vals = city_rmse[city]
        print(f"    {city:<12s} {len(vals):>4d} {np.mean(vals):>8.2f} "
              f"{np.mean(vals)/20:>8.4f}")

    # Compare with v3 N-experiment library
    v3_path = 'n_library_fitted.json'
    if os.path.exists(v3_path):
        with open(v3_path) as f:
            v3_lib = json.load(f)
        shared = [sid for sid in library if sid in v3_lib]
        v3_rmses = [v3_lib[sid]['fit_rmse'] for sid in shared]
        v4_rmses = [library[sid]['fit_rmse'] for sid in shared]
        print(f"\n  Comparison with v3 N-library ({len(shared)} shared):")
        print(f"    v3 mean RMSE: {np.mean(v3_rmses):.4f}")
        print(f"    v4 mean RMSE: {np.mean(v4_rmses):.4f}")
        print(f"    Delta:        {np.mean(v4_rmses) - np.mean(v3_rmses):+.4f}")

        better = sum(1 for a, b in zip(v4_rmses, v3_rmses) if a < b - 0.01)
        same = sum(1 for a, b in zip(v4_rmses, v3_rmses) if abs(a - b) <= 0.01)
        worse = sum(1 for a, b in zip(v4_rmses, v3_rmses) if a > b + 0.01)
        print(f"    Better: {better}, Same: {same}, Worse: {worse}")

        # Parameter scale comparison
        print(f"\n  Parameter scale comparison (mean values):")
        scale_params = ['s_rate', 'b_depletion_rate', 'b_replenish_rate',
                        'facilitation_rate', 'h_start']
        for pname in scale_params:
            v3_vals = [v3_lib[sid]['v4_params'][pname] for sid in shared
                       if pname in v3_lib[sid]['v4_params']]
            v4_vals = [library[sid]['v3_params'][pname] for sid in shared]
            if v3_vals:
                ratio = np.mean(v4_vals) / max(np.mean(v3_vals), 0.001)
                print(f"    {pname:>20s}: v3={np.mean(v3_vals):.3f}, "
                      f"v4={np.mean(v4_vals):.3f}, ratio={ratio:.1f}x")

    # Save
    out_path = 'v4_n_library_fitted.json'
    with open(out_path, 'w') as f:
        json.dump(library, f, indent=2)
    print(f"\n  Saved to {out_path} ({len(library)} subjects)")

    print(f"\n{'=' * 72}")
    print(f"  FITTING COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
