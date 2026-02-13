#!/usr/bin/env python3
"""
IPD Re-fit with Normalized Game Time
======================================

Re-fits 188 IPD subjects using normalized_time=True.
100-round games: dt = 1/99 ≈ 0.0101 per round.

Output: v4_ipd_library_fitted.json
"""

import json
import math
import time
import numpy as np

from ipd_loader import load_ipd_experiment
from vcms_engine_v4 import GameConfig, VCMSParams, run_vcms_v4
from normalized_fit import (
    predict_fast_normalized, objective, fit_subject_de,
    FIT_PARAM_NAMES, FIXED_PARAMS, params_array_to_dict,
)


# IPD config with normalized time
IPD_NORM_CONFIG = GameConfig(
    max_contrib=1, max_punish=1, has_punishment=False,
    n_signals=2, normalized_time=True,
)


def verify_fit(params_dict, rounds, actual):
    """Verify through official v4 engine with normalized time."""
    v4_params = VCMSParams(**params_dict)
    result = run_vcms_v4(v4_params, rounds, IPD_NORM_CONFIG)
    preds = result['pred_contrib']
    n = len(actual)
    correct = sum(1 for i in range(n) if preds[i] == actual[i])
    return correct / n, preds


def main():
    print("=" * 72)
    print("  IPD RE-FIT — NORMALIZED TIME")
    print("=" * 72)

    print("\nLoading IPD data...")
    sp_data = load_ipd_experiment('IPD-rand.csv')
    fp_data = load_ipd_experiment('fix.csv')
    print(f"  SP: {len(sp_data)} subjects")
    print(f"  FP: {len(fp_data)} subjects")

    # Combine with treatment labels
    subjects = {}
    for sid, rounds in sp_data.items():
        subjects[f"SP_{sid}"] = {'rounds': rounds, 'treatment': 'SP'}
    for sid, rounds in fp_data.items():
        subjects[f"FP_{sid}"] = {'rounds': rounds, 'treatment': 'FP'}

    total = len(subjects)
    n_rounds = 100
    print(f"\n  Config: normalized_time=True, dt=1/{n_rounds-1}={1/(n_rounds-1):.4f}")
    print(f"  Parameters: {len(FIT_PARAM_NAMES)} free, {len(FIXED_PARAMS)} fixed")
    print(f"  Total: {total} subjects to fit")

    library = {}
    all_acc = []
    all_rmse = []
    sp_acc = []
    fp_acc = []
    type_acc = {'mostly-D': [], 'mixed': [], 'mostly-C': []}

    t0 = time.time()
    sids = sorted(subjects.keys())

    for idx, sid in enumerate(sids):
        info = subjects[sid]
        rounds = info['rounds']
        actual = [r.contribution for r in rounds]
        coop_rate = sum(actual) / len(actual)

        # Fit
        best_x, rmse, nfev = fit_subject_de(rounds, actual, max_c=1, seed=42)
        params_dict = params_array_to_dict(best_x)

        # Verify
        verified_acc, verified_preds = verify_fit(params_dict, rounds, actual)

        # Classify
        if coop_rate < 0.2:
            stype = 'mostly-D'
        elif coop_rate > 0.8:
            stype = 'mostly-C'
        else:
            stype = 'mixed'

        library[sid] = {
            'treatment': info['treatment'],
            'subject_type': stype,
            'v3_params': params_dict,  # v3_params key for transfer test compat
            'v4_params': params_dict,
            'contribution_trajectory': actual,
            'others_mean_trajectory': [r.others_mean for r in rounds],
            'punishment_sent_trajectory': [0] * len(rounds),
            'punishment_received_trajectory': [0] * len(rounds),
            'fit_rmse': float(rmse),
            'fit_accuracy': float(verified_acc),
            'fit_pred': verified_preds,
            'cooperation_rate': float(coop_rate),
            'engine': 'v4_normalized_time',
            'dt': 1.0 / (len(rounds) - 1),
        }

        all_acc.append(verified_acc)
        all_rmse.append(rmse)
        if info['treatment'] == 'SP':
            sp_acc.append(verified_acc)
        else:
            fp_acc.append(verified_acc)
        type_acc[stype].append(verified_acc)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"  [{idx + 1:>3d}/{total}] {sid:<25s} "
                  f"acc={verified_acc:.0%} rmse={rmse:.4f} "
                  f"({elapsed:>5.0f}s, ~{eta:>4.0f}s rem)")

    total_time = time.time() - t0
    print(f"\n  Fitting complete: {total_time:.0f}s "
          f"({total_time / total:.1f}s per subject)")

    # Quality report
    print(f"\n{'=' * 72}")
    print(f"  FITTING QUALITY")
    print(f"{'=' * 72}")

    print(f"\n  Overall: acc={np.mean(all_acc):.1%} "
          f"(median={np.median(all_acc):.1%}), "
          f"rmse={np.mean(all_rmse):.4f}")
    print(f"  SP:      acc={np.mean(sp_acc):.1%}")
    print(f"  FP:      acc={np.mean(fp_acc):.1%}")

    print(f"\n  By type:")
    for stype in ['mostly-D', 'mixed', 'mostly-C']:
        vals = type_acc[stype]
        if vals:
            print(f"    {stype:<10s}: acc={np.mean(vals):.1%} "
                  f"(median={np.median(vals):.1%}, n={len(vals)})")

    # Compare with v3 IPD library
    import os
    v3_path = 'ipd_library_fitted.json'
    if os.path.exists(v3_path):
        with open(v3_path) as f:
            v3_lib = json.load(f)
        shared = [sid for sid in library if sid in v3_lib]
        v3_accs = [v3_lib[sid]['fit_accuracy'] for sid in shared]
        v4_accs = [library[sid]['fit_accuracy'] for sid in shared]
        print(f"\n  Comparison with v3 IPD library ({len(shared)} shared):")
        print(f"    v3 mean accuracy: {np.mean(v3_accs):.1%}")
        print(f"    v4 mean accuracy: {np.mean(v4_accs):.1%}")
        print(f"    Delta:            {np.mean(v4_accs) - np.mean(v3_accs):+.1%}")

        # Parameter scale comparison
        print(f"\n  Parameter scale comparison (mean values):")
        scale_params = ['s_rate', 'b_depletion_rate', 'b_replenish_rate',
                        'facilitation_rate', 'h_start']
        for pname in scale_params:
            v3_vals = [v3_lib[sid]['v4_params'][pname] for sid in shared
                       if pname in v3_lib[sid].get('v4_params', {})]
            v4_vals = [library[sid]['v3_params'][pname] for sid in shared]
            if v3_vals:
                ratio = np.mean(v4_vals) / max(np.mean(v3_vals), 0.001)
                print(f"    {pname:>20s}: v3={np.mean(v3_vals):.3f}, "
                      f"v4={np.mean(v4_vals):.3f}, ratio={ratio:.1f}x")

    # Save
    out_path = 'v4_ipd_library_fitted.json'
    with open(out_path, 'w') as f:
        json.dump(library, f, indent=2)
    print(f"\n  Saved to {out_path} ({len(library)} subjects)")

    print(f"\n{'=' * 72}")
    print(f"  FITTING COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
