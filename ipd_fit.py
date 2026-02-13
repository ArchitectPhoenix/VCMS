#!/usr/bin/env python3
"""
IPD Subject Fitting
===================

Fit VCMS v4 parameters to individual IPD subjects. Builds an IPD library
that fills the coverage gap identified in the transfer test — the PGG
library lacks "persistent cooperator under adversity" profiles.

Approach:
- Fast forward pass (no traces, no s_exploitation) for optimizer speed
- 15 free parameters, 5 fixed (punishment-only params + s_exploitation)
- scipy.optimize.differential_evolution per subject
- Saves to ipd_library_fitted.json

Fixed parameters (irrelevant for no-punishment games):
  s_frac=0.677, s_thresh=1.800, p_scale=5.830 (PGG library medians)
  v_self_weight=0.0, s_exploitation_rate=0.0
"""

import json
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

from ipd_loader import load_ipd_experiment
from vcms_engine_v4 import (
    IPD_CONFIG, VCMSParams, PARAM_BOUNDS,
    run_vcms_v4, v3_params_to_v4,
)


# ================================================================
# FIXED PARAMETERS (irrelevant for IPD, set to PGG library medians)
# ================================================================

FIXED_PARAMS = {
    's_frac': 0.677,
    's_thresh': 1.800,
    'p_scale': 5.830,
    'v_self_weight': 0.0,
    's_exploitation_rate': 0.0,
}

# Parameters to optimize (15 free)
FIT_PARAM_NAMES = [
    'alpha', 'v_rep', 'v_ref', 'c_base', 'inertia',
    's_dir', 's_rate', 's_initial',
    'b_initial', 'b_depletion_rate', 'b_replenish_rate', 'acute_threshold',
    'facilitation_rate', 'h_strength', 'h_start',
]

FIT_BOUNDS = [(PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1]) for p in FIT_PARAM_NAMES]


# ================================================================
# FAST FORWARD PASS (no traces, no s_exploitation, inlined horizon)
# ================================================================

def predict_fast(x, rounds, max_c):
    """
    Minimal VCMS forward pass for optimizer.
    ~10x faster than run_vcms_v4 (no dict allocation, no trace storage).
    """
    alpha, v_rep, v_ref, c_base, inertia, s_dir_raw, s_rate, s_initial, \
        b_initial, b_depletion_rate, b_replenish_rate, acute_threshold, \
        facilitation_rate, h_strength, h_start = x

    s_dir = 1.0 if s_dir_raw >= 0 else -1.0
    w = max(-0.3, min(0.95, inertia))

    v_level = 0.0
    disposition = 0.0
    strain = s_initial
    B = b_initial
    m_eval = 0.0
    c_prev_norm = 0.0

    n = len(rounds)
    preds = [0] * n

    for i in range(n):
        rd = rounds[i]
        v_group_raw = rd.others_mean / max_c
        v_group = min(1.0, v_rep * v_group_raw)

        if i == 0:
            v_level = v_group
            disposition = rd.contribution / max_c
        else:
            v_level = alpha * v_group + (1.0 - alpha) * v_level
            disposition = 0.15 * c_prev_norm + 0.85 * disposition

        reference = v_ref * v_level + (1.0 - v_ref) * disposition

        # Strain
        if i > 0:
            gap = c_prev_norm - reference
            directed_gap = gap * s_dir
            gap_strain = max(0.0, directed_gap)
            strain += s_rate * gap_strain

        # Budget
        if i > 0:
            experience = v_group_raw - c_prev_norm
            if experience < 0:
                magnitude = -experience
                depletion = b_depletion_rate * magnitude
                if magnitude > acute_threshold:
                    depletion *= 5.0
                B -= depletion
            elif experience > 0:
                B += b_replenish_rate * experience
            B = max(0.0, B)
            m_eval += facilitation_rate * experience

        # Affordability (gate=0 for no-punishment, discharge=0)
        affordability = B / (B + strain + 0.01)

        # Contribution
        if i == 0:
            c_norm = c_base
        else:
            c_target = v_ref * v_level + (1.0 - v_ref) * c_base
            c_target_adj = max(0.0, min(1.0, c_target + m_eval))
            c_norm = (1.0 - abs(w)) * c_target_adj + w * c_prev_norm

        # Horizon
        if n > 1 and h_strength > 0.0 and i >= h_start:
            denom = n - 1 - h_start
            if denom > 0:
                progress = min(1.0, (i - h_start) / denom)
                h_factor = 1.0 - h_strength * progress
            else:
                h_factor = (1.0 - h_strength) if i >= n - 1 else 1.0
        else:
            h_factor = 1.0

        c_out_norm = max(0.0, min(1.0, c_norm)) * affordability * h_factor
        c_out = round(c_out_norm * max_c)
        preds[i] = max(0, min(max_c, c_out))

        # State update (teacher forcing)
        c_prev_norm = rd.contribution / max_c

    return preds


# ================================================================
# OBJECTIVE FUNCTION
# ================================================================

def objective(x, rounds, actual, max_c):
    """RMSE of contribution predictions. For binary: sqrt(1-accuracy)."""
    preds = predict_fast(x, rounds, max_c)
    n = len(actual)
    sse = sum((actual[i] - preds[i]) ** 2 for i in range(n))
    return math.sqrt(sse / n)


# ================================================================
# FIT ONE SUBJECT
# ================================================================

def fit_subject(rounds, actual, max_c=1, seed=42):
    """
    Fit 15 VCMS parameters to one IPD subject via differential evolution.

    Returns (best_params_array, rmse, n_evaluations).
    """
    result = differential_evolution(
        objective, FIT_BOUNDS,
        args=(rounds, actual, max_c),
        maxiter=100, popsize=10, tol=0.005,
        seed=seed, polish=True,
        disp=False,
    )
    return result.x, result.fun, result.nfev


def params_array_to_dict(x):
    """Convert optimizer array to full v4 params dict (15 fitted + 5 fixed)."""
    d = {}
    for i, name in enumerate(FIT_PARAM_NAMES):
        d[name] = float(x[i])
    d.update(FIXED_PARAMS)
    return d


# ================================================================
# VERIFICATION: fitted params reproduce through official engine
# ================================================================

def verify_fit(params_dict, rounds, actual):
    """Run fitted params through the full v4 engine to verify."""
    v4_params = VCMSParams(**params_dict)
    result = run_vcms_v4(v4_params, rounds, IPD_CONFIG)
    preds = result['pred_contrib']
    n = len(actual)
    correct = sum(1 for i in range(n) if preds[i] == actual[i])
    return correct / n, preds


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  IPD SUBJECT FITTING")
    print("  Building VCMS library for IPD subjects")
    print("=" * 72)

    # Load data
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
    print(f"  Total: {total} subjects to fit")
    print(f"  Parameters: {len(FIT_PARAM_NAMES)} free, "
          f"{len(FIXED_PARAMS)} fixed")
    print(f"  Fixed: {FIXED_PARAMS}")

    # Fit all subjects
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
        best_x, rmse, nfev = fit_subject(rounds, actual, max_c=1, seed=42)
        params_dict = params_array_to_dict(best_x)

        # Verify through official engine
        verified_acc, verified_preds = verify_fit(params_dict, rounds, actual)

        # Classify
        if coop_rate < 0.2:
            stype = 'mostly-D'
        elif coop_rate > 0.8:
            stype = 'mostly-C'
        else:
            stype = 'mixed'

        # Store
        library[sid] = {
            'treatment': info['treatment'],
            'subject_type': stype,
            'v4_params': params_dict,
            'contribution_trajectory': actual,
            'others_mean_trajectory': [r.others_mean for r in rounds],
            'punishment_sent_trajectory': [0] * len(rounds),
            'punishment_received_trajectory': [0] * len(rounds),
            'fit_rmse': float(rmse),
            'fit_accuracy': float(verified_acc),
            'fit_pred': verified_preds,
            'cooperation_rate': float(coop_rate),
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

    # ================================================================
    # QUALITY REPORT
    # ================================================================

    print(f"\n{'=' * 72}")
    print(f"  FITTING QUALITY")
    print(f"{'=' * 72}")

    print(f"\n  Overall: acc={np.mean(all_acc):.1%} "
          f"(median={np.median(all_acc):.1%}), "
          f"rmse={np.mean(all_rmse):.4f}")
    print(f"  SP:      acc={np.mean(sp_acc):.1%} "
          f"(median={np.median(sp_acc):.1%})")
    print(f"  FP:      acc={np.mean(fp_acc):.1%} "
          f"(median={np.median(fp_acc):.1%})")

    print(f"\n  By type:")
    for stype in ['mostly-D', 'mixed', 'mostly-C']:
        vals = type_acc[stype]
        if vals:
            print(f"    {stype:<10s}: acc={np.mean(vals):.1%} "
                  f"(median={np.median(vals):.1%}, n={len(vals)})")

    # Accuracy distribution
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    hist, _ = np.histogram(all_acc, bins=bins)
    print(f"\n  Accuracy distribution:")
    for i in range(len(bins) - 1):
        bar = '#' * (hist[i] * 2)
        print(f"    {bins[i]:>4.0%}-{bins[i+1]:>4.0%}: {hist[i]:>3d} {bar}")

    # ================================================================
    # PARAMETER PROFILES
    # ================================================================

    print(f"\n{'=' * 72}")
    print(f"  PARAMETER PROFILES")
    print(f"{'=' * 72}")

    # Compare parameter distributions by subject type
    print(f"\n  {'Parameter':<20s} {'mostly-D':>12s} {'mixed':>12s} "
          f"{'mostly-C':>12s}")
    print(f"  {'-' * 60}")

    for pname in FIT_PARAM_NAMES:
        vals_by_type = {}
        for stype in ['mostly-D', 'mixed', 'mostly-C']:
            vals = [library[sid]['v4_params'][pname]
                    for sid in library
                    if library[sid]['subject_type'] == stype]
            vals_by_type[stype] = vals

        row = f"  {pname:<20s}"
        for stype in ['mostly-D', 'mixed', 'mostly-C']:
            vals = vals_by_type[stype]
            if vals:
                row += f"  {np.mean(vals):>10.3f}"
            else:
                row += f"  {'—':>10s}"
        print(row)

    # Compare with PGG library
    print(f"\n  Comparison with PGG library:")
    pgg_lib = json.load(open('v3_library_fitted.json'))

    print(f"\n  {'Parameter':<20s} {'PGG':>10s} {'IPD-all':>10s} "
          f"{'IPD-D':>10s} {'IPD-C':>10s}")
    print(f"  {'-' * 65}")

    for pname in FIT_PARAM_NAMES:
        pgg_vals = [rec['v3_params'].get(pname, 0)
                    for rec in pgg_lib.values()
                    if pname in rec['v3_params']]
        ipd_vals = [library[sid]['v4_params'][pname] for sid in library]
        ipd_d = [library[sid]['v4_params'][pname] for sid in library
                 if library[sid]['subject_type'] == 'mostly-D']
        ipd_c = [library[sid]['v4_params'][pname] for sid in library
                 if library[sid]['subject_type'] == 'mostly-C']

        pgg_m = np.mean(pgg_vals) if pgg_vals else float('nan')
        ipd_m = np.mean(ipd_vals) if ipd_vals else float('nan')
        d_m = np.mean(ipd_d) if ipd_d else float('nan')
        c_m = np.mean(ipd_c) if ipd_c else float('nan')

        print(f"  {pname:<20s} {pgg_m:>10.3f} {ipd_m:>10.3f} "
              f"{d_m:>10.3f} {c_m:>10.3f}")

    # ================================================================
    # SAVE
    # ================================================================

    out_path = 'ipd_library_fitted.json'
    with open(out_path, 'w') as f:
        json.dump(library, f, indent=2)
    print(f"\n  Library saved to {out_path} ({len(library)} subjects)")

    print(f"\n{'=' * 72}")
    print(f"  FITTING COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
