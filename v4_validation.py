#!/usr/bin/env python3
"""
v4 Engine Validation
====================

1. BACKWARD COMPATIBILITY: v4 with PGG_P_CONFIG + s_exploitation_rate=0
   must produce identical predictions to v3 on all library subjects.

2. N-EXPERIMENT ABLATION: Four independent changes, tested separately:
   a. Discharge fix only (gate=0 in no-punishment game)
   b. S_exploitation only (new strain channel)
   c. Bandwidth scaling only (faster alpha with fewer signals)
   d. All combined

3. DECLINE RATE RATIO: Does the 0.67 rate ratio improve?

4. SHAPE ANALYSIS: Does shape classification accuracy change?
"""

import json
import math
import time
import numpy as np
from collections import defaultdict

from pgg_p_loader import load_p_experiment
from pgg_vcms_agent_v3 import (
    VCMSParams as V3Params,
    run_vcms_agent as run_v3,
    MAX_CONTRIB,
)
from vcms_engine_v4 import (
    VCMSParams as V4Params,
    GameConfig, PGG_P_CONFIG, PGG_N_CONFIG, PGG_N_BW_CONFIG,
    run_vcms_v4, v3_params_to_v4,
)
from v3_n_experiment_test import (
    load_n_experiment_data, load_library,
    carry_forward_predict, rmse_window,
)
from v3_n_experiment_diagnostics import classify_curvature, compute_slope


# ================================================================
# ROUND RECONSTRUCTION
# ================================================================

class SimpleRound:
    """Minimal round data for agent forward pass."""
    def __init__(self, contribution, others_mean, pun_sent, pun_recv):
        self.contribution = contribution
        self.others_mean = others_mean
        self.punishment_sent_total = pun_sent
        self.punishment_received_total = pun_recv


def library_to_rounds(rec):
    """Reconstruct round data from library entry."""
    rounds = []
    n = len(rec['contribution_trajectory'])
    for t in range(n):
        rounds.append(SimpleRound(
            contribution=rec['contribution_trajectory'][t],
            others_mean=rec['others_mean_trajectory'][t],
            pun_sent=rec['punishment_sent_trajectory'][t],
            pun_recv=rec['punishment_received_trajectory'][t],
        ))
    return rounds


# ================================================================
# ENSEMBLE PREDICTOR (v4)
# ================================================================

def vcms_v4_ensemble_predict(target_rounds, library, game_config,
                              s_exploitation_rate=0.0, v_self_weight=0.0):
    """
    v4 ensemble predictor — same elimination logic as v3, using v4 engine.
    """
    candidates = {}
    for sid, rec in library.items():
        candidates[sid] = v3_params_to_v4(
            rec['v3_params'],
            s_exploitation_rate=s_exploitation_rate,
            v_self_weight=v_self_weight,
        )

    survivors = list(candidates.keys())
    pred_c_list = []
    obs_c_list = []
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}

    n = min(10, len(target_rounds))
    for t in range(n):
        cand_preds_c = {}
        for sid in survivors:
            result = run_vcms_v4(
                candidates[sid], target_rounds[:t + 1], game_config)
            cand_preds_c[sid] = result['pred_contrib'][t]
            cand_pred_history[sid] = list(result['pred_contrib'][:t + 1])

        if not distances:
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001)
                       for sid in survivors}
        total_w = sum(weights.values())

        pc = sum((weights[sid] / total_w) * cand_preds_c[sid] for sid in survivors)
        pred_c_list.append(int(round(pc)))
        obs_c_list.append(target_rounds[t].contribution)

        new_distances = {}
        for sid in survivors:
            hist = cand_pred_history[sid]
            nc = min(t + 1, len(hist))
            if nc == 0:
                new_distances[sid] = 0.0
                continue
            c_d = sum((obs_c_list[i] - hist[i]) ** 2
                      for i in range(nc)) / (MAX_CONTRIB ** 2 * nc)
            new_distances[sid] = math.sqrt(c_d)

        if new_distances:
            best = min(new_distances.values())
            thresh = max(best * 3.0, 0.5)
        else:
            thresh = 0.5

        new_surv = [s for s in survivors if new_distances.get(s, 999) <= thresh]
        if not new_surv and new_distances:
            new_surv = [min(new_distances, key=new_distances.get)]

        survivors = new_surv
        distances = {s: new_distances[s] for s in survivors}

    return pred_c_list


# ================================================================
# TEST 1: BACKWARD COMPATIBILITY
# ================================================================

def test_backward_compat(library):
    """v4 with PGG_P_CONFIG + s_exploitation_rate=0 must match v3 exactly."""
    print("\n" + "=" * 72)
    print("  TEST 1: BACKWARD COMPATIBILITY (v4 == v3)")
    print("=" * 72)

    mismatches = 0
    total = 0
    max_diff_c = 0
    max_diff_p = 0

    for sid, rec in library.items():
        rounds = library_to_rounds(rec)

        # v3
        v3_params = V3Params(**rec['v3_params'])
        v3_result = run_v3(v3_params, rounds)

        # v4 with zero exploitation, P-experiment config
        v4_params = v3_params_to_v4(rec['v3_params'],
                                     s_exploitation_rate=0.0,
                                     v_self_weight=0.0)
        v4_result = run_vcms_v4(v4_params, rounds, PGG_P_CONFIG)

        # Compare
        for t in range(len(rounds)):
            total += 1
            c3, c4 = v3_result['pred_contrib'][t], v4_result['pred_contrib'][t]
            p3, p4 = v3_result['pred_punish'][t], v4_result['pred_punish'][t]
            if c3 != c4 or p3 != p4:
                mismatches += 1
                max_diff_c = max(max_diff_c, abs(c3 - c4))
                max_diff_p = max(max_diff_p, abs(p3 - p4))

    if mismatches == 0:
        print(f"\n  PASS: {total} round-predictions across {len(library)} subjects — "
              f"all identical")
    else:
        print(f"\n  FAIL: {mismatches}/{total} mismatches "
              f"(max diff: C={max_diff_c}, P={max_diff_p})")

    return mismatches == 0


# ================================================================
# TEST 2: N-EXPERIMENT ABLATION
# ================================================================

def test_n_experiment_ablation(n_rounds, library, city_map):
    """Test each v4 change independently on N-experiment data."""
    print("\n" + "=" * 72)
    print("  TEST 2: N-EXPERIMENT ABLATION")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    declining = set(sid for sid in sids
                    if compute_slope([r.contribution for r in n_rounds[sid]]) < -0.3)
    nondecl = set(sids) - declining

    # Define variants
    # Each variant: (label, game_config, s_exploitation_rate, v_self_weight)
    S_EXPLOIT_RATE = 1.0
    V_SELF_WEIGHT = 1.0

    variants = [
        # Pure discharge fix (gate=0 in N-experiment, no S_exploitation, no BW scaling)
        ('discharge_fix',   PGG_N_CONFIG,    0.0,             0.0),
        # S_exploitation only (with discharge fix since PGG_N_CONFIG has_punishment=False)
        ('s_exploit',       PGG_N_CONFIG,    S_EXPLOIT_RATE,  V_SELF_WEIGHT),
        # Bandwidth scaling only (with discharge fix)
        ('bandwidth',       PGG_N_BW_CONFIG, 0.0,             0.0),
        # All combined
        ('full_v4',         PGG_N_BW_CONFIG, S_EXPLOIT_RATE,  V_SELF_WEIGHT),
    ]

    # Also need v3 baseline (already computed, but let's recompute for consistency)
    # v3 on N-experiment = v4 with PGG_P_CONFIG (has_punishment=True, gate fires normally)
    # Actually: v3 treats N-experiment identically to P except punishment=0
    # So v3 baseline = v4 with has_punishment=True, s_exploit=0
    variants.insert(0, ('v3_baseline', PGG_P_CONFIG, 0.0, 0.0))

    results = {label: {'declining': [], 'nondecl': [], 'all': []}
               for label, _, _, _ in variants}

    print(f"\n  {len(sids)} subjects ({len(declining)} declining, {len(nondecl)} non-declining)")
    print(f"  Variants: {[v[0] for v in variants]}")

    t0 = time.time()
    for idx, sid in enumerate(sids):
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]
        is_decl = sid in declining

        for label, gc, s_rate, vsw in variants:
            preds = vcms_v4_ensemble_predict(
                rounds, library, gc,
                s_exploitation_rate=s_rate, v_self_weight=vsw)
            err = rmse_window(preds, actual, 1)
            results[label]['all'].append(err)
            if is_decl:
                results[label]['declining'].append(err)
            else:
                results[label]['nondecl'].append(err)

        if (idx + 1) % 30 == 0 or idx == len(sids) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(sids) - idx - 1)
            print(f"    [{idx+1:>3d}/{len(sids)}] {elapsed:>5.0f}s elapsed, ~{eta:>4.0f}s remaining")

    # CF baseline
    cf_all = []
    cf_decl = []
    cf_nondecl = []
    for sid in sids:
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]
        cf = carry_forward_predict(rounds)
        err = rmse_window(cf, actual, 1)
        cf_all.append(err)
        if sid in declining:
            cf_decl.append(err)
        else:
            cf_nondecl.append(err)

    # Results table
    print(f"\n  {'Variant':<18s} {'All':>8s} {'Decl':>8s} {'NonDecl':>8s} "
          f"{'Δ All':>8s} {'Δ Decl':>8s} {'Δ NonDecl':>10s}")
    print("  " + "-" * 70)
    print(f"  {'CF':<18s} {np.mean(cf_all):>8.4f} {np.mean(cf_decl):>8.4f} "
          f"{np.mean(cf_nondecl):>8.4f} {'—':>8s} {'—':>8s} {'—':>10s}")

    for label, _, _, _ in variants:
        all_m = np.mean(results[label]['all'])
        decl_m = np.mean(results[label]['declining'])
        nondecl_m = np.mean(results[label]['nondecl'])
        d_all = all_m - np.mean(cf_all)
        d_decl = decl_m - np.mean(cf_decl)
        d_nondecl = nondecl_m - np.mean(cf_nondecl)
        print(f"  {label:<18s} {all_m:>8.4f} {decl_m:>8.4f} {nondecl_m:>8.4f} "
              f"{d_all:>+8.4f} {d_decl:>+8.4f} {d_nondecl:>+10.4f}")

    return results, cf_all, cf_decl, cf_nondecl


# ================================================================
# TEST 3: S_EXPLOITATION RATE SWEEP
# ================================================================

def test_srate_sweep(n_rounds, library, city_map):
    """Sweep s_exploitation_rate with fixed v_self_weight=1.0."""
    print("\n" + "=" * 72)
    print("  TEST 3: S_EXPLOITATION RATE SWEEP (v_self_weight=1.0)")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    declining = set(sid for sid in sids
                    if compute_slope([r.contribution for r in n_rounds[sid]]) < -0.3)

    sweep_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    gc = PGG_N_BW_CONFIG  # full v4 config (discharge fix + bandwidth)

    results = {v: {'all': [], 'declining': []} for v in sweep_values}

    t0 = time.time()
    for idx, sid in enumerate(sids):
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]
        is_decl = sid in declining

        for rate in sweep_values:
            preds = vcms_v4_ensemble_predict(
                rounds, library, gc,
                s_exploitation_rate=rate, v_self_weight=1.0)
            err = rmse_window(preds, actual, 1)
            results[rate]['all'].append(err)
            if is_decl:
                results[rate]['declining'].append(err)

        if (idx + 1) % 30 == 0 or idx == len(sids) - 1:
            elapsed = time.time() - t0
            print(f"    [{idx+1:>3d}/{len(sids)}] {elapsed:.0f}s")

    print(f"\n  {'Rate':>6s} {'All RMSE':>9s} {'Decl RMSE':>10s}")
    print("  " + "-" * 30)
    for rate in sweep_values:
        all_m = np.mean(results[rate]['all'])
        decl_m = np.mean(results[rate]['declining'])
        print(f"  {rate:>6.2f} {all_m:>9.4f} {decl_m:>10.4f}")

    return results


# ================================================================
# TEST 4: DECLINE RATE RATIO + SHAPE ANALYSIS
# ================================================================

def test_decline_analysis(n_rounds, library, city_map):
    """Rate ratio and shape classification under best v4 variant."""
    print("\n" + "=" * 72)
    print("  TEST 4: DECLINE RATE RATIO + SHAPE ANALYSIS")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    declining = [sid for sid in sids
                 if compute_slope([r.contribution for r in n_rounds[sid]]) < -0.3]

    gc = PGG_N_BW_CONFIG

    # Compare v3 baseline vs full v4
    configs = [
        ('v3_baseline', PGG_P_CONFIG, 0.0, 0.0),
        ('full_v4',     gc,           1.0, 1.0),
    ]

    for label, game_config, s_rate, vsw in configs:
        rate_ratios = []
        shape_match = 0
        shape_total = 0
        shape_confusion = defaultdict(lambda: defaultdict(int))

        for sid in declining:
            rounds = n_rounds[sid]
            actual = [r.contribution for r in rounds]

            preds = vcms_v4_ensemble_predict(
                rounds, library, game_config,
                s_exploitation_rate=s_rate, v_self_weight=vsw)

            # Rate ratio
            actual_slope = compute_slope(actual)
            pred_slope = compute_slope(preds)
            if abs(actual_slope) > 0.1:
                rate_ratios.append(pred_slope / actual_slope)

            # Shape
            actual_shape, _, _ = classify_curvature(actual)
            pred_shape, _, _ = classify_curvature(preds)
            shape_total += 1
            if actual_shape == pred_shape:
                shape_match += 1
            shape_confusion[actual_shape][pred_shape] += 1

        print(f"\n  --- {label} ---")
        print(f"  Rate ratio (pred_slope / actual_slope):")
        print(f"    median={np.median(rate_ratios):.3f}, mean={np.mean(rate_ratios):.3f}")
        print(f"    in [0.8,1.2]: {sum(1 for r in rate_ratios if 0.8 <= r <= 1.2)}/{len(rate_ratios)}")
        print(f"    in [0.5,1.5]: {sum(1 for r in rate_ratios if 0.5 <= r <= 1.5)}/{len(rate_ratios)}")

        print(f"  Shape classification: {shape_match}/{shape_total} "
              f"({100*shape_match/shape_total:.0f}%)")

        all_shapes = ['concave', 'linear', 'convex']
        header = f"    {'':>10s}"
        for s in all_shapes:
            header += f" {s:>10s}"
        print(header)
        for actual_s in all_shapes:
            row = f"    {actual_s:>10s}"
            for pred_s in all_shapes:
                row += f" {shape_confusion[actual_s][pred_s]:>10d}"
            print(row)


# ================================================================
# TEST 5: V_SELF_WEIGHT SENSITIVITY
# ================================================================

def test_vsw_sensitivity(n_rounds, library, city_map):
    """Check if v_self_weight matters independently from s_exploitation_rate."""
    print("\n" + "=" * 72)
    print("  TEST 5: V_SELF_WEIGHT SENSITIVITY (s_exploitation_rate=1.0)")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    gc = PGG_N_BW_CONFIG

    vsw_values = [0.25, 0.5, 0.75, 1.0]
    results = {v: [] for v in vsw_values}

    t0 = time.time()
    for idx, sid in enumerate(sids):
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]

        for vsw in vsw_values:
            preds = vcms_v4_ensemble_predict(
                rounds, library, gc,
                s_exploitation_rate=1.0, v_self_weight=vsw)
            err = rmse_window(preds, actual, 1)
            results[vsw].append(err)

        if (idx + 1) % 50 == 0 or idx == len(sids) - 1:
            elapsed = time.time() - t0
            print(f"    [{idx+1:>3d}/{len(sids)}] {elapsed:.0f}s")

    print(f"\n  {'VSW':>6s} {'RMSE':>8s}")
    print("  " + "-" * 18)
    for vsw in vsw_values:
        print(f"  {vsw:>6.2f} {np.mean(results[vsw]):>8.4f}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  VCMS v4 VALIDATION")
    print("=" * 72)

    library = load_library()
    n_rounds, city_map = load_n_experiment_data()
    print(f"  Library: {len(library)} P-experiment subjects")
    print(f"  Test:    {len(n_rounds)} N-experiment subjects")

    # 1. Backward compat
    compat = test_backward_compat(library)
    if not compat:
        print("\n  *** BACKWARD COMPAT FAILED — stopping ***")
        return

    # 2. N-experiment ablation
    test_n_experiment_ablation(n_rounds, library, city_map)

    # 3. S_exploitation rate sweep
    test_srate_sweep(n_rounds, library, city_map)

    # 4. Decline analysis
    test_decline_analysis(n_rounds, library, city_map)

    # 5. V_self_weight sensitivity
    test_vsw_sensitivity(n_rounds, library, city_map)

    print("\n" + "=" * 72)
    print("  VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
