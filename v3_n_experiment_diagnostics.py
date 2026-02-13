#!/usr/bin/env python3
"""
N-experiment deep diagnostics: B-dynamics, decline shapes, stability mechanisms.

Three questions:
1. Does VCMS advantage concentrate in library candidates with cooperation-driven
   vs punishment-driven B-dynamics?
2. For declining subjects: does VCMS get shape right but miss rate?
3. For non-declining subjects: high c_base or low s_rate?
"""

import json
import math
import numpy as np
from collections import defaultdict

from pgg_p_loader import load_p_experiment, PRoundData
from pgg_vcms_agent_v3 import VCMSParams, run_vcms_agent, MAX_CONTRIB

from v3_n_experiment_test import (
    load_n_experiment_data, load_library,
    vcms_predict, carry_forward_predict, rmse_window,
)


# ================================================================
# HELPERS
# ================================================================

def classify_curvature(traj):
    """Classify trajectory shape: concave (decelerating), linear, convex (accelerating)."""
    t = np.arange(len(traj))
    if len(traj) < 3:
        return 'linear', 0.0, 0.0
    coeffs = np.polyfit(t, traj, 2)  # a*t^2 + b*t + c
    curvature = coeffs[0]  # a > 0 → convex (upward), a < 0 → concave (downward)
    slope = coeffs[1] + 2 * coeffs[0] * np.mean(t)  # average slope
    return ('convex' if curvature > 0.05 else 'concave' if curvature < -0.05 else 'linear',
            curvature, slope)


def compute_slope(traj):
    """Linear slope of trajectory."""
    t = np.arange(len(traj))
    return np.polyfit(t, traj, 1)[0]


def vcms_predict_with_survivors(target_rounds, library):
    """Like vcms_predict but also returns which library candidates survived to the end."""
    candidates = {}
    for sid, rec in library.items():
        candidates[sid] = VCMSParams(**rec['v3_params'])

    survivors = list(candidates.keys())
    pred_c_list = []
    obs_c_list = []
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}

    n = min(10, len(target_rounds))
    for t in range(n):
        cand_preds_c = {}
        for sid in survivors:
            result = run_vcms_agent(candidates[sid], target_rounds[:t + 1])
            cand_preds_c[sid] = result['pred_contrib'][t]
            cand_pred_history[sid] = list(result['pred_contrib'][:t + 1])

        if not distances:
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001) for sid in survivors}
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

    return pred_c_list, survivors, distances


# ================================================================
# Q1: B-DYNAMICS CONSTRAINT ANALYSIS
# ================================================================

def analyze_b_dynamics(n_rounds, library, city_map):
    """Split VCMS advantage by library candidate B-parameter regime."""
    print("\n" + "=" * 72)
    print("  Q1: B-DYNAMICS CONSTRAINT ANALYSIS")
    print("=" * 72)

    # Characterize each library subject's B-regime
    # Key: did punishment actually drain B during fitting?
    lib_pun_recv = {}
    lib_b_params = {}
    for sid, rec in library.items():
        pr_traj = rec['punishment_received_trajectory']
        total_pun = sum(pr_traj)
        mean_pun = np.mean(pr_traj)
        lib_pun_recv[sid] = {'total': total_pun, 'mean': mean_pun}
        lib_b_params[sid] = {
            'b_initial': rec['v3_params']['b_initial'],
            'b_depletion_rate': rec['v3_params']['b_depletion_rate'],
            'b_replenish_rate': rec['v3_params']['b_replenish_rate'],
            'acute_threshold': rec['v3_params']['acute_threshold'],
        }

    # Split library at median punishment received
    all_total_pun = [lib_pun_recv[s]['total'] for s in library]
    median_pun = np.median(all_total_pun)
    print(f"\n  Library punishment received: median={median_pun:.0f}, "
          f"mean={np.mean(all_total_pun):.1f}, max={max(all_total_pun):.0f}")

    high_pun_lib = {s for s in library if lib_pun_recv[s]['total'] > median_pun}
    low_pun_lib = {s for s in library if lib_pun_recv[s]['total'] <= median_pun}
    print(f"  High-punishment library: {len(high_pun_lib)} subjects")
    print(f"  Low-punishment library:  {len(low_pun_lib)} subjects")

    # B-param comparison
    for label, group in [('High-pun', high_pun_lib), ('Low-pun', low_pun_lib)]:
        bi = [lib_b_params[s]['b_initial'] for s in group]
        bd = [lib_b_params[s]['b_depletion_rate'] for s in group]
        br = [lib_b_params[s]['b_replenish_rate'] for s in group]
        at = [lib_b_params[s]['acute_threshold'] for s in group]
        print(f"\n  {label} B-params:")
        print(f"    b_initial:        mean={np.mean(bi):.3f}, std={np.std(bi):.3f}")
        print(f"    b_depletion_rate: mean={np.mean(bd):.3f}, std={np.std(bd):.3f}")
        print(f"    b_replenish_rate: mean={np.mean(br):.3f}, std={np.std(br):.3f}")
        print(f"    acute_threshold:  mean={np.mean(at):.3f}, std={np.std(at):.3f}")

    # Now: for each N-experiment subject, track which library candidates
    # survived to the end — were they from the high-pun or low-pun group?
    sids = sorted(n_rounds.keys())

    vcms_scores = []
    cf_scores = []
    survivor_composition = []  # fraction of survivors from low-pun group

    print(f"\n  Running survivor analysis on {len(sids)} N-experiment subjects...")

    for idx, sid in enumerate(sids):
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]

        preds, survivors, _ = vcms_predict_with_survivors(rounds, library)
        cf_preds = carry_forward_predict(rounds)

        vcms_err = rmse_window(preds, actual, 1)
        cf_err = rmse_window(cf_preds, actual, 1)

        n_low = sum(1 for s in survivors if s in low_pun_lib)
        frac_low = n_low / len(survivors) if survivors else 0.5

        vcms_scores.append(vcms_err)
        cf_scores.append(cf_err)
        survivor_composition.append(frac_low)

    # Split N-experiment subjects by survivor composition
    frac_arr = np.array(survivor_composition)
    vcms_arr = np.array(vcms_scores)
    cf_arr = np.array(cf_scores)
    delta_arr = vcms_arr - cf_arr  # negative = VCMS wins

    # Quartile analysis
    q25, q50, q75 = np.percentile(frac_arr, [25, 50, 75])
    print(f"\n  Survivor low-pun fraction: median={q50:.2f}, q25={q25:.2f}, q75={q75:.2f}")

    buckets = [
        ('Mostly high-pun survivors', frac_arr < q25),
        ('Mixed survivors',          (frac_arr >= q25) & (frac_arr <= q75)),
        ('Mostly low-pun survivors', frac_arr > q75),
    ]

    print(f"\n  {'Group':<30s} {'n':>4s} {'VCMS':>7s} {'CF':>7s} {'Δ':>8s} {'VCMS wins':>10s}")
    print("  " + "-" * 70)

    for label, mask in buckets:
        n = mask.sum()
        if n == 0:
            continue
        mv = vcms_arr[mask].mean()
        mc = cf_arr[mask].mean()
        md = delta_arr[mask].mean()
        wins = (delta_arr[mask] < 0).sum()
        print(f"  {label:<30s} {n:>4d} {mv:>7.4f} {mc:>7.4f} {md:>+8.4f} {wins:>4d}/{n} "
              f"({100*wins/n:.0f}%)")

    # Also: direct correlation
    corr = np.corrcoef(frac_arr, delta_arr)[0, 1]
    print(f"\n  Correlation(low_pun_fraction, VCMS-CF delta): r={corr:+.3f}")
    print(f"  (negative r = low-pun survivors → bigger VCMS advantage)")

    return frac_arr, vcms_arr, cf_arr, delta_arr


# ================================================================
# Q2: FAILURE MODE ANALYSIS — DECLINE SHAPE
# ================================================================

def analyze_decline_shapes(n_rounds, library, city_map):
    """For declining N-experiment subjects, compare predicted vs actual shape."""
    print("\n" + "=" * 72)
    print("  Q2: DECLINE SHAPE ANALYSIS")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    declining_sids = []
    for sid in sids:
        actual = [r.contribution for r in n_rounds[sid]]
        slope = compute_slope(actual)
        if slope < -0.3:
            declining_sids.append(sid)

    print(f"\n  {len(declining_sids)} declining subjects (slope < -0.3)")

    # For each declining subject, compute actual and predicted shape
    shape_match = {'match': 0, 'mismatch': 0}
    rate_errors = []  # (actual_slope, predicted_slope) for shape-matched subjects
    shape_details = defaultdict(lambda: defaultdict(int))  # actual_shape → pred_shape → count

    for sid in declining_sids:
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]
        preds = vcms_predict(rounds, library)

        actual_shape, actual_curv, actual_slope = classify_curvature(actual)
        pred_shape, pred_curv, pred_slope = classify_curvature(preds)

        shape_details[actual_shape][pred_shape] += 1

        if actual_shape == pred_shape:
            shape_match['match'] += 1
        else:
            shape_match['mismatch'] += 1

        rate_errors.append({
            'sid': sid,
            'actual_slope': compute_slope(actual),
            'pred_slope': compute_slope(preds),
            'actual_shape': actual_shape,
            'pred_shape': pred_shape,
            'actual_curv': actual_curv,
            'pred_curv': pred_curv,
        })

    total = shape_match['match'] + shape_match['mismatch']
    print(f"\n  Shape classification (concave/linear/convex):")
    print(f"    Match:    {shape_match['match']:>3d}/{total} ({100*shape_match['match']/total:.0f}%)")
    print(f"    Mismatch: {shape_match['mismatch']:>3d}/{total} ({100*shape_match['mismatch']/total:.0f}%)")

    # Confusion matrix
    all_shapes = ['concave', 'linear', 'convex']
    print(f"\n  Shape confusion matrix (rows=actual, cols=predicted):")
    header = f"    {'':>10s}"
    for s in all_shapes:
        header += f" {s:>10s}"
    print(header)

    for actual_s in all_shapes:
        row = f"    {actual_s:>10s}"
        for pred_s in all_shapes:
            row += f" {shape_details[actual_s][pred_s]:>10d}"
        print(row)

    # Rate analysis: for shape-matched subjects, how close is the slope?
    matched = [r for r in rate_errors if r['actual_shape'] == r['pred_shape']]
    mismatched = [r for r in rate_errors if r['actual_shape'] != r['pred_shape']]

    if matched:
        actual_slopes = [r['actual_slope'] for r in matched]
        pred_slopes = [r['pred_slope'] for r in matched]
        slope_errs = [r['pred_slope'] - r['actual_slope'] for r in matched]
        print(f"\n  Shape-matched subjects (n={len(matched)}):")
        print(f"    Actual slope:  mean={np.mean(actual_slopes):+.3f}, std={np.std(actual_slopes):.3f}")
        print(f"    Pred slope:    mean={np.mean(pred_slopes):+.3f}, std={np.std(pred_slopes):.3f}")
        print(f"    Slope error:   mean={np.mean(slope_errs):+.3f} (pred-actual), std={np.std(slope_errs):.3f}")

        # Does model under-predict or over-predict decline rate?
        underpred = sum(1 for e in slope_errs if e > 0)  # pred slope > actual slope → underestimates decline
        print(f"    Under-predicts decline: {underpred}/{len(matched)} "
              f"({100*underpred/len(matched):.0f}%)")

    # Slope correlation across ALL declining subjects
    all_actual = [r['actual_slope'] for r in rate_errors]
    all_pred = [r['pred_slope'] for r in rate_errors]
    corr = np.corrcoef(all_actual, all_pred)[0, 1]
    print(f"\n  Slope correlation (all declining, n={len(declining_sids)}): r={corr:.3f}")

    # Rate ratio: how severe is the rate mismatch?
    rate_ratios = []
    for r in rate_errors:
        if abs(r['actual_slope']) > 0.1:
            rate_ratios.append(r['pred_slope'] / r['actual_slope'])
    if rate_ratios:
        print(f"  Rate ratio (pred_slope / actual_slope): "
              f"median={np.median(rate_ratios):.2f}, mean={np.mean(rate_ratios):.2f}")
        print(f"    Ratio < 0.5 (model declines too slowly): "
              f"{sum(1 for r in rate_ratios if r < 0.5)}/{len(rate_ratios)}")
        print(f"    Ratio 0.5-1.5 (reasonable match): "
              f"{sum(1 for r in rate_ratios if 0.5 <= r <= 1.5)}/{len(rate_ratios)}")
        print(f"    Ratio > 1.5 (model declines too fast): "
              f"{sum(1 for r in rate_ratios if r > 1.5)}/{len(rate_ratios)}")

    return rate_errors


# ================================================================
# Q3: NON-DECLINING SUBJECTS — MECHANISM
# ================================================================

def analyze_nondeclining(n_rounds, library, city_map):
    """For stable/rising N-experiment subjects, identify which mechanism VCMS uses."""
    print("\n" + "=" * 72)
    print("  Q3: NON-DECLINING SUBJECTS — STABILITY MECHANISM")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    nondecl = []
    for sid in sids:
        actual = [r.contribution for r in n_rounds[sid]]
        slope = compute_slope(actual)
        if slope >= -0.3:
            nondecl.append(sid)

    print(f"\n  {len(nondecl)} non-declining subjects (slope ≥ -0.3)")

    # Classify: stable-high, stable-low, rising
    stable_high = []  # mean > 10, |slope| ≤ 0.3
    stable_low = []   # mean ≤ 10, |slope| ≤ 0.3
    rising = []       # slope > 0.3

    for sid in nondecl:
        actual = [r.contribution for r in n_rounds[sid]]
        slope = compute_slope(actual)
        mean_c = np.mean(actual)
        if slope > 0.3:
            rising.append(sid)
        elif mean_c > 10:
            stable_high.append(sid)
        else:
            stable_low.append(sid)

    print(f"    Stable-high (mean>10, |slope|≤0.3): {len(stable_high)}")
    print(f"    Stable-low  (mean≤10, |slope|≤0.3): {len(stable_low)}")
    print(f"    Rising (slope>0.3):                  {len(rising)}")

    # For each group: does VCMS correctly predict stability?
    # And what do the survivor library params look like?
    for label, group in [('Stable-high', stable_high),
                         ('Stable-low', stable_low),
                         ('Rising', rising)]:
        if not group:
            continue

        print(f"\n  --- {label} (n={len(group)}) ---")

        correctly_stable = 0
        vcms_slopes = []
        cf_slopes = []
        survivor_params = defaultdict(list)

        for sid in group:
            rounds = n_rounds[sid]
            actual = [r.contribution for r in rounds]
            actual_slope = compute_slope(actual)

            preds, survivors, _ = vcms_predict_with_survivors(rounds, library)
            cf_preds = carry_forward_predict(rounds)

            pred_slope = compute_slope(preds)
            cf_slope = compute_slope(cf_preds)
            vcms_slopes.append(pred_slope)
            cf_slopes.append(cf_slope)

            # Is VCMS prediction also non-declining?
            if pred_slope >= -0.3:
                correctly_stable += 1

            # Aggregate survivor params
            for s in survivors:
                rec = library[s]
                for param in ['c_base', 's_rate', 'b_initial', 'b_depletion_rate',
                              'inertia', 's_dir', 'alpha']:
                    survivor_params[param].append(rec['v3_params'][param])

        print(f"    VCMS correctly predicts non-decline: "
              f"{correctly_stable}/{len(group)} ({100*correctly_stable/len(group):.0f}%)")
        print(f"    VCMS pred slope: mean={np.mean(vcms_slopes):+.3f}, "
              f"std={np.std(vcms_slopes):.3f}")
        print(f"    CF pred slope:   mean={np.mean(cf_slopes):+.3f}, "
              f"std={np.std(cf_slopes):.3f}")

        # Key question: high c_base or low s_rate?
        if survivor_params:
            print(f"\n    Survivor library params (across all survivors, n_obs={len(survivor_params['c_base'])}):")
            for param in ['c_base', 's_rate', 'b_initial', 'b_depletion_rate', 'inertia']:
                vals = survivor_params[param]
                # Compare to full library
                lib_vals = [library[s]['v3_params'][param] for s in library]
                print(f"      {param:<20s}: surv mean={np.mean(vals):.3f} "
                      f"(lib mean={np.mean(lib_vals):.3f}, "
                      f"Δ={np.mean(vals) - np.mean(lib_vals):+.3f})")

        # VCMS vs CF head-to-head
        vcms_wins = 0
        for sid in group:
            rounds = n_rounds[sid]
            actual = [r.contribution for r in rounds]
            vcms_preds = vcms_predict(rounds, library)
            cf_preds = carry_forward_predict(rounds)
            if rmse_window(vcms_preds, actual, 1) < rmse_window(cf_preds, actual, 1):
                vcms_wins += 1
        print(f"\n    VCMS beats CF: {vcms_wins}/{len(group)} ({100*vcms_wins/len(group):.0f}%)")

    # Across ALL non-declining: mechanism decomposition
    # Run with knockout channels to see which matters most
    print(f"\n  --- KNOCKOUT ANALYSIS ON NON-DECLINING SUBJECTS ---")

    knockouts_to_test = ['no_strain', 'no_budget', 'no_v_tracking', 'no_facilitation']

    # For a sample (first 30 to keep runtime manageable)
    sample = nondecl[:30]
    print(f"  (sampling {len(sample)} of {len(nondecl)} subjects for knockout analysis)")

    baseline_scores = []
    knockout_scores = {k: [] for k in knockouts_to_test}

    for sid in sample:
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]

        # Find best single library candidate (top survivor)
        _, survivors, distances = vcms_predict_with_survivors(rounds, library)
        if not survivors:
            continue
        best_sid = min(distances, key=distances.get) if distances else survivors[0]
        best_params = VCMSParams(**library[best_sid]['v3_params'])

        # Baseline
        result = run_vcms_agent(best_params, rounds)
        base_rmse = rmse_window(result['pred_contrib'], actual, 1)
        baseline_scores.append(base_rmse)

        # Knockouts
        for ko in knockouts_to_test:
            result_ko = run_vcms_agent(best_params, rounds, knockout=ko)
            ko_rmse = rmse_window(result_ko['pred_contrib'], actual, 1)
            knockout_scores[ko].append(ko_rmse)

    print(f"\n  {'Condition':<20s} {'RMSE':>7s} {'Δ vs base':>10s}")
    print("  " + "-" * 40)
    print(f"  {'baseline':<20s} {np.mean(baseline_scores):>7.4f} {'—':>10s}")
    for ko in knockouts_to_test:
        delta = np.mean(knockout_scores[ko]) - np.mean(baseline_scores)
        print(f"  {ko:<20s} {np.mean(knockout_scores[ko]):>7.4f} {delta:>+10.4f}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  N-EXPERIMENT DEEP DIAGNOSTICS")
    print("=" * 72)

    library = load_library()
    n_rounds, city_map = load_n_experiment_data()
    print(f"  Library: {len(library)} P-experiment subjects")
    print(f"  Test:    {len(n_rounds)} N-experiment subjects")

    analyze_b_dynamics(n_rounds, library, city_map)
    analyze_decline_shapes(n_rounds, library, city_map)
    analyze_nondeclining(n_rounds, library, city_map)

    print("\n" + "=" * 72)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
