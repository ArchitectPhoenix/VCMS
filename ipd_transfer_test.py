#!/usr/bin/env python3
"""
IPD Cross-Game Transfer Test
=============================

Tests whether a PGG-fitted VCMS library (196 P-experiment subjects)
can predict behavior in the Iterated Prisoner's Dilemma.

Key approach: precompute all 196 candidate trajectories per target
subject (teacher forcing makes them independent), then do elimination
as lightweight post-processing on stored predictions.

Metrics:
- Accuracy from round k (k=1, 5, 10, 20, 50)
- Cohen's kappa (base-rate adjusted)
- Transition accuracy (correctly predicted action changes)
- Cooperation trajectory RMSE (10-round windowed)

Analysis:
- Per-subject-type breakdown (mostly-D / mixed / mostly-C)
- Survivor analysis: which PGG candidates match which IPD types?
- Dynamics traces for selected subjects
- SP vs FP comparison
"""

import json
import math
import time
import numpy as np
from collections import defaultdict

from ipd_loader import (
    load_ipd_experiment,
    tit_for_tat, win_stay_lose_shift,
    always_cooperate, always_defect,
    carry_forward_ipd, ipd_accuracy,
)
from vcms_engine_v4 import (
    IPD_CONFIG, run_vcms_v4, v3_params_to_v4,
)


# ================================================================
# PRECOMPUTE ENSEMBLE
# ================================================================

def precompute_candidate_predictions(target_rounds, library, game_config):
    """
    Run all library candidates on target subject's actual rounds.

    Under teacher forcing, each candidate's prediction trajectory depends
    only on actual data + parameters. So we run each candidate once on
    the full sequence, then use stored predictions for ensemble selection.

    Returns {lib_sid: pred_contrib_list}.
    """
    predictions = {}
    for lib_sid, rec in library.items():
        params = v3_params_to_v4(rec['v3_params'])
        result = run_vcms_v4(params, target_rounds, game_config)
        predictions[lib_sid] = result['pred_contrib']
    return predictions


def ensemble_from_precomputed(predictions, actual, max_contrib=1):
    """
    Run ensemble selection on precomputed predictions.

    Same elimination logic as PGG ensemble: distance-weighted voting
    with progressive elimination using max(best * 3.0, 0.5) threshold.

    Returns (ensemble_preds, survivor_counts, final_survivors).
    """
    n = len(actual)
    survivors = list(predictions.keys())
    pred_list = []
    distances = {}
    survivor_counts = []

    for t in range(n):
        # Weighted prediction from survivors
        cand_preds = {sid: predictions[sid][t] for sid in survivors}

        if not distances:
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances[sid] + 0.001) for sid in survivors}
        total_w = sum(weights.values())

        pc = sum((weights[sid] / total_w) * cand_preds[sid] for sid in survivors)
        pred_list.append(int(round(pc)))

        # Update distances using full prediction history
        new_distances = {}
        for sid in survivors:
            preds = predictions[sid]
            c_d = sum((actual[i] - preds[i]) ** 2
                      for i in range(t + 1)) / (max_contrib ** 2 * (t + 1))
            new_distances[sid] = math.sqrt(c_d)

        # Elimination
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
        survivor_counts.append(len(survivors))

    return pred_list, survivor_counts, list(survivors)


# ================================================================
# METRICS
# ================================================================

def accuracy_from_round(pred, actual, k):
    """Fraction correct from round k onwards."""
    correct = 0
    total = 0
    for i in range(k, min(len(pred), len(actual))):
        if pred[i] == actual[i]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def cohens_kappa(pred, actual, from_round=0):
    """Cohen's kappa for binary predictions from from_round onwards."""
    p = pred[from_round:]
    a = actual[from_round:]
    n = len(a)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = sum(1 for i in range(n) if p[i] == a[i]) / n

    # Expected agreement by chance
    p_actual_1 = sum(a) / n
    p_pred_1 = sum(p) / n
    p_e = p_actual_1 * p_pred_1 + (1 - p_actual_1) * (1 - p_pred_1)

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def transition_accuracy(pred, actual, from_round=1):
    """
    Fraction of correctly predicted action changes.

    An action change occurs at round t when actual[t] != actual[t-1].
    We check whether pred[t] correctly predicted the new action.
    """
    correct = 0
    total = 0
    for i in range(max(1, from_round), min(len(pred), len(actual))):
        if actual[i] != actual[i - 1]:  # action change
            total += 1
            if pred[i] == actual[i]:
                correct += 1
    return correct / total if total > 0 else 0.0


def trajectory_rmse(pred, actual, window=10):
    """
    RMSE of cooperation rates in sliding windows.

    Compares the smoothed cooperation trajectory rather than
    individual predictions.
    """
    if len(pred) < window or len(actual) < window:
        return float('inf')

    n_windows = len(actual) - window + 1
    errors = []
    for start in range(n_windows):
        pred_rate = np.mean(pred[start:start + window])
        actual_rate = np.mean(actual[start:start + window])
        errors.append((pred_rate - actual_rate) ** 2)

    return math.sqrt(np.mean(errors))


# ================================================================
# BASELINES
# ================================================================

def run_baselines(data):
    """Run all baselines on all subjects. Returns {name: {sid: pred_list}}."""
    baseline_fns = {
        'TFT': tit_for_tat,
        'WSLS': win_stay_lose_shift,
        'Always-C': always_cooperate,
        'Always-D': always_defect,
        'Carry-Fwd': carry_forward_ipd,
    }

    results = {}
    for name, fn in baseline_fns.items():
        results[name] = {}
        for sid, rounds in data.items():
            results[name][sid] = fn(rounds)

    return results


# ================================================================
# DATA LOADING
# ================================================================

def load_library():
    """Load the PGG P-experiment fitted library."""
    with open('v3_library_fitted.json') as f:
        return json.load(f)


def classify_subject(coop_rate):
    """Classify subject by cooperation rate."""
    if coop_rate < 0.2:
        return 'mostly-D'
    elif coop_rate > 0.8:
        return 'mostly-C'
    else:
        return 'mixed'


# ================================================================
# MAIN TRANSFER TEST
# ================================================================

def run_transfer_test(data, library, label):
    """
    Run the full transfer test on one IPD dataset.

    Returns per-subject results dict.
    """
    print(f"\n{'=' * 72}")
    print(f"  TRANSFER TEST: {label}")
    print(f"{'=' * 72}")

    sids = sorted(data.keys())
    n = len(sids)

    # Classify subjects
    coop_rates = {}
    subject_types = {}
    for sid in sids:
        rounds = data[sid]
        cr = sum(r.contribution for r in rounds) / len(rounds)
        coop_rates[sid] = cr
        subject_types[sid] = classify_subject(cr)

    type_counts = defaultdict(int)
    for t in subject_types.values():
        type_counts[t] += 1
    print(f"\n  {n} subjects: {dict(type_counts)}")

    # Run baselines
    print(f"  Running baselines...")
    baseline_preds = run_baselines(data)

    # Run VCMS ensemble (precompute approach)
    print(f"  Running VCMS ensemble (precompute)...")
    vcms_preds = {}
    vcms_survivors = {}
    vcms_survivor_counts = {}

    t0 = time.time()
    for idx, sid in enumerate(sids):
        rounds = data[sid]
        actual = [r.contribution for r in rounds]

        # Precompute all 196 candidate predictions on this subject
        predictions = precompute_candidate_predictions(rounds, library, IPD_CONFIG)

        # Run ensemble selection on precomputed predictions
        preds, surv_counts, final_surv = ensemble_from_precomputed(
            predictions, actual, max_contrib=IPD_CONFIG.max_contrib)

        vcms_preds[sid] = preds
        vcms_survivors[sid] = final_surv
        vcms_survivor_counts[sid] = surv_counts

        if (idx + 1) % 10 == 0 or idx == n - 1:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n - idx - 1)
            print(f"    [{idx + 1:>3d}/{n}] {elapsed:>5.0f}s elapsed, "
                  f"~{eta:>4.0f}s remaining")

    total_time = time.time() - t0
    print(f"  Completed in {total_time:.1f}s "
          f"({total_time / n:.2f}s per subject)")

    # ================================================================
    # COMPUTE METRICS
    # ================================================================

    k_values = [1, 5, 10, 20, 50]
    methods = ['VCMS', 'TFT', 'WSLS', 'Carry-Fwd', 'Always-C', 'Always-D']

    # Accuracy from round k
    print(f"\n  --- Accuracy from round k ---")
    acc = {m: {k: [] for k in k_values} for m in methods}
    for sid in sids:
        actual = [r.contribution for r in data[sid]]
        for k in k_values:
            acc['VCMS'][k].append(accuracy_from_round(vcms_preds[sid], actual, k))
            for name in methods[1:]:
                acc[name][k].append(
                    accuracy_from_round(baseline_preds[name][sid], actual, k))

    header = f"    {'Method':<12s}"
    for k in k_values:
        header += f"  k={k:<4d}"
    print(header)
    print(f"    {'-' * (12 + 8 * len(k_values))}")
    for m in methods:
        row = f"    {m:<12s}"
        for k in k_values:
            row += f"  {np.mean(acc[m][k]):>.1%} "
        print(row)

    # Delta vs carry-forward
    print()
    for m in methods:
        if m == 'Carry-Fwd':
            continue
        row = f"    {'Δ ' + m:<12s}"
        for k in k_values:
            delta = np.mean(acc[m][k]) - np.mean(acc['Carry-Fwd'][k])
            row += f"  {delta:>+.1%} "
        print(row)

    # Cohen's kappa
    print(f"\n  --- Cohen's kappa (from round 2) ---")
    kappas = {m: [] for m in methods}
    for sid in sids:
        actual = [r.contribution for r in data[sid]]
        kappas['VCMS'].append(cohens_kappa(vcms_preds[sid], actual, from_round=1))
        for name in methods[1:]:
            kappas[name].append(
                cohens_kappa(baseline_preds[name][sid], actual, from_round=1))

    print(f"    {'Method':<12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}")
    print(f"    {'-' * 36}")
    for m in methods:
        vals = kappas[m]
        print(f"    {m:<12s} {np.mean(vals):>8.3f} {np.median(vals):>8.3f} "
              f"{np.std(vals):>8.3f}")

    # Transition accuracy
    print(f"\n  --- Transition accuracy (action changes) ---")
    trans_acc = {m: [] for m in methods}
    for sid in sids:
        actual = [r.contribution for r in data[sid]]
        trans_acc['VCMS'].append(
            transition_accuracy(vcms_preds[sid], actual, from_round=1))
        for name in methods[1:]:
            trans_acc[name].append(
                transition_accuracy(baseline_preds[name][sid], actual,
                                    from_round=1))

    print(f"    {'Method':<12s} {'Mean':>8s} {'Median':>8s}")
    print(f"    {'-' * 30}")
    for m in methods:
        vals = trans_acc[m]
        print(f"    {m:<12s} {np.mean(vals):>8.1%} {np.median(vals):>8.1%}")

    # Trajectory RMSE
    print(f"\n  --- Trajectory RMSE (10-round windows) ---")
    traj_rmse = {m: [] for m in methods}
    for sid in sids:
        actual = [r.contribution for r in data[sid]]
        traj_rmse['VCMS'].append(
            trajectory_rmse(vcms_preds[sid], actual, window=10))
        for name in methods[1:]:
            traj_rmse[name].append(
                trajectory_rmse(baseline_preds[name][sid], actual, window=10))

    print(f"    {'Method':<12s} {'Mean':>8s} {'Median':>8s}")
    print(f"    {'-' * 30}")
    for m in methods:
        vals = traj_rmse[m]
        print(f"    {m:<12s} {np.mean(vals):>8.4f} {np.median(vals):>8.4f}")

    # ================================================================
    # PER-TYPE BREAKDOWN
    # ================================================================

    print(f"\n  --- Per-type accuracy (from round 2) ---")
    types = ['mostly-D', 'mixed', 'mostly-C']
    type_acc = {m: {t: [] for t in types} for m in methods}

    for sid in sids:
        actual = [r.contribution for r in data[sid]]
        stype = subject_types[sid]
        type_acc['VCMS'][stype].append(
            accuracy_from_round(vcms_preds[sid], actual, 1))
        for name in methods[1:]:
            type_acc[name][stype].append(
                accuracy_from_round(baseline_preds[name][sid], actual, 1))

    header = f"    {'Method':<12s}"
    for t in types:
        n_t = type_counts[t]
        header += f"  {t} (n={n_t})"
    print(header)
    print(f"    {'-' * 60}")
    for m in methods:
        row = f"    {m:<12s}"
        for t in types:
            vals = type_acc[m][t]
            if vals:
                row += f"  {np.mean(vals):>14.1%}"
            else:
                row += f"  {'—':>14s}"
        print(row)

    # Per-type kappa
    print(f"\n  --- Per-type kappa (from round 2) ---")
    type_kappa = {m: {t: [] for t in types} for m in methods}
    for sid in sids:
        actual = [r.contribution for r in data[sid]]
        stype = subject_types[sid]
        type_kappa['VCMS'][stype].append(
            cohens_kappa(vcms_preds[sid], actual, from_round=1))
        for name in methods[1:]:
            type_kappa[name][stype].append(
                cohens_kappa(baseline_preds[name][sid], actual, from_round=1))

    header = f"    {'Method':<12s}"
    for t in types:
        header += f"  {t:>14s}"
    print(header)
    print(f"    {'-' * 60}")
    for m in methods:
        row = f"    {m:<12s}"
        for t in types:
            vals = type_kappa[m][t]
            if vals:
                row += f"  {np.mean(vals):>14.3f}"
            else:
                row += f"  {'—':>14s}"
        print(row)

    # ================================================================
    # SURVIVOR ANALYSIS
    # ================================================================

    print(f"\n  --- Survivor analysis ---")

    # Load library behavioral profiles for survivor analysis
    surv_profiles = defaultdict(list)
    for sid in sids:
        stype = subject_types[sid]
        for lib_sid in vcms_survivors[sid]:
            if lib_sid in library:
                lib_profile = library[lib_sid].get('behavioral_profile', {})
                surv_profiles[stype].append({
                    'lib_sid': lib_sid,
                    'target_sid': sid,
                    'lib_c_base': library[lib_sid]['v3_params'].get('c_base', 0),
                    'lib_s_dir': library[lib_sid]['v3_params'].get('s_dir', 0),
                    'lib_inertia': library[lib_sid]['v3_params'].get('inertia', 0),
                    'lib_coop_rate': np.mean(
                        library[lib_sid].get('contribution_trajectory', [])) / 20.0,
                })

    for stype in types:
        profiles = surv_profiles[stype]
        if not profiles:
            print(f"\n    {stype}: no survivors")
            continue

        c_bases = [p['lib_c_base'] for p in profiles]
        s_dirs = [p['lib_s_dir'] for p in profiles]
        inertias = [p['lib_inertia'] for p in profiles]
        lib_coops = [p['lib_coop_rate'] for p in profiles]

        # Unique library candidates used
        unique_lib = set(p['lib_sid'] for p in profiles)

        print(f"\n    {stype} → {len(unique_lib)} unique library candidates "
              f"({len(profiles)} total matches)")
        print(f"      c_base:  mean={np.mean(c_bases):.3f}, "
              f"std={np.std(c_bases):.3f}")
        print(f"      s_dir:   mean={np.mean(s_dirs):+.2f} "
              f"(prosocial: {sum(1 for s in s_dirs if s > 0)}/{len(s_dirs)})")
        print(f"      inertia: mean={np.mean(inertias):.3f}")
        print(f"      lib cooperation rate: mean={np.mean(lib_coops):.1%}")

    # Survivor count trajectory
    print(f"\n  --- Survivor count trajectory ---")
    round_checkpoints = [1, 5, 10, 20, 50, 100]
    header = f"    {'':>12s}"
    for r in round_checkpoints:
        header += f"  r={r:<4d}"
    print(header)

    for stype in types:
        type_sids = [s for s in sids if subject_types[s] == stype]
        if not type_sids:
            continue
        row = f"    {stype:<12s}"
        for r in round_checkpoints:
            counts = []
            for sid in type_sids:
                sc = vcms_survivor_counts[sid]
                idx = min(r - 1, len(sc) - 1)
                counts.append(sc[idx])
            row += f"  {np.mean(counts):>5.1f} "
        print(row)

    # ================================================================
    # DYNAMICS TRACES (selected subjects)
    # ================================================================

    print(f"\n  --- Dynamics traces (best/worst per type) ---")
    for stype in types:
        type_sids = [s for s in sids if subject_types[s] == stype]
        if not type_sids:
            continue

        # Find best and worst VCMS accuracy within type
        type_accs = {s: accuracy_from_round(vcms_preds[s],
                     [r.contribution for r in data[s]], 1) for s in type_sids}
        best_sid = max(type_accs, key=type_accs.get)
        worst_sid = min(type_accs, key=type_accs.get)

        for trace_sid, trace_label in [(best_sid, 'best'), (worst_sid, 'worst')]:
            rounds = data[trace_sid]
            actual = [r.contribution for r in rounds]

            # Re-run the best surviving candidate to get dynamics trace
            if vcms_survivors[trace_sid]:
                best_lib_sid = vcms_survivors[trace_sid][0]
                params = v3_params_to_v4(library[best_lib_sid]['v3_params'])
                result = run_vcms_v4(params, rounds, IPD_CONFIG)
                trace = result['trace']

                print(f"\n    {stype} {trace_label}: {trace_sid} "
                      f"(acc={type_accs[trace_sid]:.1%}, "
                      f"coop={coop_rates[trace_sid]:.1%}, "
                      f"survivor={best_lib_sid})")

                # Print key dynamics at checkpoints
                checkpoints = [0, 4, 9, 19, 49, 99]
                checkpoints = [c for c in checkpoints if c < len(trace)]
                header = f"      {'Round':>5s} {'Act':>3s} {'Pred':>4s} " \
                         f"{'V_lvl':>6s} {'Strain':>7s} {'B':>6s} " \
                         f"{'Afford':>7s} {'c_norm':>6s}"
                print(header)
                for c in checkpoints:
                    t = trace[c]
                    print(f"      {c + 1:>5d} {actual[c]:>3d} "
                          f"{result['pred_contrib'][c]:>4d} "
                          f"{t['v']['v_level']:>6.3f} "
                          f"{t['s_accum']['strain_pre_discharge']:>7.3f} "
                          f"{t['budget']['b_post']:>6.3f} "
                          f"{t['routing']['affordability']:>7.3f} "
                          f"{t['c_output']['c_norm']:>6.3f}")

    return {
        'acc': acc, 'kappas': kappas, 'trans_acc': trans_acc,
        'traj_rmse': traj_rmse, 'subject_types': subject_types,
        'coop_rates': coop_rates, 'vcms_survivors': vcms_survivors,
        'vcms_survivor_counts': vcms_survivor_counts,
    }


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  IPD CROSS-GAME TRANSFER TEST")
    print("  PGG P-experiment library → IPD predictions")
    print("=" * 72)

    # Load library
    print("\nLoading PGG P-experiment library...")
    library = load_library()
    print(f"  {len(library)} library subjects")

    # Load IPD data
    print("\nLoading IPD data...")
    sp_data = load_ipd_experiment('IPD-rand.csv')
    fp_data = load_ipd_experiment('fix.csv')
    print(f"  SP: {len(sp_data)} subjects (Stranger Pairing)")
    print(f"  FP: {len(fp_data)} subjects (Fixed Pairing)")

    # Run transfer tests
    sp_results = run_transfer_test(sp_data, library, "SP — STRANGER PAIRING")
    fp_results = run_transfer_test(fp_data, library, "FP — FIXED PAIRING")

    # ================================================================
    # SP vs FP COMPARISON
    # ================================================================

    print(f"\n{'=' * 72}")
    print(f"  SP vs FP COMPARISON")
    print(f"{'=' * 72}")

    k_values = [1, 5, 10, 20, 50]
    methods = ['VCMS', 'TFT', 'WSLS', 'Carry-Fwd', 'Always-D']

    print(f"\n  --- Accuracy comparison (from round k) ---")
    for k in [1, 10, 50]:
        print(f"\n    k={k}:")
        print(f"      {'Method':<12s} {'SP':>8s} {'FP':>8s} {'Diff':>8s}")
        print(f"      {'-' * 32}")
        for m in methods:
            sp_m = np.mean(sp_results['acc'][m][k])
            fp_m = np.mean(fp_results['acc'][m][k])
            print(f"      {m:<12s} {sp_m:>7.1%} {fp_m:>8.1%} "
                  f"{fp_m - sp_m:>+8.1%}")

    print(f"\n  --- Kappa comparison ---")
    print(f"    {'Method':<12s} {'SP':>8s} {'FP':>8s} {'Diff':>8s}")
    print(f"    {'-' * 32}")
    for m in methods:
        sp_k = np.mean(sp_results['kappas'][m])
        fp_k = np.mean(fp_results['kappas'][m])
        print(f"    {m:<12s} {sp_k:>8.3f} {fp_k:>8.3f} "
              f"{fp_k - sp_k:>+8.3f}")

    print(f"\n  --- Transition accuracy comparison ---")
    print(f"    {'Method':<12s} {'SP':>8s} {'FP':>8s} {'Diff':>8s}")
    print(f"    {'-' * 32}")
    for m in methods:
        sp_t = np.mean(sp_results['trans_acc'][m])
        fp_t = np.mean(fp_results['trans_acc'][m])
        print(f"    {m:<12s} {sp_t:>7.1%} {fp_t:>8.1%} "
              f"{fp_t - sp_t:>+8.1%}")

    # Theory prediction: FP accuracy > SP accuracy (repeated interaction)
    sp_vcms_acc = np.mean(sp_results['acc']['VCMS'][1])
    fp_vcms_acc = np.mean(fp_results['acc']['VCMS'][1])
    print(f"\n  Theory test: FP VCMS accuracy > SP VCMS accuracy?")
    print(f"    SP: {sp_vcms_acc:.1%}, FP: {fp_vcms_acc:.1%}, "
          f"Diff: {fp_vcms_acc - sp_vcms_acc:+.1%}")
    print(f"    {'CONFIRMED' if fp_vcms_acc > sp_vcms_acc else 'FALSIFIED'}")

    print(f"\n{'=' * 72}")
    print(f"  TRANSFER TEST COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
