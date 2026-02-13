#!/usr/bin/env python3
"""
IPD Data Characterization
=========================

Characterize SP (Stranger Pairing) and FP (Fixed Pairing)
IPD experimental datasets before running VCMS transfer test.

Produces:
- Basic stats (subjects, rounds, cooperation rates)
- Cooperation trajectory by round (1-100)
- Per-subject cooperation rate distribution
- Transition matrix: P(C_t | own_{t-1}, partner_{t-1})
- Baseline accuracy (TFT, WSLS, CF, always-C, always-D)
- SP vs FP comparison
"""

import numpy as np
from collections import defaultdict

from ipd_loader import (
    load_ipd_experiment,
    tit_for_tat, win_stay_lose_shift,
    always_cooperate, always_defect,
    carry_forward_ipd, ipd_accuracy,
)


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def cooperation_trajectory(data, window=10):
    """Mean cooperation rate by round across all subjects."""
    by_round = defaultdict(list)
    for sid, rounds in data.items():
        for r in rounds:
            by_round[r.period].append(r.contribution)

    periods = sorted(by_round.keys())
    rates = [np.mean(by_round[p]) for p in periods]
    return periods, rates


def transition_matrix(data):
    """
    Compute P(C_t | own_{t-1}, partner_{t-1}).

    Returns 2x2 dict indexed by (own_prev, partner_prev) = P(cooperate),
    plus raw counts.
    """
    counts = defaultdict(lambda: [0, 0])  # [n_defect, n_cooperate]

    for sid, rounds in data.items():
        for i in range(1, len(rounds)):
            own_prev = rounds[i - 1].contribution
            partner_prev = 1 if rounds[i - 1].partner_action == 'C' else 0
            current = rounds[i].contribution
            counts[(own_prev, partner_prev)][current] += 1

    matrix = {}
    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        total = sum(counts[key])
        matrix[key] = counts[key][1] / total if total > 0 else 0.0

    return matrix, counts


def window_accuracy(fn, data, w_start, w_end):
    """Baseline accuracy over a specific round window."""
    accs = []
    for sid, rounds in data.items():
        preds = fn(rounds)
        actual = [r.contribution for r in rounds]
        correct = 0
        total = 0
        for i in range(w_start, min(w_end, len(preds), len(actual))):
            if preds[i] == actual[i]:
                correct += 1
            total += 1
        if total > 0:
            accs.append(correct / total)
    return np.mean(accs) if accs else 0.0


# ================================================================
# CHARACTERIZE ONE DATASET
# ================================================================

def characterize_dataset(data, label):
    """Full characterization of one IPD dataset."""
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    sids = sorted(data.keys())
    n_subjects = len(sids)

    # Basic stats
    rounds_per_subject = [len(data[sid]) for sid in sids]
    coop_rates = [sum(r.contribution for r in data[sid]) / len(data[sid])
                  for sid in sids]

    print(f"\n  Subjects: {n_subjects}")
    print(f"  Rounds per subject: {min(rounds_per_subject)}-{max(rounds_per_subject)} "
          f"(mean={np.mean(rounds_per_subject):.0f})")
    print(f"  Overall cooperation rate: {np.mean(coop_rates):.1%}")
    print(f"  Cooperation rate: mean={np.mean(coop_rates):.3f}, "
          f"median={np.median(coop_rates):.3f}, std={np.std(coop_rates):.3f}")

    # Partner structure
    partners_per = [len(set(r.partner_id for r in data[sid])) for sid in sids]
    print(f"  Partners per subject: {min(partners_per)}-{max(partners_per)} "
          f"(mean={np.mean(partners_per):.1f})")

    # Subject type distribution
    n_mostly_d = sum(1 for c in coop_rates if c < 0.2)
    n_mixed = sum(1 for c in coop_rates if 0.2 <= c <= 0.8)
    n_mostly_c = sum(1 for c in coop_rates if c > 0.8)
    print(f"\n  Subject types:")
    print(f"    Mostly-D (<20% coop): {n_mostly_d} ({100 * n_mostly_d / n_subjects:.0f}%)")
    print(f"    Mixed (20-80% coop):  {n_mixed} ({100 * n_mixed / n_subjects:.0f}%)")
    print(f"    Mostly-C (>80% coop): {n_mostly_c} ({100 * n_mostly_c / n_subjects:.0f}%)")

    # Cooperation trajectory by 10-round windows
    periods, rates = cooperation_trajectory(data)
    print(f"\n  Cooperation trajectory (10-round windows):")
    for w_start in range(0, len(rates), 10):
        w_end = min(w_start + 10, len(rates))
        w_rate = np.mean(rates[w_start:w_end])
        print(f"    Rounds {w_start + 1:>3d}-{w_end:>3d}: {w_rate:.1%}")

    first_10 = np.mean(rates[:10])
    last_10 = np.mean(rates[-10:])
    print(f"\n  First 10 rounds: {first_10:.1%}")
    print(f"  Last 10 rounds:  {last_10:.1%}")
    trend = ('declining' if last_10 < first_10 - 0.05
             else 'rising' if last_10 > first_10 + 0.05
             else 'stable')
    print(f"  Trajectory:      {trend}")

    # Transition matrix
    matrix, counts = transition_matrix(data)
    print(f"\n  Transition matrix P(C_t | own_{{t-1}}, partner_{{t-1}}):")
    print(f"    {'':>20s} {'partner D':>14s} {'partner C':>14s}")
    for own_prev in [0, 1]:
        own_label = 'own C' if own_prev == 1 else 'own D'
        n0 = sum(counts[(own_prev, 0)])
        n1 = sum(counts[(own_prev, 1)])
        p0 = matrix[(own_prev, 0)]
        p1 = matrix[(own_prev, 1)]
        print(f"    {own_label:>20s} {p0:.3f} (n={n0:>5d})  {p1:.3f} (n={n1:>5d})")

    # Reactivity signals
    tft_signal = matrix[(0, 1)] - matrix[(0, 0)]
    wsls_signal = matrix[(1, 1)] - matrix[(1, 0)]
    print(f"\n  Reactivity analysis:")
    print(f"    TFT signal  P(C|D,pC) - P(C|D,pD) = {tft_signal:+.3f}")
    print(f"    WSLS signal P(C|C,pC) - P(C|C,pD) = {wsls_signal:+.3f}")

    # Baseline accuracy (from round 2 onwards)
    baselines = {
        'TFT': tit_for_tat,
        'WSLS': win_stay_lose_shift,
        'Always-C': always_cooperate,
        'Always-D': always_defect,
        'Carry-Fwd': carry_forward_ipd,
    }

    print(f"\n  Baseline accuracy (from round 2 onwards):")
    acc_results = {}
    for name, fn in baselines.items():
        accs = []
        for sid, rounds in data.items():
            actual = [r.contribution for r in rounds]
            preds = fn(rounds)
            acc = ipd_accuracy(preds, actual, from_round=1)
            accs.append(acc)
        acc_results[name] = accs

    print(f"    {'Baseline':<12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}")
    print(f"    {'-' * 36}")
    for name in ['TFT', 'WSLS', 'Carry-Fwd', 'Always-C', 'Always-D']:
        vals = acc_results[name]
        print(f"    {name:<12s} {np.mean(vals):>7.1%} {np.median(vals):>8.1%} "
              f"{np.std(vals):>8.3f}")

    # Accuracy by round window
    windows = [(1, 10), (10, 30), (30, 50), (50, 80), (80, 100)]
    print(f"\n  Baseline accuracy by window:")
    header = f"    {'':>12s}"
    for w_s, w_e in windows:
        header += f" {w_s + 1:>2d}-{w_e:<3d}"
    print(header)

    for name in ['TFT', 'WSLS', 'Carry-Fwd', 'Always-D']:
        fn = baselines[name]
        row = f"    {name:<12s}"
        for w_s, w_e in windows:
            acc = window_accuracy(fn, data, w_s, w_e)
            row += f" {acc:>5.1%} "
        print(row)

    return coop_rates, matrix, acc_results


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  IPD DATA CHARACTERIZATION")
    print("=" * 72)

    # Load both datasets
    print("\nLoading SP (Stranger Pairing) data...")
    sp_data = load_ipd_experiment('IPD-rand.csv')
    print(f"  Loaded {len(sp_data)} subjects")

    print("\nLoading FP (Fixed Pairing) data...")
    fp_data = load_ipd_experiment('fix.csv')
    print(f"  Loaded {len(fp_data)} subjects")

    # Characterize each
    sp_coop, sp_matrix, sp_acc = characterize_dataset(
        sp_data, "SP — STRANGER PAIRING (IPD-rand.csv)")
    fp_coop, fp_matrix, fp_acc = characterize_dataset(
        fp_data, "FP — FIXED PAIRING (fix.csv)")

    # SP vs FP comparison
    print(f"\n{'=' * 72}")
    print(f"  SP vs FP COMPARISON")
    print(f"{'=' * 72}")

    sp_mean = np.mean(sp_coop)
    fp_mean = np.mean(fp_coop)
    print(f"\n  {'Metric':<35s} {'SP':>10s} {'FP':>10s} {'Diff':>10s}")
    print(f"  {'-' * 65}")
    print(f"  {'Mean cooperation rate':<35s} "
          f"{sp_mean:>10.1%} {fp_mean:>10.1%} {fp_mean - sp_mean:>+10.1%}")
    print(f"  {'Median cooperation rate':<35s} "
          f"{np.median(sp_coop):>10.1%} {np.median(fp_coop):>10.1%}")
    print(f"  {'Std cooperation rate':<35s} "
          f"{np.std(sp_coop):>10.3f} {np.std(fp_coop):>10.3f}")

    # Transition matrix comparison
    print(f"\n  Transition matrix comparison:")
    for key_label, key in [('P(C|DD)', (0, 0)), ('P(C|DC)', (0, 1)),
                            ('P(C|CD)', (1, 0)), ('P(C|CC)', (1, 1))]:
        sp_v = sp_matrix[key]
        fp_v = fp_matrix[key]
        print(f"    {key_label:<12s}  SP={sp_v:.3f}  FP={fp_v:.3f}  "
              f"Diff={fp_v - sp_v:+.3f}")

    # Baseline accuracy comparison
    print(f"\n  Baseline accuracy comparison:")
    print(f"    {'Baseline':<12s} {'SP':>8s} {'FP':>8s} {'Diff':>8s}")
    print(f"    {'-' * 32}")
    for name in ['TFT', 'WSLS', 'Carry-Fwd', 'Always-C', 'Always-D']:
        sp_m = np.mean(sp_acc[name])
        fp_m = np.mean(fp_acc[name])
        print(f"    {name:<12s} {sp_m:>7.1%} {fp_m:>8.1%} {fp_m - sp_m:>+8.1%}")

    print(f"\n{'=' * 72}")
    print(f"  CHARACTERIZATION COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
