#!/usr/bin/env python3
"""
IPD Refinement Test
===================

Three refinements to improve IPD cross-game transfer while preserving
PGG findings:

  1. Strain decay (habituation) — prevents runaway strain over long horizons
  2. Horizon scaling — scales h_start proportionally to game length
  3. Tighter elimination — forces earlier commitment for binary actions

Structure:
  Phase 1: Strain decay sweep on SP (find best value)
  Phase 2: Individual ablation (each refinement independently)
  Phase 3: Combined best on both SP and FP, per-type breakdown

All PGG configs have strain_decay=0 and horizon_scaling=False by default,
so PGG results are guaranteed unchanged.
"""

import json
import math
import time
import numpy as np
from collections import defaultdict
from dataclasses import replace

from ipd_loader import (
    load_ipd_experiment,
    tit_for_tat, win_stay_lose_shift,
    always_cooperate, always_defect,
    carry_forward_ipd, ipd_accuracy,
)
from vcms_engine_v4 import (
    GameConfig, IPD_CONFIG, PGG_P_CONFIG,
    run_vcms_v4, v3_params_to_v4,
)


# ================================================================
# PRECOMPUTE + ENSEMBLE (reused from transfer test, with config-aware elimination)
# ================================================================

def precompute_predictions(target_rounds, library, game_config):
    """Run all library candidates once on target subject's rounds."""
    predictions = {}
    for lib_sid, rec in library.items():
        params = v3_params_to_v4(rec['v3_params'])
        result = run_vcms_v4(params, target_rounds, game_config)
        predictions[lib_sid] = result['pred_contrib']
    return predictions


def ensemble_select(predictions, actual, game_config):
    """
    Ensemble selection using game-config-aware elimination.

    Uses gc.elim_floor and gc.elim_mult instead of hardcoded 0.5/3.0.
    """
    n = len(actual)
    max_c = game_config.max_contrib
    elim_floor = game_config.elim_floor
    elim_mult = game_config.elim_mult

    survivors = list(predictions.keys())
    pred_list = []
    distances = {}
    survivor_counts = []

    for t in range(n):
        cand_preds = {sid: predictions[sid][t] for sid in survivors}

        if not distances:
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances[sid] + 0.001) for sid in survivors}
        total_w = sum(weights.values())

        pc = sum((weights[sid] / total_w) * cand_preds[sid] for sid in survivors)
        pred_list.append(int(round(pc)))

        new_distances = {}
        for sid in survivors:
            preds = predictions[sid]
            c_d = sum((actual[i] - preds[i]) ** 2
                      for i in range(t + 1)) / (max_c ** 2 * (t + 1))
            new_distances[sid] = math.sqrt(c_d)

        if new_distances:
            best = min(new_distances.values())
            thresh = max(best * elim_mult, elim_floor)
        else:
            thresh = elim_floor

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
    correct = sum(1 for i in range(k, min(len(pred), len(actual)))
                  if pred[i] == actual[i])
    total = min(len(pred), len(actual)) - k
    return correct / total if total > 0 else 0.0


def cohens_kappa(pred, actual, from_round=0):
    p = pred[from_round:]
    a = actual[from_round:]
    n = len(a)
    if n == 0:
        return 0.0
    p_o = sum(1 for i in range(n) if p[i] == a[i]) / n
    p_a1 = sum(a) / n
    p_p1 = sum(p) / n
    p_e = p_a1 * p_p1 + (1 - p_a1) * (1 - p_p1)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def trajectory_rmse(pred, actual, window=10):
    if len(pred) < window or len(actual) < window:
        return float('inf')
    n_windows = len(actual) - window + 1
    errors = []
    for start in range(n_windows):
        pred_rate = np.mean(pred[start:start + window])
        actual_rate = np.mean(actual[start:start + window])
        errors.append((pred_rate - actual_rate) ** 2)
    return math.sqrt(np.mean(errors))


def classify_subject(coop_rate):
    if coop_rate < 0.2:
        return 'mostly-D'
    elif coop_rate > 0.8:
        return 'mostly-C'
    return 'mixed'


# ================================================================
# RUN ONE VARIANT
# ================================================================

def run_variant(data, library, game_config, label, verbose=True):
    """
    Run one config variant on a dataset. Returns metrics dict.

    Optimization: precomputed predictions depend on (strain_decay,
    horizon_scaling). Elimination params only affect ensemble selection.
    """
    sids = sorted(data.keys())
    coop_rates = {sid: sum(r.contribution for r in data[sid]) / len(data[sid])
                  for sid in sids}
    types = {sid: classify_subject(coop_rates[sid]) for sid in sids}

    all_acc = []
    all_kappa = []
    all_traj = []
    type_acc = defaultdict(list)
    type_kappa = defaultdict(list)
    surv_counts = []

    t0 = time.time()
    for idx, sid in enumerate(sids):
        rounds = data[sid]
        actual = [r.contribution for r in rounds]

        predictions = precompute_predictions(rounds, library, game_config)
        preds, sc, final_surv = ensemble_select(predictions, actual, game_config)

        acc = accuracy_from_round(preds, actual, 1)
        kap = cohens_kappa(preds, actual, from_round=1)
        trmse = trajectory_rmse(preds, actual, window=10)

        all_acc.append(acc)
        all_kappa.append(kap)
        all_traj.append(trmse)
        type_acc[types[sid]].append(acc)
        type_kappa[types[sid]].append(kap)
        surv_counts.append(sc[-1] if sc else 0)

        if verbose and ((idx + 1) % 20 == 0 or idx == len(sids) - 1):
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(sids) - idx - 1)
            print(f"      [{idx + 1:>3d}/{len(sids)}] {elapsed:>4.0f}s, "
                  f"~{eta:>3.0f}s rem")

    return {
        'label': label,
        'acc': np.mean(all_acc),
        'kappa': np.mean(all_kappa),
        'traj_rmse': np.mean(all_traj),
        'type_acc': {t: np.mean(v) for t, v in type_acc.items()},
        'type_kappa': {t: np.mean(v) for t, v in type_kappa.items()},
        'mean_survivors': np.mean(surv_counts),
        'all_acc': all_acc,
        'all_kappa': all_kappa,
    }


# ================================================================
# BASELINES (for reference)
# ================================================================

def compute_baselines(data):
    """Compute CF and TFT accuracy for comparison."""
    cf_accs = []
    tft_accs = []
    for sid, rounds in data.items():
        actual = [r.contribution for r in rounds]
        cf_accs.append(accuracy_from_round(carry_forward_ipd(rounds), actual, 1))
        tft_accs.append(accuracy_from_round(tit_for_tat(rounds), actual, 1))
    return np.mean(cf_accs), np.mean(tft_accs)


# ================================================================
# PHASE 1: STRAIN DECAY SWEEP
# ================================================================

def phase1_decay_sweep(data, library, label):
    """Sweep strain_decay values to find optimal."""
    print(f"\n{'=' * 72}")
    print(f"  PHASE 1: STRAIN DECAY SWEEP — {label}")
    print(f"{'=' * 72}")

    decay_values = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]
    results = {}

    for decay in decay_values:
        config = replace(IPD_CONFIG, strain_decay=decay)
        print(f"\n    strain_decay = {decay:.2f}")
        r = run_variant(data, library, config,
                        f"decay={decay:.2f}", verbose=True)
        results[decay] = r

    # Results table
    cf_acc, tft_acc = compute_baselines(data)

    print(f"\n  {'Decay':>6s} {'Acc':>8s} {'Kappa':>8s} {'TrajRMSE':>9s} "
          f"{'Surv':>6s} {'Δ CF':>8s}")
    print(f"  {'-' * 52}")
    print(f"  {'CF':<6s} {cf_acc:>8.1%} {'—':>8s} {'—':>9s} {'—':>6s} {'—':>8s}")
    print(f"  {'TFT':<6s} {tft_acc:>8.1%} {'—':>8s} {'—':>9s} {'—':>6s} {'—':>8s}")

    for decay in decay_values:
        r = results[decay]
        delta = r['acc'] - cf_acc
        print(f"  {decay:>6.2f} {r['acc']:>8.1%} {r['kappa']:>8.3f} "
              f"{r['traj_rmse']:>9.4f} {r['mean_survivors']:>6.1f} "
              f"{delta:>+8.1%}")

    # Find best by accuracy
    best_decay = max(results, key=lambda d: results[d]['acc'])
    print(f"\n  Best decay: {best_decay:.2f} "
          f"(acc={results[best_decay]['acc']:.1%}, "
          f"kappa={results[best_decay]['kappa']:.3f})")

    return best_decay, results


# ================================================================
# PHASE 2: INDIVIDUAL ABLATION
# ================================================================

def phase2_ablation(data, library, best_decay, label):
    """Test each refinement independently."""
    print(f"\n{'=' * 72}")
    print(f"  PHASE 2: ABLATION — {label}")
    print(f"{'=' * 72}")

    variants = {
        'baseline': replace(IPD_CONFIG),
        'decay_only': replace(IPD_CONFIG,
                              strain_decay=best_decay),
        'horizon_only': replace(IPD_CONFIG,
                                horizon_scaling=True, reference_rounds=10),
        'elim_only': replace(IPD_CONFIG,
                             elim_floor=0.3, elim_mult=2.0),
        'decay+horizon': replace(IPD_CONFIG,
                                 strain_decay=best_decay,
                                 horizon_scaling=True, reference_rounds=10),
        'all_combined': replace(IPD_CONFIG,
                                strain_decay=best_decay,
                                horizon_scaling=True, reference_rounds=10,
                                elim_floor=0.3, elim_mult=2.0),
    }

    results = {}
    for vname, config in variants.items():
        print(f"\n    Variant: {vname}")
        r = run_variant(data, library, config, vname, verbose=True)
        results[vname] = r

    cf_acc, tft_acc = compute_baselines(data)

    # Summary table
    print(f"\n  {'Variant':<16s} {'Acc':>8s} {'Kappa':>8s} {'TrajRMSE':>9s} "
          f"{'Surv':>6s} {'Δ base':>8s}")
    print(f"  {'-' * 60}")
    print(f"  {'CF':<16s} {cf_acc:>8.1%}")
    print(f"  {'TFT':<16s} {tft_acc:>8.1%}")

    base_acc = results['baseline']['acc']
    for vname in variants:
        r = results[vname]
        delta = r['acc'] - base_acc
        print(f"  {vname:<16s} {r['acc']:>8.1%} {r['kappa']:>8.3f} "
              f"{r['traj_rmse']:>9.4f} {r['mean_survivors']:>6.1f} "
              f"{delta:>+8.1%}")

    # Per-type breakdown for baseline vs best
    print(f"\n  Per-type accuracy:")
    types = ['mostly-D', 'mixed', 'mostly-C']
    header = f"    {'Variant':<16s}"
    for t in types:
        header += f"  {t:>10s}"
    print(header)
    print(f"    {'-' * 50}")

    for vname in ['baseline', 'all_combined']:
        r = results[vname]
        row = f"    {vname:<16s}"
        for t in types:
            val = r['type_acc'].get(t, float('nan'))
            row += f"  {val:>10.1%}" if not np.isnan(val) else f"  {'—':>10s}"
        print(row)

    # Per-type kappa
    print(f"\n  Per-type kappa:")
    print(header)
    print(f"    {'-' * 50}")
    for vname in ['baseline', 'all_combined']:
        r = results[vname]
        row = f"    {vname:<16s}"
        for t in types:
            val = r['type_kappa'].get(t, float('nan'))
            row += f"  {val:>10.3f}" if not np.isnan(val) else f"  {'—':>10s}"
        print(row)

    return results


# ================================================================
# PHASE 3: BEST CONFIG ON BOTH TREATMENTS
# ================================================================

def phase3_comparison(sp_data, fp_data, library, best_decay):
    """Run the best combined config on both SP and FP."""
    print(f"\n{'=' * 72}")
    print(f"  PHASE 3: BEST REFINEMENT ON BOTH TREATMENTS")
    print(f"  (decay={best_decay:.2f}, horizon_scaling, elim 0.3/2.0)")
    print(f"{'=' * 72}")

    best_config = replace(IPD_CONFIG,
                          strain_decay=best_decay,
                          horizon_scaling=True, reference_rounds=10,
                          elim_floor=0.3, elim_mult=2.0)

    baseline_config = replace(IPD_CONFIG)

    treatments = [
        ('SP', sp_data),
        ('FP', fp_data),
    ]

    for treat_label, data in treatments:
        print(f"\n  --- {treat_label} ---")
        cf_acc, tft_acc = compute_baselines(data)

        print(f"    Running baseline...")
        base = run_variant(data, library, baseline_config,
                           f'{treat_label}_baseline', verbose=True)
        print(f"    Running refined...")
        refined = run_variant(data, library, best_config,
                              f'{treat_label}_refined', verbose=True)

        print(f"\n    {'':>16s} {'Baseline':>10s} {'Refined':>10s} {'Δ':>8s}")
        print(f"    {'-' * 48}")
        print(f"    {'CF':<16s} {cf_acc:>10.1%}")
        print(f"    {'TFT':<16s} {tft_acc:>10.1%}")
        print(f"    {'Accuracy':<16s} {base['acc']:>10.1%} "
              f"{refined['acc']:>10.1%} "
              f"{refined['acc'] - base['acc']:>+8.1%}")
        print(f"    {'Kappa':<16s} {base['kappa']:>10.3f} "
              f"{refined['kappa']:>10.3f} "
              f"{refined['kappa'] - base['kappa']:>+8.3f}")
        print(f"    {'Traj RMSE':<16s} {base['traj_rmse']:>10.4f} "
              f"{refined['traj_rmse']:>10.4f} "
              f"{refined['traj_rmse'] - base['traj_rmse']:>+8.4f}")
        print(f"    {'Survivors':<16s} {base['mean_survivors']:>10.1f} "
              f"{refined['mean_survivors']:>10.1f}")

        # Per-type
        types = ['mostly-D', 'mixed', 'mostly-C']
        print(f"\n    Per-type accuracy:")
        header = f"      {'Type':<12s} {'Baseline':>10s} {'Refined':>10s} {'Δ':>8s}"
        print(header)
        print(f"      {'-' * 42}")
        for t in types:
            b_val = base['type_acc'].get(t, float('nan'))
            r_val = refined['type_acc'].get(t, float('nan'))
            if not np.isnan(b_val) and not np.isnan(r_val):
                print(f"      {t:<12s} {b_val:>10.1%} {r_val:>10.1%} "
                      f"{r_val - b_val:>+8.1%}")
            else:
                print(f"      {t:<12s} {'—':>10s} {'—':>10s}")

    # Dynamics trace for worst mostly-C case under refined config
    print(f"\n  --- Dynamics trace: FP mostly-C under refined config ---")
    fp_sids = sorted(fp_data.keys())
    fp_coops = {sid: sum(r.contribution for r in fp_data[sid]) / len(fp_data[sid])
                for sid in fp_sids}
    mostly_c = [s for s in fp_sids if fp_coops[s] > 0.8]

    if mostly_c:
        # Find a high-coop subject that was poorly predicted before
        # Run with best_config to see improved dynamics
        trace_sid = mostly_c[len(mostly_c) // 2]  # median mostly-C subject
        rounds = fp_data[trace_sid]
        actual = [r.contribution for r in rounds]

        # Run with baseline
        base_preds_all = precompute_predictions(rounds, library, baseline_config)
        base_preds, _, base_surv = ensemble_select(
            base_preds_all, actual, baseline_config)

        # Run with refined
        ref_preds_all = precompute_predictions(rounds, library, best_config)
        ref_preds, _, ref_surv = ensemble_select(
            ref_preds_all, actual, best_config)

        base_acc = accuracy_from_round(base_preds, actual, 1)
        ref_acc = accuracy_from_round(ref_preds, actual, 1)

        print(f"    Subject: {trace_sid} (coop={fp_coops[trace_sid]:.1%})")
        print(f"    Baseline acc: {base_acc:.1%}, Refined acc: {ref_acc:.1%}")

        # Run best survivor with each config to get traces
        if ref_surv:
            from ipd_transfer_test import load_library
            best_lib_sid = ref_surv[0]
            lib = load_library()

            for cfg_label, cfg in [('baseline', baseline_config),
                                    ('refined', best_config)]:
                params = v3_params_to_v4(lib[best_lib_sid]['v3_params'])
                result = run_vcms_v4(params, rounds, cfg)
                trace = result['trace']

                print(f"\n    {cfg_label} dynamics (survivor={best_lib_sid}):")
                checkpoints = [0, 4, 9, 19, 49, 99]
                checkpoints = [c for c in checkpoints if c < len(trace)]
                print(f"      {'Round':>5s} {'Act':>3s} {'Pred':>4s} "
                      f"{'V_lvl':>6s} {'Strain':>7s} {'B':>6s} "
                      f"{'Afford':>7s} {'h_fac':>6s}")
                for c in checkpoints:
                    t = trace[c]
                    print(f"      {c + 1:>5d} {actual[c]:>3d} "
                          f"{result['pred_contrib'][c]:>4d} "
                          f"{t['v']['v_level']:>6.3f} "
                          f"{t['s_accum']['strain_pre_discharge']:>7.3f} "
                          f"{t['budget']['b_post']:>6.3f} "
                          f"{t['routing']['affordability']:>7.3f} "
                          f"{t['c_output']['h_factor']:>6.3f}")


# ================================================================
# BACKWARD COMPAT CHECK
# ================================================================

def check_pgg_compat(library):
    """Quick check: PGG configs still produce identical results."""
    print(f"\n  PGG backward compat check...")
    from pgg_vcms_agent_v3 import VCMSParams as V3Params, run_vcms_agent as run_v3

    # Check 5 random library subjects
    sids = sorted(library.keys())[:5]
    mismatches = 0
    total = 0

    for sid in sids:
        rec = library[sid]
        # Reconstruct rounds
        rounds = []
        n = len(rec['contribution_trajectory'])
        for t in range(n):
            from v4_validation import SimpleRound
            rounds.append(SimpleRound(
                contribution=rec['contribution_trajectory'][t],
                others_mean=rec['others_mean_trajectory'][t],
                pun_sent=rec['punishment_sent_trajectory'][t],
                pun_recv=rec['punishment_received_trajectory'][t],
            ))

        v3_params = V3Params(**rec['v3_params'])
        v3_result = run_v3(v3_params, rounds)

        v4_params = v3_params_to_v4(rec['v3_params'])
        v4_result = run_vcms_v4(v4_params, rounds, PGG_P_CONFIG)

        for t in range(n):
            total += 1
            if (v3_result['pred_contrib'][t] != v4_result['pred_contrib'][t] or
                    v3_result['pred_punish'][t] != v4_result['pred_punish'][t]):
                mismatches += 1

    if mismatches == 0:
        print(f"    PASS: {total} predictions across {len(sids)} subjects identical")
    else:
        print(f"    FAIL: {mismatches}/{total} mismatches!")
    return mismatches == 0


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  IPD REFINEMENT TEST")
    print("  Strain decay + Horizon scaling + Tighter elimination")
    print("=" * 72)

    # Load data
    print("\nLoading data...")
    library = json.load(open('v3_library_fitted.json'))
    sp_data = load_ipd_experiment('IPD-rand.csv')
    fp_data = load_ipd_experiment('fix.csv')
    print(f"  Library: {len(library)} PGG subjects")
    print(f"  SP: {len(sp_data)} subjects, FP: {len(fp_data)} subjects")

    # Backward compat
    if not check_pgg_compat(library):
        print("  *** PGG BACKWARD COMPAT FAILED — aborting ***")
        return

    # Phase 1: decay sweep on SP
    best_decay, decay_results = phase1_decay_sweep(sp_data, library, "SP")

    # Phase 2: ablation on SP
    ablation_results = phase2_ablation(sp_data, library, best_decay, "SP")

    # Phase 3: best config on both treatments
    phase3_comparison(sp_data, fp_data, library, best_decay)

    print(f"\n{'=' * 72}")
    print(f"  REFINEMENT TEST COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
