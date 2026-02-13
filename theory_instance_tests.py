#!/usr/bin/env python3
"""
Theory Instance Tests
=====================

Three tests proposed after theory instance feedback:

1. COMBINED LIBRARY TRANSFER
   576-subject v4 library (P + N + IPD) → IPD prediction.
   Key question: does adding N-experiment subjects improve
   committed cooperator identification?

2. PROSOCIAL STRAIN (s_exploitation_rate)
   Re-fit the 15 rising-type N-experiment subjects with
   s_exploitation_rate as a free parameter. Test whether
   prosocial strain (strain from under-contributing) improves
   fit quality for this worst-fitted type.

3. BUDGET RESILIENCE SIMULATION
   Run HIGH vs LOW cooperator parameter profiles through hostile
   environments (low others_mean). Test whether HIGH cooperators'
   budget is structurally self-sustaining (replenish/deplete > 1
   even under adversity), confirming dispositional resilience.
"""

import json
import math
import time
import os
import numpy as np
from collections import defaultdict, Counter
from scipy.optimize import differential_evolution

from dataclasses import dataclass
from pgg_p_loader import load_p_experiment
from vcms_engine_v4 import (
    GameConfig, VCMSParams, run_vcms_v4, v3_params_to_v4,
    PARAM_BOUNDS_NORMALIZED,
)
from ipd_loader import load_ipd_experiment
from ipd_transfer_test import (
    precompute_candidate_predictions, ensemble_from_precomputed,
    accuracy_from_round, cohens_kappa, trajectory_rmse,
    classify_subject, run_baselines,
)
from normalized_fit import (
    predict_fast_normalized, objective, fit_subject_de,
    FIT_PARAM_NAMES, FIXED_PARAMS, params_array_to_dict, FIT_BOUNDS,
)
from v3_n_experiment_test import N_EXPERIMENT_CSVS


@dataclass
class SimpleRound:
    """Minimal round interface for VCMS engine simulation."""
    contribution: int
    others_mean: float
    punishment_sent_total: int
    punishment_received_total: int


# Normalized-time configs
IPD_NORM_CONFIG = GameConfig(
    max_contrib=1, max_punish=1, has_punishment=False,
    n_signals=2, normalized_time=True,
)

PGG_N_NORM_CONFIG = GameConfig(
    max_contrib=20, max_punish=30, has_punishment=False,
    n_signals=2, normalized_time=True,
)


# ================================================================
# TEST 1: COMBINED LIBRARY TRANSFER
# ================================================================

def test_combined_library_transfer():
    """
    Run IPD transfer test with combined 576-subject library vs
    P-only (176) and P+N (388) subsets.

    Tests whether N-experiment subjects improve committed cooperator
    (mostly-C) prediction, since N:stable-high and IPD:mostly-C
    occupy the same parameter region.
    """
    print("\n" + "=" * 72)
    print("  TEST 1: COMBINED LIBRARY TRANSFER")
    print("  P-only (176) vs P+N (388) vs P+N+IPD (576) → IPD prediction")
    print("=" * 72)

    # Load libraries
    with open('v4_library_fitted.json') as f:
        p_lib = json.load(f)
    with open('v4_n_library_fitted.json') as f:
        n_lib = json.load(f)
    with open('v4_ipd_library_fitted.json') as f:
        ipd_lib = json.load(f)

    print(f"\n  P-library:   {len(p_lib)} subjects")
    print(f"  N-library:   {len(n_lib)} subjects")
    print(f"  IPD-library: {len(ipd_lib)} subjects")

    # Build combined libraries
    # P+N = 388 subjects (cross-game, no IPD self-prediction advantage)
    pn_lib = {}
    for sid, rec in p_lib.items():
        pn_lib[f"P_{sid}"] = rec
    for sid, rec in n_lib.items():
        pn_lib[f"N_{sid}"] = rec

    # P+N+IPD = 576 (includes IPD subjects — leave-one-out needed)
    full_lib = dict(pn_lib)
    for sid, rec in ipd_lib.items():
        full_lib[f"IPD_{sid}"] = rec

    print(f"  P+N combined: {len(pn_lib)} subjects")
    print(f"  Full combined: {len(full_lib)} subjects")

    # Load IPD test data — use FP (fixed pairing, stronger signal)
    fp_data = load_ipd_experiment('fix.csv')
    sp_data = load_ipd_experiment('IPD-rand.csv')
    print(f"\n  FP test set: {len(fp_data)} subjects")
    print(f"  SP test set: {len(sp_data)} subjects")

    # Monkey-patch config for normalized time
    import ipd_transfer_test as itt
    original_config = itt.IPD_CONFIG
    itt.IPD_CONFIG = IPD_NORM_CONFIG

    # We'll run the transfer on FP data with three library sizes.
    # For the full library, we need to exclude the target's own IPD entry.
    results = {}
    for lib_label, library in [("P-only (176)", p_lib),
                                ("P+N (388)", pn_lib)]:
        print(f"\n  Running transfer: {lib_label}...")
        r = _run_transfer_metrics(fp_data, library, lib_label)
        results[lib_label] = r

    # P+N+IPD with leave-one-out on IPD entries
    print(f"\n  Running transfer: P+N+IPD (576, LOO on IPD)...")
    r = _run_transfer_metrics_loo(fp_data, full_lib, ipd_lib, "P+N+IPD (576)")
    results["P+N+IPD (576)"] = r

    # Also run on SP
    print(f"\n  Running transfer on SP data: P+N (388)...")
    r_sp_pn = _run_transfer_metrics(sp_data, pn_lib, "P+N SP")
    results["P+N SP"] = r_sp_pn

    # Restore
    itt.IPD_CONFIG = original_config

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f"\n{'=' * 72}")
    print(f"  COMBINED LIBRARY TRANSFER RESULTS (FP)")
    print(f"{'=' * 72}")

    types = ['mostly-D', 'mixed', 'mostly-C']
    print(f"\n  --- Overall accuracy (from round 2) ---")
    print(f"  {'Library':<22s} {'Overall':>8s} {'mostly-D':>10s} "
          f"{'mixed':>8s} {'mostly-C':>10s} {'Traj RMSE':>10s}")
    print(f"  {'-' * 70}")

    for lib_label in ["P-only (176)", "P+N (388)", "P+N+IPD (576)"]:
        r = results[lib_label]
        print(f"  {lib_label:<22s} {r['overall_acc']:>7.1%} "
              f"{r['type_acc'].get('mostly-D', 0):>9.1%} "
              f"{r['type_acc'].get('mixed', 0):>8.1%} "
              f"{r['type_acc'].get('mostly-C', 0):>9.1%} "
              f"{r['traj_rmse']:>10.4f}")

    print(f"\n  --- SP comparison (P+N library) ---")
    r = results["P+N SP"]
    print(f"  {'P+N SP':<22s} {r['overall_acc']:>7.1%} "
          f"{r['type_acc'].get('mostly-D', 0):>9.1%} "
          f"{r['type_acc'].get('mixed', 0):>8.1%} "
          f"{r['type_acc'].get('mostly-C', 0):>9.1%} "
          f"{r['traj_rmse']:>10.4f}")

    # Delta analysis
    print(f"\n  --- Delta from P-only baseline ---")
    base = results["P-only (176)"]
    for lib_label in ["P+N (388)", "P+N+IPD (576)"]:
        r = results[lib_label]
        print(f"\n  {lib_label}:")
        print(f"    Overall:  {r['overall_acc'] - base['overall_acc']:+.1%}")
        for t in types:
            delta = r['type_acc'].get(t, 0) - base['type_acc'].get(t, 0)
            print(f"    {t:<10s}: {delta:+.1%}")
        print(f"    Traj RMSE: {r['traj_rmse'] - base['traj_rmse']:+.4f}")

    # Survivor analysis for mostly-C
    print(f"\n  --- Survivor origin for mostly-C (FP, P+N library) ---")
    r_pn = results["P+N (388)"]
    if 'survivors_by_origin' in r_pn:
        origins = r_pn['survivors_by_origin'].get('mostly-C', {})
        total_s = sum(origins.values())
        for origin, count in sorted(origins.items(), key=lambda x: -x[1]):
            print(f"    {origin}: {count} ({count/max(total_s,1):.0%})")

    return results


def _run_transfer_metrics(data, library, label):
    """Run transfer test and collect summary metrics."""
    sids = sorted(data.keys())
    subject_types = {}
    for sid in sids:
        rounds = data[sid]
        cr = sum(r.contribution for r in rounds) / len(rounds)
        subject_types[sid] = classify_subject(cr)

    all_acc = []
    type_acc = defaultdict(list)
    all_traj = []
    survivors_by_origin = defaultdict(lambda: defaultdict(int))

    t0 = time.time()
    for idx, sid in enumerate(sids):
        rounds = data[sid]
        actual = [r.contribution for r in rounds]

        predictions = precompute_candidate_predictions(rounds, library, IPD_NORM_CONFIG)
        preds, surv_counts, final_surv = ensemble_from_precomputed(
            predictions, actual, max_contrib=1)

        acc = accuracy_from_round(preds, actual, 1)
        all_acc.append(acc)
        type_acc[subject_types[sid]].append(acc)
        all_traj.append(trajectory_rmse(preds, actual, window=10))

        # Track survivor origins
        stype = subject_types[sid]
        for s in final_surv:
            if s.startswith('P_'):
                survivors_by_origin[stype]['P-experiment'] += 1
            elif s.startswith('N_'):
                survivors_by_origin[stype]['N-experiment'] += 1
            elif s.startswith('IPD_'):
                survivors_by_origin[stype]['IPD'] += 1
            else:
                survivors_by_origin[stype]['P-experiment'] += 1  # original P lib

        if (idx + 1) % 20 == 0 or idx == len(sids) - 1:
            elapsed = time.time() - t0
            print(f"    [{idx + 1:>3d}/{len(sids)}] {elapsed:.0f}s")

    return {
        'overall_acc': np.mean(all_acc),
        'type_acc': {t: np.mean(v) for t, v in type_acc.items()},
        'traj_rmse': np.mean(all_traj),
        'survivors_by_origin': dict(survivors_by_origin),
    }


def _run_transfer_metrics_loo(data, full_lib, ipd_lib, label):
    """Run transfer with leave-one-out for IPD entries."""
    sids = sorted(data.keys())
    subject_types = {}
    for sid in sids:
        rounds = data[sid]
        cr = sum(r.contribution for r in rounds) / len(rounds)
        subject_types[sid] = classify_subject(cr)

    all_acc = []
    type_acc = defaultdict(list)
    all_traj = []

    t0 = time.time()
    for idx, sid in enumerate(sids):
        rounds = data[sid]
        actual = [r.contribution for r in rounds]

        # Exclude this subject's own IPD entry
        loo_lib = {k: v for k, v in full_lib.items() if k != f"IPD_{sid}"}

        predictions = precompute_candidate_predictions(rounds, loo_lib, IPD_NORM_CONFIG)
        preds, surv_counts, final_surv = ensemble_from_precomputed(
            predictions, actual, max_contrib=1)

        acc = accuracy_from_round(preds, actual, 1)
        all_acc.append(acc)
        type_acc[subject_types[sid]].append(acc)
        all_traj.append(trajectory_rmse(preds, actual, window=10))

        if (idx + 1) % 20 == 0 or idx == len(sids) - 1:
            elapsed = time.time() - t0
            print(f"    [{idx + 1:>3d}/{len(sids)}] {elapsed:.0f}s")

    return {
        'overall_acc': np.mean(all_acc),
        'type_acc': {t: np.mean(v) for t, v in type_acc.items()},
        'traj_rmse': np.mean(all_traj),
    }


# ================================================================
# TEST 2: PROSOCIAL STRAIN
# ================================================================

def predict_fast_with_exploitation(x, rounds, max_c):
    """
    Fast forward pass with s_exploitation_rate as a free parameter.

    16 parameters: the 15 standard + s_exploitation_rate.
    When a subject contributes LESS than the group mean,
    prosocial strain accumulates — strain from under-contributing.
    """
    alpha, v_rep, v_ref, c_base, inertia, s_dir_raw, s_rate, s_initial, \
        b_initial, b_depletion_rate, b_replenish_rate, acute_threshold, \
        facilitation_rate, h_strength, h_start, s_exploitation_rate = x

    s_dir = 1.0 if s_dir_raw >= 0 else -1.0
    w = max(-0.3, min(0.95, inertia))

    n = len(rounds)
    dt = 1.0 / (n - 1) if n > 1 else 1.0
    h_start_round = h_start * (n - 1)

    v_level = 0.0
    disposition = 0.0
    strain = s_initial
    B = b_initial
    m_eval = 0.0
    c_prev_norm = 0.0

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

        # Strain: standard gap + prosocial exploitation
        if i > 0:
            gap = c_prev_norm - reference
            directed_gap = gap * s_dir
            gap_strain = max(0.0, directed_gap)
            strain += dt * s_rate * gap_strain

            # Prosocial strain: strain from contributing LESS than group
            # exploitation_gap > 0 when group mean > own contribution
            exploitation_gap = max(0.0, v_group_raw - c_prev_norm)
            strain += dt * s_exploitation_rate * exploitation_gap

        # Budget
        if i > 0:
            experience = v_group_raw - c_prev_norm
            if experience < 0:
                magnitude = -experience
                depletion = dt * b_depletion_rate * magnitude
                if magnitude > acute_threshold:
                    depletion *= 5.0
                B -= depletion
            elif experience > 0:
                B += dt * b_replenish_rate * experience
            B = max(0.0, B)
            m_eval += dt * facilitation_rate * experience

        affordability = B / (B + strain + 0.01)

        if i == 0:
            c_norm = c_base
        else:
            c_target = v_ref * v_level + (1.0 - v_ref) * c_base
            c_target_adj = max(0.0, min(1.0, c_target + m_eval))
            c_norm = (1.0 - abs(w)) * c_target_adj + w * c_prev_norm

        # Horizon
        if n > 1 and h_strength > 0.0 and i >= h_start_round:
            denom = n - 1 - h_start_round
            if denom > 0:
                progress = min(1.0, (i - h_start_round) / denom)
                h_factor = 1.0 - h_strength * progress
            else:
                h_factor = (1.0 - h_strength) if i >= n - 1 else 1.0
        else:
            h_factor = 1.0

        c_out_norm = max(0.0, min(1.0, c_norm)) * affordability * h_factor
        c_out = round(c_out_norm * max_c)
        preds[i] = max(0, min(max_c, c_out))

        c_prev_norm = rd.contribution / max_c

    return preds


def test_prosocial_strain():
    """
    Re-fit rising-type N-experiment subjects with s_exploitation_rate
    as a free parameter (16 params instead of 15).

    Hypothesis: subjects who INCREASE cooperation over time experience
    prosocial strain — strain from under-contributing relative to the
    group. This pushes contributions UP rather than down. The current
    model can only push contributions down through strain.

    Also test on declining and stable-high for comparison.
    """
    print("\n" + "=" * 72)
    print("  TEST 2: PROSOCIAL STRAIN (s_exploitation_rate)")
    print("  Re-fitting rising-type subjects with exploitation strain channel")
    print("=" * 72)

    # Load N-experiment data
    all_data = {}
    city_map = {}
    for city, csv_path in N_EXPERIMENT_CSVS:
        if not os.path.exists(csv_path):
            continue
        data = load_p_experiment(csv_path)
        for sid, rounds in data.items():
            all_data[sid] = rounds
            city_map[sid] = city

    # Load existing v4 library for comparison
    with open('v4_n_library_fitted.json') as f:
        n_lib = json.load(f)

    # Identify subjects by type
    type_sids = defaultdict(list)
    for sid, rec in n_lib.items():
        type_sids[rec['subject_type']].append(sid)

    # 16-parameter bounds (15 standard + s_exploitation_rate)
    exploit_bounds = list(FIT_BOUNDS) + [(0.0, 18.0)]  # 18 = 2.0 * 9 normalized

    def objective_exploit(x, rounds, actual, max_c):
        preds = predict_fast_with_exploitation(x, rounds, max_c)
        n = len(actual)
        sse = sum((actual[i] - preds[i]) ** 2 for i in range(n))
        return math.sqrt(sse / n)

    # Test on: rising (target), declining (control), stable-high (control)
    test_types = ['rising', 'declining', 'stable-high']

    for stype in test_types:
        sids = type_sids[stype]
        n_test = len(sids)
        print(f"\n  --- {stype} ({n_test} subjects) ---")

        v4_rmses = []
        exploit_rmses = []
        exploit_rates = []
        improvements = []

        for idx, sid in enumerate(sids):
            if sid not in all_data:
                continue
            rounds = all_data[sid]
            actual = [r.contribution for r in rounds]
            v4_rmse = n_lib[sid]['fit_rmse']

            # Fit with exploitation channel
            result = differential_evolution(
                objective_exploit, exploit_bounds,
                args=(rounds, actual, 20),
                maxiter=100, popsize=10, tol=0.005,
                seed=42, polish=True, disp=False,
            )
            exploit_rmse = result.fun
            s_exploit_val = result.x[15]

            v4_rmses.append(v4_rmse)
            exploit_rmses.append(exploit_rmse)
            exploit_rates.append(s_exploit_val)
            improvements.append(v4_rmse - exploit_rmse)

            if stype == 'rising' or (idx + 1) == len(sids):
                preds_v4 = n_lib[sid].get('fit_pred', [])
                preds_exploit = predict_fast_with_exploitation(
                    result.x, rounds, 20)
                print(f"    {sid}: v4={v4_rmse:.2f} → exploit={exploit_rmse:.2f} "
                      f"(Δ={v4_rmse - exploit_rmse:+.2f}) "
                      f"s_exploit_rate={s_exploit_val:.3f}")
                if stype == 'rising':
                    print(f"          actual={actual}")
                    print(f"          v4_pred={preds_v4}")
                    print(f"          ex_pred={list(preds_exploit)}")

        if v4_rmses:
            print(f"\n    Summary ({stype}):")
            print(f"      v4 mean RMSE:      {np.mean(v4_rmses):.3f}")
            print(f"      exploit mean RMSE: {np.mean(exploit_rmses):.3f}")
            print(f"      Mean improvement:  {np.mean(improvements):+.3f}")
            print(f"      Mean s_exploit:    {np.mean(exploit_rates):.3f}")
            print(f"      Median s_exploit:  {np.median(exploit_rates):.3f}")

            n_better = sum(1 for d in improvements if d > 0.1)
            n_same = sum(1 for d in improvements if abs(d) <= 0.1)
            n_worse = sum(1 for d in improvements if d < -0.1)
            print(f"      Better/Same/Worse: {n_better}/{n_same}/{n_worse}")

    return True


# ================================================================
# TEST 3: BUDGET RESILIENCE SIMULATION
# ================================================================

def _simulate_self_play(params, others_mean, n_rounds, game_config):
    """
    Inline self-play simulation: agent's own output feeds back as
    its contribution each round. Environment provides fixed others_mean.

    Inlines the VCMS forward pass for efficiency (~100x faster than
    calling run_vcms_v4 in a loop).

    Returns (budget_trajectory, contribution_list).
    """
    p = params
    gc = game_config
    max_c = gc.max_contrib
    dt = 1.0 / (n_rounds - 1) if gc.normalized_time and n_rounds > 1 else 1.0

    s_dir = 1.0 if p.s_dir >= 0 else -1.0
    w = max(-0.3, min(0.95, p.inertia))

    if gc.normalized_time:
        h_start_round = p.h_start * (n_rounds - 1)
    else:
        h_start_round = p.h_start

    v_level = 0.0
    disposition = 0.0
    strain = p.s_initial
    B = p.b_initial
    m_eval = 0.0
    c_prev_norm = 0.0
    v_group_raw = others_mean / max_c

    budget_traj = []
    contribs = []

    for i in range(n_rounds):
        v_group = min(1.0, p.v_rep * v_group_raw)

        if i == 0:
            v_level = v_group
            # First action = c_base (self-play: no external contribution)
            c_prev_norm = p.c_base
            disposition = c_prev_norm
        else:
            v_level = p.alpha * v_group + (1.0 - p.alpha) * v_level
            disposition = 0.15 * c_prev_norm + 0.85 * disposition

        reference = p.v_ref * v_level + (1.0 - p.v_ref) * disposition

        # Strain
        if i > 0:
            gap = c_prev_norm - reference
            gap_strain = max(0.0, gap * s_dir)
            strain += dt * p.s_rate * gap_strain

        # Budget
        if i > 0:
            experience = v_group_raw - c_prev_norm
            if experience < 0:
                magnitude = -experience
                depletion = dt * p.b_depletion_rate * magnitude
                if magnitude > p.acute_threshold:
                    depletion *= 5.0
                B -= depletion
            elif experience > 0:
                B += dt * p.b_replenish_rate * experience
            B = max(0.0, B)
            m_eval += dt * p.facilitation_rate * experience

        affordability = B / (B + strain + 0.01)

        # Contribution
        if i == 0:
            c_norm = p.c_base
        else:
            c_target = p.v_ref * v_level + (1.0 - p.v_ref) * p.c_base
            c_target_adj = max(0.0, min(1.0, c_target + m_eval))
            c_norm = (1.0 - abs(w)) * c_target_adj + w * c_prev_norm

        # Horizon
        h_factor = 1.0
        if n_rounds > 1 and p.h_strength > 0.0 and i >= h_start_round:
            denom = n_rounds - 1 - h_start_round
            if denom > 0:
                progress = min(1.0, (i - h_start_round) / denom)
                h_factor = 1.0 - p.h_strength * progress
            elif i >= n_rounds - 1:
                h_factor = 1.0 - p.h_strength

        c_out_norm = max(0.0, min(1.0, c_norm)) * affordability * h_factor
        c_out = max(0, min(max_c, round(c_out_norm * max_c)))

        budget_traj.append(B)
        contribs.append(c_out)

        # Self-play: agent's output is its own contribution next round
        c_prev_norm = c_out / max_c

    return budget_traj, contribs


def test_budget_resilience():
    """
    Simulate HIGH vs LOW cooperator parameter profiles in hostile
    environments. Hostile = others_mean stays low (defecting group).

    Tests whether HIGH cooperators' budget is structurally self-
    sustaining (replenish > deplete even under adversity), confirming
    dispositional resilience vs environment-contingent cooperation.
    """
    print("\n" + "=" * 72)
    print("  TEST 3: BUDGET RESILIENCE SIMULATION")
    print("  HIGH vs LOW cooperator profiles in hostile environments")
    print("=" * 72)

    # Load all three libraries
    with open('v4_library_fitted.json') as f:
        p_lib = json.load(f)
    with open('v4_n_library_fitted.json') as f:
        n_lib = json.load(f)
    with open('v4_ipd_library_fitted.json') as f:
        ipd_lib = json.load(f)

    # Unified phenotype mapping
    def get_phenotype(rec, lib_name):
        if lib_name == 'P':
            bp = rec.get('behavioral_profile', '')
            # behavioral_profile is a string like 'cooperator', 'cooperative-enforcer', etc.
            if isinstance(bp, dict):
                bp = bp.get('type', '')
            if any(k in bp for k in ('cooperator', 'enforcer')):
                return 'HIGH'
            elif any(k in bp for k in ('free-rider', 'antisocial')):
                return 'LOW'
            return 'MID'
        elif lib_name == 'N':
            stype = rec.get('subject_type', '')
            if stype == 'stable-high':
                return 'HIGH'
            elif stype == 'stable-low':
                return 'LOW'
            return 'MID'
        elif lib_name == 'IPD':
            stype = rec.get('subject_type', '')
            if stype == 'mostly-C':
                return 'HIGH'
            elif stype == 'mostly-D':
                return 'LOW'
            return 'MID'

    # Collect parameter profiles by phenotype
    profiles = {'HIGH': [], 'LOW': []}
    for lib, lib_name in [(p_lib, 'P'), (n_lib, 'N'), (ipd_lib, 'IPD')]:
        for sid, rec in lib.items():
            pheno = get_phenotype(rec, lib_name)
            if pheno in profiles:
                profiles[pheno].append({
                    'params': rec['v3_params'],
                    'source': lib_name,
                    'sid': sid,
                })

    print(f"\n  HIGH cooperators: {len(profiles['HIGH'])} subjects")
    print(f"  LOW cooperators:  {len(profiles['LOW'])} subjects")

    # Simulate environments
    # Create synthetic rounds with varying hostility levels
    hostility_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_sim_rounds = 10  # Same as PGG length

    print(f"\n  Simulation: {n_sim_rounds} rounds, hostility = 1 - others_mean/max_c")
    print(f"  Environment: others contribute at (1-hostility) × max_contrib")

    results = {}

    for pheno in ['HIGH', 'LOW']:
        print(f"\n  --- {pheno} cooperators ---")
        pheno_results = {h: {'budget_final': [], 'budget_min': [],
                             'contributions': [], 'budget_trajectory': []}
                         for h in hostility_levels}

        for subject in profiles[pheno]:
            params = subject['params']
            v4_params = v3_params_to_v4(params)

            for hostility in hostility_levels:
                others_mean = (1.0 - hostility) * 20  # 0-20 scale

                # Self-play simulation: agent's own predictions feed back
                # as contribution each round (no teacher forcing).
                # This tests what the agent DOES, not what it predicts
                # given externally provided actions.
                budget_traj, contribs = _simulate_self_play(
                    v4_params, others_mean, n_sim_rounds, PGG_N_NORM_CONFIG)

                budget_final = budget_traj[-1] if budget_traj else 0
                budget_min = min(budget_traj) if budget_traj else 0
                mean_contrib = np.mean(contribs)

                pheno_results[hostility]['budget_final'].append(budget_final)
                pheno_results[hostility]['budget_min'].append(budget_min)
                pheno_results[hostility]['contributions'].append(mean_contrib)
                pheno_results[hostility]['budget_trajectory'].append(budget_traj)

        results[pheno] = pheno_results

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print(f"\n{'=' * 72}")
    print(f"  BUDGET RESILIENCE RESULTS")
    print(f"{'=' * 72}")

    print(f"\n  --- Mean final budget by hostility ---")
    print(f"  {'Hostility':>10s} {'HIGH B_final':>12s} {'LOW B_final':>12s} "
          f"{'HIGH B_min':>11s} {'LOW B_min':>11s}")
    print(f"  {'-' * 58}")

    for h in hostility_levels:
        h_high = results['HIGH'][h]
        h_low = results['LOW'][h]
        print(f"  {h:>10.1f} {np.mean(h_high['budget_final']):>12.3f} "
              f"{np.mean(h_low['budget_final']):>12.3f} "
              f"{np.mean(h_high['budget_min']):>11.3f} "
              f"{np.mean(h_low['budget_min']):>11.3f}")

    print(f"\n  --- Mean contribution output by hostility ---")
    print(f"  {'Hostility':>10s} {'HIGH contrib':>13s} {'LOW contrib':>12s} "
          f"{'HIGH coop%':>11s} {'LOW coop%':>11s}")
    print(f"  {'-' * 58}")

    for h in hostility_levels:
        h_high = results['HIGH'][h]
        h_low = results['LOW'][h]
        high_c = np.mean(h_high['contributions'])
        low_c = np.mean(h_low['contributions'])
        print(f"  {h:>10.1f} {high_c:>13.1f} {low_c:>12.1f} "
              f"{high_c/20:>10.0%} {low_c/20:>11.0%}")

    # Budget sustainability test: at what hostility does budget hit zero?
    print(f"\n  --- Budget sustainability threshold ---")
    for pheno in ['HIGH', 'LOW']:
        collapse_h = None
        for h in hostility_levels:
            mean_min = np.mean(results[pheno][h]['budget_min'])
            if mean_min < 0.1:
                collapse_h = h
                break
        if collapse_h is not None:
            print(f"  {pheno}: budget collapses at hostility = {collapse_h:.1f}")
        else:
            print(f"  {pheno}: budget survives all hostility levels")

    # Per-source breakdown
    print(f"\n  --- Budget resilience by library source (hostility=0.8) ---")
    h = 0.8
    for pheno in ['HIGH', 'LOW']:
        source_budgets = defaultdict(list)
        for i, subject in enumerate(profiles[pheno]):
            b_final = results[pheno][h]['budget_final'][i]
            source_budgets[subject['source']].append(b_final)

        print(f"\n  {pheno}:")
        for source in ['P', 'N', 'IPD']:
            if source in source_budgets:
                vals = source_budgets[source]
                print(f"    {source}: n={len(vals)}, "
                      f"mean B_final={np.mean(vals):.3f}, "
                      f"median={np.median(vals):.3f}")

    # The key test: is HIGH budget self-sustaining even under hostility?
    print(f"\n  --- Dispositional resilience test ---")
    print(f"  (Is HIGH budget structurally self-sustaining regardless of environment?)")
    h_mild = 0.3
    h_hostile = 0.7
    h_extreme = 0.9

    for h, label in [(h_mild, "mild"), (h_hostile, "hostile"), (h_extreme, "extreme")]:
        high_final = np.mean(results['HIGH'][h]['budget_final'])
        high_initial = np.mean([s['params']['b_initial'] for s in profiles['HIGH']])
        low_final = np.mean(results['LOW'][h]['budget_final'])
        low_initial = np.mean([s['params']['b_initial'] for s in profiles['LOW']])

        high_retained = high_final / max(high_initial, 0.001)
        low_retained = low_final / max(low_initial, 0.001)

        print(f"\n  {label} (hostility={h}):")
        print(f"    HIGH: B_initial={high_initial:.2f} → B_final={high_final:.2f} "
              f"({high_retained:.0%} retained)")
        print(f"    LOW:  B_initial={low_initial:.2f} → B_final={low_final:.2f} "
              f"({low_retained:.0%} retained)")

    return results


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  THEORY INSTANCE TESTS")
    print("  Combined library transfer + Prosocial strain + Budget resilience")
    print("=" * 72)

    t0 = time.time()

    # Test 1: Combined library transfer
    transfer_results = test_combined_library_transfer()

    # Test 2: Prosocial strain
    test_prosocial_strain()

    # Test 3: Budget resilience
    test_budget_resilience()

    total = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  ALL TESTS COMPLETE — {total:.0f}s total")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
