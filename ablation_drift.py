#!/usr/bin/env python3
"""
Ablation study: disentangling recency weighting vs delta-aware scoring
in the VCMS drift predictor.

Four configurations:
  1. Base drift    (RECENCY_DECAY=1.0, DELTA_BLEND=0.0) — uniform weighting, level-only
  2. Recency only  (RECENCY_DECAY=0.7, DELTA_BLEND=0.0)
  3. Delta only    (RECENCY_DECAY=1.0, DELTA_BLEND=0.5)
  4. Both          (RECENCY_DECAY=0.7, DELTA_BLEND=0.5) — the actual vcms_drift method

Reference baselines:
  - carry_forward: C(t) = C(t-1)
  - vcms_ensemble: base VCMS ensemble (level predictions, converted to drift form)
"""

import json
import math
import os
import sys
import time
import numpy as np
from collections import Counter, defaultdict

from pgg_p_loader import load_p_experiment
from pgg_vcms_agent_v3 import (
    VCMSParams, PARAM_NAMES, PARAM_BOUNDS,
    run_vcms_agent, make_vcms_params, vcms_objective,
    MAX_CONTRIB, MAX_PUNISH,
)

# ================================================================
# CONSTANTS
# ================================================================

K_VALUES = [1, 2, 3, 5]
N_ROUNDS = 10
MAX_CONTRIB_VAL = 20
RANDOM_SEED = 42
SUBSAMPLE_N = 40

ABLATION_CONFIGS = {
    'base_drift':   {'RECENCY_DECAY': 1.0, 'DELTA_BLEND': 0.0},
    'recency_only': {'RECENCY_DECAY': 0.7, 'DELTA_BLEND': 0.0},
    'delta_only':   {'RECENCY_DECAY': 1.0, 'DELTA_BLEND': 0.5},
    'both':         {'RECENCY_DECAY': 0.7, 'DELTA_BLEND': 0.5},
}

ALL_METHODS = list(ABLATION_CONFIGS.keys()) + ['carry_forward', 'vcms_ensemble']


# ================================================================
# DATA LOADING (from v3_cross_validation.py)
# ================================================================

def load_all_data():
    """Load all city CSVs and the fitted library."""
    with open('v3_library_fitted.json') as f:
        library = json.load(f)

    city_csvs = [
        ('Samara', 'HerrmannThoeniGaechterDATA_SAMARA_P-EXPERIMENT_TRUNCATED_SESSION1.csv'),
        ('Samara', 'HerrmannThoeniGaechterDATA_SAMARA_P-EXPERIMENT_TRUNCATED_SESSION2.csv'),
        ('Samara', 'HerrmannThoeniGaechterDATA_SAMARA_P-EXPERIMENT_TRUNCATED_SESSION3.csv'),
        ('Chengdu', 'HerrmannThoeniGaechterDATA_CHENGDU_P-EXPERIMENT_TRUNCATED_SESSION1.csv'),
        ('Zurich', 'HerrmannThoeniGaechterDATA_ZURICH_P-EXPERIMENT_TRUNCATED_SESSION1.csv'),
        ('Athens', 'HerrmannThoeniGaechterDATA_ATHENS_P-EXPERIMENT_TRUNCATED_SESSION1.csv'),
        ('Boston', 'HerrmannThoeniGaechterDATA_BOSTON_P-EXPERIMENT_TRUNCATED_SESSION1.csv'),
        ('Istanbul', 'HerrmannThoeniGaechterDATA_ISTANBUL_P-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ]

    all_rounds = {}
    city_map = {}

    for city, csv_path in city_csvs:
        if not os.path.exists(csv_path):
            continue
        data = load_p_experiment(csv_path)
        for sid, rounds in data.items():
            if sid in library:
                all_rounds[sid] = rounds
                city_map[sid] = city

    return library, all_rounds, city_map


# ================================================================
# STRATIFIED SUBSAMPLE
# ================================================================

def stratified_subsample(library, all_rounds, n_target, seed):
    """
    Sample ~n_target subjects proportionally from each behavioral profile.
    Returns list of sids.
    """
    rng = np.random.RandomState(seed)

    # Group sids by profile
    profile_sids = defaultdict(list)
    for sid in sorted(all_rounds.keys()):
        profile = library[sid]['behavioral_profile']
        profile_sids[profile].append(sid)

    total = len(all_rounds)
    selected = []

    for profile, sids in sorted(profile_sids.items()):
        # Proportional allocation, at least 1 per profile
        n_profile = max(1, round(len(sids) / total * n_target))
        n_profile = min(n_profile, len(sids))
        chosen = rng.choice(sids, size=n_profile, replace=False).tolist()
        selected.extend(chosen)

    return selected


# ================================================================
# BASELINES
# ================================================================

def carry_forward_predict(rounds):
    """Predict C(t) = C(t-1). Round 1: use round 1 actual."""
    pred = []
    for t in range(min(N_ROUNDS, len(rounds))):
        if t == 0:
            pred.append(rounds[0].contribution)
        else:
            pred.append(rounds[t - 1].contribution)
    return pred


# ================================================================
# VCMS ENSEMBLE (base level predictor, converted to drift)
# ================================================================

def vcms_predict_as_drift(target_rounds, library, exclude_sids):
    """
    Run base VCMS ensemble (no recency/delta enhancements),
    then convert level predictions to CF + drift form.
    """
    candidates = {}
    for sid, rec in library.items():
        if sid in exclude_sids:
            continue
        candidates[sid] = {
            'params': VCMSParams(**rec['v3_params']),
        }

    if not candidates:
        return [10] * min(N_ROUNDS, len(target_rounds))

    survivors = list(candidates.keys())
    ensemble_preds = []
    obs_c_list = []
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}

    n = min(N_ROUNDS, len(target_rounds))

    for t in range(n):
        cand_preds_c = {}
        if t == 0:
            for sid in survivors:
                result = run_vcms_agent(
                    candidates[sid]['params'], target_rounds[:1])
                cand_preds_c[sid] = result['pred_contrib'][0]
                cand_pred_history[sid].append(result['pred_contrib'][0])
        else:
            for sid in survivors:
                result = run_vcms_agent(
                    candidates[sid]['params'], target_rounds[:t + 1])
                cand_preds_c[sid] = result['pred_contrib'][t]
                cand_pred_history[sid] = list(result['pred_contrib'][:t + 1])

        if not distances:
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001)
                       for sid in survivors}
        total_w = sum(weights.values())

        pc = 0.0
        for sid in survivors:
            pc += (weights[sid] / total_w) * cand_preds_c[sid]
        ensemble_preds.append(int(round(pc)))

        obs_c_list.append(target_rounds[t].contribution)

        # Standard elimination (no recency, no delta)
        new_distances = {}
        for sid in survivors:
            hist = cand_pred_history[sid]
            nc = min(t + 1, len(hist))
            if nc == 0:
                new_distances[sid] = 0.0
                continue
            c_d = sum((obs_c_list[i] - hist[i]) ** 2
                      for i in range(nc)) / (MAX_CONTRIB_VAL ** 2 * nc)
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

    # Convert to drift form
    drift_preds = []
    for t in range(n):
        if t == 0:
            drift_preds.append(ensemble_preds[t])
        else:
            cf_baseline = target_rounds[t - 1].contribution
            vcms_drift = ensemble_preds[t] - ensemble_preds[t - 1]
            pred = cf_baseline + vcms_drift
            pred = max(0, min(MAX_CONTRIB_VAL, round(pred)))
            drift_preds.append(pred)

    return drift_preds


# ================================================================
# PARAMETERIZED DRIFT PREDICTOR
# ================================================================

def vcms_predict_drift_parameterized(target_rounds, library, exclude_sids,
                                      recency_decay=0.7, delta_blend=0.5):
    """
    Hybrid CF + VCMS drift predictor with parameterized elimination.

    pred_t = C_{t-1} + drift_t
    where drift_t is the ensemble's predicted change from round t-1 to t.

    Parameters:
      recency_decay: exponential decay weight for recency (1.0 = uniform)
      delta_blend:   weight of delta score vs level score (0.0 = level only)
    """
    candidates = {}
    for sid, rec in library.items():
        if sid in exclude_sids:
            continue
        candidates[sid] = {
            'params': VCMSParams(**rec['v3_params']),
        }

    if not candidates:
        return [10] * min(N_ROUNDS, len(target_rounds))

    survivors = list(candidates.keys())
    ensemble_preds = []
    obs_c_list = []
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}

    n = min(N_ROUNDS, len(target_rounds))

    for t in range(n):
        # Run v3 for all survivors on target's environment
        cand_preds_c = {}
        if t == 0:
            for sid in survivors:
                result = run_vcms_agent(
                    candidates[sid]['params'], target_rounds[:1])
                cand_preds_c[sid] = result['pred_contrib'][0]
                cand_pred_history[sid].append(result['pred_contrib'][0])
        else:
            for sid in survivors:
                result = run_vcms_agent(
                    candidates[sid]['params'], target_rounds[:t + 1])
                cand_preds_c[sid] = result['pred_contrib'][t]
                cand_pred_history[sid] = list(result['pred_contrib'][:t + 1])

        # Weighted prediction
        if distances:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001)
                       for sid in survivors}
        else:
            weights = {sid: 1.0 for sid in survivors}
        total_w = sum(weights.values())

        pc = sum((weights[sid] / total_w) * cand_preds_c[sid]
                 for sid in survivors)
        ensemble_preds.append(int(round(pc)))

        # Observe actual
        obs_c_list.append(target_rounds[t].contribution)

        # Enhanced elimination: recency-weighted + delta-aware
        new_distances = {}
        for sid in survivors:
            hist = cand_pred_history[sid]
            nc = min(t + 1, len(hist))
            if nc == 0:
                new_distances[sid] = 0.0
                continue

            # Level component: recency-weighted MSE
            total_w_level = 0.0
            weighted_sse_level = 0.0
            for i in range(nc):
                w = recency_decay ** (nc - 1 - i)
                err = (obs_c_list[i] - hist[i]) / MAX_CONTRIB_VAL
                weighted_sse_level += w * err ** 2
                total_w_level += w
            level_dist = math.sqrt(weighted_sse_level / total_w_level)

            # Delta component: recency-weighted MSE on trajectory shape
            if nc >= 2:
                total_w_delta = 0.0
                weighted_sse_delta = 0.0
                for i in range(1, nc):
                    w = recency_decay ** (nc - 1 - i)
                    actual_delta = (obs_c_list[i] - obs_c_list[i - 1]) / MAX_CONTRIB_VAL
                    pred_delta = (hist[i] - hist[i - 1]) / MAX_CONTRIB_VAL
                    weighted_sse_delta += w * (actual_delta - pred_delta) ** 2
                    total_w_delta += w
                delta_dist = math.sqrt(weighted_sse_delta / total_w_delta)
            else:
                delta_dist = level_dist

            new_distances[sid] = (1 - delta_blend) * level_dist + delta_blend * delta_dist

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

    # Convert ensemble level predictions to CF + drift
    drift_preds = []
    for t in range(n):
        if t == 0:
            drift_preds.append(ensemble_preds[t])
        else:
            cf_baseline = target_rounds[t - 1].contribution
            vcms_drift = ensemble_preds[t] - ensemble_preds[t - 1]
            pred = cf_baseline + vcms_drift
            pred = max(0, min(MAX_CONTRIB_VAL, round(pred)))
            drift_preds.append(pred)

    return drift_preds


# ================================================================
# SCORING
# ================================================================

def rmse_window(pred, actual, k):
    """RMSE on rounds k+1..10 (0-indexed: rounds k..9), normalized by MAX_CONTRIB."""
    errors = []
    for t in range(k, min(len(pred), len(actual))):
        errors.append(((pred[t] - actual[t]) / MAX_CONTRIB_VAL) ** 2)
    return math.sqrt(np.mean(errors)) if errors else float('inf')


# ================================================================
# MAIN ABLATION
# ================================================================

def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 72)
    print("  ABLATION STUDY: Recency Weighting vs Delta-Aware Scoring")
    print("  in the VCMS Drift Predictor")
    print("=" * 72)

    # Load data
    print("\nLoading data...")
    library, all_rounds, city_map = load_all_data()
    total_subjects = len(all_rounds)
    print(f"  Total subjects with library entries: {total_subjects}")

    # Stratified subsample
    print(f"\nStratified subsample of ~{SUBSAMPLE_N} subjects...")
    subsample_sids = stratified_subsample(library, all_rounds, SUBSAMPLE_N, RANDOM_SEED)
    print(f"  Selected {len(subsample_sids)} subjects")

    # Show profile distribution in subsample
    profile_counts = Counter(library[sid]['behavioral_profile'] for sid in subsample_sids)
    for profile, count in sorted(profile_counts.items(), key=lambda x: -x[1]):
        total_profile = sum(1 for s in all_rounds if library[s]['behavioral_profile'] == profile)
        print(f"    {profile:<25s}: {count:>2d} / {total_profile} ({count/len(subsample_sids)*100:.0f}%)")

    # Run ablation
    print(f"\nRunning LOO cross-validation on {len(subsample_sids)} subjects...")
    print(f"  K values: {K_VALUES}")
    print(f"  Configurations: {list(ABLATION_CONFIGS.keys())}")
    print(f"  References: carry_forward, vcms_ensemble")
    print()

    scores = {k: {m: [] for m in ALL_METHODS} for k in K_VALUES}

    t0 = time.time()
    n = len(subsample_sids)

    for idx, sid in enumerate(subsample_sids):
        exclude = {sid}
        rounds = all_rounds[sid]
        actual_c = [r.contribution for r in rounds]

        preds = {}

        # Ablation configurations
        for config_name, config in ABLATION_CONFIGS.items():
            preds[config_name] = vcms_predict_drift_parameterized(
                rounds, library, exclude,
                recency_decay=config['RECENCY_DECAY'],
                delta_blend=config['DELTA_BLEND'],
            )

        # Reference: carry forward
        preds['carry_forward'] = carry_forward_predict(rounds)

        # Reference: base VCMS ensemble (converted to drift form)
        preds['vcms_ensemble'] = vcms_predict_as_drift(rounds, library, exclude)

        # Score
        for k in K_VALUES:
            for m in ALL_METHODS:
                scores[k][m].append(rmse_window(preds[m], actual_c, k))

        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (n - idx - 1)
        if (idx + 1) % 5 == 0 or idx == 0 or idx == n - 1:
            print(f"  [{idx+1:>3d}/{n}] {sid:<20s} "
                  f"({elapsed:>5.0f}s elapsed, ~{eta:>5.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nCompleted in {total_time:.1f}s ({total_time/n:.1f}s per subject)")

    # ================================================================
    # RESULTS TABLE
    # ================================================================

    # Compute aggregates
    agg = {}
    for k in K_VALUES:
        agg[k] = {}
        for m in ALL_METHODS:
            vals = scores[k][m]
            agg[k][m] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'median': float(np.median(vals)),
            }

    # Print main results
    print("\n" + "=" * 72)
    print("  MEAN RMSE BY CONFIGURATION AND K (normalized by MAX_CONTRIB=20)")
    print("=" * 72)

    col_w = 12
    header = f"  {'Configuration':<20s}"
    for k in K_VALUES:
        header += f"  {'k='+str(k):>{col_w}s}"
    print(header)
    print("  " + "-" * (20 + (col_w + 2) * len(K_VALUES)))

    # Print ablation configs first, then references
    display_order = list(ABLATION_CONFIGS.keys()) + ['carry_forward', 'vcms_ensemble']
    labels = {
        'base_drift':     'Base drift',
        'recency_only':   'Recency only',
        'delta_only':     'Delta only',
        'both':           'Both (actual)',
        'carry_forward':  'Carry forward',
        'vcms_ensemble':  'VCMS ensemble',
    }

    for m in display_order:
        label = labels.get(m, m)
        row = f"  {label:<20s}"
        for k in K_VALUES:
            val = agg[k][m]['mean']
            row += f"  {val:>{col_w}.4f}"
        print(row)

    # Print std dev table
    print("\n" + "=" * 72)
    print("  STD DEV OF RMSE")
    print("=" * 72)

    header = f"  {'Configuration':<20s}"
    for k in K_VALUES:
        header += f"  {'k='+str(k):>{col_w}s}"
    print(header)
    print("  " + "-" * (20 + (col_w + 2) * len(K_VALUES)))

    for m in display_order:
        label = labels.get(m, m)
        row = f"  {label:<20s}"
        for k in K_VALUES:
            val = agg[k][m]['std']
            row += f"  {val:>{col_w}.4f}"
        print(row)

    # Delta table: improvement over carry_forward
    print("\n" + "=" * 72)
    print("  DELTA vs CARRY FORWARD (negative = better than CF)")
    print("=" * 72)

    header = f"  {'Configuration':<20s}"
    for k in K_VALUES:
        header += f"  {'k='+str(k):>{col_w}s}"
    print(header)
    print("  " + "-" * (20 + (col_w + 2) * len(K_VALUES)))

    for m in display_order:
        if m == 'carry_forward':
            continue
        label = labels.get(m, m)
        row = f"  {label:<20s}"
        for k in K_VALUES:
            delta = agg[k][m]['mean'] - agg[k]['carry_forward']['mean']
            sign = '+' if delta >= 0 else ''
            row += f"  {sign}{delta:>{col_w-1}.4f}"
        print(row)

    # Component contribution analysis
    print("\n" + "=" * 72)
    print("  COMPONENT CONTRIBUTION ANALYSIS")
    print("  (marginal effect of each component)")
    print("=" * 72)

    header = f"  {'Component':<28s}"
    for k in K_VALUES:
        header += f"  {'k='+str(k):>{col_w}s}"
    print(header)
    print("  " + "-" * (28 + (col_w + 2) * len(K_VALUES)))

    # Recency effect = base_drift - recency_only
    # (negative means recency_only is better, i.e., recency helps)
    row = f"  {'Recency (R-only - Base)':<28s}"
    for k in K_VALUES:
        val = agg[k]['recency_only']['mean'] - agg[k]['base_drift']['mean']
        sign = '+' if val >= 0 else ''
        row += f"  {sign}{val:>{col_w-1}.4f}"
    print(row)

    # Delta effect = base_drift - delta_only
    row = f"  {'Delta (D-only - Base)':<28s}"
    for k in K_VALUES:
        val = agg[k]['delta_only']['mean'] - agg[k]['base_drift']['mean']
        sign = '+' if val >= 0 else ''
        row += f"  {sign}{val:>{col_w-1}.4f}"
    print(row)

    # Interaction = both - (recency_only + delta_only - base_drift)
    # If additive: both = recency_only + delta_only - base_drift
    # Interaction = both - recency_only - delta_only + base_drift
    row = f"  {'Interaction':<28s}"
    for k in K_VALUES:
        additive_pred = (agg[k]['recency_only']['mean']
                         + agg[k]['delta_only']['mean']
                         - agg[k]['base_drift']['mean'])
        interaction = agg[k]['both']['mean'] - additive_pred
        sign = '+' if interaction >= 0 else ''
        row += f"  {sign}{interaction:>{col_w-1}.4f}"
    print(row)

    # Combined effect = both - base_drift
    row = f"  {'Combined (Both - Base)':<28s}"
    for k in K_VALUES:
        val = agg[k]['both']['mean'] - agg[k]['base_drift']['mean']
        sign = '+' if val >= 0 else ''
        row += f"  {sign}{val:>{col_w-1}.4f}"
    print(row)

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    best_config_by_k = {}
    ablation_names = list(ABLATION_CONFIGS.keys())
    for k in K_VALUES:
        best = min(ablation_names, key=lambda m: agg[k][m]['mean'])
        best_config_by_k[k] = best
        print(f"  k={k}: Best ablation config = {labels[best]:<20s} "
              f"(RMSE={agg[k][best]['mean']:.4f})")

    print()
    print(f"  Subsample size: {len(subsample_sids)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Random seed: {RANDOM_SEED}")


if __name__ == '__main__':
    main()
