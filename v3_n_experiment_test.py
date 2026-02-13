#!/usr/bin/env python3
"""
N-experiment generalization test for VCMS v3.

Tests whether dynamics fitted on P-experiment subjects (with punishment)
predict contribution trajectories in the N-experiment (no punishment).

This is a true out-of-distribution test:
  - Library fitted on P-experiment (punishment present)
  - Test subjects from N-experiment (no punishment channel)
  - Punishment_sent = 0, punishment_received = 0 for all rounds
  - V/C/M/S/B contribution dynamics still operate:
    * Strain still accumulates from cooperation gaps
    * Budget still responds to exploitation (no pun drain though)
    * But: no discharge through P, no punishment-received B drain

If the model generalizes, it means the contribution-side architecture
captures real psychological structure, not P-experiment artifacts.
"""

import json
import math
import os
import sys
import time
import numpy as np
from collections import defaultdict, Counter

from pgg_p_loader import load_p_experiment, PRoundData
from pgg_vcms_agent_v3 import (
    VCMSParams, PARAM_NAMES, run_vcms_agent, MAX_CONTRIB,
)

K_VALUES = [1, 2, 3, 5]
N_ROUNDS = 10
METHODS = ['vcms_drift', 'vcms', 'carry_forward', 'profile_mean']


# ================================================================
# DATA LOADING
# ================================================================

N_EXPERIMENT_CSVS = [
    ('Athens',    'HerrmannThoeniGaechterDATA_ATHENS_N-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ('Boston',    'HerrmannThoeniGaechterDATA_BOSTON_N-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ('Chengdu',   'HerrmannThoeniGaechterDATA_CHENGDU_N-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ('Istanbul',  'HerrmannThoeniGaechterDATA_ISTANBUL_N-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ('Melbourne', 'HerrmannThoeniGaechterDATA_MELBOURNE_N-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ('StGallen',  'HerrmannThoeniGaechterDATA_STGALLEN_N-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ('Samara',    'HerrmannThoeniGaechterDATA_SAMARA_N-EXPERIMENT_TRUNCATED_SESSION1.csv'),
    ('Samara',    'HerrmannThoeniGaechterDATA_SAMARA_N-EXPERIMENT_TRUNCATED_SESSION2.csv'),
    ('Samara',    'HerrmannThoeniGaechterDATA_SAMARA_N-EXPERIMENT_TRUNCATED_SESSION3.csv'),
    ('Samara',    'HerrmannThoeniGaechterDATA_SAMARA_N-EXPERIMENT_TRUNCATED_SESSION4.csv'),
    ('Samara',    'HerrmannThoeniGaechterDATA_SAMARA_N-EXPERIMENT_TRUNCATED_SESSION5.csv'),
]


def load_n_experiment_data():
    """Load all N-experiment subjects. Return {sid: rounds}, city_map."""
    all_rounds = {}
    city_map = {}

    for city, csv_path in N_EXPERIMENT_CSVS:
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        data = load_p_experiment(csv_path)
        for sid, rounds in data.items():
            all_rounds[sid] = rounds
            city_map[sid] = city

    return all_rounds, city_map


def load_library():
    """Load the P-experiment fitted library."""
    with open('v3_library_fitted.json') as f:
        return json.load(f)


# ================================================================
# BASELINES
# ================================================================

def carry_forward_predict(rounds):
    """C(t) = C(t-1). Round 1 = actual."""
    pred = []
    for t in range(min(N_ROUNDS, len(rounds))):
        if t == 0:
            pred.append(rounds[0].contribution)
        else:
            pred.append(rounds[t - 1].contribution)
    return pred


def profile_mean_predict(rounds, library):
    """
    Predict C as mean trajectory across all library subjects.
    (No profile matching — N-experiment subjects don't have P-experiment profiles.)
    """
    pred = []
    for t in range(N_ROUNDS):
        cs = [rec['contribution_trajectory'][t]
              for rec in library.values()
              if t < len(rec['contribution_trajectory'])]
        pred.append(round(np.mean(cs)) if cs else 10)
    return pred


# ================================================================
# VCMS PREDICTORS (adapted from v3_cross_validation.py)
# ================================================================

def vcms_predict(target_rounds, library):
    """VCMS ensemble predictor — full library, no exclusions."""
    candidates = {}
    for sid, rec in library.items():
        candidates[sid] = {
            'params': VCMSParams(**rec['v3_params']),
        }

    survivors = list(candidates.keys())
    pred_c_list = []
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

        pc = sum((weights[sid] / total_w) * cand_preds_c[sid]
                 for sid in survivors)
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


def vcms_predict_drift(target_rounds, library):
    """CF + VCMS drift with recency + delta scoring."""
    RECENCY_DECAY = 0.7
    DELTA_BLEND = 0.5

    candidates = {}
    for sid, rec in library.items():
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

        if distances:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001)
                       for sid in survivors}
        else:
            weights = {sid: 1.0 for sid in survivors}
        total_w = sum(weights.values())

        pc = sum((weights[sid] / total_w) * cand_preds_c[sid]
                 for sid in survivors)
        ensemble_preds.append(int(round(pc)))

        obs_c_list.append(target_rounds[t].contribution)

        # Recency + delta elimination
        new_distances = {}
        for sid in survivors:
            hist = cand_pred_history[sid]
            nc = min(t + 1, len(hist))
            if nc == 0:
                new_distances[sid] = 0.0
                continue

            total_w_level = 0.0
            weighted_sse_level = 0.0
            for i in range(nc):
                w = RECENCY_DECAY ** (nc - 1 - i)
                err = (obs_c_list[i] - hist[i]) / MAX_CONTRIB
                weighted_sse_level += w * err ** 2
                total_w_level += w
            level_dist = math.sqrt(weighted_sse_level / total_w_level)

            if nc >= 2:
                total_w_delta = 0.0
                weighted_sse_delta = 0.0
                for i in range(1, nc):
                    w = RECENCY_DECAY ** (nc - 1 - i)
                    actual_delta = (obs_c_list[i] - obs_c_list[i - 1]) / MAX_CONTRIB
                    pred_delta = (hist[i] - hist[i - 1]) / MAX_CONTRIB
                    weighted_sse_delta += w * (actual_delta - pred_delta) ** 2
                    total_w_delta += w
                delta_dist = math.sqrt(weighted_sse_delta / total_w_delta)
            else:
                delta_dist = level_dist

            new_distances[sid] = (1 - DELTA_BLEND) * level_dist + DELTA_BLEND * delta_dist

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

    # Convert to drift
    drift_preds = []
    for t in range(n):
        if t == 0:
            drift_preds.append(ensemble_preds[t])
        else:
            cf_baseline = target_rounds[t - 1].contribution
            vcms_drift = ensemble_preds[t] - ensemble_preds[t - 1]
            pred = cf_baseline + vcms_drift
            pred = max(0, min(MAX_CONTRIB, round(pred)))
            drift_preds.append(pred)

    return drift_preds


# ================================================================
# SCORING
# ================================================================

def rmse_window(pred, actual, k):
    """RMSE on rounds k+1..10, normalized by MAX_CONTRIB."""
    errors = []
    for t in range(k, min(len(pred), len(actual))):
        errors.append(((pred[t] - actual[t]) / MAX_CONTRIB) ** 2)
    return math.sqrt(np.mean(errors)) if errors else float('inf')


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  N-EXPERIMENT GENERALIZATION TEST")
    print("  P-experiment-fitted library → N-experiment predictions")
    print("=" * 72)

    # Load
    print("\nLoading P-experiment library...")
    library = load_library()
    print(f"  {len(library)} library subjects (P-experiment)")

    print("\nLoading N-experiment data...")
    n_rounds, city_map = load_n_experiment_data()
    print(f"  {len(n_rounds)} N-experiment subjects")

    cities = sorted(set(city_map.values()))
    for city in cities:
        n = sum(1 for c in city_map.values() if c == city)
        print(f"    {city}: {n} subjects")

    # Quick characterization of N-experiment behavior
    all_c = []
    for sid, rounds in n_rounds.items():
        for r in rounds:
            all_c.append(r.contribution)
    print(f"\n  N-experiment contributions: mean={np.mean(all_c):.1f}, "
          f"median={np.median(all_c):.0f}, std={np.std(all_c):.1f}")

    # Run predictions
    print(f"\nRunning predictions on {len(n_rounds)} subjects...")
    sids = sorted(n_rounds.keys())
    scores = {k: {m: [] for m in METHODS} for k in K_VALUES}
    per_subject = {}

    t0 = time.time()
    n = len(sids)

    for idx, sid in enumerate(sids):
        rounds = n_rounds[sid]
        actual_c = [r.contribution for r in rounds]

        preds = {}
        preds['vcms'] = vcms_predict(rounds, library)
        preds['vcms_drift'] = vcms_predict_drift(rounds, library)
        preds['carry_forward'] = carry_forward_predict(rounds)
        preds['profile_mean'] = profile_mean_predict(rounds, library)

        per_subject[sid] = {
            'actual_c': actual_c,
            'city': city_map[sid],
            'preds': preds,
        }

        for k in K_VALUES:
            for m in METHODS:
                scores[k][m].append(rmse_window(preds[m], actual_c, k))

        if (idx + 1) % 20 == 0 or idx == 0 or idx == n - 1:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n - idx - 1)
            print(f"  [{idx+1:>3d}/{n}] {sid:<10s} "
                  f"({elapsed:>5.0f}s elapsed, ~{eta:>5.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nCompleted in {total_time:.1f}s")

    # ================================================================
    # RESULTS
    # ================================================================

    # Aggregate
    agg = {}
    for k in K_VALUES:
        agg[k] = {}
        for m in METHODS:
            vals = scores[k][m]
            agg[k][m] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
            }

    # Main table
    print(f"\n{'=' * 72}")
    print(f"  N-EXPERIMENT RESULTS (mean RMSE, normalized)")
    print(f"  Library: {len(library)} P-experiment subjects")
    print(f"  Test:    {len(n_rounds)} N-experiment subjects")
    print(f"{'=' * 72}")

    header = f"  {'Method':<20s}"
    for k in K_VALUES:
        header += f"  k={k:<6d}"
    print(header)
    print("  " + "-" * (20 + 10 * len(K_VALUES)))

    for m in METHODS:
        row = f"  {m:<20s}"
        for k in K_VALUES:
            row += f"  {agg[k][m]['mean']:.4f}  "
        print(row)

    # Delta vs carry_forward
    print()
    ref = 'carry_forward'
    for m in METHODS:
        if m == ref:
            continue
        row = f"  {'Δ vs CF: ' + m:<20s}"
        for k in K_VALUES:
            delta = agg[k][m]['mean'] - agg[k][ref]['mean']
            sign = '+' if delta >= 0 else ''
            row += f"  {sign}{delta:.4f}"
        print(row)

    # Per-city breakdown
    print(f"\n{'=' * 72}")
    print(f"  PER-CITY BREAKDOWN")
    print(f"{'=' * 72}")

    for city in cities:
        city_sids = [s for s in sids if city_map[s] == city]
        city_indices = [sids.index(s) for s in city_sids]

        print(f"\n  {city} (n={len(city_sids)})")
        header = f"    {'Method':<20s}"
        for k in [1, 3, 5]:
            header += f"  k={k:<6d}"
        print(header)

        for m in METHODS:
            row = f"    {m:<20s}"
            for k in [1, 3, 5]:
                vals = [scores[k][m][i] for i in city_indices]
                row += f"  {np.mean(vals):.4f}  "
            print(row)

    # N-experiment trajectory patterns
    print(f"\n{'=' * 72}")
    print(f"  N-EXPERIMENT TRAJECTORY CHARACTERIZATION")
    print(f"{'=' * 72}")

    slopes = []
    for sid in sids:
        ct = np.array([r.contribution for r in n_rounds[sid]])
        slope = np.polyfit(np.arange(len(ct)), ct, 1)[0]
        slopes.append(slope)

    slopes = np.array(slopes)
    print(f"\n  Contribution slopes: mean={np.mean(slopes):.2f}, "
          f"median={np.median(slopes):.2f}")
    print(f"  Declining (slope < -0.3):  {(slopes < -0.3).sum():>3d} "
          f"({100*(slopes < -0.3).mean():.0f}%)")
    print(f"  Stable (|slope| ≤ 0.3):   {(np.abs(slopes) <= 0.3).sum():>3d} "
          f"({100*(np.abs(slopes) <= 0.3).mean():.0f}%)")
    print(f"  Rising (slope > 0.3):      {(slopes > 0.3).sum():>3d} "
          f"({100*(slopes > 0.3).mean():.0f}%)")

    # The key question: does N-experiment show more decay than P-experiment?
    p_slopes = []
    for sid, rec in library.items():
        ct = np.array(rec['contribution_trajectory'])
        p_slopes.append(np.polyfit(np.arange(len(ct)), ct, 1)[0])
    p_slopes = np.array(p_slopes)

    print(f"\n  Comparison:")
    print(f"    P-experiment mean slope: {np.mean(p_slopes):+.2f}")
    print(f"    N-experiment mean slope: {np.mean(slopes):+.2f}")
    print(f"    N-experiment declines {'more' if np.mean(slopes) < np.mean(p_slopes) else 'less'} "
          f"than P-experiment")

    # Save
    out = {
        'aggregate': {str(k): {m: agg[k][m] for m in METHODS} for k in K_VALUES},
        'n_subjects': len(n_rounds),
        'library_size': len(library),
    }
    with open('v3_n_experiment_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to v3_n_experiment_results.json")


if __name__ == '__main__':
    main()
