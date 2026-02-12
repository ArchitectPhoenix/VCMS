#!/usr/bin/env python3
"""
Cross-validation evaluation of VCMS v3 sequential predictor.

Evaluation matrix:
  Methods: VCMS, Profile-mean, Carry-forward, Group-mean-regression, EWA
  CV modes: LOO (leave-one-out), LOCO (leave-one-city-out)
  Windows: k=1,2,3,5 (score on rounds k+1..10 only)

The convergence curve across k values shows how quickly the model
identifies who someone is and translates that to predictive accuracy.
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
from scipy.optimize import minimize as scipy_minimize

K_VALUES = [1, 2, 3, 5]
N_ROUNDS = 10
METHODS = ['vcms_drift', 'vcms', 'vcms_online', 'profile_mean', 'carry_forward', 'group_mean', 'ewa']
# vcms_drift: hybrid CF + VCMS — carry-forward baseline plus v3 predicted drift.
#   Guaranteed >= carry-forward when drift is zero; better when v3 anticipates transitions.
# vcms_online is included for comparison but underperforms the ensemble (vcms).
# Online parameter estimation loses the diversity of the ensemble — collapsing
# to a single parameterization is less robust than weighted averaging across
# multiple plausible candidate dynamics.

# PGG parameters (Herrmann et al. design)
ENDOWMENT = 20
MPCR = 0.4
GROUP_SIZE = 4


# ================================================================
# DATA LOADING
# ================================================================

def load_all_data():
    """Load all city CSVs and the fitted library. Return library, rounds, city_map."""
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
# BASELINES
# ================================================================

def profile_mean_predict(target_profile, training_lib):
    """Predict C as mean trajectory of same-profile subjects in training set."""
    same = [v for v in training_lib.values()
            if v['behavioral_profile'] == target_profile]
    if not same:
        same = list(training_lib.values())

    pred = []
    for t in range(N_ROUNDS):
        cs = [s['contribution_trajectory'][t] for s in same
              if t < len(s['contribution_trajectory'])]
        pred.append(round(np.mean(cs)) if cs else 10)
    return pred


def carry_forward_predict(rounds):
    """Predict C(t) = C(t-1). Round 1: use round 1 actual (no prediction)."""
    pred = []
    for t in range(min(N_ROUNDS, len(rounds))):
        if t == 0:
            pred.append(rounds[0].contribution)
        else:
            pred.append(rounds[t - 1].contribution)
    return pred


def group_mean_regression_predict(rounds, training_lib):
    """Predict C(t) = a + b * others_mean(t-1), regression fitted on training set."""
    xs, ys = [], []
    for rec in training_lib.values():
        ct = rec['contribution_trajectory']
        om = rec['others_mean_trajectory']
        for t in range(1, min(len(ct), len(om))):
            xs.append(om[t - 1])
            ys.append(ct[t])

    if len(xs) > 1:
        xs_arr, ys_arr = np.array(xs), np.array(ys)
        xm, ym = xs_arr.mean(), ys_arr.mean()
        b = np.sum((xs_arr - xm) * (ys_arr - ym)) / (np.sum((xs_arr - xm) ** 2) + 1e-10)
        a = ym - b * xm
    else:
        a, b = 10.0, 0.0

    pred = []
    for t in range(min(N_ROUNDS, len(rounds))):
        if t == 0:
            pred.append(int(round(np.clip(a + b * 10.0, 0, MAX_CONTRIB))))
        else:
            pred.append(int(round(np.clip(
                a + b * rounds[t - 1].others_mean, 0, MAX_CONTRIB))))
    return pred


def ewa_predict(rounds, phi=0.5, delta=0.5, rho=0.5, lam=0.2):
    """
    EWA (Experience-Weighted Attraction) for PGG contributions.

    Camerer & Ho (1999). Actions = contribution levels 0..20.
    Payoff: pi(c, others_mean) = (E - c) + MPCR * (c + (G-1) * others_mean)
    """
    n_actions = MAX_CONTRIB + 1
    A = np.zeros(n_actions)
    N_exp = 1.0

    pred = []
    for t in range(min(N_ROUNDS, len(rounds))):
        # Predict from current attractions via softmax
        shifted = lam * (A - A.max())
        exp_a = np.exp(shifted)
        probs = exp_a / exp_a.sum()
        expected_c = np.sum(np.arange(n_actions) * probs)
        pred.append(int(round(np.clip(expected_c, 0, MAX_CONTRIB))))

        # Observe and update
        actual_c = rounds[t].contribution
        others_mean = rounds[t].others_mean
        total_others = others_mean * (GROUP_SIZE - 1)

        for j in range(n_actions):
            payoff_j = (ENDOWMENT - j) + MPCR * (j + total_others)
            indicator = 1.0 if j == actual_c else 0.0
            reinforcement = (delta + (1.0 - delta) * indicator) * payoff_j
            A[j] = (phi * N_exp * A[j] + reinforcement) / (rho * N_exp + 1.0)

        N_exp = rho * N_exp + 1.0

    return pred


# ================================================================
# VCMS PREDICTOR (with exclusion)
# ================================================================

def vcms_predict(target_rounds, library, exclude_sids):
    """
    Run VCMS combined predictor, excluding specified subjects from pool.

    For each round:
      1. Run v3 for all survivors on target's environment
      2. Predict from inverse-distance-weighted v3 outputs
      3. Observe actual
      4. Eliminate on v3 PREDICTION accuracy (not library trajectory distance)

    Key insight: elimination must score how well each candidate's v3 dynamics
    predict the TARGET's behavior in the TARGET's environment, not how similar
    the library trajectory is. The library trajectory was in a different group.
    """
    # Build candidate pool
    candidates = {}
    for sid, rec in library.items():
        if sid in exclude_sids:
            continue
        candidates[sid] = {
            'params': VCMSParams(**rec['v3_params']),
        }

    survivors = list(candidates.keys())
    pred_c_list = []
    obs_c_list = []
    # Track each candidate's v3 predictions on the target's environment
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}

    n = min(N_ROUNDS, len(target_rounds))

    for t in range(n):
        # Run v3 for all survivors on target's environment
        cand_preds_c = {}
        if t == 0:
            # R1: v3 can't predict without prior rounds; use R1 from v3 init
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
                # Store full prediction history for this candidate
                cand_pred_history[sid] = list(result['pred_contrib'][:t + 1])

        # Weighted prediction
        if not distances:
            # First round or no distances yet: equal weights
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001)
                       for sid in survivors}
        total_w = sum(weights.values())

        pc = 0.0
        for sid in survivors:
            pc += (weights[sid] / total_w) * cand_preds_c[sid]
        pred_c_list.append(int(round(pc)))

        # Observe actual
        obs_c_list.append(target_rounds[t].contribution)

        # Eliminate on v3 PREDICTION accuracy (how well did this candidate's
        # dynamics predict the target's actual behavior in the target's env?)
        new_distances = {}
        for sid in survivors:
            hist = cand_pred_history[sid]
            nc = min(t + 1, len(hist))
            if nc == 0:
                new_distances[sid] = 0.0
                continue
            # MSE between candidate's v3 predictions and target's actual C
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
# ONLINE PARAMETER ESTIMATION
# ================================================================

# Pre-compute bounds and ranges for regularization
_BOUNDS = [PARAM_BOUNDS[n] for n in PARAM_NAMES]
_RANGES = np.array([b[1] - b[0] for b in _BOUNDS])
_LO = np.array([b[0] for b in _BOUNDS])
_HI = np.array([b[1] for b in _BOUNDS])


def _online_objective(x, rounds, prior_vec, reg_lambda):
    """v3 RMSE + L2 regularization toward library prior."""
    rmse = vcms_objective(x, rounds, PARAM_NAMES, None)
    reg = np.sum(((x - prior_vec) / _RANGES) ** 2)
    return rmse + reg_lambda * reg


def quick_online_fit(observed_rounds, prior_params_dict):
    """
    Quick parameter fit on observed rounds, regularized toward library prior.

    Uses L-BFGS-B starting from the prior. Regularization strength scales
    inversely with observation count: strong prior when few rounds, weaker
    as data accumulates.
    """
    prior_vec = np.array([prior_params_dict[n] for n in PARAM_NAMES])
    n_obs = len(observed_rounds)
    reg_lambda = 1.0 / max(1, n_obs)

    result = scipy_minimize(
        _online_objective,
        x0=prior_vec,
        args=(observed_rounds, prior_vec, reg_lambda),
        method='L-BFGS-B',
        bounds=_BOUNDS,
        options={'maxiter': 30, 'ftol': 1e-5},
    )

    fitted_vec = np.clip(result.x, _LO, _HI)
    return {n: float(fitted_vec[i]) for i, n in enumerate(PARAM_NAMES)}


def vcms_predict_online(target_rounds, library, exclude_sids):
    """
    Hybrid VCMS predictor: ensemble for rounds 1-2, online fitting from round 3+.

    Rounds 1-2: Use ensemble predictor (Fix 1) — not enough data to fit.
    Rounds 3+:  Find best library match so far, use as prior for quick online
                parameter estimation on observed rounds, predict from fitted params.
    """
    # Build candidate pool
    candidates = {}
    for sid, rec in library.items():
        if sid in exclude_sids:
            continue
        candidates[sid] = {
            'params': VCMSParams(**rec['v3_params']),
            'params_dict': rec['v3_params'],
        }

    survivors = list(candidates.keys())
    pred_c_list = []
    obs_c_list = []
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}
    online_params = None  # current online-fitted params

    n = min(N_ROUNDS, len(target_rounds))
    ONLINE_START = 3  # start online fitting at round 3 (need 2 observations)

    for t in range(n):
        # === PREDICTION ===
        if t < ONLINE_START:
            # Ensemble prediction (same as Fix 1)
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

        else:
            # Online fitting: use best match as prior, fit on observed data
            if distances:
                best_sid = min(distances, key=distances.get)
            else:
                best_sid = survivors[0] if survivors else list(candidates.keys())[0]

            prior = candidates[best_sid]['params_dict']
            online_params = quick_online_fit(target_rounds[:t], prior)

            # Predict round t from online-fitted params
            fitted = VCMSParams(**online_params)
            result = run_vcms_agent(fitted, target_rounds[:t + 1])
            pc = result['pred_contrib'][t]
            pred_c_list.append(pc)

        # === OBSERVE ===
        obs_c_list.append(target_rounds[t].contribution)

        # === ELIMINATE (keep running for prior selection) ===
        if t < ONLINE_START:
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

            new_surv = [s for s in survivors
                        if new_distances.get(s, 999) <= thresh]
            if not new_surv and new_distances:
                new_surv = [min(new_distances, key=new_distances.get)]

            survivors = new_surv
            distances = {s: new_distances[s] for s in survivors}

    return pred_c_list


def vcms_predict_drift(target_rounds, library, exclude_sids):
    """
    Hybrid CF + VCMS drift predictor.

    pred_t = C_{t-1} + drift_t

    where drift_t is the VCMS ensemble's predicted change from round t-1 to t.
    Carry-forward is the implicit baseline (drift_t = 0 recovers CF exactly).

    This focuses VCMS modeling power on predicting *transitions* — when and how
    behavior changes — rather than predicting levels that CF handles for free.

    For R1 (no prior observation): uses the ensemble prediction directly.
    """
    # Get the full ensemble predictions
    ensemble_preds = vcms_predict(target_rounds, library, exclude_sids)

    n = min(N_ROUNDS, len(target_rounds))
    drift_preds = []

    for t in range(n):
        if t == 0:
            # No prior observation — use ensemble directly
            drift_preds.append(ensemble_preds[t])
        else:
            # CF baseline: last actual observation
            cf_baseline = target_rounds[t - 1].contribution
            # VCMS drift: what does the ensemble predict will change?
            vcms_drift = ensemble_preds[t] - ensemble_preds[t - 1]
            # Apply drift to CF baseline
            pred = cf_baseline + vcms_drift
            pred = max(0, min(MAX_CONTRIB, round(pred)))
            drift_preds.append(pred)

    return drift_preds


# ================================================================
# SCORING
# ================================================================

def rmse_window(pred, actual, k):
    """RMSE on rounds k+1..10 (0-indexed: rounds k..9), normalized by MAX_CONTRIB."""
    errors = []
    for t in range(k, min(len(pred), len(actual))):
        errors.append(((pred[t] - actual[t]) / MAX_CONTRIB) ** 2)
    return math.sqrt(np.mean(errors)) if errors else float('inf')


# ================================================================
# MAIN EVALUATION LOOP
# ================================================================

def run_cv(library, all_rounds, city_map, mode='loo', verbose=True):
    """
    Run cross-validation in LOO or LOCO mode.

    Returns:
      agg: {k: {method: {'mean', 'median', 'std'}}}
      per_subject: {sid: {method: pred_c_list}}
    """
    cities = sorted(set(city_map.values()))
    sids = sorted(all_rounds.keys())
    n = len(sids)

    # Build exclusion sets
    if mode == 'loo':
        exclusions = {sid: {sid} for sid in sids}
    else:
        exclusions = {}
        for sid in sids:
            city = city_map[sid]
            exclusions[sid] = {s for s, c in city_map.items() if c == city}

    # Pre-fit group-mean regression per fold (cache by exclusion key)
    # For LOO the regression barely changes per fold; for LOCO it's per-city
    regression_cache = {}

    # Results storage
    scores = {k: {m: [] for m in METHODS} for k in K_VALUES}
    per_subject = {}

    t0 = time.time()

    for idx, sid in enumerate(sids):
        exclude = exclusions[sid]
        rounds = all_rounds[sid]
        actual_c = [r.contribution for r in rounds]
        profile = library[sid]['behavioral_profile']

        # Training library for this fold
        train_lib = {s: library[s] for s in library if s not in exclude}

        # --- Run all methods ---
        preds = {}

        # 1. VCMS (ensemble with Fix 1)
        preds['vcms'] = vcms_predict(rounds, library, exclude)

        # 1b. VCMS drift (hybrid CF + ensemble drift)
        preds['vcms_drift'] = vcms_predict_drift(rounds, library, exclude)

        # 1c. VCMS with online parameter estimation
        preds['vcms_online'] = vcms_predict_online(rounds, library, exclude)

        # 2. Profile-mean
        preds['profile_mean'] = profile_mean_predict(profile, train_lib)

        # 3. Carry-forward
        preds['carry_forward'] = carry_forward_predict(rounds)

        # 4. Group-mean regression
        preds['group_mean'] = group_mean_regression_predict(rounds, train_lib)

        # 5. EWA
        preds['ewa'] = ewa_predict(rounds)

        per_subject[sid] = {
            'actual_c': actual_c,
            'city': city_map[sid],
            'profile': profile,
            'preds': {m: preds[m] for m in METHODS},
        }

        # Score for each k
        for k in K_VALUES:
            for m in METHODS:
                scores[k][m].append(rmse_window(preds[m], actual_c, k))

        if verbose and ((idx + 1) % 20 == 0 or idx == 0 or idx == n - 1):
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n - idx - 1)
            print(f"  [{idx+1}/{n}] {sid} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

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

    return agg, per_subject, scores


def run_loco_by_city(library, all_rounds, city_map, verbose=True):
    """Run LOCO and also return per-city breakdowns."""
    cities = sorted(set(city_map.values()))

    # Run full LOCO
    agg, per_subject, scores = run_cv(
        library, all_rounds, city_map, mode='loco', verbose=verbose)

    # Per-city breakdown
    city_agg = {}
    for city in cities:
        city_sids = [s for s, c in city_map.items() if c == city]
        city_indices = [sorted(all_rounds.keys()).index(s) for s in city_sids]
        city_agg[city] = {}
        for k in K_VALUES:
            city_agg[city][k] = {}
            for m in METHODS:
                vals = [scores[k][m][i] for i in city_indices]
                city_agg[city][k][m] = {
                    'mean': float(np.mean(vals)),
                    'n': len(vals),
                }

    return agg, per_subject, city_agg


# ================================================================
# DISPLAY
# ================================================================

def print_matrix(title, agg):
    """Print a comparison matrix."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

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

    # Delta vs vcms_drift (primary method)
    ref = 'vcms_drift' if 'vcms_drift' in METHODS else 'vcms'
    print()
    print(f"  {'Δ vs ' + ref:<20s}")
    for m in METHODS:
        if m == ref:
            continue
        row = f"  {m:<20s}"
        for k in K_VALUES:
            delta = agg[k][m]['mean'] - agg[k][ref]['mean']
            sign = '+' if delta >= 0 else ''
            row += f"  {sign}{delta:.4f}"
        print(row)


def print_city_breakdown(city_agg):
    """Print per-city LOCO results."""
    ref = 'vcms_drift' if 'vcms_drift' in METHODS else 'vcms'
    print(f"\n{'=' * 70}")
    print(f"  LOCO BY CITY ({ref} vs carry_forward vs profile_mean)")
    print(f"{'=' * 70}")

    cities = sorted(city_agg.keys())
    header = f"  {'City':<10s} {'n':>3s}"
    for k in [1, 3, 5]:
        header += f"  {ref[:5]} k={k}  CF k={k}  PM k={k}"
    print(header)

    for city in cities:
        n = city_agg[city][K_VALUES[0]][ref]['n']
        row = f"  {city:<10s} {n:>3d}"
        for k in [1, 3, 5]:
            v = city_agg[city][k][ref]['mean']
            cf = city_agg[city][k]['carry_forward']['mean']
            pm = city_agg[city][k]['profile_mean']['mean']
            row += f"  {v:.4f} {cf:.4f} {pm:.4f}"
        print(row)


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("Loading data...")
    library, all_rounds, city_map = load_all_data()

    cities = sorted(set(city_map.values()))
    print(f"  {len(all_rounds)} subjects across {len(cities)} cities: "
          f"{', '.join(f'{c}({sum(1 for s,cc in city_map.items() if cc==c)})' for c in cities)}")

    # Parse args
    run_loo = '--loo' in sys.argv or '--all' in sys.argv or len(sys.argv) == 1
    run_loco = '--loco' in sys.argv or '--all' in sys.argv or len(sys.argv) == 1

    results = {}

    if run_loo:
        print(f"\n{'#' * 70}")
        print(f"  LOO CROSS-VALIDATION (n={len(all_rounds)})")
        print(f"{'#' * 70}")
        loo_agg, loo_per, _ = run_cv(
            library, all_rounds, city_map, mode='loo', verbose=True)
        print_matrix("LOO RESULTS (mean RMSE, normalized)", loo_agg)
        results['loo'] = loo_agg

    if run_loco:
        print(f"\n{'#' * 70}")
        print(f"  LOCO CROSS-VALIDATION (n={len(all_rounds)})")
        print(f"{'#' * 70}")
        loco_agg, loco_per, city_agg = run_loco_by_city(
            library, all_rounds, city_map, verbose=True)
        print_matrix("LOCO RESULTS (mean RMSE, normalized)", loco_agg)
        print_city_breakdown(city_agg)
        results['loco'] = loco_agg
        results['loco_by_city'] = {
            city: {str(k): {m: v for m, v in methods.items()}
                   for k, methods in k_data.items()}
            for city, k_data in city_agg.items()
        }

    # LOO vs LOCO gap
    if run_loo and run_loco:
        print(f"\n{'=' * 70}")
        print(f"  LOO vs LOCO GAP (LOCO - LOO)")
        print(f"{'=' * 70}")
        header = f"  {'Method':<20s}"
        for k in K_VALUES:
            header += f"  k={k:<6d}"
        print(header)
        for m in METHODS:
            row = f"  {m:<20s}"
            for k in K_VALUES:
                gap = loco_agg[k][m]['mean'] - loo_agg[k][m]['mean']
                row += f"  {gap:+.4f} "
            print(row)
        print("\n  Small gap → dynamics generalize. Large gap → model is a lookup table.")

    # Save
    out_path = 'v3_cross_validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
