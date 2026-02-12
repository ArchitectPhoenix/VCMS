"""
VCMS v3 Combined Predictor
============================

Match on behavior (what V can see).
Predict from dynamics (what the model learned).

Architecture:
  For each new subject, maintain 32 library candidates.
  Each round:
    1. OBSERVE actual C/P at round t
    2. ELIMINATE candidates whose behavioral trajectory diverges
       (behavioral distance on raw C/P — what's visible)
    3. For surviving candidates, RUN v3 with their fitted params
       on the current subject's actual environment
    4. PREDICT round t+1 from inverse-distance-weighted v3 outputs
    5. Track elimination curve for forward shadow

This gives:
  - Convergence curve from behavioral elimination (V calibrating)
  - Model-grounded predictions from fitted dynamics (v3 does the work)
  - Forward shadow from elimination acceleration
"""

import math
import json
import os
import sys
import numpy as np
from collections import Counter

from pgg_vcms_agent_v3 import (
    VCMSParams, PARAM_NAMES, DEFAULTS,
    run_vcms_agent, MAX_CONTRIB, MAX_PUNISH,
)
from pgg_p_loader import load_p_experiment


# =============================================================================
# Library
# =============================================================================

class FittedLibrary:
    def __init__(self, library_path):
        with open(library_path) as f:
            self.raw = json.load(f)
        self.subjects = {}
        for sid, record in self.raw.items():
            self.subjects[sid] = {
                'profile': record['behavioral_profile'],
                'params': VCMSParams(**record['v3_params']),
                'actual_c': record['contribution_trajectory'],
                'actual_p': record['punishment_sent_trajectory'],
                'v3_rmse': record['v3_rmse'],
            }
        print(f"  Library: {len(self.subjects)} subjects")
        profiles = Counter(s['profile'] for s in self.subjects.values())
        print(f"  Profiles: {profiles}")

    def get_all_sids(self):
        return list(self.subjects.keys())


# =============================================================================
# Behavioral distance (for elimination — what V can see)
# =============================================================================

def behavioral_distance(actual_c, actual_p, lib_c, lib_p, t, p_weight=2.0):
    n = min(t + 1, len(actual_c), len(lib_c))
    if n == 0:
        return 0.0
    c_dist = sum((actual_c[i] - lib_c[i]) ** 2 for i in range(n))
    c_dist = c_dist / (MAX_CONTRIB ** 2 * n)
    p_n = min(t + 1, len(actual_p), len(lib_p))
    if p_n > 0:
        p_dist = sum((actual_p[i] - lib_p[i]) ** 2 for i in range(p_n))
        p_dist = p_dist / (MAX_PUNISH ** 2 * p_n)
    else:
        p_dist = 0.0
    return math.sqrt(c_dist + p_weight * p_dist)


# =============================================================================
# Combined predictor
# =============================================================================

class CombinedPredictor:
    def __init__(self, library, threshold=0.5):
        self.library = library
        self.threshold = threshold

    def predict_subject(self, rounds, verbose=False):
        n = len(rounds)
        survivors = list(self.library.get_all_sids())

        predictions_c = []
        predictions_p = []
        actuals_c = []
        actuals_p = []
        rmse_by_round = []
        elimination_curve = []
        effective_candidates = []
        behavioral_distances = {}  # carried forward for weighting

        for t in range(n):
            # ==============================================================
            # STEP 1: PREDICT round t from surviving candidates' dynamics
            # Run v3 with each survivor's fitted params on rounds 0..t-1
            # to get their prediction for round t
            # ==============================================================
            n_candidates = len(survivors)

            if t == 0:
                # No prior rounds — use library trajectory means as prior
                pred_c = int(round(np.mean([
                    self.library.subjects[s]['actual_c'][0]
                    for s in survivors
                ])))
                pred_p = int(round(np.mean([
                    self.library.subjects[s]['actual_p'][0]
                    for s in survivors
                ])))
            else:
                # Run v3 for each survivor on actual environment 0..t-1
                # v3 predicts round t from accumulated state
                candidate_preds = {}
                for sid in survivors:
                    params = self.library.subjects[sid]['params']
                    result = run_vcms_agent(params, rounds[:t])
                    # v3 produces predictions for rounds 0..t-1
                    # The prediction for round t-1 uses state from 0..t-2
                    # We need to run on rounds 0..t to get pred for round t
                    result_extended = run_vcms_agent(params, rounds[:t+1])
                    # pred[t] is computed before round t's actual data
                    # affects state — no leakage
                    candidate_preds[sid] = (
                        result_extended['pred_contrib'][t],
                        result_extended['pred_punish'][t],
                    )

                # Weight by behavioral distance (from last elimination)
                weights = {}
                for sid in survivors:
                    d = behavioral_distances.get(sid, 0.0)
                    weights[sid] = 1.0 / (d + 0.001)
                total_w = sum(weights.values())

                pred_c = 0.0
                pred_p = 0.0
                for sid in survivors:
                    w = weights[sid] / total_w
                    pc, pp = candidate_preds[sid]
                    pred_c += w * pc
                    pred_p += w * pp

                pred_c = int(round(pred_c))
                pred_p = int(round(pred_p))

            predictions_c.append(pred_c)
            predictions_p.append(pred_p)
            effective_candidates.append(n_candidates)

            # ==============================================================
            # STEP 2: OBSERVE actual at round t
            # ==============================================================
            actual_c = rounds[t].contribution
            actual_p = rounds[t].punishment_sent_total
            actuals_c.append(actual_c)
            actuals_p.append(actual_p)

            err_c = (pred_c - actual_c) / MAX_CONTRIB
            err_p = (pred_p - actual_p) / MAX_PUNISH if MAX_PUNISH > 0 else 0
            rmse_by_round.append(math.sqrt((err_c ** 2 + err_p ** 2) / 2))

            # ==============================================================
            # STEP 3: ELIMINATE on behavioral distance (what V sees)
            # ==============================================================
            distances = {}
            for sid in survivors:
                d = behavioral_distance(
                    actuals_c, actuals_p,
                    self.library.subjects[sid]['actual_c'],
                    self.library.subjects[sid]['actual_p'],
                    t, p_weight=2.0
                )
                distances[sid] = d

            # Adaptive threshold
            if distances:
                best_d = min(distances.values())
                adaptive_thresh = max(best_d * 3.0, self.threshold)
            else:
                adaptive_thresh = self.threshold

            new_survivors = [sid for sid in survivors
                             if distances.get(sid, 999) <= adaptive_thresh]
            if not new_survivors and distances:
                new_survivors = [min(distances, key=distances.get)]

            survivors = new_survivors
            behavioral_distances = {sid: distances[sid] for sid in survivors}
            elimination_curve.append(len(survivors))

        # Derivatives
        E = elimination_curve
        E_prime = [0] + [E[i] - E[i-1] for i in range(1, len(E))]
        E_double_prime = [0] + [E_prime[i] - E_prime[i-1] for i in range(1, len(E_prime))]

        # Forward shadow events
        forward_shadow_events = []
        for t in range(len(E_double_prime)):
            if E_double_prime[t] < -2 and t + 1 < n:
                c_change = abs(actuals_c[t+1] - actuals_c[t])
                if c_change >= 3:
                    forward_shadow_events.append({
                        'round': t + 1,
                        'E_double_prime': E_double_prime[t],
                        'c_change_next': c_change,
                    })

        overall_rmse = math.sqrt(np.mean([e ** 2 for e in rmse_by_round]))

        final_profiles = dict(Counter(
            self.library.subjects[sid]['profile'] for sid in survivors
        ))

        return {
            'pred_c': predictions_c,
            'pred_p': predictions_p,
            'actual_c': actuals_c,
            'actual_p': actuals_p,
            'rmse_combined': overall_rmse,
            'rmse_by_round': rmse_by_round,
            'elimination_curve': elimination_curve,
            'elimination_E_prime': E_prime,
            'elimination_E_double_prime': E_double_prime,
            'forward_shadow_events': forward_shadow_events,
            'effective_candidates': effective_candidates,
            'final_survivor_count': len(survivors),
            'final_profiles': final_profiles,
        }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_city(library_path, city_csv, threshold=0.5, verbose=True, sid_filter=None):
    library = FittedLibrary(library_path)
    data = load_p_experiment(city_csv)
    if sid_filter:
        sid_filter = set(str(s) for s in sid_filter)
        data = {k: v for k, v in data.items() if k in sid_filter}
    predictor = CombinedPredictor(library, threshold=threshold)

    all_results = {}
    for sid, rounds in sorted(data.items()):
        result = predictor.predict_subject(rounds, verbose=verbose)
        all_results[sid] = result
        if verbose:
            print(f"  {sid}: RMSE={result['rmse_combined']:.4f}, "
                  f"elim={result['elimination_curve']}, "
                  f"final={result['final_profiles']}")
            if result['forward_shadow_events']:
                for fs in result['forward_shadow_events']:
                    print(f"    ⚡ E''={fs['E_double_prime']} at R{fs['round']}, "
                          f"ΔC={fs['c_change_next']}")

    # Aggregate
    n_rounds = 10
    print(f"\n{'=' * 60}")
    print(f"COMBINED PREDICTOR — {os.path.basename(city_csv)}")
    print(f"{'=' * 60}")
    for t in range(n_rounds):
        errors_t = [r['rmse_by_round'][t] for r in all_results.values()]
        m = np.mean(errors_t)
        s = np.std(errors_t)
        bar = '#' * int(m * 50)
        print(f'  R{t+1:2d}: {m:.4f} +/- {s:.4f}  {bar}')

    rmses = [r['rmse_combined'] for r in all_results.values()]
    print(f'  Overall: {np.mean(rmses):.4f}')

    n_fs = sum(len(r['forward_shadow_events']) for r in all_results.values())
    print(f'  Forward shadow events: {n_fs}')
    for sid, r in all_results.items():
        for fs in r['forward_shadow_events']:
            print(f'    {sid}: E"={fs["E_double_prime"]} R{fs["round"]}, dC={fs["c_change_next"]}')

    for t in range(n_rounds):
        surv = [r['elimination_curve'][t] for r in all_results.values()]
        print(f'  R{t+1:2d} survivors: {np.mean(surv):.1f} +/- {np.std(surv):.1f}')

    return all_results


if __name__ == '__main__':
    library_path = 'v3_library_fitted.json'
    if not os.path.exists(library_path):
        print("ERROR: v3_library_fitted.json not found")
        sys.exit(1)

    city_csv = None
    threshold = 0.5
    sid_filter = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--csv' and i + 1 < len(args):
            city_csv = args[i + 1]; i += 2
        elif args[i] == '--threshold' and i + 1 < len(args):
            threshold = float(args[i + 1]); i += 2
        elif args[i] == '--sids' and i + 1 < len(args):
            sid_filter = args[i + 1].split(','); i += 2
        else:
            i += 1

    if not city_csv:
        for f in sorted(os.listdir('.')):
            if 'P-EXPERIMENT' in f and f.endswith('.csv'):
                city_csv = f; break

    if not city_csv:
        print("ERROR: No CSV found"); sys.exit(1)

    evaluate_city(library_path, city_csv, threshold=threshold, sid_filter=sid_filter)
