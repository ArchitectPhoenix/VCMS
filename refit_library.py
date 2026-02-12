#!/usr/bin/env python3
"""
Refit all 176 subjects with the 18-parameter v3 model (horizon-aware).
Rebuilds v3_library_fitted.json.
"""
import os
import sys
import json
import time
import numpy as np

from pgg_p_loader import load_p_experiment
from pgg_vcms_fit_v3 import fit_subject
from pgg_vcms_agent_v3 import (
    VCMSParams, PARAM_NAMES, run_vcms_agent, make_vcms_params,
)


def classify_profile(contrib_traj, pun_traj):
    """Quick behavioral classification."""
    avg_c = np.mean(contrib_traj)
    avg_p = np.mean(pun_traj)
    if avg_c >= 15 and avg_p >= 3:
        return 'cooperative-enforcer'
    elif avg_c >= 15:
        return 'cooperator'
    elif avg_c <= 5:
        return 'free-rider'
    elif np.std(contrib_traj) > 5:
        return 'volatile'
    else:
        return 'moderate'


def main():
    csv_files = sorted(f for f in os.listdir('.')
                       if f.endswith('.csv') and 'P-EXPERIMENT' in f)

    # Load all subjects
    subjects = {}
    for csv_file in csv_files:
        parts = csv_file.split('_')
        city = parts[1]  # BOSTON, SAMARA, etc.
        session_tag = csv_file.replace('.csv', '')
        data = load_p_experiment(csv_file)
        for sid, rounds in data.items():
            subjects[sid] = {
                'rounds': rounds,
                'city': city,
                'session': session_tag,
            }

    print(f"Refitting {len(subjects)} subjects with 18 params (v3 + horizon)")
    print(f"{'='*70}")

    library = {}
    t_start = time.time()

    for idx, (sid, info) in enumerate(sorted(subjects.items())):
        rounds = info['rounds']
        t0 = time.time()

        fit = fit_subject(rounds, maxiter=600, seed=42, verbose=False)

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        rate = (idx + 1) / total_elapsed if total_elapsed > 0 else 1
        remaining = (len(subjects) - idx - 1) / rate if rate > 0 else 0

        p = fit['best_params']
        h_str = p.get('h_strength', 0)
        h_st = p.get('h_start', 7)

        print(f"  [{idx+1}/{len(subjects)}] {sid}: "
              f"RMSE={fit['best_rmse']:.5f}, "
              f"h_str={h_str:.3f}, h_start={h_st:.1f}, "
              f"{elapsed:.1f}s "
              f"(~{remaining:.0f}s remaining)")

        # Build library entry
        params = make_vcms_params(
            [fit['best_params'][n] for n in PARAM_NAMES],
            PARAM_NAMES, None
        )
        result = run_vcms_agent(params, rounds)

        contrib_traj = [r.contribution for r in rounds]
        pun_traj = [r.punishment_sent_total for r in rounds]
        others_traj = [r.others_mean for r in rounds]
        pun_recv_traj = [r.punishment_received_total for r in rounds]

        library[sid] = {
            'behavioral_profile': classify_profile(contrib_traj, pun_traj),
            'population': info['city'],
            'session': info['session'],
            'contribution_trajectory': contrib_traj,
            'punishment_sent_trajectory': pun_traj,
            'punishment_received_trajectory': pun_recv_traj,
            'others_mean_trajectory': others_traj,
            'v3_params': fit['best_params'],
            'v3_rmse': fit['best_rmse'],
            'v3_pred_c': result['pred_contrib'],
            'v3_pred_p': result['pred_punish'],
        }

    # Save
    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open('v3_library_fitted.json', 'w') as f:
        json.dump(library, f, indent=2, default=to_json)

    total = time.time() - t_start
    rmses = [v['v3_rmse'] for v in library.values()]
    print(f"\n{'='*70}")
    print(f"Done in {total:.0f}s")
    print(f"Mean RMSE: {np.mean(rmses):.5f}")
    print(f"Median RMSE: {np.median(rmses):.5f}")

    # Horizon stats
    h_strengths = [v['v3_params']['h_strength'] for v in library.values()]
    h_active = sum(1 for h in h_strengths if h > 0.05)
    print(f"Horizon active (h_strength > 0.05): {h_active}/{len(library)} "
          f"({100*h_active/len(library):.0f}%)")
    print(f"Mean h_strength: {np.mean(h_strengths):.3f}")

    print(f"\nLibrary saved to v3_library_fitted.json")


if __name__ == '__main__':
    main()
