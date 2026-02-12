"""
Fit v3 params for all library subjects and build state trajectory library.

This is the bridge between "address book lookup" and "actual model."
After this runs, each library subject has:
  - Fitted v3 parameters (the model's representation of their dynamics)
  - State trajectories under those params (B, S, A, m_eval, gate per round)
  - Predicted C/P under those params

The sequential predictor can then match on STATE DYNAMICS, not raw behavior.
"""

import os
import sys
import json
import time
import numpy as np

from pgg_p_loader import load_p_experiment
from pgg_vcms_agent_v3 import (
    VCMSParams, PARAM_NAMES, PARAM_BOUNDS, DEFAULTS,
    run_vcms_agent, make_vcms_params, vcms_objective,
)
from pgg_vcms_fit_v3 import fit_subject


def build_v3_library(library_path, data_dirs, output_path, maxiter=600):
    """
    Fit v3 parameters for every subject in the canonical library.
    
    Args:
        library_path: path to p_experiment_canonical_library.json
        data_dirs: list of directories to search for CSV files
        output_path: where to save the v3-augmented library
        maxiter: DE iterations per subject
    """
    with open(library_path) as f:
        library = json.load(f)

    # Load all city data
    all_data = {}
    csv_files = []
    for d in data_dirs:
        for fname in os.listdir(d):
            if fname.endswith('.csv') and 'P-EXPERIMENT' in fname:
                csv_files.append(os.path.join(d, fname))

    for csv_path in csv_files:
        city_data = load_p_experiment(csv_path)
        for sid, rounds in city_data.items():
            all_data[sid] = rounds
        print(f"  Loaded {len(city_data)} subjects from {os.path.basename(csv_path)}")

    print(f"\nTotal subjects available: {len(all_data)}")
    print(f"Library subjects to fit: {len(library)}")

    # Check which library subjects we have round data for
    missing = [sid for sid in library if sid not in all_data]
    if missing:
        print(f"WARNING: {len(missing)} library subjects not found in data: {missing}")

    # Fit each library subject
    results = {}
    total_t0 = time.time()

    for i, (sid, record) in enumerate(sorted(library.items())):
        if sid not in all_data:
            print(f"  [{i+1}/{len(library)}] {sid}: SKIPPED (no round data)")
            continue

        rounds = all_data[sid]
        print(f"  [{i+1}/{len(library)}] {sid} ({record['behavioral_profile']})...", end=' ')
        t0 = time.time()

        fit_result = fit_subject(rounds, maxiter=maxiter, seed=42, verbose=False)

        elapsed = time.time() - t0
        print(f"RMSE={fit_result['best_rmse']:.4f} ({elapsed:.1f}s)")

        # Run agent with fitted params to get state trajectories
        params = VCMSParams(**fit_result['best_params'])
        agent_result = run_vcms_agent(params, rounds)

        # Extract state trajectory
        state_trajectory = []
        for t_data in agent_result['trace']:
            state_trajectory.append({
                'B': t_data['budget']['b_post'],
                'S': t_data['state']['strain_end'],
                'A': t_data['routing']['affordability'],
                'm_eval': t_data['m_eval']['m_eval_acc'],
                'gate': t_data['routing']['gate'],
            })

        # Store everything
        results[sid] = {
            # Original library data
            'behavioral_profile': record['behavioral_profile'],
            'population': record.get('population', ''),
            'session': record.get('session', ''),
            'contribution_trajectory': record['contribution_trajectory'],
            'punishment_sent_trajectory': record['punishment_sent_trajectory'],
            'punishment_received_trajectory': record['punishment_received_trajectory'],
            'others_mean_trajectory': record['others_mean_trajectory'],

            # V3 fitted model
            'v3_params': fit_result['best_params'],
            'v3_param_names': PARAM_NAMES,
            'v3_rmse': fit_result['best_rmse'],
            'v3_pred_c': agent_result['pred_contrib'],
            'v3_pred_p': agent_result['pred_punish'],
            'v3_state_trajectory': state_trajectory,
        }

    total_elapsed = time.time() - total_t0
    print(f"\nTotal fitting time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Subjects fitted: {len(results)}/{len(library)}")

    # Summary stats
    rmses = [r['v3_rmse'] for r in results.values()]
    print(f"RMSE: mean={np.mean(rmses):.4f}, median={np.median(rmses):.4f}, "
          f"max={np.max(rmses):.4f}")

    # Save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    return results


if __name__ == '__main__':
    library_path = os.path.join('.', 'p_experiment_canonical_library.json')
    data_dirs = ['.']
    output_path = os.path.join('.', 'v3_library_fitted.json')

    maxiter = 600
    if '--fast' in sys.argv:
        maxiter = 100
        print("FAST MODE: 100 iterations (for testing)")

    build_v3_library(library_path, data_dirs, output_path, maxiter=maxiter)
