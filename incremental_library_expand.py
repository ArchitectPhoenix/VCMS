"""
Incrementally expand the v3 library with new subjects from a CSV.

Only fits subjects not already in the library, preserving existing fits.
Outputs a new v3_library_fitted.json with the expanded set.
"""

import os
import sys
import json
import time
import numpy as np

from pgg_p_loader import load_p_experiment
from pgg_vcms_agent_v3 import VCMSParams, PARAM_NAMES, run_vcms_agent
from pgg_vcms_fit_v3 import fit_subject


def expand_library(csv_path, population, session, subject_filter=None,
                   library_path='v3_library_fitted.json',
                   canonical_path='p_experiment_canonical_library.json',
                   maxiter=600):
    """
    Add subjects from csv_path to both the canonical and fitted libraries.

    Args:
        csv_path: path to P-experiment CSV
        population: population name (e.g. 'Zurich')
        session: session label (e.g. 'Zurich_P_S1')
        subject_filter: optional set/list of subject IDs to include
        library_path: path to v3_library_fitted.json
        canonical_path: path to p_experiment_canonical_library.json
        maxiter: DE iterations per subject

    Returns:
        (n_added, library_path) tuple
    """
    # Load existing libraries
    with open(library_path) as f:
        fitted_lib = json.load(f)
    with open(canonical_path) as f:
        canonical_lib = json.load(f)

    # Load new data
    all_data = load_p_experiment(csv_path)

    # Filter subjects if requested
    if subject_filter:
        subject_filter = set(str(s) for s in subject_filter)
        all_data = {k: v for k, v in all_data.items() if k in subject_filter}

    # Skip subjects already in library
    new_sids = [sid for sid in sorted(all_data.keys()) if sid not in fitted_lib]
    if not new_sids:
        print(f"  All {len(all_data)} subjects already in library, nothing to add.")
        return 0, library_path

    print(f"  Fitting {len(new_sids)} new subjects (skipping {len(all_data) - len(new_sids)} existing)...")

    n_added = 0
    for i, sid in enumerate(new_sids):
        rounds = all_data[sid]
        contribs = [r.contribution for r in rounds]
        pun_sent = [r.punishment_sent_total for r in rounds]
        pun_recv = [r.punishment_received_total for r in rounds]
        others_mean = [r.others_mean for r in rounds]
        tvs = [r.total_voluntary_spend for r in rounds]

        mc = np.mean(contribs)
        mps = np.mean(pun_sent)
        anti = sum(r.antisocial_punishment for r in rounds)
        pro = sum(r.prosocial_punishment for r in rounds)
        total_pun = anti + pro
        ratio = anti / total_pun if total_pun > 0 else 0.0

        # Profile classification
        if mps < 0.5 and mc > 10:
            profile = "cooperator"
        elif mps < 0.5 and mc <= 5:
            profile = "free-rider"
        elif mps >= 0.5 and mc > 10:
            profile = "cooperative-enforcer"
        elif mps >= 0.5 and mc <= 5 and ratio > 0.5:
            profile = "antisocial-controller"
        elif mps >= 0.5 and mc <= 5:
            profile = "punitive-free-rider"
        else:
            profile = "mixed"

        print(f"    [{i+1}/{len(new_sids)}] {sid} ({profile})...", end=' ', flush=True)
        t0 = time.time()

        # Fit v3 params
        fit_result = fit_subject(rounds, maxiter=maxiter, seed=42, verbose=False)
        params = VCMSParams(**fit_result['best_params'])
        agent_result = run_vcms_agent(params, rounds)

        # State trajectory
        state_trajectory = []
        for t_data in agent_result['trace']:
            state_trajectory.append({
                'B': t_data['budget']['b_post'],
                'S': t_data['state']['strain_end'],
                'A': t_data['routing']['affordability'],
                'm_eval': t_data['m_eval']['m_eval_acc'],
                'gate': t_data['routing']['gate'],
            })

        elapsed = time.time() - t0
        print(f"RMSE={fit_result['best_rmse']:.4f} ({elapsed:.1f}s)")

        # Add to canonical library
        canonical_lib[sid] = {
            'params': {},
            'params_list': [],
            'actual': contribs,
            'predicted': [],
            'rmse': 0.0,
            'method': 'pending_v3_fit',
            'n_free': 0,
            'null_channels': [],
            'population': population,
            'session': session,
            'knockout_active_channels': [],
            'knockout_null_channels': [],
            'active_channels': [],
            'gap_classification': 'antisocial' if ratio > 0.5 else 'prosocial',
            'behavioral_profile': profile,
            'contribution_trajectory': contribs,
            'punishment_sent_trajectory': pun_sent,
            'punishment_received_trajectory': pun_recv,
            'total_voluntary_spend_trajectory': tvs,
            'others_mean_trajectory': [float(x) for x in others_mean],
            'mean_contribution': float(mc),
            'mean_punishment_sent': float(mps),
            'mean_punishment_received': float(np.mean(pun_recv)),
            'mean_total_voluntary_spend': float(np.mean(tvs)),
            'antisocial_punishment_total': int(anti),
            'prosocial_punishment_total': int(pro),
            'antisocial_ratio': float(ratio),
            'group_id': rounds[0].group_id,
        }

        # Add to fitted library
        fitted_lib[sid] = {
            'behavioral_profile': profile,
            'population': population,
            'session': session,
            'contribution_trajectory': contribs,
            'punishment_sent_trajectory': pun_sent,
            'punishment_received_trajectory': pun_recv,
            'others_mean_trajectory': [float(x) for x in others_mean],
            'v3_params': fit_result['best_params'],
            'v3_param_names': PARAM_NAMES,
            'v3_rmse': fit_result['best_rmse'],
            'v3_pred_c': agent_result['pred_contrib'],
            'v3_pred_p': agent_result['pred_punish'],
            'v3_state_trajectory': state_trajectory,
        }
        n_added += 1

    # Save both
    with open(canonical_path, 'w') as f:
        json.dump(canonical_lib, f, indent=2)
    with open(library_path, 'w') as f:
        json.dump(fitted_lib, f, indent=2)

    print(f"  Added {n_added} subjects. Library now has {len(fitted_lib)} subjects.")
    return n_added, library_path


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python incremental_library_expand.py <csv> <population> <session> [--sids 401,402,...]")
        sys.exit(1)

    csv_path = sys.argv[1]
    population = sys.argv[2]
    session = sys.argv[3]

    sids = None
    for arg in sys.argv[4:]:
        if arg.startswith('--sids='):
            sids = arg.split('=')[1].split(',')

    expand_library(csv_path, population, session, subject_filter=sids)
