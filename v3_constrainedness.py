#!/usr/bin/env python3
"""
Parameter constrainedness mask for the VCMS v3 library.

For each subject × parameter pair, determines whether the parameter had
gradient during fitting (constrained) or was free to wander because the
observable it acts through was at floor/ceiling (unconstrained).

Unconstrained parameters are noise — their fitted values carry no
information about the subject. Any downstream analysis touching a
parameter must filter to the constrained population for that parameter.

Multiplicative unconstrained condition:
  When a parameter only expresses through a variable that's at floor or
  ceiling for a given subject, the loss surface is flat along that
  parameter's axis. The optimizer finds a value, but that value is
  arbitrary.

The 9 always-constrained parameters (c_base, alpha, inertia, s_dir,
s_rate, s_initial, b_initial, facilitation_rate, v_ref) are Layer 0-1
primitives that express in every environment. The conditionally-
constrained parameters (p_scale, h_start, h_strength, s_frac, s_thresh,
b_depletion_rate, b_replenish_rate, acute_threshold, v_rep) are Layer 2+
— they only activate under specific conditions.

Output format (v3_library_constrainedness.json):
  {
    "<sid>": {
      "mask": {"<param>": true/false, ...},  // true = constrained
      "n_constrained": int,
      "n_unconstrained": int,
      "unconstrained_params": [list of param names]
    },
    "_summary": { ... }
  }
"""

import json
import math
import numpy as np
from collections import defaultdict, Counter

from pgg_vcms_agent_v3 import PARAM_NAMES, MAX_CONTRIB, MAX_PUNISH


# ================================================================
# Constrainedness criteria
# ================================================================

# Thresholds for floor/ceiling detection
C_FLOOR = 2.0          # mean C ≤ this → contribution floor
P_FLOOR = 1.0          # mean P_sent ≤ this → punishment floor
H_STRENGTH_FLOOR = 0.05  # h_strength ≤ this → h_start unconstrained
GAP_FLOOR = 0.05       # mean |gap| ≤ this → no strain signal
EXP_TOLERANCE = 0.05   # experience within this of 0 → no signal
OM_FLOOR = 1.0         # mean others_mean ≤ this → group at floor


def compute_constrainedness(rec):
    """
    Compute constrainedness mask for a single subject.

    Args:
        rec: library record with 'v3_params', 'contribution_trajectory',
             'punishment_sent_trajectory', 'others_mean_trajectory',
             'punishment_received_trajectory'

    Returns:
        dict: {param_name: bool} where True = constrained (has gradient)
    """
    p = rec['v3_params']
    ct = np.array(rec['contribution_trajectory'])
    pt = np.array(rec['punishment_sent_trajectory'])
    om = np.array(rec['others_mean_trajectory'])
    pr = np.array(rec['punishment_received_trajectory'])

    mean_c = np.mean(ct)
    mean_p = np.mean(pt)
    mean_om = np.mean(om)

    c_norm = ct / MAX_CONTRIB
    om_norm = om / MAX_CONTRIB

    # Experience signal per round (from round 2 onward)
    if len(c_norm) > 1 and len(om_norm) > 1:
        experiences = om_norm[1:] - c_norm[:-1]
    else:
        experiences = np.array([0.0])

    # Gap signal per round
    if len(c_norm) > 1 and len(om_norm) > 1:
        gaps = np.abs(c_norm[:-1] - om_norm[:-1])
    else:
        gaps = np.array([0.0])

    mask = {}

    # --- Always constrained ---
    mask['c_base'] = True
    mask['alpha'] = True
    mask['inertia'] = True
    mask['s_dir'] = True
    mask['s_rate'] = True
    mask['s_initial'] = True
    mask['b_initial'] = True
    mask['facilitation_rate'] = True
    mask['v_ref'] = True

    # --- Conditionally constrained ---

    # h_strength: unconstrained when C ≈ 0 (discount × 0 = 0)
    mask['h_strength'] = mean_c > C_FLOOR

    # h_start: unconstrained when h_strength ≈ 0 OR C ≈ 0
    mask['h_start'] = (p['h_strength'] > H_STRENGTH_FLOOR) and (mean_c > C_FLOOR)

    # p_scale: unconstrained when P ≈ 0 (no punishment output)
    mask['p_scale'] = mean_p > P_FLOOR

    # s_frac: unconstrained when strain ≈ 0 AND P ≈ 0
    # (s_frac controls what fraction of strain exits as punishment;
    # needs both strain to exist and punishment to express)
    mask['s_frac'] = not (np.mean(gaps) < GAP_FLOOR and mean_p <= P_FLOOR)

    # s_thresh: unconstrained when s_frac ≈ 0 OR strain ≈ 0
    # (s_thresh gates the discharge; if nothing to discharge, no gradient)
    mask['s_thresh'] = mask['s_frac'] and (p['s_frac'] > 0.05)

    # b_depletion_rate: unconstrained when experience ≥ 0 always
    # (depletion only fires on negative experience)
    mask['b_depletion_rate'] = bool(np.any(experiences < -EXP_TOLERANCE))

    # b_replenish_rate: unconstrained when experience ≤ 0 always
    # (replenishment only fires on positive experience)
    mask['b_replenish_rate'] = bool(np.any(experiences > EXP_TOLERANCE))

    # acute_threshold: unconstrained when no exploitation events
    # (acute pathway only fires on negative experience exceeding threshold)
    mask['acute_threshold'] = bool(np.any(experiences < -EXP_TOLERANCE))

    # v_rep: unconstrained when group contributes nothing
    # (v_rep scales the group signal; no signal → no gradient)
    mask['v_rep'] = mean_om > OM_FLOOR

    return mask


def build_constrainedness_metadata(library):
    """
    Build full constrainedness metadata for the library.

    Returns:
        dict with per-subject masks and summary statistics.
    """
    metadata = {}
    param_counts = {p: {'constrained': 0, 'unconstrained': 0}
                    for p in PARAM_NAMES}
    profile_stats = defaultdict(lambda: defaultdict(
        lambda: {'constrained': 0, 'unconstrained': 0}))

    for sid, rec in sorted(library.items()):
        mask = compute_constrainedness(rec)
        unconstrained = [p for p in PARAM_NAMES if not mask.get(p, True)]
        n_constrained = sum(1 for p in PARAM_NAMES if mask.get(p, True))

        metadata[sid] = {
            'mask': {p: bool(mask.get(p, True)) for p in PARAM_NAMES},
            'n_constrained': n_constrained,
            'n_unconstrained': len(PARAM_NAMES) - n_constrained,
            'unconstrained_params': unconstrained,
        }

        profile = rec.get('behavioral_profile', 'unknown')
        for p in PARAM_NAMES:
            if mask.get(p, True):
                param_counts[p]['constrained'] += 1
                profile_stats[profile][p]['constrained'] += 1
            else:
                param_counts[p]['unconstrained'] += 1
                profile_stats[profile][p]['unconstrained'] += 1

    n = len(library)

    # Summary
    summary = {
        'total_subjects': n,
        'total_cells': n * len(PARAM_NAMES),
        'per_parameter': {
            p: {
                'constrained': param_counts[p]['constrained'],
                'unconstrained': param_counts[p]['unconstrained'],
                'pct_constrained': round(
                    100 * param_counts[p]['constrained'] / n, 1),
            }
            for p in PARAM_NAMES
        },
        'always_constrained': [
            p for p in PARAM_NAMES
            if param_counts[p]['unconstrained'] == 0
        ],
        'conditionally_constrained': [
            p for p in PARAM_NAMES
            if 0 < param_counts[p]['unconstrained'] < n
        ],
        'per_profile': {
            profile: {
                p: {
                    'constrained': stats[p]['constrained'],
                    'pct': round(100 * stats[p]['constrained'] /
                                 (stats[p]['constrained'] +
                                  stats[p]['unconstrained']), 1)
                }
                for p in PARAM_NAMES
                if stats[p]['constrained'] + stats[p]['unconstrained'] > 0
            }
            for profile, stats in sorted(profile_stats.items())
        },
    }

    # Per-subject constrainedness distribution
    n_constrained_dist = Counter(
        metadata[sid]['n_constrained'] for sid in metadata
        if sid != '_summary'
    )
    summary['subject_constrainedness_distribution'] = dict(
        sorted(n_constrained_dist.items()))

    metadata['_summary'] = summary
    return metadata


# ================================================================
# Convenience functions for downstream use
# ================================================================

_CACHED_METADATA = None


def load_constrainedness(path='v3_library_constrainedness.json'):
    """Load and cache the constrainedness metadata."""
    global _CACHED_METADATA
    if _CACHED_METADATA is None:
        with open(path) as f:
            _CACHED_METADATA = json.load(f)
    return _CACHED_METADATA


def is_constrained(sid, param, metadata=None):
    """Check if a parameter is constrained for a given subject."""
    if metadata is None:
        metadata = load_constrainedness()
    return metadata.get(sid, {}).get('mask', {}).get(param, True)


def constrained_sids(param, library, metadata=None):
    """Return list of sids where param is constrained."""
    if metadata is None:
        metadata = load_constrainedness()
    return [sid for sid in library
            if metadata.get(sid, {}).get('mask', {}).get(param, True)]


def constrained_values(param, library, metadata=None):
    """Return {sid: value} for subjects where param is constrained."""
    if metadata is None:
        metadata = load_constrainedness()
    return {
        sid: rec['v3_params'][param]
        for sid, rec in library.items()
        if metadata.get(sid, {}).get('mask', {}).get(param, True)
    }


# ================================================================
# Main: generate and save
# ================================================================

if __name__ == '__main__':
    with open('v3_library_fitted.json') as f:
        library = json.load(f)

    print("Computing constrainedness mask...")
    metadata = build_constrainedness_metadata(library)
    summary = metadata['_summary']

    # Save
    out_path = 'v3_library_constrainedness.json'
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved to {out_path}")

    # Print summary
    n = summary['total_subjects']
    print(f"\n{'=' * 65}")
    print(f"  CONSTRAINEDNESS SUMMARY ({n} subjects × {len(PARAM_NAMES)} params)")
    print(f"{'=' * 65}")

    print(f"\n  {'Parameter':<22s}  {'Constrained':>12s}  {'Unconstrained':>14s}  {'%':>6s}")
    print(f"  {'-' * 22}  {'-' * 12}  {'-' * 14}  {'-' * 6}")

    for p in PARAM_NAMES:
        s = summary['per_parameter'][p]
        print(f"  {p:<22s}  {s['constrained']:>12d}  {s['unconstrained']:>14d}  "
              f"{s['pct_constrained']:>5.1f}%")

    print(f"\n  Always constrained:        {', '.join(summary['always_constrained'])}")
    print(f"  Conditionally constrained: {', '.join(summary['conditionally_constrained'])}")

    # Subject-level stats
    print(f"\n  Subjects by # constrained params:")
    for n_c, count in sorted(summary['subject_constrainedness_distribution'].items()):
        bar = '#' * count
        print(f"    {n_c:>2d}/18 constrained: {count:>3d} subjects  {bar}")

    total_unconstrained = sum(
        summary['per_parameter'][p]['unconstrained'] for p in PARAM_NAMES)
    total_cells = summary['total_cells']
    print(f"\n  Total cells: {total_cells}")
    print(f"  Unconstrained: {total_unconstrained} ({100*total_unconstrained/total_cells:.1f}%)")
    print(f"  Constrained: {total_cells - total_unconstrained} "
          f"({100*(total_cells - total_unconstrained)/total_cells:.1f}%)")

    # Per-profile breakdown for key parameters
    print(f"\n  {'Profile':<25s}  {'p_scale':>8s}  {'h_str':>7s}  {'h_start':>8s}  "
          f"{'b_depl':>7s}  {'b_repl':>7s}")
    print(f"  {'-' * 25}  {'-' * 8}  {'-' * 7}  {'-' * 8}  {'-' * 7}  {'-' * 7}")

    for profile in sorted(summary['per_profile'].keys()):
        pdata = summary['per_profile'][profile]
        row = f"  {profile:<25s}"
        for param in ['p_scale', 'h_strength', 'h_start', 'b_depletion_rate',
                       'b_replenish_rate']:
            if param in pdata:
                row += f"  {pdata[param]['pct']:>6.0f}%"
            else:
                row += f"  {'N/A':>7s}"
        print(row)
