#!/usr/bin/env python3
"""
Normalized-Time Fitting Module
===============================

Shared fast forward pass and fitting routines for normalized-time
parameter fitting. Used by both N-experiment and IPD re-fits.

dt = 1/(n_rounds-1) scales all temporal rate parameters so they
represent "per unit of game progress" rather than "per round."
"""

import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

from vcms_engine_v4 import PARAM_BOUNDS_NORMALIZED


# Parameters to optimize (15 free — same as ipd_fit.py)
FIT_PARAM_NAMES = [
    'alpha', 'v_rep', 'v_ref', 'c_base', 'inertia',
    's_dir', 's_rate', 's_initial',
    'b_initial', 'b_depletion_rate', 'b_replenish_rate', 'acute_threshold',
    'facilitation_rate', 'h_strength', 'h_start',
]

FIT_BOUNDS = [(PARAM_BOUNDS_NORMALIZED[p][0], PARAM_BOUNDS_NORMALIZED[p][1])
              for p in FIT_PARAM_NAMES]

# Fixed parameters (irrelevant for no-punishment games)
FIXED_PARAMS = {
    's_frac': 0.677,
    's_thresh': 1.800,
    'p_scale': 5.830,
    'v_self_weight': 0.0,
    's_exploitation_rate': 0.0,
}


def predict_fast_normalized(x, rounds, max_c):
    """
    Minimal VCMS forward pass with normalized game time.

    dt = 1/(n_rounds-1) applied to: s_rate, b_depletion_rate,
    b_replenish_rate, facilitation_rate.
    h_start is in [0,1] game progress, converted to round index.

    ~10x faster than run_vcms_v4 (no dict allocation, no trace storage).
    """
    alpha, v_rep, v_ref, c_base, inertia, s_dir_raw, s_rate, s_initial, \
        b_initial, b_depletion_rate, b_replenish_rate, acute_threshold, \
        facilitation_rate, h_strength, h_start = x

    s_dir = 1.0 if s_dir_raw >= 0 else -1.0
    w = max(-0.3, min(0.95, inertia))

    n = len(rounds)
    dt = 1.0 / (n - 1) if n > 1 else 1.0

    # Convert h_start from [0,1] game progress to round index
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

        # Strain (dt-scaled)
        if i > 0:
            gap = c_prev_norm - reference
            directed_gap = gap * s_dir
            gap_strain = max(0.0, directed_gap)
            strain += dt * s_rate * gap_strain

        # Budget (dt-scaled)
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

        # Affordability (gate=0 for no-punishment, discharge=0)
        affordability = B / (B + strain + 0.01)

        # Contribution
        if i == 0:
            c_norm = c_base
        else:
            c_target = v_ref * v_level + (1.0 - v_ref) * c_base
            c_target_adj = max(0.0, min(1.0, c_target + m_eval))
            c_norm = (1.0 - abs(w)) * c_target_adj + w * c_prev_norm

        # Horizon (h_start_round is in round units)
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

        # State update (teacher forcing)
        c_prev_norm = rd.contribution / max_c

    return preds


def objective(x, rounds, actual, max_c):
    """RMSE of normalized-time contribution predictions."""
    preds = predict_fast_normalized(x, rounds, max_c)
    n = len(actual)
    sse = sum((actual[i] - preds[i]) ** 2 for i in range(n))
    return math.sqrt(sse / n)


def fit_subject_de(rounds, actual, max_c, seed=42, maxiter=100, popsize=10):
    """Fit via differential evolution (fast, for IPD/N-experiment)."""
    result = differential_evolution(
        objective, FIT_BOUNDS,
        args=(rounds, actual, max_c),
        maxiter=maxiter, popsize=popsize, tol=0.005,
        seed=seed, polish=True,
        disp=False,
    )
    return result.x, result.fun, result.nfev


def params_array_to_dict(x):
    """Convert optimizer array to full params dict (15 fitted + 5 fixed)."""
    d = {}
    for i, name in enumerate(FIT_PARAM_NAMES):
        d[name] = float(x[i])
    d.update(FIXED_PARAMS)
    return d
