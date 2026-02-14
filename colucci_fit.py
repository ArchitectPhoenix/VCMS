#!/usr/bin/env python3
"""
Fit VCMS parameters to Colucci, Franco & Valori (2023) endowment data.

For each of the ~516 subjects:
  1. Reads their condition (buyer/now/day/week/month) and 3 item valuations
  2. Uses differential evolution to find VCMS parameters that best explain
     their valuations via the endowment adapter
  3. Stores fitted parameters as a JSON library

Then runs fitted parameters through PGG and IPD engines to verify that
behavioral phenotype distributions match existing cooperation-game libraries.

Design choices:
  - Sellers (415 subjects): fit 10 params (all adapter-used params)
  - Buyers (101 subjects): fit 2 params (c_base, inertia — only ones that
    affect buyer WTP)
  - Unfitted params: fixed at library medians
  - Regularization: L2 penalty toward library medians prevents overfitting
    with only 3 observations per subject
  - Objective: relative RMSE (normalized by item value) + regularization
"""

import csv
import json
import math
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from scipy.optimize import differential_evolution

from endowment_adapter import compute_valuation, ITEM_TYPES, ITEM_VALUES
from federation_sim import AgentParams
from vcms_engine_v4 import (
    VCMSParams, GameConfig, PGG_P_CONFIG,
    run_vcms_v4, v3_params_to_v4,
)
from v4_validation import SimpleRound


# ================================================================
# CONSTANTS
# ================================================================

CSV_PATH = "dataset_ColucciFrancoValori2023.csv"
OUTPUT_PATH = "v4_colucci_library_fitted.json"

# Library medians from 196 P-experiment fitted subjects
LIBRARY_MEDIANS = {
    'alpha': 0.4833,
    'v_rep': 1.3106,
    'v_ref': 0.5575,
    'c_base': 0.7694,
    'inertia': 0.0755,
    's_dir': 1.0,           # snapped to +1
    's_rate': 0.6613,
    's_initial': 0.2289,
    'b_initial': 3.6614,
    'b_depletion_rate': 0.6084,
    'b_replenish_rate': 1.3650,
    'acute_threshold': 0.5389,
    'facilitation_rate': 0.3890,
    'h_strength': 0.0162,
    'h_start': 5.2377,
}

# Library standard deviations (for regularization weighting)
LIBRARY_STDS = {
    'alpha': 0.2831,
    'v_rep': 0.4680,
    'v_ref': 0.3501,
    'c_base': 0.3231,
    'inertia': 0.3398,
    's_rate': 0.6823,
    's_initial': 2.0041,
    'facilitation_rate': 0.3182,
    'b_initial': 1.3638,
    'b_replenish_rate': 0.5497,
}

# VCMSParams-only fields (not in AgentParams) — fixed at library medians
VCMS_EXTRA = {
    's_frac': 0.677,
    's_thresh': 1.800,
    'p_scale': 5.830,
    'v_self_weight': 0.0,
    's_exploitation_rate': 0.0,
}

# Parameters used by the endowment adapter's seller path
SELLER_FREE_NAMES = [
    'alpha', 'v_rep', 'v_ref', 'c_base', 'inertia',
    's_rate', 's_initial', 'facilitation_rate',
    'b_initial', 'b_replenish_rate',
]
SELLER_BOUNDS = [
    (0.01, 0.99),   # alpha
    (0.5, 2.0),     # v_rep
    (0.0, 1.0),     # v_ref
    (0.0, 1.0),     # c_base
    (-0.3, 0.95),   # inertia
    (0.0, 2.0),     # s_rate
    (0.0, 10.0),    # s_initial
    (0.0, 1.0),     # facilitation_rate
    (0.1, 5.0),     # b_initial
    (0.0, 2.0),     # b_replenish_rate
]

# Only c_base and inertia affect buyer WTP
BUYER_FREE_NAMES = ['c_base', 'inertia']
BUYER_BOUNDS = [(0.0, 1.0), (-0.3, 0.95)]

# Regularization strength (relative to data fit)
REG_LAMBDA = 0.05


# ================================================================
# DATA LOADING
# ================================================================

def load_colucci_data(csv_path=CSV_PATH):
    """Load Colucci CSV, return dict of per-subject records."""
    subjects = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get('PROLIFIC_PID', '').strip()
            if not pid or pid == 'NA':
                continue
            if row.get('Sex', '') == 'DATA EXPIRED':
                continue
            cond = row.get('cond', '').strip()
            if cond not in ('buyer', 'now', 'day', 'week', 'month'):
                continue
            try:
                mug = float(row['Mug_assess'])
                amazon = float(row['Amazon_assess'])
                spotify = float(row['Spotify_assess'])
            except (ValueError, TypeError, KeyError):
                continue
            subjects[pid] = {
                'condition': cond,
                'valuations': {
                    'mug': mug,
                    'amazon': amazon,
                    'spotify': spotify,
                },
                'sex': row.get('Sex', ''),
                'age': row.get('age', ''),
            }
    return subjects


# ================================================================
# PARAMETER CONSTRUCTION
# ================================================================

def build_agent_params(free_values, free_names):
    """Build AgentParams from free values + fixed medians."""
    p = dict(LIBRARY_MEDIANS)
    for name, val in zip(free_names, free_values):
        p[name] = val
    return AgentParams(
        alpha=p['alpha'],
        v_rep=p['v_rep'],
        v_ref=p['v_ref'],
        c_base=p['c_base'],
        inertia=max(-0.3, min(0.95, p['inertia'])),
        s_dir=1.0 if p['s_dir'] >= 0 else -1.0,
        s_rate=p['s_rate'],
        s_initial=p['s_initial'],
        b_initial=p['b_initial'],
        b_depletion_rate=p['b_depletion_rate'],
        b_replenish_rate=p['b_replenish_rate'],
        acute_threshold=p['acute_threshold'],
        facilitation_rate=p['facilitation_rate'],
        h_strength=p['h_strength'],
        h_start=p['h_start'],
    )


def agent_params_to_v3_dict(ap):
    """Convert AgentParams to a v3_params dict for storage."""
    return {
        'alpha': ap.alpha,
        'v_rep': ap.v_rep,
        'v_ref': ap.v_ref,
        'c_base': ap.c_base,
        'inertia': ap.inertia,
        's_dir': ap.s_dir,
        's_rate': ap.s_rate,
        's_initial': ap.s_initial,
        'b_initial': ap.b_initial,
        'b_depletion_rate': ap.b_depletion_rate,
        'b_replenish_rate': ap.b_replenish_rate,
        'acute_threshold': ap.acute_threshold,
        'facilitation_rate': ap.facilitation_rate,
        'h_strength': ap.h_strength,
        'h_start': ap.h_start,
        # VCMSParams extras for PGG/IPD
        's_frac': VCMS_EXTRA['s_frac'],
        's_thresh': VCMS_EXTRA['s_thresh'],
        'p_scale': VCMS_EXTRA['p_scale'],
    }


def v3_dict_to_vcms_params(d):
    """Convert a v3_params dict to VCMSParams for engine runs."""
    return v3_params_to_v4(d, s_exploitation_rate=0.0, v_self_weight=0.0)


# ================================================================
# FITTING OBJECTIVE
# ================================================================

def objective(x, free_names, condition, actual_vals):
    """
    Relative RMSE + L2 regularization toward library medians.

    Data term: sqrt(mean((pred_i - actual_i)^2 / item_value_i^2))
    Reg term: sqrt(mean(((x_j - median_j) / std_j)^2))
    """
    params = build_agent_params(x, free_names)

    # Data fit: relative squared error
    sse = 0.0
    for item in ITEM_TYPES:
        predicted = compute_valuation(params, item, condition)
        actual = actual_vals[item]
        rel_err = (predicted - actual) / ITEM_VALUES[item]
        sse += rel_err ** 2
    data_term = math.sqrt(sse / len(ITEM_TYPES))

    # Regularization: L2 toward medians, normalized by std
    reg_sum = 0.0
    for name, val in zip(free_names, x):
        if name in LIBRARY_STDS:
            reg_sum += ((val - LIBRARY_MEDIANS[name]) / LIBRARY_STDS[name]) ** 2
    n_reg = max(1, sum(1 for n in free_names if n in LIBRARY_STDS))
    reg_term = math.sqrt(reg_sum / n_reg)

    return data_term + REG_LAMBDA * reg_term


# ================================================================
# FITTING
# ================================================================

def fit_subject(condition, actual_vals, seed=42):
    """Fit VCMS parameters to one subject's 3 item valuations."""
    is_buyer = (condition == 'buyer')
    free_names = BUYER_FREE_NAMES if is_buyer else SELLER_FREE_NAMES
    bounds = BUYER_BOUNDS if is_buyer else SELLER_BOUNDS

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(free_names, condition, actual_vals),
        maxiter=300,
        popsize=15,
        tol=1e-6,
        seed=seed,
        polish=True,
    )

    params = build_agent_params(result.x, free_names)
    return params, result.fun


def fit_all_subjects(subjects):
    """Fit all Colucci subjects, return fitted library dict."""
    library = {}
    n = len(subjects)
    t0 = time.time()

    for i, (pid, rec) in enumerate(subjects.items()):
        params, rmse = fit_subject(rec['condition'], rec['valuations'], seed=42+i)
        v3d = agent_params_to_v3_dict(params)

        # Store predicted valuations
        preds = {}
        for item in ITEM_TYPES:
            preds[item] = compute_valuation(params, item, rec['condition'])

        library[pid] = {
            'condition': rec['condition'],
            'actual_valuations': rec['valuations'],
            'predicted_valuations': preds,
            'fit_rmse': round(rmse, 6),
            'v3_params': v3d,
            'sex': rec.get('sex', ''),
            'age': rec.get('age', ''),
            'source': 'endowment_fit',
            'engine': 'endowment_adapter_v2',
        }

        if (i + 1) % 50 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  Fitted {i+1}/{n} subjects "
                  f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    return library


# ================================================================
# AUTONOMOUS SIMULATION (two-pass)
# ================================================================
# The VCMS engine uses teacher forcing: rd.contribution drives state
# updates, pred_contrib is the prediction. For autonomous simulation
# (no real data), we run two passes:
#   Pass 1: contribution = c_base * max_c  (rough estimate)
#   Pass 2: contribution = Pass 1 predictions (self-consistent)
# This gives realistic internal state evolution.


def _build_pgg_rounds(contributions, n_rounds, others_mean, max_c):
    """Build PGG round objects from a contribution sequence."""
    rounds = []
    for t in range(n_rounds):
        c = contributions[t] if t < len(contributions) else max_c * 0.5
        rounds.append(SimpleRound(
            contribution=c,
            others_mean=others_mean,
            pun_sent=0,
            pun_recv=0,
        ))
    return rounds


def simulate_pgg(v3_dict, n_rounds=10, others_mean_frac=0.5):
    """
    Autonomous two-pass PGG simulation.
    Returns contribution trajectory and average contribution.
    """
    gc = PGG_P_CONFIG
    max_c = gc.max_contrib  # 20
    others_mean = others_mean_frac * max_c
    v4p = v3_dict_to_vcms_params(v3_dict)

    # Pass 1: estimate with c_base
    est_c = round(v3_dict['c_base'] * max_c)
    rounds1 = _build_pgg_rounds([est_c] * n_rounds, n_rounds, others_mean, max_c)
    result1 = run_vcms_v4(v4p, rounds1, gc)

    # Pass 2: use Pass 1 predictions as teacher-forced contributions
    rounds2 = _build_pgg_rounds(result1['pred_contrib'], n_rounds, others_mean, max_c)
    result2 = run_vcms_v4(v4p, rounds2, gc)

    contribs = result2['pred_contrib']
    avg_c = np.mean(contribs)
    return contribs, avg_c


def classify_pgg_phenotype(contribs, c_base, inertia, alpha,
                           b_depletion_rate, b_replenish_rate, max_c=20):
    """
    Classify PGG behavioral phenotype from parameters + simulated trajectory.

    Criteria from build_phenotype_pools:
      CC: c_base > 0.65, inertia > 0.3, high contribution
      EC: 0.35 <= c_base <= 0.75, inertia < 0.25, alpha > 0.3
      CD: c_base < 0.4, low contribution
      DL: c_base > 0.55, declining trajectory, b_depletion > b_replenish
    """
    avg_c = np.mean(contribs)
    frac = avg_c / max_c

    # Check for declining trajectory (slope < -0.3 per round in normalized units)
    if len(contribs) >= 5:
        first_half = np.mean(contribs[:len(contribs)//2])
        second_half = np.mean(contribs[len(contribs)//2:])
        declining = (first_half - second_half) / max_c > 0.05
    else:
        declining = False

    if c_base > 0.65 and inertia > 0.3 and frac > 0.5:
        return 'CC'
    if 0.35 <= c_base <= 0.75 and inertia < 0.25 and alpha > 0.3:
        return 'EC'
    if c_base < 0.4:
        return 'CD'
    if c_base > 0.55 and declining and b_depletion_rate > b_replenish_rate:
        return 'DL'
    if frac > 0.5:
        return 'high-other'
    return 'low-other'


# ================================================================
# IPD SIMULATION
# ================================================================

IPD_CONFIG = GameConfig(
    max_contrib=1, max_punish=1, has_punishment=False,
    n_signals=2, normalized_time=False,
)


def simulate_ipd(v3_dict, n_rounds=50, partner_coop_rate=0.5):
    """
    Autonomous two-pass IPD simulation.
    Returns cooperation trajectory and cooperation rate.
    """
    rng = np.random.RandomState(42)
    partner_choices = [1 if rng.random() < partner_coop_rate else 0
                       for _ in range(n_rounds)]
    v4p = v3_dict_to_vcms_params(v3_dict)

    # Pass 1: estimate with c_base
    est_c = 1 if v3_dict['c_base'] > 0.5 else 0
    rounds1 = []
    for t in range(n_rounds):
        rounds1.append(SimpleRound(
            contribution=est_c,
            others_mean=partner_choices[t],
            pun_sent=0, pun_recv=0,
        ))
    result1 = run_vcms_v4(v4p, rounds1, IPD_CONFIG)

    # Pass 2: use Pass 1 predictions
    rounds2 = []
    for t in range(n_rounds):
        c = result1['pred_contrib'][t]
        rounds2.append(SimpleRound(
            contribution=c,
            others_mean=partner_choices[t],
            pun_sent=0, pun_recv=0,
        ))
    result2 = run_vcms_v4(v4p, rounds2, IPD_CONFIG)

    choices = result2['pred_contrib']
    coop_rate = np.mean(choices)
    return choices, coop_rate


def classify_ipd_type(coop_rate):
    """Classify IPD behavioral type from cooperation rate."""
    if coop_rate > 0.6:
        return 'mostly-C'
    elif coop_rate > 0.3:
        return 'mixed'
    else:
        return 'mostly-D'


# ================================================================
# REFERENCE LIBRARY PHENOTYPE DISTRIBUTION
# ================================================================

def get_library_distributions():
    """Compute phenotype distributions from existing fitted libraries."""
    # PGG-P library
    with open('v3_library_fitted.json') as f:
        p_lib = json.load(f)

    p_profiles = Counter()
    for sid, rec in p_lib.items():
        p_profiles[rec['behavioral_profile']] += 1

    # IPD library
    with open('v4_ipd_library_fitted.json') as f:
        ipd_lib = json.load(f)

    ipd_types = Counter()
    for sid, rec in ipd_lib.items():
        ipd_types[rec.get('subject_type', 'unknown')] += 1

    # Also compute parameter-based phenotype classification for PGG
    pgg_phenotypes = Counter()
    for sid, rec in p_lib.items():
        p = rec['v3_params']
        label = rec['behavioral_profile']
        if label in ('cooperator', 'cooperative-enforcer') and p['c_base'] > 0.65 and p['inertia'] > 0.3:
            pgg_phenotypes['CC'] += 1
        elif 0.35 <= p['c_base'] <= 0.75 and p['inertia'] < 0.25 and p['alpha'] > 0.3:
            pgg_phenotypes['EC'] += 1
        elif label in ('free-rider', 'punitive-free-rider', 'antisocial-controller') and p['c_base'] < 0.4:
            pgg_phenotypes['CD'] += 1
        else:
            pgg_phenotypes['other'] += 1

    # Simulate PGG cooperation levels for reference library
    pgg_coop_levels = Counter()
    pgg_avg_cs = []
    for sid, rec in p_lib.items():
        contribs, avg_c = simulate_pgg(rec['v3_params'])
        pgg_avg_cs.append(avg_c)
        if avg_c / 20 > 0.6:
            pgg_coop_levels['high'] += 1
        elif avg_c / 20 >= 0.3:
            pgg_coop_levels['mid'] += 1
        else:
            pgg_coop_levels['low'] += 1

    # Simulate IPD cooperation rates for reference library
    ipd_coop_rates = []
    ipd_sim_types = Counter()
    for sid, rec in ipd_lib.items():
        choices, coop_rate = simulate_ipd(rec['v3_params'])
        ipd_coop_rates.append(coop_rate)
        ipd_sim_types[classify_ipd_type(coop_rate)] += 1

    return {
        'pgg_profiles': dict(p_profiles),
        'pgg_phenotypes': dict(pgg_phenotypes),
        'pgg_coop_levels': dict(pgg_coop_levels),
        'pgg_avg_cs': pgg_avg_cs,
        'ipd_types': dict(ipd_types),
        'ipd_sim_types': dict(ipd_sim_types),
        'ipd_coop_rates': ipd_coop_rates,
    }


# ================================================================
# REPORTING
# ================================================================

def print_fit_summary(library):
    """Print fitting quality summary."""
    print("\n" + "=" * 72)
    print("  FITTING SUMMARY")
    print("=" * 72)

    by_cond = defaultdict(list)
    for pid, rec in library.items():
        by_cond[rec['condition']].append(rec)

    print(f"\n  Total subjects fitted: {len(library)}")
    print(f"\n  {'Condition':<10s} {'N':>5s} {'RMSE mean':>10s} {'RMSE med':>10s}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*10}")

    all_rmses = []
    for cond in ['buyer', 'now', 'day', 'week', 'month']:
        recs = by_cond[cond]
        if not recs:
            continue
        rmses = [r['fit_rmse'] for r in recs]
        all_rmses.extend(rmses)
        print(f"  {cond:<10s} {len(recs):>5d} {np.mean(rmses):>10.4f} {np.median(rmses):>10.4f}")

    print(f"  {'ALL':<10s} {len(all_rmses):>5d} {np.mean(all_rmses):>10.4f} {np.median(all_rmses):>10.4f}")

    # Parameter distributions
    print(f"\n  Fitted parameter distributions:")
    print(f"  {'Parameter':<20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Lib Med':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    param_names = ['alpha', 'v_rep', 'v_ref', 'c_base', 'inertia',
                   's_rate', 's_initial', 'facilitation_rate',
                   'b_initial', 'b_replenish_rate']
    for pname in param_names:
        vals = [rec['v3_params'][pname] for rec in library.values()]
        lib_med = LIBRARY_MEDIANS.get(pname, 0)
        print(f"  {pname:<20s} {np.mean(vals):>8.3f} {np.median(vals):>8.3f} "
              f"{np.std(vals):>8.3f} {lib_med:>8.3f}")

    # Valuation accuracy
    print(f"\n  Valuation fit quality:")
    print(f"  {'Item':<10s} {'MAE ($)':>10s} {'MAE (%)':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")

    for item in ITEM_TYPES:
        errors = []
        rel_errors = []
        for rec in library.values():
            actual = rec['actual_valuations'][item]
            predicted = rec['predicted_valuations'][item]
            errors.append(abs(predicted - actual))
            rel_errors.append(abs(predicted - actual) / ITEM_VALUES[item])
        print(f"  {item:<10s} {np.mean(errors):>10.2f} {np.mean(rel_errors)*100:>9.1f}%")


def print_phenotype_comparison(library, ref_dists):
    """Run PGG/IPD simulations and compare phenotype distributions."""
    print("\n" + "=" * 72)
    print("  PHENOTYPE VERIFICATION: Endowment-fitted → PGG/IPD")
    print("=" * 72)

    # --- PGG Phenotype Classification ---
    print("\n  Running PGG simulation (autonomous two-pass) for all subjects...")
    pgg_phenotypes = Counter()
    pgg_avg_cs = []

    for pid, rec in library.items():
        contribs, avg_c = simulate_pgg(rec['v3_params'])
        p = rec['v3_params']
        phenotype = classify_pgg_phenotype(
            contribs, p['c_base'], p['inertia'], p['alpha'],
            p['b_depletion_rate'], p['b_replenish_rate'])
        pgg_phenotypes[phenotype] += 1
        pgg_avg_cs.append(avg_c)

    n_total = len(library)
    print(f"\n  PGG Phenotype Distribution (endowment-fitted, n={n_total}):")
    for ph in ['CC', 'EC', 'CD', 'DL', 'high-other', 'low-other']:
        count = pgg_phenotypes.get(ph, 0)
        pct = 100 * count / n_total if n_total > 0 else 0
        print(f"    {ph:<15s}: {count:>4d} ({pct:>5.1f}%)")

    print(f"\n  PGG avg contribution: mean={np.mean(pgg_avg_cs):.1f}/20, "
          f"median={np.median(pgg_avg_cs):.1f}/20")

    # Cooperation level distribution (unified)
    high = sum(1 for c in pgg_avg_cs if c / 20 > 0.6)
    mid = sum(1 for c in pgg_avg_cs if 0.3 <= c / 20 <= 0.6)
    low = sum(1 for c in pgg_avg_cs if c / 20 < 0.3)
    print(f"\n  PGG Cooperation Level (endowment-fitted):")
    print(f"    high (>60%):  {high:>4d} ({100*high/n_total:>5.1f}%)")
    print(f"    mid (30-60%): {mid:>4d} ({100*mid/n_total:>5.1f}%)")
    print(f"    low (<30%):   {low:>4d} ({100*low/n_total:>5.1f}%)")

    # Reference library phenotype dist
    ref_pgg = ref_dists['pgg_phenotypes']
    ref_total = sum(ref_pgg.values())
    print(f"\n  Reference PGG Phenotype Distribution (P-library, n={ref_total}):")
    for ph in ['CC', 'EC', 'CD', 'other']:
        count = ref_pgg.get(ph, 0)
        pct = 100 * count / ref_total if ref_total > 0 else 0
        print(f"    {ph:<15s}: {count:>4d} ({pct:>5.1f}%)")

    # Reference cooperation level (simulated)
    ref_coop = ref_dists['pgg_coop_levels']
    ref_avg = ref_dists['pgg_avg_cs']
    print(f"\n  Reference PGG Cooperation Level (simulated, n={ref_total}):")
    for level in ['high', 'mid', 'low']:
        count = ref_coop.get(level, 0)
        pct = 100 * count / ref_total if ref_total > 0 else 0
        print(f"    {level:<15s}: {count:>4d} ({pct:>5.1f}%)")
    print(f"    avg contribution: {np.mean(ref_avg):.1f}/20")

    # --- IPD Classification ---
    print(f"\n  Running IPD simulation for all fitted subjects...")
    ipd_types = Counter()
    ipd_coop_rates = []

    for pid, rec in library.items():
        choices, coop_rate = simulate_ipd(rec['v3_params'])
        ipd_type = classify_ipd_type(coop_rate)
        ipd_types[ipd_type] += 1
        ipd_coop_rates.append(coop_rate)

    print(f"\n  IPD Type Distribution (endowment-fitted, n={n_total}):")
    for t in ['mostly-C', 'mixed', 'mostly-D']:
        count = ipd_types.get(t, 0)
        pct = 100 * count / n_total if n_total > 0 else 0
        print(f"    {t:<15s}: {count:>4d} ({pct:>5.1f}%)")

    print(f"\n  IPD cooperation rate: mean={np.mean(ipd_coop_rates):.3f}, "
          f"median={np.median(ipd_coop_rates):.3f}")

    # Reference IPD dist (from library labels)
    ref_ipd = ref_dists['ipd_types']
    ref_ipd_total = sum(ref_ipd.values())
    print(f"\n  Reference IPD Type Distribution (library labels, n={ref_ipd_total}):")
    for t in ['mostly-C', 'mixed', 'mostly-D']:
        count = ref_ipd.get(t, 0)
        pct = 100 * count / ref_ipd_total if ref_ipd_total > 0 else 0
        print(f"    {t:<15s}: {count:>4d} ({pct:>5.1f}%)")

    # Reference IPD dist (simulated with same method)
    ref_ipd_sim = ref_dists['ipd_sim_types']
    ref_ipd_rates = ref_dists['ipd_coop_rates']
    print(f"\n  Reference IPD Type Distribution (simulated, n={ref_ipd_total}):")
    for t in ['mostly-C', 'mixed', 'mostly-D']:
        count = ref_ipd_sim.get(t, 0)
        pct = 100 * count / ref_ipd_total if ref_ipd_total > 0 else 0
        print(f"    {t:<15s}: {count:>4d} ({pct:>5.1f}%)")
    print(f"    avg coop rate: {np.mean(ref_ipd_rates):.3f}")

    # --- By endowment condition ---
    print(f"\n  PGG Cooperation Level by Endowment Condition:")
    print(f"  {'Condition':<10s} {'N':>5s} {'high':>6s} {'mid':>6s} {'low':>6s} {'avg_c':>8s}")
    print(f"  {'-'*10} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")

    by_cond = defaultdict(list)
    for pid, rec in library.items():
        by_cond[rec['condition']].append(pid)

    for cond in ['buyer', 'now', 'day', 'week', 'month']:
        pids = by_cond[cond]
        if not pids:
            continue
        cond_avg_cs = []
        for pid in pids:
            p = library[pid]['v3_params']
            contribs, avg_c = simulate_pgg(p)
            cond_avg_cs.append(avg_c)

        n = len(pids)
        h = sum(1 for c in cond_avg_cs if c / 20 > 0.6)
        m = sum(1 for c in cond_avg_cs if 0.3 <= c / 20 <= 0.6)
        lo = sum(1 for c in cond_avg_cs if c / 20 < 0.3)
        print(f"  {cond:<10s} {n:>5d} {100*h/n:>5.1f}% {100*m/n:>5.1f}% "
              f"{100*lo/n:>5.1f}% {np.mean(cond_avg_cs):>7.1f}")


# ================================================================
# MAIN
# ================================================================

def main():
    import os

    print("=" * 72)
    print("  COLUCCI ENDOWMENT FIT → VCMS PARAMETER EXTRACTION")
    print("=" * 72)

    # Check if we can skip fitting (library already exists)
    skip_fit = '--sim-only' in sys.argv

    if skip_fit and os.path.exists(OUTPUT_PATH):
        print(f"\n  Loading previously fitted library from {OUTPUT_PATH}...")
        with open(OUTPUT_PATH) as f:
            library = json.load(f)
        print(f"  Loaded {len(library)} fitted subjects")
    else:
        # Phase 1: Load data
        print("\n  Loading Colucci dataset...")
        subjects = load_colucci_data()
        print(f"  Loaded {len(subjects)} valid subjects")
        cond_counts = Counter(s['condition'] for s in subjects.values())
        for cond in ['buyer', 'now', 'day', 'week', 'month']:
            print(f"    {cond}: {cond_counts.get(cond, 0)}")

        # Phase 2: Fit
        print(f"\n  Fitting VCMS parameters (DE, sellers={len(SELLER_FREE_NAMES)} free, "
              f"buyers={len(BUYER_FREE_NAMES)} free, reg_lambda={REG_LAMBDA})...")
        library = fit_all_subjects(subjects)

        # Phase 3: Save
        def sanitize(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(sanitize(library), f, indent=2)
        print(f"\n  Saved fitted library to {OUTPUT_PATH}")

    # Phase 4: Fit summary
    print_fit_summary(library)

    # Phase 5: Phenotype verification
    print("\n  Loading reference library distributions (simulating PGG/IPD)...")
    ref_dists = get_library_distributions()
    print_phenotype_comparison(library, ref_dists)

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == '__main__':
    main()
