#!/usr/bin/env python3
"""
VCMS Behavioral Economics Adapter — Endowment Effect (Phase 1: Transfer Test)
=============================================================================

Tests whether VCMS parameters fitted on cooperation games predict endowment
effect valuations (WTA/WTP gaps) with zero retuning. The endowment effect is
modeled as a strain phenomenon from topology change: losing a possessed item
triggers the same V→S→B cascade that drives cooperation dynamics.

Phase 1: Transfer Test
  - 576 agents from fitted library (P:176, N:212, IPD:188)
  - 3 item types: mug, Amazon gift card, Spotify subscription
  - 5 conditions: buyer, seller-now, seller-day, seller-week, seller-month
  - 8,640 total valuations (576 × 3 × 5)
  - 6 predictions (PT1-PT6) evaluated against CFV benchmarks

All adapter constants are fixed a priori — no fitting, no free parameters
beyond the already-fitted VCMS cooperation parameters.

Reference: Colucci, Franco & Valori (2024), N=516, MIT license.
"""

import json
import math
import time
import numpy as np
from collections import defaultdict

from federation_sim import (
    load_libraries, extract_agent_params, build_phenotype_pools,
    AgentParams,
)
from phenotype_geometry import unified_label


# ================================================================
# ADAPTER CONSTANTS (fixed a priori — not fitted)
# ================================================================

# Item salience: how strongly the item anchors into V-channel reference
# Mug = tangible physical object (high), Amazon = abstract monetary (low),
# Spotify = experiential digital (medium)
ITEM_SALIENCE = {"mug": 0.8, "amazon": 0.4, "spotify": 0.6}

# Item value anchors (from CFV buyer WTP means, in dollars)
ITEM_VALUES = {"mug": 3.86, "amazon": 78.38, "spotify": 12.24}

# Seller possession-duration multipliers (V-channel integration timescale)
# Longer possession → deeper integration into reference topology
DURATION_MULT = {"now": 1.0, "day": 2.0, "week": 3.5, "month": 5.0}

# Facilitation multipliers (M_eval competes with integration over time)
# Zero at "now" (no experience-based adaptation at instant of endowment).
# Very gentle sublinear growth: fac/dur ratio stays below 0.15,
# ensuring effective integration is strictly monotonic with duration.
# Fac/dur ratios: 0.00, 0.075, 0.114, 0.140.
FACILITATION_MULT = {"now": 0.0, "day": 0.15, "week": 0.40, "month": 0.70}

# Scaling constants
BUYER_COST_SCALE = 0.15       # buyer discount from (1 - c_base)
BUYER_INERTIA_SCALE = 0.1     # buyer discount from inertia
SELLER_INERTIA_AMP = 0.5      # seller inertia amplification of premium
STRAIN_S_INITIAL_SCALE = 5.0  # s_initial normalization denominator

# Conditions and items
SELLER_CONDITIONS = ["now", "day", "week", "month"]
ALL_CONDITIONS = ["buyer"] + SELLER_CONDITIONS
ITEM_TYPES = ["mug", "amazon", "spotify"]

# Game tag → phenotype_geometry game label
GAME_TAG_MAP = {"P": "PGG-P", "N": "PGG-N", "IPD": "IPD"}


# ================================================================
# CORE ADAPTER FUNCTION
# ================================================================

def compute_valuation(params, item_type, condition):
    """
    Map VCMS cooperation parameters to endowment effect valuation.

    Args:
        params: AgentParams with alpha, c_base, inertia, s_initial,
                s_rate, facilitation_rate
        item_type: "mug", "amazon", or "spotify"
        condition: "buyer", "now", "day", "week", or "month"

    Returns:
        Valuation in dollars (WTP for buyers, WTA for sellers).

    Channel mapping:
        V → Reference integration (alpha × salience × duration)
        C → Topology change cost ((1 - c_base) × integration × value)
        M → Facilitation (facilitation_rate competes with integration)
        S → Strain premium (s_rate × integration × (1 + s_initial/5))
        Inertia → Possession attachment (amplifies seller premium)
    """
    item_value = ITEM_VALUES[item_type]
    item_salience = ITEM_SALIENCE[item_type]

    if condition == "buyer":
        # Buyer: WTP = value minus discounts from topology change resistance
        buyer_discount = (1.0 - params.c_base) * BUYER_COST_SCALE * item_value
        inertia_discount = item_value * params.inertia * BUYER_INERTIA_SCALE
        wtp = item_value - buyer_discount - inertia_discount
        return max(0.0, wtp)

    # Seller: WTA = value plus premium from integration/strain/inertia
    duration_mult = DURATION_MULT[condition]
    fac_mult = FACILITATION_MULT[condition]

    # V-channel: reference integration depth
    V_integration = params.alpha * item_salience * duration_mult

    # M-channel: facilitation competes with integration
    facilitation = params.facilitation_rate * fac_mult
    effective_integration = max(0.0, V_integration - facilitation)

    # C-channel: topology change cost
    change_cost = (1.0 - params.c_base) * effective_integration * item_value

    # S-channel: strain premium from anticipated loss
    strain_premium = (params.s_rate * effective_integration
                      * (1.0 + params.s_initial / STRAIN_S_INITIAL_SCALE))

    # Inertia: amplifies the total premium for sellers
    inertia_amp = 1.0 + params.inertia * SELLER_INERTIA_AMP

    wta = item_value + (change_cost + strain_premium) * inertia_amp
    return wta


# ================================================================
# LIBRARY LOADING
# ================================================================

def load_all_agents():
    """
    Load all 576 agents from the three v4 fitted libraries.

    Returns:
        agents: list of dicts with sid, game, phenotype, unified, params
        pool_map: dict mapping sid → pool phenotype (CC/EC/CD/DL) for
                  agents that qualify; others absent
    """
    libs = load_libraries()

    agents = []
    for tag, lib in libs.items():
        game_label = GAME_TAG_MAP[tag]
        for sid, rec in lib.items():
            phenotype = rec.get('behavioral_profile',
                                rec.get('subject_type', 'unknown'))
            params = extract_agent_params(rec['v3_params'])
            unified = unified_label(game_label, phenotype)
            agents.append({
                "sid": sid,
                "game": tag,
                "phenotype": phenotype,
                "unified": unified,
                "params": params,
            })

    # Build pool mapping (CC/EC/CD/DL)
    pools = build_phenotype_pools(libs)
    pool_map = {}
    for pool_name, members in pools.items():
        for (tag, sid, ap) in members:
            # An agent can appear in multiple pools; keep the first match
            if sid not in pool_map:
                pool_map[sid] = pool_name

    return agents, pool_map


# ================================================================
# TRANSFER TEST
# ================================================================

def run_transfer_test(agents, pool_map):
    """
    Run all 576 agents through the adapter for all items and conditions.

    Returns:
        results: dict[sid] → {game, phenotype, unified, pool,
                               valuations: {item: {condition: $}},
                               wta_wtp: {item: ratio}}
    """
    results = {}

    for agent in agents:
        sid = agent["sid"]
        params = agent["params"]
        valuations = {}
        wta_wtp = {}

        for item in ITEM_TYPES:
            vals = {}
            for cond in ALL_CONDITIONS:
                vals[cond] = compute_valuation(params, item, cond)
            valuations[item] = vals

            # WTA/WTP ratio: seller-now vs buyer
            wtp = vals["buyer"]
            wta = vals["now"]
            wta_wtp[item] = wta / wtp if wtp > 0 else float('inf')

        results[sid] = {
            "game": agent["game"],
            "phenotype": agent["phenotype"],
            "unified": agent["unified"],
            "pool": pool_map.get(sid),
            "valuations": valuations,
            "wta_wtp": wta_wtp,
        }

    return results


# ================================================================
# ANALYSIS
# ================================================================

def compute_aggregate_metrics(results):
    """Compute all aggregate metrics from transfer test results."""
    metrics = {}

    # --- Per-item aggregates ---
    for item in ITEM_TYPES:
        wtps = []
        wtas_by_cond = {c: [] for c in SELLER_CONDITIONS}
        ratios = []

        for r in results.values():
            wtps.append(r["valuations"][item]["buyer"])
            for c in SELLER_CONDITIONS:
                wtas_by_cond[c].append(r["valuations"][item][c])
            ratios.append(r["wta_wtp"][item])

        metrics[item] = {
            "wtp_mean": np.mean(wtps),
            "wtp_std": np.std(wtps),
            "wtp_median": np.median(wtps),
        }
        for c in SELLER_CONDITIONS:
            metrics[item][f"wta_{c}_mean"] = np.mean(wtas_by_cond[c])
            metrics[item][f"wta_{c}_std"] = np.std(wtas_by_cond[c])
            metrics[item][f"wta_{c}_median"] = np.median(wtas_by_cond[c])

        metrics[item]["ratio_mean"] = np.mean(ratios)
        metrics[item]["ratio_std"] = np.std(ratios)
        metrics[item]["ratio_median"] = np.median(ratios)
        metrics[item]["ee_frac"] = np.mean([r > 1.05 for r in ratios])

    # --- Per-phenotype (unified: high/mid/low) ---
    for label in ["high", "mid", "low"]:
        subset = [r for r in results.values() if r["unified"] == label]
        if not subset:
            continue
        for item in ITEM_TYPES:
            ratios = [r["wta_wtp"][item] for r in subset]
            key = f"unified_{label}_{item}"
            metrics[key] = {
                "ratio_mean": np.mean(ratios),
                "ratio_std": np.std(ratios),
                "n": len(subset),
            }

    # --- Per-phenotype (pool: CC/EC/CD/DL) ---
    for pool in ["CC", "EC", "CD", "DL"]:
        subset = [r for r in results.values() if r["pool"] == pool]
        if not subset:
            continue
        for item in ITEM_TYPES:
            ratios = [r["wta_wtp"][item] for r in subset]
            key = f"pool_{pool}_{item}"
            metrics[key] = {
                "ratio_mean": np.mean(ratios),
                "ratio_std": np.std(ratios),
                "n": len(subset),
            }
        # Time profile for this pool
        for item in ITEM_TYPES:
            wta_by_cond = {}
            for c in SELLER_CONDITIONS:
                vals = [r["valuations"][item][c] for r in subset]
                wta_by_cond[c] = np.mean(vals)
            metrics[f"pool_{pool}_{item}_time"] = wta_by_cond
            # Time effect: month/now ratio
            if wta_by_cond["now"] > 0:
                metrics[f"pool_{pool}_{item}_time_ratio"] = (
                    wta_by_cond["month"] / wta_by_cond["now"]
                )

    # --- Time profiles (all agents) ---
    for item in ITEM_TYPES:
        wta_by_cond = {}
        for c in SELLER_CONDITIONS:
            vals = [r["valuations"][item][c] for r in results.values()]
            wta_by_cond[c] = np.mean(vals)
        metrics[f"time_profile_{item}"] = wta_by_cond

    # --- Non-monotonicity ---
    n_nonmono = 0
    for r in results.values():
        for item in ITEM_TYPES:
            if r["valuations"][item]["month"] < r["valuations"][item]["week"]:
                n_nonmono += 1
                break  # count agent once
    metrics["nonmono_frac"] = n_nonmono / len(results)

    # --- No-EE fraction (mug, WTA/WTP < 1.1) ---
    mug_ratios = [r["wta_wtp"]["mug"] for r in results.values()]
    metrics["no_ee_frac_mug"] = np.mean([r < 1.1 for r in mug_ratios])

    # --- Distribution statistics ---
    for item in ITEM_TYPES:
        ratios = [r["wta_wtp"][item] for r in results.values()]
        ratios_arr = np.array(ratios)
        metrics[f"dist_{item}"] = {
            "mean": np.mean(ratios_arr),
            "std": np.std(ratios_arr),
            "median": np.median(ratios_arr),
            "p10": np.percentile(ratios_arr, 10),
            "p25": np.percentile(ratios_arr, 25),
            "p75": np.percentile(ratios_arr, 75),
            "p90": np.percentile(ratios_arr, 90),
            "skew": float(np.mean(((ratios_arr - np.mean(ratios_arr))
                                    / max(np.std(ratios_arr), 1e-9)) ** 3)),
            "frac_lt_1.05": np.mean(ratios_arr < 1.05),
            "frac_lt_1.1": np.mean(ratios_arr < 1.1),
            "frac_gt_2.0": np.mean(ratios_arr > 2.0),
            "frac_gt_3.0": np.mean(ratios_arr > 3.0),
        }

    # --- Parameter sensitivity (correlation with mug WTA/WTP) ---
    param_names = ["alpha", "c_base", "inertia", "s_initial",
                   "s_rate", "facilitation_rate"]
    mug_ratios = np.array([r["wta_wtp"]["mug"] for r in results.values()])
    agents_list = list(results.values())
    # Need params; reconstruct from results ordering
    # Actually we need the agents list — pass it separately or embed
    # For now, store raw ratios for later correlation
    metrics["mug_ratios"] = mug_ratios

    return metrics


def compute_parameter_correlations(agents, results):
    """Compute correlation of each VCMS parameter with WTA/WTP ratios."""
    param_names = ["alpha", "c_base", "inertia", "s_initial",
                   "s_rate", "facilitation_rate"]
    correlations = {}

    for item in ITEM_TYPES:
        ratios = []
        param_vals = {p: [] for p in param_names}
        for agent in agents:
            sid = agent["sid"]
            if sid in results:
                ratios.append(results[sid]["wta_wtp"][item])
                for p in param_names:
                    param_vals[p].append(getattr(agent["params"], p))

        ratios = np.array(ratios)
        for p in param_names:
            pv = np.array(param_vals[p])
            if np.std(pv) > 1e-9 and np.std(ratios) > 1e-9:
                corr = np.corrcoef(pv, ratios)[0, 1]
            else:
                corr = 0.0
            correlations[f"{item}_{p}"] = corr

    return correlations


# ================================================================
# CFV BENCHMARKS
# ================================================================

def get_cfv_benchmarks():
    """
    Return CFV (Colucci, Franco & Valori 2024) benchmark ranges.

    These are approximate target ranges from the published study.
    Phase 1 asks whether the model's predictions fall in plausible
    territory, not whether they match exactly.
    """
    return {
        "mug_wta_wtp_range": (1.5, 4.0),
        "amazon_wta_wtp_range": (1.0, 2.0),
        "spotify_wta_wtp_range": (1.0, 3.0),
        "time_increases": True,  # WTA generally increases with duration
        "distribution_right_skewed": True,
        "substantial_mass_near_1": True,  # some agents show no EE
    }


# ================================================================
# PREDICTIONS
# ================================================================

def evaluate_predictions(metrics, results):
    """
    Evaluate 6 pre-registered predictions.

    Returns list of dicts: {id, description, criterion, actual, supported}
    """
    predictions = []

    # PT1: Aggregate endowment effect exists
    mug_ratio = metrics["mug"]["ratio_mean"]
    pt1 = {
        "id": "PT1",
        "description": "Aggregate EE exists: mean WTA/WTP(mug) > 1.5",
        "tolerance": "[1.2, 4.0]",
        "actual": f"{mug_ratio:.3f}",
        "supported": mug_ratio > 1.5,
        "in_tolerance": 1.2 <= mug_ratio <= 4.0,
    }
    predictions.append(pt1)

    # PT2: CD shows highest endowment effect
    cd_key = "pool_CD_mug"
    cc_key = "pool_CC_mug"
    ec_key = "pool_EC_mug"
    cd_ratio = metrics.get(cd_key, {}).get("ratio_mean", 0)
    cc_ratio = metrics.get(cc_key, {}).get("ratio_mean", 0)
    ec_ratio = metrics.get(ec_key, {}).get("ratio_mean", 0)
    pt2 = {
        "id": "PT2",
        "description": "CD highest EE: CD WTA/WTP > CC and EC",
        "tolerance": "CD > max(CC, EC)",
        "actual": f"CD={cd_ratio:.3f}, CC={cc_ratio:.3f}, EC={ec_ratio:.3f}",
        "supported": cd_ratio > cc_ratio and cd_ratio > ec_ratio,
    }
    predictions.append(pt2)

    # PT3: EC weakest time effect
    ec_time = metrics.get("pool_EC_mug_time_ratio", 0)
    cc_time = metrics.get("pool_CC_mug_time_ratio", 0)
    cd_time = metrics.get("pool_CD_mug_time_ratio", 0)
    pt3 = {
        "id": "PT3",
        "description": "EC weakest time effect: EC month/now < CC and CD",
        "tolerance": "EC < min(CC, CD)",
        "actual": f"EC={ec_time:.3f}, CC={cc_time:.3f}, CD={cd_time:.3f}",
        "supported": ec_time < cc_time and ec_time < cd_time,
    }
    predictions.append(pt3)

    # PT4: 20-40% of agents show no endowment effect
    no_ee = metrics["no_ee_frac_mug"]
    pt4 = {
        "id": "PT4",
        "description": "No-EE fraction: 20-40% have WTA/WTP(mug) < 1.1",
        "tolerance": "[0.10, 0.50]",
        "actual": f"{no_ee:.1%}",
        "supported": 0.20 <= no_ee <= 0.40,
        "in_tolerance": 0.10 <= no_ee <= 0.50,
    }
    predictions.append(pt4)

    # PT5: Item salience ordering
    mug_r = metrics["mug"]["ratio_mean"]
    spotify_r = metrics["spotify"]["ratio_mean"]
    amazon_r = metrics["amazon"]["ratio_mean"]
    pt5 = {
        "id": "PT5",
        "description": "Salience ordering: mug > spotify > amazon",
        "tolerance": "strict ordering",
        "actual": f"mug={mug_r:.3f}, spotify={spotify_r:.3f}, amazon={amazon_r:.3f}",
        "supported": mug_r > spotify_r > amazon_r,
    }
    predictions.append(pt5)

    # PT6: Non-monotonic time profile in >10% of agents
    nonmono = metrics["nonmono_frac"]
    pt6 = {
        "id": "PT6",
        "description": "Non-monotonic: >10% show WTA decrease week→month",
        "tolerance": "[0.05, 0.60]",
        "actual": f"{nonmono:.1%}",
        "supported": nonmono > 0.10,
        "in_tolerance": 0.05 <= nonmono <= 0.60,
    }
    predictions.append(pt6)

    return predictions


# ================================================================
# REPORTING
# ================================================================

def print_header(agents, pool_map):
    """Print transfer test header."""
    print("=" * 78)
    print("  VCMS ENDOWMENT EFFECT ADAPTER — PHASE 1: TRANSFER TEST")
    print("  576 cooperation-fitted agents → endowment valuations (zero retuning)")
    print("=" * 78)

    # Agent counts
    games = defaultdict(int)
    unified_counts = defaultdict(int)
    pool_counts = defaultdict(int)
    for a in agents:
        games[a["game"]] += 1
        unified_counts[a["unified"]] += 1
    for p in pool_map.values():
        pool_counts[p] += 1

    print(f"\n  Agents: {len(agents)} total")
    print(f"    P-experiment: {games['P']:>4d}   "
          f"N-experiment: {games['N']:>4d}   "
          f"IPD: {games['IPD']:>4d}")
    print(f"    Unified — high: {unified_counts['high']:>4d}   "
          f"mid: {unified_counts['mid']:>4d}   "
          f"low: {unified_counts['low']:>4d}")
    in_pool = len(pool_map)
    print(f"    Pools  — CC: {pool_counts.get('CC', 0):>4d}   "
          f"EC: {pool_counts.get('EC', 0):>4d}   "
          f"CD: {pool_counts.get('CD', 0):>4d}   "
          f"DL: {pool_counts.get('DL', 0):>4d}   "
          f"({in_pool} in pool, {len(agents) - in_pool} unassigned)")

    n_valuations = len(agents) * len(ITEM_TYPES) * len(ALL_CONDITIONS)
    print(f"\n  Valuations: {n_valuations:,d} "
          f"({len(agents)} × {len(ITEM_TYPES)} items × {len(ALL_CONDITIONS)} conditions)")
    print(f"  Items: mug (${ITEM_VALUES['mug']:.2f}), "
          f"Amazon (${ITEM_VALUES['amazon']:.2f}), "
          f"Spotify (${ITEM_VALUES['spotify']:.2f})")
    print(f"  Conditions: buyer, seller-now, seller-day, seller-week, seller-month")


def print_main_table(metrics):
    """Print core valuation results table."""
    print(f"\n{'=' * 78}")
    print(f"  VALUATION RESULTS")
    print(f"{'=' * 78}")

    print(f"\n  {'Item':<10s} {'Value($)':>8s} {'WTP($)':>8s} "
          f"{'WTA-now':>8s} {'WTA-day':>8s} {'WTA-wk':>8s} {'WTA-mo':>8s} "
          f"{'WTA/WTP':>8s} {'EE%':>6s}")
    print(f"  {'-' * 74}")

    for item in ITEM_TYPES:
        m = metrics[item]
        label = {"mug": "Mug", "amazon": "Amazon", "spotify": "Spotify"}[item]
        print(f"  {label:<10s} {ITEM_VALUES[item]:>8.2f} {m['wtp_mean']:>8.2f} "
              f"{m['wta_now_mean']:>8.2f} {m['wta_day_mean']:>8.2f} "
              f"{m['wta_week_mean']:>8.2f} {m['wta_month_mean']:>8.2f} "
              f"{m['ratio_mean']:>8.3f} {m['ee_frac']:>5.0%}")

    print(f"\n  WTA/WTP ratio = seller-now / buyer (immediate endowment effect)")
    print(f"  EE% = fraction of agents with WTA/WTP > 1.05")


def print_phenotype_breakdown(metrics):
    """Print WTA/WTP breakdown by phenotype."""
    print(f"\n{'=' * 78}")
    print(f"  PHENOTYPE BREAKDOWN (WTA/WTP ratios)")
    print(f"{'=' * 78}")

    # Unified axis
    print(f"\n  --- Unified cooperation axis ---")
    print(f"  {'Phenotype':<10s}", end="")
    for item in ITEM_TYPES:
        label = {"mug": "Mug", "amazon": "Amazon", "spotify": "Spotify"}[item]
        print(f" {label:>12s}", end="")
    print(f" {'N':>6s}")
    print(f"  {'-' * 52}")

    for label in ["high", "mid", "low"]:
        print(f"  {label:<10s}", end="")
        n = 0
        for item in ITEM_TYPES:
            key = f"unified_{label}_{item}"
            if key in metrics:
                m = metrics[key]
                print(f" {m['ratio_mean']:>12.3f}", end="")
                n = m["n"]
            else:
                print(f" {'—':>12s}", end="")
        print(f" {n:>6d}")

    # Pool phenotypes
    print(f"\n  --- Pool phenotypes (parameter-based) ---")
    print(f"  {'Pool':<10s}", end="")
    for item in ITEM_TYPES:
        label = {"mug": "Mug", "amazon": "Amazon", "spotify": "Spotify"}[item]
        print(f" {label:>12s}", end="")
    print(f" {'N':>6s}")
    print(f"  {'-' * 52}")

    for pool in ["CC", "EC", "CD", "DL"]:
        print(f"  {pool:<10s}", end="")
        n = 0
        for item in ITEM_TYPES:
            key = f"pool_{pool}_{item}"
            if key in metrics:
                m = metrics[key]
                print(f" {m['ratio_mean']:>12.3f}", end="")
                n = m["n"]
            else:
                print(f" {'—':>12s}", end="")
        print(f" {n:>6d}")


def print_time_profiles(metrics):
    """Print WTA time profiles across possession durations."""
    print(f"\n{'=' * 78}")
    print(f"  TIME-DURATION PROFILES (mean WTA by possession duration)")
    print(f"{'=' * 78}")

    # Aggregate profiles
    print(f"\n  --- All agents ---")
    print(f"  {'Item':<10s}", end="")
    for c in SELLER_CONDITIONS:
        print(f" {c:>10s}", end="")
    print(f" {'mo/now':>8s}")
    print(f"  {'-' * 52}")

    for item in ITEM_TYPES:
        label = {"mug": "Mug", "amazon": "Amazon", "spotify": "Spotify"}[item]
        tp = metrics[f"time_profile_{item}"]
        print(f"  {label:<10s}", end="")
        for c in SELLER_CONDITIONS:
            print(f" ${tp[c]:>9.2f}", end="")
        ratio = tp["month"] / tp["now"] if tp["now"] > 0 else 0
        print(f" {ratio:>8.3f}")

    # Per-pool profiles (mug only for compactness)
    print(f"\n  --- Pool phenotypes (mug) ---")
    print(f"  {'Pool':<10s}", end="")
    for c in SELLER_CONDITIONS:
        print(f" {c:>10s}", end="")
    print(f" {'mo/now':>8s}")
    print(f"  {'-' * 52}")

    for pool in ["CC", "EC", "CD", "DL"]:
        key = f"pool_{pool}_mug_time"
        if key not in metrics:
            continue
        tp = metrics[key]
        print(f"  {pool:<10s}", end="")
        for c in SELLER_CONDITIONS:
            print(f" ${tp[c]:>9.2f}", end="")
        ratio = tp["month"] / tp["now"] if tp["now"] > 0 else 0
        print(f" {ratio:>8.3f}")

    nonmono = metrics["nonmono_frac"]
    print(f"\n  Non-monotonic agents (WTA decreases week→month): {nonmono:.1%}")


def print_distribution_analysis(metrics):
    """Print distribution analysis of WTA/WTP ratios."""
    print(f"\n{'=' * 78}")
    print(f"  DISTRIBUTION ANALYSIS (WTA/WTP ratio distributions)")
    print(f"{'=' * 78}")

    for item in ITEM_TYPES:
        label = {"mug": "Mug", "amazon": "Amazon", "spotify": "Spotify"}[item]
        d = metrics[f"dist_{item}"]
        print(f"\n  --- {label} ---")
        print(f"    Mean:   {d['mean']:>8.3f}   Median: {d['median']:>8.3f}   "
              f"Std: {d['std']:>8.3f}   Skew: {d['skew']:>8.3f}")
        print(f"    P10:    {d['p10']:>8.3f}   P25:    {d['p25']:>8.3f}   "
              f"P75: {d['p75']:>8.3f}   P90: {d['p90']:>8.3f}")
        print(f"    <1.05:  {d['frac_lt_1.05']:>7.1%}   <1.1:  {d['frac_lt_1.1']:>7.1%}   "
              f">2.0: {d['frac_gt_2.0']:>7.1%}   >3.0: {d['frac_gt_3.0']:>7.1%}")


def print_parameter_sensitivity(correlations):
    """Print parameter sensitivity analysis."""
    print(f"\n{'=' * 78}")
    print(f"  PARAMETER SENSITIVITY (correlation with WTA/WTP ratio)")
    print(f"{'=' * 78}")

    param_names = ["alpha", "c_base", "inertia", "s_initial",
                   "s_rate", "facilitation_rate"]

    print(f"\n  {'Parameter':<20s}", end="")
    for item in ITEM_TYPES:
        label = {"mug": "Mug", "amazon": "Amz", "spotify": "Sptfy"}[item]
        print(f" {label:>10s}", end="")
    print(f"  {'Channel':>12s}")
    print(f"  {'-' * 64}")

    channel_map = {
        "alpha": "V (Visibility)",
        "c_base": "C (Cost)",
        "inertia": "M (Memory)",
        "s_initial": "S (Strain)",
        "s_rate": "S (Strain)",
        "facilitation_rate": "M (Facilit.)",
    }

    for p in param_names:
        print(f"  {p:<20s}", end="")
        for item in ITEM_TYPES:
            key = f"{item}_{p}"
            r = correlations.get(key, 0.0)
            sign = "+" if r >= 0 else ""
            print(f" {sign}{r:>9.3f}", end="")
        print(f"  {channel_map[p]:>12s}")

    print(f"\n  Interpretation:")
    print(f"    Positive r → higher parameter → higher WTA/WTP (stronger EE)")
    print(f"    Negative r → higher parameter → lower WTA/WTP (weaker EE)")


def print_predictions(predictions):
    """Print predictions scorecard."""
    print(f"\n{'=' * 78}")
    print(f"  PREDICTIONS SCORECARD")
    print(f"{'=' * 78}")

    n_supported = sum(1 for p in predictions if p["supported"])
    print(f"\n  {n_supported}/{len(predictions)} predictions supported\n")

    for p in predictions:
        status = "SUPPORTED" if p["supported"] else "NOT SUPPORTED"
        marker = "+" if p["supported"] else "-"
        print(f"  [{marker}] {p['id']}: {p['description']}")
        print(f"      Actual: {p['actual']}")
        tol_info = p.get("tolerance", "")
        if tol_info:
            print(f"      Tolerance: {tol_info}", end="")
            if "in_tolerance" in p:
                tol_status = " (in range)" if p["in_tolerance"] else " (OUT OF RANGE)"
                print(tol_status, end="")
            print()
        print()


def print_cfv_comparison(metrics):
    """Print comparison against CFV benchmark ranges."""
    benchmarks = get_cfv_benchmarks()

    print(f"\n{'=' * 78}")
    print(f"  CFV BENCHMARK COMPARISON")
    print(f"  (Colucci, Franco & Valori 2024, N=516)")
    print(f"{'=' * 78}")

    print(f"\n  {'Item':<10s} {'Model':>10s} {'CFV range':>14s} {'Status':>12s}")
    print(f"  {'-' * 48}")

    for item, bkey in [("mug", "mug_wta_wtp_range"),
                        ("amazon", "amazon_wta_wtp_range"),
                        ("spotify", "spotify_wta_wtp_range")]:
        label = {"mug": "Mug", "amazon": "Amazon", "spotify": "Spotify"}[item]
        model_val = metrics[item]["ratio_mean"]
        lo, hi = benchmarks[bkey]
        in_range = lo <= model_val <= hi
        status = "IN RANGE" if in_range else "OUT OF RANGE"
        print(f"  {label:<10s} {model_val:>10.3f} [{lo:.1f}, {hi:.1f}]{'':<4s} {status:>12s}")

    print(f"\n  Note: CFV ranges are approximate plausibility bounds from the")
    print(f"  published study. Phase 1 tests model plausibility, not exact fit.")


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()

    # Load agents
    agents, pool_map = load_all_agents()

    # Print header
    print_header(agents, pool_map)

    # Run transfer test
    print(f"\n  Computing valuations...")
    results = run_transfer_test(agents, pool_map)

    # Compute metrics
    metrics = compute_aggregate_metrics(results)
    correlations = compute_parameter_correlations(agents, results)

    # Report
    print_main_table(metrics)
    print_phenotype_breakdown(metrics)
    print_time_profiles(metrics)
    print_distribution_analysis(metrics)
    print_parameter_sensitivity(correlations)

    # Predictions
    predictions = evaluate_predictions(metrics, results)
    print_predictions(predictions)

    # CFV comparison
    print_cfv_comparison(metrics)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 78}")
    print(f"  TRANSFER TEST COMPLETE")
    print(f"  {len(agents)} agents × {len(ITEM_TYPES)} items × "
          f"{len(ALL_CONDITIONS)} conditions = "
          f"{len(agents) * len(ITEM_TYPES) * len(ALL_CONDITIONS):,d} valuations")
    n_supported = sum(1 for p in predictions if p["supported"])
    print(f"  Predictions: {n_supported}/{len(predictions)} supported")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 78}")


if __name__ == '__main__':
    main()
