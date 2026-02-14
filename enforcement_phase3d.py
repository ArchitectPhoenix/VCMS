#!/usr/bin/env python3
"""
Federation Dynamics Phase 3D — Internalized Removal Cost Reanalysis
====================================================================

No new simulations. Reanalysis of existing Phase 1–3C data with internalized
removal costs. Every removal event is assigned a cost, exposing the accounting
fiction that enforcement-based governance is "free."

Two calibration levels:
  CAP-21: 2.0 units per removal (conservative, 21% salary-equivalent)
  EG-40:  4.0 units per removal (reasonable, 40% salary-equivalent)

Recomputes cost-adjusted navigability for all conditions across all phases,
performs a crossover sweep (per-removal cost 0.0–6.0), and tests 5 predictions.

Depends on: enforcement_sim.py, enforcement_phase2.py, enforcement_phase3.py
"""

import math
import time
import numpy as np

from enforcement_sim import (
    load_libraries, build_enforcement_pools, sample_population_blueprint,
    instantiate_population, assign_groups, create_replacement,
    run_baseline, run_punishment, run_threshold_exclusion,
    run_sustainability_exclusion, run_voluntary_exit,
    compute_run_metrics, aggregate_metrics,
    N_ROUNDS, N_RUNS, N_AGENTS, N_PER_GROUP, N_GROUPS,
    MAX_CONTRIB,
)

from enforcement_phase3 import (
    REHAB_CONDITIONS, VISIBILITY_CONDITIONS, STRUCTURAL_CONDITIONS,
    FULL_VIS,
    simulate_phase3,
    compute_phase3_metrics, compute_visibility_metrics,
    compute_structural_metrics,
)


# ================================================================
# PHASE 3D CONSTANTS
# ================================================================

CAP_21_COST = 2.0     # Conservative: 21% salary-equivalent per removal
EG_40_COST = 4.0      # Reasonable: 40% salary-equivalent per removal
SWEEP_MIN = 0.0
SWEEP_MAX = 6.0
SWEEP_STEP = 0.5


# ================================================================
# REMOVAL COUNT EXTRACTION
# ================================================================

def _get_total_removals(agg, phase='3c'):
    """Extract total removal count from aggregated metrics.

    Removal sources:
      - sustain_removal_count: sustainability exclusion removals (Phase 1+)
      - removal_count: threshold exclusion removals (Phase 1)
      - hybrid_removal_count: hybrid escalation removals (Phase 3A+)
    """
    sustain = agg.get('sustain_removal_count', {})
    if isinstance(sustain, dict):
        sustain = sustain.get('median', 0)

    threshold = agg.get('removal_count', {})
    if isinstance(threshold, dict):
        threshold = threshold.get('median', 0)

    hybrid = agg.get('hybrid_removal_count', {})
    if isinstance(hybrid, dict):
        hybrid = hybrid.get('median', 0)

    return sustain + threshold + hybrid


def _get_navigability(agg):
    """Extract navigability from aggregated metrics."""
    nav = agg.get('navigability', {})
    if isinstance(nav, dict):
        return nav.get('median', 0)
    return nav if isinstance(nav, (int, float)) else 0


def _get_structural_cost(agg):
    """Extract structural cost from aggregated metrics."""
    sc = agg.get('total_structural_cost', {})
    if isinstance(sc, dict):
        return sc.get('median', 0)
    return sc if isinstance(sc, (int, float)) else 0


def _get_intervention_cost(agg):
    """Extract intervention cost from aggregated metrics."""
    ic = agg.get('total_intervention_cost', {})
    if isinstance(ic, dict):
        return ic.get('median', 0)
    return ic if isinstance(ic, (int, float)) else 0


def _get_metric(agg, key, default=0):
    """Generic metric extraction from aggregated dict."""
    val = agg.get(key, {})
    if isinstance(val, dict):
        return val.get('median', default)
    return val if isinstance(val, (int, float)) else default


def _cost_adj_nav(navigability, total_cost):
    """Cost-adjusted navigability: nav / (1 + log(1 + cost))."""
    if total_cost <= 0:
        return navigability
    return navigability / (1.0 + math.log1p(total_cost))


# ================================================================
# DATA COLLECTION: Run all phases
# ================================================================

def collect_phase1_data(n_runs=N_RUNS, n_rounds=100, seed=42):
    """Run Phase 1 conditions and collect aggregated metrics."""
    print("\n  Phase 1: Enforcement mechanisms...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    rng = np.random.default_rng(seed)

    conditions = {
        'baseline':             lambda agents, p, r: run_baseline(agents, n_rounds),
        'punishment':           lambda agents, p, r: run_punishment(agents, n_rounds),
        'threshold_K3':         lambda agents, p, r: run_threshold_exclusion(
                                    agents, p, r, n_rounds, K=3),
        'threshold_K5':         lambda agents, p, r: run_threshold_exclusion(
                                    agents, p, r, n_rounds, K=5),
        'sustainability':       lambda agents, p, r: run_sustainability_exclusion(
                                    agents, p, r, n_rounds),
        'voluntary_r10':        lambda agents, p, r: run_voluntary_exit(
                                    agents, r, n_rounds, eval_freq=10,
                                    formation='random'),
        'voluntary_r10_sorted': lambda agents, p, r: run_voluntary_exit(
                                    agents, r, n_rounds, eval_freq=10,
                                    formation='sorted'),
    }

    all_metrics = {name: [] for name in conditions}

    for run in range(n_runs):
        bp = sample_population_blueprint(pools, rng)
        for cond_name, cond_fn in conditions.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2**31))
            result = cond_fn(agents, pools, cond_rng)
            metrics = compute_run_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    return {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}


def collect_phase3a_data(n_runs=N_RUNS, n_rounds=100, seed=42):
    """Run Phase 3A rehabilitation conditions and collect aggregated metrics."""
    print("  Phase 3A: Rehabilitation...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in REHAB_CONDITIONS}

    for run in range(n_runs):
        bp = sample_population_blueprint(pools, rng)
        for cond_name, mechs in REHAB_CONDITIONS.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2**31))
            result = simulate_phase3(agents, pools, cond_rng, n_rounds,
                                     mechanisms=mechs)
            metrics = compute_phase3_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    return {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}


def collect_phase3b_data(n_runs=N_RUNS, n_rounds=100, seed=45):
    """Run Phase 3B visibility conditions and collect aggregated metrics."""
    print("  Phase 3B: Visibility...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in VISIBILITY_CONDITIONS}

    for run in range(n_runs):
        bp = sample_population_blueprint(pools, rng)
        for cond_name, (mechs, vis) in VISIBILITY_CONDITIONS.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2**31))
            result = simulate_phase3(agents, pools, cond_rng, n_rounds,
                                     mechanisms=mechs, visibility=vis)
            metrics = compute_visibility_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    return {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}


def collect_phase3c_data(n_runs=N_RUNS, n_rounds=100, seed=46):
    """Run Phase 3C structural conditions and collect aggregated metrics."""
    print("  Phase 3C: Structural...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in STRUCTURAL_CONDITIONS}

    for run in range(n_runs):
        bp = sample_population_blueprint(pools, rng)
        for cond_name, (mechs, vis, struct) in STRUCTURAL_CONDITIONS.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2**31))
            result = simulate_phase3(agents, pools, cond_rng, n_rounds,
                                     mechanisms=mechs, visibility=vis,
                                     structural=struct)
            metrics = compute_structural_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    return {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}


# ================================================================
# REANALYSIS
# ================================================================

def build_unified_table(p1, p3a, p3b, p3c):
    """Build a unified condition table across all phases.

    Returns list of dicts with keys:
      phase, condition, ss_coop, ttfr, removals, structural_cost,
      intervention_cost, navigability, dignity_floor, gini
    """
    rows = []

    # Phase 1
    for name, agg in p1.items():
        rows.append({
            'phase': '1',
            'condition': name,
            'ss_coop': _get_metric(agg, 'steady_state_coop'),
            'ttfr': _get_metric(agg, 'system_ttfr'),
            'removals': _get_total_removals(agg, phase='1'),
            'structural_cost': 0.0,
            'intervention_cost': 0.0,
            'navigability': 0.0,  # Not computed in Phase 1
            'dignity_floor': 0.0,  # Not computed in Phase 1
            'gini': _get_metric(agg, 'gini'),
        })

    # Phase 3A
    for name, agg in p3a.items():
        rows.append({
            'phase': '3A',
            'condition': name,
            'ss_coop': _get_metric(agg, 'steady_state_coop'),
            'ttfr': _get_metric(agg, 'system_ttfr'),
            'removals': _get_total_removals(agg, phase='3a'),
            'structural_cost': 0.0,
            'intervention_cost': _get_intervention_cost(agg),
            'navigability': 0.0,  # Not computed in Phase 3A
            'dignity_floor': 0.0,  # Not computed in Phase 3A
            'gini': _get_metric(agg, 'gini'),
        })

    # Phase 3B
    for name, agg in p3b.items():
        rows.append({
            'phase': '3B',
            'condition': name,
            'ss_coop': _get_metric(agg, 'steady_state_coop'),
            'ttfr': _get_metric(agg, 'system_ttfr'),
            'removals': _get_total_removals(agg, phase='3b'),
            'structural_cost': 0.0,
            'intervention_cost': _get_intervention_cost(agg),
            'navigability': _get_navigability(agg),
            'dignity_floor': _get_metric(agg, 'dignity_floor'),
            'gini': _get_metric(agg, 'gini'),
        })

    # Phase 3C
    for name, agg in p3c.items():
        rows.append({
            'phase': '3C',
            'condition': name,
            'ss_coop': _get_metric(agg, 'steady_state_coop'),
            'ttfr': _get_metric(agg, 'system_ttfr'),
            'removals': _get_total_removals(agg, phase='3c'),
            'structural_cost': _get_structural_cost(agg),
            'intervention_cost': _get_intervention_cost(agg),
            'navigability': _get_navigability(agg),
            'dignity_floor': _get_metric(agg, 'dignity_floor'),
            'gini': _get_metric(agg, 'gini'),
        })

    return rows


def compute_true_costs(rows, per_removal_cost):
    """Add true cost columns to each row."""
    for row in rows:
        removal_cost = row['removals'] * per_removal_cost
        row['removal_cost'] = removal_cost
        row['total_true_cost'] = (row['structural_cost'] +
                                  row['intervention_cost'] +
                                  removal_cost)
        row['original_cost'] = row['structural_cost'] + row['intervention_cost']
        if row['navigability'] > 0:
            row['original_cost_adj_nav'] = _cost_adj_nav(
                row['navigability'], row['original_cost'])
            row['true_cost_adj_nav'] = _cost_adj_nav(
                row['navigability'], row['total_true_cost'])
        else:
            row['original_cost_adj_nav'] = 0.0
            row['true_cost_adj_nav'] = 0.0
    return rows


# ================================================================
# CROSSOVER ANALYSIS
# ================================================================

def crossover_sweep(rows_template):
    """Sweep per-removal cost from 0.0 to 6.0 and find SV5 vs B2 crossover.

    Uses Phase 3C rows (which have navigability) for the comparison.
    Returns (sweep_data, crossover_point).
    """
    # Find the B2 and SV5 rows from Phase 3C
    b2_row = None
    sv5_row = None
    for row in rows_template:
        if row['phase'] == '3C' and row['condition'] == 'B2_sustain':
            b2_row = row
        if row['phase'] == '3C' and row['condition'] == 'SV5_all+vis':
            sv5_row = row

    if not b2_row or not sv5_row:
        return [], None

    sweep_data = []
    crossover_point = None
    prev_b2_better = None

    cost_val = SWEEP_MIN
    while cost_val <= SWEEP_MAX + 0.001:
        # B2 costs
        b2_removal_cost = b2_row['removals'] * cost_val
        b2_total = b2_row['structural_cost'] + b2_row['intervention_cost'] + b2_removal_cost
        b2_cnav = _cost_adj_nav(b2_row['navigability'], b2_total)

        # SV5 costs (no removals)
        sv5_total = sv5_row['structural_cost'] + sv5_row['intervention_cost']
        sv5_cnav = _cost_adj_nav(sv5_row['navigability'], sv5_total)

        sweep_data.append({
            'per_removal_cost': round(cost_val, 1),
            'b2_total_cost': b2_total,
            'b2_cost_adj_nav': b2_cnav,
            'sv5_total_cost': sv5_total,
            'sv5_cost_adj_nav': sv5_cnav,
        })

        b2_better = b2_cnav > sv5_cnav
        if prev_b2_better is not None and prev_b2_better and not b2_better:
            # Crossover happened between prev cost and this cost
            # Linear interpolation
            prev_cost = round(cost_val - SWEEP_STEP, 1)
            prev_b2 = sweep_data[-2]['b2_cost_adj_nav']
            prev_sv5 = sweep_data[-2]['sv5_cost_adj_nav']
            curr_b2 = b2_cnav
            curr_sv5 = sv5_cnav
            # Find where b2 = sv5
            # prev_b2 - prev_sv5 > 0, curr_b2 - curr_sv5 < 0
            gap_prev = prev_b2 - prev_sv5
            gap_curr = curr_b2 - curr_sv5
            if gap_prev != gap_curr:
                frac = gap_prev / (gap_prev - gap_curr)
                crossover_point = prev_cost + frac * SWEEP_STEP
        prev_b2_better = b2_better

        cost_val += SWEEP_STEP

    # If SV5 was always better (even at cost=0), crossover is at 0
    if crossover_point is None:
        if sweep_data and sweep_data[0]['sv5_cost_adj_nav'] >= sweep_data[0]['b2_cost_adj_nav']:
            crossover_point = 0.0

    return sweep_data, crossover_point


def extended_crossover_sweep(rows_template):
    """Sweep all conditions that have removals against SV5.

    Returns dict of condition → crossover_point.
    """
    # Find SV5 from Phase 3C
    sv5_row = None
    for row in rows_template:
        if row['phase'] == '3C' and row['condition'] == 'SV5_all+vis':
            sv5_row = row
    if not sv5_row:
        return {}

    sv5_total = sv5_row['structural_cost'] + sv5_row['intervention_cost']
    sv5_cnav = _cost_adj_nav(sv5_row['navigability'], sv5_total)

    # Only compare conditions from phases that have navigability (3B, 3C)
    removal_rows = [r for r in rows_template
                    if r['removals'] > 0 and r['navigability'] > 0]

    results = {}
    for row in removal_rows:
        prev_better = None
        crossover = None

        cost_val = SWEEP_MIN
        while cost_val <= SWEEP_MAX + 0.001:
            removal_cost = row['removals'] * cost_val
            total = row['structural_cost'] + row['intervention_cost'] + removal_cost
            cnav = _cost_adj_nav(row['navigability'], total)

            row_better = cnav > sv5_cnav
            if prev_better is not None and prev_better and not row_better:
                prev_cost = round(cost_val - SWEEP_STEP, 1)
                # Recompute prev
                prev_removal = row['removals'] * prev_cost
                prev_total = row['structural_cost'] + row['intervention_cost'] + prev_removal
                prev_cnav = _cost_adj_nav(row['navigability'], prev_total)
                gap_prev = prev_cnav - sv5_cnav
                gap_curr = cnav - sv5_cnav
                if gap_prev != gap_curr:
                    frac = gap_prev / (gap_prev - gap_curr)
                    crossover = prev_cost + frac * SWEEP_STEP
            prev_better = row_better
            cost_val += SWEEP_STEP

        if crossover is None:
            if not prev_better:
                crossover = 0.0  # SV5 always dominated
            # else: row always better (crossover above sweep range)

        key = f"{row['phase']}:{row['condition']}"
        results[key] = crossover

    return results


# ================================================================
# REPORTING
# ================================================================

def print_header():
    """Print Phase 3D header."""
    print("=" * 140)
    print("FEDERATION DYNAMICS PHASE 3D")
    print("Internalized Removal Cost Reanalysis")
    print("=" * 140)
    print()
    print("Every removal-based mechanism previously reported structural cost = 0.")
    print("This is an accounting fiction. Removal externalizes costs to three parties:")
    print("  1. The removed agent (lost access, severed relationships, dignity)")
    print("  2. The group (integration disruption, v_level recalibration)")
    print("  3. The system (displaced agent still exists beyond measurement boundary)")
    print()
    print("Calibration from worker replacement cost literature:")
    print(f"  CAP-21 (conservative):  {CAP_21_COST:.1f} units/removal  "
          f"(21% salary-equivalent, median of 30 case studies)")
    print(f"  EG-40  (reasonable):    {EG_40_COST:.1f} units/removal  "
          f"(40% salary-equivalent, average of 37 case studies)")
    print()


def print_removal_inventory(rows):
    """Print removal counts for all conditions that have any."""
    print("\n" + "=" * 140)
    print("REMOVAL INVENTORY: All conditions with removals")
    print("=" * 140)

    header = (f"{'Phase':<6} {'Condition':<22} {'Removals':>9} "
              f"{'RemCost@2.0':>12} {'RemCost@4.0':>12} "
              f"{'OrigCost':>9} {'TrueCost@2.0':>13} {'TrueCost@4.0':>13}")
    print(header)
    print("-" * 140)

    removal_rows = [r for r in rows if r['removals'] > 0]
    removal_rows.sort(key=lambda r: r['removals'], reverse=True)

    for row in removal_rows:
        rc_cap = row['removals'] * CAP_21_COST
        rc_eg = row['removals'] * EG_40_COST
        orig = row['structural_cost'] + row['intervention_cost']
        tc_cap = orig + rc_cap
        tc_eg = orig + rc_eg
        print(f"{row['phase']:<6} {row['condition']:<22} {row['removals']:>9.0f} "
              f"{rc_cap:>12.1f} {rc_eg:>12.1f} "
              f"{orig:>9.1f} {tc_cap:>13.1f} {tc_eg:>13.1f}")

    print()
    total_removals = sum(r['removals'] for r in removal_rows)
    print(f"  Total removal events across all conditions: {total_removals:.0f}")


def print_cost_adjusted_comparison(rows, per_removal_cost, label):
    """Print cost-adjusted navigability comparison at a given per-removal cost.

    Only includes Phase 3B and 3C conditions (which have navigability).
    """
    print(f"\n{'=' * 140}")
    print(f"COST-ADJUSTED NAVIGABILITY AT {label} ({per_removal_cost:.1f} units/removal)")
    print("=" * 140)

    nav_rows = [r for r in rows if r['navigability'] > 0]
    compute_true_costs(nav_rows, per_removal_cost)

    header = (f"{'Phase':<6} {'Condition':<22} {'SS-Coop':>8} {'DignFl':>7} "
              f"{'Navig':>7} {'Removals':>9} {'RemCost':>8} "
              f"{'OrigCost':>9} {'TrueCost':>9} "
              f"{'OrigCNav':>9} {'TrueCNav':>9} {'Delta':>7}")
    print(header)
    print("-" * 140)

    # Sort by true cost-adjusted navigability descending
    nav_rows.sort(key=lambda r: r['true_cost_adj_nav'], reverse=True)

    for row in nav_rows:
        delta = row['true_cost_adj_nav'] - row['original_cost_adj_nav']
        print(f"{row['phase']:<6} {row['condition']:<22} "
              f"{row['ss_coop']:>8.1f} {row['dignity_floor']:>7.1%} "
              f"{row['navigability']:>7.3f} {row['removals']:>9.0f} "
              f"{row['removal_cost']:>8.1f} "
              f"{row['original_cost']:>9.1f} {row['total_true_cost']:>9.1f} "
              f"{row['original_cost_adj_nav']:>9.3f} "
              f"{row['true_cost_adj_nav']:>9.3f} "
              f"{delta:>+7.3f}")


def print_crossover_sweep(sweep_data, crossover_point):
    """Print the crossover sweep chart."""
    print(f"\n{'=' * 140}")
    print("CROSSOVER ANALYSIS: SV5 vs B2 at varying per-removal costs")
    print("=" * 140)
    print("  At what per-removal cost does SV5's cost-adjusted navigability exceed B2's?")
    print()

    header = (f"{'RemovalCost':>11} {'B2_TotalCost':>13} {'B2_CostNav':>11} "
              f"{'SV5_TotalCost':>14} {'SV5_CostNav':>12} {'Winner':>8}")
    print(header)
    print("-" * 80)

    for d in sweep_data:
        winner = "B2" if d['b2_cost_adj_nav'] > d['sv5_cost_adj_nav'] else "SV5"
        if abs(d['b2_cost_adj_nav'] - d['sv5_cost_adj_nav']) < 0.001:
            winner = "TIE"
        print(f"{d['per_removal_cost']:>11.1f} {d['b2_total_cost']:>13.1f} "
              f"{d['b2_cost_adj_nav']:>11.3f} "
              f"{d['sv5_total_cost']:>14.1f} {d['sv5_cost_adj_nav']:>12.3f} "
              f"{winner:>8}")

    print()
    if crossover_point is not None:
        print(f"  CROSSOVER POINT: {crossover_point:.2f} units per removal")
        if crossover_point < CAP_21_COST:
            print(f"  → Below CAP-21 conservative estimate ({CAP_21_COST:.1f})")
            print(f"  → SV5 dominates B2 on cost-adjusted navigability at ANY "
                  f"empirically calibrated removal cost")
        elif crossover_point < EG_40_COST:
            print(f"  → Between CAP-21 ({CAP_21_COST:.1f}) and EG-40 ({EG_40_COST:.1f})")
            print(f"  → SV5 dominates B2 at the reasonable estimate but not conservative")
        else:
            print(f"  → Above EG-40 reasonable estimate ({EG_40_COST:.1f})")
            print(f"  → Enforcement remains cost-competitive even with internalized costs")
    else:
        print("  No crossover found in sweep range.")


def print_extended_crossovers(crossover_results):
    """Print crossover points for all conditions with removals."""
    print(f"\n{'=' * 140}")
    print("EXTENDED CROSSOVER: All removal conditions vs SV5")
    print("=" * 140)
    print("  Per-removal cost at which SV5 dominates each condition on cost-adjusted navigability.")
    print()

    header = f"{'Condition':<30} {'Crossover':>10} {'vs CAP-21':>10} {'vs EG-40':>10}"
    print(header)
    print("-" * 70)

    for key in sorted(crossover_results.keys()):
        cp = crossover_results[key]
        if cp is not None:
            vs_cap = "below" if cp < CAP_21_COST else "above"
            vs_eg = "below" if cp < EG_40_COST else "above"
            print(f"{key:<30} {cp:>10.2f} {vs_cap:>10} {vs_eg:>10}")
        else:
            print(f"{key:<30} {'never':>10} {'—':>10} {'—':>10}")


def print_phase_summary(rows, per_removal_cost, label):
    """Print summary table showing best condition per phase at given removal cost."""
    print(f"\n{'=' * 140}")
    print(f"PHASE SUMMARY AT {label} ({per_removal_cost:.1f} units/removal)")
    print("=" * 140)

    compute_true_costs(rows, per_removal_cost)

    # Group by phase, pick best navigability condition where available,
    # otherwise pick best cooperation
    phases = {}
    for row in rows:
        p = row['phase']
        if p not in phases:
            phases[p] = []
        phases[p].append(row)

    header = (f"{'Phase':<6} {'Best Condition':<22} {'SS-Coop':>8} {'DignFl':>7} "
              f"{'Navig':>7} {'Removals':>9} "
              f"{'OrigCost':>9} {'TrueCost':>9} {'TrueCNav':>9}")
    print(header)
    print("-" * 100)

    for phase in ['1', '3A', '3B', '3C']:
        if phase not in phases:
            continue
        phase_rows = phases[phase]

        # For phases with navigability, pick highest true_cost_adj_nav
        # For phases without, pick highest ss_coop
        has_nav = any(r['navigability'] > 0 for r in phase_rows)
        if has_nav:
            best = max(phase_rows, key=lambda r: r['true_cost_adj_nav'])
        else:
            best = max(phase_rows, key=lambda r: r['ss_coop'])

        print(f"{phase:<6} {best['condition']:<22} "
              f"{best['ss_coop']:>8.1f} "
              f"{best['dignity_floor']:>7.1%}" if best['dignity_floor'] > 0 else f"{'—':>7}",
              end="")
        if has_nav:
            print(f" {best['navigability']:>7.3f} ", end="")
        else:
            print(f" {'—':>7} ", end="")
        print(f"{best['removals']:>9.0f} "
              f"{best['original_cost']:>9.1f} "
              f"{best['total_true_cost']:>9.1f} ", end="")
        if has_nav:
            print(f"{best['true_cost_adj_nav']:>9.3f}")
        else:
            print(f"{'—':>9}")


def print_phase_summary_table(rows, per_removal_cost, label):
    """Print full phase summary table (corrected formatting)."""
    print(f"\n{'=' * 140}")
    print(f"PHASE SUMMARY AT {label} ({per_removal_cost:.1f} units/removal)")
    print("=" * 140)

    compute_true_costs(rows, per_removal_cost)

    phases = {}
    for row in rows:
        p = row['phase']
        if p not in phases:
            phases[p] = []
        phases[p].append(row)

    header = (f"{'Phase':<6} {'Best Condition':<22} {'SS-Coop':>8} {'DignFl':>7} "
              f"{'Navig':>7} {'Removals':>9} "
              f"{'OrigCost':>9} {'TrueCost':>9} {'TrueCNav':>9}")
    print(header)
    print("-" * 100)

    for phase in ['1', '3A', '3B', '3C']:
        if phase not in phases:
            continue
        phase_rows = phases[phase]

        has_nav = any(r['navigability'] > 0 for r in phase_rows)
        if has_nav:
            best = max(phase_rows, key=lambda r: r['true_cost_adj_nav'])
        else:
            best = max(phase_rows, key=lambda r: r['ss_coop'])

        dfl_str = f"{best['dignity_floor']:>7.1%}" if best['dignity_floor'] > 0 else f"{'—':>7}"
        nav_str = f"{best['navigability']:>7.3f}" if has_nav else f"{'—':>7}"
        cnav_str = f"{best['true_cost_adj_nav']:>9.3f}" if has_nav else f"{'—':>9}"

        print(f"{phase:<6} {best['condition']:<22} "
              f"{best['ss_coop']:>8.1f} {dfl_str} "
              f"{nav_str} {best['removals']:>9.0f} "
              f"{best['original_cost']:>9.1f} "
              f"{best['total_true_cost']:>9.1f} {cnav_str}")


def print_predictions(rows, sweep_data, crossover_point, crossover_results):
    """Evaluate Phase 3D predictions."""
    print(f"\n{'=' * 140}")
    print("PHASE 3D PREDICTIONS SCORECARD")
    print("=" * 140)

    scored = []

    # Find key rows from Phase 3C
    b2_3c = None
    sv5_3c = None
    for row in rows:
        if row['phase'] == '3C' and row['condition'] == 'B2_sustain':
            b2_3c = row
        if row['phase'] == '3C' and row['condition'] == 'SV5_all+vis':
            sv5_3c = row

    # P1: Crossover point is below CAP-21 (2.0)
    p1 = crossover_point is not None and crossover_point < CAP_21_COST
    cp_str = f"{crossover_point:.2f}" if crossover_point is not None else "N/A"
    scored.append(('P1',
        f"Crossover point below CAP-21 ({CAP_21_COST:.1f}): "
        f"crossover={cp_str}",
        p1))

    # P2: At CAP-21, every structural-architecture condition dominates B2
    if b2_3c:
        compute_true_costs(rows, CAP_21_COST)
        b2_cnav_cap = _cost_adj_nav(b2_3c['navigability'], b2_3c['total_true_cost'])
        struct_names = ['S1_floor', 'S3_match', 'SV1_floor+vis', 'SV3_match+vis',
                        'SV4_all_struct', 'SV5_all+vis']
        all_dominate = True
        domination_details = []
        for sname in struct_names:
            srow = None
            for r in rows:
                if r['phase'] == '3C' and r['condition'] == sname:
                    srow = r
                    break
            if srow:
                s_cnav = _cost_adj_nav(srow['navigability'], srow['total_true_cost'])
                dominates = s_cnav > b2_cnav_cap
                domination_details.append(f"{sname}={s_cnav:.3f}")
                if not dominates:
                    all_dominate = False
            else:
                all_dominate = False

        p2 = all_dominate
        scored.append(('P2',
            f"At CAP-21, structural conditions dominate B2 (B2 cnav={b2_cnav_cap:.3f}): "
            f"{'; '.join(domination_details)}",
            p2))
    else:
        scored.append(('P2', "B2 not found in Phase 3C data", False))

    # P3: At EG-40, enforcement is the most expensive regime
    if b2_3c:
        compute_true_costs(rows, EG_40_COST)
        b2_true_cost = b2_3c['total_true_cost']
        # Check if B2 cost exceeds all structural conditions
        max_struct_cost = 0
        for r in rows:
            if r['phase'] == '3C' and r['condition'].startswith('S'):
                if r['total_true_cost'] > max_struct_cost:
                    max_struct_cost = r['total_true_cost']

        # Also check threshold from Phase 1
        thresh_rows = [r for r in rows if r['phase'] == '1'
                       and r['condition'].startswith('threshold')]
        max_thresh_cost = 0
        for r in thresh_rows:
            tc = r['removals'] * EG_40_COST
            if tc > max_thresh_cost:
                max_thresh_cost = tc

        p3 = max_thresh_cost > max_struct_cost
        scored.append(('P3',
            f"At EG-40, enforcement is most expensive: "
            f"threshold_K3 cost={max_thresh_cost:.1f}, "
            f"B2 cost={b2_true_cost:.1f}, "
            f"max structural cost={max_struct_cost:.1f}",
            p3))
    else:
        scored.append(('P3', "B2 not found in Phase 3C data", False))

    # P4: H2 graduated becomes more cost-competitive than B2
    h2_rows = [r for r in rows if r['condition'] == 'H2_graduated']
    b2_rows = [r for r in rows if r['condition'] == 'B2_sustain']
    if h2_rows and b2_rows:
        # Use Phase 3A versions (both present there)
        h2 = [r for r in h2_rows if r['phase'] == '3A']
        b2 = [r for r in b2_rows if r['phase'] == '3A']
        if h2 and b2:
            h2 = h2[0]
            b2 = b2[0]
            compute_true_costs([h2, b2], CAP_21_COST)
            p4 = h2['total_true_cost'] < b2['total_true_cost']
            scored.append(('P4',
                f"H2 cheaper than B2 at CAP-21: "
                f"H2 total={h2['total_true_cost']:.1f} "
                f"({h2['removals']:.0f} removals × {CAP_21_COST} + "
                f"{h2['intervention_cost']:.1f} intervention) vs "
                f"B2 total={b2['total_true_cost']:.1f} "
                f"({b2['removals']:.0f} removals × {CAP_21_COST})",
                p4))
        else:
            scored.append(('P4', "H2/B2 not found in Phase 3A data", False))
    else:
        scored.append(('P4', "H2/B2 not found", False))

    # P5: Ranking inversion is robust — even at per_removal_cost = 1.0
    if b2_3c and sv5_3c:
        compute_true_costs([b2_3c, sv5_3c], 1.0)
        b2_cnav_1 = _cost_adj_nav(b2_3c['navigability'], b2_3c['total_true_cost'])
        sv5_cnav_1 = _cost_adj_nav(sv5_3c['navigability'], sv5_3c['total_true_cost'])
        # "Meaningfully changes" = B2's cost-adjusted nav drops by at least 10%
        compute_true_costs([b2_3c], 0.0)
        b2_cnav_0 = _cost_adj_nav(b2_3c['navigability'], b2_3c['total_true_cost'])
        drop_pct = (b2_cnav_0 - b2_cnav_1) / max(b2_cnav_0, 0.001)
        p5 = drop_pct > 0.10
        scored.append(('P5',
            f"Robust at cost=1.0 (half conservative): "
            f"B2 cnav drops {b2_cnav_0:.3f} → {b2_cnav_1:.3f} ({drop_pct:.1%} drop), "
            f"SV5 cnav={sv5_cnav_1:.3f}",
            p5))
    else:
        scored.append(('P5', "B2/SV5 not found in Phase 3C data", False))

    # Print scorecard
    supported = sum(1 for _, _, s in scored if s)
    total = len(scored)
    print(f"\n  Score: {supported}/{total} predictions supported\n")
    for pid, desc, result_val in scored:
        status = "SUPPORTED" if result_val else "NOT SUPPORTED"
        print(f"  {pid}: {status}")
        print(f"      {desc}")
        print()


def print_conclusion(crossover_point, rows):
    """Print the conclusion."""
    print(f"\n{'=' * 140}")
    print("CONCLUSION")
    print("=" * 140)

    if crossover_point is not None and crossover_point < CAP_21_COST:
        print()
        print(f"  The crossover point is {crossover_point:.2f} units per removal.")
        print(f"  The conservative empirical estimate (CAP-21) is {CAP_21_COST:.1f}.")
        print(f"  The reasonable estimate (EG-40) is {EG_40_COST:.1f}.")
        print()
        print("  At ANY empirically calibrated removal cost, SV5 (structural architecture")
        print("  + visibility) dominates B2 (sustainability exclusion) on cost-adjusted")
        print("  navigability. The 'expensive' structural architecture is CHEAPER than the")
        print("  'free' enforcement once you count what enforcement actually costs.")
        print()
        print("  The cooperation-dignity tradeoff from Phase 3C is confirmed and strengthened:")
        print("  SV5 dominates B2 not just on navigability but on total cost.")
        print()
        print("  This is a V-primitive result: making the full cost visible changes the")
        print("  analysis. High V couples action to consequence. Here, high V in the")
        print("  accounting couples removal to its actual cost.")

        # Find threshold K3 cost at EG-40
        for row in rows:
            if row['phase'] == '1' and row['condition'] == 'threshold_K3':
                tc = row['removals'] * EG_40_COST
                print()
                print(f"  Most expensive mechanism at EG-40: threshold_K3 with "
                      f"{row['removals']:.0f} removals × {EG_40_COST:.1f} = {tc:.1f} units")
                print(f"  (vs SV5 structural cost of ~137 units)")
                break
    else:
        print()
        print("  Removal is genuinely cheaper than structure even when its costs are")
        print("  internalized. The case for structural architecture remains values-based")
        print("  rather than cost-based — a meaningful result either way.")

    print()
    print("  Either way, the data decides.")
    print()


# ================================================================
# RUNNER
# ================================================================

def run_phase3d(n_runs=N_RUNS, n_rounds=100):
    """Run the full Phase 3D reanalysis."""
    print_header()

    print("Collecting data from all phases...")
    print("(Re-running simulations with matched seeds for fresh aggregated metrics)")
    print()

    t0 = time.time()

    p1 = collect_phase1_data(n_runs, n_rounds, seed=42)
    p3a = collect_phase3a_data(n_runs, n_rounds, seed=42)
    p3b = collect_phase3b_data(n_runs, n_rounds, seed=45)
    p3c = collect_phase3c_data(n_runs, n_rounds, seed=46)

    elapsed = time.time() - t0
    print(f"\nData collection complete: {elapsed:.1f}s")

    # Build unified table
    rows = build_unified_table(p1, p3a, p3b, p3c)

    # 1. Removal inventory
    print_removal_inventory(rows)

    # 2. Cost-adjusted comparison at CAP-21
    print_cost_adjusted_comparison(
        [dict(r) for r in rows], CAP_21_COST, "CAP-21")

    # 3. Cost-adjusted comparison at EG-40
    print_cost_adjusted_comparison(
        [dict(r) for r in rows], EG_40_COST, "EG-40")

    # 4. Crossover sweep
    sweep_data, crossover_point = crossover_sweep(rows)
    print_crossover_sweep(sweep_data, crossover_point)

    # 5. Extended crossover (all removal conditions)
    crossover_results = extended_crossovers = extended_crossover_sweep(rows)
    print_extended_crossovers(crossover_results)

    # 6. Phase summary tables
    print_phase_summary_table([dict(r) for r in rows], CAP_21_COST, "CAP-21")
    print_phase_summary_table([dict(r) for r in rows], EG_40_COST, "EG-40")

    # 7. Predictions
    # Reset true costs (predictions compute their own)
    for row in rows:
        row.pop('removal_cost', None)
        row.pop('total_true_cost', None)
        row.pop('original_cost', None)
        row.pop('original_cost_adj_nav', None)
        row.pop('true_cost_adj_nav', None)
    print_predictions(rows, sweep_data, crossover_point, crossover_results)

    # 8. Conclusion
    print_conclusion(crossover_point, rows)

    return rows


if __name__ == '__main__':
    run_phase3d()
