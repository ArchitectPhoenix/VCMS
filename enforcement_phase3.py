#!/usr/bin/env python3
"""
Federation Dynamics Phase 3 — Rehabilitation: Parameter Intervention as Alternative to Removal
==============================================================================================

Tests enforcement mechanisms that operate on agent *state* — modifying the conditions
that produce harmful behavior rather than removing the person exhibiting it.

Same trigger as sustainability exclusion (budget slope < 0 AND cooperation slope < 0
for 3+ rounds). Different response: intervene on agent parameters instead of removing.

Conditions:
  B1  Baseline (no mechanism)
  B2  Sustainability exclusion (Phase 1 removal)
  R1a-c  Strain reduction (25/50/75%)
  R2a-c  Budget support (0.5/1.0/2.0)
  R3a-c  Replenishment boost (1.5x/2.0x/3.0x for 10 rounds)
  R4  Comprehensive (strain + budget + replenishment)
  R5  Targeted diagnosis (root-cause matching)
  H1  Hybrid: rehabilitate first, remove if no improvement
  H2  Hybrid: graduated response (strain → comprehensive → removal)

Population: 40 agents, 10 groups of 4, same distribution as Phase 1-2.
100 runs × 100 rounds per condition. ~90M agent-steps total.

Depends on: enforcement_sim.py, enforcement_phase2.py
"""

import copy
import math
import time
import numpy as np
from collections import defaultdict

from enforcement_sim import (
    load_libraries, build_enforcement_pools, sample_population_blueprint,
    instantiate_population, assign_groups, create_replacement,
    step_group, record_round,
    Agent, EnforcementAgentParams, AgentState,
    _linear_slope, gini_coefficient, bimodality_coefficient,
    compute_run_metrics, aggregate_metrics,
    N_ROUNDS, N_RUNS, N_AGENTS, N_PER_GROUP, N_GROUPS,
    MAX_CONTRIB, MAX_PUNISH, ANCHOR_RATE, ACUTE_MULT, EPS,
    RUPTURE_B_FRAC, RUPTURE_CONSEC,
    THRESHOLD_SD_MULT, HEALTH_WINDOW, HEALTH_TRIGGER_CONSEC,
    LEAVE_THRESHOLD,
)

from enforcement_phase2 import (
    compute_phase2_metrics, _legibility, _is_ruptured, _pearson,
)


# ================================================================
# PHASE 3 CONSTANTS
# ================================================================

REHAB_BASE_COST = 1.0
REHAB_BOOST_DURATION = 10
REHAB_PATIENCE = 10          # H1: rounds to wait before removal fallback
IMPROVEMENT_WINDOW = 10      # rounds to measure conversion
RECIDIVISM_WINDOW = 20       # rounds to measure recidivism


# ================================================================
# TARGET SELECTION
# ================================================================

def _select_highest_strain(members):
    """Select agent with highest current strain."""
    return max(range(len(members)), key=lambda j: members[j].state.strain)


def _select_lowest_budget(members):
    """Select agent with lowest current budget."""
    return min(range(len(members)), key=lambda j: members[j].state.B)


def _select_lowest_ratio(members):
    """Select agent with lowest b_replenish / b_depletion ratio."""
    return min(range(len(members)),
               key=lambda j: members[j].params.b_replenish_rate /
                              max(members[j].params.b_depletion_rate, EPS))


def _select_highest_impact(members, contribs):
    """Select agent whose removal most improves group mean (sustainability target)."""
    total = sum(contribs)
    n = len(members)
    gmean = total / n
    best_imp, worst_j = -float('inf'), 0
    for j in range(n):
        imp = (total - contribs[j]) / max(n - 1, 1) - gmean
        if imp > best_imp:
            best_imp, worst_j = imp, j
    return worst_j


# ================================================================
# INTERVENTION APPLICATION
# ================================================================

def _apply_strain_reduction(agent, reduction_factor):
    """Reduce agent's current strain by fraction."""
    agent.state.strain *= (1.0 - reduction_factor)


def _apply_budget_support(agent, amount):
    """Inject budget directly."""
    agent.state.B += amount


def _apply_replenish_boost(agent, factor, duration):
    """Set temporary replenishment boost (applied each round in simulation loop)."""
    agent._rehab_boost_remaining = duration
    agent._rehab_boost_factor = factor


def _apply_comprehensive(agent, reduction=0.50, amount=1.0, factor=1.5, duration=10):
    """Apply all three: strain reduction + budget support + replenishment boost."""
    _apply_strain_reduction(agent, reduction)
    _apply_budget_support(agent, amount)
    _apply_replenish_boost(agent, factor, duration)


def _apply_targeted(agent, all_agents):
    """Diagnose root cause, then apply matching intervention. Returns sub-mode name."""
    all_strains = [ag.state.strain for ag in all_agents]
    all_budgets = [ag.state.B for ag in all_agents]
    median_strain = sorted(all_strains)[len(all_strains) // 2]
    median_budget = sorted(all_budgets)[len(all_budgets) // 2]

    if agent.state.strain > median_strain:
        _apply_strain_reduction(agent, 0.50)
        return 'strain_reduction'
    elif agent.state.B < median_budget:
        _apply_budget_support(agent, 1.0)
        return 'budget_support'
    elif agent.params.b_replenish_rate / max(agent.params.b_depletion_rate, EPS) < 1.0:
        _apply_replenish_boost(agent, 1.5, REHAB_BOOST_DURATION)
        return 'replenish_boost'
    else:
        _apply_comprehensive(agent)
        return 'comprehensive'


# ================================================================
# INTERVENTION COST
# ================================================================

def _compute_cost(mode, kwargs):
    """Compute intervention cost based on mode and parameters."""
    base_cost = kwargs.get('base_cost', REHAB_BASE_COST)
    if mode == 'strain_reduction':
        return base_cost * kwargs.get('reduction', 0.50)
    elif mode == 'budget_support':
        return base_cost
    elif mode == 'replenish_boost':
        factor = kwargs.get('factor', 1.5)
        duration = kwargs.get('duration', REHAB_BOOST_DURATION)
        return base_cost * (factor - 1.0) * duration / 10.0
    elif mode == 'comprehensive':
        return base_cost * 0.50 + base_cost + base_cost * 0.5 * 10 / 10.0
    elif mode == 'targeted':
        return 0.1 + base_cost * 0.50  # diagnosis + estimated treatment
    elif mode == 'hybrid_rehab_first':
        return base_cost * 0.50 + base_cost + base_cost * 0.5 * 10 / 10.0
    elif mode == 'hybrid_graduated':
        return base_cost * 0.50  # first intervention is strain reduction
    return base_cost


def _deduct_cost(members, cost):
    """Distribute intervention cost evenly across group members."""
    per_member = cost / max(len(members), 1)
    for ag in members:
        ag.state.B = max(0, ag.state.B - per_member)


# ================================================================
# AGENT REHABILITATION STATE HELPERS
# ================================================================

def _init_rehab_state(agent):
    """Initialize rehabilitation tracking attributes on an agent."""
    if not hasattr(agent, '_rehab_history'):
        agent._rehab_history = []
    if not hasattr(agent, '_rehab_boost_remaining'):
        agent._rehab_boost_remaining = 0
    if not hasattr(agent, '_rehab_boost_factor'):
        agent._rehab_boost_factor = 1.0
    if not hasattr(agent, '_rehab_count'):
        agent._rehab_count = 0
    if not hasattr(agent, '_rehab_last_round'):
        agent._rehab_last_round = -1
    if not hasattr(agent, '_rehab_last_contrib'):
        agent._rehab_last_contrib = -1.0


def _apply_boosts(members, dt):
    """Apply active replenishment boosts for all group members. Call after step_group."""
    for ag in members:
        if not hasattr(ag, '_rehab_boost_remaining'):
            continue
        if ag._rehab_boost_remaining > 0:
            # Flat supplement proportional to natural replenishment rate.
            # 0.5 = expected mean positive experience (modeling simplification).
            ag.state.B += dt * ag.params.b_replenish_rate * (ag._rehab_boost_factor - 1.0) * 0.5
            ag._rehab_boost_remaining -= 1


# ================================================================
# VISIBILITY EFFECTS — FULL PARAMETER TRANSPARENCY
# ================================================================

def _apply_visibility(groups, round_contribs, dt, vis_config):
    """
    Apply effects of full parameter/state visibility between all agents in each group.

    All agents see all groupmates' parameters (c_base, s_initial, alpha, inertia,
    b_depletion_rate, b_replenish_rate) and current state (B, strain). This
    knowledge is operationalized as three between-round state modifications:

    1. Empathetic strain modulation: knowing WHY someone defects (constraint vs choice)
       reduces strain from their gap. Gated by observer's alpha (social sensitivity).

    2. Solidarity transfer: agents above group median budget share a fraction with
       agents below. Sharing rate gated by sharer's c_base (prosocial tendency).

    3. Informed reference: agent's v_level shifts toward mean of capable agents
       (those with budget above vulnerability threshold), preventing downward spiral
       from constrained agents dragging the reference down.
    """
    empathy = vis_config.get('empathy', False)
    solidarity = vis_config.get('solidarity', False)
    reference = vis_config.get('reference', False)

    for gid, members in groups.items():
        if len(members) <= 1:
            continue

        budgets = [ag.state.B for ag in members]
        n = len(members)
        sorted_b = sorted(budgets)
        median_b = sorted_b[n // 2]
        vulnerability_threshold = max(median_b * 0.3, 0.05)

        # --- Empathetic strain modulation ---
        if empathy:
            for ag in members:
                # Count groupmates whose defection is from constraint (low B), not choice
                constrained_count = 0
                for m in members:
                    if m is ag:
                        continue
                    if m.state.B < vulnerability_threshold:
                        constrained_count += 1

                if constrained_count > 0:
                    # Empathy strength = observer's social sensitivity (alpha)
                    empathy_strength = ag.params.alpha
                    constrained_frac = constrained_count / max(n - 1, 1)
                    # Reduce strain: understanding reduces frustration from structural gaps
                    # Max 30% reduction when all groupmates are constrained + full alpha
                    reduction = empathy_strength * constrained_frac * 0.3
                    ag.state.strain *= (1.0 - reduction)

        # --- Solidarity transfer ---
        if solidarity:
            above = [ag for ag in members if ag.state.B > median_b]
            below = [ag for ag in members if ag.state.B <= median_b and ag.state.B < median_b]
            if above and below:
                total_shared = 0.0
                for ag in above:
                    # Sharing gated by c_base: prosocial agents share more
                    # Rate: up to 10% of excess per round, scaled by dt
                    share_rate = ag.params.c_base * 0.1
                    excess = ag.state.B - median_b
                    share = excess * share_rate * dt
                    if share > 0:
                        ag.state.B -= share
                        total_shared += share

                if total_shared > 0:
                    per_recipient = total_shared / len(below)
                    for ag in below:
                        ag.state.B += per_recipient

        # --- Informed reference ---
        if reference:
            contribs = round_contribs.get(gid)
            if contribs is None:
                continue

            # Identify capable agents (those with budget to cooperate)
            capable_contribs = []
            for j, m in enumerate(members):
                if m.state.B > vulnerability_threshold and j < len(contribs):
                    capable_contribs.append(contribs[j])

            # Only adjust if some agents are constrained (otherwise no shift needed)
            if capable_contribs and len(capable_contribs) < n:
                capable_mean_norm = (sum(capable_contribs) /
                                     max(len(capable_contribs), 1)) / MAX_CONTRIB

                for ag in members:
                    # Blend toward capable-only reference
                    # Strength gated by alpha: socially sensitive agents adjust more
                    blend = ag.params.alpha * 0.3
                    ag.state.v_level = ((1.0 - blend) * ag.state.v_level +
                                        blend * capable_mean_norm)


# ================================================================
# SIMULATE — PHASE 3 EXTENDED LOOP
# ================================================================

def simulate_phase3(agents, pools, rng, n_rounds=100, mechanisms=None,
                     visibility=None):
    """
    Run n_rounds with pluggable mechanisms, including rehabilitation and visibility.

    Extends Phase 2 simulate() with:
      - 'rehabilitation' mechanism (strain reduction, budget support, replenishment
        boost, comprehensive, targeted, hybrid_rehab_first, hybrid_graduated)
      - Per-round boost application
      - Intervention cost accounting
      - Hybrid escalation (rehabilitate → remove)
      - Full visibility effects (empathy, solidarity, informed reference)

    mechanisms: list of (name, kwargs). Names:
      'none', 'punishment', 'threshold', 'sustainability', 'rehabilitation'
    visibility: dict with keys 'empathy', 'solidarity', 'reference' (bool each),
      or None for no visibility effects.
    """
    groups = assign_groups(agents)
    dt = 1.0 / (n_rounds - 1) if n_rounds > 1 else 1.0
    all_agents = list(agents)
    events = []
    uid_counter = max(ag.uid for ag in agents) + 1

    prev_c = {gid: None for gid in groups}
    prev_p = {gid: None for gid in groups}

    # Mechanism state
    below_count = {}                      # threshold
    g_budget_hist = defaultdict(list)     # sustainability / rehabilitation trigger
    g_coop_hist = defaultdict(list)
    degrading = defaultdict(int)

    # Intervention cost tracking
    total_intervention_cost = 0.0

    if mechanisms is None:
        mechanisms = [('none', {})]

    # Initialize rehab state for all agents
    for ag in agents:
        _init_rehab_state(ag)

    for rnd in range(n_rounds):
        mech_names = {m[0] for m in mechanisms}
        has_pun = 'punishment' in mech_names
        has_rehab = 'rehabilitation' in mech_names

        # --- Step all active groups ---
        round_contribs = {}
        for gid in list(groups.keys()):
            members = groups[gid]
            if not members:
                del groups[gid]
                continue

            contribs, ps, pr = step_group(
                members, rnd, n_rounds, dt,
                has_punishment=has_pun,
                prev_contribs=prev_c.get(gid),
                prev_pun_recv=prev_p.get(gid) if has_pun else None,
            )

            # Apply active replenishment boosts (after VCMS pass, before recording)
            if has_rehab:
                _apply_boosts(members, dt)

            record_round(members, contribs, ps, pr)
            round_contribs[gid] = contribs
            prev_c[gid] = contribs
            prev_p[gid] = pr if has_pun else None

            # Group-level tracking for trigger detection
            g_budget_hist[gid].append(
                sum(ag.state.B for ag in members) / max(len(members), 1))
            g_coop_hist[gid].append(
                sum(contribs) / max(len(contribs), 1))

        # --- Apply mechanisms in order ---
        for mech_name, mech_kwargs in mechanisms:

            if mech_name == 'threshold':
                K = mech_kwargs.get('K', 3)
                for gid in list(groups.keys()):
                    members = groups[gid]
                    if gid not in round_contribs or rnd == 0 or len(members) < 2:
                        continue
                    contribs = round_contribs[gid]
                    mean_c = sum(contribs) / len(contribs)
                    sd_c = (sum((c - mean_c) ** 2 for c in contribs) / len(contribs)) ** 0.5
                    thresh_val = mean_c - THRESHOLD_SD_MULT * sd_c

                    removals = []
                    for j, ag in enumerate(members):
                        if contribs[j] < thresh_val:
                            below_count[ag.uid] = below_count.get(ag.uid, 0) + 1
                        else:
                            below_count[ag.uid] = 0
                        if below_count.get(ag.uid, 0) >= K:
                            removals.append((j, ag))

                    for j, old_ag in sorted(removals, reverse=True, key=lambda x: x[0]):
                        old_ag.active_to = rnd
                        below_count.pop(old_ag.uid, None)
                        repl = create_replacement(
                            old_ag.phenotype, pools, rng, uid_counter,
                            active_from=rnd + 1)
                        uid_counter += 1
                        repl.group_id = gid
                        _init_rehab_state(repl)
                        members[j] = repl
                        all_agents.append(repl)
                        below_count[repl.uid] = 0
                        events.append({
                            'type': 'threshold_removal', 'round': rnd,
                            'group': gid, 'removed_uid': old_ag.uid,
                            'removed_phenotype': old_ag.phenotype,
                        })

            elif mech_name == 'sustainability':
                for gid in list(groups.keys()):
                    members = groups[gid]
                    if gid not in round_contribs or len(members) <= 1:
                        continue
                    if len(g_budget_hist[gid]) < HEALTH_WINDOW:
                        continue
                    contribs = round_contribs[gid]
                    b_sl = _linear_slope(g_budget_hist[gid][-HEALTH_WINDOW:])
                    c_sl = _linear_slope(g_coop_hist[gid][-HEALTH_WINDOW:])

                    if b_sl < 0 and c_sl < 0:
                        degrading[gid] += 1
                    else:
                        degrading[gid] = 0

                    if degrading[gid] >= HEALTH_TRIGGER_CONSEC:
                        total = sum(contribs)
                        n_m = len(members)
                        gmean = total / n_m
                        best_imp, worst_j = -float('inf'), -1
                        for j in range(n_m):
                            imp = (total - contribs[j]) / max(n_m - 1, 1) - gmean
                            if imp > best_imp:
                                best_imp, worst_j = imp, j

                        if worst_j >= 0:
                            old_ag = members[worst_j]
                            old_ag.active_to = rnd
                            repl = create_replacement(
                                old_ag.phenotype, pools, rng, uid_counter,
                                active_from=rnd + 1)
                            uid_counter += 1
                            repl.group_id = gid
                            _init_rehab_state(repl)
                            members[worst_j] = repl
                            all_agents.append(repl)
                            degrading[gid] = 0
                            events.append({
                                'type': 'sustainability_removal', 'round': rnd,
                                'group': gid, 'removed_uid': old_ag.uid,
                                'removed_phenotype': old_ag.phenotype,
                                'impact': best_imp,
                            })

            elif mech_name == 'rehabilitation':
                mode = mech_kwargs.get('mode', 'strain_reduction')

                for gid in list(groups.keys()):
                    members = groups[gid]
                    if gid not in round_contribs or len(members) <= 1:
                        continue
                    if len(g_budget_hist[gid]) < HEALTH_WINDOW:
                        continue
                    contribs = round_contribs[gid]

                    # --- Trigger detection (identical to sustainability) ---
                    b_sl = _linear_slope(g_budget_hist[gid][-HEALTH_WINDOW:])
                    c_sl = _linear_slope(g_coop_hist[gid][-HEALTH_WINDOW:])

                    if b_sl < 0 and c_sl < 0:
                        degrading[gid] += 1
                    else:
                        degrading[gid] = 0

                    if degrading[gid] < HEALTH_TRIGGER_CONSEC:
                        continue

                    # --- Trigger fired: intervene instead of remove ---

                    # Target selection depends on mode
                    if mode == 'strain_reduction':
                        target_j = _select_highest_strain(members)
                    elif mode == 'budget_support':
                        target_j = _select_lowest_budget(members)
                    elif mode == 'replenish_boost':
                        target_j = _select_lowest_ratio(members)
                    elif mode in ('comprehensive', 'targeted',
                                  'hybrid_rehab_first', 'hybrid_graduated'):
                        target_j = _select_highest_impact(members, contribs)
                    else:
                        target_j = _select_highest_impact(members, contribs)

                    target = members[target_j]
                    pre_strain = target.state.strain
                    pre_budget = target.state.B

                    # --- Hybrid logic ---
                    if mode == 'hybrid_rehab_first':
                        patience = mech_kwargs.get('patience', REHAB_PATIENCE)
                        if target._rehab_count > 0:
                            # Check if prior intervention worked
                            elapsed = rnd - target._rehab_last_round
                            if elapsed >= patience:
                                # Check improvement: current contribution vs at intervention
                                idx_now = rnd - target.active_from
                                c_now = (target.contrib_history[idx_now]
                                         if 0 <= idx_now < len(target.contrib_history) else 0)
                                if c_now <= target._rehab_last_contrib:
                                    # Not improved → remove
                                    target.active_to = rnd
                                    repl = create_replacement(
                                        target.phenotype, pools, rng, uid_counter,
                                        active_from=rnd + 1)
                                    uid_counter += 1
                                    repl.group_id = gid
                                    _init_rehab_state(repl)
                                    members[target_j] = repl
                                    all_agents.append(repl)
                                    degrading[gid] = 0
                                    events.append({
                                        'type': 'sustainability_removal', 'round': rnd,
                                        'group': gid, 'removed_uid': target.uid,
                                        'removed_phenotype': target.phenotype,
                                        'escalated_from': 'hybrid_rehab_first',
                                    })
                                    continue
                                # Improved → still in system, don't intervene again yet
                                degrading[gid] = 0
                                continue

                        # Apply comprehensive rehabilitation
                        _apply_comprehensive(target)
                        cost = _compute_cost('comprehensive', mech_kwargs)
                        _deduct_cost(members, cost)
                        total_intervention_cost += cost
                        target._rehab_count += 1
                        target._rehab_last_round = rnd
                        idx_at = rnd - target.active_from
                        target._rehab_last_contrib = (
                            target.contrib_history[idx_at]
                            if 0 <= idx_at < len(target.contrib_history) else 0)
                        target._rehab_history.append((rnd, 'comprehensive', {}))
                        degrading[gid] = 0
                        events.append({
                            'type': 'rehabilitation', 'round': rnd,
                            'group': gid, 'target_uid': target.uid,
                            'target_phenotype': target.phenotype,
                            'mode': 'hybrid_rehab_first', 'sub_mode': 'comprehensive',
                            'cost': cost,
                            'pre_strain': pre_strain, 'pre_budget': pre_budget,
                            'post_strain': target.state.strain,
                            'post_budget': target.state.B,
                        })
                        continue

                    elif mode == 'hybrid_graduated':
                        prior_count = target._rehab_count
                        if prior_count >= 2:
                            # Exhausted rehabilitation → remove
                            target.active_to = rnd
                            repl = create_replacement(
                                target.phenotype, pools, rng, uid_counter,
                                active_from=rnd + 1)
                            uid_counter += 1
                            repl.group_id = gid
                            _init_rehab_state(repl)
                            members[target_j] = repl
                            all_agents.append(repl)
                            degrading[gid] = 0
                            events.append({
                                'type': 'sustainability_removal', 'round': rnd,
                                'group': gid, 'removed_uid': target.uid,
                                'removed_phenotype': target.phenotype,
                                'escalated_from': 'hybrid_graduated',
                                'prior_interventions': prior_count,
                            })
                            continue
                        elif prior_count == 1:
                            # Second trigger → comprehensive
                            _apply_comprehensive(target)
                            sub_mode = 'comprehensive'
                            cost = _compute_cost('comprehensive', mech_kwargs)
                        else:
                            # First trigger → strain reduction (mildest)
                            _apply_strain_reduction(target, 0.50)
                            sub_mode = 'strain_reduction'
                            cost = _compute_cost('strain_reduction',
                                                 {'reduction': 0.50, **mech_kwargs})

                        _deduct_cost(members, cost)
                        total_intervention_cost += cost
                        target._rehab_count += 1
                        target._rehab_last_round = rnd
                        idx_at = rnd - target.active_from
                        target._rehab_last_contrib = (
                            target.contrib_history[idx_at]
                            if 0 <= idx_at < len(target.contrib_history) else 0)
                        target._rehab_history.append((rnd, sub_mode, {}))
                        degrading[gid] = 0
                        events.append({
                            'type': 'rehabilitation', 'round': rnd,
                            'group': gid, 'target_uid': target.uid,
                            'target_phenotype': target.phenotype,
                            'mode': 'hybrid_graduated', 'sub_mode': sub_mode,
                            'cost': cost,
                            'pre_strain': pre_strain, 'pre_budget': pre_budget,
                            'post_strain': target.state.strain,
                            'post_budget': target.state.B,
                        })
                        continue

                    # --- Standard (non-hybrid) rehabilitation ---
                    sub_mode = mode
                    if mode == 'strain_reduction':
                        reduction = mech_kwargs.get('reduction', 0.50)
                        _apply_strain_reduction(target, reduction)
                    elif mode == 'budget_support':
                        amount = mech_kwargs.get('amount', 1.0)
                        _apply_budget_support(target, amount)
                    elif mode == 'replenish_boost':
                        factor = mech_kwargs.get('factor', 1.5)
                        duration = mech_kwargs.get('duration', REHAB_BOOST_DURATION)
                        _apply_replenish_boost(target, factor, duration)
                    elif mode == 'comprehensive':
                        _apply_comprehensive(target)
                    elif mode == 'targeted':
                        sub_mode = _apply_targeted(target, members)

                    cost = _compute_cost(mode, mech_kwargs)
                    _deduct_cost(members, cost)
                    total_intervention_cost += cost

                    target._rehab_count += 1
                    target._rehab_last_round = rnd
                    idx_at = rnd - target.active_from
                    target._rehab_last_contrib = (
                        target.contrib_history[idx_at]
                        if 0 <= idx_at < len(target.contrib_history) else 0)
                    target._rehab_history.append((rnd, sub_mode, dict(mech_kwargs)))
                    degrading[gid] = 0

                    events.append({
                        'type': 'rehabilitation', 'round': rnd,
                        'group': gid, 'target_uid': target.uid,
                        'target_phenotype': target.phenotype,
                        'mode': mode, 'sub_mode': sub_mode,
                        'cost': cost,
                        'pre_strain': pre_strain, 'pre_budget': pre_budget,
                        'post_strain': target.state.strain,
                        'post_budget': target.state.B,
                    })

        # Apply visibility effects (end of round, after all mechanisms)
        if visibility:
            _apply_visibility(groups, round_contribs, dt, visibility)

        # Clean up empty groups
        for gid in list(groups.keys()):
            if not groups[gid]:
                del groups[gid]
                prev_c.pop(gid, None)

    return {
        'agents': all_agents,
        'groups': groups,
        'events': events,
        'total_intervention_cost': total_intervention_cost,
    }


# ================================================================
# PHASE 3 METRICS
# ================================================================

def compute_phase3_metrics(result, n_rounds=100):
    """Phase 1-2 metrics plus rehabilitation-specific metrics."""
    base = compute_phase2_metrics(result, n_rounds)
    events = result['events']
    agents = result['agents']

    interventions = [e for e in events if e['type'] == 'rehabilitation']
    removals = [e for e in events if e['type'] == 'sustainability_removal']

    base['intervention_count'] = len(interventions)
    base['total_intervention_cost'] = result.get('total_intervention_cost', 0.0)
    base['hybrid_removal_count'] = len(removals)

    # Build uid → agent lookup for efficiency
    uid_map = {}
    for ag in agents:
        uid_map[ag.uid] = ag

    # Conversion rate: fraction of interventions where target's contribution
    # improved within IMPROVEMENT_WINDOW rounds
    conversions = 0
    measurable = 0
    for e in interventions:
        uid = e['target_uid']
        rnd = e['round']
        agent = uid_map.get(uid)
        if agent is None:
            continue
        idx_at = rnd - agent.active_from
        idx_after = idx_at + IMPROVEMENT_WINDOW
        if idx_at >= 0 and idx_after < len(agent.contrib_history):
            measurable += 1
            c_before = agent.contrib_history[idx_at]
            c_after = agent.contrib_history[idx_after]
            if c_after > c_before:
                conversions += 1
    base['conversion_rate'] = conversions / max(measurable, 1)

    # Recidivism rate: improved within 10 rounds but regressed within 20
    recidivists = 0
    converted_count = 0
    for e in interventions:
        uid = e['target_uid']
        rnd = e['round']
        agent = uid_map.get(uid)
        if agent is None:
            continue
        idx_at = rnd - agent.active_from
        idx_10 = idx_at + IMPROVEMENT_WINDOW
        idx_20 = idx_at + RECIDIVISM_WINDOW
        if idx_at >= 0 and idx_10 < len(agent.contrib_history):
            c_before = agent.contrib_history[idx_at]
            c_10 = agent.contrib_history[idx_10]
            if c_10 > c_before:
                converted_count += 1
                if idx_20 < len(agent.contrib_history):
                    c_20 = agent.contrib_history[idx_20]
                    if c_20 <= c_before:
                        recidivists += 1
    base['recidivism_rate'] = recidivists / max(converted_count, 1)

    # Cost efficiency: steady-state cooperation per unit of intervention cost
    ss_coop = base.get('steady_state_coop', 0)
    total_cost = base['total_intervention_cost']
    base['cost_efficiency'] = ss_coop / max(total_cost, 0.01)

    # Per-phenotype intervention response
    for ptype in ['CC', 'EC', 'CD', 'DL', 'MX']:
        ptype_interv = [e for e in interventions
                        if e.get('target_phenotype') == ptype]
        ptype_conv = 0
        ptype_meas = 0
        for e in ptype_interv:
            uid = e['target_uid']
            rnd = e['round']
            agent = uid_map.get(uid)
            if agent is None:
                continue
            idx_at = rnd - agent.active_from
            idx_after = idx_at + IMPROVEMENT_WINDOW
            if idx_at >= 0 and idx_after < len(agent.contrib_history):
                ptype_meas += 1
                if agent.contrib_history[idx_after] > agent.contrib_history[idx_at]:
                    ptype_conv += 1
        base[f'{ptype}_intervention_count'] = len(ptype_interv)
        base[f'{ptype}_conversion_rate'] = ptype_conv / max(ptype_meas, 1)

    return base


# ================================================================
# CONDITION DEFINITIONS
# ================================================================

REHAB_CONDITIONS = {
    'B1_baseline':      [('none', {})],
    'B2_sustain':       [('sustainability', {})],
    'R1a_strain25':     [('rehabilitation', {'mode': 'strain_reduction', 'reduction': 0.25})],
    'R1b_strain50':     [('rehabilitation', {'mode': 'strain_reduction', 'reduction': 0.50})],
    'R1c_strain75':     [('rehabilitation', {'mode': 'strain_reduction', 'reduction': 0.75})],
    'R2a_budget05':     [('rehabilitation', {'mode': 'budget_support', 'amount': 0.5})],
    'R2b_budget10':     [('rehabilitation', {'mode': 'budget_support', 'amount': 1.0})],
    'R2c_budget20':     [('rehabilitation', {'mode': 'budget_support', 'amount': 2.0})],
    'R3a_boost15':      [('rehabilitation', {'mode': 'replenish_boost', 'factor': 1.5, 'duration': 10})],
    'R3b_boost20':      [('rehabilitation', {'mode': 'replenish_boost', 'factor': 2.0, 'duration': 10})],
    'R3c_boost30':      [('rehabilitation', {'mode': 'replenish_boost', 'factor': 3.0, 'duration': 10})],
    'R4_comprehensive': [('rehabilitation', {'mode': 'comprehensive'})],
    'R5_targeted':      [('rehabilitation', {'mode': 'targeted'})],
    'H1_rehab_first':   [('rehabilitation', {'mode': 'hybrid_rehab_first', 'patience': 10})],
    'H2_graduated':     [('rehabilitation', {'mode': 'hybrid_graduated'})],
}


# ================================================================
# RUNNER
# ================================================================

def run_phase3(n_runs=N_RUNS, n_rounds=100, seed=42):
    """Run all Phase 3 rehabilitation conditions."""
    print("=" * 100)
    print("FEDERATION DYNAMICS PHASE 3")
    print("Rehabilitation: Parameter Intervention as Alternative to Removal")
    print("=" * 100)

    print("\nLoading libraries and building pools...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    for name, pool in pools.items():
        print(f"  {name}: {len(pool)} subjects")

    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in REHAB_CONDITIONS}

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Run {run + 1}/{n_runs}  ({elapsed:.1f}s)")
        bp = sample_population_blueprint(pools, rng)

        for cond_name, mechs in REHAB_CONDITIONS.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2 ** 31))
            result = simulate_phase3(agents, pools, cond_rng, n_rounds,
                                     mechanisms=mechs)
            metrics = compute_phase3_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    elapsed = time.time() - t0
    print(f"\nAll simulations complete: {elapsed:.1f}s "
          f"({n_runs * len(REHAB_CONDITIONS)} runs, "
          f"~{n_runs * len(REHAB_CONDITIONS) * n_rounds * N_AGENTS / 1e6:.0f}M agent-steps)")

    # Aggregate
    aggregated = {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}

    # Report
    print_condition_comparison(aggregated)
    print_phenotype_response(aggregated)
    print_cost_analysis(aggregated)
    print_frontier_analysis(aggregated)
    print_predictions_scorecard(aggregated)

    return aggregated


# ================================================================
# REPORTING
# ================================================================

def print_condition_comparison(results):
    """Main condition comparison table."""
    print("\n" + "=" * 120)
    print("CONDITION COMPARISON")
    print("=" * 120)

    header = (f"{'Condition':<20} {'SS-Coop':>8} {'TTFR':>5} {'Rupt':>5} "
              f"{'Gini':>6} {'Intrv':>6} {'Conv%':>6} {'Recid%':>7} "
              f"{'Cost':>7} {'CostEff':>8} {'Removals':>8}")
    print(header)
    print("-" * 120)

    for name in REHAB_CONDITIONS:
        agg = results[name]
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        rupt = agg.get('rupture_count', {}).get('median', 0)
        gini = agg.get('gini', {}).get('median', 0)
        interv = agg.get('intervention_count', {}).get('median', 0)
        conv = agg.get('conversion_rate', {}).get('median', 0)
        recid = agg.get('recidivism_rate', {}).get('median', 0)
        cost = agg.get('total_intervention_cost', {}).get('median', 0)
        ceff = agg.get('cost_efficiency', {}).get('median', 0)
        removals = agg.get('hybrid_removal_count', {}).get('median', 0)
        sustain_rm = agg.get('sustain_removal_count', agg.get('sustain_removals', {}))
        if isinstance(sustain_rm, dict):
            sustain_rm = sustain_rm.get('median', 0)
        total_rm = removals + sustain_rm if isinstance(sustain_rm, (int, float)) else removals
        print(f"{name:<20} {ss:>8.1f} {ttfr:>5.0f} {rupt:>5.0f} "
              f"{gini:>6.3f} {interv:>6.0f} {conv:>6.1%} {recid:>7.1%} "
              f"{cost:>7.1f} {ceff:>8.1f} {total_rm:>8.0f}")


def print_phenotype_response(results):
    """Per-phenotype intervention response table."""
    print("\n" + "=" * 120)
    print("PER-PHENOTYPE INTERVENTION RESPONSE")
    print("=" * 120)

    rehab_conds = [n for n in REHAB_CONDITIONS if n.startswith(('R', 'H'))]

    header = f"{'Condition':<20}"
    for pt in ['CD', 'DL', 'EC', 'CC']:
        header += f" {'#' + pt:>5} {'Conv':>5}"
    print(header)
    print("-" * 80)

    for name in rehab_conds:
        agg = results[name]
        row = f"{name:<20}"
        for pt in ['CD', 'DL', 'EC', 'CC']:
            cnt = agg.get(f'{pt}_intervention_count', {}).get('median', 0)
            conv = agg.get(f'{pt}_conversion_rate', {}).get('median', 0)
            row += f" {cnt:>5.0f} {conv:>5.1%}"
        print(row)


def print_cost_analysis(results):
    """Cost-adjusted comparison."""
    print("\n" + "=" * 120)
    print("COST-ADJUSTED ANALYSIS")
    print("=" * 120)

    b2_ss = results['B2_sustain'].get('steady_state_coop', {}).get('median', 0)
    b2_ttfr = results['B2_sustain'].get('system_ttfr', {}).get('median', 0)

    header = (f"{'Condition':<20} {'SS-Coop':>8} {'vs B2':>7} {'TTFR':>5} "
              f"{'vs B2':>6} {'Cost':>7} {'Coop/Cost':>9}")
    print(header)
    print("-" * 80)

    for name in REHAB_CONDITIONS:
        agg = results[name]
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        cost = agg.get('total_intervention_cost', {}).get('median', 0)
        coop_per_cost = ss / max(cost, 0.01)
        ss_diff = ss - b2_ss
        ttfr_diff = ttfr - b2_ttfr
        print(f"{name:<20} {ss:>8.1f} {ss_diff:>+7.1f} {ttfr:>5.0f} "
              f"{ttfr_diff:>+6.0f} {cost:>7.1f} {coop_per_cost:>9.1f}")


def print_frontier_analysis(results):
    """Check if any rehabilitation condition sits above the compliance-sustainability frontier."""
    print("\n" + "=" * 120)
    print("FRONTIER ANALYSIS: Does rehabilitation move the compliance-sustainability curve?")
    print("=" * 120)

    b2_ss = results['B2_sustain'].get('steady_state_coop', {}).get('median', 0)
    b2_ttfr = results['B2_sustain'].get('system_ttfr', {}).get('median', 0)
    b1_ss = results['B1_baseline'].get('steady_state_coop', {}).get('median', 0)
    b1_ttfr = results['B1_baseline'].get('system_ttfr', {}).get('median', 0)

    print(f"  B1 baseline:  SS-Coop={b1_ss:.1f}, TTFR={b1_ttfr:.0f}")
    print(f"  B2 sustain:   SS-Coop={b2_ss:.1f}, TTFR={b2_ttfr:.0f}")
    print(f"  Frontier: any condition with SS-Coop >= B2 AND TTFR >= B2 is above the frontier")
    print()

    above_frontier = []
    for name in REHAB_CONDITIONS:
        if name.startswith('B'):
            continue
        agg = results[name]
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        above = ss >= b2_ss and ttfr >= b2_ttfr
        marker = " <<<" if above else ""
        if above:
            above_frontier.append(name)
        print(f"  {name:<20} SS-Coop={ss:>6.1f} ({ss - b2_ss:>+5.1f})  "
              f"TTFR={ttfr:>4.0f} ({ttfr - b2_ttfr:>+4.0f}){marker}")

    print()
    if above_frontier:
        print(f"  ABOVE FRONTIER: {', '.join(above_frontier)}")
        print("  → Rehabilitation is not just an alternative — it's a strictly superior operation")
    else:
        print("  NO CONDITIONS ABOVE FRONTIER")
        print("  → Rehabilitation trades off against removal; case for it is values-based, not physics-based")


def print_predictions_scorecard(results):
    """Evaluate all 6 Phase 3 predictions."""
    print("\n" + "=" * 120)
    print("PHASE 3 PREDICTIONS SCORECARD")
    print("=" * 120)

    scored = []

    # P1: Rehabilitation matches removal on cooperation for at least one condition
    b2_ss = results['B2_sustain'].get('steady_state_coop', {}).get('median', 0)
    best_rehab_ss = 0
    best_rehab_name = ''
    for name in REHAB_CONDITIONS:
        if name.startswith('B'):
            continue
        ss = results[name].get('steady_state_coop', {}).get('median', 0)
        if ss > best_rehab_ss:
            best_rehab_ss = ss
            best_rehab_name = name
    within_10pct = best_rehab_ss >= b2_ss * 0.90
    p1 = within_10pct
    scored.append(('P1', f"Rehabilitation matches removal on cooperation (within 10%): "
                         f"best rehab={best_rehab_name} SS={best_rehab_ss:.1f} vs B2={b2_ss:.1f} "
                         f"({best_rehab_ss / max(b2_ss, 0.01):.1%})",
                   p1))

    # P2: Rehabilitation exceeds removal on TTFR
    b2_ttfr = results['B2_sustain'].get('system_ttfr', {}).get('median', 0)
    best_rehab_ttfr = 0
    best_ttfr_name = ''
    for name in REHAB_CONDITIONS:
        if name.startswith('B'):
            continue
        ttfr = results[name].get('system_ttfr', {}).get('median', 0)
        if ttfr > best_rehab_ttfr:
            best_rehab_ttfr = ttfr
            best_ttfr_name = name
    p2 = best_rehab_ttfr > b2_ttfr
    scored.append(('P2', f"Rehabilitation exceeds removal on TTFR: "
                         f"best={best_ttfr_name} TTFR={best_rehab_ttfr:.0f} vs B2={b2_ttfr:.0f}",
                   p2))

    # P3: Strain reduction works for CD, replenishment boost works for DL
    r1b_cd_conv = results['R1b_strain50'].get('CD_conversion_rate', {}).get('median', 0)
    r3b_dl_conv = results['R3b_boost20'].get('DL_conversion_rate', {}).get('median', 0)
    # Check if R5 targeted outperforms uniform
    r5_conv = results['R5_targeted'].get('conversion_rate', {}).get('median', 0)
    r1b_conv = results['R1b_strain50'].get('conversion_rate', {}).get('median', 0)
    r3b_conv = results['R3b_boost20'].get('conversion_rate', {}).get('median', 0)
    uniform_avg = (r1b_conv + r3b_conv) / 2.0
    p3_cd = r1b_cd_conv > 0.3  # meaningful conversion for CD under strain reduction
    p3_dl = r3b_dl_conv > 0.3  # meaningful conversion for DL under replenishment boost
    p3_targeted = r5_conv > uniform_avg
    p3 = p3_cd and p3_dl
    scored.append(('P3', f"Strain works for CD (conv={r1b_cd_conv:.1%}), "
                         f"boost works for DL (conv={r3b_dl_conv:.1%}), "
                         f"targeted > uniform ({r5_conv:.1%} vs {uniform_avg:.1%}): "
                         f"CD={'yes' if p3_cd else 'no'}, DL={'yes' if p3_dl else 'no'}, "
                         f"targeted={'yes' if p3_targeted else 'no'}",
                   p3))

    # P4: Hybrid graduated (H2) outperforms both pure removal and pure rehabilitation
    def _composite(agg):
        ss = agg.get('steady_state_coop', {}).get('median', 0) / MAX_CONTRIB
        ttfr = min(agg.get('system_ttfr', {}).get('median', 0), 100) / 100
        rupt = 1.0 - min(agg.get('rupture_count', {}).get('median', 0), N_AGENTS) / N_AGENTS
        return (ss + ttfr + rupt) / 3.0

    h2_score = _composite(results['H2_graduated'])
    b2_score = _composite(results['B2_sustain'])
    r4_score = _composite(results['R4_comprehensive'])
    p4 = h2_score > b2_score and h2_score > r4_score
    scored.append(('P4', f"H2 graduated > both B2 and R4 on composite: "
                         f"H2={h2_score:.3f} vs B2={b2_score:.3f}, R4={r4_score:.3f}",
                   p4))

    # P5: Rehabilitation reduces removal rate below B2's
    b2_removals = results['B2_sustain'].get('sustain_removals',
                  results['B2_sustain'].get('hybrid_removal_count', {}))
    if isinstance(b2_removals, dict):
        b2_removals = b2_removals.get('median', 0)
    h2_removals = results['H2_graduated'].get('hybrid_removal_count', {}).get('median', 0)
    p5 = h2_removals < b2_removals
    scored.append(('P5', f"H2 reduces removals below B2: "
                         f"H2={h2_removals:.0f} vs B2={b2_removals:.0f}",
                   p5))

    # P6: Break-even cost is positive (rehabilitation is dynamically competitive)
    # Find if any rehab condition matches B2 cooperation and has positive cost
    competitive_with_cost = False
    for name in REHAB_CONDITIONS:
        if name.startswith('B'):
            continue
        ss = results[name].get('steady_state_coop', {}).get('median', 0)
        cost = results[name].get('total_intervention_cost', {}).get('median', 0)
        if ss >= b2_ss * 0.90 and cost > 0:
            competitive_with_cost = True
            break
    p6 = competitive_with_cost
    scored.append(('P6', f"Break-even cost is positive (rehab competitive at positive cost): "
                         f"{'found' if competitive_with_cost else 'not found'}",
                   p6))

    # Print scorecard
    supported = sum(1 for _, _, s in scored if s)
    total = len(scored)
    print(f"\n  Score: {supported}/{total} predictions supported\n")
    for pid, desc, result in scored:
        status = "SUPPORTED" if result else "NOT SUPPORTED"
        print(f"  {pid}: {status}")
        print(f"      {desc}")
        print()


# ================================================================
# PHASE 3B: FULL VISIBILITY EXTENSION
# ================================================================
#
# All agents see all groupmates' parameters and state. Tests whether
# visibility can increase navigable agency while maintaining baseline
# dignity. Varies mechanisms with visibility always ON.

FULL_VIS = {'empathy': True, 'solidarity': True, 'reference': True}
EMPATHY_ONLY = {'empathy': True, 'solidarity': False, 'reference': False}
SOLIDARITY_ONLY = {'empathy': False, 'solidarity': True, 'reference': False}
REFERENCE_ONLY = {'empathy': False, 'solidarity': False, 'reference': True}

VISIBILITY_CONDITIONS = {
    # --- Controls (no visibility) ---
    'B1_none':           ([('none', {})],           None),
    'B2_sustain':        ([('sustainability', {})],  None),
    'B3_rehab_grad':     ([('rehabilitation', {'mode': 'hybrid_graduated'})], None),
    # --- Visibility components (no enforcement) ---
    'V1_empathy':        ([('none', {})],           EMPATHY_ONLY),
    'V2_solidarity':     ([('none', {})],           SOLIDARITY_ONLY),
    'V3_reference':      ([('none', {})],           REFERENCE_ONLY),
    'V4_full_vis':       ([('none', {})],           FULL_VIS),
    # --- Visibility + enforcement ---
    'V5_vis+sustain':    ([('sustainability', {})],  FULL_VIS),
    'V6_vis+rehab_comp': ([('rehabilitation', {'mode': 'comprehensive'})], FULL_VIS),
    'V7_vis+rehab_tgt':  ([('rehabilitation', {'mode': 'targeted'})],      FULL_VIS),
    'V8_vis+graduated':  ([('rehabilitation', {'mode': 'hybrid_graduated'})], FULL_VIS),
}


def compute_visibility_metrics(result, n_rounds=100):
    """Phase 3 metrics plus dignity, agency, and navigability indices."""
    base = compute_phase3_metrics(result, n_rounds)
    agents = result['agents']

    active_agents = [ag for ag in agents if ag.active_to == -1]

    # --- Dignity metrics ---

    # Per-phenotype survival rates
    phenotype_survivals = {}
    for ptype in ['CC', 'EC', 'CD', 'DL', 'MX']:
        total = sum(1 for ag in agents if ag.phenotype == ptype)
        survived = sum(1 for ag in active_agents if ag.phenotype == ptype
                       and not _is_ruptured(ag))
        phenotype_survivals[ptype] = survived / max(total, 1)

    # Dignity floor: minimum survival rate across phenotypes with agents present
    valid_survivals = [v for pt, v in phenotype_survivals.items()
                       if sum(1 for ag in agents if ag.phenotype == pt) > 0]
    base['dignity_floor'] = min(valid_survivals) if valid_survivals else 0.0

    # Budget floor: 10th percentile of final budgets
    final_budgets = []
    for ag in active_agents:
        if ag.budget_history:
            final_budgets.append(ag.budget_history[-1])
    if final_budgets:
        sorted_fb = sorted(final_budgets)
        p10_idx = max(0, len(sorted_fb) // 10)
        base['budget_floor_p10'] = sorted_fb[p10_idx]
    else:
        base['budget_floor_p10'] = 0.0

    # Sustained dignity: fraction of agents maintaining B > 20% of initial throughout
    sustained_count = 0
    for ag in active_agents:
        threshold = ag.params.b_initial * 0.2
        if ag.budget_history and all(b >= threshold for b in ag.budget_history):
            sustained_count += 1
    base['sustained_dignity'] = sustained_count / max(len(active_agents), 1)

    # --- Agency metrics ---

    # Cooperation agency: SD of steady-state cooperation (diversity of choices)
    final_contribs = []
    for ag in active_agents:
        if ag.contrib_history:
            final_contribs.append(ag.contrib_history[-1])
    if len(final_contribs) > 1:
        mean_fc = sum(final_contribs) / len(final_contribs)
        base['cooperation_agency'] = (
            sum((c - mean_fc) ** 2 for c in final_contribs) / len(final_contribs)
        ) ** 0.5
    else:
        base['cooperation_agency'] = 0.0

    # Budget-cooperation correlation: do choices lead to outcomes?
    if len(active_agents) > 3:
        contribs_for_corr = []
        budgets_for_corr = []
        for ag in active_agents:
            if ag.contrib_history and ag.budget_history:
                contribs_for_corr.append(sum(ag.contrib_history) / len(ag.contrib_history))
                budgets_for_corr.append(ag.budget_history[-1])
        if len(contribs_for_corr) > 3:
            base['budget_coop_corr'] = _pearson(contribs_for_corr, budgets_for_corr)
        else:
            base['budget_coop_corr'] = 0.0
    else:
        base['budget_coop_corr'] = 0.0

    # --- Navigability index ---
    # Combines dignity (everyone survives) and agency (choices matter)
    # navigability = dignity_floor * (1 - gini) * (1 + normalized_agency)
    gini_val = base.get('gini', 0.5)
    agency_norm = base['cooperation_agency'] / max(MAX_CONTRIB, 1)
    base['navigability'] = base['dignity_floor'] * (1.0 - gini_val) * (1.0 + agency_norm)

    return base


def run_visibility_test(n_runs=N_RUNS, n_rounds=100, seed=45):
    """Run all visibility conditions."""
    print("\n" + "=" * 120)
    print("PHASE 3B: FULL VISIBILITY — NAVIGABLE AGENCY AND BASELINE DIGNITY")
    print("=" * 120)

    print("\nLoading libraries and building pools...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    for name, pool in pools.items():
        print(f"  {name}: {len(pool)} subjects")

    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in VISIBILITY_CONDITIONS}

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Run {run + 1}/{n_runs}  ({elapsed:.1f}s)")
        bp = sample_population_blueprint(pools, rng)

        for cond_name, (mechs, vis) in VISIBILITY_CONDITIONS.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2 ** 31))
            result = simulate_phase3(agents, pools, cond_rng, n_rounds,
                                     mechanisms=mechs, visibility=vis)
            metrics = compute_visibility_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    elapsed = time.time() - t0
    n_conds = len(VISIBILITY_CONDITIONS)
    print(f"\nAll simulations complete: {elapsed:.1f}s "
          f"({n_runs * n_conds} runs, "
          f"~{n_runs * n_conds * n_rounds * N_AGENTS / 1e6:.0f}M agent-steps)")

    aggregated = {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}

    print_visibility_comparison(aggregated)
    print_dignity_analysis(aggregated)
    print_phenotype_dignity(aggregated)
    print_agency_analysis(aggregated)
    print_visibility_predictions(aggregated)

    return aggregated


# ================================================================
# VISIBILITY REPORTING
# ================================================================

def print_visibility_comparison(results):
    """Main visibility condition comparison."""
    print("\n" + "=" * 130)
    print("VISIBILITY CONDITION COMPARISON")
    print("=" * 130)

    header = (f"{'Condition':<20} {'SS-Coop':>8} {'TTFR':>5} {'Rupt':>5} "
              f"{'Gini':>6} {'DignFl':>7} {'BudFl':>6} {'SustDig':>7} "
              f"{'Agency':>7} {'B-C r':>6} {'Navig':>6} "
              f"{'Intrv':>5} {'Remov':>5}")
    print(header)
    print("-" * 130)

    for name in VISIBILITY_CONDITIONS:
        agg = results[name]
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        rupt = agg.get('rupture_count', {}).get('median', 0)
        gini = agg.get('gini', {}).get('median', 0)
        dfloor = agg.get('dignity_floor', {}).get('median', 0)
        bfloor = agg.get('budget_floor_p10', {}).get('median', 0)
        sdig = agg.get('sustained_dignity', {}).get('median', 0)
        agency = agg.get('cooperation_agency', {}).get('median', 0)
        bcorr = agg.get('budget_coop_corr', {}).get('median', 0)
        navig = agg.get('navigability', {}).get('median', 0)
        interv = agg.get('intervention_count', {}).get('median', 0)
        remov = agg.get('hybrid_removal_count', {}).get('median', 0)
        print(f"{name:<20} {ss:>8.1f} {ttfr:>5.0f} {rupt:>5.0f} "
              f"{gini:>6.3f} {dfloor:>7.1%} {bfloor:>6.2f} {sdig:>7.1%} "
              f"{agency:>7.2f} {bcorr:>+6.2f} {navig:>6.3f} "
              f"{interv:>5.0f} {remov:>5.0f}")


def print_dignity_analysis(results):
    """Dignity floor analysis — who maintains baseline dignity?"""
    print("\n" + "=" * 130)
    print("DIGNITY ANALYSIS: Which configurations maintain baseline dignity for ALL phenotypes?")
    print("=" * 130)

    b1_dfloor = results['B1_none'].get('dignity_floor', {}).get('median', 0)
    b2_dfloor = results['B2_sustain'].get('dignity_floor', {}).get('median', 0)

    print(f"  B1 baseline dignity floor:  {b1_dfloor:.1%}")
    print(f"  B2 sustain dignity floor:   {b2_dfloor:.1%}")
    print()

    for name in VISIBILITY_CONDITIONS:
        if name.startswith('B'):
            continue
        agg = results[name]
        dfloor = agg.get('dignity_floor', {}).get('median', 0)
        sdig = agg.get('sustained_dignity', {}).get('median', 0)
        rupt = agg.get('rupture_count', {}).get('median', 0)
        vs_b1 = dfloor - b1_dfloor
        vs_b2 = dfloor - b2_dfloor
        marker = " <<<" if dfloor > b2_dfloor else ""
        print(f"  {name:<20} floor={dfloor:>6.1%} (vs B1: {vs_b1:>+5.1%}, "
              f"vs B2: {vs_b2:>+5.1%})  sustained={sdig:>5.1%}  "
              f"ruptures={rupt:>4.0f}{marker}")


def print_phenotype_dignity(results):
    """Per-phenotype survival under each visibility condition."""
    print("\n" + "=" * 130)
    print("PER-PHENOTYPE SURVIVAL (dignity by type)")
    print("=" * 130)

    header = f"{'Condition':<20}"
    for pt in ['EC', 'CC', 'CD', 'DL']:
        header += f" {'Surv_' + pt:>8} {'Bud_' + pt:>8}"
    print(header)
    print("-" * 100)

    for name in VISIBILITY_CONDITIONS:
        agg = results[name]
        row = f"{name:<20}"
        for pt in ['EC', 'CC', 'CD', 'DL']:
            surv = agg.get(f'{pt}_survival', {}).get('median', 0)
            bud = agg.get(f'{pt}_budget_T100', {}).get('median', 0)
            row += f" {surv:>8.1%} {bud:>8.2f}"
        print(row)


def print_agency_analysis(results):
    """Agency analysis — do choices matter under visibility?"""
    print("\n" + "=" * 130)
    print("AGENCY ANALYSIS: Does visibility enable meaningful choice?")
    print("=" * 130)

    header = (f"{'Condition':<20} {'Agency':>7} {'B-C Corr':>8} {'Gini':>6} "
              f"{'Navig':>7} {'SS-Coop':>8}")
    print(header)
    print("-" * 70)

    for name in VISIBILITY_CONDITIONS:
        agg = results[name]
        agency = agg.get('cooperation_agency', {}).get('median', 0)
        bcorr = agg.get('budget_coop_corr', {}).get('median', 0)
        gini = agg.get('gini', {}).get('median', 0)
        navig = agg.get('navigability', {}).get('median', 0)
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        print(f"{name:<20} {agency:>7.2f} {bcorr:>+8.3f} {gini:>6.3f} "
              f"{navig:>7.3f} {ss:>8.1f}")


def print_visibility_predictions(results):
    """Evaluate visibility-specific predictions."""
    print("\n" + "=" * 130)
    print("VISIBILITY PREDICTIONS SCORECARD")
    print("=" * 130)

    scored = []

    b1 = results['B1_none']
    b2 = results['B2_sustain']
    v4 = results['V4_full_vis']
    v8 = results['V8_vis+graduated']

    # PV1: Full visibility alone (no enforcement) improves dignity floor over baseline
    b1_dfl = b1.get('dignity_floor', {}).get('median', 0)
    v4_dfl = v4.get('dignity_floor', {}).get('median', 0)
    pv1 = v4_dfl > b1_dfl
    scored.append(('PV1', f"Full visibility improves dignity floor over baseline: "
                          f"V4={v4_dfl:.1%} vs B1={b1_dfl:.1%}",
                   pv1))

    # PV2: Solidarity is the strongest single visibility component
    v1_dfl = results['V1_empathy'].get('dignity_floor', {}).get('median', 0)
    v2_dfl = results['V2_solidarity'].get('dignity_floor', {}).get('median', 0)
    v3_dfl = results['V3_reference'].get('dignity_floor', {}).get('median', 0)
    strongest = max([('empathy', v1_dfl), ('solidarity', v2_dfl),
                     ('reference', v3_dfl)], key=lambda x: x[1])
    pv2 = strongest[0] == 'solidarity'
    scored.append(('PV2', f"Solidarity is strongest component: "
                          f"empathy={v1_dfl:.1%}, solidarity={v2_dfl:.1%}, "
                          f"reference={v3_dfl:.1%} → strongest={strongest[0]}",
                   pv2))

    # PV3: Visibility + graduated hybrid outperforms either alone on navigability
    b3_nav = results['B3_rehab_grad'].get('navigability', {}).get('median', 0)
    v4_nav = v4.get('navigability', {}).get('median', 0)
    v8_nav = v8.get('navigability', {}).get('median', 0)
    pv3 = v8_nav > max(b3_nav, v4_nav)
    scored.append(('PV3', f"Vis+graduated > either alone on navigability: "
                          f"V8={v8_nav:.3f} vs B3_grad={b3_nav:.3f}, V4_vis={v4_nav:.3f}",
                   pv3))

    # PV4: Full visibility reduces rupture count below baseline
    b1_rupt = b1.get('rupture_count', {}).get('median', 0)
    v4_rupt = v4.get('rupture_count', {}).get('median', 0)
    pv4 = v4_rupt < b1_rupt
    scored.append(('PV4', f"Full visibility reduces ruptures below baseline: "
                          f"V4={v4_rupt:.0f} vs B1={b1_rupt:.0f}",
                   pv4))

    # PV5: Visibility + sustainability produces higher navigability than
    #       sustainability alone (visibility enhances care-first enforcement)
    b2_nav = b2.get('navigability', {}).get('median', 0)
    v5_nav = results['V5_vis+sustain'].get('navigability', {}).get('median', 0)
    pv5 = v5_nav > b2_nav
    scored.append(('PV5', f"Vis+sustain > sustain alone on navigability: "
                          f"V5={v5_nav:.3f} vs B2={b2_nav:.3f}",
                   pv5))

    # PV6: EC survival improves most from visibility (transparency tax inverts)
    b1_ec_surv = b1.get('EC_survival', {}).get('median', 0)
    v4_ec_surv = v4.get('EC_survival', {}).get('median', 0)
    b1_cd_surv = b1.get('CD_survival', {}).get('median', 0)
    v4_cd_surv = v4.get('CD_survival', {}).get('median', 0)
    ec_gain = v4_ec_surv - b1_ec_surv
    cd_gain = v4_cd_surv - b1_cd_surv
    pv6 = ec_gain > cd_gain
    scored.append(('PV6', f"EC gains most from visibility (transparency tax inverts): "
                          f"EC gain={ec_gain:+.1%}, CD gain={cd_gain:+.1%}",
                   pv6))

    # PV7: There exists a configuration where navigability > B2 AND cooperation
    #       is within 20% of B2 (the "sweet spot" — agency + dignity + performance)
    b2_ss = b2.get('steady_state_coop', {}).get('median', 0)
    b2_nav_val = b2.get('navigability', {}).get('median', 0)
    sweet_spot = None
    for name in VISIBILITY_CONDITIONS:
        if name.startswith('B'):
            continue
        agg = results[name]
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        nav = agg.get('navigability', {}).get('median', 0)
        if nav > b2_nav_val and ss >= b2_ss * 0.80:
            sweet_spot = name
            break
    pv7 = sweet_spot is not None
    scored.append(('PV7', f"Sweet spot exists (nav > B2 AND coop within 20%): "
                          f"{'found: ' + sweet_spot if sweet_spot else 'not found'}",
                   pv7))

    # Print scorecard
    supported = sum(1 for _, _, s in scored if s)
    total = len(scored)
    print(f"\n  Score: {supported}/{total} predictions supported\n")
    for pid, desc, result in scored:
        status = "SUPPORTED" if result else "NOT SUPPORTED"
        print(f"  {pid}: {status}")
        print(f"      {desc}")
        print()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'visibility':
        run_visibility_test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'both':
        run_phase3()
        run_visibility_test()
    else:
        run_phase3()
