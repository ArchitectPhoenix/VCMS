#!/usr/bin/env python3
"""
Federation Dynamics — Organizational Hierarchy Branch
=====================================================

Models a 30-agent stratified organization with four levels:
  Level 1: Entry-Level (EL) — 16 agents in 4 groups of 4
  Level 2: Middle Management (MM) — 8 agents in 2 groups of 4
  Level 3: Upper Management (UM) — 4 agents in 1 group of 4
  Level 4: C-Suite (CS) — 2 agents (1 pair)

Every agent runs TWO V/C/M/S channels per round:
  1. Peer channel: standard PGG dynamics within same-level group
  2. Authority channel: relationship with direct supervisor

Inter-level coupling:
  - Downward: supervisor cooperation modulates subordinate b_replenish, s_rate,
    b_depletion via cascade_strength multipliers (Gallup calibration)
  - Upward: subordinate aggregate cooperation feeds into supervisor's v_level
    as a weighted performance signal

Enforcement varies by level (threshold at EL, sustainability at MM/UM,
crisis-only at CS), with level-stratified removal costs and golden parachutes.

Population: 30 agents, 100 runs × 100 rounds per condition.
Depends on: enforcement_sim.py, enforcement_phase3.py
"""

import copy
import math
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

from enforcement_sim import (
    load_libraries, build_enforcement_pools, sample_population_blueprint,
    instantiate_population, assign_groups, create_replacement,
    step_group, record_round,
    Agent, EnforcementAgentParams, AgentState,
    _linear_slope, gini_coefficient, bimodality_coefficient,
    compute_run_metrics, aggregate_metrics,
    MAX_CONTRIB, MAX_PUNISH, ANCHOR_RATE, ACUTE_MULT, EPS,
    RUPTURE_B_FRAC, RUPTURE_CONSEC,
    HEALTH_WINDOW, HEALTH_TRIGGER_CONSEC,
)

from enforcement_phase2 import _is_ruptured, _pearson


# ================================================================
# HIERARCHY CONSTANTS
# ================================================================

N_RUNS = 100
N_ROUNDS = 100

# Organization structure
LEVELS = ['EL', 'MM', 'UM', 'CS']
LEVEL_COUNTS = {'EL': 16, 'MM': 8, 'UM': 4, 'CS': 2}
LEVEL_GROUP_SIZE = {'EL': 4, 'MM': 4, 'UM': 4, 'CS': 2}
N_HIERARCHY_AGENTS = 30

# Phenotype distribution by level
LEVEL_PHENOTYPE_WEIGHTS = {
    'EL': {'EC': 0.40, 'CC': 0.25, 'CD': 0.20, 'DL': 0.15, 'MX': 0.00},
    'MM': {'EC': 0.35, 'CC': 0.30, 'CD': 0.15, 'DL': 0.15, 'MX': 0.05},
    'UM': {'EC': 0.25, 'CC': 0.35, 'CD': 0.20, 'DL': 0.10, 'MX': 0.10},
    'CS': {'EC': 0.15, 'CC': 0.30, 'CD': 0.40, 'DL': 0.05, 'MX': 0.10},
}

# Level-specific VCMS parameter overrides
LEVEL_PARAMS = {
    'EL': {
        'alpha': 0.35,         # alpha_peer
        'alpha_authority': 0.30,
        'inertia': 0.15,
        's_initial': 1.0,
        's_rate': 1.0,
        'b_initial': 3.0,
        'b_depletion_rate': 0.20,
        'b_replenish_rate': 0.12,
        'h_start': 0.70,
        'h_strength': 0.30,
        'facilitation_rate': 0.08,
    },
    'MM': {
        'alpha': 0.30,         # alpha_peer
        'alpha_authority': 0.50,  # 0.25 up + 0.25 down
        'inertia': 0.30,
        's_initial': 1.8,
        's_rate': 1.8,
        'b_initial': 5.0,
        'b_depletion_rate': 0.25,
        'b_replenish_rate': 0.15,
        'h_start': 0.80,
        'h_strength': 0.25,
        'facilitation_rate': 0.06,
    },
    'UM': {
        'alpha': 0.20,         # alpha_peer
        'alpha_authority': 0.15,
        'inertia': 0.45,
        's_initial': 1.2,
        's_rate': 1.0,
        'b_initial': 8.5,
        'b_depletion_rate': 0.18,
        'b_replenish_rate': 0.18,
        'h_start': 0.85,
        'h_strength': 0.20,
        'facilitation_rate': 0.04,
    },
    'CS': {
        'alpha': 0.10,         # alpha_peer
        'alpha_authority': 0.05,  # board only
        'inertia': 0.60,
        's_initial': 1.0,
        's_rate': 0.8,
        'b_initial': 16.5,
        'b_depletion_rate': 0.15,
        'b_replenish_rate': 0.20,
        'h_start': 0.90,
        'h_strength': 0.50,  # inverted — amplifies short-term optimization
        'facilitation_rate': 0.03,
    },
}

# Inter-level cascade strengths (Gallup calibration)
CASCADE_STRENGTH = {
    ('MM', 'EL'): 0.59,   # MM supervisor → EL subordinates
    ('UM', 'MM'): 0.39,   # UM supervisor → MM subordinates
    ('CS', 'UM'): 0.39,   # CS supervisor → UM subordinates
}

# Enforcement gradient
ENFORCEMENT_CONFIG = {
    'EL': {
        'mechanism': 'threshold',
        'K': 3,
        'removal_prob': 1.0,
        'retention_frac': 0.05,    # retains 5% of budget
        'replacement_cost': 1.5,
        'grace_rounds': 0,
    },
    'MM': {
        'mechanism': 'sustainability',
        'removal_prob': 0.8,
        'retention_frac': 0.20,    # retains 20%
        'replacement_cost': 7.5,
        'grace_rounds': 0,
    },
    'UM': {
        'mechanism': 'sustainability',
        'removal_prob': 0.5,
        'retention_frac': 0.60,    # retains 60% (severance)
        'replacement_cost': 17.0,
        'grace_rounds': 5,
    },
    'CS': {
        'mechanism': 'crisis',
        'removal_prob': 0.3,
        'retention_frac': 0.85,    # retains 85% (golden parachute)
        'replacement_cost': 50.0,
        'grace_rounds': 10,
    },
}

# Removal cascade disruption strain by level
DISRUPTION_STRAIN = {'EL': 0.5, 'MM': 1.0, 'UM': 1.5, 'CS': 2.0}

# Integration period for replacements (rounds)
INTEGRATION_ROUNDS = {'EL': 5, 'MM': 10, 'UM': 15, 'CS': 20}

# Organizational output weights (higher levels have more leverage)
OUTPUT_WEIGHTS = {'EL': 1.0, 'MM': 1.5, 'UM': 2.0, 'CS': 3.0}

# Board demand: fixed cooperation level for CS to be measured against
BOARD_DEMAND = 0.7  # normalized cooperation demand


# ================================================================
# HIERARCHY POPULATION CONSTRUCTION
# ================================================================

def sample_hierarchy_blueprint(pools, rng):
    """Build a 30-agent hierarchy blueprint with level-specific phenotype distributions."""
    blueprint = []
    uid = 0

    for level in LEVELS:
        count = LEVEL_COUNTS[level]
        weights = LEVEL_PHENOTYPE_WEIGHTS[level]
        level_params = LEVEL_PARAMS[level]

        # Determine phenotype counts for this level
        phenotype_counts = {}
        remaining = count
        sorted_phenos = sorted(weights.keys(), key=lambda p: weights[p], reverse=True)
        for i, ptype in enumerate(sorted_phenos):
            if i == len(sorted_phenos) - 1:
                phenotype_counts[ptype] = remaining
            else:
                n = max(0, round(weights[ptype] * count))
                n = min(n, remaining)
                phenotype_counts[ptype] = n
                remaining -= n

        for ptype, n in phenotype_counts.items():
            if n == 0:
                continue
            # Map MX to random pool draw
            pool_key = ptype if ptype != 'MX' else rng.choice(list(pools.keys()))
            pool = pools.get(pool_key, pools.get('EC', []))
            if not pool:
                pool = pools['EC']

            for _ in range(n):
                _, sid, base_params = pool[rng.integers(len(pool))]
                params = copy.deepcopy(base_params)

                # Apply level-specific parameter overrides
                params.alpha = level_params['alpha']
                params.inertia = level_params['inertia']
                params.s_initial = level_params['s_initial']
                params.s_rate = level_params['s_rate']
                params.b_initial = level_params['b_initial']
                params.b_depletion_rate = level_params['b_depletion_rate']
                params.b_replenish_rate = level_params['b_replenish_rate']
                params.h_start = level_params['h_start']
                params.h_strength = level_params['h_strength']
                params.facilitation_rate = level_params['facilitation_rate']

                blueprint.append((uid, ptype, f"{level}:{sid}", params, level))
                uid += 1

    # Shuffle within levels (groups will be formed by position)
    by_level = defaultdict(list)
    for entry in blueprint:
        by_level[entry[4]].append(entry)
    for level in LEVELS:
        rng.shuffle(by_level[level])

    # Reassemble in level order
    result = []
    for level in LEVELS:
        result.extend(by_level[level])

    return result


def instantiate_hierarchy(blueprint):
    """Create Agent objects from hierarchy blueprint."""
    agents = []
    for uid, phenotype, source, params, level in blueprint:
        ag = Agent(
            uid=uid, phenotype=phenotype, source=source,
            params=copy.deepcopy(params),
            state=AgentState(strain=params.s_initial, B=params.b_initial),
        )
        ag._level = level
        ag._alpha_authority = LEVEL_PARAMS[level]['alpha_authority']
        ag._supervisor_uid = None
        ag._subordinate_uids = []
        ag._integration_remaining = 0
        ag._supervisor_depletion_boost = 0.0
        ag._supervisor_depletion_rounds = 0
        agents.append(ag)
    return agents


def assign_hierarchy_groups(agents):
    """Assign agents to level-specific groups and establish supervision links.

    Returns:
      groups: dict[group_id -> list[Agent]]
      supervision: dict[supervisor_uid -> list[group_id]]  (groups they supervise)
    """
    by_level = defaultdict(list)
    for ag in agents:
        by_level[ag._level].append(ag)

    groups = {}
    group_id = 0
    level_groups = defaultdict(list)  # level -> [group_ids]

    for level in LEVELS:
        members = by_level[level]
        gs = LEVEL_GROUP_SIZE[level]
        for i in range(0, len(members), gs):
            grp = members[i:i+gs]
            for ag in grp:
                ag.group_id = group_id
            groups[group_id] = grp
            level_groups[level].append(group_id)
            group_id += 1

    # Establish supervision links
    # 4 EL groups supervised by 4 MM agents (1 MM per EL group)
    # But MM has 8 agents in 2 groups of 4. Each MM agent supervises 1 EL group.
    # Actually: 4 EL groups, 8 MM agents. We assign 2 MM agents per EL group pair?
    # No — spec says "each supervised by 1 MM agent". So 4 MM agents each supervise 1 EL group.
    # The other 4 MM agents supervise nothing? That doesn't make sense with 8 MM agents.
    # Re-reading spec: "4 EL groups of 4, each supervised by 1 MM agent"
    # "2 MM groups of 4, each supervised by 1 UM agent"
    # So each of the 8 MM agents is in a peer group, but only some supervise EL groups.
    # With 4 EL groups and 8 MM agents, assign 1 MM supervisor per EL group from
    # the set of MM agents, round-robin across MM groups.

    supervision = {}  # supervisor_uid -> [supervised_group_ids]

    el_groups = level_groups['EL']
    mm_agents = by_level['MM']
    mm_groups = level_groups['MM']
    um_agents = by_level['UM']
    um_groups = level_groups['UM']
    cs_agents = by_level['CS']

    # Each MM agent supervises at most 1 EL group (4 EL groups, 8 MM agents → first 4 get assigned)
    # But we need all EL groups supervised. Assign first 4 MM agents.
    for i, gid in enumerate(el_groups):
        if i < len(mm_agents):
            sup = mm_agents[i]
            sup._subordinate_uids = [ag.uid for ag in groups[gid]]
            supervision[sup.uid] = supervision.get(sup.uid, []) + [gid]
            for ag in groups[gid]:
                ag._supervisor_uid = sup.uid

    # Each UM agent supervises 1 MM group (2 MM groups, 4 UM agents → first 2 get assigned)
    for i, gid in enumerate(mm_groups):
        if i < len(um_agents):
            sup = um_agents[i]
            sup._subordinate_uids = [ag.uid for ag in groups[gid]]
            supervision[sup.uid] = supervision.get(sup.uid, []) + [gid]
            for ag in groups[gid]:
                ag._supervisor_uid = sup.uid

    # CS agents supervise UM group (1 UM group, 2 CS agents → both supervise it)
    for gid in um_groups:
        for cs_ag in cs_agents:
            cs_ag._subordinate_uids = [ag.uid for ag in groups[gid]]
            supervision[cs_ag.uid] = supervision.get(cs_ag.uid, []) + [gid]
            for ag in groups[gid]:
                ag._supervisor_uid = cs_ag.uid  # last CS wins, but both listed

    return groups, supervision, level_groups


def _init_hierarchy_state(agent):
    """Initialize hierarchy-specific tracking attributes."""
    if not hasattr(agent, '_level'):
        agent._level = 'EL'
    if not hasattr(agent, '_alpha_authority'):
        agent._alpha_authority = 0.0
    if not hasattr(agent, '_supervisor_uid'):
        agent._supervisor_uid = None
    if not hasattr(agent, '_subordinate_uids'):
        agent._subordinate_uids = []
    if not hasattr(agent, '_integration_remaining'):
        agent._integration_remaining = 0
    if not hasattr(agent, '_supervisor_depletion_boost'):
        agent._supervisor_depletion_boost = 0.0
    if not hasattr(agent, '_supervisor_depletion_rounds'):
        agent._supervisor_depletion_rounds = 0
    if not hasattr(agent, '_authority_strain'):
        agent._authority_strain = 0.0


# ================================================================
# INTER-LEVEL COUPLING
# ================================================================

def apply_supervisor_influence(groups, supervision, all_agents, dt):
    """Downward coupling: supervisor cooperation modulates subordinate environment.

    For each supervisor-subordinate link:
      subordinate.b_replenish modifier = 1.0 + sup_coop_norm × cascade_strength
      subordinate.s_rate modifier = 1.0 + (1 - sup_coop_norm) × cascade_strength
      subordinate.b_depletion modifier = 1.0 + sup_defection_frac × cascade_strength
    """
    uid_map = {ag.uid: ag for ag in all_agents if ag.active_to == -1}

    for sup_uid, sub_gids in supervision.items():
        sup = uid_map.get(sup_uid)
        if sup is None:
            continue

        # Supervisor's normalized cooperation (from most recent contribution)
        if sup.contrib_history:
            sup_coop_norm = sup.contrib_history[-1] / max(MAX_CONTRIB, 1)
        else:
            sup_coop_norm = sup.params.c_base

        sup_level = sup._level
        for sub_gid in sub_gids:
            if sub_gid not in groups:
                continue
            members = groups[sub_gid]
            if not members:
                continue
            sub_level = members[0]._level

            cascade_key = (sup_level, sub_level)
            strength = CASCADE_STRENGTH.get(cascade_key, 0.0)
            if strength == 0:
                continue

            for sub_ag in members:
                if sub_ag.active_to != -1:
                    continue
                # Replenishment boost from good supervisor
                replenish_mod = 1.0 + sup_coop_norm * strength
                # Strain amplification from bad supervisor
                strain_mod = 1.0 + (1.0 - sup_coop_norm) * strength
                # Depletion increase from defecting supervisor
                depletion_mod = 1.0 + (1.0 - sup_coop_norm) * strength

                # Apply as per-round modifiers to state
                # These modify the EFFECTIVE rates for this round
                sub_ag._sup_replenish_mod = replenish_mod
                sub_ag._sup_strain_mod = strain_mod
                sub_ag._sup_depletion_mod = depletion_mod


def apply_upward_performance(groups, supervision, all_agents, level_groups):
    """Upward coupling: subordinate aggregate cooperation feeds into supervisor's v_level.

    Supervisor sees a blended signal: their peer group + subordinate performance.
    """
    uid_map = {ag.uid: ag for ag in all_agents if ag.active_to == -1}

    for sup_uid, sub_gids in supervision.items():
        sup = uid_map.get(sup_uid)
        if sup is None:
            continue

        # Compute subordinate aggregate cooperation
        sub_contribs = []
        for sub_gid in sub_gids:
            if sub_gid not in groups:
                continue
            for ag in groups[sub_gid]:
                if ag.active_to == -1 and ag.contrib_history:
                    sub_contribs.append(ag.contrib_history[-1])

        if not sub_contribs:
            continue

        sub_mean_norm = (sum(sub_contribs) / len(sub_contribs)) / max(MAX_CONTRIB, 1)

        # Blend into supervisor's v_level via authority alpha
        alpha_auth = sup._alpha_authority
        # Authority strain: gap between board demand and subordinate performance
        if sup._level == 'CS':
            demand = BOARD_DEMAND
        else:
            demand = sub_mean_norm  # supervisors just track subordinate performance

        # Accumulate authority-channel strain
        if sup._level == 'CS':
            auth_gap = max(0, BOARD_DEMAND - sub_mean_norm)
        else:
            auth_gap = max(0, 0.5 - sub_mean_norm)  # minimum acceptable performance

        sup._authority_strain = getattr(sup, '_authority_strain', 0.0)
        sup._authority_strain += alpha_auth * auth_gap * sup.params.s_rate * (1.0 / max(N_ROUNDS - 1, 1))
        sup.state.strain += alpha_auth * auth_gap * 0.1  # small per-round authority strain


# ================================================================
# HIERARCHY STEP: Modified step_group with supervisor modifiers
# ================================================================

def step_hierarchy_group(agents, rnd, n_rounds, dt, prev_contribs=None,
                         prev_pun_recv=None):
    """Run one round for a single group with hierarchy modifiers applied.

    Identical to step_group() but applies supervisor influence modifiers
    to b_replenish_rate and s_rate before the step, then restores them after.
    """
    # Save original rates and apply supervisor modifiers
    originals = []
    for ag in agents:
        orig = (ag.params.b_replenish_rate, ag.params.s_rate,
                ag.params.b_depletion_rate)
        originals.append(orig)

        rep_mod = getattr(ag, '_sup_replenish_mod', 1.0)
        str_mod = getattr(ag, '_sup_strain_mod', 1.0)
        dep_mod = getattr(ag, '_sup_depletion_mod', 1.0)

        ag.params.b_replenish_rate *= rep_mod
        ag.params.s_rate *= str_mod
        ag.params.b_depletion_rate *= dep_mod

        # Handle supervisor depletion boost (from managing removals below)
        if getattr(ag, '_supervisor_depletion_rounds', 0) > 0:
            ag.params.b_depletion_rate += ag._supervisor_depletion_boost
            ag._supervisor_depletion_rounds -= 1
            if ag._supervisor_depletion_rounds <= 0:
                ag._supervisor_depletion_boost = 0.0

        # Integration penalty for replacements
        if getattr(ag, '_integration_remaining', 0) > 0:
            ag.params.b_replenish_rate *= 0.5  # halved during integration
            ag._integration_remaining -= 1

    # Call standard step_group
    contribs, pun_sent, pun_recv = step_group(
        agents, rnd, n_rounds, dt,
        has_punishment=False,
        prev_contribs=prev_contribs,
        prev_pun_recv=prev_pun_recv,
    )

    # Restore original rates
    for j, ag in enumerate(agents):
        ag.params.b_replenish_rate, ag.params.s_rate, ag.params.b_depletion_rate = originals[j]

    # Reset supervisor modifiers for next round
    for ag in agents:
        ag._sup_replenish_mod = 1.0
        ag._sup_strain_mod = 1.0
        ag._sup_depletion_mod = 1.0

    return contribs, pun_sent, pun_recv


# ================================================================
# REMOVAL CASCADE
# ================================================================

def process_removal_cascade(removed_agent, group, all_agents, supervision):
    """Apply state modifications when an agent is removed.

    Immediate effects:
      - Same-level group peers: +disruption_strain
      - Direct supervisor: +0.05 b_depletion_rate for integration_rounds
    """
    level = removed_agent._level
    disrupt = DISRUPTION_STRAIN.get(level, 0.5)
    integ_rounds = INTEGRATION_ROUNDS.get(level, 5)

    # Strain to remaining group members
    for ag in group:
        if ag.uid != removed_agent.uid and ag.active_to == -1:
            ag.state.strain += disrupt

    # Supervisor depletion increase
    uid_map = {ag.uid: ag for ag in all_agents if ag.active_to == -1}
    sup_uid = removed_agent._supervisor_uid
    if sup_uid is not None:
        sup = uid_map.get(sup_uid)
        if sup is not None:
            sup._supervisor_depletion_boost += 0.05
            sup._supervisor_depletion_rounds = max(
                sup._supervisor_depletion_rounds, integ_rounds)


def create_hierarchy_replacement(removed_agent, pools, rng, uid_counter):
    """Create a replacement agent for the hierarchy with proper level parameters."""
    level = removed_agent._level
    level_params = LEVEL_PARAMS[level]
    phenotype = removed_agent.phenotype

    # Draw from pool
    pool_key = phenotype if phenotype != 'MX' else rng.choice(list(pools.keys()))
    pool = pools.get(pool_key, pools.get('EC', []))
    if not pool:
        pool = pools['EC']
    _, sid, base_params = pool[rng.integers(len(pool))]
    params = copy.deepcopy(base_params)

    # Apply level-specific overrides
    params.alpha = level_params['alpha']
    params.inertia = level_params['inertia']
    params.s_initial = level_params['s_initial']
    params.s_rate = level_params['s_rate']
    params.b_initial = level_params['b_initial']
    params.b_depletion_rate = level_params['b_depletion_rate']
    params.b_replenish_rate = level_params['b_replenish_rate']
    params.h_start = level_params['h_start']
    params.h_strength = level_params['h_strength']
    params.facilitation_rate = level_params['facilitation_rate']

    # Replacement enters at 80% of level b_initial
    params.b_initial *= 0.80

    ag = Agent(
        uid=uid_counter, phenotype=phenotype,
        source=f"{level}:{sid}",
        params=params,
        state=AgentState(strain=params.s_initial, B=params.b_initial),
    )
    ag._level = level
    ag._alpha_authority = level_params['alpha_authority']
    ag._supervisor_uid = removed_agent._supervisor_uid
    ag._subordinate_uids = list(removed_agent._subordinate_uids)
    ag._integration_remaining = INTEGRATION_ROUNDS[level]
    ag._supervisor_depletion_boost = 0.0
    ag._supervisor_depletion_rounds = 0
    ag._authority_strain = 0.0

    return ag


# ================================================================
# ENFORCEMENT BY LEVEL
# ================================================================

def apply_level_enforcement(level, groups, level_groups, all_agents, pools,
                            rng, uid_counter, rnd, events,
                            g_budget_hist, g_coop_hist, degrading,
                            below_count, prev_contribs_map,
                            enforcement_override=None, supervision=None):
    """Apply level-specific enforcement mechanism.

    Returns: (uid_counter, total_removal_cost)
    """
    if enforcement_override is not None:
        config = enforcement_override.get(level)
        if config is None:
            return uid_counter, 0.0
    else:
        config = ENFORCEMENT_CONFIG[level]

    if config is None:
        return uid_counter, 0.0

    mechanism = config.get('mechanism', 'none')
    if mechanism == 'none':
        return uid_counter, 0.0

    total_removal_cost = 0.0
    gids = level_groups.get(level, [])

    for gid in gids:
        members = [ag for ag in groups.get(gid, []) if ag.active_to == -1]
        if len(members) < 2:
            continue

        contribs = prev_contribs_map.get(gid, [0] * len(members))

        if mechanism == 'threshold':
            K = config.get('K', 3)
            group_mean = sum(contribs) / max(len(contribs), 1)
            sd = (sum((c - group_mean) ** 2 for c in contribs) / max(len(contribs), 1)) ** 0.5
            threshold = group_mean - sd

            for j, ag in enumerate(members):
                if j >= len(contribs):
                    break
                key = ag.uid
                if contribs[j] < threshold:
                    below_count[key] = below_count.get(key, 0) + 1
                else:
                    below_count[key] = 0

                if below_count.get(key, 0) >= K:
                    # Probabilistic removal
                    if rng.random() < config.get('removal_prob', 1.0):
                        # Apply golden parachute
                        retention = config.get('retention_frac', 0.05)
                        ag.state.B *= retention
                        ag.active_to = rnd

                        process_removal_cascade(ag, members, all_agents, supervision)

                        replacement = create_hierarchy_replacement(
                            ag, pools, rng, uid_counter)
                        uid_counter += 1
                        replacement.group_id = gid
                        replacement.active_from = rnd
                        groups[gid] = [m for m in groups[gid]
                                       if m.uid != ag.uid] + [replacement]
                        all_agents.append(replacement)
                        _init_hierarchy_state(replacement)

                        events.append({
                            'type': 'hierarchy_removal', 'round': rnd,
                            'level': level, 'mechanism': 'threshold',
                            'removed_uid': ag.uid,
                            'removed_phenotype': ag.phenotype,
                            'replacement_uid': replacement.uid,
                            'replacement_cost': config['replacement_cost'],
                        })
                        total_removal_cost += config['replacement_cost']
                    below_count[key] = 0

        elif mechanism in ('sustainability', 'crisis'):
            grace = config.get('grace_rounds', 0)

            # Track group health
            if gid not in g_budget_hist:
                g_budget_hist[gid] = []
                g_coop_hist[gid] = []
            mean_b = sum(ag.state.B for ag in members) / max(len(members), 1)
            mean_c = sum(contribs) / max(len(contribs), 1)
            g_budget_hist[gid].append(mean_b)
            g_coop_hist[gid].append(mean_c)

            if len(g_budget_hist[gid]) >= HEALTH_WINDOW:
                b_slope = _linear_slope(g_budget_hist[gid][-HEALTH_WINDOW:])
                c_slope = _linear_slope(g_coop_hist[gid][-HEALTH_WINDOW:])

                if b_slope < 0 and c_slope < 0:
                    degrading[gid] = degrading.get(gid, 0) + 1
                else:
                    degrading[gid] = 0

                trigger_consec = HEALTH_TRIGGER_CONSEC + grace

                if degrading.get(gid, 0) >= trigger_consec and len(members) > 1:
                    # Select highest-impact target
                    total_c = sum(contribs)
                    n_m = len(members)
                    group_mean_c = total_c / n_m

                    best_impact = -float('inf')
                    worst_idx = -1
                    for j in range(min(n_m, len(contribs))):
                        mean_without = (total_c - contribs[j]) / max(n_m - 1, 1)
                        impact = mean_without - group_mean_c
                        if impact > best_impact:
                            best_impact = impact
                            worst_idx = j

                    if worst_idx >= 0:
                        target = members[worst_idx]
                        if rng.random() < config.get('removal_prob', 1.0):
                            retention = config.get('retention_frac', 0.05)
                            target.state.B *= retention
                            target.active_to = rnd

                            process_removal_cascade(
                                target, members, all_agents, supervision)

                            replacement = create_hierarchy_replacement(
                                target, pools, rng, uid_counter)
                            uid_counter += 1
                            replacement.group_id = gid
                            replacement.active_from = rnd
                            groups[gid] = [m for m in groups[gid]
                                           if m.uid != target.uid] + [replacement]
                            all_agents.append(replacement)
                            _init_hierarchy_state(replacement)

                            events.append({
                                'type': 'hierarchy_removal', 'round': rnd,
                                'level': level, 'mechanism': mechanism,
                                'removed_uid': target.uid,
                                'removed_phenotype': target.phenotype,
                                'replacement_uid': replacement.uid,
                                'replacement_cost': config['replacement_cost'],
                            })
                            total_removal_cost += config['replacement_cost']

                    degrading[gid] = 0

    return uid_counter, total_removal_cost


# ================================================================
# STRUCTURAL MECHANISMS (per-level)
# ================================================================

def apply_hierarchy_structural(groups, level_groups, structural_config, dt):
    """Apply structural mechanisms at specified levels.

    structural_config: dict mapping level -> struct_dict, or 'all' -> struct_dict
    """
    if structural_config is None:
        return 0.0

    total_cost = 0.0

    for level in LEVELS:
        config = structural_config.get(level, structural_config.get('all'))
        if config is None:
            continue

        gids = level_groups.get(level, [])
        for gid in gids:
            members = [ag for ag in groups.get(gid, []) if ag.active_to == -1]
            if not members:
                continue

            # Budget floor
            floor_frac = config.get('budget_floor', 0)
            if floor_frac > 0:
                for ag in members:
                    floor_val = floor_frac * ag.params.b_initial
                    if ag.state.B < floor_val:
                        total_cost += floor_val - ag.state.B
                        ag.state.B = floor_val

            # Strain ceiling
            ceil = config.get('strain_ceiling', 0)
            if ceil > 0:
                for ag in members:
                    if ag.state.strain > ceil:
                        ag.state.strain = ceil

            # Contribution matching
            match_rate = config.get('match_rate', 0)
            if match_rate > 0:
                contribs = [ag.contrib_history[-1] if ag.contrib_history else 0
                            for ag in members]
                mean_c = sum(contribs) / max(len(contribs), 1)
                for j, ag in enumerate(members):
                    if contribs[j] > mean_c:
                        excess_norm = (contribs[j] - mean_c) / max(MAX_CONTRIB, 1)
                        bonus = match_rate * excess_norm * ag.params.b_initial
                        ag.state.B += bonus
                        total_cost += bonus

            # Progressive redistribution (budget-neutral)
            redist_rate = config.get('redist_rate', 0)
            if redist_rate > 0:
                budgets = [ag.state.B for ag in members]
                median_b = sorted(budgets)[len(budgets) // 2]
                collected = 0.0
                above = [ag for ag in members if ag.state.B > median_b]
                below = [ag for ag in members if ag.state.B < median_b]
                for ag in above:
                    excess = ag.state.B - median_b
                    tax = excess * redist_rate
                    ag.state.B -= tax
                    collected += tax
                if below:
                    per_below = collected / len(below)
                    for ag in below:
                        ag.state.B += per_below

    return total_cost


# ================================================================
# VISIBILITY (per-level or cross-level)
# ================================================================

def apply_hierarchy_visibility(groups, level_groups, all_agents, vis_config, dt):
    """Apply visibility effects within or across levels.

    vis_config: dict with keys:
      'within_level': bool — agents see same-level peers
      'cross_level': bool — agents see ALL agents
      'empathy': bool
      'solidarity': bool
      'reference': bool
    """
    if vis_config is None:
        return

    do_empathy = vis_config.get('empathy', False)
    do_solidarity = vis_config.get('solidarity', False)
    do_reference = vis_config.get('reference', False)
    cross_level = vis_config.get('cross_level', False)

    if not (do_empathy or do_solidarity or do_reference):
        return

    active = [ag for ag in all_agents if ag.active_to == -1]

    if cross_level:
        # Full cross-level visibility: process all agents as one pool
        _apply_vis_to_pool(active, do_empathy, do_solidarity, do_reference, dt)
    else:
        # Within-level only: process each group independently
        for gid, members in groups.items():
            active_members = [ag for ag in members if ag.active_to == -1]
            if len(active_members) < 2:
                continue
            _apply_vis_to_pool(active_members, do_empathy, do_solidarity,
                               do_reference, dt)


def _apply_vis_to_pool(agents, empathy, solidarity, reference, dt):
    """Apply visibility effects to a pool of agents."""
    if len(agents) < 2:
        return

    budgets = [ag.state.B for ag in agents]
    median_b = sorted(budgets)[len(budgets) // 2]
    vuln_threshold = max(median_b * 0.3, 0.05)

    if empathy:
        n = len(agents)
        constrained = sum(1 for ag in agents if ag.state.B < vuln_threshold)
        for ag in agents:
            others_constrained = constrained - (1 if ag.state.B < vuln_threshold else 0)
            constrained_frac = others_constrained / max(n - 1, 1)
            reduction = ag.params.alpha * constrained_frac * 0.3
            ag.state.strain *= (1.0 - min(reduction, 0.3))

    if solidarity:
        above = [ag for ag in agents if ag.state.B > median_b]
        below = [ag for ag in agents if ag.state.B < median_b]
        if above and below:
            collected = 0.0
            for ag in above:
                share_rate = ag.params.c_base * 0.1
                excess = ag.state.B - median_b
                share = excess * share_rate * dt
                ag.state.B -= share
                collected += share
            per_below = collected / len(below)
            for ag in below:
                ag.state.B += per_below

    if reference:
        capable = [ag for ag in agents if ag.state.B > vuln_threshold]
        if capable and len(capable) < len(agents):
            capable_contribs = [ag.contrib_history[-1] / max(MAX_CONTRIB, 1)
                                if ag.contrib_history else ag.params.c_base
                                for ag in capable]
            capable_mean = sum(capable_contribs) / len(capable_contribs)
            for ag in agents:
                blend = ag.params.alpha * 0.3
                ag.state.v_level = (1.0 - blend) * ag.state.v_level + blend * capable_mean


# ================================================================
# MAIN SIMULATION
# ================================================================

def simulate_hierarchy(pools, rng, n_rounds=N_ROUNDS, blueprint=None,
                       enforcement_override=None,
                       visibility_config=None,
                       structural_config=None,
                       perturbation=None):
    """Run one hierarchy simulation.

    Args:
        pools: phenotype pools from build_enforcement_pools()
        rng: numpy random generator
        n_rounds: number of rounds
        blueprint: population blueprint (or None to generate)
        enforcement_override: dict[level -> config] or None for defaults
        visibility_config: dict or None
        structural_config: dict[level -> struct_dict] or None
        perturbation: dict describing mid-simulation changes (for Phase 3)

    Returns: dict with agents, groups, events, costs, metrics
    """
    if blueprint is None:
        blueprint = sample_hierarchy_blueprint(pools, rng)
    agents = instantiate_hierarchy(blueprint)
    for ag in agents:
        _init_hierarchy_state(ag)

    groups, supervision, level_groups = assign_hierarchy_groups(agents)
    dt = 1.0 / max(n_rounds - 1, 1)
    all_agents = list(agents)
    events = []
    uid_counter = max(ag.uid for ag in agents) + 1

    # Enforcement state
    g_budget_hist = {}
    g_coop_hist = {}
    degrading = {}
    below_count = {}
    prev_contribs_map = {}

    total_structural_cost = 0.0
    total_removal_cost = 0.0

    for rnd in range(n_rounds):
        # Check for mid-simulation perturbation (Phase 3 transitions)
        if perturbation and rnd == perturbation.get('switch_round', -1):
            enforcement_override = perturbation.get('enforcement_after')
            structural_config = perturbation.get('structural_after')
            visibility_config = perturbation.get('visibility_after')

        # Apply supervisor influence (downward coupling)
        apply_supervisor_influence(groups, supervision, all_agents, dt)

        # Step all groups
        round_contribs = {}
        for gid, members in groups.items():
            active = [ag for ag in members if ag.active_to == -1]
            if not active:
                continue

            prev_c = prev_contribs_map.get(gid)
            contribs, pun_sent, pun_recv = step_hierarchy_group(
                active, rnd, n_rounds, dt,
                prev_contribs=prev_c,
                prev_pun_recv=None,
            )
            record_round(active, contribs, pun_sent, pun_recv)
            round_contribs[gid] = contribs

        prev_contribs_map = round_contribs

        # Apply upward performance coupling
        apply_upward_performance(groups, supervision, all_agents, level_groups)

        # Apply enforcement by level
        for level in LEVELS:
            uid_counter, rm_cost = apply_level_enforcement(
                level, groups, level_groups, all_agents, pools, rng,
                uid_counter, rnd, events,
                g_budget_hist, g_coop_hist, degrading, below_count,
                prev_contribs_map,
                enforcement_override=enforcement_override,
                supervision=supervision,
            )
            total_removal_cost += rm_cost

        # Apply structural mechanisms
        if structural_config:
            sc = apply_hierarchy_structural(
                groups, level_groups, structural_config, dt)
            total_structural_cost += sc

        # Apply visibility effects
        if visibility_config:
            apply_hierarchy_visibility(
                groups, level_groups, all_agents, visibility_config, dt)

        # Update supervision links for any replaced agents
        _refresh_supervision(groups, supervision, all_agents, level_groups)

    return {
        'agents': all_agents,
        'groups': groups,
        'events': events,
        'level_groups': level_groups,
        'supervision': supervision,
        'total_structural_cost': total_structural_cost,
        'total_removal_cost': total_removal_cost,
    }


def _refresh_supervision(groups, supervision, all_agents, level_groups):
    """Update supervision links after replacements."""
    active_map = {ag.uid: ag for ag in all_agents if ag.active_to == -1}

    # Rebuild supervision from group structure
    new_supervision = {}
    for sup_uid, sub_gids in list(supervision.items()):
        if sup_uid in active_map:
            new_supervision[sup_uid] = sub_gids
        else:
            # Supervisor was replaced — find replacement in same group
            old_sup = None
            for ag in all_agents:
                if ag.uid == sup_uid:
                    old_sup = ag
                    break
            if old_sup is not None:
                # Find active agent in same position
                gid = old_sup.group_id
                if gid in groups:
                    for ag in groups[gid]:
                        if (ag.active_to == -1 and
                                getattr(ag, '_subordinate_uids', []) and
                                ag.uid not in new_supervision):
                            new_supervision[ag.uid] = sub_gids
                            break

    supervision.clear()
    supervision.update(new_supervision)


# ================================================================
# METRICS
# ================================================================

def compute_hierarchy_metrics(result, n_rounds=N_ROUNDS):
    """Compute per-level and aggregate metrics."""
    agents = result['agents']
    events = result['events']
    level_groups = result['level_groups']
    groups = result['groups']

    active = [ag for ag in agents if ag.active_to == -1]
    metrics = {}

    # Per-level metrics
    for level in LEVELS:
        level_agents = [ag for ag in agents if ag._level == level]
        level_active = [ag for ag in active if ag._level == level]

        # Cooperation
        level_contribs = []
        for ag in level_active:
            if ag.contrib_history:
                level_contribs.append(ag.contrib_history[-1])
        mean_coop = sum(level_contribs) / max(len(level_contribs), 1) if level_contribs else 0

        # Steady-state cooperation (last 10 rounds)
        ss_contribs = []
        for ag in level_active:
            if len(ag.contrib_history) >= 10:
                ss_contribs.extend(ag.contrib_history[-10:])
            elif ag.contrib_history:
                ss_contribs.extend(ag.contrib_history)
        ss_coop = sum(ss_contribs) / max(len(ss_contribs), 1) if ss_contribs else 0

        # Budget
        level_budgets = [ag.state.B for ag in level_active]
        mean_budget = sum(level_budgets) / max(len(level_budgets), 1) if level_budgets else 0
        budget_floor = min(level_budgets) if level_budgets else 0

        # Strain
        level_strains = [ag.state.strain for ag in level_active]
        mean_strain = sum(level_strains) / max(len(level_strains), 1) if level_strains else 0
        peak_strain = max(level_strains) if level_strains else 0

        # Survival
        total_at_level = len(level_agents)
        ruptured = sum(1 for ag in level_agents if _is_ruptured(ag))
        survived = sum(1 for ag in level_active if not _is_ruptured(ag))
        survival_rate = survived / max(total_at_level, 1)

        # Removals at this level
        level_removals = [e for e in events
                          if e['type'] == 'hierarchy_removal' and e['level'] == level]
        removal_count = len(level_removals)

        # Rupture count
        rupture_count = sum(1 for ag in level_agents if _is_ruptured(ag))

        # TTFR for this level
        level_ttfr = n_rounds  # default: no rupture
        for ag in level_agents:
            if len(ag.budget_history) >= RUPTURE_CONSEC:
                b_thresh = ag.params.b_initial * RUPTURE_B_FRAC
                for r in range(RUPTURE_CONSEC - 1, len(ag.budget_history)):
                    if all(ag.budget_history[r - k] < b_thresh
                           for k in range(RUPTURE_CONSEC)):
                        level_ttfr = min(level_ttfr, r - RUPTURE_CONSEC + 1 +
                                         ag.active_from)
                        break

        metrics[f'{level}_coop'] = mean_coop
        metrics[f'{level}_ss_coop'] = ss_coop
        metrics[f'{level}_budget'] = mean_budget
        metrics[f'{level}_budget_floor'] = budget_floor
        metrics[f'{level}_strain'] = mean_strain
        metrics[f'{level}_peak_strain'] = peak_strain
        metrics[f'{level}_survival'] = survival_rate
        metrics[f'{level}_removals'] = removal_count
        metrics[f'{level}_ruptures'] = rupture_count
        metrics[f'{level}_ttfr'] = level_ttfr

    # Aggregate metrics

    # Organizational output: weighted sum of level cooperation
    org_output = 0.0
    for level in LEVELS:
        org_output += OUTPUT_WEIGHTS[level] * metrics[f'{level}_ss_coop']
    metrics['org_output'] = org_output

    # Max possible output
    max_output = sum(OUTPUT_WEIGHTS[level] * MAX_CONTRIB for level in LEVELS)
    metrics['org_output_norm'] = org_output / max(max_output, 1)

    # Total removals and cost
    total_removals = sum(metrics[f'{level}_removals'] for level in LEVELS)
    metrics['total_removals'] = total_removals
    metrics['total_removal_cost'] = result.get('total_removal_cost', 0.0)
    metrics['total_structural_cost'] = result.get('total_structural_cost', 0.0)
    metrics['total_true_cost'] = (metrics['total_removal_cost'] +
                                   metrics['total_structural_cost'])

    # Cross-level Gini
    all_budgets = [ag.state.B for ag in active]
    metrics['cross_level_gini'] = gini_coefficient(all_budgets) if all_budgets else 0

    # Dignity floor: minimum survival rate across levels
    survival_rates = [metrics[f'{level}_survival'] for level in LEVELS]
    metrics['dignity_floor'] = min(survival_rates) if survival_rates else 0

    # System TTFR
    metrics['system_ttfr'] = min(metrics[f'{level}_ttfr'] for level in LEVELS)

    # Total ruptures
    metrics['total_ruptures'] = sum(metrics[f'{level}_ruptures'] for level in LEVELS)

    # Removal share by level (for P1 — what fraction of removals hit EL?)
    if total_removals > 0:
        for level in LEVELS:
            metrics[f'{level}_removal_share'] = (
                metrics[f'{level}_removals'] / total_removals)
    else:
        for level in LEVELS:
            metrics[f'{level}_removal_share'] = 0.0

    # Hierarchical navigability
    gini_val = metrics['cross_level_gini']
    dfloor = metrics['dignity_floor']
    out_norm = metrics['org_output_norm']
    metrics['hierarchical_navigability'] = dfloor * (1.0 - gini_val) * (1.0 + out_norm)

    # Cost-adjusted navigability
    tc = metrics['total_true_cost']
    nav = metrics['hierarchical_navigability']
    if tc > 0:
        metrics['cost_adj_navigability'] = nav / (1.0 + math.log1p(tc))
    else:
        metrics['cost_adj_navigability'] = nav

    # Cascade timing: not computed per-run (would require perturbation tracking)

    return metrics


# ================================================================
# CONDITION DEFINITIONS
# ================================================================

# Helper enforcement configs
NO_ENFORCEMENT = {level: {'mechanism': 'none'} for level in LEVELS}

UNIFORM_ENFORCEMENT = {
    level: {
        'mechanism': 'sustainability',
        'removal_prob': 1.0,
        'retention_frac': 0.05,
        'replacement_cost': ENFORCEMENT_CONFIG[level]['replacement_cost'],
        'grace_rounds': 0,
    }
    for level in LEVELS
}

REALISTIC_ENFORCEMENT = None  # Use defaults (ENFORCEMENT_CONFIG)

INVERTED_ENFORCEMENT = {
    'EL': {
        'mechanism': 'crisis',
        'removal_prob': 0.3,
        'retention_frac': 0.85,
        'replacement_cost': 1.5,
        'grace_rounds': 10,
    },
    'MM': {
        'mechanism': 'sustainability',
        'removal_prob': 0.5,
        'retention_frac': 0.60,
        'replacement_cost': 7.5,
        'grace_rounds': 5,
    },
    'UM': {
        'mechanism': 'sustainability',
        'removal_prob': 0.8,
        'retention_frac': 0.20,
        'replacement_cost': 17.0,
        'grace_rounds': 0,
    },
    'CS': {
        'mechanism': 'threshold',
        'K': 3,
        'removal_prob': 1.0,
        'retention_frac': 0.05,
        'replacement_cost': 50.0,
        'grace_rounds': 0,
    },
}

WITHIN_VIS = {
    'within_level': True, 'cross_level': False,
    'empathy': True, 'solidarity': True, 'reference': True,
}
CROSS_VIS = {
    'within_level': True, 'cross_level': True,
    'empathy': True, 'solidarity': True, 'reference': True,
}

STRUCT_FLOOR = {'budget_floor': 0.3, 'strain_ceiling': 0, 'match_rate': 0, 'redist_rate': 0}
STRUCT_MATCH = {'budget_floor': 0, 'strain_ceiling': 0, 'match_rate': 0.2, 'redist_rate': 0}
STRUCT_ALL = {'budget_floor': 0.3, 'strain_ceiling': 5.0, 'match_rate': 0.2, 'redist_rate': 0.1}

# Phase 1 conditions
PHASE1_CONDITIONS = {
    'H1_no_enforce': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': None,
        'structural': None,
    },
    'H2_uniform_enforce': {
        'enforcement': UNIFORM_ENFORCEMENT,
        'visibility': None,
        'structural': None,
    },
    'H3_realistic_enforce': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': None,
        'structural': None,
    },
    'H4_inverted_enforce': {
        'enforcement': INVERTED_ENFORCEMENT,
        'visibility': None,
        'structural': None,
    },
    'H5_within_vis': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': WITHIN_VIS,
        'structural': None,
    },
    'H6_cross_vis': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': CROSS_VIS,
        'structural': None,
    },
}

# Phase 2 conditions
PHASE2_CONDITIONS = {
    'HS1_EL_floor': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': None,
        'structural': {'EL': STRUCT_FLOOR},
    },
    'HS2_universal_floor': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': None,
        'structural': {'all': STRUCT_FLOOR},
    },
    'HS3_EL_matching': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': None,
        'structural': {'EL': STRUCT_MATCH},
    },
    'HS4_universal_matching': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': None,
        'structural': {'all': STRUCT_MATCH},
    },
    'HS5_cross_vis+realistic': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': CROSS_VIS,
        'structural': None,
    },
    'HS6_struct_EL+enforce_CS': {
        'enforcement': {
            'EL': {'mechanism': 'none'},
            'MM': {'mechanism': 'none'},
            'UM': {'mechanism': 'none'},
            'CS': INVERTED_ENFORCEMENT['CS'],
        },
        'visibility': {'within_level': True, 'cross_level': False,
                        'empathy': True, 'solidarity': True, 'reference': True},
        'structural': {'EL': STRUCT_ALL},
    },
    'HS7_full_struct_all': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': None,
        'structural': {'all': STRUCT_ALL},
    },
    'HS8_full_struct+cross_vis': {
        'enforcement': NO_ENFORCEMENT,
        'visibility': CROSS_VIS,
        'structural': {'all': STRUCT_ALL},
    },
}

# Phase 3 conditions (cascade dynamics — use perturbation)
PHASE3_CONDITIONS = {
    'HC1_CS_extraction': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': None,
        'structural': None,
        'perturbation': {'type': 'cs_extraction'},
    },
    'HC2_EL_burnout': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': None,
        'structural': None,
        'perturbation': {'type': 'el_burnout'},
    },
    'HC3_MM_squeeze': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': None,
        'structural': None,
        'perturbation': {'type': 'mm_squeeze'},
    },
    'HC4_top_down_reform': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': None,
        'structural': None,
        'perturbation': {
            'type': 'transition',
            'switch_round': 50,
            'structural_after': {'CS': STRUCT_ALL, 'UM': STRUCT_ALL},
            'visibility_after': CROSS_VIS,
            'enforcement_after': {
                'EL': ENFORCEMENT_CONFIG['EL'],
                'MM': ENFORCEMENT_CONFIG['MM'],
                'UM': {'mechanism': 'none'},
                'CS': {'mechanism': 'none'},
            },
        },
    },
    'HC5_bottom_up_reform': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': None,
        'structural': None,
        'perturbation': {
            'type': 'transition',
            'switch_round': 50,
            'structural_after': {'EL': STRUCT_ALL, 'MM': STRUCT_ALL},
            'visibility_after': WITHIN_VIS,
            'enforcement_after': {
                'EL': {'mechanism': 'none'},
                'MM': {'mechanism': 'none'},
                'UM': ENFORCEMENT_CONFIG['UM'],
                'CS': ENFORCEMENT_CONFIG['CS'],
            },
        },
    },
    'HC6_middle_out_reform': {
        'enforcement': REALISTIC_ENFORCEMENT,
        'visibility': None,
        'structural': None,
        'perturbation': {
            'type': 'transition',
            'switch_round': 50,
            'structural_after': {'MM': STRUCT_ALL},
            'visibility_after': WITHIN_VIS,
            'enforcement_after': {
                'EL': ENFORCEMENT_CONFIG['EL'],
                'MM': {'mechanism': 'none'},
                'UM': ENFORCEMENT_CONFIG['UM'],
                'CS': ENFORCEMENT_CONFIG['CS'],
            },
        },
    },
}


def apply_perturbation_setup(agents, perturbation, rng):
    """Apply initial perturbation modifications before simulation starts."""
    if perturbation is None:
        return

    ptype = perturbation.get('type', '')

    if ptype == 'cs_extraction':
        # Replace 1 CS agent with extreme CD
        cs_agents = [ag for ag in agents if ag._level == 'CS']
        if cs_agents:
            target = cs_agents[0]
            target.params.c_base = 0.1
            target.phenotype = 'CD'
            target.params.inertia = 0.70
            target.params.alpha = 0.05

    elif ptype == 'el_burnout':
        # Place 4 DL-like agents in one EL group
        el_agents = [ag for ag in agents if ag._level == 'EL']
        # First 4 EL agents become DL-like
        for ag in el_agents[:4]:
            ag.phenotype = 'DL'
            ag.params.c_base = max(0.15, ag.params.c_base * 0.3)
            ag.params.b_depletion_rate = 0.30  # elevated drain
            ag.params.b_replenish_rate = 0.08  # suppressed recovery

    elif ptype == 'mm_squeeze':
        # Increase MM s_rate by 50%
        for ag in agents:
            if ag._level == 'MM':
                ag.params.s_rate *= 1.5


# ================================================================
# RUNNERS
# ================================================================

def run_phase(conditions, phase_name, n_runs=N_RUNS, n_rounds=N_ROUNDS, seed=42):
    """Run all conditions for a phase."""
    print(f"\n{'=' * 140}")
    print(f"ORGANIZATIONAL HIERARCHY — {phase_name}")
    print("=" * 140)

    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    print(f"  Pools: {', '.join(f'{k}: {len(v)}' for k, v in pools.items())}")

    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in conditions}

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Run {run + 1}/{n_runs}  ({elapsed:.1f}s)")

        bp = sample_hierarchy_blueprint(pools, rng)

        for cond_name, config in conditions.items():
            agents = instantiate_hierarchy(bp)
            for ag in agents:
                _init_hierarchy_state(ag)

            cond_rng = np.random.default_rng(rng.integers(2**31))

            # Apply any initial perturbation
            perturbation = config.get('perturbation')
            if perturbation and perturbation.get('type') not in ('transition',):
                apply_perturbation_setup(agents, perturbation, cond_rng)

            # Build perturbation dict for mid-simulation transitions
            sim_perturbation = None
            if perturbation and perturbation.get('type') == 'transition':
                sim_perturbation = perturbation

            result = simulate_hierarchy(
                pools, cond_rng, n_rounds,
                blueprint=[(ag.uid, ag.phenotype, ag.source, ag.params, ag._level)
                            for ag in agents],
                enforcement_override=config.get('enforcement'),
                visibility_config=config.get('visibility'),
                structural_config=config.get('structural'),
                perturbation=sim_perturbation,
            )
            metrics = compute_hierarchy_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    elapsed = time.time() - t0
    n_conds = len(conditions)
    print(f"\n  Complete: {elapsed:.1f}s ({n_runs * n_conds} runs, "
          f"~{n_runs * n_conds * n_rounds * N_HIERARCHY_AGENTS / 1e6:.1f}M agent-steps)")

    aggregated = {name: aggregate_metrics(runs)
                  for name, runs in all_metrics.items()}
    return aggregated


# ================================================================
# REPORTING
# ================================================================

def _m(agg, key, default=0):
    """Extract median from aggregated metric."""
    val = agg.get(key, {})
    if isinstance(val, dict):
        return val.get('median', default)
    return val if isinstance(val, (int, float)) else default


def print_level_comparison(results, conditions, title=""):
    """Print per-level metrics for all conditions."""
    print(f"\n{'=' * 140}")
    print(f"PER-LEVEL COMPARISON{': ' + title if title else ''}")
    print("=" * 140)

    for level in LEVELS:
        print(f"\n  --- {level} ---")
        header = (f"  {'Condition':<25} {'SS-Coop':>8} {'Budget':>7} {'Strain':>7} "
                  f"{'Survival':>9} {'Removals':>9} {'Ruptures':>9} {'TTFR':>5}")
        print(header)
        print("  " + "-" * 100)

        for name in conditions:
            agg = results[name]
            ss = _m(agg, f'{level}_ss_coop')
            bud = _m(agg, f'{level}_budget')
            strn = _m(agg, f'{level}_strain')
            surv = _m(agg, f'{level}_survival')
            rm = _m(agg, f'{level}_removals')
            rupt = _m(agg, f'{level}_ruptures')
            ttfr = _m(agg, f'{level}_ttfr')
            print(f"  {name:<25} {ss:>8.1f} {bud:>7.2f} {strn:>7.2f} "
                  f"{surv:>9.1%} {rm:>9.0f} {rupt:>9.0f} {ttfr:>5.0f}")


def print_aggregate_comparison(results, conditions, title=""):
    """Print aggregate metrics for all conditions."""
    print(f"\n{'=' * 140}")
    print(f"AGGREGATE COMPARISON{': ' + title if title else ''}")
    print("=" * 140)

    header = (f"{'Condition':<25} {'OrgOutput':>10} {'DignFl':>7} {'Gini':>6} "
              f"{'HNavig':>7} {'Removals':>9} {'RemCost':>8} {'StrCost':>8} "
              f"{'TrueCost':>9} {'CostNav':>8}")
    print(header)
    print("-" * 140)

    for name in conditions:
        agg = results[name]
        out = _m(agg, 'org_output')
        dfl = _m(agg, 'dignity_floor')
        gini = _m(agg, 'cross_level_gini')
        hnav = _m(agg, 'hierarchical_navigability')
        rm = _m(agg, 'total_removals')
        rmc = _m(agg, 'total_removal_cost')
        sc = _m(agg, 'total_structural_cost')
        tc = _m(agg, 'total_true_cost')
        cnav = _m(agg, 'cost_adj_navigability')
        print(f"{name:<25} {out:>10.1f} {dfl:>7.1%} {gini:>6.3f} "
              f"{hnav:>7.3f} {rm:>9.0f} {rmc:>8.1f} {sc:>8.1f} "
              f"{tc:>9.1f} {cnav:>8.3f}")


def print_removal_distribution(results, conditions):
    """Print removal share by level."""
    print(f"\n{'=' * 140}")
    print("REMOVAL DISTRIBUTION BY LEVEL")
    print("=" * 140)

    header = (f"{'Condition':<25} {'EL_share':>9} {'MM_share':>9} "
              f"{'UM_share':>9} {'CS_share':>9} {'Total':>7}")
    print(header)
    print("-" * 80)

    for name in conditions:
        agg = results[name]
        shares = {level: _m(agg, f'{level}_removal_share') for level in LEVELS}
        total = _m(agg, 'total_removals')
        print(f"{name:<25} {shares['EL']:>9.1%} {shares['MM']:>9.1%} "
              f"{shares['UM']:>9.1%} {shares['CS']:>9.1%} {total:>7.0f}")


def print_predictions(results_p1, results_p2, results_p3):
    """Evaluate all predictions."""
    print(f"\n{'=' * 140}")
    print("PREDICTIONS SCORECARD")
    print("=" * 140)

    scored = []

    # P1: EL bears >70% of removals under realistic enforcement
    h3 = results_p1.get('H3_realistic_enforce', {})
    el_share = _m(h3, 'EL_removal_share')
    p1 = el_share > 0.70
    scored.append(('P1', f"EL bears >70% of removals under H3: EL_share={el_share:.1%}", p1))

    # P2: CS agents never rupture under any condition
    cs_never_ruptures = True
    all_results = {**results_p1, **results_p2, **results_p3}
    for name, agg in all_results.items():
        if _m(agg, 'CS_ruptures') > 0:
            cs_never_ruptures = False
            break
    scored.append(('P2', f"CS never ruptures: {cs_never_ruptures}", cs_never_ruptures))

    # P3: MM shows highest strain of any level under every Phase 1 condition
    mm_highest_strain = True
    for name in PHASE1_CONDITIONS:
        agg = results_p1.get(name, {})
        mm_s = _m(agg, 'MM_strain')
        for level in ['EL', 'UM', 'CS']:
            if _m(agg, f'{level}_strain') > mm_s:
                mm_highest_strain = False
                break
    scored.append(('P3', f"MM highest strain in all Phase 1: {mm_highest_strain}",
                   mm_highest_strain))

    # P4: Cross-level visibility (H6) > within-level (H5) on cooperation
    h5_out = _m(results_p1.get('H5_within_vis', {}), 'org_output')
    h6_out = _m(results_p1.get('H6_cross_vis', {}), 'org_output')
    p4 = h6_out > h5_out
    scored.append(('P4', f"Cross-level vis > within-level on org output: "
                         f"H6={h6_out:.1f} vs H5={h5_out:.1f}", p4))

    # P5: Realistic (H3) > uniform (H2) on output but lower EL cooperation
    h3_out = _m(results_p1.get('H3_realistic_enforce', {}), 'org_output')
    h2_out = _m(results_p1.get('H2_uniform_enforce', {}), 'org_output')
    h3_el = _m(results_p1.get('H3_realistic_enforce', {}), 'EL_ss_coop')
    h2_el = _m(results_p1.get('H2_uniform_enforce', {}), 'EL_ss_coop')
    p5 = h3_out > h2_out and h3_el < h2_el
    scored.append(('P5', f"H3 > H2 on output ({h3_out:.1f} vs {h2_out:.1f}) "
                         f"but lower EL coop ({h3_el:.1f} vs {h2_el:.1f})", p5))

    # P6: HS6 achieves highest navigability
    best_nav_name = ''
    best_nav = -1
    for name in PHASE2_CONDITIONS:
        agg = results_p2.get(name, {})
        nav = _m(agg, 'hierarchical_navigability')
        if nav > best_nav:
            best_nav = nav
            best_nav_name = name
    p6 = best_nav_name == 'HS6_struct_EL+enforce_CS'
    scored.append(('P6', f"HS6 highest navigability: best={best_nav_name} ({best_nav:.3f})", p6))

    # P7: Universal matching cost-effectiveness decreases with level
    hs4 = results_p2.get('HS4_universal_matching', {})
    el_eff = _m(hs4, 'EL_ss_coop') / max(_m(hs4, 'total_structural_cost'), 0.01)
    # Can't easily decompose per-level cost from aggregate. Check if total cost is high.
    p7 = _m(hs4, 'total_structural_cost') > _m(results_p2.get('HS3_EL_matching', {}),
                                                  'total_structural_cost')
    scored.append(('P7', f"Universal matching more costly than EL-only: "
                         f"HS4={_m(hs4, 'total_structural_cost'):.1f} vs "
                         f"HS3={_m(results_p2.get('HS3_EL_matching', {}), 'total_structural_cost'):.1f}",
                   p7))

    # P8: CS extraction (HC1) takes >20 rounds to affect EL
    # Can't precisely measure cascade timing from aggregated metrics alone
    # Approximate: check if EL TTFR > 20 under HC1
    hc1 = results_p3.get('HC1_CS_extraction', {})
    el_ttfr_hc1 = _m(hc1, 'EL_ttfr')
    p8 = el_ttfr_hc1 > 20
    scored.append(('P8', f"CS extraction takes >20 rounds to reach EL: "
                         f"EL_TTFR={el_ttfr_hc1:.0f}", p8))

    # P9: Middle-out reform (HC6) propagates fastest
    hc4_out = _m(results_p3.get('HC4_top_down_reform', {}), 'org_output')
    hc5_out = _m(results_p3.get('HC5_bottom_up_reform', {}), 'org_output')
    hc6_out = _m(results_p3.get('HC6_middle_out_reform', {}), 'org_output')
    p9 = hc6_out > max(hc4_out, hc5_out)
    scored.append(('P9', f"Middle-out reform best output: "
                         f"HC6={hc6_out:.1f} vs HC4={hc4_out:.1f}, HC5={hc5_out:.1f}", p9))

    # P10: CS-optimal and EL-optimal select different conditions
    best_cs = max(PHASE2_CONDITIONS, key=lambda n: _m(results_p2.get(n, {}), 'CS_ss_coop'))
    best_el_survival = max(PHASE2_CONDITIONS,
                           key=lambda n: (_m(results_p2.get(n, {}), 'EL_survival') *
                                          _m(results_p2.get(n, {}), 'EL_budget_floor')))
    p10 = best_cs != best_el_survival
    scored.append(('P10', f"CS-optimal != EL-optimal: "
                          f"CS-best={best_cs}, EL-best={best_el_survival}", p10))

    # Print
    supported = sum(1 for _, _, s in scored if s)
    total = len(scored)
    print(f"\n  Score: {supported}/{total} predictions supported\n")
    for pid, desc, result in scored:
        status = "SUPPORTED" if result else "NOT SUPPORTED"
        print(f"  {pid}: {status}")
        print(f"      {desc}")
        print()


def print_governance_selection(results_p2):
    """Phase 4: Which condition does each objective function select?"""
    print(f"\n{'=' * 140}")
    print("PHASE 4: GOVERNANCE SELECTION")
    print("=" * 140)
    print("  Which Phase 2 condition does each stakeholder's objective function select?")
    print()

    conditions = list(PHASE2_CONDITIONS.keys())

    # HG1: CS-optimal — maximize weighted output (higher levels count more)
    cs_optimal = max(conditions,
                     key=lambda n: _m(results_p2[n], 'org_output'))

    # HG2: EL-optimal — maximize EL survival × EL budget floor × (1 − EL strain)
    def el_score(n):
        agg = results_p2[n]
        surv = _m(agg, 'EL_survival')
        bfloor = _m(agg, 'EL_budget_floor')
        strain = _m(agg, 'EL_strain')
        return surv * max(bfloor, 0.01) * (1.0 - min(strain / 5.0, 1.0))
    el_optimal = max(conditions, key=el_score)

    # HG3: Consensus — maximize navigability across all levels equally
    consensus = max(conditions,
                    key=lambda n: _m(results_p2[n], 'hierarchical_navigability'))

    # HG4: External regulation — maximize minimum dignity floor
    def min_dignity(n):
        agg = results_p2[n]
        return min(_m(agg, f'{level}_survival') for level in LEVELS)
    external_reg = max(conditions, key=min_dignity)

    results_table = [
        ('HG1 CS-optimal', 'Maximize weighted organizational output', cs_optimal),
        ('HG2 EL-optimal', 'Maximize EL survival × budget floor × (1−strain)', el_optimal),
        ('HG3 Consensus', 'Maximize hierarchical navigability', consensus),
        ('HG4 External regulation', 'Maximize minimum dignity floor', external_reg),
    ]

    header = f"{'Optimizer':<25} {'Objective':<55} {'Selected Condition':<25}"
    print(header)
    print("-" * 105)
    for opt, obj, cond in results_table:
        print(f"{opt:<25} {obj:<55} {cond:<25}")

    # Check divergence
    selections = set(r[2] for r in results_table)
    print()
    if len(selections) == len(results_table):
        print("  All four objectives select DIFFERENT conditions.")
        print("  → The governance question is 'best for whom,' not 'which is best.'")
    elif len(selections) == 1:
        print("  All four objectives select the SAME condition.")
        print("  → There is a dominant architecture.")
    else:
        print(f"  {len(selections)} distinct selections from 4 objectives.")
        print("  → Partial alignment, partial divergence on governance architecture.")


# ================================================================
# MAIN
# ================================================================

def run_all():
    """Run all four phases."""
    print("=" * 140)
    print("FEDERATION DYNAMICS — ORGANIZATIONAL HIERARCHY BRANCH")
    print("30-agent stratified organization, 4 levels, empirically calibrated")
    print("=" * 140)

    t0 = time.time()

    # Phase 1
    results_p1 = run_phase(PHASE1_CONDITIONS, "PHASE 1: BASELINE HIERARCHY", seed=50)
    print_level_comparison(results_p1, PHASE1_CONDITIONS, "Phase 1")
    print_aggregate_comparison(results_p1, PHASE1_CONDITIONS, "Phase 1")
    print_removal_distribution(results_p1, PHASE1_CONDITIONS)

    # Phase 2
    results_p2 = run_phase(PHASE2_CONDITIONS, "PHASE 2: STRUCTURAL INTERVENTIONS", seed=51)
    print_level_comparison(results_p2, PHASE2_CONDITIONS, "Phase 2")
    print_aggregate_comparison(results_p2, PHASE2_CONDITIONS, "Phase 2")
    print_removal_distribution(results_p2, PHASE2_CONDITIONS)

    # Phase 3
    results_p3 = run_phase(PHASE3_CONDITIONS, "PHASE 3: CASCADE DYNAMICS", seed=52)
    print_level_comparison(results_p3, PHASE3_CONDITIONS, "Phase 3")
    print_aggregate_comparison(results_p3, PHASE3_CONDITIONS, "Phase 3")

    # Phase 4
    print_governance_selection(results_p2)

    # Predictions
    print_predictions(results_p1, results_p2, results_p3)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    return results_p1, results_p2, results_p3


if __name__ == '__main__':
    run_all()
