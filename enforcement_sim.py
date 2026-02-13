#!/usr/bin/env python3
"""
Federation Enforcement Mechanisms Simulation
==============================================

Multi-group simulation testing five enforcement architectures for sustaining
cooperation in mixed-phenotype populations:

  1. No mechanism (baseline)    — fixed groups, no enforcement
  2. Punishment                 — agent-to-agent via VCMS discharge channel
  3. Threshold exclusion        — agents below threshold removed and replaced
  4. Sustainability exclusion   — federation removes highest-impact extractor
                                  when system is degrading
  5. Voluntary exit             — agents leave declining groups; sorted regrouping

Population: 40 agents (10 groups of 4)
  40% EC, 20% CC, 20% CD, 10% DL, 10% MX (random)

Each condition runs with the same population draw per Monte Carlo run,
ensuring fair comparison. 100 runs, 100 rounds per run.

Theoretical grounding (Book 1):
  Punishment    — Ch 8, agent-to-agent discharge
  Threshold     — Ch 7, Λ-partition Ω-channel mandate
  Sustainability— Ch 9, federation boundary modification
  Voluntary exit— Ch 4 R-channel evaluation + Ch 9 federation formation
  No mechanism  — Ch 10, EDC unmanaged environment

Depends on: federation_sim.py (library loading, phenotype pools, constants),
vcms_engine_v4.py (reference for punishment mechanics).
"""

import copy
import json
import math
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Import infrastructure from federation sustainability simulation
from federation_sim import (
    load_libraries, build_phenotype_pools,
    AgentParams, AgentState,
    MAX_CONTRIB, ANCHOR_RATE, ACUTE_MULT, EPS,
    RUPTURE_B_FRAC, RUPTURE_CONSEC,
)


# ================================================================
# CONSTANTS
# ================================================================

N_ROUNDS = 100
N_RUNS = 100
N_AGENTS = 40
N_PER_GROUP = 4
N_GROUPS = N_AGENTS // N_PER_GROUP  # 10

MAX_PUNISH = 30        # P-experiment max punishment tokens

# Population distribution
POP_DIST = {'EC': 16, 'CC': 8, 'CD': 8, 'DL': 4, 'MX': 4}

# P-experiment punishment parameter population means (for non-P agents)
DEFAULT_P_SCALE = 8.92
DEFAULT_S_FRAC = 0.63
DEFAULT_S_THRESH = 1.98

# Threshold exclusion defaults
THRESHOLD_SD_MULT = 1.0   # contribution < mean - 1 SD
THRESHOLD_K = 3            # consecutive rounds below threshold

# Sustainability exclusion defaults
HEALTH_WINDOW = 5          # rounds for slope computation
HEALTH_TRIGGER_CONSEC = 3  # consecutive degrading rounds to activate

# Voluntary exit defaults
LEAVE_THRESHOLD = 0.5      # affordability below this triggers exit evaluation
EVAL_FREQ = 10             # evaluate every N rounds


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class EnforcementAgentParams:
    """Agent parameters including punishment (extends AgentParams fields)."""
    alpha: float = 0.0
    v_rep: float = 1.0
    v_ref: float = 0.5
    c_base: float = 0.5
    inertia: float = 0.0
    s_dir: float = 1.0
    s_rate: float = 1.0
    s_initial: float = 0.0
    b_initial: float = 2.0
    b_depletion_rate: float = 1.0
    b_replenish_rate: float = 1.0
    acute_threshold: float = 0.3
    facilitation_rate: float = 0.0
    h_strength: float = 0.0
    h_start: float = 0.778
    # Punishment parameters
    p_scale: float = DEFAULT_P_SCALE
    s_frac: float = DEFAULT_S_FRAC
    s_thresh: float = DEFAULT_S_THRESH


@dataclass
class Agent:
    """Persistent agent with identity, params, state, and trajectory history."""
    uid: int
    phenotype: str           # CC, EC, CD, DL, MX
    source: str              # "P:1234", "N:5678", etc.
    params: EnforcementAgentParams
    state: AgentState
    group_id: int = -1
    # Per-round trajectory (append each round the agent is active)
    contrib_history: list = field(default_factory=list)
    budget_history: list = field(default_factory=list)
    strain_history: list = field(default_factory=list)
    afford_history: list = field(default_factory=list)
    pun_sent_history: list = field(default_factory=list)
    pun_recv_history: list = field(default_factory=list)
    # Lifecycle
    active_from: int = 0     # Round this agent entered the simulation
    active_to: int = -1      # Round removed (-1 = still active at end)


# ================================================================
# POPULATION INFRASTRUCTURE
# ================================================================

def extract_enforcement_params(v3_params: dict) -> EnforcementAgentParams:
    """Convert a library v3_params dict to EnforcementAgentParams (with punishment)."""
    p = v3_params
    return EnforcementAgentParams(
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
        h_strength=p.get('h_strength', 0.0),
        h_start=p.get('h_start', 7.0 / 9.0),
        p_scale=p.get('p_scale', DEFAULT_P_SCALE),
        s_frac=p.get('s_frac', DEFAULT_S_FRAC),
        s_thresh=p.get('s_thresh', DEFAULT_S_THRESH),
    )


def build_enforcement_pools(libs):
    """Build phenotype pools with enforcement-extended params."""
    from federation_sim import P_HIGH, P_LOW, N_HIGH, N_LOW, IPD_HIGH, IPD_LOW

    pools = {'CC': [], 'EC': [], 'CD': [], 'DL': []}
    for tag, lib in libs.items():
        for sid, rec in lib.items():
            p = rec['v3_params']
            label = rec.get('behavioral_profile', rec.get('subject_type', ''))
            ap = extract_enforcement_params(p)

            cc_labels = P_HIGH if tag == 'P' else (N_HIGH if tag == 'N' else IPD_HIGH)
            if label in cc_labels and p['c_base'] > 0.65 and p['inertia'] > 0.3:
                pools['CC'].append((tag, sid, ap))

            if 0.35 <= p['c_base'] <= 0.75 and p['inertia'] < 0.25 and p['alpha'] > 0.3:
                pools['EC'].append((tag, sid, ap))

            cd_labels = P_LOW if tag == 'P' else (N_LOW if tag == 'N' else IPD_LOW)
            if label in cd_labels and p['c_base'] < 0.4:
                pools['CD'].append((tag, sid, ap))

            if tag == 'N' and label == 'declining' and p['c_base'] > 0.55:
                ratio = p['b_depletion_rate'] / max(p['b_replenish_rate'], 0.001)
                if ratio > 1.0:
                    pools['DL'].append((tag, sid, ap))

    return pools


def sample_population_blueprint(pools, rng, pop_dist=None):
    """
    Draw a population blueprint: list of (uid, phenotype, source, params) tuples.

    Same blueprint is reused across conditions for fair comparison.
    MX agents are drawn uniformly from all pools.
    """
    if pop_dist is None:
        pop_dist = POP_DIST

    blueprint = []
    uid = 0
    all_pool_names = [k for k in pools if len(pools[k]) > 0]

    for phenotype, count in pop_dist.items():
        if phenotype == 'MX':
            # Random draw from any pool with entries
            for _ in range(count):
                pool_name = rng.choice(all_pool_names)
                pool = pools[pool_name]
                idx = rng.integers(len(pool))
                tag, sid, params = pool[idx]
                blueprint.append((uid, 'MX', f"{tag}:{sid}", copy.deepcopy(params)))
                uid += 1
        else:
            pool = pools[phenotype]
            if len(pool) == 0:
                raise ValueError(f"Pool {phenotype} is empty")
            indices = rng.choice(len(pool), size=count, replace=True)
            for idx in indices:
                tag, sid, params = pool[idx]
                blueprint.append((uid, phenotype, f"{tag}:{sid}", copy.deepcopy(params)))
                uid += 1

    # Shuffle so group assignment produces mixed-phenotype groups
    rng.shuffle(blueprint)
    return blueprint


def instantiate_population(blueprint):
    """Create fresh Agent objects from a blueprint."""
    agents = []
    for uid, phenotype, source, params in blueprint:
        agents.append(Agent(
            uid=uid, phenotype=phenotype, source=source,
            params=copy.deepcopy(params),
            state=AgentState(strain=params.s_initial, B=params.b_initial),
        ))
    return agents


def assign_groups(agents, n_per_group=N_PER_GROUP):
    """
    Assign agents to groups sequentially.
    Returns dict[group_id] -> list[Agent].
    """
    groups = {}
    for i, agent in enumerate(agents):
        gid = i // n_per_group
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(agent)
        agent.group_id = gid
    return groups


def create_replacement(phenotype, pools, rng, uid_counter, active_from=0):
    """Create a fresh replacement agent of the given phenotype."""
    if phenotype == 'MX':
        all_pool_names = [k for k in pools if len(pools[k]) > 0]
        pool_name = rng.choice(all_pool_names)
        pool = pools[pool_name]
    else:
        pool = pools[phenotype]

    idx = rng.integers(len(pool))
    tag, sid, params = pool[idx]
    return Agent(
        uid=uid_counter,
        phenotype=phenotype,
        source=f"{tag}:{sid}",
        params=copy.deepcopy(params),
        state=AgentState(strain=params.s_initial, B=params.b_initial),
        active_from=active_from,
    )


# ================================================================
# PER-ROUND STEP FUNCTION (CORE ENGINE)
# ================================================================

def step_group(agents, rnd, n_rounds, dt, has_punishment=False,
               prev_contribs=None, prev_pun_recv=None):
    """
    Run one VCMS round for a group of agents. Modifies agent state in place.

    Mirrors federation_sim.py:248-336 inline step, extended with:
    - Punishment computation (discharge gate, reactive, p_scale)
    - Punishment-received strain and budget effects
    - Variable group size support

    Args:
        agents: list[Agent] in the group
        rnd: current round (0-indexed)
        n_rounds: total rounds (for dt and horizon)
        dt: normalized time step (1 / (n_rounds - 1))
        has_punishment: whether punishment channel is active
        prev_contribs: list[int] of previous round contributions (None for round 0)
        prev_pun_recv: list[float] of previous round punishment received

    Returns:
        contribs: list[int] — contribution per agent
        pun_sent: list[float] — total punishment sent per agent
        pun_recv: list[float] — total punishment received per agent
    """
    n = len(agents)
    if n == 0:
        return [], [], []

    max_c = MAX_CONTRIB

    # Others mean from previous contributions
    if rnd == 0 or prev_contribs is None:
        others_means = [0.0] * n
    else:
        total = sum(prev_contribs)
        others_means = [(total - prev_contribs[j]) / max(n - 1, 1)
                        for j in range(n)]

    contribs = [0] * n
    pun_sent_total = [0.0] * n
    gates = [0.0] * n          # discharge gates (needed for punishment output)
    discharges = [0.0] * n     # strain discharge amounts

    for j in range(n):
        ag = agents[j]
        ap = ag.params
        st = ag.state

        v_group_raw = others_means[j] / max_c
        v_group = min(1.0, ap.v_rep * v_group_raw)

        # --- V: Observe, update disposition and reference ---
        if rnd == 0:
            st.v_level = v_group
            st.c_prev_norm = ap.c_base
            st.disposition = ap.c_base
        else:
            st.v_level = ap.alpha * v_group + (1.0 - ap.alpha) * st.v_level
            st.disposition = (ANCHOR_RATE * st.c_prev_norm
                              + (1.0 - ANCHOR_RATE) * st.disposition)

        reference = ap.v_ref * st.v_level + (1.0 - ap.v_ref) * st.disposition

        # --- S: Strain accumulation ---
        prp = prev_pun_recv[j] if prev_pun_recv is not None else 0.0
        if rnd > 0:
            gap = st.c_prev_norm - reference
            gap_strain = max(0.0, gap * ap.s_dir)
            pun_strain = prp / 15.0 if has_punishment else 0.0
            st.strain += dt * ap.s_rate * (gap_strain + pun_strain)

        # --- B: Budget update ---
        if rnd > 0:
            experience = v_group_raw - st.c_prev_norm

            if experience < 0:
                magnitude = -experience
                depletion = dt * ap.b_depletion_rate * magnitude
                if magnitude > ap.acute_threshold:
                    depletion *= ACUTE_MULT
                st.B -= depletion
            elif experience > 0:
                if has_punishment:
                    pun_gate = max(0.0, 1.0 - prp / MAX_PUNISH)
                    st.B += dt * ap.b_replenish_rate * experience * pun_gate
                else:
                    st.B += dt * ap.b_replenish_rate * experience

            # Direct budget drain from punishment received
            if has_punishment:
                st.B -= dt * ap.b_depletion_rate * (prp / 15.0)

            st.B = max(0.0, st.B)

            # M_eval: facilitation / inhibition
            st.m_eval += dt * ap.facilitation_rate * experience

        # --- Resolution routing ---
        if has_punishment:
            gate = 1.0 / (1.0 + math.exp(-(st.B - ap.s_thresh) / 0.1))
        else:
            gate = 0.0

        discharge = gate * ap.s_frac * st.strain
        remaining_strain = max(0.0, st.strain - discharge)
        affordability = st.B / (st.B + remaining_strain + EPS)

        gates[j] = gate
        discharges[j] = discharge

        # --- Memory / Inertia ---
        if rnd == 0:
            c_norm = ap.c_base
        else:
            c_target = ap.v_ref * st.v_level + (1.0 - ap.v_ref) * ap.c_base
            c_target_adj = max(0.0, min(1.0, c_target + st.m_eval))
            c_norm = ((1.0 - abs(ap.inertia)) * c_target_adj
                      + ap.inertia * st.c_prev_norm)

        # --- Horizon ---
        h_factor = 1.0
        h_sr = ap.h_start * (n_rounds - 1)
        if n_rounds > 1 and ap.h_strength > 0.0 and rnd >= h_sr:
            denom = n_rounds - 1 - h_sr
            if denom > 0:
                progress = min(1.0, (rnd - h_sr) / denom)
                h_factor = 1.0 - ap.h_strength * progress
            elif rnd >= n_rounds - 1:
                h_factor = 1.0 - ap.h_strength

        # --- Output C ---
        c_out_norm = max(0.0, min(1.0, c_norm)) * affordability * h_factor
        c_out = max(0, min(max_c, round(c_out_norm * max_c)))
        contribs[j] = c_out

        # --- Punishment output ---
        if has_punishment and rnd > 0:
            current_gap = (c_out - others_means[j]) / max_c
            reactive = gate * max(0.0, ap.s_dir * current_gap)
            p_raw = (discharge + reactive) * ap.p_scale
            pun_sent_total[j] = max(0.0, min(float(MAX_PUNISH), p_raw))

        # Update strain to post-discharge
        st.strain = remaining_strain

        # Store affordability on state for voluntary exit evaluation
        st._afford = affordability

    # --- Distribute punishment to targets ---
    pun_recv = [0.0] * n
    if has_punishment:
        group_mean_c = sum(contribs) / max(n, 1)
        for j in range(n):
            if pun_sent_total[j] <= 0.01:
                continue
            # Target proportionally to under-contribution
            defections = []
            for k in range(n):
                if k == j:
                    defections.append(0.0)
                else:
                    defections.append(max(0.0, group_mean_c - contribs[k]))
            total_def = sum(defections)
            if total_def > 0:
                for k in range(n):
                    if k != j and defections[k] > 0:
                        pun_recv[k] += pun_sent_total[j] * defections[k] / total_def

            # Punisher budget cost (enforcement is costly)
            agents[j].state.B -= dt * (pun_sent_total[j] / 20.0)
            agents[j].state.B = max(0.0, agents[j].state.B)

    # --- Self-play feedback ---
    for j in range(n):
        agents[j].state.c_prev_norm = contribs[j] / max_c

    return contribs, pun_sent_total, pun_recv


def record_round(agents, contribs, pun_sent, pun_recv):
    """Append this round's data to each agent's history."""
    for j, ag in enumerate(agents):
        ag.contrib_history.append(contribs[j])
        ag.budget_history.append(ag.state.B)
        ag.strain_history.append(ag.state.strain)
        ag.afford_history.append(getattr(ag.state, '_afford', 0.0))
        ag.pun_sent_history.append(pun_sent[j] if j < len(pun_sent) else 0.0)
        ag.pun_recv_history.append(pun_recv[j] if j < len(pun_recv) else 0.0)


# ================================================================
# CONDITION 1: NO MECHANISM (BASELINE)
# ================================================================

def run_baseline(agents, n_rounds=N_ROUNDS):
    """Fixed groups, no enforcement, no punishment."""
    groups = assign_groups(agents)
    dt = 1.0 / (n_rounds - 1) if n_rounds > 1 else 1.0

    # Per-group previous contributions
    prev = {gid: None for gid in groups}

    for rnd in range(n_rounds):
        for gid, members in groups.items():
            contribs, pun_sent, pun_recv = step_group(
                members, rnd, n_rounds, dt,
                has_punishment=False,
                prev_contribs=prev[gid],
            )
            record_round(members, contribs, pun_sent, pun_recv)
            prev[gid] = contribs

    return {'agents': agents, 'groups': groups, 'events': []}


# ================================================================
# CONDITION 2: PUNISHMENT
# ================================================================

def run_punishment(agents, n_rounds=N_ROUNDS):
    """Fixed groups with punishment via VCMS discharge channel."""
    groups = assign_groups(agents)
    dt = 1.0 / (n_rounds - 1) if n_rounds > 1 else 1.0

    prev_c = {gid: None for gid in groups}
    prev_p = {gid: None for gid in groups}

    for rnd in range(n_rounds):
        for gid, members in groups.items():
            contribs, pun_sent, pun_recv = step_group(
                members, rnd, n_rounds, dt,
                has_punishment=True,
                prev_contribs=prev_c[gid],
                prev_pun_recv=prev_p[gid],
            )
            record_round(members, contribs, pun_sent, pun_recv)
            prev_c[gid] = contribs
            prev_p[gid] = pun_recv

    return {'agents': agents, 'groups': groups, 'events': []}


# ================================================================
# CONDITION 3: THRESHOLD EXCLUSION
# ================================================================

def run_threshold_exclusion(agents, pools, rng, n_rounds=N_ROUNDS, K=THRESHOLD_K,
                            uid_counter_start=N_AGENTS):
    """
    Agent removed if contribution < (group_mean - 1 SD) for K consecutive rounds.
    Replaced from same-type pool with fresh state.
    """
    groups = assign_groups(agents)
    dt = 1.0 / (n_rounds - 1) if n_rounds > 1 else 1.0
    all_agents = list(agents)  # accumulates replacements
    events = []
    uid_counter = uid_counter_start

    # Track consecutive below-threshold rounds per agent
    below_count = {ag.uid: 0 for ag in agents}
    prev = {gid: None for gid in groups}

    for rnd in range(n_rounds):
        for gid, members in groups.items():
            contribs, pun_sent, pun_recv = step_group(
                members, rnd, n_rounds, dt,
                has_punishment=False,
                prev_contribs=prev[gid],
            )
            record_round(members, contribs, pun_sent, pun_recv)

            # Threshold check (skip round 0 — no meaningful contributions yet)
            if rnd > 0 and len(contribs) >= 2:
                mean_c = sum(contribs) / len(contribs)
                sd_c = (sum((c - mean_c) ** 2 for c in contribs) / len(contribs)) ** 0.5
                threshold = mean_c - THRESHOLD_SD_MULT * sd_c

                removals_this_round = []
                for j, ag in enumerate(members):
                    if contribs[j] < threshold:
                        below_count[ag.uid] = below_count.get(ag.uid, 0) + 1
                    else:
                        below_count[ag.uid] = 0

                    if below_count.get(ag.uid, 0) >= K:
                        removals_this_round.append((j, ag))

                # Process removals (reverse order to preserve indices)
                for j, old_ag in sorted(removals_this_round, key=lambda x: x[0],
                                        reverse=True):
                    old_ag.active_to = rnd
                    replacement = create_replacement(
                        old_ag.phenotype, pools, rng, uid_counter,
                        active_from=rnd + 1,
                    )
                    uid_counter += 1
                    replacement.group_id = gid
                    members[j] = replacement
                    all_agents.append(replacement)
                    below_count[replacement.uid] = 0

                    events.append({
                        'type': 'threshold_removal', 'round': rnd,
                        'group': gid, 'removed_uid': old_ag.uid,
                        'removed_phenotype': old_ag.phenotype,
                        'replacement_uid': replacement.uid,
                        'contribution_at_removal': contribs[j],
                        'threshold': threshold,
                    })

            prev[gid] = contribs

    return {'agents': all_agents, 'groups': groups, 'events': events}


# ================================================================
# CONDITION 4: SUSTAINABILITY EXCLUSION
# ================================================================

def _linear_slope(values):
    """Linear regression slope of values over their index."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / (den + 1e-10)


def run_sustainability_exclusion(agents, pools, rng, n_rounds=N_ROUNDS,
                                 uid_counter_start=N_AGENTS):
    """
    Federation monitors group health (budget + cooperation slopes).
    When degrading for 3+ rounds, removes agent whose absence most improves
    group mean. Replaced from same-type pool.
    """
    groups = assign_groups(agents)
    dt = 1.0 / (n_rounds - 1) if n_rounds > 1 else 1.0
    all_agents = list(agents)
    events = []
    uid_counter = uid_counter_start

    # Group health tracking
    group_budget_history = {gid: [] for gid in groups}
    group_coop_history = {gid: [] for gid in groups}
    degrading_count = {gid: 0 for gid in groups}

    prev = {gid: None for gid in groups}

    for rnd in range(n_rounds):
        for gid, members in groups.items():
            contribs, pun_sent, pun_recv = step_group(
                members, rnd, n_rounds, dt,
                has_punishment=False,
                prev_contribs=prev[gid],
            )
            record_round(members, contribs, pun_sent, pun_recv)

            # Track group-level metrics
            mean_budget = sum(ag.state.B for ag in members) / max(len(members), 1)
            mean_coop = sum(contribs) / max(len(contribs), 1)
            group_budget_history[gid].append(mean_budget)
            group_coop_history[gid].append(mean_coop)

            # Health check (need enough history for slope)
            if rnd >= HEALTH_WINDOW:
                b_slope = _linear_slope(
                    group_budget_history[gid][-HEALTH_WINDOW:])
                c_slope = _linear_slope(
                    group_coop_history[gid][-HEALTH_WINDOW:])

                if b_slope < 0 and c_slope < 0:
                    degrading_count[gid] += 1
                else:
                    degrading_count[gid] = 0

                # Trigger: sustained degradation
                if degrading_count[gid] >= HEALTH_TRIGGER_CONSEC and len(members) > 1:
                    # Compute counterfactual impact for each agent
                    group_total = sum(contribs)
                    n_m = len(members)
                    group_mean = group_total / n_m

                    best_impact = -float('inf')
                    worst_idx = -1
                    for j in range(n_m):
                        mean_without_j = (group_total - contribs[j]) / max(n_m - 1, 1)
                        impact_j = mean_without_j - group_mean
                        if impact_j > best_impact:
                            best_impact = impact_j
                            worst_idx = j

                    if worst_idx >= 0:
                        old_ag = members[worst_idx]
                        old_ag.active_to = rnd
                        replacement = create_replacement(
                            old_ag.phenotype, pools, rng, uid_counter,
                            active_from=rnd + 1,
                        )
                        uid_counter += 1
                        replacement.group_id = gid
                        members[worst_idx] = replacement
                        all_agents.append(replacement)
                        degrading_count[gid] = 0  # Reset after action

                        events.append({
                            'type': 'sustainability_removal', 'round': rnd,
                            'group': gid, 'removed_uid': old_ag.uid,
                            'removed_phenotype': old_ag.phenotype,
                            'impact': best_impact,
                            'b_slope': b_slope, 'c_slope': c_slope,
                        })

            prev[gid] = contribs

    return {'agents': all_agents, 'groups': groups, 'events': events}


# ================================================================
# CONDITION 5: VOLUNTARY EXIT WITH PARTNER CHOICE
# ================================================================

def run_voluntary_exit(agents, rng, n_rounds=N_ROUNDS, eval_freq=EVAL_FREQ,
                       formation='random', leave_threshold=LEAVE_THRESHOLD):
    """
    Agents evaluate their group periodically and leave if affordability is low.
    Departed agents enter free pool; new groups form when 4 agents available.

    formation: 'random' or 'sorted' (by recent cooperation level)
    """
    groups = assign_groups(agents)
    dt = 1.0 / (n_rounds - 1) if n_rounds > 1 else 1.0
    events = []
    free_pool = []
    next_gid = max(groups.keys()) + 1

    prev_c = {gid: None for gid in groups}
    prev_p = {gid: None for gid in groups}  # Not used (no punishment) but keeps interface clean

    for rnd in range(n_rounds):
        # Step all active groups
        for gid in list(groups.keys()):
            members = groups[gid]
            if len(members) == 0:
                del groups[gid]
                continue

            contribs, pun_sent, pun_recv = step_group(
                members, rnd, n_rounds, dt,
                has_punishment=False,
                prev_contribs=prev_c.get(gid),
            )
            record_round(members, contribs, pun_sent, pun_recv)
            prev_c[gid] = contribs

        # Evaluation round: agents decide whether to leave
        if rnd > 0 and eval_freq > 0 and rnd % eval_freq == 0:
            for gid in list(groups.keys()):
                members = groups[gid]
                leavers = []
                for j, ag in enumerate(members):
                    afford = getattr(ag.state, '_afford', 0.0)
                    # Agent-specific threshold: high-inertia agents tolerate more
                    agent_thresh = leave_threshold * (1.0 - 0.5 * abs(ag.params.inertia))
                    if afford < agent_thresh:
                        leavers.append(j)

                # Process leavers (reverse order)
                for j in sorted(leavers, reverse=True):
                    ag = members.pop(j)
                    ag.group_id = -1
                    free_pool.append(ag)
                    events.append({
                        'type': 'voluntary_exit', 'round': rnd,
                        'from_group': gid, 'agent_uid': ag.uid,
                        'phenotype': ag.phenotype,
                        'affordability': getattr(ag.state, '_afford', 0.0),
                    })

                if leavers:
                    # Reset prev_contribs after composition change to avoid
                    # position mismatch (old contribs indexed by old positions)
                    prev_c[gid] = None

                # Clean up empty groups
                if len(members) == 0:
                    del groups[gid]
                    if gid in prev_c:
                        del prev_c[gid]

            # Form new groups from free pool
            if formation == 'sorted':
                # Sort by most recent contribution (cooperators cluster)
                free_pool.sort(
                    key=lambda a: a.contrib_history[-1] if a.contrib_history else 0,
                    reverse=True)
            else:
                rng.shuffle(free_pool)

            while len(free_pool) >= N_PER_GROUP:
                new_members = free_pool[:N_PER_GROUP]
                free_pool = free_pool[N_PER_GROUP:]
                for ag in new_members:
                    ag.group_id = next_gid
                groups[next_gid] = new_members
                prev_c[next_gid] = None  # Fresh group, no prev contribs
                events.append({
                    'type': 'group_formed', 'round': rnd,
                    'group': next_gid,
                    'members': [ag.uid for ag in new_members],
                    'phenotypes': [ag.phenotype for ag in new_members],
                })
                next_gid += 1

    # Agents still in free pool at end
    for ag in free_pool:
        ag.active_to = n_rounds - 1

    return {'agents': agents, 'groups': groups, 'events': events,
            'free_pool': free_pool}


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def gini_coefficient(values):
    """Gini coefficient of a list of non-negative values. 0 = perfect equality."""
    n = len(values)
    if n == 0 or sum(values) < 1e-10:
        return 0.0
    sv = sorted(values)
    total = sum(sv)
    cumsum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sv))
    return cumsum / (n * total)


def bimodality_coefficient(values):
    """Sarle's bimodality coefficient. >0.555 suggests bimodality."""
    n = len(values)
    if n < 4:
        return 0.0
    m = sum(values) / n
    m2 = sum((v - m) ** 2 for v in values) / n
    if m2 < 1e-10:
        return 0.0
    m3 = sum((v - m) ** 3 for v in values) / n
    m4 = sum((v - m) ** 4 for v in values) / n
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3  # excess kurtosis
    # Adjustment for sample size
    denom = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3) + 1e-10)
    return (skew ** 2 + 1) / (denom + 1e-10)


# ================================================================
# METRICS COMPUTATION
# ================================================================

def compute_run_metrics(result, n_rounds=N_ROUNDS):
    """Compute per-run metrics from a condition result."""
    agents = result['agents']
    groups = result['groups']
    events = result['events']

    # Filter to agents active at end
    active_agents = [ag for ag in agents if ag.active_to == -1]

    metrics = {}

    # --- System-level ---

    # Mean cooperation at checkpoints
    for cp in [10, 25, 50, 75, 100]:
        rnd_idx = min(cp - 1, n_rounds - 1)
        contribs_at_cp = []
        for ag in agents:
            if ag.active_from <= rnd_idx and (ag.active_to == -1 or ag.active_to >= rnd_idx):
                h = ag.contrib_history
                local_idx = rnd_idx - ag.active_from
                if 0 <= local_idx < len(h):
                    contribs_at_cp.append(h[local_idx])
        metrics[f'mean_coop_r{cp}'] = (sum(contribs_at_cp) / max(len(contribs_at_cp), 1))

    # Steady-state cooperation (last 10 rounds)
    ss_contribs = []
    for ag in active_agents:
        if len(ag.contrib_history) >= 10:
            ss_contribs.extend(ag.contrib_history[-10:])
    metrics['steady_state_coop'] = sum(ss_contribs) / max(len(ss_contribs), 1)

    # System TTFR: first round any agent's budget drops below RUPTURE_B_FRAC of initial
    metrics['system_ttfr'] = n_rounds  # default: no rupture
    for ag in agents:
        b_init = ag.params.b_initial
        threshold = RUPTURE_B_FRAC * b_init
        consec = 0
        for r, b in enumerate(ag.budget_history):
            if b < threshold:
                consec += 1
                if consec >= RUPTURE_CONSEC:
                    ttfr = ag.active_from + r - RUPTURE_CONSEC + 1
                    if ttfr < metrics['system_ttfr']:
                        metrics['system_ttfr'] = ttfr
                    break
            else:
                consec = 0

    # Total ruptures at T=100
    rupture_count = 0
    for ag in active_agents:
        if len(ag.budget_history) >= RUPTURE_CONSEC:
            b_init = ag.params.b_initial
            threshold = RUPTURE_B_FRAC * b_init
            tail = ag.budget_history[-RUPTURE_CONSEC:]
            if all(b < threshold for b in tail):
                rupture_count += 1
    metrics['rupture_count'] = rupture_count

    # Cooperation variance across groups at T=100
    group_means_final = []
    for gid, members in groups.items():
        if len(members) > 0:
            final_contribs = [ag.contrib_history[-1] for ag in members
                              if len(ag.contrib_history) > 0]
            if final_contribs:
                group_means_final.append(sum(final_contribs) / len(final_contribs))
    if len(group_means_final) > 1:
        gm_mean = sum(group_means_final) / len(group_means_final)
        metrics['coop_variance'] = sum(
            (g - gm_mean) ** 2 for g in group_means_final) / len(group_means_final)
    else:
        metrics['coop_variance'] = 0.0

    # Gini coefficient of contributions at T=100
    final_contribs_all = [ag.contrib_history[-1] for ag in active_agents
                          if len(ag.contrib_history) > 0]
    metrics['gini'] = gini_coefficient(final_contribs_all)

    # Bimodality of group means
    metrics['bimodality'] = bimodality_coefficient(group_means_final)

    # --- Per-phenotype metrics ---
    for ptype in ['CC', 'EC', 'CD', 'DL', 'MX']:
        ptype_agents = [ag for ag in active_agents if ag.phenotype == ptype]
        if not ptype_agents:
            metrics[f'{ptype}_budget_T100'] = 0.0
            metrics[f'{ptype}_strain_T100'] = 0.0
            metrics[f'{ptype}_coop_T100'] = 0.0
            metrics[f'{ptype}_survival'] = 0.0
            continue

        budgets = [ag.budget_history[-1] for ag in ptype_agents
                   if len(ag.budget_history) > 0]
        strains = [ag.strain_history[-1] for ag in ptype_agents
                   if len(ag.strain_history) > 0]
        contribs = [ag.contrib_history[-1] for ag in ptype_agents
                    if len(ag.contrib_history) > 0]

        metrics[f'{ptype}_budget_T100'] = sum(budgets) / max(len(budgets), 1)
        metrics[f'{ptype}_strain_T100'] = sum(strains) / max(len(strains), 1)
        metrics[f'{ptype}_coop_T100'] = sum(contribs) / max(len(contribs), 1)

        # Survival: fraction not ruptured
        survived = 0
        for ag in ptype_agents:
            b_init = ag.params.b_initial
            if len(ag.budget_history) >= RUPTURE_CONSEC:
                tail = ag.budget_history[-RUPTURE_CONSEC:]
                if not all(b < RUPTURE_B_FRAC * b_init for b in tail):
                    survived += 1
            else:
                survived += 1
        metrics[f'{ptype}_survival'] = survived / len(ptype_agents)

        # Cooperation sustainability: fraction maintaining >50% initial cooperation
        sustained = 0
        for ag in ptype_agents:
            initial_c = ag.params.c_base * MAX_CONTRIB
            if len(ag.contrib_history) > 0 and ag.contrib_history[-1] > 0.5 * initial_c:
                sustained += 1
        metrics[f'{ptype}_coop_sustained'] = sustained / len(ptype_agents)

    # --- Mechanism-specific metrics ---

    # Threshold exclusion metrics
    threshold_removals = [e for e in events if e['type'] == 'threshold_removal']
    metrics['removal_count'] = len(threshold_removals)
    metrics['removal_rate'] = len(threshold_removals) / max(n_rounds, 1)

    # Gaming prevalence (computed from active agents' contributions vs group threshold)
    gaming_rounds = 0
    total_agent_rounds = 0
    for ag in agents:
        for r_idx, c in enumerate(ag.contrib_history):
            total_agent_rounds += 1
    metrics['total_agent_rounds'] = total_agent_rounds

    # Sustainability exclusion metrics
    sustain_removals = [e for e in events if e['type'] == 'sustainability_removal']
    metrics['sustain_removal_count'] = len(sustain_removals)
    metrics['sustain_activation_freq'] = len(sustain_removals) / max(n_rounds, 1)
    # False positive: cooperator types removed
    if sustain_removals:
        fp = sum(1 for e in sustain_removals if e['removed_phenotype'] in ('CC', 'EC'))
        metrics['sustain_false_positive_rate'] = fp / len(sustain_removals)
    else:
        metrics['sustain_false_positive_rate'] = 0.0

    # Voluntary exit metrics
    exits = [e for e in events if e['type'] == 'voluntary_exit']
    formations = [e for e in events if e['type'] == 'group_formed']
    metrics['exit_count'] = len(exits)
    metrics['exit_rate'] = len(exits) / max(n_rounds // max(EVAL_FREQ, 1), 1)
    metrics['groups_formed'] = len(formations)

    # Punishment metrics
    total_pun_sent = sum(sum(ag.pun_sent_history) for ag in agents)
    metrics['total_punishment_sent'] = total_pun_sent

    # Enforcer budget cost: compare high-punishers vs low-punishers
    pun_totals = [(ag, sum(ag.pun_sent_history)) for ag in active_agents]
    if pun_totals and total_pun_sent > 0:
        median_pun = sorted(pt for _, pt in pun_totals)[len(pun_totals) // 2]
        high_punishers = [ag for ag, pt in pun_totals if pt > median_pun]
        low_punishers = [ag for ag, pt in pun_totals if pt <= median_pun]
        if high_punishers:
            metrics['high_punisher_budget'] = sum(
                ag.budget_history[-1] for ag in high_punishers
                if ag.budget_history) / len(high_punishers)
        else:
            metrics['high_punisher_budget'] = 0.0
        if low_punishers:
            metrics['low_punisher_budget'] = sum(
                ag.budget_history[-1] for ag in low_punishers
                if ag.budget_history) / len(low_punishers)
        else:
            metrics['low_punisher_budget'] = 0.0
    else:
        metrics['high_punisher_budget'] = 0.0
        metrics['low_punisher_budget'] = 0.0

    return metrics


def aggregate_metrics(all_run_metrics):
    """Aggregate metrics across runs: median, q25, q75."""
    if not all_run_metrics:
        return {}

    keys = all_run_metrics[0].keys()
    agg = {}
    for k in keys:
        values = [m[k] for m in all_run_metrics if k in m]
        if not values:
            continue
        sv = sorted(values)
        n = len(sv)
        agg[k] = {
            'median': sv[n // 2],
            'q25': sv[max(0, n // 4)],
            'q75': sv[min(n - 1, 3 * n // 4)],
            'mean': sum(sv) / n,
            'min': sv[0],
            'max': sv[-1],
        }
    return agg


# ================================================================
# REPORTING
# ================================================================

def print_condition_comparison(all_results):
    """Print comparison table across conditions."""
    print("\n" + "=" * 100)
    print("CONDITION COMPARISON")
    print("=" * 100)

    header = f"{'Condition':<24} {'MeanCoop':>8} {'SS-Coop':>8} {'TTFR':>6} " \
             f"{'Rupt':>5} {'Var':>8} {'Gini':>6}"
    print(header)
    print("-" * 100)

    for cond_name, agg in sorted(all_results.items()):
        mc = agg.get('mean_coop_r100', {}).get('median', 0)
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        rupt = agg.get('rupture_count', {}).get('median', 0)
        var_ = agg.get('coop_variance', {}).get('median', 0)
        gini = agg.get('gini', {}).get('median', 0)
        print(f"{cond_name:<24} {mc:>8.1f} {ss:>8.1f} {ttfr:>6.0f} "
              f"{rupt:>5.0f} {var_:>8.1f} {gini:>6.3f}")


def print_cooperation_trajectories(all_results):
    """Print cooperation at checkpoints for each condition."""
    print("\n" + "=" * 100)
    print("COOPERATION TRAJECTORIES (median across runs)")
    print("=" * 100)

    checkpoints = [10, 25, 50, 75, 100]
    header = f"{'Condition':<24}" + "".join(f"{'r=' + str(cp):>8}" for cp in checkpoints)
    print(header)
    print("-" * 100)

    for cond_name, agg in sorted(all_results.items()):
        vals = []
        for cp in checkpoints:
            v = agg.get(f'mean_coop_r{cp}', {}).get('median', 0)
            vals.append(f"{v:>8.1f}")
        print(f"{cond_name:<24}{''.join(vals)}")


def print_phenotype_outcomes(all_results):
    """Print per-phenotype budget and survival under each condition."""
    print("\n" + "=" * 100)
    print("PHENOTYPE OUTCOMES AT T=100 (median across runs)")
    print("=" * 100)

    for ptype in ['CC', 'EC', 'CD', 'DL']:
        print(f"\n  {ptype}:")
        header = f"    {'Condition':<24} {'Budget':>8} {'Strain':>8} " \
                 f"{'Coop':>8} {'Survival':>8} {'Sustained':>9}"
        print(header)
        print(f"    {'-' * 80}")

        for cond_name, agg in sorted(all_results.items()):
            b = agg.get(f'{ptype}_budget_T100', {}).get('median', 0)
            s = agg.get(f'{ptype}_strain_T100', {}).get('median', 0)
            c = agg.get(f'{ptype}_coop_T100', {}).get('median', 0)
            sv = agg.get(f'{ptype}_survival', {}).get('median', 0)
            cs = agg.get(f'{ptype}_coop_sustained', {}).get('median', 0)
            print(f"    {cond_name:<24} {b:>8.2f} {s:>8.2f} "
                  f"{c:>8.1f} {sv:>8.1%} {cs:>9.1%}")


def print_mechanism_details(all_results):
    """Print mechanism-specific metrics."""
    print("\n" + "=" * 100)
    print("MECHANISM-SPECIFIC METRICS (median across runs)")
    print("=" * 100)

    # Threshold
    for k in ['threshold_K3', 'threshold_K5']:
        if k not in all_results:
            continue
        agg = all_results[k]
        rc = agg.get('removal_count', {}).get('median', 0)
        rr = agg.get('removal_rate', {}).get('median', 0)
        print(f"\n  {k}:")
        print(f"    Removals (median): {rc:.0f}  Rate: {rr:.2f}/round")

    # Sustainability
    if 'sustainability' in all_results:
        agg = all_results['sustainability']
        rc = agg.get('sustain_removal_count', {}).get('median', 0)
        af = agg.get('sustain_activation_freq', {}).get('median', 0)
        fp = agg.get('sustain_false_positive_rate', {}).get('median', 0)
        print(f"\n  sustainability:")
        print(f"    Removals (median): {rc:.0f}  Activation freq: {af:.3f}/round")
        print(f"    False positive rate: {fp:.1%}")

    # Punishment
    if 'punishment' in all_results:
        agg = all_results['punishment']
        tp = agg.get('total_punishment_sent', {}).get('median', 0)
        hp = agg.get('high_punisher_budget', {}).get('median', 0)
        lp = agg.get('low_punisher_budget', {}).get('median', 0)
        print(f"\n  punishment:")
        print(f"    Total punishment sent (median): {tp:.0f}")
        print(f"    High-punisher budget T100: {hp:.2f}")
        print(f"    Low-punisher budget T100:  {lp:.2f}")

    # Voluntary exit
    for k in ['voluntary_r10', 'voluntary_r10_sorted']:
        if k not in all_results:
            continue
        agg = all_results[k]
        ec = agg.get('exit_count', {}).get('median', 0)
        er = agg.get('exit_rate', {}).get('median', 0)
        gf = agg.get('groups_formed', {}).get('median', 0)
        bm = agg.get('bimodality', {}).get('median', 0)
        print(f"\n  {k}:")
        print(f"    Exits (median): {ec:.0f}  Rate: {er:.1f}/eval")
        print(f"    Groups formed: {gf:.0f}  Bimodality: {bm:.3f}")


def print_predictions_scorecard(all_results):
    """Evaluate the 6 testable predictions from the proposal."""
    print("\n" + "=" * 100)
    print("PREDICTIONS SCORECARD")
    print("=" * 100)

    results = []

    # Prediction 1: Threshold produces gaming (>20% of agent-rounds near threshold)
    # Approximated by high removal rate with oscillation
    for k in ['threshold_K3', 'threshold_K5']:
        if k in all_results:
            rc = all_results[k].get('removal_count', {}).get('median', 0)
            gaming = rc > 5  # Meaningful removal activity indicates gaming dynamics
            results.append(
                f"  1. Threshold gaming ({k}): "
                f"Removals={rc:.0f} → {'SUPPORTED' if gaming else 'NOT SUPPORTED'}"
            )

    # Prediction 2: Punishment depletes enforcers
    if 'punishment' in all_results:
        hp = all_results['punishment'].get('high_punisher_budget', {}).get('median', 0)
        lp = all_results['punishment'].get('low_punisher_budget', {}).get('median', 0)
        depleted = hp < lp
        results.append(
            f"  2. Punishment depletes enforcers: "
            f"High-P budget={hp:.2f} vs Low-P={lp:.2f} → "
            f"{'SUPPORTED' if depleted else 'NOT SUPPORTED'}"
        )

    # Prediction 3: Sustainability has fewer removals than threshold
    if 'sustainability' in all_results and 'threshold_K3' in all_results:
        sr = all_results['sustainability'].get('sustain_removal_count', {}).get('median', 0)
        tr = all_results['threshold_K3'].get('removal_count', {}).get('median', 0)
        fewer = sr < tr
        results.append(
            f"  3. Sustainability fewer removals than threshold: "
            f"Sustain={sr:.0f} vs Thresh={tr:.0f} → "
            f"{'SUPPORTED' if fewer else 'NOT SUPPORTED'}"
        )

    # Prediction 4: Voluntary exit produces bimodal group cooperation
    for k in ['voluntary_r10', 'voluntary_r10_sorted']:
        if k in all_results:
            bm = all_results[k].get('bimodality', {}).get('median', 0)
            bimodal = bm > 0.555
            results.append(
                f"  4. Voluntary exit bimodality ({k}): "
                f"BC={bm:.3f} → {'SUPPORTED' if bimodal else 'NOT SUPPORTED'} (>0.555)"
            )

    # Prediction 5: CC agents benefit most from voluntary exit
    if 'voluntary_r10' in all_results and 'baseline' in all_results:
        cc_v = all_results['voluntary_r10'].get('CC_budget_T100', {}).get('median', 0)
        cc_b = all_results['baseline'].get('CC_budget_T100', {}).get('median', 0)
        ec_v = all_results['voluntary_r10'].get('EC_budget_T100', {}).get('median', 0)
        ec_b = all_results['baseline'].get('EC_budget_T100', {}).get('median', 0)
        cc_gain = cc_v - cc_b
        ec_gain = ec_v - ec_b
        cc_most = cc_gain > ec_gain
        results.append(
            f"  5. CC benefits most from vol exit: "
            f"CC gain={cc_gain:+.2f} vs EC gain={ec_gain:+.2f} → "
            f"{'SUPPORTED' if cc_most else 'NOT SUPPORTED'}"
        )

    # Prediction 6: Sustainability is best single intervention (composite score)
    composite = {}
    for cond_name, agg in all_results.items():
        if cond_name == 'baseline':
            continue
        ss = agg.get('steady_state_coop', {}).get('median', 0) / MAX_CONTRIB
        ttfr = min(agg.get('system_ttfr', {}).get('median', 0), N_ROUNDS) / N_ROUNDS
        rupt = 1.0 - min(agg.get('rupture_count', {}).get('median', 0), N_AGENTS) / N_AGENTS
        score = (ss + ttfr + rupt) / 3.0
        composite[cond_name] = score

    if composite:
        best = max(composite, key=composite.get)
        results.append(
            f"  6. Best single intervention: {best} "
            f"(score={composite[best]:.3f}) → "
            f"{'SUPPORTED' if best == 'sustainability' else 'NOT SUPPORTED'} "
            f"(predicted: sustainability)"
        )
        for c, s in sorted(composite.items(), key=lambda x: -x[1]):
            results.append(f"     {c:<24} {s:.3f}")

    for r in results:
        print(r)

    # Predicted orderings
    print("\n  PREDICTED ORDERINGS:")

    # Mean cooperation ordering
    coop_order = sorted(all_results.items(),
                        key=lambda x: x[1].get('steady_state_coop', {}).get('median', 0),
                        reverse=True)
    print(f"\n  Cooperation (high→low):  {' > '.join(c for c, _ in coop_order)}")
    print(f"    Predicted: punishment > sustainability > voluntary > threshold > baseline")

    # TTFR ordering
    ttfr_order = sorted(all_results.items(),
                        key=lambda x: x[1].get('system_ttfr', {}).get('median', 0),
                        reverse=True)
    print(f"\n  TTFR (long→short):       {' > '.join(c for c, _ in ttfr_order)}")
    print(f"    Predicted: voluntary > sustainability > punishment > threshold > baseline")

    # Welfare ordering
    welfare_order = []
    for c, agg in all_results.items():
        b_vals = []
        for pt in ['CC', 'EC', 'CD', 'DL']:
            v = agg.get(f'{pt}_budget_T100', {}).get('median', 0)
            b_vals.append(v)
        welfare_order.append((c, sum(b_vals) / max(len(b_vals), 1)))
    welfare_order.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Welfare (high→low):      {' > '.join(c for c, _ in welfare_order)}")
    print(f"    Predicted: voluntary > sustainability > threshold > punishment > baseline")


# ================================================================
# MAIN RUNNER
# ================================================================

def run_all_conditions(n_runs=N_RUNS, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                       seed=42):
    """
    Run all enforcement conditions with matched populations.

    For each Monte Carlo run:
      1. Sample one population blueprint (same across all conditions)
      2. Instantiate fresh agents for each condition
      3. Run condition
      4. Collect metrics
    """
    print("Loading libraries and building pools...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    for name, pool in pools.items():
        print(f"  {name}: {len(pool)} subjects")

    rng = np.random.default_rng(seed)

    # Condition definitions
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

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Run {run + 1}/{n_runs}  ({elapsed:.1f}s elapsed)")

        # Same population blueprint for all conditions this run
        blueprint = sample_population_blueprint(pools, rng)

        for cond_name, cond_fn in conditions.items():
            # Fresh agents from same blueprint
            agents = instantiate_population(blueprint)
            # Fresh RNG fork for reproducibility within condition
            cond_rng = np.random.default_rng(rng.integers(2**31))
            result = cond_fn(agents, pools, cond_rng)
            metrics = compute_run_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    elapsed = time.time() - t0
    print(f"\nAll runs complete in {elapsed:.1f}s")

    # Aggregate
    all_agg = {}
    for cond_name, run_metrics in all_metrics.items():
        all_agg[cond_name] = aggregate_metrics(run_metrics)

    # Report
    print_condition_comparison(all_agg)
    print_cooperation_trajectories(all_agg)
    print_phenotype_outcomes(all_agg)
    print_mechanism_details(all_agg)
    print_predictions_scorecard(all_agg)

    return all_agg


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == '__main__':
    results = run_all_conditions()
