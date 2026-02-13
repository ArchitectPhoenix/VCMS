#!/usr/bin/env python3
"""
Federation Dynamics Phase 2 — Transition, Combination, and EC Protection
=========================================================================

Three connected tests building on Phase 1 enforcement mechanisms results:

Test 1 (Transition): Can systems recover from punishment architecture?
  7 conditions × 100 runs × 200 rounds. Mechanism switch at round 100.

Test 2 (Combination): Do sustainability + voluntary exit outperform either alone?
  6 conditions × 100 runs × 100 rounds.

Test 3 (EC Protection): Does care-first enforcement protect transparent agents?
  Part A: Cross-condition analytical (no new sim)
  Part B: Inertia sensitivity — 3 variants × 2 conditions × 100 runs
  Part C: Canary index — legibility analysis from trajectory data

Total: ~104M agent-steps, ~40 seconds.
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


# ================================================================
# GENERALIZED SIMULATION LOOP
# ================================================================

def simulate(agents, pools, rng, n_rounds=100, mechanisms=None,
             switch_round=None, mechanisms_after=None):
    """
    Run n_rounds with pluggable, switchable mechanisms.

    mechanisms: list of (name, kwargs) active for rounds [0, switch_round).
      name: 'none' | 'punishment' | 'threshold' | 'sustainability' | 'voluntary'
    mechanisms_after: list active for rounds [switch_round, n_rounds).
      If None, same as mechanisms (no switch).
    switch_round: round at which to change mechanisms (None = no switch).

    dt = 1/(n_rounds-1) for full horizon. Agent state is continuous across switch.

    Mechanism ordering per round:
      1. step_group (VCMS forward pass, punishment if active)
      2. threshold / sustainability exclusion
      3. voluntary exit evaluation + pool regrouping

    In combinations with voluntary exit, exclusion sends agents to pool (not replaced).
    Without voluntary exit, exclusion replaces agents (Phase 1 behavior).
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
    g_budget_hist = defaultdict(list)     # sustainability
    g_coop_hist = defaultdict(list)
    degrading = defaultdict(int)
    free_pool = []                        # voluntary
    next_gid = max(groups.keys()) + 1

    if mechanisms is None:
        mechanisms = [('none', {})]
    if mechanisms_after is None:
        mechanisms_after = mechanisms

    for rnd in range(n_rounds):
        active = mechanisms_after if (switch_round is not None and rnd >= switch_round) else mechanisms
        mech_names = {m[0] for m in active}
        has_pun = 'punishment' in mech_names
        has_vol = 'voluntary' in mech_names

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
            record_round(members, contribs, ps, pr)
            round_contribs[gid] = contribs
            prev_c[gid] = contribs
            prev_p[gid] = pr if has_pun else None

            # Global group-level tracking
            g_budget_hist[gid].append(
                sum(ag.state.B for ag in members) / max(len(members), 1))
            g_coop_hist[gid].append(
                sum(contribs) / max(len(contribs), 1))

        # --- Apply mechanisms in order ---
        for mech_name, mech_kwargs in active:

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
                        if has_vol:
                            members.pop(j)
                            old_ag.group_id = -1
                            free_pool.append(old_ag)
                        else:
                            repl = create_replacement(
                                old_ag.phenotype, pools, rng, uid_counter,
                                active_from=rnd + 1)
                            uid_counter += 1
                            repl.group_id = gid
                            members[j] = repl
                            all_agents.append(repl)
                            below_count[repl.uid] = 0
                        events.append({
                            'type': 'threshold_removal', 'round': rnd,
                            'group': gid, 'removed_uid': old_ag.uid,
                            'removed_phenotype': old_ag.phenotype,
                        })
                    if removals and has_vol:
                        prev_c[gid] = None

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
                            if has_vol:
                                members.pop(worst_j)
                                old_ag.group_id = -1
                                free_pool.append(old_ag)
                                prev_c[gid] = None
                            else:
                                repl = create_replacement(
                                    old_ag.phenotype, pools, rng, uid_counter,
                                    active_from=rnd + 1)
                                uid_counter += 1
                                repl.group_id = gid
                                members[worst_j] = repl
                                all_agents.append(repl)
                            degrading[gid] = 0
                            events.append({
                                'type': 'sustainability_removal', 'round': rnd,
                                'group': gid, 'removed_uid': old_ag.uid,
                                'removed_phenotype': old_ag.phenotype,
                                'impact': best_imp,
                            })

            elif mech_name == 'voluntary':
                ef = mech_kwargs.get('eval_freq', 10)
                fm = mech_kwargs.get('formation', 'random')
                lt = mech_kwargs.get('leave_threshold', LEAVE_THRESHOLD)

                if rnd > 0 and ef > 0 and rnd % ef == 0:
                    sustain_fired_groups = {e['group'] for e in events
                                            if e['round'] == rnd and
                                            e['type'] == 'sustainability_removal'}

                    for gid in list(groups.keys()):
                        members = groups[gid]
                        if not members:
                            continue
                        leavers = []
                        for j, ag in enumerate(members):
                            afford = getattr(ag.state, '_afford', 0.0)
                            agent_thresh = lt * (1.0 - 0.5 * abs(ag.params.inertia))
                            if afford < agent_thresh:
                                leavers.append(j)

                        for j in sorted(leavers, reverse=True):
                            ag = members.pop(j)
                            ag.group_id = -1
                            free_pool.append(ag)
                            events.append({
                                'type': 'voluntary_exit', 'round': rnd,
                                'from_group': gid, 'agent_uid': ag.uid,
                                'phenotype': ag.phenotype,
                                'redundant': gid in sustain_fired_groups,
                            })

                        if leavers:
                            prev_c[gid] = None
                        if not members:
                            del groups[gid]
                            prev_c.pop(gid, None)

                    # Form new groups from pool
                    if fm == 'sorted':
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
                        prev_c[next_gid] = None
                        events.append({
                            'type': 'group_formed', 'round': rnd,
                            'group': next_gid,
                            'members': [ag.uid for ag in new_members],
                        })
                        next_gid += 1

        # Clean up empty groups
        for gid in list(groups.keys()):
            if not groups[gid]:
                del groups[gid]
                prev_c.pop(gid, None)

    for ag in free_pool:
        if ag.active_to == -1:
            ag.active_to = n_rounds - 1

    return {'agents': all_agents, 'groups': groups, 'events': events,
            'free_pool': free_pool}


# ================================================================
# PHASE 2 METRICS
# ================================================================

def compute_phase2_metrics(result, n_rounds=100, transition_round=None):
    """Compute Phase 1 base metrics plus transition, combination, and legibility metrics."""
    agents = result['agents']
    events = result['events']
    base = compute_run_metrics(result, n_rounds)

    active_agents = [ag for ag in agents if ag.active_to == -1]

    # Extra checkpoints for 200-round sims
    if n_rounds > 100:
        for cp in [110, 125, 150, 175, 200]:
            rnd_idx = min(cp - 1, n_rounds - 1)
            vals = []
            for ag in agents:
                if ag.active_from <= rnd_idx and (ag.active_to == -1 or ag.active_to >= rnd_idx):
                    local_idx = rnd_idx - ag.active_from
                    if 0 <= local_idx < len(ag.contrib_history):
                        vals.append(ag.contrib_history[local_idx])
            base[f'mean_coop_r{cp}'] = sum(vals) / max(len(vals), 1)

    # --- Transition metrics ---
    if transition_round is not None and transition_round < n_rounds:
        tr = transition_round

        # Damage profile at switch point
        for ptype in ['CC', 'EC', 'CD', 'DL', 'MX']:
            ptype_agents = [ag for ag in agents
                            if ag.phenotype == ptype and ag.active_from <= tr - 1
                            and (ag.active_to == -1 or ag.active_to >= tr - 1)]
            if ptype_agents:
                budgets = [ag.budget_history[tr - 1 - ag.active_from]
                           for ag in ptype_agents
                           if tr - 1 - ag.active_from < len(ag.budget_history)]
                strains = [ag.strain_history[tr - 1 - ag.active_from]
                           for ag in ptype_agents
                           if tr - 1 - ag.active_from < len(ag.strain_history)]
                contribs = [ag.contrib_history[tr - 1 - ag.active_from]
                            for ag in ptype_agents
                            if tr - 1 - ag.active_from < len(ag.contrib_history)]
                base[f'{ptype}_budget_at_switch'] = sum(budgets) / max(len(budgets), 1)
                base[f'{ptype}_strain_at_switch'] = sum(strains) / max(len(strains), 1)
                base[f'{ptype}_coop_at_switch'] = sum(contribs) / max(len(contribs), 1)
            else:
                base[f'{ptype}_budget_at_switch'] = 0.0
                base[f'{ptype}_strain_at_switch'] = 0.0
                base[f'{ptype}_coop_at_switch'] = 0.0

        # Cooperation dip at switch
        vals_pre = []
        vals_post = []
        for ag in agents:
            if ag.active_from <= tr and (ag.active_to == -1 or ag.active_to >= tr):
                idx_pre = tr - 1 - ag.active_from
                idx_post = tr - ag.active_from
                if 0 <= idx_pre < len(ag.contrib_history):
                    vals_pre.append(ag.contrib_history[idx_pre])
                if 0 <= idx_post < len(ag.contrib_history):
                    vals_post.append(ag.contrib_history[idx_post])
        pre_mean = sum(vals_pre) / max(len(vals_pre), 1)
        post_mean = sum(vals_post) / max(len(vals_post), 1)
        base['coop_dip_at_switch'] = post_mean - pre_mean

        # Recovery slopes (first 20 post-switch rounds)
        recovery_window = min(20, n_rounds - tr)
        coop_traj = []
        budget_traj = []
        for r_offset in range(recovery_window):
            r = tr + r_offset
            c_vals, b_vals = [], []
            for ag in agents:
                if ag.active_from <= r and (ag.active_to == -1 or ag.active_to >= r):
                    idx = r - ag.active_from
                    if 0 <= idx < len(ag.contrib_history):
                        c_vals.append(ag.contrib_history[idx])
                    if 0 <= idx < len(ag.budget_history):
                        b_vals.append(ag.budget_history[idx])
            coop_traj.append(sum(c_vals) / max(len(c_vals), 1))
            budget_traj.append(sum(b_vals) / max(len(b_vals), 1))

        base['coop_recovery_slope'] = _linear_slope(coop_traj)
        base['budget_recovery_slope'] = _linear_slope(budget_traj)

        # Per-phenotype budget recovery slope
        for ptype in ['CC', 'EC', 'CD', 'DL']:
            ptraj = []
            for r_offset in range(recovery_window):
                r = tr + r_offset
                vals = []
                for ag in agents:
                    if (ag.phenotype == ptype and ag.active_from <= r
                            and (ag.active_to == -1 or ag.active_to >= r)):
                        idx = r - ag.active_from
                        if 0 <= idx < len(ag.budget_history):
                            vals.append(ag.budget_history[idx])
                ptraj.append(sum(vals) / max(len(vals), 1))
            base[f'{ptype}_budget_recovery_slope'] = _linear_slope(ptraj)

        # Time to stabilization
        base['time_to_stabilization'] = n_rounds  # default: didn't stabilize
        for r in range(tr + 10, n_rounds):
            window_start = r - 10
            window = []
            for wr in range(window_start, r):
                c_vals = []
                for ag in agents:
                    if ag.active_from <= wr and (ag.active_to == -1 or ag.active_to >= wr):
                        idx = wr - ag.active_from
                        if 0 <= idx < len(ag.contrib_history):
                            c_vals.append(ag.contrib_history[idx])
                window.append(sum(c_vals) / max(len(c_vals), 1))
            if abs(_linear_slope(window)) < 0.05:
                base['time_to_stabilization'] = r
                break

        # Steady-state cooperation (last 20 rounds)
        ss_vals = []
        for r in range(max(tr, n_rounds - 20), n_rounds):
            c_vals = []
            for ag in active_agents:
                idx = r - ag.active_from
                if 0 <= idx < len(ag.contrib_history):
                    c_vals.append(ag.contrib_history[idx])
            ss_vals.append(sum(c_vals) / max(len(c_vals), 1))
        base['post_switch_steady_state'] = sum(ss_vals) / max(len(ss_vals), 1)

    # --- Combination metrics ---
    sustain_removals = [e for e in events if e['type'] == 'sustainability_removal']
    vol_exits = [e for e in events if e['type'] == 'voluntary_exit']
    base['sustain_removals'] = len(sustain_removals)
    base['voluntary_exits'] = len(vol_exits)
    base['total_mobility'] = len(sustain_removals) + len(vol_exits)
    # Redundancy: voluntary exits in groups where sustainability already fired
    redundant = sum(1 for e in vol_exits if e.get('redundant', False))
    base['redundancy_count'] = redundant
    base['redundancy_rate'] = redundant / max(len(vol_exits), 1)

    # --- Legibility metrics ---
    legibilities = []
    ruptured_legibility = []
    survived_legibility = []
    for ag in agents:
        leg = _legibility(ag)
        legibilities.append(leg)
        if _is_ruptured(ag):
            ruptured_legibility.append(leg)
        elif ag.active_to == -1:
            survived_legibility.append(leg)

    base['mean_legibility'] = sum(legibilities) / max(len(legibilities), 1)
    base['ruptured_legibility'] = (sum(ruptured_legibility) / max(len(ruptured_legibility), 1)
                                   if ruptured_legibility else 0.0)
    base['survived_legibility'] = (sum(survived_legibility) / max(len(survived_legibility), 1)
                                   if survived_legibility else 0.0)
    base['legibility_gap'] = base['ruptured_legibility'] - base['survived_legibility']

    # Environmental volatility: mean abs round-to-round cooperation change
    all_deltas = []
    for ag in active_agents:
        for i in range(1, len(ag.contrib_history)):
            all_deltas.append(abs(ag.contrib_history[i] - ag.contrib_history[i - 1]))
    base['env_volatility'] = sum(all_deltas) / max(len(all_deltas), 1)

    # EC-specific metrics
    ec_agents = [ag for ag in active_agents if ag.phenotype == 'EC']
    cc_agents = [ag for ag in active_agents if ag.phenotype == 'CC']
    if ec_agents and cc_agents:
        ec_budget = sum(ag.budget_history[-1] for ag in ec_agents if ag.budget_history) / len(ec_agents)
        cc_budget = sum(ag.budget_history[-1] for ag in cc_agents if ag.budget_history) / len(cc_agents)
        base['transparency_tax'] = cc_budget - ec_budget
    else:
        base['transparency_tax'] = 0.0

    return base


def _legibility(agent):
    """Pearson correlation between contribution and budget trajectories."""
    cx = agent.contrib_history
    cy = agent.budget_history
    n = min(len(cx), len(cy))
    if n < 3:
        return 0.0
    cx, cy = cx[:n], cy[:n]
    mx = sum(cx) / n
    my = sum(cy) / n
    cov = sum((cx[i] - mx) * (cy[i] - my) for i in range(n)) / n
    sx = (sum((x - mx) ** 2 for x in cx) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in cy) / n) ** 0.5
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return cov / (sx * sy)


def _is_ruptured(agent):
    """Check if agent is ruptured (budget < 10% of initial for 3+ consecutive rounds at end)."""
    if len(agent.budget_history) < RUPTURE_CONSEC:
        return False
    b_init = agent.params.b_initial
    threshold = RUPTURE_B_FRAC * b_init
    tail = agent.budget_history[-RUPTURE_CONSEC:]
    return all(b < threshold for b in tail)


# ================================================================
# TEST 1: TRANSITION DYNAMICS
# ================================================================

TRANSITION_CONDITIONS = {
    'T1_pun→none':   (('punishment', {}),     ('none', {})),
    'T2_pun→sustain': (('punishment', {}),     ('sustainability', {})),
    'T3_pun→vol':    (('punishment', {}),     ('voluntary', {'eval_freq': 10})),
    'T4_pun→thresh': (('punishment', {}),     ('threshold', {'K': 3})),
    'T5_none→sustain': (('none', {}),          ('sustainability', {})),
    'T6_sustain_full': (('sustainability', {}), ('sustainability', {})),
    'T7_pun_full':   (('punishment', {}),     ('punishment', {})),
}


def run_test1(pools, n_runs=N_RUNS, n_rounds=200, switch_round=100, seed=42):
    """Run all transition conditions."""
    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in TRANSITION_CONDITIONS}

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"    Test 1: run {run + 1}/{n_runs}  ({time.time() - t0:.1f}s)")
        bp = sample_population_blueprint(pools, rng)

        for cond_name, (mech_before, mech_after) in TRANSITION_CONDITIONS.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2 ** 31))
            result = simulate(
                agents, pools, cond_rng, n_rounds,
                mechanisms=[mech_before],
                switch_round=switch_round,
                mechanisms_after=[mech_after],
            )
            metrics = compute_phase2_metrics(result, n_rounds, transition_round=switch_round)
            all_metrics[cond_name].append(metrics)

    print(f"    Test 1 complete: {time.time() - t0:.1f}s")
    return {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}


# ================================================================
# TEST 2: MECHANISM COMBINATION
# ================================================================

COMBINATION_CONDITIONS = {
    'C1_sustain_only':     [('sustainability', {})],
    'C2_vol_r10_only':     [('voluntary', {'eval_freq': 10})],
    'C3_sustain+vol_r10':  [('sustainability', {}), ('voluntary', {'eval_freq': 10})],
    'C4_sustain+vol_r5':   [('sustainability', {}), ('voluntary', {'eval_freq': 5})],
    'C5_thresh+vol_r10':   [('threshold', {'K': 3}), ('voluntary', {'eval_freq': 10})],
    'C6_pun+vol_r10':      [('punishment', {}), ('voluntary', {'eval_freq': 10})],
}


def run_test2(pools, n_runs=N_RUNS, n_rounds=100, seed=43):
    """Run all combination conditions."""
    rng = np.random.default_rng(seed)
    all_metrics = {name: [] for name in COMBINATION_CONDITIONS}

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"    Test 2: run {run + 1}/{n_runs}  ({time.time() - t0:.1f}s)")
        bp = sample_population_blueprint(pools, rng)

        for cond_name, mechs in COMBINATION_CONDITIONS.items():
            agents = instantiate_population(bp)
            cond_rng = np.random.default_rng(rng.integers(2 ** 31))
            result = simulate(agents, pools, cond_rng, n_rounds, mechanisms=mechs)
            metrics = compute_phase2_metrics(result, n_rounds)
            all_metrics[cond_name].append(metrics)

    print(f"    Test 2 complete: {time.time() - t0:.1f}s")
    return {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}


# ================================================================
# TEST 3: EC PROTECTION
# ================================================================

def run_test3b(pools, n_runs=N_RUNS, n_rounds=100, seed=44):
    """Test 3B: Inertia sensitivity — 3 EC inertia variants × 2 conditions."""
    rng = np.random.default_rng(seed)
    variants = {'EC_original': None, 'EC_medium': 0.35, 'EC_high': 0.50}
    conditions = {
        'punishment': [('punishment', {})],
        'sustainability': [('sustainability', {})],
    }

    all_metrics = {}
    for var_name, target_inertia in variants.items():
        for cond_name, mechs in conditions.items():
            key = f'{var_name}_{cond_name}'
            all_metrics[key] = []

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"    Test 3B: run {run + 1}/{n_runs}  ({time.time() - t0:.1f}s)")
        bp = sample_population_blueprint(pools, rng)

        for var_name, target_inertia in variants.items():
            # Modify EC agents' inertia if needed
            if target_inertia is not None:
                mod_bp = []
                for uid, phen, src, params in bp:
                    p = copy.deepcopy(params)
                    if phen == 'EC':
                        p.inertia = target_inertia
                    mod_bp.append((uid, phen, src, p))
            else:
                mod_bp = bp

            for cond_name, mechs in conditions.items():
                key = f'{var_name}_{cond_name}'
                agents = instantiate_population(mod_bp)
                cond_rng = np.random.default_rng(rng.integers(2 ** 31))
                result = simulate(agents, pools, cond_rng, n_rounds, mechanisms=mechs)
                metrics = compute_phase2_metrics(result, n_rounds)
                all_metrics[key].append(metrics)

    print(f"    Test 3B complete: {time.time() - t0:.1f}s")
    return {name: aggregate_metrics(runs) for name, runs in all_metrics.items()}


def analyze_test3a(all_condition_results):
    """Test 3A: Cross-condition EC survival vs environmental volatility."""
    print("\n" + "=" * 100)
    print("TEST 3A: EC SURVIVAL vs ENVIRONMENTAL VOLATILITY")
    print("=" * 100)
    header = f"  {'Condition':<30} {'EC Surv':>8} {'EC Budget':>9} {'Env Vol':>8} {'Leg Gap':>8}"
    print(header)
    print(f"  {'-' * 80}")

    ec_survivals = []
    volatilities = []
    for name, agg in sorted(all_condition_results.items()):
        ec_surv = agg.get('EC_survival', {}).get('median', 1.0)
        ec_bud = agg.get('EC_budget_T100', {}).get('median', 0.0)
        vol = agg.get('env_volatility', {}).get('median', 0.0)
        leg_gap = agg.get('legibility_gap', {}).get('median', 0.0)
        ec_survivals.append(ec_surv)
        volatilities.append(vol)
        print(f"  {name:<30} {ec_surv:>8.1%} {ec_bud:>9.2f} {vol:>8.2f} {leg_gap:>+8.3f}")

    # Correlation
    if len(ec_survivals) > 2:
        corr = _pearson(ec_survivals, volatilities)
        print(f"\n  EC survival vs volatility correlation: {corr:+.3f}")
        print(f"  {'SUPPORTED' if corr < -0.3 else 'NOT SUPPORTED'}: "
              f"inverse correlation (predicted < -0.3)")


def analyze_test3c(all_condition_results):
    """Test 3C: Canary index — legibility of ruptured vs survived agents."""
    print("\n" + "=" * 100)
    print("TEST 3C: CANARY INDEX — LEGIBILITY AND RUPTURE")
    print("=" * 100)
    header = f"  {'Condition':<30} {'Rupt Leg':>9} {'Surv Leg':>9} {'Gap':>8} {'Tax':>8}"
    print(header)
    print(f"  {'-' * 80}")

    for name, agg in sorted(all_condition_results.items()):
        rl = agg.get('ruptured_legibility', {}).get('median', 0.0)
        sl = agg.get('survived_legibility', {}).get('median', 0.0)
        gap = agg.get('legibility_gap', {}).get('median', 0.0)
        tax = agg.get('transparency_tax', {}).get('median', 0.0)
        print(f"  {name:<30} {rl:>9.3f} {sl:>9.3f} {gap:>+8.3f} {tax:>+8.3f}")


def _pearson(x, y):
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return cov / (sx * sy)


# ================================================================
# REPORTING
# ================================================================

def print_test1_results(results):
    """Print transition dynamics results."""
    print("\n" + "=" * 100)
    print("TEST 1: TRANSITION DYNAMICS (200 rounds, switch at round 100)")
    print("=" * 100)

    # Condition comparison
    header = (f"{'Condition':<22} {'Coop@100':>8} {'Coop@200':>8} {'Dip@Sw':>7} "
              f"{'RecSlope':>8} {'BudSlope':>8} {'Stabil':>7} {'SS-Post':>7} {'TTFR':>5}")
    print(header)
    print("-" * 100)

    for name, agg in sorted(results.items()):
        c100 = agg.get('mean_coop_r100', {}).get('median', 0)
        c200 = agg.get('mean_coop_r200', {}).get('median', 0)
        dip = agg.get('coop_dip_at_switch', {}).get('median', 0)
        rec_sl = agg.get('coop_recovery_slope', {}).get('median', 0)
        bud_sl = agg.get('budget_recovery_slope', {}).get('median', 0)
        stab = agg.get('time_to_stabilization', {}).get('median', 0)
        ss = agg.get('post_switch_steady_state', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        print(f"{name:<22} {c100:>8.1f} {c200:>8.1f} {dip:>+7.2f} "
              f"{rec_sl:>+8.3f} {bud_sl:>+8.4f} {stab:>7.0f} {ss:>7.1f} {ttfr:>5.0f}")

    # Cooperation trajectory at all checkpoints
    print(f"\n  Cooperation Trajectory (median):")
    cps = [10, 25, 50, 75, 100, 110, 125, 150, 175, 200]
    header2 = f"  {'Condition':<22}" + "".join(f"{'r=' + str(cp):>7}" for cp in cps)
    print(header2)
    print(f"  {'-' * 90}")
    for name, agg in sorted(results.items()):
        vals = [f"{agg.get(f'mean_coop_r{cp}', {}).get('median', 0):>7.1f}" for cp in cps]
        print(f"  {name:<22}{''.join(vals)}")

    # Damage profile at switch
    print(f"\n  Damage Profile at Round 100 (median):")
    header3 = f"  {'Condition':<22} {'CC Bud':>7} {'EC Bud':>7} {'CD Bud':>7} {'DL Bud':>7}"
    print(header3)
    print(f"  {'-' * 60}")
    for name, agg in sorted(results.items()):
        vals = []
        for pt in ['CC', 'EC', 'CD', 'DL']:
            v = agg.get(f'{pt}_budget_at_switch', {}).get('median', 0)
            vals.append(f"{v:>7.2f}")
        print(f"  {name:<22}{''.join(vals)}")

    # Per-phenotype recovery slopes
    print(f"\n  Budget Recovery Slopes by Phenotype (first 20 post-switch rounds):")
    header4 = f"  {'Condition':<22} {'CC':>8} {'EC':>8} {'CD':>8} {'DL':>8}"
    print(header4)
    print(f"  {'-' * 60}")
    for name, agg in sorted(results.items()):
        vals = []
        for pt in ['CC', 'EC', 'CD', 'DL']:
            v = agg.get(f'{pt}_budget_recovery_slope', {}).get('median', 0)
            vals.append(f"{v:>+8.4f}")
        print(f"  {name:<22}{''.join(vals)}")


def print_test2_results(results):
    """Print combination results."""
    print("\n" + "=" * 100)
    print("TEST 2: MECHANISM COMBINATION (100 rounds)")
    print("=" * 100)

    header = (f"{'Condition':<24} {'SS-Coop':>8} {'TTFR':>5} {'Rupt':>5} "
              f"{'Gini':>6} {'Bimod':>6} {'SustRm':>6} {'VolEx':>6} "
              f"{'TotMob':>6} {'Redund':>6}")
    print(header)
    print("-" * 100)

    for name, agg in sorted(results.items()):
        ss = agg.get('steady_state_coop', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        rupt = agg.get('rupture_count', {}).get('median', 0)
        gini = agg.get('gini', {}).get('median', 0)
        bimod = agg.get('bimodality', {}).get('median', 0)
        sr = agg.get('sustain_removals', {}).get('median', 0)
        ve = agg.get('voluntary_exits', {}).get('median', 0)
        tm = agg.get('total_mobility', {}).get('median', 0)
        rr = agg.get('redundancy_rate', {}).get('median', 0)
        print(f"{name:<24} {ss:>8.1f} {ttfr:>5.0f} {rupt:>5.0f} "
              f"{gini:>6.3f} {bimod:>6.3f} {sr:>6.0f} {ve:>6.0f} "
              f"{tm:>6.0f} {rr:>6.1%}")

    # Per-phenotype outcomes
    print(f"\n  Per-Phenotype Budget at T=100 (median):")
    header2 = f"  {'Condition':<24} {'CC':>8} {'EC':>8} {'CD':>8} {'DL':>8}"
    print(header2)
    print(f"  {'-' * 60}")
    for name, agg in sorted(results.items()):
        vals = []
        for pt in ['CC', 'EC', 'CD', 'DL']:
            v = agg.get(f'{pt}_budget_T100', {}).get('median', 0)
            vals.append(f"{v:>8.2f}")
        print(f"  {name:<24}{''.join(vals)}")


def print_test3b_results(results):
    """Print inertia sensitivity results."""
    print("\n" + "=" * 100)
    print("TEST 3B: INERTIA SENSITIVITY — EC VARIANTS UNDER PUNISHMENT vs SUSTAINABILITY")
    print("=" * 100)

    header = (f"{'Variant + Condition':<35} {'EC Surv':>8} {'EC Bud':>8} "
              f"{'EC Coop':>8} {'EC Strain':>9} {'TTFR':>5}")
    print(header)
    print("-" * 100)

    for name, agg in sorted(results.items()):
        ec_surv = agg.get('EC_survival', {}).get('median', 0)
        ec_bud = agg.get('EC_budget_T100', {}).get('median', 0)
        ec_coop = agg.get('EC_coop_T100', {}).get('median', 0)
        ec_strain = agg.get('EC_strain_T100', {}).get('median', 0)
        ttfr = agg.get('system_ttfr', {}).get('median', 0)
        print(f"{name:<35} {ec_surv:>8.1%} {ec_bud:>8.2f} "
              f"{ec_coop:>8.1f} {ec_strain:>9.2f} {ttfr:>5.0f}")


def print_predictions_scorecard(t1, t2, t3b, all_cond):
    """Evaluate all Phase 2 predictions."""
    print("\n" + "=" * 100)
    print("PHASE 2 PREDICTIONS SCORECARD")
    print("=" * 100)

    results = []

    # --- Test 1 Predictions ---
    results.append("\n  TEST 1: TRANSITION")

    # P1-T: Cooperation dips at switch for T1-T4
    dips = {}
    for name in ['T1_pun→none', 'T2_pun→sustain', 'T3_pun→vol', 'T4_pun→thresh']:
        if name in t1:
            dips[name] = t1[name].get('coop_dip_at_switch', {}).get('median', 0)
    all_negative = all(d < 0 for d in dips.values()) if dips else False
    results.append(f"  P1-T: Coop dips at switch: {dips} → {'SUPPORTED' if all_negative else 'NOT SUPPORTED'}")

    # P2-T: Budget recovery before cooperation recovery
    budget_first_count = 0
    for name in ['T2_pun→sustain', 'T3_pun→vol', 'T4_pun→thresh']:
        if name in t1:
            bsl = t1[name].get('budget_recovery_slope', {}).get('median', 0)
            csl = t1[name].get('coop_recovery_slope', {}).get('median', 0)
            if bsl > csl:
                budget_first_count += 1
    results.append(f"  P2-T: Budget recovers before coop: {budget_first_count}/3 conditions → "
                   f"{'SUPPORTED' if budget_first_count >= 2 else 'NOT SUPPORTED'}")

    # P3-T: Hysteresis — T2 < T6 but T2 > T7
    if 'T2_pun→sustain' in t1 and 'T6_sustain_full' in t1 and 'T7_pun_full' in t1:
        t2_ss = t1['T2_pun→sustain'].get('post_switch_steady_state', {}).get('median', 0)
        t6_ss = t1['T6_sustain_full'].get('post_switch_steady_state', {}).get('median', 0)
        t7_ss = t1['T7_pun_full'].get('post_switch_steady_state', {}).get('median', 0)
        hysteresis = t2_ss < t6_ss and t2_ss > t7_ss
        results.append(f"  P3-T: Hysteresis: T2={t2_ss:.1f} < T6={t6_ss:.1f} and > T7={t7_ss:.1f} → "
                       f"{'SUPPORTED' if hysteresis else 'NOT SUPPORTED'}")
        results.append(f"         Hysteresis gap: {t6_ss - t2_ss:.2f} (cost of delayed reform)")

    # P4-T: DL agents recover slowest
    if 'T2_pun→sustain' in t1:
        slopes = {}
        for pt in ['CC', 'EC', 'CD', 'DL']:
            slopes[pt] = t1['T2_pun→sustain'].get(
                f'{pt}_budget_recovery_slope', {}).get('median', 0)
        slowest = min(slopes, key=slopes.get)
        results.append(f"  P4-T: DL recovers slowest: slopes={slopes} → "
                       f"{'SUPPORTED' if slowest == 'DL' else 'NOT SUPPORTED'} (slowest: {slowest})")

    # P5-T: pun→vol is slowest recovery
    recovery_speeds = {}
    for name in ['T1_pun→none', 'T2_pun→sustain', 'T3_pun→vol', 'T4_pun→thresh']:
        if name in t1:
            recovery_speeds[name] = t1[name].get('time_to_stabilization', {}).get('median', 200)
    if recovery_speeds:
        slowest_recovery = max(recovery_speeds, key=recovery_speeds.get)
        results.append(f"  P5-T: pun→vol slowest recovery: {recovery_speeds} → "
                       f"{'SUPPORTED' if slowest_recovery == 'T3_pun→vol' else 'NOT SUPPORTED'}")

    # --- Test 2 Predictions ---
    results.append("\n  TEST 2: COMBINATION")

    # P1-C: Combined outperforms either alone
    def _composite(agg):
        ss = agg.get('steady_state_coop', {}).get('median', 0) / MAX_CONTRIB
        ttfr = min(agg.get('system_ttfr', {}).get('median', 0), 100) / 100
        rupt = 1.0 - min(agg.get('rupture_count', {}).get('median', 0), N_AGENTS) / N_AGENTS
        return (ss + ttfr + rupt) / 3.0

    composites = {name: _composite(agg) for name, agg in t2.items()}
    c3_score = composites.get('C3_sustain+vol_r10', 0)
    c1_score = composites.get('C1_sustain_only', 0)
    c2_score = composites.get('C2_vol_r10_only', 0)
    combined_better = c3_score > max(c1_score, c2_score)
    results.append(f"  P1-C: Combined > either alone: C3={c3_score:.3f} vs C1={c1_score:.3f}, C2={c2_score:.3f} → "
                   f"{'SUPPORTED' if combined_better else 'NOT SUPPORTED'}")

    # P2-C: Total mobility < sum of individual
    if 'C1_sustain_only' in t2 and 'C2_vol_r10_only' in t2 and 'C3_sustain+vol_r10' in t2:
        c1_mob = t2['C1_sustain_only'].get('total_mobility', {}).get('median', 0)
        c2_mob = t2['C2_vol_r10_only'].get('total_mobility', {}).get('median', 0)
        c3_mob = t2['C3_sustain+vol_r10'].get('total_mobility', {}).get('median', 0)
        less_mob = c3_mob < (c1_mob + c2_mob)
        results.append(f"  P2-C: Mobility reduction: C3={c3_mob:.0f} < C1+C2={c1_mob + c2_mob:.0f} → "
                       f"{'SUPPORTED' if less_mob else 'NOT SUPPORTED'}")

    # P3-C: r5 achieves bimodality > 0.555
    if 'C4_sustain+vol_r5' in t2:
        bimod = t2['C4_sustain+vol_r5'].get('bimodality', {}).get('median', 0)
        results.append(f"  P3-C: r5 bimodality: BC={bimod:.3f} → "
                       f"{'SUPPORTED' if bimod > 0.555 else 'NOT SUPPORTED'} (>0.555)")

    # P4-C: pun+vol has highest exit rate
    exit_rates = {}
    for name, agg in t2.items():
        ve = agg.get('voluntary_exits', {}).get('median', 0)
        exit_rates[name] = ve
    if exit_rates:
        highest_exit = max(exit_rates, key=exit_rates.get)
        results.append(f"  P4-C: pun+vol highest exits: {exit_rates} → "
                       f"{'SUPPORTED' if highest_exit == 'C6_pun+vol_r10' else 'NOT SUPPORTED'}")

    # --- Test 3 Predictions ---
    results.append("\n  TEST 3: EC PROTECTION")

    # P2-E: Inertia mediates punishment vulnerability
    if t3b:
        orig_pun = t3b.get('EC_original_punishment', {}).get('EC_survival', {}).get('median', 1)
        med_pun = t3b.get('EC_medium_punishment', {}).get('EC_survival', {}).get('median', 1)
        high_pun = t3b.get('EC_high_punishment', {}).get('EC_survival', {}).get('median', 1)
        inertia_protects = high_pun > orig_pun
        results.append(f"  P2-E: Inertia protects ECs under punishment: "
                       f"orig={orig_pun:.1%} med={med_pun:.1%} high={high_pun:.1%} → "
                       f"{'SUPPORTED' if inertia_protects else 'NOT SUPPORTED'}")

    # P3-E: Control-first kills legible agents (legibility gap > 0 under punishment)
    control_first_names = [n for n in all_cond if 'pun' in n.lower() or 'thresh' in n.lower()]
    care_first_names = [n for n in all_cond if 'sustain' in n.lower() or 'vol' in n.lower()]
    cf_gaps = [all_cond[n].get('legibility_gap', {}).get('median', 0) for n in control_first_names if n in all_cond]
    cf_care_gaps = [all_cond[n].get('legibility_gap', {}).get('median', 0) for n in care_first_names if n in all_cond]
    if cf_gaps and cf_care_gaps:
        mean_cf = sum(cf_gaps) / len(cf_gaps)
        mean_care = sum(cf_care_gaps) / len(cf_care_gaps)
        results.append(f"  P3-E: Control-first legibility gap: {mean_cf:+.3f} vs care-first: {mean_care:+.3f} → "
                       f"{'SUPPORTED' if mean_cf > mean_care else 'NOT SUPPORTED'}")

    # P4-E: Transparency tax under punishment > under sustainability
    pun_tax = all_cond.get('T7_pun_full', all_cond.get('C6_pun+vol_r10', {}))
    sus_tax = all_cond.get('T6_sustain_full', all_cond.get('C1_sustain_only', {}))
    if pun_tax and sus_tax:
        pt = pun_tax.get('transparency_tax', {}).get('median', 0)
        st = sus_tax.get('transparency_tax', {}).get('median', 0)
        results.append(f"  P4-E: Transparency tax: punishment={pt:+.3f} vs sustainability={st:+.3f} → "
                       f"{'SUPPORTED' if pt > st else 'NOT SUPPORTED'}")

    for r in results:
        print(r)


# ================================================================
# MAIN RUNNER
# ================================================================

def run_phase2(n_runs=N_RUNS, seed=42):
    """Run all Phase 2 tests."""
    print("=" * 100)
    print("FEDERATION DYNAMICS PHASE 2")
    print("Transition, Combination, and EC Protection")
    print("=" * 100)

    print("\nLoading libraries and building pools...")
    libs = load_libraries()
    pools = build_enforcement_pools(libs)
    for name, pool in pools.items():
        print(f"  {name}: {len(pool)} subjects")

    t_total = time.time()

    # Test 1: Transition dynamics
    print("\n--- TEST 1: Transition Dynamics (7 conditions × 200 rounds) ---")
    t1_results = run_test1(pools, n_runs, n_rounds=200, switch_round=100, seed=seed)

    # Test 2: Mechanism combination
    print("\n--- TEST 2: Mechanism Combination (6 conditions × 100 rounds) ---")
    t2_results = run_test2(pools, n_runs, n_rounds=100, seed=seed + 1)

    # Test 3B: EC inertia sensitivity
    print("\n--- TEST 3B: EC Inertia Sensitivity (6 conditions × 100 rounds) ---")
    t3b_results = run_test3b(pools, n_runs, n_rounds=100, seed=seed + 2)

    elapsed = time.time() - t_total
    print(f"\nAll simulations complete in {elapsed:.1f}s")

    # Reporting
    print_test1_results(t1_results)
    print_test2_results(t2_results)
    print_test3b_results(t3b_results)

    # Collect all condition results for cross-condition EC analysis
    all_cond = {}
    all_cond.update(t1_results)
    all_cond.update(t2_results)
    all_cond.update(t3b_results)

    analyze_test3a(all_cond)
    analyze_test3c(all_cond)

    print_predictions_scorecard(t1_results, t2_results, t3b_results, all_cond)

    return {'test1': t1_results, 'test2': t2_results, 'test3b': t3b_results}


if __name__ == '__main__':
    run_phase2()
