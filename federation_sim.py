#!/usr/bin/env python3
"""
Federation Sustainability Simulation
======================================

Multi-agent coupled PGG simulation: 4 VCMS agents interact through
endogenous group mean signals over 50 rounds, no punishment.

Agents are drawn from the 576-subject fitted library (P + N + IPD)
using phenotype-based parameter criteria. Each round, agents compute
contributions via inline VCMS v4 dynamics, and the group mean of the
OTHER 3 agents' contributions becomes each agent's others_mean input
for the next round.

Self-play (no teacher forcing): agents' predicted contributions ARE
their actual contributions. Defection lowers group mean → cooperators'
experience becomes negative → budget drains → cooperation drops further.

Phenotypes:
  CC (Committed Cooperator): high c_base, high inertia — persist through momentum
  EC (Evaluative Cooperator): moderate c_base, low inertia, high alpha — track environment
  CD (Comfortable Defector): low c_base — individually stable, systemically extractive
  DL (Decliner): high c_base, structural budget drain — cooperate then decline

Key comparison: 3CC+1CD vs 3EC+1CD — does evaluative cooperation
produce longer time-to-rupture than committed cooperation under
extraction pressure?

Theory predictions (falsifiable):
  1. CC groups are fragile to extraction (inertia delays then worsens collapse)
  2. EC groups degrade gracefully (earlier but gentler decline, longer TTFR)
  3. CD never rupture but shorten federation TTFR nonlinearly
  4. Trajectory shape: CC → collapse events; EC → declining events
"""

import json
import math
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ================================================================
# CONSTANTS
# ================================================================

N_ROUNDS = 100         # Simulation horizon (extended from 50 for cascade visibility)
N_RUNS = 100           # Monte Carlo runs per composition
N_AGENTS = 4           # Agents per group
MAX_CONTRIB = 20       # PGG max contribution

# From vcms_engine_v4.py
ANCHOR_RATE = 0.15     # Disposition EMA rate (~7 round half-life)
ACUTE_MULT = 5.0       # Acute event amplifier
EPS = 0.01             # Numerical stability floor

# Rupture detection
RUPTURE_B_FRAC = 0.1   # Budget below 10% of initial
RUPTURE_CONSEC = 3     # Must persist for 3 consecutive rounds
STEADY_STATE_WINDOW = 10  # Last 10 rounds for steady-state metrics

# Cooperation collapse detection (distinct from budget rupture)
COOP_COLLAPSE_THRESH = 2  # Group mean contribution below this
COOP_COLLAPSE_CONSEC = 5  # Must persist for 5 consecutive rounds

# Phenotype label sets (from phenotype_geometry.py)
P_HIGH = {'cooperator', 'cooperative-enforcer'}
P_LOW = {'free-rider', 'punitive-free-rider', 'antisocial-controller'}
N_HIGH = {'stable-high'}
N_LOW = {'stable-low'}
IPD_HIGH = {'mostly-C'}
IPD_LOW = {'mostly-D'}


# ================================================================
# AGENT PARAMETERS AND STATE
# ================================================================

@dataclass
class AgentParams:
    """Frozen parameters for one agent (15 relevant for no-punishment PGG)."""
    alpha: float
    v_rep: float
    v_ref: float
    c_base: float
    inertia: float   # pre-clamped to [-0.3, 0.95]
    s_dir: float     # snapped to +1 or -1
    s_rate: float
    s_initial: float
    b_initial: float
    b_depletion_rate: float
    b_replenish_rate: float
    acute_threshold: float
    facilitation_rate: float
    h_strength: float
    h_start: float   # in [0,1] game progress (normalized)


@dataclass
class AgentState:
    """Mutable state for one VCMS agent during federation simulation."""
    v_level: float = 0.0
    disposition: float = 0.0
    strain: float = 0.0
    B: float = 0.0
    m_eval: float = 0.0
    c_prev_norm: float = 0.0


# ================================================================
# LIBRARY LOADING AND PHENOTYPE POOLS
# ================================================================

def extract_agent_params(v3_params: dict) -> AgentParams:
    """Convert a library v3_params dict to AgentParams."""
    p = v3_params
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
        h_strength=p.get('h_strength', 0.0),
        h_start=p.get('h_start', 7.0 / 9.0),
    )


def load_libraries():
    """Load all three v4 normalized-time libraries."""
    libs = {}
    for fname, tag in [('v4_library_fitted.json', 'P'),
                        ('v4_n_library_fitted.json', 'N'),
                        ('v4_ipd_library_fitted.json', 'IPD')]:
        with open(fname) as f:
            libs[tag] = json.load(f)
    return libs


def build_phenotype_pools(libs):
    """
    Build the four phenotype pools from parameter criteria.

    CC: c_base >0.65, inertia >0.3, cooperator-type label
    EC: c_base 0.35-0.75, inertia <0.25, alpha >0.3 (parameter only)
    CD: c_base <0.4, defector-type label
    DL: N:declining, c_base >0.55, b_depletion/b_replenish >1.0
    """
    pools = {'CC': [], 'EC': [], 'CD': [], 'DL': []}

    for tag, lib in libs.items():
        for sid, rec in lib.items():
            p = rec['v3_params']
            label = rec.get('behavioral_profile', rec.get('subject_type', ''))
            ap = extract_agent_params(p)

            # CC: Committed Cooperator
            cc_labels = P_HIGH if tag == 'P' else (N_HIGH if tag == 'N' else IPD_HIGH)
            if label in cc_labels and p['c_base'] > 0.65 and p['inertia'] > 0.3:
                pools['CC'].append((tag, sid, ap))

            # EC: Evaluative Cooperator (parameter-based, cross-library)
            if 0.35 <= p['c_base'] <= 0.75 and p['inertia'] < 0.25 and p['alpha'] > 0.3:
                pools['EC'].append((tag, sid, ap))

            # CD: Comfortable Defector
            cd_labels = P_LOW if tag == 'P' else (N_LOW if tag == 'N' else IPD_LOW)
            if label in cd_labels and p['c_base'] < 0.4:
                pools['CD'].append((tag, sid, ap))

            # DL: Decliner
            if tag == 'N' and label == 'declining' and p['c_base'] > 0.55:
                ratio = p['b_depletion_rate'] / max(p['b_replenish_rate'], 0.001)
                if ratio > 1.0:
                    pools['DL'].append((tag, sid, ap))

    return pools


def sample_group(composition, pools, rng):
    """
    Sample a group of 4 agents for one simulation run.

    composition: {'CC': 3, 'CD': 1}
    Returns list of (phenotype, sid, AgentParams) tuples.
    """
    group = []
    for phenotype, count in composition.items():
        pool = pools[phenotype]
        indices = rng.choice(len(pool), size=count, replace=True)
        for idx in indices:
            tag, sid, params = pool[idx]
            group.append((phenotype, f"{tag}:{sid}", params))
    return group


# ================================================================
# INLINE STEP-WISE 4-AGENT ENGINE
# ================================================================

def simulate_federation(agents, n_rounds=N_ROUNDS):
    """
    Run a 4-agent coupled PGG simulation.

    Each round:
    1. Compute others_mean for each agent (mean of other 3 agents' prev contributions)
    2. Run one VCMS step per agent (inline, no dict allocation)
    3. Record contributions, budgets, strain, affordability
    4. Feed predicted contribution back as actual (self-play)

    Returns dict with per-agent trajectories.
    """
    n = len(agents)
    dt = 1.0 / (n_rounds - 1) if n_rounds > 1 else 1.0
    max_c = MAX_CONTRIB

    # Initialize states and extract params
    states = []
    params_list = []
    h_start_rounds = []

    for phenotype, sid, ap in agents:
        states.append(AgentState(
            strain=ap.s_initial,
            B=ap.b_initial,
        ))
        params_list.append(ap)
        h_start_rounds.append(ap.h_start * (n_rounds - 1))

    # Output storage
    contrib_traj = [[0] * n_rounds for _ in range(n)]
    budget_traj = [[0.0] * n_rounds for _ in range(n)]
    strain_traj = [[0.0] * n_rounds for _ in range(n)]
    afford_traj = [[0.0] * n_rounds for _ in range(n)]

    # Previous contributions for computing others_mean
    prev_contribs = [0] * n

    for rnd in range(n_rounds):
        # Step 1: Compute others_mean for each agent
        if rnd == 0:
            # No prior contributions. Round 0 only sets v_level and disposition;
            # no strain or budget update occurs. Using 0.0 avoids look-ahead.
            others_means = [0.0] * n
        else:
            total = sum(prev_contribs)
            others_means = [(total - prev_contribs[j]) / (n - 1) for j in range(n)]

        # Step 2: Run one VCMS step per agent
        round_contribs = [0] * n

        for j in range(n):
            ap = params_list[j]
            st = states[j]

            v_group_raw = others_means[j] / max_c
            v_group = min(1.0, ap.v_rep * v_group_raw)

            # --- V: Observe, update reference ---
            if rnd == 0:
                st.v_level = v_group
                st.c_prev_norm = ap.c_base
                st.disposition = ap.c_base
            else:
                st.v_level = ap.alpha * v_group + (1.0 - ap.alpha) * st.v_level
                st.disposition = ANCHOR_RATE * st.c_prev_norm + (1.0 - ANCHOR_RATE) * st.disposition

            reference = ap.v_ref * st.v_level + (1.0 - ap.v_ref) * st.disposition

            # --- S: Strain accumulation ---
            if rnd > 0:
                gap = st.c_prev_norm - reference
                gap_strain = max(0.0, gap * ap.s_dir)
                st.strain += dt * ap.s_rate * gap_strain

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
                    st.B += dt * ap.b_replenish_rate * experience
                st.B = max(0.0, st.B)

                # M_eval: facilitation
                st.m_eval += dt * ap.facilitation_rate * experience

            # --- Affordability (gate=0, discharge=0 for no-punishment) ---
            affordability = st.B / (st.B + st.strain + EPS)

            # --- Contribution ---
            if rnd == 0:
                c_norm = ap.c_base
            else:
                c_target = ap.v_ref * st.v_level + (1.0 - ap.v_ref) * ap.c_base
                c_target_adj = max(0.0, min(1.0, c_target + st.m_eval))
                c_norm = (1.0 - abs(ap.inertia)) * c_target_adj + ap.inertia * st.c_prev_norm

            # --- Horizon ---
            h_factor = 1.0
            h_sr = h_start_rounds[j]
            if n_rounds > 1 and ap.h_strength > 0.0 and rnd >= h_sr:
                denom = n_rounds - 1 - h_sr
                if denom > 0:
                    progress = min(1.0, (rnd - h_sr) / denom)
                    h_factor = 1.0 - ap.h_strength * progress
                elif rnd >= n_rounds - 1:
                    h_factor = 1.0 - ap.h_strength

            # --- Output ---
            c_out_norm = max(0.0, min(1.0, c_norm)) * affordability * h_factor
            c_out = max(0, min(max_c, round(c_out_norm * max_c)))

            round_contribs[j] = c_out
            contrib_traj[j][rnd] = c_out
            budget_traj[j][rnd] = st.B
            strain_traj[j][rnd] = st.strain
            afford_traj[j][rnd] = affordability

        # Step 3: Self-play feedback
        for j in range(n):
            prev_contribs[j] = round_contribs[j]
            states[j].c_prev_norm = round_contribs[j] / max_c

    return {
        'contributions': contrib_traj,
        'budgets': budget_traj,
        'strains': strain_traj,
        'affordabilities': afford_traj,
        'agents': [(phen, sid) for phen, sid, _ in agents],
    }


# ================================================================
# METRICS
# ================================================================

def compute_metrics(sim_result):
    """Compute federation-level and agent-level metrics from one run."""
    contribs = sim_result['contributions']
    budgets = sim_result['budgets']
    agents = sim_result['agents']
    n_agents = len(contribs)
    n_rounds = len(contribs[0])

    # --- TTFR: time-to-first-rupture ---
    ttfr = n_rounds  # default: no rupture
    rupture_rounds = {}  # agent_idx -> round of rupture

    for j in range(n_agents):
        b_init = budgets[j][0]
        if b_init < EPS:
            # Edge case: agent starts with near-zero budget
            continue
        threshold = RUPTURE_B_FRAC * b_init
        consec = 0
        for rnd in range(n_rounds):
            if budgets[j][rnd] < threshold:
                consec += 1
                if consec >= RUPTURE_CONSEC:
                    rupture_rnd = rnd - RUPTURE_CONSEC + 1
                    rupture_rounds[j] = rupture_rnd
                    if rupture_rnd < ttfr:
                        ttfr = rupture_rnd
                    break
            else:
                consec = 0

    # --- Cascade count ---
    if rupture_rounds:
        first_rupture = min(rupture_rounds.values())
        cascade_count = sum(1 for r in rupture_rounds.values() if r > first_rupture)
    else:
        cascade_count = 0

    # --- Steady-state cooperation ---
    ss_start = max(0, n_rounds - STEADY_STATE_WINDOW)
    group_mean_last = np.mean([
        np.mean(contribs[j][ss_start:]) for j in range(n_agents)
    ])
    group_mean_first = np.mean([
        np.mean(contribs[j][:min(10, n_rounds)]) for j in range(n_agents)
    ])

    # --- Cooperation collapse: group mean drops below threshold ---
    coop_collapse_round = n_rounds  # default: no collapse
    coop_consec = 0
    for rnd in range(n_rounds):
        group_mean = np.mean([contribs[j][rnd] for j in range(n_agents)])
        if group_mean < COOP_COLLAPSE_THRESH:
            coop_consec += 1
            if coop_consec >= COOP_COLLAPSE_CONSEC:
                coop_collapse_round = rnd - COOP_COLLAPSE_CONSEC + 1
                break
        else:
            coop_consec = 0

    # --- Time to half cooperation: when group mean drops below 50% of initial ---
    half_coop_round = n_rounds
    if group_mean_first > 2:
        half_thresh = group_mean_first * 0.5
        for rnd in range(n_rounds):
            group_mean = np.mean([contribs[j][rnd] for j in range(n_agents)])
            if group_mean < half_thresh:
                half_coop_round = rnd
                break

    # --- Cooperation at checkpoints (rounds 10, 25, 50, 75, 100) ---
    checkpoints = [10, 25, 50, 75, 100]
    coop_at_checkpoint = {}
    for cp in checkpoints:
        if cp <= n_rounds:
            idx = cp - 1  # 0-indexed
            coop_at_checkpoint[cp] = np.mean([contribs[j][idx] for j in range(n_agents)])

    # --- Trajectory shape ---
    if group_mean_first - group_mean_last > 8 and group_mean_last < 3:
        trajectory_shape = 'collapse'
    elif group_mean_first - group_mean_last > 3:
        trajectory_shape = 'declining'
    else:
        trajectory_shape = 'stable'

    # --- Per-agent metrics ---
    agent_metrics = []
    for j in range(n_agents):
        agent_metrics.append({
            'phenotype': agents[j][0],
            'survived': j not in rupture_rounds,
            'mean_contrib': np.mean(contribs[j]),
            'mean_contrib_last': np.mean(contribs[j][ss_start:]),
            'mean_budget': np.mean(budgets[j]),
            'min_budget': min(budgets[j]),
        })

    return {
        'ttfr': ttfr,
        'cascade_count': cascade_count,
        'n_ruptured': len(rupture_rounds),
        'coop_collapse_round': coop_collapse_round,
        'half_coop_round': half_coop_round,
        'coop_at_checkpoint': coop_at_checkpoint,
        'steady_state_coop': group_mean_last,
        'group_mean_first': group_mean_first,
        'trajectory_shape': trajectory_shape,
        'agent_metrics': agent_metrics,
    }


# ================================================================
# COMPOSITION RUNNER
# ================================================================

COMPOSITIONS = {
    # Homogeneous baselines
    '4CC':              {'CC': 4},
    '4EC':              {'EC': 4},
    '4CD':              {'CD': 4},
    '4DL':              {'DL': 4},
    # Mixed — key comparisons
    '3CC+1CD':          {'CC': 3, 'CD': 1},
    '3EC+1CD':          {'EC': 3, 'CD': 1},
    '2CC+2CD':          {'CC': 2, 'CD': 2},
    '2EC+2CD':          {'EC': 2, 'CD': 2},
    # Cooperator mix
    '2CC+2EC':          {'CC': 2, 'EC': 2},
    # Max diversity
    '1CC+1EC+1DL+1CD':  {'CC': 1, 'EC': 1, 'DL': 1, 'CD': 1},
}


def run_composition(composition, pools, n_runs=N_RUNS, seed=42):
    """Run n_runs simulations of a given group composition."""
    rng = np.random.default_rng(seed)
    all_metrics = []

    for _ in range(n_runs):
        group = sample_group(composition, pools, rng)
        sim_result = simulate_federation(group)
        metrics = compute_metrics(sim_result)
        all_metrics.append(metrics)

    return aggregate_metrics(all_metrics, composition)


def aggregate_metrics(all_metrics, composition):
    """Aggregate metrics across runs into distributional statistics."""
    ttfrs = [m['ttfr'] for m in all_metrics]
    cascades = [m['cascade_count'] for m in all_metrics]
    n_ruptured = [m['n_ruptured'] for m in all_metrics]
    coop_collapses = [m['coop_collapse_round'] for m in all_metrics]
    half_coops = [m['half_coop_round'] for m in all_metrics]
    steady_states = [m['steady_state_coop'] for m in all_metrics]
    first_means = [m['group_mean_first'] for m in all_metrics]
    shapes = [m['trajectory_shape'] for m in all_metrics]

    # Aggregate checkpoint cooperation
    all_checkpoints = defaultdict(list)
    for m in all_metrics:
        for cp, val in m['coop_at_checkpoint'].items():
            all_checkpoints[cp].append(val)

    # Per-phenotype aggregation
    phenotype_survival = defaultdict(list)
    phenotype_contrib = defaultdict(list)
    phenotype_contrib_last = defaultdict(list)
    phenotype_budget = defaultdict(list)

    for m in all_metrics:
        for am in m['agent_metrics']:
            p = am['phenotype']
            phenotype_survival[p].append(am['survived'])
            phenotype_contrib[p].append(am['mean_contrib'])
            phenotype_contrib_last[p].append(am['mean_contrib_last'])
            phenotype_budget[p].append(am['mean_budget'])

    # Shape counts
    shape_counts = defaultdict(int)
    for s in shapes:
        shape_counts[s] += 1

    return {
        'composition': composition,
        'n_runs': len(all_metrics),
        'ttfr': {
            'median': float(np.median(ttfrs)),
            'mean': float(np.mean(ttfrs)),
            'q25': float(np.percentile(ttfrs, 25)),
            'q75': float(np.percentile(ttfrs, 75)),
            'min': int(np.min(ttfrs)),
            'max': int(np.max(ttfrs)),
            'no_rupture_frac': float(np.mean([t == N_ROUNDS for t in ttfrs])),
        },
        'cascade': {
            'mean': float(np.mean(cascades)),
            'median': float(np.median(cascades)),
        },
        'n_ruptured': {
            'mean': float(np.mean(n_ruptured)),
            'median': float(np.median(n_ruptured)),
        },
        'steady_state': {
            'median': float(np.median(steady_states)),
            'mean': float(np.mean(steady_states)),
            'q25': float(np.percentile(steady_states, 25)),
            'q75': float(np.percentile(steady_states, 75)),
        },
        'coop_collapse': {
            'median': float(np.median(coop_collapses)),
            'mean': float(np.mean(coop_collapses)),
            'no_collapse_frac': float(np.mean([c == N_ROUNDS for c in coop_collapses])),
        },
        'half_coop': {
            'median': float(np.median(half_coops)),
            'mean': float(np.mean(half_coops)),
            'q25': float(np.percentile(half_coops, 25)),
            'q75': float(np.percentile(half_coops, 75)),
        },
        'coop_at_checkpoint': {
            cp: float(np.median(vals)) for cp, vals in sorted(all_checkpoints.items())
        },
        'initial_coop': {
            'mean': float(np.mean(first_means)),
        },
        'trajectory_shapes': dict(shape_counts),
        'phenotype_survival': {
            p: float(np.mean(v)) for p, v in phenotype_survival.items()
        },
        'phenotype_contrib': {
            p: {
                'median': float(np.median(v)),
                'q25': float(np.percentile(v, 25)),
                'q75': float(np.percentile(v, 75)),
            }
            for p, v in phenotype_contrib.items()
        },
        'phenotype_contrib_last': {
            p: {
                'median': float(np.median(v)),
                'q25': float(np.percentile(v, 25)),
                'q75': float(np.percentile(v, 75)),
            }
            for p, v in phenotype_contrib_last.items()
        },
        'phenotype_budget': {
            p: {
                'median': float(np.median(v)),
                'q25': float(np.percentile(v, 25)),
                'q75': float(np.percentile(v, 75)),
            }
            for p, v in phenotype_budget.items()
        },
    }


# ================================================================
# RESULTS REPORTING
# ================================================================

def print_results(all_results):
    """Print comprehensive results tables."""

    print(f"\n{'=' * 80}")
    print(f"  FEDERATION SUSTAINABILITY RESULTS")
    print(f"  {N_ROUNDS} rounds, {N_RUNS} runs per composition, {N_AGENTS} agents per group")
    print(f"{'=' * 80}")

    # Table 1: Federation-level summary
    print(f"\n  --- Federation summary ---")
    print(f"  {'Composition':<20s} {'Initial':>8s} {'SS coop':>8s} "
          f"{'T½ coop':>8s} {'Collapse':>9s} {'TTFR':>6s} {'No-rupt':>8s}")
    print(f"  {'-' * 69}")

    for label, r in all_results.items():
        t = r['ttfr']
        ss = r['steady_state']
        ic = r['initial_coop']
        hc = r['half_coop']
        cc = r['coop_collapse']
        print(f"  {label:<20s} {ic['mean']:>8.1f} {ss['median']:>8.1f} "
              f"{hc['median']:>8.0f} {1 - cc['no_collapse_frac']:>8.0%} "
              f"{t['median']:>6.0f} {t['no_rupture_frac']:>7.0%}")

    # Table 1b: Cooperation trajectory at checkpoints
    checkpoints = [10, 25, 50, 75, 100]
    avail_checkpoints = [cp for cp in checkpoints if cp <= N_ROUNDS]
    print(f"\n  --- Group cooperation at round checkpoints (median) ---")
    header = f"  {'Composition':<20s}"
    for cp in avail_checkpoints:
        header += f"  r={cp:<4d}"
    print(header)
    print(f"  {'-' * (20 + 8 * len(avail_checkpoints))}")
    for label, r in all_results.items():
        row = f"  {label:<20s}"
        for cp in avail_checkpoints:
            val = r['coop_at_checkpoint'].get(cp, 0)
            row += f"  {val:>6.1f}"
        print(row)

    # Table 2: Trajectory shapes
    print(f"\n  --- Trajectory shapes (fraction of runs) ---")
    print(f"  {'Composition':<20s} {'stable':>8s} {'declining':>10s} {'collapse':>10s}")
    print(f"  {'-' * 50}")
    for label, r in all_results.items():
        shapes = r['trajectory_shapes']
        total = r['n_runs']
        print(f"  {label:<20s} "
              f"{shapes.get('stable', 0)/total:>8.0%} "
              f"{shapes.get('declining', 0)/total:>10.0%} "
              f"{shapes.get('collapse', 0)/total:>10.0%}")

    # Table 3: Per-phenotype survival
    all_phenos = ['CC', 'EC', 'CD', 'DL']
    print(f"\n  --- Per-phenotype survival rate ---")
    header = f"  {'Composition':<20s}"
    for p in all_phenos:
        header += f"  {p:>6s}"
    print(header)
    print(f"  {'-' * (20 + 8 * len(all_phenos))}")
    for label, r in all_results.items():
        row = f"  {label:<20s}"
        for p in all_phenos:
            if p in r['phenotype_survival']:
                row += f"  {r['phenotype_survival'][p]:>5.0%}"
            else:
                row += f"  {'--':>6s}"
        print(row)

    # Table 4: Per-phenotype contribution (median)
    print(f"\n  --- Per-phenotype mean contribution (median across runs) ---")
    header = f"  {'Composition':<20s}"
    for p in all_phenos:
        header += f"  {p:>6s}"
    print(header)
    print(f"  {'-' * (20 + 8 * len(all_phenos))}")
    for label, r in all_results.items():
        row = f"  {label:<20s}"
        for p in all_phenos:
            if p in r['phenotype_contrib']:
                row += f"  {r['phenotype_contrib'][p]['median']:>6.1f}"
            else:
                row += f"  {'--':>6s}"
        print(row)

    # Table 5: Per-phenotype steady-state contribution
    print(f"\n  --- Per-phenotype steady-state contribution (last {STEADY_STATE_WINDOW} rounds, median) ---")
    header = f"  {'Composition':<20s}"
    for p in all_phenos:
        header += f"  {p:>6s}"
    print(header)
    print(f"  {'-' * (20 + 8 * len(all_phenos))}")
    for label, r in all_results.items():
        row = f"  {label:<20s}"
        for p in all_phenos:
            if p in r['phenotype_contrib_last']:
                row += f"  {r['phenotype_contrib_last'][p]['median']:>6.1f}"
            else:
                row += f"  {'--':>6s}"
        print(row)

    # Table 6: Per-phenotype budget (median)
    print(f"\n  --- Per-phenotype mean budget (median across runs) ---")
    header = f"  {'Composition':<20s}"
    for p in all_phenos:
        header += f"  {p:>6s}"
    print(header)
    print(f"  {'-' * (20 + 8 * len(all_phenos))}")
    for label, r in all_results.items():
        row = f"  {label:<20s}"
        for p in all_phenos:
            if p in r['phenotype_budget']:
                row += f"  {r['phenotype_budget'][p]['median']:>6.2f}"
            else:
                row += f"  {'--':>6s}"
        print(row)


def print_key_comparisons(all_results):
    """Print the theory-testing comparisons."""

    print(f"\n{'=' * 80}")
    print(f"  KEY THEORY COMPARISONS")
    print(f"{'=' * 80}")

    # --- Comparison 1: CC vs EC under extraction ---
    if '3CC+1CD' in all_results and '3EC+1CD' in all_results:
        cc = all_results['3CC+1CD']
        ec = all_results['3EC+1CD']
        print(f"\n  --- Prediction 1+2: CC vs EC under extraction (1 CD) ---")
        print(f"  Theory: CC delays adjustment via inertia → worse eventual collapse")
        print(f"  Theory: EC adjusts early → lower peak but more sustainable")
        print(f"\n  {'Metric':<24s} {'3CC+1CD':>10s} {'3EC+1CD':>10s} {'Delta':>10s}")
        print(f"  {'-' * 56}")
        comparisons = [
            ('Initial cooperation', cc['initial_coop']['mean'], ec['initial_coop']['mean']),
            ('SS cooperation', cc['steady_state']['median'], ec['steady_state']['median']),
            ('T½ cooperation', cc['half_coop']['median'], ec['half_coop']['median']),
            ('Coop collapse %', (1 - cc['coop_collapse']['no_collapse_frac']) * 100,
             (1 - ec['coop_collapse']['no_collapse_frac']) * 100),
            ('TTFR median', cc['ttfr']['median'], ec['ttfr']['median']),
            ('Budget rupture %', (1 - cc['ttfr']['no_rupture_frac']) * 100,
             (1 - ec['ttfr']['no_rupture_frac']) * 100),
        ]
        for metric, cc_v, ec_v in comparisons:
            print(f"  {metric:<24s} {cc_v:>10.1f} {ec_v:>10.1f} {ec_v - cc_v:>+10.1f}")

        # Shape comparison
        cc_collapse = cc['trajectory_shapes'].get('collapse', 0) / cc['n_runs']
        ec_collapse = ec['trajectory_shapes'].get('collapse', 0) / ec['n_runs']
        cc_declining = cc['trajectory_shapes'].get('declining', 0) / cc['n_runs']
        ec_declining = ec['trajectory_shapes'].get('declining', 0) / ec['n_runs']
        cc_stable = cc['trajectory_shapes'].get('stable', 0) / cc['n_runs']
        ec_stable = ec['trajectory_shapes'].get('stable', 0) / ec['n_runs']
        print(f"\n  Trajectory shapes:")
        print(f"    3CC+1CD: {cc_stable:.0%} stable, {cc_declining:.0%} declining, {cc_collapse:.0%} collapse")
        print(f"    3EC+1CD: {ec_stable:.0%} stable, {ec_declining:.0%} declining, {ec_collapse:.0%} collapse")

        # Cooperation trajectory at checkpoints
        print(f"\n  Cooperation trajectory (median):")
        checkpoints = sorted(set(cc['coop_at_checkpoint'].keys()) & set(ec['coop_at_checkpoint'].keys()))
        if checkpoints:
            header = f"    {'':>12s}"
            for cp in checkpoints:
                header += f"  r={cp:<4d}"
            print(header)
            for lab, r in [('3CC+1CD', cc), ('3EC+1CD', ec)]:
                row = f"    {lab:>12s}"
                for cp in checkpoints:
                    row += f"  {r['coop_at_checkpoint'].get(cp, 0):>6.1f}"
                print(row)

        # Verdict
        ec_more_stable = ec_stable > cc_stable
        ec_less_collapse = ec_collapse < cc_collapse
        cc_higher_half = cc['half_coop']['median'] > ec['half_coop']['median']
        print(f"\n  Verdict: EC more stable trajectories? {'YES' if ec_more_stable else 'NO'}")
        print(f"  Verdict: CC more collapse events? {'YES' if ec_less_collapse else 'NO'}")
        print(f"  Verdict: CC delays T½ longer? {'YES' if cc_higher_half else 'NO'}")

    # --- Comparison 2: Marginal cost of extraction (using SS cooperation and T½) ---
    if all(k in all_results for k in ['4CC', '3CC+1CD', '2CC+2CD']):
        print(f"\n  --- Prediction 3: Marginal cost of extraction (CC) ---")
        print(f"  {'':>12s} {'SS coop':>8s} {'T½':>6s} {'Collapse%':>10s}")
        print(f"  {'-' * 38}")
        for lab in ['4CC', '3CC+1CD', '2CC+2CD']:
            r = all_results[lab]
            collapse_pct = 1 - r['coop_collapse']['no_collapse_frac']
            print(f"  {lab:>12s} {r['steady_state']['median']:>8.1f} "
                  f"{r['half_coop']['median']:>6.0f} {collapse_pct:>9.0%}")

        ss_4 = all_results['4CC']['steady_state']['median']
        ss_3 = all_results['3CC+1CD']['steady_state']['median']
        ss_2 = all_results['2CC+2CD']['steady_state']['median']
        drop1_ss = ss_4 - ss_3
        drop2_ss = ss_3 - ss_2
        print(f"\n  SS coop drop: 1st CD = {drop1_ss:+.1f}, 2nd CD = {drop2_ss:+.1f}")
        print(f"  Nonlinear? {'YES' if drop2_ss > drop1_ss else 'NO'}")

        # Same for EC
        if all(k in all_results for k in ['4EC', '3EC+1CD', '2EC+2CD']):
            print(f"\n  --- Marginal cost of extraction (EC) ---")
            print(f"  {'':>12s} {'SS coop':>8s} {'T½':>6s} {'Collapse%':>10s}")
            print(f"  {'-' * 38}")
            for lab in ['4EC', '3EC+1CD', '2EC+2CD']:
                r = all_results[lab]
                collapse_pct = 1 - r['coop_collapse']['no_collapse_frac']
                print(f"  {lab:>12s} {r['steady_state']['median']:>8.1f} "
                      f"{r['half_coop']['median']:>6.0f} {collapse_pct:>9.0%}")

            ss_4e = all_results['4EC']['steady_state']['median']
            ss_3e = all_results['3EC+1CD']['steady_state']['median']
            ss_2e = all_results['2EC+2CD']['steady_state']['median']
            drop1e = ss_4e - ss_3e
            drop2e = ss_3e - ss_2e
            print(f"\n  SS coop drop: 1st CD = {drop1e:+.1f}, 2nd CD = {drop2e:+.1f}")
            print(f"  Nonlinear? {'YES' if drop2e > drop1e else 'NO'}")

    # --- Comparison 3: CD never rupture ---
    print(f"\n  --- Prediction 3: CD never rupture ---")
    cd_ever_ruptured = False
    for label, r in all_results.items():
        if 'CD' in r['phenotype_survival']:
            surv = r['phenotype_survival']['CD']
            if surv < 1.0:
                cd_ever_ruptured = True
                print(f"  {label}: CD survival = {surv:.0%} — RUPTURE DETECTED")
    if not cd_ever_ruptured:
        print(f"  CD survival = 100% across all compositions — CONFIRMED")

    # --- Comparison 4: Cooperator mix ---
    if '2CC+2EC' in all_results:
        mix = all_results['2CC+2EC']
        print(f"\n  --- Cooperator mix (2CC+2EC, no extractors) ---")
        print(f"  TTFR median: {mix['ttfr']['median']:.0f}")
        print(f"  SS cooperation: {mix['steady_state']['median']:.1f}")
        print(f"  No-rupture: {mix['ttfr']['no_rupture_frac']:.0%}")
        if 'CC' in mix['phenotype_survival'] and 'EC' in mix['phenotype_survival']:
            print(f"  CC survival: {mix['phenotype_survival']['CC']:.0%}")
            print(f"  EC survival: {mix['phenotype_survival']['EC']:.0%}")

    # --- Overall theory scorecard ---
    print(f"\n{'=' * 80}")
    print(f"  THEORY SCORECARD")
    print(f"{'=' * 80}")

    if '3CC+1CD' in all_results and '3EC+1CD' in all_results:
        cc_r = all_results['3CC+1CD']
        ec_r = all_results['3EC+1CD']

        # P1: EC produces more stable trajectories
        cc_stable = cc_r['trajectory_shapes'].get('stable', 0) / cc_r['n_runs']
        ec_stable = ec_r['trajectory_shapes'].get('stable', 0) / ec_r['n_runs']
        p1 = ec_stable > cc_stable

        # P2: CC has more collapse events
        p2 = (cc_r['trajectory_shapes'].get('collapse', 0) >
              ec_r['trajectory_shapes'].get('collapse', 0))

        # P3: CD never rupture
        p3 = not cd_ever_ruptured

        print(f"\n  P1: EC more stable than CC under extraction: "
              f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
              f"(EC {ec_stable:.0%} vs CC {cc_stable:.0%} stable)")
        print(f"  P2: CC more collapse events than EC: "
              f"{'CONFIRMED' if p2 else 'FALSIFIED'}")
        print(f"  P3: CD never rupture: "
              f"{'CONFIRMED' if p3 else 'FALSIFIED'}")

        if all(k in all_results for k in ['4CC', '3CC+1CD', '2CC+2CD']):
            ss_4 = all_results['4CC']['steady_state']['median']
            ss_3 = all_results['3CC+1CD']['steady_state']['median']
            ss_2 = all_results['2CC+2CD']['steady_state']['median']
            p4 = (ss_3 - ss_2) > (ss_4 - ss_3)
            print(f"  P4: Nonlinear extraction cost (SS coop): "
                  f"{'CONFIRMED' if p4 else 'FALSIFIED'}")


# ================================================================
# MAIN
# ================================================================

def main():
    t_start = time.time()

    print("=" * 80)
    print("  FEDERATION SUSTAINABILITY SIMULATION")
    print(f"  {N_ROUNDS} rounds, {N_RUNS} runs, {N_AGENTS} agents per group")
    print(f"  576-subject library (P:176 + N:212 + IPD:188)")
    print("=" * 80)

    # Load libraries
    print("\nLoading libraries...")
    libs = load_libraries()
    for tag, lib in libs.items():
        print(f"  {tag}: {len(lib)} subjects")

    # Build phenotype pools
    print("\nBuilding phenotype pools...")
    pools = build_phenotype_pools(libs)
    for phenotype, pool in sorted(pools.items()):
        sources = defaultdict(int)
        params_summary = defaultdict(list)
        for tag, sid, ap in pool:
            sources[tag] += 1
            params_summary['c_base'].append(ap.c_base)
            params_summary['inertia'].append(ap.inertia)
            params_summary['alpha'].append(ap.alpha)
            params_summary['s_dir'].append(ap.s_dir)
            params_summary['b_initial'].append(ap.b_initial)
        print(f"\n  {phenotype}: {len(pool)} subjects ({dict(sources)})")
        print(f"    c_base:  {np.mean(params_summary['c_base']):.3f} "
              f"(std={np.std(params_summary['c_base']):.3f})")
        print(f"    inertia: {np.mean(params_summary['inertia']):.3f} "
              f"(std={np.std(params_summary['inertia']):.3f})")
        print(f"    alpha:   {np.mean(params_summary['alpha']):.3f}")
        print(f"    s_dir:   {np.mean([s > 0 for s in params_summary['s_dir']]):.0%} prosocial")
        print(f"    b_initial: {np.mean(params_summary['b_initial']):.2f}")

    # Verify pool sizes
    for p, pool in pools.items():
        if len(pool) < 20:
            print(f"\n  WARNING: {p} pool has only {len(pool)} subjects (< 20)")

    # Run all compositions
    all_results = {}
    print(f"\n{'=' * 80}")
    print(f"  RUNNING SIMULATIONS")
    print(f"{'=' * 80}")

    for label, composition in COMPOSITIONS.items():
        t0 = time.time()
        result = run_composition(composition, pools, n_runs=N_RUNS, seed=42)
        elapsed = time.time() - t0
        ttfr = result['ttfr']['median']
        ss = result['steady_state']['median']
        print(f"  {label:<20s} TTFR={ttfr:>5.0f}  SS={ss:>5.1f}  ({elapsed:.1f}s)")
        all_results[label] = result

    # Print full results
    print_results(all_results)
    print_key_comparisons(all_results)

    total = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"  COMPLETE — {total:.1f}s total")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
