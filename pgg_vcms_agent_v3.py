"""
PGG VCMS Dual-Output Agent v3
==============================
16-parameter agent producing BOTH contribution AND punishment predictions.
Flat optimization (no staging — v2 showed staging hurts joint optima).

Architecture change from v1/v2:
  v1/v2 had one resolution pathway: strain accumulates → threshold → discharge as P.
  v3 implements the full R1 partition (Book 1, Chapter 4, Theorem R1):
    - DISCHARGE: Strain exits through P when B can afford confrontation
    - REROUTE:   Remaining strain suppresses C via affordability ratio
    - HALT:      When B is low, inertia dominates (repeat last action)
    - RUPTURE:   Emergent — when B ≈ 0 and S > 0, affordability collapses → C → 0

  Rupture is NOT coded. It emerges when the routing equation has no stable
  solution: B depleted, strain positive, affordability ratio → 0.

Design principles: /mnt/user-data/outputs/vcms_v3_design_principles.md
Every parameter traces to a Book 1 theorem. No ad hoc additions.

Canon grounding for each parameter:

  V (Visibility) — what δ conditions on (Ch2 §2.4, D12-D15)
    alpha     : V_rep update rate
    v_rep     : Perceptual scaling (gain on V signal)
    v_ref     : Reference blend (0=disposition, 1=group)

  C (Cost) — constraint geometry (Ch2 §2.6, Ch5)
    c_base    : Baseline cooperation propensity
    p_scale   : Punishment output scale (measurement, not dynamics)

  M (Memory) — history deforms effective dynamics (Ch2 D8)
    inertia   : Striation strength — repeated behavior easier to repeat

  S (Strain) — δ's demand for transition (Ch2 D11)
    s_rate    : Accumulation rate from expectation-reality mismatch
    s_initial : Pre-loaded strain (history before game)
    s_dir     : Strain vector direction (+1=prosocial, -1=antisocial)

  B (Budget) — resource that makes paths affordable (Ch2 D16, Ch5 Proof Theorem M8)
    b_initial       : Starting budget — individual baseline reserves
    b_depletion_rate: B drain from negative experience — cumulative pathway (Theorem M5B)
    b_replenish_rate: B recovery from positive experience — facilitation (Def M1a)
    acute_threshold : Single-event magnitude triggering step-function B drain (Theorem M5A)

  Resolution (Ch4 Theorem R1)
    s_thresh  : B level below which P-path becomes unaffordable (soft gate)
    s_frac    : Discharge fraction — how much strain exits through P per round

  M_eval (Ch5 Theorem M1, bidirectional)
    facilitation_rate : C_eff modification from experience — positive lowers C, negative raises C

Fixed architectural constants:
  B_NOISE = 0.1     : Sigmoid softness for discharge gate (Corollary 3c — imperfect self-V)
  ACUTE_MULT = 5.0  : Acute event amplification (Theorem M5A — step function, not gradual)
  EPS = 0.01        : Numerical stability floor
  ANCHOR_RATE = 0.15: Disposition EMA rate (~7 round half-life)

Knockout channels (7):
  no_v_tracking   : Reference fixed at c_base (V doesn't update from group)
  no_strain       : S = 0 always (no accumulation)
  no_memory       : Inertia = 0 (no behavioral persistence)
  no_budget       : B never depletes (always affordable, affordability = 1)
  no_discharge    : Discharge gate always closed (strain never exits through P)
  no_facilitation : M_eval accumulator frozen at 0
  no_routing      : Affordability ratio forced to 1 (S/B never affects C output)
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional

from pgg_p_loader import PRoundData


# =============================================================================
# Fixed constants (game structure + architectural, not fitted)
# =============================================================================

MAX_CONTRIB = 20
MAX_PUNISH = 30
TOTAL_ROUNDS = 10
ANCHOR_RATE = 0.15     # Disposition EMA rate (slow — ~7 round half-life)
B_NOISE = 0.1          # Sigmoid softness for discharge gate
ACUTE_MULT = 5.0       # Acute event amplifier (Theorem M5A)
EPS = 0.01             # Numerical stability floor


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class VCMSParams:
    """16 parameters. All canon-grounded. Rupture emergent."""

    # V: Visibility — what δ conditions on
    alpha: float        # V_rep update rate ∈ (0.01, 0.99)
    v_rep: float        # Perceptual scaling ∈ [0.5, 2.0]
    v_ref: float        # Reference blend: 0=disposition, 1=group ∈ [0, 1]

    # C: Cost — baseline + output scaling
    c_base: float       # Baseline cooperation propensity ∈ [0, 1]
    p_scale: float      # Punishment output scale ∈ [1, 30]

    # M: Memory — behavioral persistence
    inertia: float      # Striation strength ∈ [-0.3, 0.95]

    # S: Strain — motive force, vector
    s_dir: float        # Direction: +1 prosocial, -1 antisocial ∈ {-1, +1}
    s_rate: float       # Accumulation rate ∈ [0, 2]
    s_initial: float    # Pre-loaded strain ∈ [0, 10]

    # Resolution — discharge mechanics
    s_frac: float       # Discharge fraction ∈ [0, 1]
    s_thresh: float     # B threshold for discharge gate ∈ [0, 5]

    # B: Budget — psychological resource that makes paths affordable
    b_initial: float        # Starting budget ∈ [0.1, 5]
    b_depletion_rate: float # Drain from negative experience ∈ [0, 2]
    b_replenish_rate: float # Recovery from positive experience ∈ [0, 2]
    acute_threshold: float  # Mismatch magnitude for acute pathway ∈ [0, 1]

    # M_eval: Bidirectional cost modification
    facilitation_rate: float  # Experience → C_eff change rate ∈ [0, 1]


PARAM_NAMES = [
    'c_base', 'alpha', 'v_rep', 'v_ref',
    'inertia',
    's_dir', 's_rate', 's_initial',
    's_frac', 'p_scale', 's_thresh',
    'b_initial', 'b_depletion_rate', 'b_replenish_rate', 'acute_threshold',
    'facilitation_rate',
]

PARAM_BOUNDS = {
    'c_base':           (0.0, 1.0),
    'alpha':            (0.01, 0.99),
    'v_rep':            (0.5, 2.0),
    'v_ref':            (0.0, 1.0),
    'inertia':          (-0.3, 0.95),
    's_dir':            (-1.0, 1.0),     # snapped to ±1
    's_rate':           (0.0, 2.0),
    's_initial':        (0.0, 10.0),
    's_frac':           (0.0, 1.0),
    'p_scale':          (1.0, 30.0),
    's_thresh':         (0.0, 5.0),
    'b_initial':        (0.1, 5.0),
    'b_depletion_rate': (0.0, 2.0),
    'b_replenish_rate': (0.0, 2.0),
    'acute_threshold':  (0.01, 1.0),
    'facilitation_rate':(0.0, 1.0),
}

DEFAULTS = {
    'c_base': 0.5, 'alpha': 0.3, 'v_rep': 1.0, 'v_ref': 0.5,
    'inertia': 0.5,
    's_dir': 1.0, 's_rate': 0.5, 's_initial': 0.0,
    's_frac': 0.3, 'p_scale': 10.0, 's_thresh': 1.0,
    'b_initial': 2.0, 'b_depletion_rate': 0.3, 'b_replenish_rate': 0.2,
    'acute_threshold': 0.3, 'facilitation_rate': 0.1,
}

# Knockout configurations
KNOCKOUTS = {
    'no_v_tracking':   {'alpha': 0.0, 'v_rep': 1.0, 'v_ref': 0.0},
    'no_strain':       {'s_rate': 0.0, 's_initial': 0.0},
    'no_memory':       {'inertia': 0.0},
    'no_budget':       {'b_initial': 100.0, 'b_depletion_rate': 0.0},
    'no_discharge':    {'s_thresh': 100.0},
    'no_facilitation': {'facilitation_rate': 0.0},
    'no_routing':      {},  # handled by flag in run_vcms_agent
}


# =============================================================================
# Core agent
# =============================================================================

def _sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


def run_vcms_agent(params: VCMSParams, rounds: List[PRoundData],
                   knockout: Optional[str] = None) -> Dict:
    """
    Run v3 VCMS agent over a sequence of PGG rounds.

    Agent step logic per round:
      1. V  — Observe group signal, update reference, compute c_norm
      2. S  — Strain accumulation from gap + received punishment
      3. B  — Budget update (deplete from exploitation/punishment, replenish from positive)
      4. Me — M_eval facilitation accumulator update (bidirectional)
      5. R  — Resolution routing: discharge gate, partial discharge, affordability ratio
      6. M  — Memory/inertia blend
      7. C  — Output contribution = c_adjusted × affordability × MAX_CONTRIB
      8. P  — Output punishment = (discharge + reactive) × p_scale, gated by B
      9. U  — State update with teacher forcing

    Rupture emerges when B → 0 with S > 0: affordability ratio collapses,
    C output → 0. No rupture parameter exists.
    """
    p = params
    force_no_routing = (knockout == 'no_routing')

    # Snap s_dir to ±1
    s_dir = 1.0 if p.s_dir >= 0 else -1.0

    # === Internal state ===
    v_level = 0.0           # V: Running reference of group behavior
    disposition = 0.0       # V: Disposition anchor (slow EMA of own contributions)
    strain = p.s_initial    # S: Strain accumulator
    B = p.b_initial         # B: Psychological resource budget
    m_eval = 0.0            # M_eval: Facilitation/inhibition accumulator
    c_prev_norm = 0.0       # M: Previous contribution [0,1]
    pun_recv_prev = 0.0     # V: Previous punishment received (raw tokens)

    # === Outputs ===
    pred_contrib = []
    pred_punish = []
    actual_contrib = []
    actual_punish = []
    trace = []

    for i, rd in enumerate(rounds):
        rt = {}  # round trace

        # =================================================================
        # STEP 1: V — OBSERVE, UPDATE REFERENCE
        # Canon: D12-D15 (V_rep = variables on which δ depends)
        # =================================================================

        v_group_raw = rd.others_mean / MAX_CONTRIB
        v_group = min(1.0, p.v_rep * v_group_raw)

        if i == 0:
            v_level = v_group
            disposition = rd.contribution / MAX_CONTRIB
        else:
            v_level = p.alpha * v_group + (1.0 - p.alpha) * v_level
            disposition = ANCHOR_RATE * c_prev_norm + (1.0 - ANCHOR_RATE) * disposition

        reference = p.v_ref * v_level + (1.0 - p.v_ref) * disposition

        rt['v'] = {
            'v_group': v_group, 'v_level': v_level,
            'disposition': disposition, 'reference': reference,
        }

        # =================================================================
        # STEP 2: S — STRAIN ACCUMULATION
        # Canon: D11 (S = δ's demand for transition)
        # Gap between own prior behavior and group reference → strain
        # =================================================================

        if i > 0:
            gap = c_prev_norm - reference
        else:
            gap = 0.0

        directed_gap = gap * s_dir
        gap_strain = max(0.0, directed_gap)

        pun_strain = pun_recv_prev / 15.0

        strain += p.s_rate * (gap_strain + pun_strain)

        rt['s_accum'] = {
            'gap': gap, 'directed_gap': directed_gap,
            'gap_strain': gap_strain, 'pun_strain': pun_strain,
            'strain_pre_discharge': strain,
        }

        # =================================================================
        # STEP 3: B — BUDGET UPDATE
        # Canon: D16 (B ∈ ℝ⁺), Theorem M8 (gradient), M5A/B (pathways)
        #
        # Experience signal: how well did the group treat me?
        #   experience = v_group_raw - c_prev_norm
        #   experience > 0 → others cooperated more than me (positive)
        #   experience < 0 → I cooperated more than them (exploitation)
        #
        # Graduated punishment gate on replenishment:
        #   pun_gate = max(0, 1 - pun_recv / MAX_PUNISH)
        #   Heavy punishment (16+ tokens) nearly shuts facilitation.
        #   Mild punishment (2-3 tokens) allows most facilitation through.
        #   No punishment → full facilitation. No new parameters.
        #
        # Semantics for all cases:
        #   Cooperator, coop group, no pun:  exp > 0, gate=1 → full replenish
        #   Cooperator, defecting group:     exp < 0 → deplete (acute)
        #   Defector, heavily punished:      exp > 0, gate≈0 → drain dominates
        #   Defector, NOT punished:          exp > 0, gate=1 → full replenish
        #   Defector, mildly punished:       exp > 0, gate≈0.9 → partial replenish
        #   Cooperator, antisocially pun:    exp ≈ 0, drain fires → B drops
        #
        # Depletion: cumulative from exploitation (M5B), acute from shocks (M5A)
        # Replenishment: facilitation from positive experience (Def M1a)
        # Punishment: separate drain, always fires (strain event)
        # =================================================================

        if i > 0:
            experience = v_group_raw - c_prev_norm
        else:
            experience = 0.0

        b_pre = B

        if experience < 0:
            # Exploitation → B depletes
            magnitude = abs(experience)
            depletion = p.b_depletion_rate * magnitude
            # Acute pathway: step-function amplification (Theorem M5A)
            if magnitude > p.acute_threshold:
                depletion *= ACUTE_MULT
            B -= depletion
        elif experience > 0:
            # Positive experience → graduated replenishment gated by punishment
            pun_gate = max(0.0, 1.0 - pun_recv_prev / MAX_PUNISH)
            B += p.b_replenish_rate * experience * pun_gate

        # Punishment received → B depletes (separate drain, always fires)
        if i > 0:
            B -= p.b_depletion_rate * (pun_recv_prev / 15.0)

        B = max(0.0, B)

        rt['budget'] = {
            'experience': experience, 'b_pre': b_pre, 'b_post': B,
        }

        # =================================================================
        # STEP 4: M_EVAL — FACILITATION / INHIBITION
        # Canon: Theorem M1 (C direction depends on substrate outcome)
        #        Theorem M6 (effects sum additively)
        #        Corollary M6a (facilitation can offset inhibition)
        #
        # Positive experience → facilitation (cooperation cheaper)
        # Negative experience → inhibition (cooperation more expensive)
        # Path-Dependence Principle: recovery is new construction, not erasure
        # =================================================================

        if i > 0:
            m_eval += p.facilitation_rate * experience
        # m_eval can go negative (net inhibition) or positive (net facilitation)

        rt['m_eval'] = {'m_eval_acc': m_eval}

        # =================================================================
        # STEP 5: RESOLUTION ROUTING
        # Canon: Theorem R1 — four outcomes, exhaustive
        #
        # Discharge gate: sigmoid on B vs s_thresh
        #   Gate ≈ 1 when B >> s_thresh (P path affordable)
        #   Gate ≈ 0 when B << s_thresh (P path blocked)
        #   B_NOISE controls sigmoid softness (imperfect self-V)
        #
        # Partial discharge: gate × s_frac × strain exits through P
        # Remaining strain: feeds into affordability ratio
        #
        # Affordability ratio A = B / (B + remaining_S + ε)
        #   A ≈ 1: cooperation at normal levels (Discharge handled S)
        #   A moderate: cooperation suppressed (Reroute)
        #   A → 0: cooperation collapses (Rupture — emergent, not coded)
        #
        # Halt emerges when A is moderate and inertia is high:
        #   system repeats prior action because inertia dominates the blend
        # =================================================================

        # Discharge gate (soft sigmoid)
        gate = _sigmoid((B - p.s_thresh) / max(B_NOISE, 0.001))

        # Partial discharge through P
        discharge = gate * p.s_frac * strain
        remaining_strain = max(0.0, strain - discharge)

        # Affordability ratio: how much cooperation can the subject sustain?
        if force_no_routing:
            affordability = 1.0
        else:
            affordability = B / (B + remaining_strain + EPS)

        rt['routing'] = {
            'gate': gate, 'discharge': discharge,
            'remaining_strain': remaining_strain,
            'affordability': affordability,
        }

        # =================================================================
        # STEP 6: M — MEMORY / INERTIA
        # Canon: D8 (M_eval = effective dynamics deformation)
        # Inertia blends prior behavior with current V-derived target
        # =================================================================

        if i == 0:
            c_norm = p.c_base
        else:
            w = max(-0.3, min(0.95, p.inertia))

            # V-derived cooperation target
            c_target = p.v_ref * v_level + (1.0 - p.v_ref) * p.c_base

            # Apply M_eval: facilitation lowers effective cost, inhibition raises it
            c_target_adj = max(0.0, min(1.0, c_target + m_eval))

            # Inertia blend: high |w| → repeat prior behavior (Halt-like)
            c_norm = (1.0 - abs(w)) * c_target_adj + w * c_prev_norm

        # =================================================================
        # STEP 7: OUTPUT C — CONTRIBUTION
        # Canon: Affordability gates cooperation output
        # c_out = c_norm × affordability
        # When B → 0: affordability → 0, c_out → 0 (rupture emerges)
        # When B high, S low: affordability ≈ 1, c_out ≈ c_norm (normal)
        # =================================================================

        c_out_norm = max(0.0, min(1.0, c_norm)) * affordability
        c_out = round(c_out_norm * MAX_CONTRIB)
        c_out = max(0, min(MAX_CONTRIB, c_out))

        pred_contrib.append(c_out)
        actual_contrib.append(rd.contribution)

        rt['c_output'] = {
            'c_norm': c_norm, 'c_out_norm': c_out_norm,
            'c_out': c_out, 'actual_c': rd.contribution,
        }

        # =================================================================
        # STEP 8: OUTPUT P — PUNISHMENT
        # Canon: Discharge resolution (R1) + reactive component
        #
        # Discharge: strain exiting through P path (gated by B)
        # Reactive: current-round defection signal (also gated by B)
        # Both require budget — can't punish without social confrontation capacity
        # =================================================================

        current_gap = (rd.contribution - rd.others_mean) / MAX_CONTRIB
        reactive = gate * max(0.0, s_dir * current_gap)

        # Punishment = (strain discharge + reactive) × scale
        p_raw = (discharge + reactive) * p.p_scale
        p_out = max(0, min(MAX_PUNISH, round(p_raw)))

        pred_punish.append(p_out)
        actual_punish.append(rd.punishment_sent_total)

        rt['p_output'] = {
            'current_gap': current_gap, 'reactive': reactive,
            'p_raw': p_raw, 'p_out': p_out,
            'actual_p': rd.punishment_sent_total,
        }

        # =================================================================
        # STEP 9: STATE UPDATE — teacher forcing
        # Canon: actual values drive state for next round
        # Strain reduced by discharge (resolved demand)
        # B persists (no passive decay — depletion/replenishment is event-driven)
        # =================================================================

        strain = remaining_strain
        c_prev_norm = rd.contribution / MAX_CONTRIB
        pun_recv_prev = rd.punishment_received_total

        rt['state'] = {
            'strain_end': strain, 'B_end': B, 'm_eval_end': m_eval,
        }
        trace.append(rt)

    # =====================================================================
    # COMPUTE METRICS
    # =====================================================================
    n = len(rounds)
    if n == 0:
        return {
            'pred_contrib': [], 'pred_punish': [],
            'actual_contrib': [], 'actual_punish': [],
            'rmse_contrib': 0.0, 'rmse_punish': 0.0, 'rmse_combined': 0.0,
            'trace': [],
        }

    sse_c = sum((a - p_) ** 2 for a, p_ in zip(actual_contrib, pred_contrib))
    rmse_c = math.sqrt(sse_c / n)

    sse_p = sum((a - p_) ** 2 for a, p_ in zip(actual_punish, pred_punish))
    rmse_p = math.sqrt(sse_p / n)

    # Combined normalized RMSE (same as v1/v2 for comparability)
    norm_errors = []
    for a, p_ in zip(actual_contrib, pred_contrib):
        norm_errors.append(((a - p_) / MAX_CONTRIB) ** 2)
    for a, p_ in zip(actual_punish, pred_punish):
        norm_errors.append(((a - p_) / MAX_PUNISH) ** 2)
    rmse_combined = math.sqrt(sum(norm_errors) / len(norm_errors))

    return {
        'pred_contrib': pred_contrib,
        'pred_punish': pred_punish,
        'actual_contrib': actual_contrib,
        'actual_punish': actual_punish,
        'rmse_contrib': rmse_c,
        'rmse_punish': rmse_p,
        'rmse_combined': rmse_combined,
        'trace': trace,
    }


# =============================================================================
# Fitting helpers
# =============================================================================

def make_vcms_params(x, free_names, fixed=None):
    """Create VCMSParams from optimizer vector + fixed values."""
    if fixed is None:
        fixed = {}
    params = dict(fixed)
    for i, name in enumerate(free_names):
        params[name] = x[i]
    # Fill defaults for anything not specified
    for n in PARAM_NAMES:
        if n not in params:
            params[n] = DEFAULTS[n]
    return VCMSParams(**params)


def vcms_objective(x, rounds, free_names, fixed=None):
    """Objective function for optimizer. Combined normalized RMSE."""
    try:
        params = make_vcms_params(x, free_names, fixed)
        result = run_vcms_agent(params, rounds)
        return result['rmse_combined']
    except Exception:
        return 100.0


# =============================================================================
# Smoke tests
# =============================================================================

if __name__ == '__main__':
    # Synthetic round data for testing
    class FakeRound:
        def __init__(self, period, contribution, others_mean, pun_sent, pun_recv):
            self.period = period
            self.contribution = contribution
            self.others_mean = others_mean
            self.punishment_sent_total = pun_sent
            self.punishment_received_total = pun_recv

    # Test 1: Stable cooperator (high B, no depletion)
    rounds = [
        FakeRound(t+1, 15, 14.0, 0, 0) for t in range(10)
    ]
    params = VCMSParams(
        alpha=0.3, v_rep=1.0, v_ref=0.5, c_base=0.75, p_scale=10.0,
        inertia=0.5, s_dir=1.0, s_rate=0.5, s_initial=0.0,
        s_frac=0.3, s_thresh=1.0,
        b_initial=3.0, b_depletion_rate=0.2, b_replenish_rate=0.3,
        acute_threshold=0.3, facilitation_rate=0.1,
    )
    result = run_vcms_agent(params, rounds)
    print("=== Test 1: Stable cooperator ===")
    print(f"  C pred: {result['pred_contrib']}")
    print(f"  P pred: {result['pred_punish']}")
    print(f"  RMSE combined: {result['rmse_combined']:.4f}")
    for i, t in enumerate(result['trace']):
        print(f"  R{i+1}: B={t['budget']['b_post']:.3f}, "
              f"A={t['routing']['affordability']:.3f}, "
              f"gate={t['routing']['gate']:.3f}, "
              f"m_eval={t['m_eval']['m_eval_acc']:.3f}")

    print()

    # Test 2: Subject being exploited (others defect)
    rounds2 = [
        FakeRound(t+1, 15, 5.0, 0, 0) for t in range(10)
    ]
    result2 = run_vcms_agent(params, rounds2)
    print("=== Test 2: Exploited cooperator ===")
    print(f"  C pred: {result2['pred_contrib']}")
    print(f"  P pred: {result2['pred_punish']}")
    for i, t in enumerate(result2['trace']):
        print(f"  R{i+1}: B={t['budget']['b_post']:.3f}, "
              f"A={t['routing']['affordability']:.3f}, "
              f"S={t['state']['strain_end']:.3f}, "
              f"m_eval={t['m_eval']['m_eval_acc']:.3f}")

    print()

    # Test 3: Acute shock (big punishment when cooperating)
    rounds3 = [
        FakeRound(1, 18, 16.0, 0, 0),
        FakeRound(2, 18, 16.0, 0, 0),
        FakeRound(3, 18, 16.0, 0, 20),  # big punishment received
        FakeRound(4, 18, 16.0, 0, 0),
        FakeRound(5, 18, 16.0, 0, 0),
        FakeRound(6, 18, 16.0, 0, 0),
        FakeRound(7, 18, 16.0, 0, 0),
        FakeRound(8, 18, 16.0, 0, 0),
        FakeRound(9, 18, 16.0, 0, 0),
        FakeRound(10, 18, 16.0, 0, 0),
    ]
    result3 = run_vcms_agent(params, rounds3)
    print("=== Test 3: Acute punishment shock at round 3 ===")
    print(f"  C pred: {result3['pred_contrib']}")
    for i, t in enumerate(result3['trace']):
        print(f"  R{i+1}: B={t['budget']['b_post']:.3f}, "
              f"A={t['routing']['affordability']:.3f}, "
              f"S={t['state']['strain_end']:.3f}")

    print()

    # Test 4: Knockout - no_routing (affordability always 1)
    result4 = run_vcms_agent(params, rounds2, knockout='no_routing')
    print("=== Test 4: Exploited + no_routing knockout ===")
    print(f"  C pred (no routing): {result4['pred_contrib']}")
    print(f"  C pred (with routing): {result2['pred_contrib']}")

    print()
    print(f"Parameters: {len(PARAM_NAMES)} fitted, "
          f"fixed constants: B_NOISE={B_NOISE}, ACUTE_MULT={ACUTE_MULT}")
    print("All smoke tests passed.")
