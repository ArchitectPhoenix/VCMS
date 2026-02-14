"""
VCMS v4 Engine — Adapter Architecture
======================================

Game-agnostic dynamics engine with adapter pattern for I/O mapping.

Changes from v3:
  1. S_exploitation: new strain channel from self-state awareness of
     over-contribution (Book 1 Ch 2 §2.3 — V includes self-observation)
  2. GameConfig: adapter separating game-specific properties from agent dynamics
  3. Discharge gate properly disabled for games without punishment pathway
     (Book 1 Ch 4 §4.3 — Discharge requires payable action path)
  4. Optional V-bandwidth scaling: fewer signals → faster alpha
     (Book 1 Ch 2 §2.3 — High visibility couples action to consequence)
  5. Strain decay: habituation preventing runaway accumulation in long games
     (Book 1 Ch 2 — repeated exposure reduces strain impact)
  6. Horizon scaling: h_start proportional to game length, not absolute round
  7. Ensemble elimination parameters in GameConfig for game-adapted selection
  8. Normalized game time: temporal rate parameters operate in [0,1] game
     progress rather than raw rounds. dt = 1/(n_rounds-1) per step.
     Subsumes horizon_scaling and strain_decay for cross-game transfer.

Backward compatibility:
  s_exploitation_rate=0, v_self_weight=0, PGG_P_CONFIG → identical to v3
  strain_decay=0, horizon_scaling=False, normalized_time=False → identical to v4.0

New parameters (20 total = 18 from v3 + 2 new):
  v_self_weight:       Self-state visibility (0=oblivious, 1=fully aware) [0,1]
  s_exploitation_rate: Exploitation strain accumulation rate [0,2]

Game-specific (adapter, via GameConfig):
  has_punishment:       Whether discharge pathway exists
  use_bandwidth_scaling: Whether alpha adapts to V-channel count
  n_signals:            Number of independent information channels
  strain_decay:         Per-round strain decay rate (habituation)
  horizon_scaling:      Scale h_start proportionally to game length
  normalized_time:      Temporal rates in [0,1] game progress (subsumes above two)
  elim_floor/elim_mult: Ensemble elimination tuning for action space
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional


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
# Game Adapter Configuration
# =============================================================================

@dataclass
class GameConfig:
    """
    Game-specific adapter configuration.

    Separates game structure (what observables exist, what actions are available)
    from agent dynamics (how the agent processes information and decides).

    The adapter defines the I/O contract between the game and the VCMS engine:
    - Input: how game observables map to VCMS signals (V_level, experience, etc.)
    - Output: how VCMS decisions map to game actions (contribution, punishment)
    - Structure: what pathways exist (punishment available? how many signals?)
    """
    max_contrib: int = 20
    max_punish: int = 30
    has_punishment: bool = True          # Does this game provide a discharge pathway?
    n_signals: int = 4                   # V-channel bandwidth (independent info channels)
    use_bandwidth_scaling: bool = False  # Scale alpha by channel bandwidth?
    reference_signals: int = 4           # Baseline signal count for bandwidth scaling
    strain_decay: float = 0.0           # Per-round strain decay (0 = no decay, habituation)
    horizon_scaling: bool = False        # Scale h_start proportionally to game length?
    reference_rounds: int = 10           # Reference game length for horizon scaling
    normalized_time: bool = False        # Temporal rates in normalized game progress [0,1]?
    elim_floor: float = 0.5             # Ensemble elimination distance floor
    elim_mult: float = 3.0              # Ensemble elimination distance multiplier


# Pre-configured adapters for standard games
PGG_P_CONFIG = GameConfig(
    max_contrib=20, max_punish=30, has_punishment=True,
    n_signals=4,  # contribution, others_mean, punishment_sent, punishment_received
)

PGG_N_CONFIG = GameConfig(
    max_contrib=20, max_punish=30, has_punishment=False,
    n_signals=2,  # contribution, others_mean only
)

PGG_N_BW_CONFIG = GameConfig(
    max_contrib=20, max_punish=30, has_punishment=False,
    n_signals=2, use_bandwidth_scaling=True, reference_signals=4,
)

IPD_CONFIG = GameConfig(
    max_contrib=1,    # binary: 1=cooperate, 0=defect
    max_punish=1,     # no punishment in standard IPD (placeholder for RMSE normalization)
    has_punishment=False,
    n_signals=2,      # own action, partner's action
)


# =============================================================================
# Agent Parameters
# =============================================================================

@dataclass
class VCMSParams:
    """
    20 agent-intrinsic parameters. All canon-grounded. Rupture emergent.

    The first 18 match v3 exactly. Two new parameters add self-state
    awareness (V operating on self per Ch 2 §2.3).
    """

    # V: Visibility — what δ conditions on (Ch2 §2.4, D12-D15)
    alpha: float        # V_rep update rate ∈ (0.01, 0.99)
    v_rep: float        # Perceptual scaling ∈ [0.5, 2.0]
    v_ref: float        # Reference blend: 0=disposition, 1=group ∈ [0, 1]

    # C: Cost — baseline + output scaling (Ch2 §2.6, Ch5)
    c_base: float       # Baseline cooperation propensity ∈ [0, 1]
    p_scale: float      # Punishment output scale ∈ [1, 30]

    # M: Memory — behavioral persistence (Ch2 D8)
    inertia: float      # Striation strength ∈ [-0.3, 0.95]

    # S: Strain — motive force, vector (Ch2 D11)
    s_dir: float        # Direction: +1 prosocial, -1 antisocial ∈ {-1, +1}
    s_rate: float       # Accumulation rate ∈ [0, 2]
    s_initial: float    # Pre-loaded strain ∈ [0, 10]

    # Resolution — discharge mechanics (Ch4 Theorem R1)
    s_frac: float       # Discharge fraction ∈ [0, 1]
    s_thresh: float     # B threshold for discharge gate ∈ [0, 5]

    # B: Budget — resource that makes paths affordable (Ch2 D16, Ch5 Theorem M8)
    b_initial: float        # Starting budget ∈ [0.1, 5]
    b_depletion_rate: float # Drain from negative experience ∈ [0, 2]
    b_replenish_rate: float # Recovery from positive experience ∈ [0, 2]
    acute_threshold: float  # Mismatch magnitude for acute pathway ∈ [0, 1]

    # M_eval: Bidirectional cost modification (Ch5 Theorem M1)
    facilitation_rate: float  # Experience → C_eff change rate ∈ [0, 1]

    # H: Horizon awareness
    h_strength: float = 0.0    # Max contribution discount at final round ∈ [0, 1]
    h_start: float = 7.0       # Round index where discounting begins ∈ [0, 9]

    # NEW v4: Self-state awareness (Ch2 §2.3 — V includes self-observation)
    v_self_weight: float = 0.0       # Self-state visibility ∈ [0, 1]
    s_exploitation_rate: float = 0.0  # Exploitation strain rate ∈ [0, 2]


PARAM_NAMES = [
    'c_base', 'alpha', 'v_rep', 'v_ref',
    'inertia',
    's_dir', 's_rate', 's_initial',
    's_frac', 'p_scale', 's_thresh',
    'b_initial', 'b_depletion_rate', 'b_replenish_rate', 'acute_threshold',
    'facilitation_rate',
    'h_strength', 'h_start',
    'v_self_weight', 's_exploitation_rate',
]

PARAM_BOUNDS = {
    'c_base':              (0.0, 1.0),
    'alpha':               (0.01, 0.99),
    'v_rep':               (0.5, 2.0),
    'v_ref':               (0.0, 1.0),
    'inertia':             (-0.3, 0.95),
    's_dir':               (-1.0, 1.0),
    's_rate':              (0.0, 2.0),
    's_initial':           (0.0, 10.0),
    's_frac':              (0.0, 1.0),
    'p_scale':             (1.0, 30.0),
    's_thresh':            (0.0, 5.0),
    'b_initial':           (0.1, 5.0),
    'b_depletion_rate':    (0.0, 2.0),
    'b_replenish_rate':    (0.0, 2.0),
    'acute_threshold':     (0.01, 1.0),
    'facilitation_rate':   (0.0, 1.0),
    'h_strength':          (0.0, 1.0),
    'h_start':             (0.0, 9.0),
    'v_self_weight':       (0.0, 1.0),
    's_exploitation_rate': (0.0, 2.0),
}

# Bounds for normalized-time fitting.
# Rate parameters scale by (n_rounds-1) so bounds widen proportionally.
# h_start is now in [0,1] game progress instead of [0,9] rounds.
PARAM_BOUNDS_NORMALIZED = {
    **PARAM_BOUNDS,
    's_rate':              (0.0, 18.0),    # 2.0 * 9 (10-round equivalent)
    'b_depletion_rate':    (0.0, 18.0),    # 2.0 * 9
    'b_replenish_rate':    (0.0, 18.0),    # 2.0 * 9
    'facilitation_rate':   (0.0, 9.0),     # 1.0 * 9
    'h_start':             (0.0, 1.0),     # fractional game progress
}


# =============================================================================
# Helpers
# =============================================================================

def _sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


def _horizon_factor(t, T, h_strength, h_start):
    """Horizon discount: ramps from 1.0 at h_start to (1-h_strength) at T-1."""
    if T <= 1 or h_strength <= 0.0 or t < h_start:
        return 1.0
    denom = T - 1 - h_start
    if denom <= 0:
        return 1.0 - h_strength if t >= T - 1 else 1.0
    progress = min(1.0, (t - h_start) / denom)
    return 1.0 - h_strength * progress


def _bandwidth_scale(alpha, game_config):
    """
    Scale alpha by V-channel bandwidth.

    Fewer independent signals → each signal carries more weight → faster updating.
    Formula: effective_alpha = alpha × sqrt(reference_signals / n_signals)

    Canon: Ch2 §2.3 — "High visibility couples action to consequence."
    More V-channels = higher effective V = more calibrated updating.
    Fewer V-channels = lower effective V = more responsive updating.
    """
    if not game_config.use_bandwidth_scaling:
        return alpha
    scaling = math.sqrt(game_config.reference_signals / max(1, game_config.n_signals))
    return min(0.99, alpha * scaling)


def v3_params_to_v4(v3_dict, s_exploitation_rate=0.0, v_self_weight=0.0):
    """Convert a v3 library params dict to v4 VCMSParams."""
    return VCMSParams(
        alpha=v3_dict['alpha'],
        v_rep=v3_dict['v_rep'],
        v_ref=v3_dict['v_ref'],
        c_base=v3_dict['c_base'],
        p_scale=v3_dict['p_scale'],
        inertia=v3_dict['inertia'],
        s_dir=v3_dict['s_dir'],
        s_rate=v3_dict['s_rate'],
        s_initial=v3_dict['s_initial'],
        s_frac=v3_dict['s_frac'],
        s_thresh=v3_dict['s_thresh'],
        b_initial=v3_dict['b_initial'],
        b_depletion_rate=v3_dict['b_depletion_rate'],
        b_replenish_rate=v3_dict['b_replenish_rate'],
        acute_threshold=v3_dict['acute_threshold'],
        facilitation_rate=v3_dict['facilitation_rate'],
        h_strength=v3_dict.get('h_strength', 0.0),
        h_start=v3_dict.get('h_start', 7.0),
        v_self_weight=v_self_weight,
        s_exploitation_rate=s_exploitation_rate,
    )


# =============================================================================
# Core Engine
# =============================================================================

def run_vcms_v4(params: VCMSParams, rounds, game_config: GameConfig,
                knockout: Optional[str] = None) -> Dict:
    """
    Run v4 VCMS agent over a sequence of game rounds.

    Architecture: Game Observables → [Input Adapter] → VCMS Dynamics →
                  [Output Adapter] → Game Actions

    The dynamics engine is game-agnostic. The GameConfig defines:
    - What observables are available (n_signals, has_punishment)
    - How VCMS decisions map to actions (max_contrib, max_punish)

    Changes from v3 (all gated by new parameters, backward-compatible):
    1. S_exploitation: strain from self-awareness of over-contribution
    2. Discharge gate: forced closed when has_punishment=False
    3. Alpha: optionally scaled by V-channel bandwidth
    """
    p = params
    gc = game_config
    force_no_routing = (knockout == 'no_routing')
    n_rounds = len(rounds)

    # Snap s_dir to ±1
    s_dir = 1.0 if p.s_dir >= 0 else -1.0

    # Compute effective alpha (bandwidth scaling)
    effective_alpha = _bandwidth_scale(p.alpha, gc)

    # ---- Normalized game time ----
    # When enabled, temporal rate parameters operate in [0,1] game progress.
    # dt = 1/(n_rounds-1) per step. Rates mean "per unit of game progress"
    # rather than "per round", making them transfer across game lengths.
    if gc.normalized_time and n_rounds > 1:
        dt = 1.0 / (n_rounds - 1)
    else:
        dt = 1.0  # Legacy: rates are per-round

    # Compute effective h_start
    if gc.normalized_time:
        # h_start is in [0,1] game progress — convert to round index
        effective_h_start = p.h_start * (n_rounds - 1)
    elif gc.horizon_scaling and gc.reference_rounds > 0:
        effective_h_start = p.h_start * (n_rounds / gc.reference_rounds)
    else:
        effective_h_start = p.h_start

    # === Internal state ===
    v_level = 0.0
    disposition = 0.0
    strain = p.s_initial
    B = p.b_initial
    m_eval = 0.0
    c_prev_norm = 0.0
    pun_recv_prev = 0.0

    # === Outputs ===
    pred_contrib = []
    pred_punish = []
    actual_contrib = []
    actual_punish = []
    trace = []

    for i, rd in enumerate(rounds):
        rt = {}

        # =============================================================
        # INPUT ADAPTER: Game Observables → VCMS Signals
        # =============================================================
        # These lines are the game-specific input mapping.
        # For PGG: contribution, others_mean, punishment → VCMS signals
        # For other games: different mapping, same VCMS signals
        v_group_raw = rd.others_mean / gc.max_contrib
        v_self_raw = rd.contribution / gc.max_contrib  # V(self) — agent's own state

        # =============================================================
        # STEP 1: V — OBSERVE, UPDATE REFERENCE
        # =============================================================
        v_group = min(1.0, p.v_rep * v_group_raw)

        if i == 0:
            v_level = v_group
            disposition = rd.contribution / gc.max_contrib
        else:
            v_level = effective_alpha * v_group + (1.0 - effective_alpha) * v_level
            disposition = ANCHOR_RATE * c_prev_norm + (1.0 - ANCHOR_RATE) * disposition

        reference = p.v_ref * v_level + (1.0 - p.v_ref) * disposition

        rt['v'] = {
            'v_group': v_group, 'v_level': v_level,
            'disposition': disposition, 'reference': reference,
        }

        # =============================================================
        # STEP 2: S — STRAIN ACCUMULATION
        # Two channels:
        #   S_social_gap: discrepancy between own prior and group reference
        #   S_exploitation: self-awareness of over-contribution (NEW v4)
        # =============================================================

        # Channel 1: Social gap strain (identical to v3)
        if i > 0:
            gap = c_prev_norm - reference
        else:
            gap = 0.0

        directed_gap = gap * s_dir
        gap_strain = max(0.0, directed_gap)
        pun_strain = pun_recv_prev / 15.0

        # Channel 2: Exploitation strain (NEW v4)
        # S_exploitation = s_exploitation_rate × max(0, V_self - V_level) × v_self_weight
        # High when cooperator in defecting group. Zero for comfortable defectors.
        if i > 0 and p.s_exploitation_rate > 0.0 and p.v_self_weight > 0.0:
            exploitation_gap = max(0.0, c_prev_norm - v_group_raw)
            s_exploitation = p.s_exploitation_rate * exploitation_gap * p.v_self_weight
        else:
            s_exploitation = 0.0

        # Total strain accumulation (dt-scaled)
        strain += dt * p.s_rate * (gap_strain + pun_strain) + dt * s_exploitation

        # Strain decay (habituation — prevents runaway in long games)
        # Note: with normalized_time, strain rates are already temporally scaled,
        # so strain_decay is typically unnecessary. Kept for backward compat.
        if gc.strain_decay > 0:
            strain *= (1.0 - gc.strain_decay)

        rt['s_accum'] = {
            'gap': gap, 'directed_gap': directed_gap,
            'gap_strain': gap_strain, 'pun_strain': pun_strain,
            's_exploitation': s_exploitation,
            'strain_pre_discharge': strain,
        }

        # =============================================================
        # STEP 3: B — BUDGET UPDATE
        # =============================================================
        if i > 0:
            experience = v_group_raw - c_prev_norm
        else:
            experience = 0.0

        b_pre = B

        if experience < 0:
            magnitude = abs(experience)
            depletion = dt * p.b_depletion_rate * magnitude
            if magnitude > p.acute_threshold:
                depletion *= ACUTE_MULT
            B -= depletion
        elif experience > 0:
            pun_gate = max(0.0, 1.0 - pun_recv_prev / gc.max_punish)
            B += dt * p.b_replenish_rate * experience * pun_gate

        if i > 0:
            B -= dt * p.b_depletion_rate * (pun_recv_prev / 15.0)

        B = max(0.0, B)

        rt['budget'] = {
            'experience': experience, 'b_pre': b_pre, 'b_post': B,
        }

        # =============================================================
        # STEP 4: M_EVAL — FACILITATION / INHIBITION
        # =============================================================
        if i > 0:
            m_eval += dt * p.facilitation_rate * experience

        rt['m_eval'] = {'m_eval_acc': m_eval}

        # =============================================================
        # STEP 5: RESOLUTION ROUTING
        # v4 change: gate forced to 0 when game has no punishment pathway
        # (Book 1 Ch 4 §4.3: Discharge requires payable action path)
        # =============================================================

        if gc.has_punishment:
            gate = _sigmoid((B - p.s_thresh) / max(B_NOISE, 0.001))
        else:
            # No discharge pathway — all strain feeds into affordability
            gate = 0.0

        discharge = gate * p.s_frac * strain
        remaining_strain = max(0.0, strain - discharge)

        if force_no_routing:
            affordability = 1.0
        else:
            affordability = B / (B + remaining_strain + EPS)

        rt['routing'] = {
            'gate': gate, 'discharge': discharge,
            'remaining_strain': remaining_strain,
            'affordability': affordability,
        }

        # =============================================================
        # STEP 6: M — MEMORY / INERTIA
        # =============================================================
        if i == 0:
            c_norm = p.c_base
        else:
            w = max(-0.3, min(0.95, p.inertia))
            c_target = p.v_ref * v_level + (1.0 - p.v_ref) * p.c_base
            c_target_adj = max(0.0, min(1.0, c_target + m_eval))
            c_norm = (1.0 - abs(w)) * c_target_adj + w * c_prev_norm

        # =============================================================
        # STEP 7: OUTPUT C — via output adapter
        # =============================================================
        h_factor = _horizon_factor(i, len(rounds), p.h_strength, effective_h_start)
        c_out_norm = max(0.0, min(1.0, c_norm)) * affordability * h_factor
        c_out = round(c_out_norm * gc.max_contrib)
        c_out = max(0, min(gc.max_contrib, c_out))

        pred_contrib.append(c_out)
        actual_contrib.append(rd.contribution)

        rt['c_output'] = {
            'c_norm': c_norm, 'c_out_norm': c_out_norm,
            'h_factor': h_factor,
            'c_out': c_out, 'actual_c': rd.contribution,
        }

        # =============================================================
        # STEP 8: OUTPUT P — PUNISHMENT (only if game has punishment)
        # =============================================================
        if gc.has_punishment:
            current_gap = (rd.contribution - rd.others_mean) / gc.max_contrib
            reactive = gate * max(0.0, s_dir * current_gap)
            p_raw = (discharge + reactive) * p.p_scale
            p_out = max(0, min(gc.max_punish, round(p_raw)))
        else:
            current_gap = 0.0
            reactive = 0.0
            p_raw = 0.0
            p_out = 0

        pred_punish.append(p_out)
        actual_punish.append(rd.punishment_sent_total)

        rt['p_output'] = {
            'current_gap': current_gap, 'reactive': reactive,
            'p_raw': p_raw, 'p_out': p_out,
            'actual_p': rd.punishment_sent_total,
        }

        # =============================================================
        # STEP 9: STATE UPDATE — teacher forcing
        # =============================================================
        strain = remaining_strain
        c_prev_norm = rd.contribution / gc.max_contrib
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

    norm_errors = []
    for a, p_ in zip(actual_contrib, pred_contrib):
        norm_errors.append(((a - p_) / gc.max_contrib) ** 2)
    for a, p_ in zip(actual_punish, pred_punish):
        norm_errors.append(((a - p_) / gc.max_punish) ** 2)
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
