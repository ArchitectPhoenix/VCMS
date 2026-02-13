"""
IPD Data Loader + VCMS Adapter
===============================

Loads Iterated Prisoner's Dilemma experimental data and adapts it
to the VCMS engine interface.

Key adapter mapping:
  IPD action (C=1, D=0) → rd.contribution (binary cooperation level)
  Partner action (C=1, D=0) → rd.others_mean (environment signal)
  No punishment → rd.punishment_sent_total = 0, rd.punishment_received_total = 0

With IPD_CONFIG (max_contrib=1), the engine's output adapter naturally
thresholds: round(c_out_norm * 1) = 0 (defect) or 1 (cooperate).

Payoff matrix (experimental data: Herrmann et al.):
  CC = (3,3)   Mutual cooperation
  CD = (0,4)   Sucker / Temptation
  DC = (4,0)   Temptation / Sucker
  DD = (1,1)   Mutual defection

  R=3 (reward), T=4 (temptation), S=0 (sucker), P=1 (punishment)
  Satisfies T > R > P > S and 2R > T + S

Usage:
    from ipd_loader import load_ipd_experiment, IPDRound
    data = load_ipd_experiment('IPD-rand.csv')
    for sid, rounds in data.items():
        for rd in rounds:
            print(rd.contribution, rd.others_mean, rd.payoff)
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Standard IPD payoff matrix
PAYOFF_R = 3  # Reward (mutual cooperation)
PAYOFF_T = 4  # Temptation (defect while partner cooperates)
PAYOFF_S = 0  # Sucker (cooperate while partner defects)
PAYOFF_P = 1  # Punishment (mutual defection)


@dataclass
class IPDRound:
    """
    One round of IPD data, adapted to VCMS engine interface.

    The VCMS engine reads:
      .contribution           → own action (1=C, 0=D)
      .others_mean            → partner action (1.0=C, 0.0=D)
      .punishment_sent_total  → 0 (no punishment in IPD)
      .punishment_received_total → 0

    Additional IPD-specific fields:
      .own_action             → 'C' or 'D'
      .partner_action         → 'C' or 'D'
      .payoff                 → this round's payoff
      .period                 → round number
      .subject_id             → subject identifier
      .partner_id             → partner identifier
    """
    # VCMS engine interface (required)
    contribution: int              # 1=cooperate, 0=defect
    others_mean: float             # partner cooperation: 1.0 or 0.0
    punishment_sent_total: int     # always 0
    punishment_received_total: int # always 0

    # IPD-specific
    own_action: str          # 'C' or 'D'
    partner_action: str      # 'C' or 'D'
    payoff: int              # this round's payoff
    period: int              # round number (1-indexed)
    subject_id: str = ''
    partner_id: str = ''

    @staticmethod
    def from_actions(own: str, partner: str, period: int,
                     subject_id: str = '', partner_id: str = '',
                     payoff_matrix: Optional[Dict] = None) -> 'IPDRound':
        """
        Create IPDRound from action strings.

        own, partner: 'C' or 'D' (case-insensitive)
        """
        own_c = own.strip().upper()
        partner_c = partner.strip().upper()

        own_binary = 1 if own_c == 'C' else 0
        partner_binary = 1.0 if partner_c == 'C' else 0.0

        # Compute payoff
        if payoff_matrix:
            payoff = payoff_matrix.get((own_c, partner_c), 0)
        else:
            if own_c == 'C' and partner_c == 'C':
                payoff = PAYOFF_R
            elif own_c == 'C' and partner_c == 'D':
                payoff = PAYOFF_S
            elif own_c == 'D' and partner_c == 'C':
                payoff = PAYOFF_T
            else:
                payoff = PAYOFF_P

        return IPDRound(
            contribution=own_binary,
            others_mean=partner_binary,
            punishment_sent_total=0,
            punishment_received_total=0,
            own_action=own_c,
            partner_action=partner_c,
            payoff=payoff,
            period=period,
            subject_id=subject_id,
            partner_id=partner_id,
        )


def load_ipd_csv(csv_path: str,
                 subject_col: str = 'subject_id',
                 round_col: str = 'round',
                 action_col: str = 'action',
                 partner_action_col: str = 'partner_action',
                 partner_id_col: Optional[str] = 'partner_id',
                 payoff_col: Optional[str] = 'payoff',
                 cooperate_value: str = 'C',
                 defect_value: str = 'D',
                 delimiter: str = ',',
                 ) -> Dict[str, List[IPDRound]]:
    """
    Load IPD data from CSV.

    Flexible column mapping to handle different experimental data formats.
    Returns dict of subject_id -> list of IPDRound (sorted by round).

    Common format expectations:
      subject_id, round, action, partner_action [, partner_id] [, payoff]

    Action values can be: 'C'/'D', 'cooperate'/'defect', 1/0, etc.
    Set cooperate_value/defect_value to match your data.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return {}

    # Normalize column names (strip whitespace, lowercase)
    # Build a mapping from our expected names to actual header names
    actual_headers = list(rows[0].keys())
    header_map = {}
    for expected in [subject_col, round_col, action_col, partner_action_col,
                     partner_id_col, payoff_col]:
        if expected is None:
            continue
        # Try exact match first, then case-insensitive
        if expected in actual_headers:
            header_map[expected] = expected
        else:
            for h in actual_headers:
                if h.strip().lower() == expected.strip().lower():
                    header_map[expected] = h
                    break

    def get_val(row, col):
        if col is None or col not in header_map:
            return None
        return row.get(header_map[col], '').strip()

    def parse_action(val):
        """Convert action value to 'C' or 'D'."""
        v = str(val).strip().upper()
        if v in ('C', 'COOPERATE', 'COOP', '1', 'TRUE'):
            return 'C'
        elif v in ('D', 'DEFECT', 'DEF', '0', 'FALSE'):
            return 'D'
        elif v == cooperate_value.upper():
            return 'C'
        elif v == defect_value.upper():
            return 'D'
        else:
            raise ValueError(f"Cannot parse action value: {val!r}")

    # Parse rows
    data = defaultdict(list)
    for row in rows:
        sid = get_val(row, subject_col)
        if not sid:
            continue

        try:
            period = int(get_val(row, round_col))
        except (ValueError, TypeError):
            continue

        own_action = parse_action(get_val(row, action_col))
        partner_action_val = get_val(row, partner_action_col)
        if partner_action_val is None:
            continue
        partner_action = parse_action(partner_action_val)

        pid = get_val(row, partner_id_col) or ''

        payoff_val = get_val(row, payoff_col)
        custom_payoff = None
        if payoff_val:
            try:
                custom_payoff = {
                    (own_action, partner_action): int(float(payoff_val))
                }
            except (ValueError, TypeError):
                custom_payoff = None

        rd = IPDRound.from_actions(
            own=own_action,
            partner=partner_action,
            period=period,
            subject_id=sid,
            partner_id=pid,
            payoff_matrix=custom_payoff,
        )
        data[sid].append(rd)

    # Sort by period
    result = {}
    for sid in sorted(data.keys()):
        result[sid] = sorted(data[sid], key=lambda r: r.period)

    return result


def load_ipd_action_matrix(csv_path: str,
                           cooperate_value: str = 'C') -> Dict[str, List[IPDRound]]:
    """
    Load IPD data from a paired action matrix format.

    Expected format: each row is a round with paired actions.
    Columns: round, player1_action, player2_action [, player1_payoff, player2_payoff]

    Returns data for both players (subject IDs = 'player1', 'player2').
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    headers = list(rows[0].keys())

    # Try to identify columns
    round_col = None
    p1_col = None
    p2_col = None
    for h in headers:
        hl = h.lower().strip()
        if hl in ('round', 'period', 't', 'trial'):
            round_col = h
        elif hl in ('player1', 'p1', 'player1_action', 'p1_action', 'action1'):
            p1_col = h
        elif hl in ('player2', 'p2', 'player2_action', 'p2_action', 'action2'):
            p2_col = h

    if not all([round_col, p1_col, p2_col]):
        raise ValueError(f"Cannot identify columns. Headers: {headers}")

    data = {'player1': [], 'player2': []}
    for row in rows:
        try:
            period = int(row[round_col])
        except (ValueError, TypeError):
            continue

        p1_action = 'C' if str(row[p1_col]).strip().upper() in ('C', 'COOPERATE', '1') else 'D'
        p2_action = 'C' if str(row[p2_col]).strip().upper() in ('C', 'COOPERATE', '1') else 'D'

        data['player1'].append(IPDRound.from_actions(
            own=p1_action, partner=p2_action, period=period,
            subject_id='player1', partner_id='player2'))
        data['player2'].append(IPDRound.from_actions(
            own=p2_action, partner=p1_action, period=period,
            subject_id='player2', partner_id='player1'))

    for sid in data:
        data[sid].sort(key=lambda r: r.period)

    return data


def load_ipd_experiment(csv_path: str) -> Dict[str, List[IPDRound]]:
    """
    Load IPD experimental data (IPD-rand.csv or fix.csv format).

    Convenience wrapper with column mapping for the semicolon-delimited
    experimental data files.
    """
    return load_ipd_csv(
        csv_path,
        subject_col='player',
        round_col='round',
        action_col='action_player',
        partner_action_col='action_opponent',
        partner_id_col='opponent',
        payoff_col='payoff',
        delimiter=';',
    )


# ================================================================
# IPD BASELINES
# ================================================================

def tit_for_tat(rounds: List[IPDRound]) -> List[int]:
    """TFT: cooperate on round 1, then copy partner's last action."""
    preds = []
    for i, rd in enumerate(rounds):
        if i == 0:
            preds.append(1)  # cooperate first
        else:
            preds.append(1 if rounds[i - 1].partner_action == 'C' else 0)
    return preds


def win_stay_lose_shift(rounds: List[IPDRound]) -> List[int]:
    """WSLS/Pavlov: if last payoff was good (R or T), repeat; else switch."""
    preds = []
    for i, rd in enumerate(rounds):
        if i == 0:
            preds.append(1)  # cooperate first
        else:
            last_payoff = rounds[i - 1].payoff
            if last_payoff >= PAYOFF_R:  # R=3 or T=4 → good outcome
                preds.append(rounds[i - 1].contribution)  # stay
            else:  # S=0 or P=1 → bad outcome
                preds.append(1 - rounds[i - 1].contribution)  # shift
    return preds


def always_cooperate(rounds: List[IPDRound]) -> List[int]:
    """Always cooperate."""
    return [1] * len(rounds)


def always_defect(rounds: List[IPDRound]) -> List[int]:
    """Always defect."""
    return [0] * len(rounds)


def carry_forward_ipd(rounds: List[IPDRound]) -> List[int]:
    """Predict current action = last action. Round 1 = actual."""
    preds = []
    for i, rd in enumerate(rounds):
        if i == 0:
            preds.append(rd.contribution)
        else:
            preds.append(rounds[i - 1].contribution)
    return preds


# ================================================================
# IPD METRICS
# ================================================================

def ipd_accuracy(pred: List[int], actual: List[int],
                 from_round: int = 0) -> float:
    """Fraction of correct predictions from from_round onwards."""
    correct = 0
    total = 0
    for i in range(from_round, min(len(pred), len(actual))):
        if pred[i] == actual[i]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def ipd_summary(data: Dict[str, List[IPDRound]]):
    """Print summary statistics for IPD dataset."""
    import numpy as np

    print(f"\nIPD DATASET SUMMARY — {len(data)} subjects")
    print(f"{'=' * 60}")

    all_coop_rates = []
    all_rounds = []
    partner_changes = defaultdict(int)

    for sid in sorted(data.keys()):
        rounds = data[sid]
        n_rounds = len(rounds)
        all_rounds.append(n_rounds)

        coop_rate = sum(r.contribution for r in rounds) / n_rounds
        all_coop_rates.append(coop_rate)

        partners = set(r.partner_id for r in rounds)
        partner_changes[len(partners)] += 1

        actions = ''.join(r.own_action for r in rounds)
        partner_actions = ''.join(r.partner_action for r in rounds)

        print(f"  {sid:>8s}: {n_rounds:>3d} rounds, "
              f"coop={coop_rate:.0%}, "
              f"own={actions[:20]}{'...' if len(actions) > 20 else ''}, "
              f"partner={partner_actions[:20]}{'...' if len(partner_actions) > 20 else ''}")

    print(f"\n  Rounds per subject: {min(all_rounds)}-{max(all_rounds)} "
          f"(mean={np.mean(all_rounds):.0f})")
    print(f"  Cooperation rate: mean={np.mean(all_coop_rates):.0%}, "
          f"median={np.median(all_coop_rates):.0%}")
    print(f"  Partner counts: {dict(partner_changes)}")

    # Classify strategies
    n_tft = 0
    n_wsls = 0
    n_allc = 0
    n_alld = 0
    for sid, rounds in data.items():
        if len(rounds) < 3:
            continue
        tft_pred = tit_for_tat(rounds)
        wsls_pred = win_stay_lose_shift(rounds)
        actual = [r.contribution for r in rounds]

        tft_acc = ipd_accuracy(tft_pred, actual, from_round=1)
        wsls_acc = ipd_accuracy(wsls_pred, actual, from_round=1)
        coop_rate = sum(actual) / len(actual)

        if coop_rate > 0.95:
            n_allc += 1
        elif coop_rate < 0.05:
            n_alld += 1
        elif tft_acc > 0.9:
            n_tft += 1
        elif wsls_acc > 0.9:
            n_wsls += 1

    print(f"\n  Strategy approximations (>90% accuracy):")
    print(f"    TFT-like:  {n_tft}")
    print(f"    WSLS-like: {n_wsls}")
    print(f"    All-C:     {n_allc}")
    print(f"    All-D:     {n_alld}")
    print(f"    Other:     {len(data) - n_tft - n_wsls - n_allc - n_alld}")
