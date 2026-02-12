"""
PGG P-Experiment Data Loader
=============================
Loads Herrmann et al. punishment experiment data.

Each round has 3 rows (one per partner). This loader collapses them into
per-round records with:
  - contribution: subject's contribution (prosocial spend)
  - others_individual: list of 3 partner contributions
  - others_mean: mean of partner contributions
  - punishment_sent: list of 3 punishment amounts (to each partner)
  - punishment_sent_total: sum of punishment sent (control spend)
  - punishment_received_total: total punishment received from all partners
  - total_voluntary_spend: contribution + punishment_sent_total
  - antisocial_punishment: punishment sent to partners who contributed >= subject
  - prosocial_punishment: punishment sent to partners who contributed < subject

Also computes per-partner targeting features:
  - punishment vs relative contribution (did subject punish up or down?)

Usage:
    from pgg_p_loader import load_p_experiment, PRoundData
    data = load_p_experiment('path/to/csv')
    for sid, rounds in data.items():
        for rd in rounds:
            print(rd.contribution, rd.punishment_sent_total, rd.punishment_received_total)
"""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PPartnerInteraction:
    """Single partner interaction within a round."""
    partner_contribution: int
    punishment_sent: int      # punishment I sent to this partner
    relative_position: float  # partner_contrib - my_contrib (positive = they gave more)


@dataclass 
class PRoundData:
    """One round of P-experiment data for a single subject."""
    subject_id: str
    group_id: str
    period: int
    contribution: int         # prosocial spend

    # Partner details
    others_individual: List[int]           # each partner's contribution
    others_mean: float                     # mean of partner contributions
    partner_interactions: List[PPartnerInteraction]  # per-partner detail

    # Punishment aggregates
    punishment_sent: List[int]             # punishment to each partner
    punishment_sent_total: int             # sum = control spend
    punishment_received_total: int         # damage taken from all partners

    # Derived
    total_voluntary_spend: int             # contribution + punishment_sent_total
    antisocial_punishment: int             # punishment sent to partners who contributed >= me
    prosocial_punishment: int              # punishment sent to partners who contributed < me

    # For N-experiment compatibility
    @property
    def others(self):
        """Mean others contribution, matching N-experiment RoundData interface."""
        return self.others_mean


def load_p_experiment(csv_path: str) -> Dict[str, List[PRoundData]]:
    """
    Load P-experiment CSV and return dict of subject_id -> list of PRoundData.

    Handles:
      - Citation row at top of file
      - 3 rows per subject per round (one per partner)
      - Collapsing into single round records with aggregated punishment
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = None
        rows = []
        for row in reader:
            if not row or not row[0]:
                continue
            if 'DATA from' in row[0]:
                continue
            if row[0] == 'sessionid':
                header = row
                continue
            if header and len(row) >= len(header):
                rows.append(row)

    if not header:
        raise ValueError(f"No header row found in {csv_path}")

    # Column indices
    sid_idx = header.index('subjectid')
    per_idx = header.index('period')
    con_idx = header.index('senderscontribution')
    oth_idx = header.index('otherscontribution')
    pun_idx = header.index('punishment')
    rec_idx = header.index('recpun')
    gid_idx = header.index('groupid')

    # Group rows by (subject, period)
    raw = defaultdict(lambda: defaultdict(list))
    groups = {}
    for row in rows:
        sid = row[sid_idx]
        try:
            period = int(row[per_idx])
        except ValueError:
            continue
        raw[sid][period].append(row)
        groups[sid] = row[gid_idx]

    # Build PRoundData for each subject
    result = {}
    for sid in sorted(raw.keys()):
        rounds = []
        for period in sorted(raw[sid].keys()):
            partner_rows = raw[sid][period]

            # Extract contribution (same across all rows in this round)
            contribution = int(partner_rows[0][con_idx])

            # Extract per-partner data
            others_individual = []
            punishment_sent = []
            for pr in partner_rows:
                others_individual.append(int(pr[oth_idx]))
                pun_val = pr[pun_idx]
                punishment_sent.append(int(pun_val) if pun_val else 0)

            # Punishment received (same across all rows)
            recpun_val = partner_rows[0][rec_idx]
            punishment_received_total = int(recpun_val) if recpun_val else 0

            # Compute aggregates
            others_mean = sum(others_individual) / len(others_individual) if others_individual else 0.0
            punishment_sent_total = sum(punishment_sent)
            total_voluntary_spend = contribution + punishment_sent_total

            # Classify punishment targeting
            antisocial = 0  # punishing those who contributed >= me
            prosocial = 0   # punishing those who contributed < me
            partner_interactions = []
            for i, (oc, ps) in enumerate(zip(others_individual, punishment_sent)):
                relative = oc - contribution  # positive = partner gave more
                if ps > 0:
                    if oc >= contribution:
                        antisocial += ps
                    else:
                        prosocial += ps
                partner_interactions.append(PPartnerInteraction(
                    partner_contribution=oc,
                    punishment_sent=ps,
                    relative_position=relative,
                ))

            rounds.append(PRoundData(
                subject_id=sid,
                group_id=groups[sid],
                period=period,
                contribution=contribution,
                others_individual=others_individual,
                others_mean=others_mean,
                partner_interactions=partner_interactions,
                punishment_sent=punishment_sent,
                punishment_sent_total=punishment_sent_total,
                punishment_received_total=punishment_received_total,
                total_voluntary_spend=total_voluntary_spend,
                antisocial_punishment=antisocial,
                prosocial_punishment=prosocial,
            ))

        result[sid] = rounds

    return result


def print_p_summary(data: Dict[str, List[PRoundData]]):
    """Print comprehensive summary of P-experiment data."""
    import numpy as np

    print(f"\nP-EXPERIMENT SUMMARY — {len(data)} subjects")
    print(f"{'='*90}")

    all_contrib = []
    all_pun_sent = []
    all_pun_recv = []
    all_antisocial = []
    all_prosocial = []
    all_tvs = []

    print(f"\n{'SID':>5s} {'Contributions':>40s}  {'Pun Sent Total':>40s}  {'Pun Recv':>30s}")
    print(f"{'-'*120}")

    for sid in sorted(data.keys()):
        rounds = data[sid]
        contribs = [r.contribution for r in rounds]
        pun_sent = [r.punishment_sent_total for r in rounds]
        pun_recv = [r.punishment_received_total for r in rounds]
        tvs = [r.total_voluntary_spend for r in rounds]

        all_contrib.extend(contribs)
        all_pun_sent.extend(pun_sent)
        all_pun_recv.extend(pun_recv)
        all_tvs.extend(tvs)
        all_antisocial.extend([r.antisocial_punishment for r in rounds])
        all_prosocial.extend([r.prosocial_punishment for r in rounds])

        print(f"{sid:>5s} C={str(contribs):>40s}  PS={str(pun_sent):>38s}  PR={str(pun_recv)}")

    print(f"\nAGGREGATE STATISTICS:")
    print(f"  Contribution:     mean={np.mean(all_contrib):.1f}, median={np.median(all_contrib):.0f}")
    print(f"  Punishment sent:  mean={np.mean(all_pun_sent):.1f}, median={np.median(all_pun_sent):.0f}, "
          f"max={max(all_pun_sent)}")
    print(f"  Punishment recv:  mean={np.mean(all_pun_recv):.1f}, median={np.median(all_pun_recv):.0f}, "
          f"max={max(all_pun_recv)}")
    print(f"  Total vol spend:  mean={np.mean(all_tvs):.1f}, median={np.median(all_tvs):.0f}")
    print(f"  Antisocial pun:   mean={np.mean(all_antisocial):.2f} "
          f"({sum(1 for a in all_antisocial if a > 0)}/{len(all_antisocial)} rounds)")
    print(f"  Prosocial pun:    mean={np.mean(all_prosocial):.2f} "
          f"({sum(1 for a in all_prosocial if a > 0)}/{len(all_prosocial)} rounds)")

    # Per-subject punishment profiles
    print(f"\nPUNISHMENT PROFILES:")
    print(f"{'SID':>5s} {'MeanC':>6s} {'MeanPS':>7s} {'MeanPR':>7s} {'MeanTVS':>8s} "
          f"{'AntiS':>6s} {'ProS':>6s} {'Ratio':>7s}  Profile")
    print(f"{'-'*75}")

    for sid in sorted(data.keys()):
        rounds = data[sid]
        mc = np.mean([r.contribution for r in rounds])
        mps = np.mean([r.punishment_sent_total for r in rounds])
        mpr = np.mean([r.punishment_received_total for r in rounds])
        mtvs = np.mean([r.total_voluntary_spend for r in rounds])
        anti = sum(r.antisocial_punishment for r in rounds)
        pro = sum(r.prosocial_punishment for r in rounds)
        total_pun = anti + pro
        ratio = anti / total_pun if total_pun > 0 else 0.0

        # Classify
        if mps < 0.5 and mc > 10:
            profile = "cooperator"
        elif mps < 0.5 and mc <= 5:
            profile = "free-rider"
        elif mps >= 0.5 and mc > 10:
            profile = "cooperative-enforcer"
        elif mps >= 0.5 and mc <= 5 and ratio > 0.5:
            profile = "ANTISOCIAL-CONTROLLER"
        elif mps >= 0.5 and mc <= 5:
            profile = "punitive-free-rider"
        else:
            profile = "mixed"

        print(f"{sid:>5s} {mc:>6.1f} {mps:>7.1f} {mpr:>7.1f} {mtvs:>8.1f} "
              f"{anti:>6d} {pro:>6d} {ratio:>7.2f}  {profile}")

    # Targeting analysis: when subjects punish, who do they punish?
    print(f"\nTARGETING ANALYSIS:")
    upward_pun = 0    # punishing someone who contributed more
    downward_pun = 0  # punishing someone who contributed less
    equal_pun = 0     # punishing someone who contributed same
    for sid, rounds in data.items():
        for r in rounds:
            for pi in r.partner_interactions:
                if pi.punishment_sent > 0:
                    if pi.relative_position > 0:
                        upward_pun += pi.punishment_sent
                    elif pi.relative_position < 0:
                        downward_pun += pi.punishment_sent
                    else:
                        equal_pun += pi.punishment_sent

    total = upward_pun + downward_pun + equal_pun
    if total > 0:
        print(f"  Upward (antisocial):   {upward_pun:>4d} tokens ({upward_pun/total*100:.1f}%)")
        print(f"  Downward (prosocial):  {downward_pun:>4d} tokens ({downward_pun/total*100:.1f}%)")
        print(f"  Equal:                 {equal_pun:>4d} tokens ({equal_pun/total*100:.1f}%)")


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'HerrmannThoeniGaechterDATA_SAMARA_P-EXPERIMENT_TRUNCATED_SESSION1.csv'
    data = load_p_experiment(path)
    print_p_summary(data)
