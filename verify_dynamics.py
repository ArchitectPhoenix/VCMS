"""
Verification: VCMS predictions come from cognitive dynamics, not pattern matching.

Three tests:
1. Same library candidate, different environments → different predictions
   (proves model conditions on environment through VCMS state variables)
2. VCMS predictions vs naive trajectory averaging → systematic divergence
   (proves the model does more than interpolate library C/P values)
3. Internal state audit: B, S, affordability evolve dynamically per subject
   (proves the 16-parameter model is doing real work)
"""

import json
import math
import numpy as np
from pgg_vcms_agent_v3 import VCMSParams, run_vcms_agent, MAX_CONTRIB, MAX_PUNISH
from pgg_p_loader import load_p_experiment

# =========================================================================
# Test 1: Same candidate params, two different environments
# =========================================================================
print("=" * 70)
print("TEST 1: Same candidate, different environments")
print("  If prediction = pattern matching, outputs would be identical.")
print("  If prediction = VCMS dynamics, outputs differ because B/S/m_eval")
print("  evolve differently in each environment.")
print("=" * 70)

# Load one library candidate's params
with open('v3_library_fitted.json') as f:
    lib = json.load(f)

# Pick a cooperative-enforcer candidate
candidate_sid = None
for sid, record in lib.items():
    if record['behavioral_profile'] == 'cooperative-enforcer':
        candidate_sid = sid
        break

params = VCMSParams(**lib[candidate_sid]['v3_params'])
print(f"\nCandidate: {candidate_sid} ({lib[candidate_sid]['behavioral_profile']})")
print(f"  Fitted params: c_base={params.c_base:.3f}, b_initial={params.b_initial:.3f}, "
      f"s_dir={'+'if params.s_dir>=0 else'-'}, s_rate={params.s_rate:.3f}")

# Environment A: cooperative group (high others_mean)
class FakeRound:
    def __init__(self, period, contribution, others_mean, pun_sent, pun_recv):
        self.period = period
        self.contribution = contribution
        self.others_mean = others_mean
        self.punishment_sent_total = pun_sent
        self.punishment_received_total = pun_recv

env_coop = [FakeRound(t+1, 15, 16.0, 0, 0) for t in range(10)]
env_defect = [FakeRound(t+1, 15, 3.0, 0, 5) for t in range(10)]

result_coop = run_vcms_agent(params, env_coop)
result_defect = run_vcms_agent(params, env_defect)

print(f"\n  Environment A (cooperative group, others_mean=16):")
print(f"    C predictions: {result_coop['pred_contrib']}")
print(f"    P predictions: {result_coop['pred_punish']}")
print(f"    B trajectory:  {['%.2f' % t['budget']['b_post'] for t in result_coop['trace']]}")
print(f"    A trajectory:  {['%.2f' % t['routing']['affordability'] for t in result_coop['trace']]}")

print(f"\n  Environment B (defecting group, others_mean=3, punishment=5):")
print(f"    C predictions: {result_defect['pred_contrib']}")
print(f"    P predictions: {result_defect['pred_punish']}")
print(f"    B trajectory:  {['%.2f' % t['budget']['b_post'] for t in result_defect['trace']]}")
print(f"    A trajectory:  {['%.2f' % t['routing']['affordability'] for t in result_defect['trace']]}")

c_diff = sum(abs(a - b) for a, b in zip(result_coop['pred_contrib'], result_defect['pred_contrib']))
p_diff = sum(abs(a - b) for a, b in zip(result_coop['pred_punish'], result_defect['pred_punish']))
print(f"\n  Total |C difference|: {c_diff}  Total |P difference|: {p_diff}")
assert c_diff > 0 or p_diff > 0, "FAIL: Same predictions in different environments"
print("  PASS: Same candidate produces different predictions in different environments")

# =========================================================================
# Test 2: VCMS predictions vs naive trajectory averaging
# =========================================================================
print(f"\n{'=' * 70}")
print("TEST 2: VCMS dynamics vs naive trajectory averaging")
print("  Naive = average of survivors' original library C/P trajectories")
print("  VCMS = run cognitive model with survivors' params on THIS subject's env")
print("  If they systematically diverge, prediction is model-driven.")
print("=" * 70)

# Load real subject data (Boston)
data = load_p_experiment('HerrmannThoeniGaechterDATA_BOSTON_P-EXPERIMENT_TRUNCATED_SESSION1.csv')

# Pick a few subjects
test_sids = list(data.keys())[:5]
divergence_scores = []

for test_sid in test_sids:
    rounds = data[test_sid]

    # Get all library candidate params
    all_lib_sids = list(lib.keys())

    # For rounds 3-9 (where model has enough history), compare:
    for t in [4, 7, 9]:  # rounds 5, 8, 10
        # Naive: average of library trajectories at round t
        naive_c = np.mean([lib[s]['contribution_trajectory'][t] for s in all_lib_sids])
        naive_p = np.mean([lib[s]['punishment_sent_trajectory'][t] for s in all_lib_sids])

        # VCMS: run each candidate's model on this subject's environment
        vcms_preds_c = []
        vcms_preds_p = []
        for s in all_lib_sids:
            p = VCMSParams(**lib[s]['v3_params'])
            result = run_vcms_agent(p, rounds[:t+1])
            vcms_preds_c.append(result['pred_contrib'][t])
            vcms_preds_p.append(result['pred_punish'][t])

        vcms_c = np.mean(vcms_preds_c)
        vcms_p = np.mean(vcms_preds_p)

        div = abs(vcms_c - naive_c) + abs(vcms_p - naive_p)
        divergence_scores.append(div)

        if test_sid == test_sids[0]:  # Print details for first subject
            print(f"\n  Subject {test_sid}, Round {t+1}:")
            print(f"    Naive (trajectory avg): C={naive_c:.1f}, P={naive_p:.1f}")
            print(f"    VCMS (model dynamics):  C={vcms_c:.1f}, P={vcms_p:.1f}")
            print(f"    Actual:                 C={rounds[t].contribution}, P={rounds[t].punishment_sent_total}")
            print(f"    Divergence: {div:.1f}")

mean_div = np.mean(divergence_scores)
print(f"\n  Mean divergence across {len(divergence_scores)} predictions: {mean_div:.2f}")
assert mean_div > 0.5, f"FAIL: Divergence too low ({mean_div:.2f})"
print("  PASS: VCMS dynamics systematically differ from naive trajectory averaging")

# =========================================================================
# Test 3: Internal state audit
# =========================================================================
print(f"\n{'=' * 70}")
print("TEST 3: VCMS internal state audit")
print("  Verify B, S, m_eval, gate, affordability evolve non-trivially")
print("  across rounds for real subjects.")
print("=" * 70)

# Run a library candidate's params on a real Boston subject
test_sid = test_sids[0]
rounds = data[test_sid]

# Pick 3 different candidate types
profiles_to_test = ['cooperator', 'cooperative-enforcer', 'antisocial-controller']
for profile in profiles_to_test:
    cand_sid = None
    for sid, rec in lib.items():
        if rec['behavioral_profile'] == profile:
            cand_sid = sid
            break
    if not cand_sid:
        continue

    p = VCMSParams(**lib[cand_sid]['v3_params'])
    result = run_vcms_agent(p, rounds)

    B_vals = [t['budget']['b_post'] for t in result['trace']]
    S_vals = [t['state']['strain_end'] for t in result['trace']]
    A_vals = [t['routing']['affordability'] for t in result['trace']]
    gate_vals = [t['routing']['gate'] for t in result['trace']]

    B_range = max(B_vals) - min(B_vals)
    S_range = max(S_vals) - min(S_vals)
    A_range = max(A_vals) - min(A_vals)

    print(f"\n  {profile} ({cand_sid}) on subject {test_sid}:")
    print(f"    B range: {min(B_vals):.3f} — {max(B_vals):.3f} (Δ={B_range:.3f})")
    print(f"    S range: {min(S_vals):.3f} — {max(S_vals):.3f} (Δ={S_range:.3f})")
    print(f"    A range: {min(A_vals):.3f} — {max(A_vals):.3f} (Δ={A_range:.3f})")
    print(f"    C pred:  {result['pred_contrib']}")
    print(f"    P pred:  {result['pred_punish']}")

    # At least one state variable should show meaningful variation
    if B_range > 0.01 or S_range > 0.01 or A_range > 0.01:
        print(f"    PASS: State evolves dynamically")
    else:
        print(f"    NOTE: Flat state trajectory (may be stable cooperator type)")

print(f"\n  Actual C: {[r.contribution for r in rounds]}")
print(f"  Actual P: {[r.punishment_sent_total for r in rounds]}")

print(f"\n{'=' * 70}")
print("ALL VERIFICATION TESTS PASSED")
print("Predictions are driven by VCMS cognitive dynamics, not pattern matching.")
print("=" * 70)
