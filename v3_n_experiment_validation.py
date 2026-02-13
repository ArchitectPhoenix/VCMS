#!/usr/bin/env python3
"""
N-experiment theoretical validation: comfortable defectors, s_rate correction,
strain habituation.

Tests three predictions from the theory analysis:

1. COMFORTABLE DEFECTOR BASIN: Stable-low N-experiment subjects are in a
   low-strain equilibrium where v_ref tracks down to match the decaying group,
   closing the gap → low S → stable B → stable low C. The P-experiment never
   produces this because punishment injects S externally.

2. 1.5x S_RATE CORRECTION: The P-experiment discharge pathway (punishment)
   was doing ~1/3 of the strain resolution work. Without it, effective s_rate
   should be ~1.5x higher. Test: scale s_rate for all library candidates and
   rerun declining subjects.

3. STRAIN HABITUATION: The model biases toward convex (accelerating collapse)
   for linear decliners because the B-depletion feedback loop has no damping.
   Hypothesize that repeated exposure to the same gap reduces its effective
   S contribution. Test: add exponential decay on gap → S contribution when
   gap direction is consistent, see if linear decliners improve.
"""

import json
import math
import copy
import numpy as np
from collections import defaultdict

from pgg_p_loader import PRoundData
from pgg_vcms_agent_v3 import (
    VCMSParams, PARAM_NAMES, run_vcms_agent, MAX_CONTRIB,
    MAX_PUNISH, TOTAL_ROUNDS, ANCHOR_RATE, B_NOISE, ACUTE_MULT, EPS,
    _sigmoid, _horizon_factor,
)
from v3_n_experiment_test import (
    load_n_experiment_data, load_library,
    vcms_predict, carry_forward_predict, rmse_window,
)
from v3_n_experiment_diagnostics import (
    classify_curvature, compute_slope, vcms_predict_with_survivors,
)


# ================================================================
# TEST 1: COMFORTABLE DEFECTOR BASIN DYNAMICS
# ================================================================

def validate_comfortable_defector(n_rounds, library, city_map):
    """
    Trace the internal dynamics for stable-low N-experiment subjects.
    Prediction: they're in a low-strain, stable-B equilibrium where
    v_ref has tracked down to match the declining group.
    """
    print("\n" + "=" * 72)
    print("  TEST 1: COMFORTABLE DEFECTOR BASIN DYNAMICS")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    stable_low = []
    declining = []
    stable_high = []

    for sid in sids:
        actual = [r.contribution for r in n_rounds[sid]]
        slope = compute_slope(actual)
        mean_c = np.mean(actual)
        if slope >= -0.3 and mean_c <= 10:
            stable_low.append(sid)
        elif slope < -0.3:
            declining.append(sid)
        elif slope >= -0.3 and mean_c > 10:
            stable_high.append(sid)

    print(f"\n  Stable-low: {len(stable_low)}, Declining: {len(declining)}, "
          f"Stable-high: {len(stable_high)}")

    # For each stable-low subject, find the best-matching library candidate
    # and trace the internal dynamics
    sl_traces = {'strain': [], 'B': [], 'affordability': [], 'gap': [],
                 'v_level': [], 'reference': [], 'contribution': [],
                 'others_mean': []}
    decl_traces = {'strain': [], 'B': [], 'affordability': [], 'gap': [],
                   'v_level': [], 'reference': [], 'contribution': [],
                   'others_mean': []}

    for label, group, traces in [('Stable-low', stable_low, sl_traces),
                                  ('Declining', declining[:40], decl_traces)]:
        for sid in group:
            rounds = n_rounds[sid]
            _, survivors, distances = vcms_predict_with_survivors(rounds, library)
            if not survivors:
                continue
            best_sid = min(distances, key=distances.get)
            best_params = VCMSParams(**library[best_sid]['v3_params'])

            result = run_vcms_agent(best_params, rounds)
            trace = result['trace']

            for key_trace, accessor in [
                ('strain', lambda t: t['s_accum']['strain_pre_discharge']),
                ('B', lambda t: t['budget']['b_post']),
                ('affordability', lambda t: t['routing']['affordability']),
                ('gap', lambda t: t['s_accum']['gap']),
                ('v_level', lambda t: t['v']['v_level']),
                ('reference', lambda t: t['v']['reference']),
                ('contribution', lambda t: t['c_output']['actual_c']),
                ('others_mean', lambda t: t['v']['v_group']),
            ]:
                vals = [accessor(t) for t in trace]
                traces[key_trace].append(vals)

    # Average across subjects per round
    print(f"\n  AVERAGE INTERNAL DYNAMICS (per round, best-match candidate)")
    print(f"\n  {'':>5s} {'--- Stable-low (n=' + str(len(stable_low)) + ') ---':>45s} "
          f"{'--- Declining (n=40) ---':>40s}")
    print(f"  {'Rnd':>5s} {'S':>7s} {'B':>7s} {'Aff':>7s} {'Gap':>7s} {'Ref':>7s} "
          f"{'|':>3s} {'S':>7s} {'B':>7s} {'Aff':>7s} {'Gap':>7s} {'Ref':>7s}")
    print("  " + "-" * 80)

    for t in range(10):
        sl_s = np.mean([x[t] for x in sl_traces['strain'] if len(x) > t])
        sl_b = np.mean([x[t] for x in sl_traces['B'] if len(x) > t])
        sl_a = np.mean([x[t] for x in sl_traces['affordability'] if len(x) > t])
        sl_g = np.mean([x[t] for x in sl_traces['gap'] if len(x) > t])
        sl_r = np.mean([x[t] for x in sl_traces['reference'] if len(x) > t])

        d_s = np.mean([x[t] for x in decl_traces['strain'] if len(x) > t])
        d_b = np.mean([x[t] for x in decl_traces['B'] if len(x) > t])
        d_a = np.mean([x[t] for x in decl_traces['affordability'] if len(x) > t])
        d_g = np.mean([x[t] for x in decl_traces['gap'] if len(x) > t])
        d_r = np.mean([x[t] for x in decl_traces['reference'] if len(x) > t])

        print(f"  R{t+1:>3d} {sl_s:>7.3f} {sl_b:>7.3f} {sl_a:>7.3f} {sl_g:>+7.3f} {sl_r:>7.3f} "
              f"{'|':>3s} {d_s:>7.3f} {d_b:>7.3f} {d_a:>7.3f} {d_g:>+7.3f} {d_r:>7.3f}")

    # Contribution trajectories
    print(f"\n  ACTUAL CONTRIBUTION TRAJECTORIES (mean per round):")
    print(f"  {'Rnd':>5s} {'SL actual':>10s} {'SL others':>10s} "
          f"{'Decl actual':>12s} {'Decl others':>12s}")
    for t in range(10):
        sl_c = np.mean([x[t] for x in sl_traces['contribution'] if len(x) > t])
        sl_o = np.mean([x[t] for x in sl_traces['others_mean'] if len(x) > t])
        d_c = np.mean([x[t] for x in decl_traces['contribution'] if len(x) > t])
        d_o = np.mean([x[t] for x in decl_traces['others_mean'] if len(x) > t])
        print(f"  R{t+1:>3d} {sl_c:>10.1f} {sl_o:>10.3f} {d_c:>12.1f} {d_o:>12.3f}")

    # Key diagnostic: what's the gap between own contribution and others' mean
    # for stable-low at equilibrium (rounds 5-10)?
    late_gaps_sl = []
    late_gaps_decl = []
    for x in sl_traces['gap']:
        late_gaps_sl.extend(x[4:10])
    for x in decl_traces['gap']:
        late_gaps_decl.extend(x[4:10])

    print(f"\n  LATE-GAME GAP (rounds 5-10):")
    print(f"    Stable-low: mean={np.mean(late_gaps_sl):+.4f}, std={np.std(late_gaps_sl):.4f}")
    print(f"    Declining:  mean={np.mean(late_gaps_decl):+.4f}, std={np.std(late_gaps_decl):.4f}")
    print(f"    → Stable-low gap is {'smaller' if abs(np.mean(late_gaps_sl)) < abs(np.mean(late_gaps_decl)) else 'larger'} "
          f"(ratio: {abs(np.mean(late_gaps_sl)) / (abs(np.mean(late_gaps_decl)) + 1e-6):.2f}x)")


# ================================================================
# TEST 2: 1.5x S_RATE CORRECTION
# ================================================================

def run_vcms_agent_scaled(params, rounds, s_rate_mult=1.0):
    """Run VCMS agent with scaled s_rate."""
    scaled_params = VCMSParams(
        c_base=params.c_base, alpha=params.alpha, v_rep=params.v_rep,
        v_ref=params.v_ref, inertia=params.inertia,
        s_dir=params.s_dir, s_rate=params.s_rate * s_rate_mult,
        s_initial=params.s_initial,
        s_frac=params.s_frac, p_scale=params.p_scale, s_thresh=params.s_thresh,
        b_initial=params.b_initial, b_depletion_rate=params.b_depletion_rate,
        b_replenish_rate=params.b_replenish_rate,
        acute_threshold=params.acute_threshold,
        facilitation_rate=params.facilitation_rate,
        h_strength=params.h_strength, h_start=params.h_start,
    )
    return run_vcms_agent(scaled_params, rounds)


def vcms_predict_scaled(target_rounds, library, s_rate_mult=1.0):
    """VCMS ensemble with scaled s_rate for all candidates."""
    candidates = {}
    for sid, rec in library.items():
        candidates[sid] = VCMSParams(**rec['v3_params'])

    survivors = list(candidates.keys())
    pred_c_list = []
    obs_c_list = []
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}

    n = min(10, len(target_rounds))
    for t in range(n):
        cand_preds_c = {}
        for sid in survivors:
            result = run_vcms_agent_scaled(
                candidates[sid], target_rounds[:t + 1], s_rate_mult)
            cand_preds_c[sid] = result['pred_contrib'][t]
            cand_pred_history[sid] = list(result['pred_contrib'][:t + 1])

        if not distances:
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001)
                       for sid in survivors}
        total_w = sum(weights.values())

        pc = sum((weights[sid] / total_w) * cand_preds_c[sid] for sid in survivors)
        pred_c_list.append(int(round(pc)))
        obs_c_list.append(target_rounds[t].contribution)

        new_distances = {}
        for sid in survivors:
            hist = cand_pred_history[sid]
            nc = min(t + 1, len(hist))
            if nc == 0:
                new_distances[sid] = 0.0
                continue
            c_d = sum((obs_c_list[i] - hist[i]) ** 2
                      for i in range(nc)) / (MAX_CONTRIB ** 2 * nc)
            new_distances[sid] = math.sqrt(c_d)

        if new_distances:
            best = min(new_distances.values())
            thresh = max(best * 3.0, 0.5)
        else:
            thresh = 0.5

        new_surv = [s for s in survivors if new_distances.get(s, 999) <= thresh]
        if not new_surv and new_distances:
            new_surv = [min(new_distances, key=new_distances.get)]

        survivors = new_surv
        distances = {s: new_distances[s] for s in survivors}

    return pred_c_list


def test_srate_correction(n_rounds, library, city_map):
    """Test whether 1.5x s_rate correction improves declining-subject predictions."""
    print("\n" + "=" * 72)
    print("  TEST 2: S_RATE CORRECTION SWEEP")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    declining = [sid for sid in sids
                 if compute_slope([r.contribution for r in n_rounds[sid]]) < -0.3]
    nondecl = [sid for sid in sids if sid not in declining]

    print(f"\n  Declining: {len(declining)}, Non-declining: {len(nondecl)}")

    multipliers = [1.0, 1.25, 1.5, 1.75, 2.0]

    print(f"\n  Sweeping s_rate multipliers: {multipliers}")
    print(f"  (This tests on ALL subjects — declining and non-declining)")

    results = {m: {'declining': [], 'nondecl': [], 'all': []} for m in multipliers}

    import time
    t0 = time.time()

    for idx, sid in enumerate(sids):
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]
        is_decl = sid in declining

        for mult in multipliers:
            preds = vcms_predict_scaled(rounds, library, s_rate_mult=mult)
            err = rmse_window(preds, actual, 1)
            results[mult]['all'].append(err)
            if is_decl:
                results[mult]['declining'].append(err)
            else:
                results[mult]['nondecl'].append(err)

        if (idx + 1) % 30 == 0 or idx == len(sids) - 1:
            elapsed = time.time() - t0
            print(f"    [{idx+1:>3d}/{len(sids)}] {elapsed:.0f}s elapsed")

    # CF baseline
    cf_decl = []
    cf_nondecl = []
    cf_all = []
    for sid in sids:
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]
        cf = carry_forward_predict(rounds)
        err = rmse_window(cf, actual, 1)
        cf_all.append(err)
        if sid in declining:
            cf_decl.append(err)
        else:
            cf_nondecl.append(err)

    print(f"\n  {'Mult':>6s} {'All':>8s} {'Decl':>8s} {'NonDecl':>8s} "
          f"{'Δ All':>8s} {'Δ Decl':>8s} {'Δ NonDecl':>10s}")
    print("  " + "-" * 60)
    print(f"  {'CF':>6s} {np.mean(cf_all):>8.4f} {np.mean(cf_decl):>8.4f} "
          f"{np.mean(cf_nondecl):>8.4f} {'—':>8s} {'—':>8s} {'—':>10s}")

    for mult in multipliers:
        all_m = np.mean(results[mult]['all'])
        decl_m = np.mean(results[mult]['declining'])
        nondecl_m = np.mean(results[mult]['nondecl'])
        d_all = all_m - np.mean(cf_all)
        d_decl = decl_m - np.mean(cf_decl)
        d_nondecl = nondecl_m - np.mean(cf_nondecl)
        print(f"  {mult:>6.2f} {all_m:>8.4f} {decl_m:>8.4f} {nondecl_m:>8.4f} "
              f"{d_all:>+8.4f} {d_decl:>+8.4f} {d_nondecl:>+10.4f}")

    # Rate ratio at each multiplier for declining subjects
    print(f"\n  RATE RATIO (pred slope / actual slope) for declining subjects:")
    for mult in multipliers:
        ratios = []
        for sid in declining:
            rounds = n_rounds[sid]
            actual = [r.contribution for r in rounds]
            preds = vcms_predict_scaled(rounds, library, s_rate_mult=mult)
            a_slope = compute_slope(actual)
            p_slope = compute_slope(preds)
            if abs(a_slope) > 0.1:
                ratios.append(p_slope / a_slope)
        print(f"    mult={mult:.2f}: median={np.median(ratios):.3f}, "
              f"mean={np.mean(ratios):.3f}, "
              f"in [0.8,1.2]={sum(1 for r in ratios if 0.8 <= r <= 1.2)}/{len(ratios)}")

    return results


# ================================================================
# TEST 3: STRAIN HABITUATION
# ================================================================

def run_vcms_agent_habituated(params, rounds, habituation_rate=0.0):
    """
    Modified VCMS agent with strain habituation.

    When the gap direction is consistent across rounds, the effective
    S contribution decays exponentially:
      effective_gap_strain = gap_strain * habituation_factor
      habituation_factor *= (1 - habituation_rate) when gap direction is same
      habituation_factor = 1.0 when gap direction reverses

    This models Ch5 Memory Structure: facilitative boundary formation
    around the experience of defection. Repeated identical gaps become
    less psychologically salient.
    """
    p = params
    s_dir = 1.0 if p.s_dir >= 0 else -1.0

    v_level = 0.0
    disposition = 0.0
    strain = p.s_initial
    B = p.b_initial
    m_eval = 0.0
    c_prev_norm = 0.0
    pun_recv_prev = 0.0

    # Habituation state
    hab_factor = 1.0
    prev_gap_sign = 0  # +1, -1, or 0

    pred_contrib = []

    for i, rd in enumerate(rounds):
        # V
        v_group_raw = rd.others_mean / MAX_CONTRIB
        v_group = min(1.0, p.v_rep * v_group_raw)

        if i == 0:
            v_level = v_group
            disposition = rd.contribution / MAX_CONTRIB
        else:
            v_level = p.alpha * v_group + (1.0 - p.alpha) * v_level
            disposition = ANCHOR_RATE * c_prev_norm + (1.0 - ANCHOR_RATE) * disposition

        reference = p.v_ref * v_level + (1.0 - p.v_ref) * disposition

        # S
        if i > 0:
            gap = c_prev_norm - reference
        else:
            gap = 0.0

        directed_gap = gap * s_dir
        gap_strain = max(0.0, directed_gap)

        # Habituation: decay if gap direction is consistent
        current_gap_sign = 1 if gap > 0.01 else (-1 if gap < -0.01 else 0)
        if current_gap_sign != 0 and current_gap_sign == prev_gap_sign:
            hab_factor *= (1.0 - habituation_rate)
        else:
            hab_factor = 1.0
        prev_gap_sign = current_gap_sign

        # Apply habituation to gap strain only
        gap_strain_eff = gap_strain * hab_factor

        pun_strain = pun_recv_prev / 15.0
        strain += p.s_rate * (gap_strain_eff + pun_strain)

        # B
        if i > 0:
            experience = v_group_raw - c_prev_norm
        else:
            experience = 0.0

        if experience < 0:
            magnitude = abs(experience)
            depletion = p.b_depletion_rate * magnitude
            if magnitude > p.acute_threshold:
                depletion *= ACUTE_MULT
            B -= depletion
        elif experience > 0:
            pun_gate = max(0.0, 1.0 - pun_recv_prev / MAX_PUNISH)
            B += p.b_replenish_rate * experience * pun_gate

        if i > 0:
            B -= p.b_depletion_rate * (pun_recv_prev / 15.0)
        B = max(0.0, B)

        # M_eval
        if i > 0:
            m_eval += p.facilitation_rate * experience

        # Resolution
        gate = _sigmoid((B - p.s_thresh) / max(B_NOISE, 0.001))
        discharge = gate * p.s_frac * strain
        remaining_strain = max(0.0, strain - discharge)
        affordability = B / (B + remaining_strain + EPS)

        # M
        if i == 0:
            c_norm = p.c_base
        else:
            w = max(-0.3, min(0.95, p.inertia))
            c_target = p.v_ref * v_level + (1.0 - p.v_ref) * p.c_base
            c_target_adj = max(0.0, min(1.0, c_target + m_eval))
            c_norm = (1.0 - abs(w)) * c_target_adj + w * c_prev_norm

        # C
        h_factor = _horizon_factor(i, len(rounds), p.h_strength, p.h_start)
        c_out_norm = max(0.0, min(1.0, c_norm)) * affordability * h_factor
        c_out = max(0, min(MAX_CONTRIB, round(c_out_norm * MAX_CONTRIB)))
        pred_contrib.append(c_out)

        # State update
        strain = remaining_strain
        c_prev_norm = rd.contribution / MAX_CONTRIB
        pun_recv_prev = rd.punishment_received_total

    return pred_contrib


def vcms_predict_habituated(target_rounds, library, habituation_rate=0.0):
    """Ensemble predictor with habituation."""
    candidates = {}
    for sid, rec in library.items():
        candidates[sid] = VCMSParams(**rec['v3_params'])

    survivors = list(candidates.keys())
    pred_c_list = []
    obs_c_list = []
    cand_pred_history = {sid: [] for sid in candidates}
    distances = {}

    n = min(10, len(target_rounds))
    for t in range(n):
        cand_preds_c = {}
        for sid in survivors:
            preds = run_vcms_agent_habituated(
                candidates[sid], target_rounds[:t + 1], habituation_rate)
            cand_preds_c[sid] = preds[t]
            cand_pred_history[sid] = preds[:t + 1]

        if not distances:
            weights = {sid: 1.0 for sid in survivors}
        else:
            weights = {sid: 1.0 / (distances.get(sid, 0.0) + 0.001)
                       for sid in survivors}
        total_w = sum(weights.values())

        pc = sum((weights[sid] / total_w) * cand_preds_c[sid] for sid in survivors)
        pred_c_list.append(int(round(pc)))
        obs_c_list.append(target_rounds[t].contribution)

        new_distances = {}
        for sid in survivors:
            hist = cand_pred_history[sid]
            nc = min(t + 1, len(hist))
            if nc == 0:
                new_distances[sid] = 0.0
                continue
            c_d = sum((obs_c_list[i] - hist[i]) ** 2
                      for i in range(nc)) / (MAX_CONTRIB ** 2 * nc)
            new_distances[sid] = math.sqrt(c_d)

        if new_distances:
            best = min(new_distances.values())
            thresh = max(best * 3.0, 0.5)
        else:
            thresh = 0.5

        new_surv = [s for s in survivors if new_distances.get(s, 999) <= thresh]
        if not new_surv and new_distances:
            new_surv = [min(new_distances, key=new_distances.get)]

        survivors = new_surv
        distances = {s: new_distances[s] for s in survivors}

    return pred_c_list


def test_strain_habituation(n_rounds, library, city_map):
    """Test whether strain habituation fixes the convex bias for linear decliners."""
    print("\n" + "=" * 72)
    print("  TEST 3: STRAIN HABITUATION")
    print("=" * 72)

    sids = sorted(n_rounds.keys())
    declining = [sid for sid in sids
                 if compute_slope([r.contribution for r in n_rounds[sid]]) < -0.3]

    # Classify declining subjects by actual shape
    concave = []
    linear = []
    convex = []
    for sid in declining:
        actual = [r.contribution for r in n_rounds[sid]]
        shape, curv, slope = classify_curvature(actual)
        if shape == 'concave':
            concave.append(sid)
        elif shape == 'linear':
            linear.append(sid)
        else:
            convex.append(sid)

    print(f"\n  Declining subjects by actual shape:")
    print(f"    Concave: {len(concave)}, Linear: {len(linear)}, Convex: {len(convex)}")

    hab_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    print(f"\n  Sweeping habituation rates: {hab_rates}")

    import time
    t0 = time.time()

    results = {h: {'concave': [], 'linear': [], 'convex': [], 'all_decl': []}
               for h in hab_rates}
    shape_accuracy = {h: {'concave': 0, 'linear': 0, 'convex': 0} for h in hab_rates}
    shape_total = {'concave': len(concave), 'linear': len(linear), 'convex': len(convex)}

    for idx, sid in enumerate(declining):
        actual = [r.contribution for r in n_rounds[sid]]
        actual_shape, _, _ = classify_curvature(actual)
        rounds = n_rounds[sid]

        for h in hab_rates:
            preds = vcms_predict_habituated(rounds, library, habituation_rate=h)
            err = rmse_window(preds, actual, 1)
            results[h]['all_decl'].append(err)

            if actual_shape == 'concave':
                results[h]['concave'].append(err)
            elif actual_shape == 'linear':
                results[h]['linear'].append(err)
            else:
                results[h]['convex'].append(err)

            pred_shape, _, _ = classify_curvature(preds)
            if pred_shape == actual_shape:
                shape_accuracy[h][actual_shape] += 1

        if (idx + 1) % 20 == 0 or idx == len(declining) - 1:
            elapsed = time.time() - t0
            print(f"    [{idx+1:>3d}/{len(declining)}] {elapsed:.0f}s elapsed")

    # Results table
    print(f"\n  RMSE BY SHAPE (declining only):")
    print(f"  {'Hab':>5s} {'All':>8s} {'Concave':>8s} {'Linear':>8s} {'Convex':>8s}")
    print("  " + "-" * 40)
    for h in hab_rates:
        all_m = np.mean(results[h]['all_decl'])
        c_m = np.mean(results[h]['concave']) if results[h]['concave'] else 0
        l_m = np.mean(results[h]['linear']) if results[h]['linear'] else 0
        v_m = np.mean(results[h]['convex']) if results[h]['convex'] else 0
        print(f"  {h:>5.2f} {all_m:>8.4f} {c_m:>8.4f} {l_m:>8.4f} {v_m:>8.4f}")

    # Shape accuracy
    print(f"\n  SHAPE CLASSIFICATION ACCURACY:")
    print(f"  {'Hab':>5s} {'Concave':>12s} {'Linear':>12s} {'Convex':>12s}")
    print("  " + "-" * 45)
    for h in hab_rates:
        c_acc = shape_accuracy[h]['concave'] / shape_total['concave'] if shape_total['concave'] else 0
        l_acc = shape_accuracy[h]['linear'] / shape_total['linear'] if shape_total['linear'] else 0
        v_acc = shape_accuracy[h]['convex'] / shape_total['convex'] if shape_total['convex'] else 0
        print(f"  {h:>5.2f} {c_acc:>8.0%} ({shape_accuracy[h]['concave']:>2d}/{shape_total['concave']:>2d}) "
              f"{l_acc:>5.0%} ({shape_accuracy[h]['linear']:>2d}/{shape_total['linear']:>2d}) "
              f"{v_acc:>5.0%} ({shape_accuracy[h]['convex']:>2d}/{shape_total['convex']:>2d})")

    # Also test on non-declining to check for damage
    nondecl = [sid for sid in sids if sid not in declining]
    print(f"\n  NON-DECLINING SUBJECTS (collateral check, n={len(nondecl)}):")
    nondecl_results = {h: [] for h in hab_rates}
    cf_nondecl = []

    for sid in nondecl:
        rounds = n_rounds[sid]
        actual = [r.contribution for r in rounds]
        cf_nondecl.append(rmse_window(carry_forward_predict(rounds), actual, 1))
        for h in hab_rates:
            preds = vcms_predict_habituated(rounds, library, habituation_rate=h)
            nondecl_results[h].append(rmse_window(preds, actual, 1))

    print(f"  {'Hab':>5s} {'RMSE':>8s} {'Δ vs CF':>8s}")
    print("  " + "-" * 25)
    print(f"  {'CF':>5s} {np.mean(cf_nondecl):>8.4f} {'—':>8s}")
    for h in hab_rates:
        m = np.mean(nondecl_results[h])
        d = m - np.mean(cf_nondecl)
        print(f"  {h:>5.2f} {m:>8.4f} {d:>+8.4f}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 72)
    print("  N-EXPERIMENT THEORETICAL VALIDATION")
    print("=" * 72)

    library = load_library()
    n_rounds, city_map = load_n_experiment_data()
    print(f"  Library: {len(library)} P-experiment subjects")
    print(f"  Test:    {len(n_rounds)} N-experiment subjects")

    validate_comfortable_defector(n_rounds, library, city_map)
    test_srate_correction(n_rounds, library, city_map)
    test_strain_habituation(n_rounds, library, city_map)

    print("\n" + "=" * 72)
    print("  VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
