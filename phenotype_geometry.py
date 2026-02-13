#!/usr/bin/env python3
"""
Cross-Library Phenotype Geometry Analysis
==========================================

Tests whether cooperator phenotypes identified independently in three
different games converge to the same region of normalized parameter space.

Libraries (all v4 normalized-time):
  - P-experiment: 176 subjects, PGG with punishment, 10 rounds
  - N-experiment: 212 subjects, PGG without punishment, 10 rounds
  - IPD: 188 subjects, iterated prisoner's dilemma, 100 rounds

The 15 shared free parameters are compared across 576 total subjects.
Phenotypes are mapped to a unified cooperation axis:
  HIGH: P:cooperator + cooperative-enforcer, N:stable-high, IPD:mostly-C
  LOW:  P:free-rider + punitive-free-rider + antisocial-controller,
        N:stable-low, IPD:mostly-D
  MID:  P:mixed, N:declining + stable-mid + rising, IPD:mixed
"""

import json
import math
import numpy as np
from collections import defaultdict


# ================================================================
# CONFIGURATION
# ================================================================

# The 15 shared free parameters across all three games
SHARED_PARAMS = [
    'alpha', 'v_rep', 'v_ref', 'c_base', 'inertia',
    's_dir', 's_rate', 's_initial',
    'b_initial', 'b_depletion_rate', 'b_replenish_rate', 'acute_threshold',
    'facilitation_rate', 'h_strength', 'h_start',
]

# Phenotype mapping to unified cooperation axis
P_HIGH = {'cooperator', 'cooperative-enforcer'}
P_LOW = {'free-rider', 'punitive-free-rider', 'antisocial-controller'}
P_MID = {'mixed'}

N_HIGH = {'stable-high'}
N_LOW = {'stable-low'}
N_MID = {'declining', 'stable-mid', 'rising'}

IPD_HIGH = {'mostly-C'}
IPD_LOW = {'mostly-D'}
IPD_MID = {'mixed'}


def unified_label(game, phenotype):
    """Map game-specific phenotype to unified cooperation axis."""
    if game == 'PGG-P':
        if phenotype in P_HIGH:
            return 'high'
        elif phenotype in P_LOW:
            return 'low'
        else:
            return 'mid'
    elif game == 'PGG-N':
        if phenotype in N_HIGH:
            return 'high'
        elif phenotype in N_LOW:
            return 'low'
        else:
            return 'mid'
    elif game == 'IPD':
        if phenotype in IPD_HIGH:
            return 'high'
        elif phenotype in IPD_LOW:
            return 'low'
        else:
            return 'mid'
    return 'mid'


# ================================================================
# DATA LOADING
# ================================================================

def load_all_libraries():
    """Load all three v4 libraries and extract shared parameters."""

    subjects = []  # list of dicts: {game, sid, phenotype, unified, params[15]}

    # P-experiment
    with open('v4_library_fitted.json') as f:
        p_lib = json.load(f)
    for sid, rec in p_lib.items():
        phenotype = rec['behavioral_profile']
        params = [rec['v3_params'][p] for p in SHARED_PARAMS]
        # Snap s_dir to ±1
        params[5] = 1.0 if params[5] >= 0 else -1.0
        subjects.append({
            'game': 'PGG-P', 'sid': sid, 'phenotype': phenotype,
            'unified': unified_label('PGG-P', phenotype),
            'params': np.array(params),
        })

    # N-experiment
    with open('v4_n_library_fitted.json') as f:
        n_lib = json.load(f)
    for sid, rec in n_lib.items():
        phenotype = rec['subject_type']
        params = [rec['v3_params'][p] for p in SHARED_PARAMS]
        params[5] = 1.0 if params[5] >= 0 else -1.0
        subjects.append({
            'game': 'PGG-N', 'sid': sid, 'phenotype': phenotype,
            'unified': unified_label('PGG-N', phenotype),
            'params': np.array(params),
        })

    # IPD
    with open('v4_ipd_library_fitted.json') as f:
        ipd_lib = json.load(f)
    for sid, rec in ipd_lib.items():
        phenotype = rec['subject_type']
        params = [rec['v3_params'][p] for p in SHARED_PARAMS]
        params[5] = 1.0 if params[5] >= 0 else -1.0
        subjects.append({
            'game': 'IPD', 'sid': sid, 'phenotype': phenotype,
            'unified': unified_label('IPD', phenotype),
            'params': np.array(params),
        })

    return subjects


# ================================================================
# ANALYSIS 1: CROSS-GAME PHENOTYPE ALIGNMENT TABLE
# ================================================================

def phenotype_alignment_table(subjects):
    """Print mean parameter values per unified group, per game."""

    print(f"\n{'=' * 90}")
    print(f"  1. CROSS-GAME PHENOTYPE ALIGNMENT TABLE")
    print(f"{'=' * 90}")

    games = ['PGG-P', 'PGG-N', 'IPD']
    groups = ['high', 'mid', 'low']

    # Count subjects per group per game
    print(f"\n  Subject counts:")
    print(f"    {'':>12s}", end='')
    for game in games:
        print(f"  {game:>8s}", end='')
    print(f"  {'Total':>8s}")
    print(f"    {'-' * 42}")

    for group in groups:
        print(f"    {group:>12s}", end='')
        total = 0
        for game in games:
            n = sum(1 for s in subjects
                    if s['game'] == game and s['unified'] == group)
            print(f"  {n:>8d}", end='')
            total += n
        print(f"  {total:>8d}")

    # Parameter means per group per game
    # Focus on the key phenotype-defining parameters
    key_params = ['c_base', 'inertia', 's_initial', 'b_initial',
                  'b_depletion_rate', 'b_replenish_rate',
                  's_rate', 'h_strength', 'h_start']

    for group in groups:
        print(f"\n  --- {group.upper()} cooperators ---")
        print(f"    {'Parameter':<22s}", end='')
        for game in games:
            print(f"  {game:>8s}", end='')
        print(f"  {'spread':>8s}")
        print(f"    {'-' * 54}")

        for pname in key_params:
            pidx = SHARED_PARAMS.index(pname)
            print(f"    {pname:<22s}", end='')
            means = []
            for game in games:
                vals = [s['params'][pidx] for s in subjects
                        if s['game'] == game and s['unified'] == group]
                if vals:
                    m = np.mean(vals)
                    means.append(m)
                    print(f"  {m:>8.3f}", end='')
                else:
                    print(f"  {'—':>8s}", end='')

            # Cross-game spread (max - min of means)
            if len(means) >= 2:
                spread = max(means) - min(means)
                print(f"  {spread:>8.3f}", end='')
            print()

    return True


# ================================================================
# ANALYSIS 2: PAIRWISE DISTANCE IN Z-SCORED PARAMETER SPACE
# ================================================================

def pairwise_distance_analysis(subjects):
    """Centroid distances and within-group spread in z-scored space."""

    print(f"\n{'=' * 90}")
    print(f"  2. PAIRWISE DISTANCE ANALYSIS (z-scored parameter space)")
    print(f"{'=' * 90}")

    # Build full parameter matrix and z-score
    all_params = np.array([s['params'] for s in subjects])
    means = all_params.mean(axis=0)
    stds = all_params.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero (s_dir is ±1)
    z_params = (all_params - means) / stds

    # Assign z-scored params back
    for i, s in enumerate(subjects):
        s['z_params'] = z_params[i]

    games = ['PGG-P', 'PGG-N', 'IPD']
    groups = ['high', 'mid', 'low']

    # Compute centroids
    centroids = {}
    within_spread = {}
    for game in games:
        for group in groups:
            members = [s['z_params'] for s in subjects
                       if s['game'] == game and s['unified'] == group]
            if members:
                members = np.array(members)
                centroid = members.mean(axis=0)
                centroids[(game, group)] = centroid
                # Within-group spread: mean Euclidean distance to centroid
                dists = [np.linalg.norm(m - centroid) for m in members]
                within_spread[(game, group)] = np.mean(dists)

    # Print within-game spread
    print(f"\n  Within-group mean distance to centroid:")
    print(f"    {'Group':<8s}", end='')
    for game in games:
        print(f"  {game:>8s}", end='')
    print()
    print(f"    {'-' * 32}")
    for group in groups:
        print(f"    {group:<8s}", end='')
        for game in games:
            key = (game, group)
            if key in within_spread:
                print(f"  {within_spread[key]:>8.2f}", end='')
            else:
                print(f"  {'—':>8s}", end='')
        print()

    # Cross-game centroid distances (same phenotype, different games)
    print(f"\n  Cross-game centroid distances (same phenotype):")
    game_pairs = [('PGG-P', 'PGG-N'), ('PGG-P', 'IPD'), ('PGG-N', 'IPD')]
    print(f"    {'Group':<8s}", end='')
    for g1, g2 in game_pairs:
        print(f"  {g1}↔{g2:>6s}", end='')
    print()
    print(f"    {'-' * 56}")
    for group in groups:
        print(f"    {group:<8s}", end='')
        for g1, g2 in game_pairs:
            k1 = (g1, group)
            k2 = (g2, group)
            if k1 in centroids and k2 in centroids:
                dist = np.linalg.norm(centroids[k1] - centroids[k2])
                print(f"  {dist:>12.2f}", end='')
            else:
                print(f"  {'—':>12s}", end='')
        print()

    # Cross-game vs different-phenotype distances
    print(f"\n  Cross-game DIFFERENT phenotype distances (should be larger):")
    print(f"    {'Comparison':<24s}  {'Distance':>10s}")
    print(f"    {'-' * 38}")

    # Same phenotype cross-game (should be small)
    same_dists = []
    for group in groups:
        for g1, g2 in game_pairs:
            k1 = (g1, group)
            k2 = (g2, group)
            if k1 in centroids and k2 in centroids:
                d = np.linalg.norm(centroids[k1] - centroids[k2])
                same_dists.append(d)
    print(f"    {'Same phenotype, x-game':<24s}  {np.mean(same_dists):>10.2f}")

    # Different phenotype same-game (should be large)
    diff_within_dists = []
    for game in games:
        for g1 in groups:
            for g2 in groups:
                if g1 >= g2:
                    continue
                k1 = (game, g1)
                k2 = (game, g2)
                if k1 in centroids and k2 in centroids:
                    d = np.linalg.norm(centroids[k1] - centroids[k2])
                    diff_within_dists.append(d)
    print(f"    {'Diff phenotype, same game':<24s}  {np.mean(diff_within_dists):>10.2f}")

    # Different phenotype cross-game (should be largest)
    diff_cross_dists = []
    for g1, g2 in game_pairs:
        for gr1 in groups:
            for gr2 in groups:
                if gr1 == gr2:
                    continue
                k1 = (g1, gr1)
                k2 = (g2, gr2)
                if k1 in centroids and k2 in centroids:
                    d = np.linalg.norm(centroids[k1] - centroids[k2])
                    diff_cross_dists.append(d)
    print(f"    {'Diff phenotype, x-game':<24s}  {np.mean(diff_cross_dists):>10.2f}")

    ratio = np.mean(diff_within_dists) / max(np.mean(same_dists), 0.01)
    print(f"\n  Separation ratio (diff-within / same-cross): {ratio:.2f}x")
    if ratio > 1.5:
        print(f"  → Phenotype separation is STRONGER than game separation")
    elif ratio > 1.0:
        print(f"  → Phenotype separation is marginally stronger than game separation")
    else:
        print(f"  → Game separation dominates phenotype separation")

    return centroids


# ================================================================
# ANALYSIS 3: CROSS-GAME DISCRIMINANT ANALYSIS
# ================================================================

def discriminant_analysis(subjects):
    """Train LDA on one game, test on others."""

    print(f"\n{'=' * 90}")
    print(f"  3. CROSS-GAME DISCRIMINANT ANALYSIS")
    print(f"{'=' * 90}")

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    games = ['PGG-P', 'PGG-N', 'IPD']
    label_map = {'high': 0, 'mid': 1, 'low': 2}

    # Build per-game datasets
    game_data = {}
    for game in games:
        game_subjects = [s for s in subjects if s['game'] == game]
        X = np.array([s['params'] for s in game_subjects])
        y = np.array([label_map[s['unified']] for s in game_subjects])
        game_data[game] = (X, y)

    # Train on each game, test on others
    print(f"\n  Cross-game classification accuracy (3-class: high/mid/low):")
    print(f"  Chance baseline: 33.3%")
    print(f"\n    {'Train →':>12s}", end='')
    for game in games:
        print(f"  {game:>8s}", end='')
    print(f"  {'Mean':>8s}")
    print(f"    {'-' * 42}")

    for test_game in games:
        print(f"    {'Test ' + test_game:>12s}", end='')
        accs = []
        for train_game in games:
            if train_game == test_game:
                # Within-game (LOO-style, but just resubstitution for now)
                X, y = game_data[train_game]
                lda = LinearDiscriminantAnalysis()
                lda.fit(X, y)
                acc = lda.score(X, y)
                print(f"  {acc:>7.1%}", end='')
            else:
                X_train, y_train = game_data[train_game]
                X_test, y_test = game_data[test_game]
                lda = LinearDiscriminantAnalysis()
                lda.fit(X_train, y_train)
                acc = lda.score(X_test, y_test)
                accs.append(acc)
                print(f"  {acc:>7.1%}", end='')
        if accs:
            print(f"  {np.mean(accs):>7.1%}", end='')
        print()

    # Also train on ALL data and report feature importances
    print(f"\n  --- Pooled LDA (all 576 subjects) ---")
    X_all = np.array([s['params'] for s in subjects])
    y_all = np.array([label_map[s['unified']] for s in subjects])

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_all, y_all)
    pooled_acc = lda.score(X_all, y_all)
    print(f"  Pooled accuracy: {pooled_acc:.1%}")

    # Feature importance via LDA coefficients
    # For 3-class LDA, we get 2 discriminant axes
    print(f"\n  Top discriminating parameters (LD1 loadings):")
    coefs = lda.scalings_[:, 0]  # First discriminant axis
    param_importance = sorted(zip(SHARED_PARAMS, coefs),
                              key=lambda x: abs(x[1]), reverse=True)
    for pname, coef in param_importance[:8]:
        bar = '#' * int(abs(coef) * 3)
        sign = '+' if coef > 0 else '-'
        print(f"    {pname:<22s} {sign}{abs(coef):>6.3f}  {bar}")

    return lda


# ================================================================
# ANALYSIS 4: BUDGET CURVE SHAPE COMPARISON
# ================================================================

def budget_curve_comparison(subjects):
    """Simulate synthetic B-curves from mean params per phenotype per game."""

    print(f"\n{'=' * 90}")
    print(f"  4. BUDGET CURVE SHAPE COMPARISON")
    print(f"{'=' * 90}")

    games = ['PGG-P', 'PGG-N', 'IPD']
    groups = ['high', 'mid', 'low']

    # Get indices for budget-related params
    bi_idx = SHARED_PARAMS.index('b_initial')
    bd_idx = SHARED_PARAMS.index('b_depletion_rate')
    br_idx = SHARED_PARAMS.index('b_replenish_rate')
    at_idx = SHARED_PARAMS.index('acute_threshold')

    print(f"\n  Budget parameters by phenotype and game:")
    print(f"    {'Group':<6s} {'Game':<8s} {'n':>4s} {'b_init':>8s} "
          f"{'b_depl':>8s} {'b_repl':>8s} {'ratio':>8s}")
    print(f"    {'-' * 56}")

    for group in groups:
        for game in games:
            members = [s for s in subjects
                       if s['game'] == game and s['unified'] == group]
            if not members:
                continue
            b_init = np.mean([s['params'][bi_idx] for s in members])
            b_depl = np.mean([s['params'][bd_idx] for s in members])
            b_repl = np.mean([s['params'][br_idx] for s in members])
            ratio = b_repl / max(b_depl, 0.001)
            print(f"    {group:<6s} {game:<8s} {len(members):>4d} {b_init:>8.2f} "
                  f"{b_depl:>8.2f} {b_repl:>8.2f} {ratio:>8.2f}")

    # Simulate B-curves over normalized time [0, 1]
    # Use a neutral experience sequence: alternating +0.1, -0.1
    # This tests how the budget parameters respond to mild fluctuation
    print(f"\n  Simulated B-curve at t=0.0, 0.25, 0.50, 0.75, 1.00:")
    print(f"  (neutral experience: alternating ±0.1)")
    print(f"    {'Group':<6s} {'Game':<8s} {'t=0.0':>7s} {'t=0.25':>7s} "
          f"{'t=0.50':>7s} {'t=0.75':>7s} {'t=1.0':>7s} {'final/init':>10s}")
    print(f"    {'-' * 64}")

    n_steps = 100
    dt = 1.0 / (n_steps - 1)
    checkpoints = [0, 25, 50, 75, 99]

    for group in groups:
        for game in games:
            members = [s for s in subjects
                       if s['game'] == game and s['unified'] == group]
            if not members:
                continue

            b_init = np.mean([s['params'][bi_idx] for s in members])
            b_depl = np.mean([s['params'][bd_idx] for s in members])
            b_repl = np.mean([s['params'][br_idx] for s in members])
            a_thresh = np.mean([s['params'][at_idx] for s in members])

            # Simulate
            B = b_init
            b_traj = []
            for i in range(n_steps):
                b_traj.append(B)
                # Alternating experience
                exp = 0.1 if i % 2 == 0 else -0.1
                if exp < 0:
                    mag = abs(exp)
                    depl = dt * b_depl * mag
                    if mag > a_thresh:
                        depl *= 5.0
                    B -= depl
                else:
                    B += dt * b_repl * exp
                B = max(0.0, B)

            row = f"    {group:<6s} {game:<8s}"
            for c in checkpoints:
                row += f" {b_traj[c]:>7.2f}"
            ratio = b_traj[-1] / max(b_traj[0], 0.001)
            row += f" {ratio:>10.2f}"
            print(row)

    return True


# ================================================================
# ANALYSIS 5: PARAMETER CORRELATION ACROSS GAMES
# ================================================================

def parameter_correlation(subjects):
    """Check if parameter relationships are consistent across games."""

    print(f"\n{'=' * 90}")
    print(f"  5. KEY PARAMETER CORRELATIONS BY GAME")
    print(f"{'=' * 90}")

    games = ['PGG-P', 'PGG-N', 'IPD']

    # Key parameter pairs that should correlate similarly across games
    pairs = [
        ('c_base', 'b_initial'),          # cooperators start with more budget
        ('c_base', 's_initial'),           # cooperators have less pre-loaded strain
        ('b_depletion_rate', 'inertia'),   # fragile budget → needs inertia to persist
        ('c_base', 'h_strength'),          # cooperators less endgame-sensitive
    ]

    print(f"\n    {'Pair':<35s}", end='')
    for game in games:
        print(f"  {game:>8s}", end='')
    print()
    print(f"    {'-' * 60}")

    for p1_name, p2_name in pairs:
        p1_idx = SHARED_PARAMS.index(p1_name)
        p2_idx = SHARED_PARAMS.index(p2_name)

        print(f"    {p1_name + ' ↔ ' + p2_name:<35s}", end='')
        for game in games:
            vals1 = [s['params'][p1_idx] for s in subjects if s['game'] == game]
            vals2 = [s['params'][p2_idx] for s in subjects if s['game'] == game]
            if vals1 and vals2:
                corr = np.corrcoef(vals1, vals2)[0, 1]
                print(f"  {corr:>+7.2f}", end='')
            else:
                print(f"  {'—':>8s}", end='')
        print()


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 90)
    print("  CROSS-LIBRARY PHENOTYPE GEOMETRY ANALYSIS")
    print("  576 subjects × 15 parameters × 3 games")
    print("  All in normalized game time")
    print("=" * 90)

    # Load
    print("\nLoading v4 normalized-time libraries...")
    subjects = load_all_libraries()
    print(f"  Total subjects: {len(subjects)}")

    # Verify no missing params
    for s in subjects:
        assert len(s['params']) == 15, f"Missing params for {s['sid']}"
        assert not np.any(np.isnan(s['params'])), f"NaN params for {s['sid']}"
    print(f"  Parameter matrix: {len(subjects)} × {len(SHARED_PARAMS)} — clean")

    # Distribution
    for game in ['PGG-P', 'PGG-N', 'IPD']:
        n = sum(1 for s in subjects if s['game'] == game)
        types = defaultdict(int)
        for s in subjects:
            if s['game'] == game:
                types[s['unified']] += 1
        print(f"  {game}: {n} subjects — high:{types['high']}, "
              f"mid:{types['mid']}, low:{types['low']}")

    # Run analyses
    phenotype_alignment_table(subjects)
    centroids = pairwise_distance_analysis(subjects)
    lda = discriminant_analysis(subjects)
    budget_curve_comparison(subjects)
    parameter_correlation(subjects)

    print(f"\n{'=' * 90}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
