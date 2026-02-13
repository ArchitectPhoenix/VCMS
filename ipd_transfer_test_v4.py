"""
IPD Cross-Game Transfer Test — V4 Normalized Time
==================================================

Same transfer test as ipd_transfer_test.py, but using:
  - v4_library_fitted.json (normalized-time parameters)
  - IPD_CONFIG with normalized_time=True

This tests whether temporal normalization fixes the budget collapse
that caused v3 library subjects to predict defection on long IPD games.
"""

import json
import numpy as np
from vcms_engine_v4 import GameConfig
from ipd_loader import load_ipd_experiment
from ipd_transfer_test import run_transfer_test


# IPD config with normalized time
IPD_NORM_CONFIG = GameConfig(
    max_contrib=1, max_punish=1, has_punishment=False,
    n_signals=2, normalized_time=True,
)


def main():
    print("=" * 72)
    print("  IPD CROSS-GAME TRANSFER TEST — V4 NORMALIZED TIME")
    print("  PGG P-experiment library (normalized) → IPD predictions")
    print("=" * 72)

    # Load v4 library
    print("\nLoading v4 normalized-time library...")
    with open('v4_library_fitted.json') as f:
        library = json.load(f)
    print(f"  {len(library)} library subjects")

    # Also load v3 for comparison
    print("Loading v3 legacy library for comparison...")
    with open('v3_library_fitted.json') as f:
        v3_library = json.load(f)

    # Load IPD data
    print("\nLoading IPD data...")
    sp_data = load_ipd_experiment('IPD-rand.csv')
    fp_data = load_ipd_experiment('fix.csv')
    print(f"  SP: {len(sp_data)} subjects (Stranger Pairing)")
    print(f"  FP: {len(fp_data)} subjects (Fixed Pairing)")

    # Run v4 normalized transfer test
    # We need to patch the game config used in precompute_candidate_predictions.
    # The run_transfer_test function uses IPD_CONFIG internally.
    # Let's monkey-patch it for this run.
    import ipd_transfer_test as itt
    original_config = itt.IPD_CONFIG
    itt.IPD_CONFIG = IPD_NORM_CONFIG

    print(f"\n  Using normalized_time=True (dt={1.0/99:.4f} for 100-round games)")

    sp_results_v4 = run_transfer_test(sp_data, library, "V4 NORMALIZED — SP")
    fp_results_v4 = run_transfer_test(fp_data, library, "V4 NORMALIZED — FP")

    # Run v3 legacy for direct comparison
    itt.IPD_CONFIG = original_config
    sp_results_v3 = run_transfer_test(sp_data, v3_library, "V3 LEGACY — SP")
    fp_results_v3 = run_transfer_test(fp_data, v3_library, "V3 LEGACY — FP")

    # Restore
    itt.IPD_CONFIG = original_config

    # ================================================================
    # V3 vs V4 COMPARISON
    # ================================================================

    print(f"\n{'=' * 72}")
    print(f"  V3 vs V4 COMPARISON")
    print(f"{'=' * 72}")

    k_values = [1, 5, 10, 20, 50]
    for label, v3r, v4r in [("SP", sp_results_v3, sp_results_v4),
                             ("FP", fp_results_v3, fp_results_v4)]:
        print(f"\n  --- {label}: Accuracy from round k ---")
        print(f"    {'k':>6s} {'v3 VCMS':>10s} {'v4 VCMS':>10s} {'Delta':>10s} {'v3 CF':>10s}")
        print(f"    {'-' * 46}")
        for k in k_values:
            v3_acc = np.mean(v3r['acc']['VCMS'][k])
            v4_acc = np.mean(v4r['acc']['VCMS'][k])
            cf_acc = np.mean(v3r['acc']['Carry-Fwd'][k])
            print(f"    {k:>6d} {v3_acc:>9.1%} {v4_acc:>10.1%} "
                  f"{v4_acc - v3_acc:>+10.1%} {cf_acc:>10.1%}")

        print(f"\n  --- {label}: Kappa ---")
        v3_k = np.mean(v3r['kappas']['VCMS'])
        v4_k = np.mean(v4r['kappas']['VCMS'])
        cf_k = np.mean(v3r['kappas']['Carry-Fwd'])
        print(f"    v3 VCMS: {v3_k:.3f}")
        print(f"    v4 VCMS: {v4_k:.3f} ({v4_k - v3_k:+.3f})")
        print(f"    CF:      {cf_k:.3f}")

        print(f"\n  --- {label}: Per-type accuracy (from round 2) ---")
        types = ['mostly-D', 'mixed', 'mostly-C']
        print(f"    {'Type':<12s} {'v3 VCMS':>10s} {'v4 VCMS':>10s} {'Delta':>10s}")
        print(f"    {'-' * 42}")
        for t in types:
            v3_vals = v3r['acc']['VCMS'][1]  # already computed per-subject
            v4_vals = v4r['acc']['VCMS'][1]
            # Need to recompute per-type
            v3_type = [v3_vals[i] for i, sid in enumerate(sorted(v3r['subject_types'].keys()))
                       if v3r['subject_types'][sid] == t]
            v4_type = [v4_vals[i] for i, sid in enumerate(sorted(v4r['subject_types'].keys()))
                       if v4r['subject_types'][sid] == t]
            if v3_type and v4_type:
                print(f"    {t:<12s} {np.mean(v3_type):>9.1%} {np.mean(v4_type):>10.1%} "
                      f"{np.mean(v4_type) - np.mean(v3_type):>+10.1%}")

        print(f"\n  --- {label}: Trajectory RMSE ---")
        v3_tr = np.mean(v3r['traj_rmse']['VCMS'])
        v4_tr = np.mean(v4r['traj_rmse']['VCMS'])
        cf_tr = np.mean(v3r['traj_rmse']['Carry-Fwd'])
        print(f"    v3 VCMS: {v3_tr:.4f}")
        print(f"    v4 VCMS: {v4_tr:.4f} ({v4_tr - v3_tr:+.4f})")
        print(f"    CF:      {cf_tr:.4f}")

    # Survivor analysis comparison
    print(f"\n  --- Survivor parameter profiles (FP, mostly-C) ---")
    for label, results in [("v3", fp_results_v3), ("v4", fp_results_v4)]:
        survivors = results['vcms_survivors']
        lib = v3_library if label == "v3" else library
        type_map = results['subject_types']
        mostly_c_sids = [s for s in type_map if type_map[s] == 'mostly-C']
        all_surv = []
        for sid in mostly_c_sids:
            for lib_sid in survivors[sid]:
                if lib_sid in lib:
                    all_surv.append(lib[lib_sid]['v3_params'])
        if all_surv:
            print(f"\n    {label} survivors for mostly-C ({len(all_surv)} matches):")
            for pname in ['c_base', 'inertia', 'b_initial', 'b_depletion_rate',
                          'b_replenish_rate', 's_rate', 'h_start']:
                vals = [s[pname] for s in all_surv]
                print(f"      {pname:>20s}: mean={np.mean(vals):.3f}")

    print(f"\n{'=' * 72}")
    print(f"  TRANSFER TEST COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
