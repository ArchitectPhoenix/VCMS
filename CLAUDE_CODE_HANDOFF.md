# VCMS v3 — Claude Code Handoff

## What This Is

A behavioral prediction model for public goods games (PGG). Given a human subject playing a 10-round cooperation game, predict their contribution and punishment decisions each round using a library of previously-observed subjects and a cognitive dynamics model.

## The Architecture (settled)

Three components, executed sequentially each round:

### 1. OBSERVE
Record the subject's actual contribution (C) and punishment (P) at round t.

### 2. ELIMINATE (behavioral distance — what you can see)
Maintain a population of ~32 library candidates. Each is a real human subject whose behavior we've previously recorded and modeled. After observing round t, compute behavioral distance between the new subject's C/P trajectory so far and each library candidate's recorded C/P trajectory over the same rounds:

```
d = sqrt( Σ(C_actual[i] - C_lib[i])² / (MAX_C² × n)  +  2.0 × Σ(P_actual[i] - P_lib[i])² / (MAX_P² × n) )
```

P weighted 2× because punishment patterns are more diagnostic than contribution patterns. Eliminate candidates beyond an adaptive threshold (3× best distance or 0.5 floor). This is a Guess Who protocol — hard elimination, not soft weighting.

### 3. PREDICT (fitted dynamics — what the model learned)
For each surviving candidate, run the v3 cognitive model with **that candidate's own fitted parameters** on the **current subject's actual environment** (others' mean contribution, punishment received). The model answers: "If this known human type were in this situation, what would they do?"

Prediction for round t+1 = inverse-distance-weighted average of survivors' model outputs.

R1 is observation only (no prediction with zero information). Predictions start at R2.

### Forward Shadow
Track the elimination curve E(t) = number of survivors per round. Compute first and second derivatives. When E''(t) accelerates (hypothesis space polarizing), a behavioral transition is approaching. This has been validated: E'' spikes predict contribution changes 1 round ahead.

## What Exists

### Files

| File | Lines | Role |
|------|-------|------|
| `pgg_vcms_agent_v3.py` | 613 | The v3 cognitive model. 16 parameters: V (visibility), C (contribution), S (strain), M (memory/eval), B (budget). Takes a subject's environment, produces predicted C and P per round. |
| `pgg_p_loader.py` | ~300 | Loads Herrmann et al. PGG data from CSV. Returns dict of subject_id → list of round objects. |
| `pgg_vcms_fit_v3.py` | ~300 | Fitting pipeline. Differential evolution + Nelder-Mead to fit v3's 16 params to a subject's actual behavior. |
| `build_v3_library.py` | ~140 | Batch-fits v3 params for all 32 library subjects. Produces `v3_library_fitted.json`. |
| `vcms_v3_combined.py` | 320 | **The combined predictor.** Behavioral elimination + dynamic prediction. This is the current best version. |
| `vcms_v3_sequential.py` | 772 | Earlier version (behavioral matching only, predicts from raw trajectory averages). Superseded by combined. |
| `vcms_v3_state_matching.py` | ~600 | Earlier version (state matching for both elimination and prediction). Superseded by combined. |

### Data Files

| File | Contents |
|------|----------|
| `v3_library_fitted.json` | 32 Samara subjects with fitted v3 params, state trajectories, behavioral profiles |
| `p_experiment_canonical_library.json` | Same 32 subjects with behavioral classifications |
| `HerrmannThoeniGaechterDATA_*_P-EXPERIMENT_TRUNCATED_SESSION*.csv` | Raw PGG data for Boston (24 subj), Istanbul (20), Samara S1 (16), Samara S2 (16) |

### Validated Results (Combined Predictor, R2-R10)

| City | Overall RMSE | Forward Shadow Events | Notes |
|------|-------------|----------------------|-------|
| Boston (out-of-sample) | 0.199 | 3 | 24 subjects |
| Istanbul (out-of-sample) | 0.168 | 6 | 20 subjects |
| Samara S2 (in-sample city) | 0.097 | 6 | 16 subjects, converges to 0.034 by R10 |

## The v3 Agent Internals

The agent runs a per-round loop. Each round, given environment inputs (others' mean contribution, punishment received), it updates internal state and produces predicted C and P.

### State variables (persist across rounds)
- **B** (Budget): Psychological resource. Depletes from exploitation/punishment, replenishes from positive group experience. Gates output via affordability = B/(B+1).
- **S** (Strain): Accumulated tension. Builds from gap between own contribution and group norm. Discharges through a sigmoid gate when it crosses a threshold.
- **m_eval** (Memory evaluator): Accumulated facilitation/inhibition from group experience. Positive experience facilitates future cooperation; negative inhibits.

### Per-round computation
1. **V** (Visibility): Computed from others' cooperation (v_rep) and punishment signal (v_ref). Scales how much m_eval influences output.
2. **Strain update**: Gap between own contribution norm and group reference, directed by s_dir (±1 orientation). Accumulates via s_rate.
3. **Budget update**: experience = others_cooperation - my_cooperation. Negative → depletion. Positive → replenishment. Punishment received always depletes.
4. **M_eval update**: Accumulates facilitation/inhibition from experience signal.
5. **C output**: c_base + inertia×previous + m_eval×V, scaled by affordability. Clamped to [0, MAX_CONTRIB].
6. **P output**: Separate punishment channel based on gap detection and strain.

### Parameters (16 total)
```
V channel:   v_rep, v_ref
C channel:   c_base, alpha, inertia
S channel:   s_dir, s_rate, s_initial, s_thresh, s_frac
P channel:   p_scale
B channel:   b_initial, b_depletion_rate, b_replenish_rate, acute_threshold
M channel:   facilitation_rate
```

## What Needs Work

### Priority 1: Code quality
The current codebase was developed iteratively in a chat environment. It works but isn't clean:
- Three predictor files exist (sequential, state_matching, combined). Only `vcms_v3_combined.py` matters.
- No tests.
- No CLI consistency across files.
- The agent (`pgg_vcms_agent_v3.py`) has vestigial knockout logic from earlier experiments.

### Priority 2: Library expansion
Currently 32 subjects from Samara only. The fitting infrastructure exists to add Boston (24), Istanbul (20), and Samara S2 (16) — all in the CSVs. Larger library = better coverage of behavioral archetypes = better late-round predictions. The "endgame bump" (R10 error spike) is partly because the library hasn't seen enough defection-at-end patterns.

### Priority 3: Facilitation channel
`facilitation_rate` parameter is suspected vestigial — it may not be doing meaningful work distinct from `alpha`. Needs ablation: fit with and without, compare RMSE. If removal doesn't hurt, cut to 15 params.

### Priority 4: Cross-validation
Current library is fit on Samara, predictions tested on Boston/Istanbul. Should also test: fit on Boston, predict Istanbul. Fit on Istanbul, predict Samara. If the architecture is sound, convergence curves should replicate regardless of which city seeds the library.

## How to Run

```bash
# Fit library (if needed, ~1 min on 16 threads)
python3 build_v3_library.py

# Run combined predictor on a city
python3 vcms_v3_combined.py --csv HerrmannThoeniGaechterDATA_BOSTON_P-EXPERIMENT_TRUNCATED_SESSION1.csv

# Dependencies: numpy, scipy
```

## Key Principles

1. **Match on what you can see, predict from what you've learned.** Elimination uses raw behavioral distance. Prediction uses fitted dynamics. Never match on guessed internal state.

2. **R1 is observation, not prediction.** With zero information, any output is a guess. Predictions start at R2.

3. **The convergence curve is the finding.** Error decreasing over rounds = the model is learning who this person is. The shape and speed of convergence matter more than absolute error.

4. **Forward shadow is real.** E'' acceleration in the elimination curve predicts behavioral transitions before they appear in raw data. This has been validated on human subjects.

5. **Model-first accountability.** When predictions diverge from observations, check the model first — not the data, not the theory. The worked trace showed that a "budget collapse" turned out to be a bug in manual computation, not in the agent.
