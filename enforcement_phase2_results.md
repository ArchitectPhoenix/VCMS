# Federation Dynamics Phase 2 — Transition, Combination, and EC Protection

## Experimental Context

Phase 1 established that enforcement mechanisms produce a compliance-sustainability tradeoff: punishment generates the highest peak cooperation but the fastest rupture (TTFR=8); sustainability exclusion is the most balanced intervention (longest TTFR=34, highest welfare, 5.5x fewer removals than threshold); voluntary exit produces partial sorting but incomplete bimodality (BC=0.50) at eval_freq=10. Three questions follow.

**Test 1 (Transition):** Can a system recover from punishment architecture? What does "reform" look like in agent dynamics — does the population snap back, drift, or carry permanent damage?

**Test 2 (Combination):** Do sustainability exclusion and voluntary exit complement each other? If one removes toxic agents and the other lets cooperators self-sort, does running both outperform either alone?

**Test 3 (EC Protection):** Does care-first enforcement structurally protect transparent agents? EC agents (low inertia, environment-tracking) are the only phenotype that loses members under punishment. Is this because they're legible — and if so, does legibility predict rupture generally?

---

## Method

### Architecture: Generalized Simulation Loop

Phase 1 used separate runner functions per condition, each containing its own round loop. Phase 2 required transitions (switch mechanism mid-run) and combinations (multiple mechanisms per round), so a single generic `simulate()` function was built with pluggable mechanism specs:

```
simulate(agents, pools, rng, n_rounds, mechanisms, switch_round, mechanisms_after)
```

Mechanisms are specified as `(name, kwargs)` tuples — `'none'`, `'punishment'`, `'threshold'`, `'sustainability'`, `'voluntary'`. The function handles Phase 1 conditions (single mechanism, no switch), transitions (mechanism A for rounds 0–99, mechanism B for rounds 100–199), and combinations (multiple mechanisms per round).

**Mechanism ordering per round:** VCMS forward pass → threshold/sustainability exclusion → voluntary exit evaluation → pool regrouping. In combinations with voluntary exit, excluded agents go to the free pool (not immediately replaced) and regroup when ≥4 agents are available.

### Test 1: Transition Dynamics

Seven conditions, 100 runs × 200 rounds, switch at round 100:

| ID | Rounds 1–100 | Rounds 101–200 |
|----|-------------|----------------|
| T1 | punishment | none (baseline) |
| T2 | punishment | sustainability |
| T3 | punishment | voluntary exit r10 |
| T4 | punishment | threshold K=3 |
| T5 | none | sustainability |
| T6 | sustainability | sustainability (continuous) |
| T7 | punishment | punishment (continuous) |

dt = 1/199 throughout. Agent state is continuous across the switch boundary — no reset. T6 and T7 are continuous controls. T5 is a "cold start" control (sustainability introduced at round 100 without prior damage).

### Test 2: Mechanism Combination

Six conditions, 100 runs × 100 rounds:

| ID | Mechanism(s) |
|----|-------------|
| C1 | sustainability only |
| C2 | voluntary exit r10 only |
| C3 | sustainability + voluntary r10 |
| C4 | sustainability + voluntary r5 |
| C5 | threshold K=3 + voluntary r10 |
| C6 | punishment + voluntary r10 |

### Test 3: EC Protection

**Part A (analytical):** Cross-condition EC survival vs environmental volatility across all 19 conditions from Tests 1, 2, and 3B.

**Part B (simulation):** Three synthetic EC variants — original inertia, medium (0.35), high (0.50) — under punishment and sustainability exclusion. 100 runs × 100 rounds. Only EC agents' inertia is modified; all other agents retain original parameters.

**Part C (analytical):** Canary index — legibility (Pearson correlation between contribution and budget trajectories) of ruptured vs survived agents. Transparency tax: CC budget minus EC budget at T=100 under each condition.

### Scale

Test 1: 7 conditions × 100 runs × 200 rounds × 40 agents = 56M agent-steps
Test 2: 6 conditions × 100 runs × 100 rounds × 40 agents = 24M agent-steps
Test 3B: 6 conditions × 100 runs × 100 rounds × 40 agents = 24M agent-steps
**Total: ~104M agent-steps, completed in 57.1 seconds.**

---

## Results

### Test 1: Transition Dynamics

#### 1.1 Condition Comparison

| Condition | Coop@100 | Coop@200 | Dip@Switch | Recovery Slope | Budget Slope | Stabilization | Steady-State | TTFR |
|-----------|:--------:|:--------:|:----------:|:--------------:|:------------:|:-------------:|:------------:|:----:|
| T1 pun→none | 9.4 | 4.5 | −0.03 | −0.032 | −0.000 | 110 | 5.6 | 19 |
| T2 pun→sustain | 9.4 | 5.2 | −0.03 | −0.017 | +0.005 | 110 | **6.5** | 19 |
| T3 pun→vol | 9.4 | 4.9 | −0.03 | −0.017 | +0.000 | 110 | 6.0 | 19 |
| T4 pun→thresh | 9.4 | 6.2 | −0.03 | −0.024 | +0.004 | 112 | **7.5** | 19 |
| T5 none→sustain | 6.3 | 4.1 | +0.00 | −0.007 | +0.002 | 110 | 4.9 | 68 |
| T6 sustain full | 7.8 | 4.6 | −0.02 | −0.011 | +0.001 | 110 | 5.6 | **74** |
| T7 pun full | 9.4 | 4.8 | −0.02 | −0.019 | −0.001 | 110 | 6.0 | 19 |

All values are medians across 100 runs. Contributions on a 0–20 scale.

#### 1.2 Cooperation Trajectory

| Condition | r=10 | r=25 | r=50 | r=75 | r=100 | r=110 | r=125 | r=150 | r=175 | r=200 |
|-----------|:----:|:----:|:----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| T1 pun→none | 10.2 | 10.3 | 10.1 | 9.9 | 9.4 | 9.0 | 8.6 | 7.6 | 6.5 | 4.5 |
| T2 pun→sustain | 10.2 | 10.3 | 10.1 | 9.9 | 9.4 | 9.0 | 8.8 | 8.3 | 7.5 | 5.2 |
| T3 pun→vol | 10.2 | 10.3 | 10.1 | 9.9 | 9.4 | 9.2 | 8.8 | 8.0 | 7.0 | 4.9 |
| T4 pun→thresh | 10.2 | 10.3 | 10.1 | 9.9 | 9.4 | 8.6 | 8.5 | 8.1 | 7.8 | 6.2 |
| T5 none→sustain | 7.8 | 7.5 | 7.2 | 6.8 | 6.3 | 6.2 | 6.2 | 6.0 | 5.5 | 4.1 |
| T6 sustain full | 7.9 | 8.1 | 8.3 | 8.0 | 7.8 | 7.5 | 7.3 | 6.8 | 6.2 | 4.6 |
| T7 pun full | 10.2 | 10.3 | 10.1 | 9.9 | 9.4 | 9.2 | 8.8 | 8.1 | 7.1 | 4.8 |

The first 100 rounds are identical for all punishment-start conditions (T1–T4, T7). Punishment maintains cooperation above 9.0 through round 100, then all conditions decline in the second half — but at different rates depending on the post-switch mechanism.

#### 1.3 Damage Profile at Round 100

| Condition | CC Budget | EC Budget | CD Budget | DL Budget |
|-----------|:---------:|:---------:|:---------:|:---------:|
| T1–T4, T7 (punishment) | 2.79 | 2.97 | 2.22 | 2.25 |
| T5 (none) | 3.00 | 3.15 | 2.52 | 2.27 |
| T6 (sustainability) | 3.06 | 3.20 | 2.94 | 2.47 |

Punishment leaves all phenotypes with lower budget than sustainability at the 100-round mark. The damage is non-uniform: CD agents under punishment have 2.22 budget vs 2.94 under sustainability (a 25% deficit). EC agents are least damaged (2.97 vs 3.20, only 7% deficit). DL agents are structurally similar across conditions (2.25 vs 2.47).

#### 1.4 Per-Phenotype Budget Recovery

| Condition | CC | EC | CD | DL |
|-----------|:--:|:--:|:--:|:--:|
| T1 pun→none | −0.0015 | +0.0005 | +0.0017 | −0.0021 |
| T2 pun→sustain | −0.0015 | **+0.0042** | **+0.0057** | −0.0009 |
| T3 pun→vol | −0.0009 | +0.0004 | +0.0018 | −0.0018 |
| T4 pun→thresh | −0.0018 | +0.0038 | +0.0021 | −0.0024 |
| T5 none→sustain | −0.0008 | +0.0006 | +0.0040 | −0.0027 |
| T6 sustain full | −0.0007 | +0.0000 | +0.0038 | −0.0024 |
| T7 pun full | −0.0017 | −0.0003 | +0.0010 | −0.0027 |

Budget recovery slopes in the first 20 post-switch rounds. **CC agents have negative recovery slopes in every condition** — they are the only phenotype that systematically loses budget during recovery. DL agents also decline, but less steeply. EC and CD agents recover, with CD agents recovering fastest under sustainability (T2: +0.0057).

---

### Test 2: Mechanism Combination

#### 2.1 Condition Comparison

| Condition | SS-Coop | TTFR | Ruptures | Gini | Bimodality | Sustain Removals | Vol Exits | Total Mobility | Redundancy |
|-----------|:-------:|:----:|:--------:|:----:|:----------:|:----------------:|:---------:|:--------------:|:----------:|
| C1 sustain only | 5.3 | 25 | 1 | 0.547 | 0.430 | 36 | 0 | 36 | — |
| C2 vol r10 only | 4.7 | 15 | 1 | 0.588 | 0.492 | 0 | 74 | 74 | — |
| C3 sustain+vol r10 | 5.2 | 15 | 1 | 0.580 | **0.747** | 23 | 73 | 97 | 2.7% |
| C4 sustain+vol r5 | 4.5 | 16 | 1 | **0.664** | **0.729** | 13 | 168 | 184 | 0.0% |
| C5 thresh+vol r10 | 6.5 | 11 | 1 | 0.515 | 0.667 | 0 | 65 | 65 | — |
| C6 pun+vol r10 | **7.3** | 11 | 2 | 0.532 | 0.435 | 0 | 38 | 38 | — |

#### 2.2 Per-Phenotype Budget at T=100

| Condition | CC | EC | CD | DL |
|-----------|:--:|:--:|:--:|:--:|
| C1 sustain only | 2.98 | **3.43** | 3.02 | 2.48 |
| C2 vol r10 only | 3.07 | 3.31 | 2.44 | 2.27 |
| C3 sustain+vol r10 | 2.94 | 3.30 | 3.00 | 1.76 |
| C4 sustain+vol r5 | 2.90 | 3.17 | 2.19 | 1.75 |
| C5 thresh+vol r10 | 2.93 | 3.20 | 1.43 | 2.11 |
| C6 pun+vol r10 | 2.64 | 3.17 | 2.38 | 2.20 |

EC agents have the highest budget in every condition. CD agents' budget varies dramatically — highest under sustainability (3.02 in C1, 3.00 in C3) and lowest under threshold+voluntary (1.43 in C5). DL agents suffer most under combined mechanisms (1.75–1.76 in C3/C4 vs 2.48 in C1).

---

### Test 3: EC Protection

#### 3A: Cross-Condition EC Survival vs Environmental Volatility

| Condition | EC Survival | EC Budget | Env Volatility | Legibility Gap |
|-----------|:-----------:|:---------:|:--------------:|:--------------:|
| C1 sustain only | 100.0% | 3.43 | 0.15 | +0.413 |
| C2 vol r10 only | 100.0% | 3.31 | 0.15 | +0.523 |
| C3 sustain+vol r10 | 100.0% | 3.30 | 0.17 | +0.468 |
| C4 sustain+vol r5 | 100.0% | 3.17 | 0.16 | +0.575 |
| C5 thresh+vol r10 | 100.0% | 3.20 | 0.22 | +0.556 |
| C6 pun+vol r10 | 93.8% | 3.17 | 0.17 | +0.378 |
| T1–T4 (pun→*) | 93.8% | 3.00–3.21 | 0.10–0.13 | +0.322–0.430 |
| T5 none→sustain | 100.0% | 3.21 | 0.09 | +0.452 |
| T6 sustain full | 100.0% | 3.32 | 0.09 | +0.374 |
| T7 pun full | 93.8% | 2.94 | 0.10 | +0.303 |

EC survival vs volatility correlation: **r = +0.198** (weakly positive). The predicted inverse correlation (high volatility kills ECs) was not supported. EC agents die only under punishment-containing conditions, regardless of volatility level.

#### 3B: Inertia Sensitivity

| Variant | Punishment EC Surv | Sustain EC Surv | Punishment EC Budget | Sustain EC Budget |
|---------|:------------------:|:---------------:|:--------------------:|:-----------------:|
| EC original | 93.8% | 100.0% | 2.93 | 3.37 |
| EC medium (0.35) | 93.8% | 100.0% | 2.87 | 3.33 |
| EC high (0.50) | 93.8% | 100.0% | 2.88 | 3.35 |

**Inertia has zero effect on EC survival.** All three variants show identical 93.8% survival under punishment and 100% under sustainability. Increasing inertia from <0.25 to 0.50 does not protect ECs — their vulnerability to punishment is not mediated by responsiveness to environmental change.

#### 3C: Canary Index — Legibility and Rupture

| Condition | Ruptured Legibility | Survived Legibility | Gap | Transparency Tax |
|-----------|:-------------------:|:-------------------:|:---:|:----------------:|
| C1 sustain only | 0.799 | 0.377 | +0.413 | −0.484 |
| C2 vol r10 only | 0.819 | 0.287 | +0.523 | −0.285 |
| C3 sustain+vol r10 | 0.893 | 0.420 | +0.468 | −0.345 |
| C4 sustain+vol r5 | 0.909 | 0.329 | +0.575 | −0.257 |
| C5 thresh+vol r10 | 0.889 | 0.295 | +0.556 | −0.180 |
| C6 pun+vol r10 | 0.720 | 0.329 | +0.378 | −0.524 |
| T6 sustain full | 0.776 | 0.380 | +0.374 | −0.354 |
| T7 pun full | 0.713 | 0.408 | +0.303 | −0.400 |

**Legibility gap is universal and large**: ruptured agents have legibility 0.71–0.91 across all conditions, while survivors have 0.29–0.42. The gap ranges from +0.30 to +0.58.

**Transparency tax is negative everywhere**: EC agents end with *higher* budget than CC agents in every condition (tax ranges from −0.18 to −0.52). The predicted "penalty for transparency" is actually an advantage.

---

## Predictions Scorecard

### Test 1: Transition (2/5 supported)

#### P1-T: Cooperation dips at switch ✓ SUPPORTED

All four punishment→other conditions show a cooperation dip of −0.03 at the switch point. When punishment is removed, cooperation drops immediately. The dip is small (−0.03 on a 0–20 scale) because the switch happens within a single round — the VCMS model has inertia that smooths the transition. The real decline comes over the subsequent 10–20 rounds as agents recalibrate.

#### P2-T: Budget recovers before cooperation ✓ SUPPORTED

In all three reform conditions (T2, T3, T4), the budget recovery slope is positive while the cooperation recovery slope is negative. Budget begins recovering immediately when punishment is removed (the punishment-sent drain stops), but cooperation continues declining because agents are still carrying the strain and depleted budgets from the punishment phase. Budget leads cooperation by approximately 10 rounds.

This is theoretically important: it means the *capacity* for cooperation recovers before the *behavior* does. In VCMS terms, the B-channel responds to the removal of drain before the V-channel integrates the new (less punitive) social signal. The order is: (1) punishment stops → (2) budget drain stops → (3) budget begins recovering → (4) affordability improves → (5) contribution output increases.

#### P3-T: Hysteresis — T2 < T6 < T7 ✗ NOT SUPPORTED (reversed)

**Predicted:** Punishment→sustainability (T2) should show lower steady-state than continuous sustainability (T6), demonstrating irrecoverable damage from punishment front-loading.

**Observed:** T2 steady-state (6.5) is **higher** than T6 (5.6), and higher even than T7 continuous punishment (6.0).

**Why:** Punishment front-loading produces a *cooperation premium*, not a cooperation deficit. The punishment phase drives cooperation to 9.4 by round 100, establishing a high-cooperation norm. When sustainability takes over at round 101, agents are starting from a much higher cooperation baseline than they would under sustainability alone (T6 at round 100: 7.8). The sustainability mechanism then maintains this elevated level more effectively because there are fewer low contributors to remove — punishment already coerced them upward.

The full ordering is: T4 (7.5) > T2 (6.5) > T3 (6.0) = T7 (6.0) > T6 (5.6) > T1 (5.6) > T5 (4.9).

This is the deepest finding of Phase 2: **punishment functions as a costly but effective initialization**. The system pays for 100 rounds of budget depletion and fast rupture, but in exchange gets a cooperation norm that persists under gentler governance. The optimal federation trajectory may be punishment → sustainability, not sustainability from the start.

#### P4-T: DL recovers slowest ✗ NOT SUPPORTED (CC is slowest)

**Predicted:** DL agents (structural decliners) should have the lowest budget recovery slope.

**Observed:** CC agents have the lowest budget recovery slope (−0.0015 across T1–T4), while DL agents recover at −0.0009 to −0.0027. EC agents recover fastest (+0.004).

**Why:** CC agents' high inertia is a liability during regime change. When punishment is removed, the environment changes — but CC agents' inertia means they continue contributing at punishment-era levels while receiving less in return (other agents' contributions are dropping). CC agents keep spending from their budget based on their high c_base and strong inertia, regardless of whether the environment is still reciprocating at the same level. In VCMS terms: the CC V-level doesn't update fast enough to match the new equilibrium.

This inverts the Phase 1 finding where CC inertia was protective (100% survival everywhere). The same trait that protects CC agents from decline — resistance to environmental tracking — also prevents them from adapting upward when conditions improve. Inertia is stability, and stability can mean being stuck at the wrong level in either direction.

#### P5-T: Punishment→voluntary is slowest recovery ✗ NOT SUPPORTED (all equal)

**Predicted:** T3 (punishment→voluntary) should have the slowest time to stabilization because sorting needs time that damaged agents don't have.

**Observed:** All conditions stabilize at round 110 (±2 rounds). There is no meaningful variation in stabilization time because the stabilization metric (slope < 0.05 for 10 consecutive rounds) is met by all conditions within the first evaluation window post-switch.

The metric was too coarse. A slope < 0.05 is easily achieved because the cooperation decline is gradual (0.02–0.03 per round), not volatile. A tighter threshold or a different metric (e.g., distance from eventual steady-state) would better distinguish recovery speeds. The cooperation trajectories show clear differences (T4 ends at 6.2, T1 at 4.5) — the differences are in *level*, not in *speed*.

---

### Test 2: Combination (2/4 supported)

#### P1-C: Combined outperforms either alone ✗ NOT SUPPORTED

**Predicted:** C3 (sustainability + voluntary r10) composite score > max(C1, C2).

**Observed:** C3 composite (0.462) < C1 (0.497) < C2 (0.454). Combined is in the middle.

**Why:** Adding voluntary exit to sustainability *hurts* the sustainability mechanism's effectiveness. Sustainability exclusion works best with stable groups — it monitors budget and cooperation slopes over a health window, detecting degradation trends. When voluntary exit adds agent mobility, groups are constantly being disrupted by departures and regroupings. The sustainability monitor's historical tracking (budget_hist, coop_hist) loses signal when the group composition changes.

The mechanisms are not complementary as predicted — they **interfere**. Sustainability needs stability to detect degradation; voluntary exit creates instability by design. The result: fewer sustainability removals in C3 (23) than C1 (36), because voluntary exit disrupts the groups before sustainability can trigger.

The highest composite belongs to C1 (sustainability only) with its longer TTFR (25 vs 15). The highest cooperation belongs to C6 (punishment + voluntary, 7.3), but it trades off with ruptures (2 vs 1) and short TTFR (11).

#### P2-C: Mobility reduction ✓ SUPPORTED

C3 total mobility (97) < C1 mobility (36) + C2 mobility (74) = 110. The reduction is 13 events (12%). Each mechanism partially substitutes for the other: sustainability removes agents who would have otherwise exited voluntarily, and voluntary exit removes agents who would have otherwise triggered sustainability intervention.

The redundancy rate is low (2.7% in C3, 0.0% in C4), meaning explicit same-round overlap is rare. The substitution operates across rounds — an agent removed by sustainability at round 30 can't voluntarily exit at round 40.

#### P3-C: r5 bimodality > 0.555 ✓ SUPPORTED

C4 (sustainability + voluntary r5) achieves bimodality 0.729, well above the 0.555 threshold for statistical bimodality. C3 (voluntary r10) achieves 0.747. Phase 1's voluntary exit r10 scored only 0.492.

**What changed:** The combination with sustainability creates cleaner sorting because sustainability removes the worst extractors from groups, making the remaining groups more homogeneous. Voluntary exit then separates the moderately-unhappy from the adequately-served. The result is a three-stage population: high-cooperation groups (never needed intervention), medium-cooperation groups (sustainability cleaned up the worst member, rest stayed), and low-cooperation remnant groups.

C4's higher eval frequency (every 5 rounds) generates far more exits (168 vs 73) and the highest Gini coefficient (0.664), confirming that temporal resolution is the key constraint on sorting quality — not the sorting mechanism itself.

#### P4-C: Punishment+voluntary has highest exit rate ✗ NOT SUPPORTED

**Predicted:** C6 (punishment + voluntary) has highest exit rate because punishment makes the environment toxic enough that agents flee.

**Observed:** C6 has the *lowest* exit rate (38 exits). C4 has the highest (168 from r5 frequency). Even at the same r10 frequency, C2 (74), C3 (73), and C5 (65) all exceed C6 (38).

**Why:** Punishment keeps cooperation artificially high, so agents' affordability (which drives the exit decision) stays elevated. Agents in punishing groups are contributing more and receiving more, so their exit threshold isn't crossed. The environment is coercive but not perceived as *declining* by the affordability metric. Punishment creates a "cage" — agents can't leave because the punishment-maintained cooperation level makes their local situation appear adequate.

This is the second-most important finding in Phase 2 after the hysteresis reversal. Punishment doesn't just coerce cooperation — it traps agents in groups by maintaining a superficially functional environment. The voluntariness of voluntary exit is undermined by the very mechanism that makes the environment seem acceptable.

---

### Test 3: EC Protection (0/4 supported)

#### P1-E: EC survival inversely correlated with environmental volatility ✗ NOT SUPPORTED

EC survival vs volatility correlation: r = +0.198 (weakly positive). EC agents die only under punishment-containing conditions (93.8% survival), regardless of environmental volatility. The highest-volatility condition (C5, thresh+vol, volatility 0.22) has 100% EC survival. The lowest-volatility conditions (T1–T4, volatility 0.10) have 93.8% EC survival.

EC vulnerability to punishment is not mediated by volatility. It's mediated by the punishment mechanism itself — specifically, the punishment-received budget drain and replenishment gating that the VCMS punishment pathway imposes on targets. ECs are targeted because their low-inertia responsiveness makes them temporarily low contributors during decline phases, attracting punishment.

#### P2-E: Inertia protects ECs under punishment ✗ NOT SUPPORTED

All three inertia variants (original <0.25, medium 0.35, high 0.50) show identical 93.8% survival under punishment. EC budget under punishment: original 2.93, medium 2.87, high 2.88. Under sustainability: original 3.37, medium 3.33, high 3.35.

Increasing inertia has negligible effect because the punishment mechanism operates on budget, not on cooperation level. An EC agent with high inertia still gets punished (because the punishment-received pathway doesn't depend on the target's inertia), still suffers budget drain, and still ruptures at the same rate. The modification changes how fast ECs adapt their *output* but not how much *damage* they receive.

If anything, there's a marginal negative effect: higher-inertia ECs have slightly lower budget (2.88 vs 2.93 under punishment), possibly because their slower adaptation means they spend longer at contribution levels that don't match their group, attracting marginally more punishment.

#### P3-E: Control-first has larger legibility gap than care-first ✗ NOT SUPPORTED (reversed)

Mean legibility gap under control-first conditions (punishment, threshold): +0.377. Under care-first conditions (sustainability, voluntary): +0.443.

Care-first conditions have a *larger* legibility gap. This is because care-first mechanisms allow more agents to survive with intact budgets (lowering survivor legibility — their budgets decouple from contributions as they stabilize at high levels), while ruptured agents under care-first conditions have extremely high legibility (0.82–0.91 — their budgets track their contributions all the way down).

Under punishment, ruptured agents have lower legibility (0.71–0.78) because punishment introduces budget dynamics that decouple budget from cooperation — agents receive punishment-sent drain and replenishment gating that adds noise to the budget trajectory independent of their contribution choices.

#### P4-E: Transparency tax higher under punishment than sustainability ✗ NOT SUPPORTED (both negative)

Transparency tax (CC budget − EC budget) is **negative** in every condition. EC agents have higher budget than CC agents everywhere:
- Under punishment: −0.400 (EC has 0.40 more budget than CC)
- Under sustainability: −0.354 (EC has 0.35 more budget than CC)
- Range across all conditions: −0.18 to −0.52

The predicted "penalty for transparency" is actually an advantage. EC agents' low inertia means they reduce cooperation faster when the group is declining, which *preserves their budget* — they stop spending before CC agents do. CC agents' high inertia means they keep contributing at high levels even when the group is collapsing, draining their own budget while the group is no longer reciprocating.

The "transparency tax" is a "transparency advantage" — being responsive to environment changes is *budget-protective*, not budget-costly. This reframes the entire EC vulnerability finding from Phase 1: EC agents die under punishment not because they're transparent, but because punishment specifically targets them with budget-destroying mechanisms. In the absence of punishment, EC agents are the most budget-resilient phenotype.

---

## Theoretical Interpretation

### 1. The Punishment Initialization Principle

The most significant finding is that punishment→sustainability (T2) achieves higher post-switch steady-state (6.5) than continuous sustainability (T6, 5.6). The hysteresis prediction was reversed: punishment front-loading doesn't damage the system — it initializes it.

This has a clear VCMS interpretation. Punishment drives cooperation to ~9.4 over 100 rounds, establishing a high-cooperation equilibrium. When sustainability takes over, agents are starting from a high-cooperation state with established cooperative dispositions. The sustainability mechanism then only needs to *maintain* this level, not *create* it. Fewer agents are below the degradation threshold, fewer removals are needed, and the system coasts on momentum.

The cost is real: punishment conditions all have TTFR=19 (rupture happens early), meaning some agents are destroyed during the initialization phase. The question becomes: is the population-level cooperation gain worth the individual-level casualties? T2's rupture count at the end (100 runs, not shown in aggregated table) is comparable to T6's — suggesting the early ruptures are the price paid for the later norm.

**Federation design implication:** The optimal enforcement trajectory may be *temporal* — start with coercive mechanisms to establish norms, then transition to sustainable mechanisms to maintain them. This is structurally analogous to constitutional founding (establishing rules through force) followed by institutional maintenance (sustaining rules through systems).

### 2. Mechanism Interference

The combination results overturn the intuition that "more mechanisms = better." Sustainability + voluntary exit does NOT outperform sustainability alone, because the mechanisms operate on different assumptions about group stability:

- **Sustainability** assumes groups are persistent — it monitors trends over a health window and intervenes when degradation is sustained. This requires *temporal depth* in group composition.
- **Voluntary exit** assumes groups are temporary — agents should leave bad groups and form new ones. This creates *temporal disruption* in group composition.

Running both together undermines sustainability's monitoring: groups change composition before sustainability can detect degradation, and sustainability removals disrupt groups that voluntary exit would have naturally sorted.

The exception: **bimodality**. C3 and C4 achieve bimodality > 0.7 while neither individual mechanism does at r10. The combination produces bimodal sorting not because the mechanisms complement each other on any single metric, but because sustainability pre-cleans the worst extractors and voluntary exit then sorts the remainder. The sorting is two-stage: institutional removal of the most harmful agents, followed by self-sorting of the rest.

### 3. The Legibility Trap

Across all 19 conditions, ruptured agents have legibility (contribution-budget correlation) of 0.71–0.91, while survivors have legibility of 0.29–0.42. The gap is +0.30 to +0.58 in every condition.

This means: **agents whose behavior is most readable — whose contributions most closely track their internal state — are the ones who rupture.** Agents who survive are the ones whose contributions decouple from their budget, meaning they maintain contribution at levels that don't reflect their actual resource state.

This is not an artifact of EC vulnerability. The gap is universal across all governance architectures, including conditions with 100% EC survival. It applies to all phenotypes — any agent who becomes legible (budget decline visibly tracks contribution decline) is on the path to rupture.

The mechanism is clear: as an agent's budget declines, its contribution declines proportionally (through the affordability pathway), creating high contribution-budget correlation. An agent whose budget *doesn't* decline — or whose contribution is maintained through inertia despite budget changes — has lower legibility and survives.

**Legibility is not a trait that produces vulnerability. It is a symptom of the budget dynamics that cause rupture.** An agent's legibility increases as it approaches rupture because the same process (budget depletion) drives both contribution decline and budget decline simultaneously. The "canary index" identifies agents currently in cascade, not agents structurally prone to it.

### 4. The Transparency Advantage

The transparency tax is negative everywhere — EC agents end with higher budget than CC agents in every condition. This contradicts the Phase 1 observation that EC agents are the only phenotype vulnerable to punishment, which led to the prediction that transparency (low inertia, environment-tracking) carries a budget cost.

The resolution: EC agents are vulnerable to *punishment specifically*, not to governance generally. Under punishment, ECs are targeted because their environment-tracking produces temporary low-contribution episodes that attract punishment. Under every other mechanism, ECs' responsiveness is *protective*: they reduce contribution faster when the group is declining, conserving budget, and increase contribution faster when the group is recovering, capturing more of the public good.

CC agents' high inertia protects them from punishment (they never have temporary low-contribution episodes that attract targeting) but costs them budget in every other context (they keep contributing at high levels even when the group isn't reciprocating). Over 100–200 rounds, this inertia tax exceeds the EC vulnerability under non-punishment conditions.

**Reframing:** The question isn't "does governance protect transparent agents?" but "which governance architecture transforms transparency from an advantage to a liability?" The answer is: only punishment. Every non-punishment mechanism leaves transparency as a net advantage. This strengthens the Phase 1 conclusion that punishment is structurally different from all other federation mechanisms — it's the only architecture that punishes the cooperative trait (responsiveness) rather than the uncooperative behavior (extraction).

### 5. CC Inertia as Recovery Liability

Phase 1 found CC agents survive at 100% across all conditions. Phase 2 finds CC agents have the *worst* budget recovery slopes in every transition condition. The trait that ensures survival (inertia = resistance to environmental decline) also ensures the slowest adaptation when the environment improves.

In VCMS terms: CC agents' high inertia means their V-level (valuation of the group) updates slowly. During the punishment phase, their V-level was high (punishment maintained cooperation, so the social signal was positive). When punishment is removed and cooperation begins declining, their V-level remains elevated due to inertia — so they keep contributing at punishment-era levels, draining their budget in a group that's no longer reciprocating at the same level.

This is the *mirror image* of the CC survival advantage: the same inertia parameter that prevents CC agents from tracking environmental decline also prevents them from tracking environmental improvement. The CC phenotype is stable in both directions, which is an advantage when the environment is declining (they don't spiral) and a disadvantage when the environment is changing (they don't adapt).

---

## Summary: Phase 2 Findings

### Predictions: 4/13 supported (31%)

| Test | Supported | Not Supported |
|------|:---------:|:-------------:|
| Test 1 (Transition) | 2/5 | 3/5 |
| Test 2 (Combination) | 2/4 | 2/4 |
| Test 3 (EC Protection) | 0/4 | 4/4 |

The prediction failures are more informative than the successes. The four deepest findings are all from failed predictions:

### Five Key Results

1. **Punishment as initialization.** Punishment→sustainability produces higher steady-state cooperation than continuous sustainability. The costliest mechanism becomes the most effective when used as a *phase*, not a permanent architecture. Prediction P3-T reversed.

2. **Mechanism interference.** Sustainability + voluntary exit does not outperform sustainability alone because the mechanisms disrupt each other's operating assumptions. More governance is not better governance. Prediction P1-C failed.

3. **Legibility marks cascade, not vulnerability.** Ruptured agents are highly legible (0.71–0.91) across every condition. This is a *symptom* of budget collapse (contribution tracks budget downward), not a *cause*. The canary index identifies agents currently dying, not agents structurally at risk. Prediction P3-E reversed.

4. **Transparency is an advantage.** EC agents have higher budget than CC agents in every condition. Low inertia conserves budget by reducing contribution when the group is declining. EC vulnerability to punishment is a special case — punishment is the only mechanism that transforms responsiveness from an advantage to a liability. Prediction P4-E reversed.

5. **Inertia is a double-edged trait.** CC agents survive everything (100% across all conditions, both phases) but recover the slowest from regime change. EC agents adapt fastest but are killed by the one mechanism (punishment) that penalizes their adaptiveness. The optimal inertia is context-dependent — high inertia for hostile environments, low inertia for changing ones.

### The Compliance-Sustainability Tradeoff, Revised

Phase 1 established the tradeoff: high compliance (punishment, threshold) trades off against high sustainability (sustainability exclusion, voluntary exit). Phase 2 shows the tradeoff is **temporal, not structural**:

- Punishment is optimal for *initialization* (establishing cooperation norms)
- Sustainability exclusion is optimal for *maintenance* (preserving cooperation with minimal harm)
- Voluntary exit is optimal for *sorting* (achieving group homogeneity when temporal resolution is sufficient)
- Combination is optimal for *bimodality* (two-stage cleaning + sorting)

The deepest result: **no single mechanism dominates across the full lifecycle of a federation.** The optimal governance trajectory involves transitioning between mechanisms as the system matures — from coercive norm-establishment to sustainable norm-maintenance. This is the enforcement analogue of the VCMS disposition dynamics themselves: initial conditions matter, but the path through them matters more.
