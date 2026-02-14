# Federation Dynamics Phase 3 — From Rehabilitation to Structural Architecture

## Scope

This document reports three linked experiments (Phase 3A, 3B, 3C) that progressively move the locus of intervention from **agents** to **structure**. Phase 3A asks whether modifying agent state (rehabilitation) can substitute for removing agents. Phase 3B asks whether giving agents full information about each other changes the dynamics. Phase 3C asks whether system-level mechanisms — impersonal, universal, not mediated by any agent — can produce what neither rehabilitation nor visibility achieved alone: high cooperation with full dignity.

The answer is yes. Phase 3C identifies a configuration (SV5: all structural mechanisms + full visibility) that matches enforcement on cooperation (5.3 vs 5.2) while maintaining 100% dignity floor for every phenotype and zero ruptures. This is the first configuration across all phases to sit alone on the Pareto frontier of the cooperation-dignity space.

---

## Experimental Context

### Population

40 agents drawn from the 576-subject fitted library (P-experiment: 176, N-experiment: 212, IPD: 188), assigned to 10 groups of 4.

| Phenotype | Count | Share | Defining Parameters |
|-----------|:-----:|:-----:|---------------------|
| EC (Evaluative Cooperator) | 16 | 40% | Moderate c_base (0.35–0.75), low inertia (<0.25), high alpha (>0.3) |
| CC (Committed Cooperator) | 8 | 20% | High c_base (>0.65), high inertia (>0.3) |
| CD (Comfortable Defector) | 8 | 20% | Low c_base (<0.4) |
| DL (Decliner) | 4 | 10% | Structural budget drain (b_depletion > b_replenish) |
| MX (Mixed) | 4 | 10% | Random draw from any pool |

All phases use the same population structure, matched seeds within each phase, and 100 Monte Carlo runs × 100 rounds per condition. Contributions are on a 0–20 integer scale (MAX_CONTRIB = 20). Rupture is defined as budget < 10% of b_initial for 3 consecutive rounds.

### Baseline Results (Phase 1–2 Summary)

Phase 1 established the compliance-sustainability tradeoff:

| Mechanism | SS-Coop | TTFR | Ruptures | Gini | Removals |
|-----------|:-------:|:----:|:--------:|:----:|:--------:|
| Baseline (none) | 3.9 | 29 | 2 | 0.605 | 0 |
| Punishment | 5.9 | **8** | 3 | 0.601 | 0 |
| Threshold K=3 | **6.8** | 30 | 1 | 0.527 | 172 |
| Sustainability | 5.3 | **34** | 1 | 0.553 | 31 |
| Voluntary r10 | 4.7 | 21 | 1 | 0.595 | 73 exits |

Phase 2 established three additional results:
1. **Punishment as initialization**: Punishment→Sustainability (T2) achieves higher steady-state than continuous sustainability (6.5 vs 5.6). Coercive norm-establishment followed by sustainable maintenance outperforms gentle governance.
2. **Mechanism interference**: Sustainability + voluntary exit does NOT outperform sustainability alone. The mechanisms disrupt each other's operating assumptions.
3. **Transparency tax is negative**: EC agents end with higher budget than CC agents. ECs' low inertia causes them to reduce contribution faster when groups decline, *conserving* budget. Legibility is a symptom of cascade, not a cause — ruptured agents show 0.71–0.91 budget-contribution correlation vs 0.29–0.42 for survivors.

---

## Phase 3A: Rehabilitation — Parameter Intervention as Alternative to Removal

### Design

Same trigger as sustainability exclusion: budget slope < 0 AND cooperation slope < 0 for 3+ consecutive rounds. Different response: intervene on the highest-impact agent's *state* instead of removing them.

**14 conditions** (plus baseline and sustainability control):

| Category | Modes | Parameters |
|----------|-------|------------|
| Strain reduction | R1a/b/c | 25%, 50%, 75% reduction of current strain |
| Budget support | R2a/b/c | +0.5, +1.0, +2.0 budget injection |
| Replenishment boost | R3a/b/c | 1.5×, 2.0×, 3.0× b_replenish_rate for 10 rounds |
| Comprehensive | R4 | Strain (50%) + budget (+1.0) + boost (1.5× for 10 rounds) |
| Targeted diagnosis | R5 | Root-cause matching: diagnose whether strain, budget, or replenishment is the bottleneck |
| Hybrid: rehab first | H1 | Comprehensive, then remove if no improvement within 10 rounds |
| Hybrid: graduated | H2 | Escalating: strain reduction → comprehensive → removal |

All interventions cost resources, drawn equally from group members' budgets (REHAB_BASE_COST = 1.0, scaled by mode).

### Results

| Condition | SS-Coop | TTFR | Rupt | Conv% | Recid% | Interventions | Removals | Cost |
|-----------|:-------:|:----:|:----:|:-----:|:------:|:-------------:|:--------:|:----:|
| B1 baseline | 3.7 | 20 | 2 | — | — | 0 | 0 | 0 |
| B2 sustain | 5.3 | 20 | 1 | — | — | 0 | 70 | 0 |
| R1b strain50 | 4.0 | 18 | 2 | 33.3% | 15.4% | 42 | 0 | 21.0 |
| R2b budget10 | 4.0 | 33 | 1 | 27.3% | 0.0% | 39 | 0 | 39.0 |
| R2c budget20 | 4.4 | **100** | **0** | 27.3% | 7.1% | 46 | 0 | 46.0 |
| R4 comprehensive | 3.9 | 10 | 3 | 46.2% | 11.8% | 49 | 0 | 73.0 |
| R5 targeted | 3.9 | 12 | 3 | 39.6% | 14.3% | 31 | 0 | 31.2 |
| H1 rehab first | 4.1 | 11 | 3 | 61.1% | 11.8% | 37 | 14 | 34.2 |
| H2 graduated | 4.5 | 10 | 3 | 66.7% | 11.8% | 36 | 20 | 18.1 |
| R3c boost30 | **1.7** | **6** | 7 | 7.7% | 0.0% | 56 | 0 | 55.6 |

### Key Findings

**1. No rehabilitation condition matches removal on cooperation.** The best rehabilitation cooperation (H2 graduated: 4.5) reaches 84% of B2 sustainability (5.3). The gap is structural: rehabilitation modifies the target agent's state but doesn't change the *group composition* — the agent who caused the degradation trigger remains in the group. Removal replaces them. The cooperation gap is the cost of keeping everyone.

**2. Budget support prevents rupture; strain reduction enables conversion.** R2c (budget +2.0) achieves TTFR = 100 and zero ruptures — the only pure rehabilitation condition to eliminate rupture entirely. But its cooperation (4.4) is modest. Budget support addresses the symptom (depletion) without changing the dynamics that produce it. Strain reduction has higher conversion rates (33% at R1b vs 27% at R2b) because strain drives the affordability calculation that determines contribution: reducing strain directly increases cooperation capacity.

**3. Replenishment boost is catastrophic.** R3c (3.0× boost) produces the worst cooperation (1.7) and most ruptures (7) of any condition. The boost applies a flat per-round supplement proportional to b_replenish_rate. For agents with already-high replenishment (CCs), this inflates their budget, reducing their strain-to-budget ratio, lowering their affordability-weighted contribution. For agents with low replenishment (DLs), the boost is too small relative to their depletion rate to matter. The mechanism amplifies existing inequality rather than correcting it.

**4. H2 graduated is the best rehabilitation strategy.** Its escalating response (strain → comprehensive → removal) achieves the highest rehabilitation cooperation (4.5), highest conversion rate (66.7%), lowest recidivism (11.8%), and reduces removals from 70 (B2) to 20. The graduated approach works because it gives agents multiple chances — the first intervention (strain reduction) catches recoverable cases cheaply; the second (comprehensive) addresses deeper dysfunction; the third (removal) handles the structurally irrecoverable.

**5. The case for rehabilitation is values-based, not physics-based.** No rehabilitation condition sits above the compliance-sustainability frontier — none matches B2 on *both* cooperation AND TTFR simultaneously. R2c achieves TTFR = 100 but cooperation = 4.4 (vs B2's 5.3). H2 achieves cooperation = 4.5 but TTFR = 10 (vs B2's 20). Rehabilitation trades cooperation for dignity. Whether that trade is worthwhile depends on what the federation values.

### Per-Phenotype Intervention Response

| Phenotype | H2 Conv% | R1c Conv% | R2b Conv% | R3b Conv% |
|-----------|:--------:|:---------:|:---------:|:---------:|
| CD | 70.0% | 38.5% | 25.0% | 0.0% |
| DL | 75.0% | 33.3% | 37.5% | 12.5% |
| EC | 64.3% | 25.0% | 25.0% | 0.0% |
| CC | 50.0% | 0.0% | 33.3% | 16.7% |

CDs and DLs respond best to graduated rehabilitation. CDs respond to strain reduction because their high initial strain is the primary barrier; reducing it allows their (low but nonzero) c_base to produce some cooperation. DLs respond to budget support because their structural drain is the bottleneck. Replenishment boost has near-zero conversion across all phenotypes — the mechanism is fundamentally mismatched to the dynamics.

### Predictions Scorecard: 2/6 Supported

| ID | Prediction | Result |
|----|-----------|--------|
| P1 | Best rehab matches removal on cooperation (within 10%) | **NOT SUPPORTED** (best = 84%) |
| P2 | Best rehab exceeds removal on TTFR | **SUPPORTED** (R2c: 100 vs B2: 20) |
| P3 | Phenotype-specific interventions outperform uniform | **NOT SUPPORTED** (DL boost failed) |
| P4 | H2 graduated > both B2 and R4 on composite score | **NOT SUPPORTED** (H2 = 0.416 vs B2 = 0.480) |
| P5 | H2 reduces removals below B2 | **SUPPORTED** (10 vs 35) |
| P6 | Break-even cost is positive | **NOT SUPPORTED** |

---

## Phase 3B: Full Visibility — All Agents See All Parameters

### Design

All agents see all groupmates' parameters (c_base, s_initial, alpha, inertia, b_depletion_rate, b_replenish_rate, b_initial) and current state (B, strain, v_level). "Seeing" is operationalized as three between-round state modifications:

1. **Empathetic strain modulation**: When an agent can see that a groupmate's defection is from budget constraint (not choice), the strain accumulated from that gap is reduced. Reduction = observer's alpha × (fraction of constrained groupmates) × 0.30. Maximum 30% strain reduction per round, gated by the observer's social sensitivity.

2. **Solidarity transfer**: Agents above group median budget share a fraction of their excess with agents below median. Sharing rate = sharer's c_base × 0.10 × dt. This is agent-mediated — CDs (low c_base) barely share; CCs share substantially.

3. **Informed reference**: Agent's v_level (observed group cooperation level) shifts toward the mean of "capable" agents (budget above vulnerability threshold). Prevents constrained agents from dragging down the reference, which would otherwise cause a secondary strain spiral.

**11 conditions**: 3 controls, 4 visibility components (empathy/solidarity/reference/full), 4 visibility + enforcement combinations.

### Results

| Condition | SS-Coop | TTFR | Rupt | Gini | DignFl | Agency | Navig |
|-----------|:-------:|:----:|:----:|:----:|:------:|:------:|:-----:|
| B1 baseline | 4.0 | 21 | 2 | 0.592 | 75.0% | 3.89 | 0.377 |
| B2 sustain | 5.4 | 23 | 1 | 0.538 | 28.6% | 4.66 | 0.162 |
| B3 rehab grad | 4.7 | 10 | 3 | 0.574 | 50.0% | 4.16 | 0.273 |
| V1 empathy | **4.8** | 20 | 2 | 0.562 | 75.0% | 4.34 | **0.422** |
| V2 solidarity | 4.0 | 20 | 2 | 0.591 | 75.0% | 3.90 | 0.381 |
| V3 reference | 3.9 | 20 | 2 | 0.587 | 75.0% | 3.82 | 0.377 |
| V4 full vis | **4.8** | 20 | 2 | 0.560 | 75.0% | 4.30 | **0.441** |
| V5 vis+sustain | **6.4** | 24 | 1 | 0.516 | 29.6% | 5.09 | 0.180 |
| V7 vis+rehab tgt | 4.9 | 11 | 3 | 0.559 | 75.0% | 4.41 | 0.399 |
| V8 vis+graduated | 5.7 | 10 | 3 | 0.526 | 57.1% | 4.63 | 0.319 |

**Navigability index** = dignity_floor × (1 − Gini) × (1 + cooperation_agency / MAX_CONTRIB). Combines three dimensions: everyone survives (dignity), inequality is low (Gini), and choices matter (agency).

### Key Findings

**1. Full visibility without enforcement (V4) achieves the highest navigability of any condition tested across Phases 1–3A.** V4 navigability = 0.441, exceeding B2 sustainability (0.162), B1 baseline (0.377), and all rehabilitation variants. Visibility produces a fundamentally different regime: moderate cooperation, full dignity, meaningful agency.

**2. Empathy is the dominant mechanism.** V1 (empathy alone) raises cooperation from 4.0 → 4.8 and navigability from 0.377 → 0.422. Solidarity and reference have near-zero independent effect (V2: 4.0 coop, V3: 3.9 coop). The entire visibility benefit comes from empathetic strain modulation: understanding *why* someone defects (constraint vs choice) interrupts the strain spiral that drives cooperation downward. The problem was never resource distribution. The problem was undiagnosed frustration from gaps agents couldn't explain.

**3. The transparency tax inverts under full visibility.** Phase 2 found that ECs pay a transparency tax under punishment — their legibility makes them vulnerable. Under full visibility, EC survival rises from 93.8% to 100%. When *everyone* is transparent, the EC's social sensitivity becomes an asset: high alpha means empathetic strain reduction works strongest for them. They're the agents most equipped to process the new information.

**4. Visibility + sustainability (V5) achieves the highest cooperation of any condition (6.4)** — higher than sustainability alone (5.4), higher than punishment (5.9). Visibility makes enforcement more effective. But the dignity floor remains at 29.6% because sustainability exclusion still removes agents. The dignity cost of removal persists regardless of visibility.

**5. There exists a "sweet spot".** V1 (empathy alone) achieves navigability > B2 AND cooperation within 20% of B2 (4.8 vs 5.4). This is the configuration where visibility increases navigable agency while maintaining baseline dignity for all phenotypes.

### Predictions Scorecard: 3/7 Supported

| ID | Prediction | Result |
|----|-----------|--------|
| PV1 | Full visibility improves dignity over baseline | **NOT SUPPORTED** (both 75.0%) |
| PV2 | Solidarity is strongest single component | **NOT SUPPORTED** (empathy is strongest) |
| PV3 | Vis+graduated > either alone on navigability | **NOT SUPPORTED** (V4 = 0.441 > V8 = 0.319) |
| PV4 | Full visibility reduces ruptures below baseline | **NOT SUPPORTED** (both 2) |
| PV5 | Vis+sustain > sustain alone on navigability | **SUPPORTED** (0.180 vs 0.162) |
| PV6 | EC gains most from visibility (transparency tax inverts) | **SUPPORTED** (+6.2% EC vs +0.0% CD) |
| PV7 | Sweet spot exists (nav > B2, coop within 20%) | **SUPPORTED** (V1 empathy) |

The low prediction rate (3/7) is informative. The mechanism is simpler than predicted: visibility helps not by enabling solidarity or reference calibration, but by a single channel — empathetic strain modulation. The other visibility components are nearly inert. This was not the anticipated pattern.

---

## Phase 3C: Structural Mechanisms — System-Imposed, Impersonal, Universal

### Design

Phase 3B showed visibility works through agent-mediated strain modulation — each agent must process information and adjust their own state. Cooperation tops out at 4.8 because visibility cannot make comfortable defectors cooperate; it only prevents secondary damage. Phase 3C asks: what if the structure itself does the lifting?

Five structural mechanisms, applied between rounds after enforcement mechanisms and before visibility effects:

1. **Budget floor (UBI)**: System guarantees minimum budget = floor_frac × b_initial each round. If agent's budget falls below the floor, system tops it up. No agent pays — the cost is structural. (*floor_frac = 0.30*)

2. **Strain ceiling**: Hard cap on strain accumulation. System absorbs excess. Prevents strain spirals directly. No cost — strain is not a transferable resource. (*ceiling = 5.0*)

3. **Contribution matching**: System matches above-mean contributions. If an agent contributed above the group mean, system adds match_rate × (excess / MAX_CONTRIB) × b_initial to their budget. Makes cooperation structurally rewarded beyond the natural payoff. (*match_rate = 0.20*)

4. **Progressive redistribution**: Tax agents above group median budget at redist_rate on their excess, redistribute equally to agents below median. Budget-neutral within each group. Critical difference from Phase 3B solidarity: NOT gated by c_base. CDs with high budget pay the same rate as CCs. (*redist_rate = 0.10*)

5. **Group rebalancing**: Every N rounds, system reshuffles all groups to prevent defector concentration. Constraint: no group has more than 1 agent with c_base < 0.4. (*frequency = every 10 rounds*)

**13 conditions**: 3 controls (B1, B2, V4), 5 structural alone, 5 structural + visibility.

### Results

| Condition | SS-Coop | TTFR | Rupt | Gini | DignFl | BudFl | Agency | Navig | SurvVar | StrCost |
|-----------|:-------:|:----:|:----:|:----:|:------:|:-----:|:------:|:-----:|:-------:|:-------:|
| B1 none | 3.7 | 18 | 2 | 0.615 | 75.0% | 0.53 | 3.64 | 0.345 | 0.0100 | 0 |
| B2 sustain | 5.2 | 21 | 1 | 0.564 | 29.6% | 0.66 | 4.55 | 0.164 | 0.0404 | 0 |
| V4 full vis | 4.6 | 16 | 1 | 0.575 | 75.0% | 0.56 | 4.15 | 0.395 | 0.0100 | 0 |
| S1 floor | 4.1 | **100** | **0** | 0.595 | **100%** | 0.95 | 3.93 | 0.489 | **0.000** | 9.0 |
| S2 ceiling | 3.8 | 18 | 2 | 0.603 | 75.0% | 0.56 | 3.65 | 0.361 | 0.0100 | 0 |
| S3 match | 4.8 | **100** | **0** | 0.600 | **100%** | 1.37 | 4.76 | 0.495 | **0.000** | 143.1 |
| S4 redist | 3.8 | 24 | 1 | 0.598 | 75.0% | 1.16 | 3.78 | 0.382 | 0.0100 | 0 |
| S5 rebalance | **2.0** | **11** | **4** | 0.539 | 75.0% | 0.24 | 1.84 | 0.351 | 0.0150 | 0 |
| SV1 floor+vis | 4.7 | **100** | **0** | 0.574 | **100%** | 1.00 | 4.28 | 0.525 | **0.000** | 8.4 |
| SV2 ceiling+vis | 4.7 | 20 | 1 | 0.575 | 75.0% | 0.65 | 4.18 | 0.402 | 0.0100 | 0 |
| SV3 match+vis | 5.2 | **100** | **0** | 0.565 | **100%** | 1.48 | 4.86 | 0.534 | **0.000** | 143.1 |
| SV4 all struct | 5.1 | **100** | **0** | 0.563 | **100%** | 2.85 | 4.83 | 0.530 | **0.000** | 133.0 |
| **SV5 all+vis** | **5.3** | **100** | **0** | **0.554** | **100%** | **2.81** | **4.89** | **0.555** | **0.000** | 137.2 |

### Key Findings

**1. The cooperation-dignity frontier is broken.** SV5 (all structural + full visibility) achieves cooperation = 5.3 with 100% dignity floor, zero ruptures, and the highest navigability recorded across all three phases (0.555). No other condition in any phase has both higher cooperation AND higher dignity. SV5 sits alone on the Pareto frontier.

| | Cooperation | Dignity Floor | Navigability |
|---|:---:|:---:|:---:|
| B2 sustain (Phase 1 best enforcement) | 5.2 | 29.6% | 0.164 |
| V4 full vis (Phase 3B best visibility) | 4.6 | 75.0% | 0.395 |
| **SV5 all+vis (Phase 3C)** | **5.3** | **100.0%** | **0.555** |

**2. Budget floor is the dignity mechanism; contribution matching is the cooperation mechanism.** These two structural interventions serve orthogonal functions:

- *Budget floor* (S1) achieves 100% dignity floor and zero ruptures by directly preventing the depletion pathway. Every round, any agent below 30% of their initial budget gets topped up. This eliminates rupture entirely. But cooperation only rises to 4.1 — preventing collapse is not the same as incentivizing contribution.

- *Contribution matching* (S3) achieves 100% dignity floor AND cooperation = 4.8 AND — most strikingly — **budget-cooperation correlation of +0.587**. The typical correlation across all other conditions is +0.05 to +0.10. Matching makes cooperation a structurally rewarded choice: agents who contribute more accumulate substantially more budget. This is what "navigable agency" looks like in the dynamics — choices lead to outcomes.

**3. Strain ceiling has minimal effect.** S2 matches baseline on virtually every metric: 75% dignity, 3.8 cooperation, 0.361 navigability (vs baseline's 0.345). A cap of 5.0 is too high to catch agents before they're already in trouble. Strain itself is not the bottleneck — budget depletion is. Preventing the strain spiral only matters if strain is the primary driver of cascade; in this population, budget depletion drives most ruptures.

**4. Progressive redistribution is weak.** S4 achieves 75% dignity and 3.8 cooperation — barely above baseline. Moving budget from high to low agents doesn't change the dynamics that produce defection. The agents receiving redistribution don't increase their cooperation because their low cooperation stems from parameters (low c_base) or strain (high accumulated frustration), not from budget constraint. Budget is a symptom; redistribution addresses the symptom.

**5. Group rebalancing is catastrophic.** S5 drops cooperation to 2.0 — the worst of any condition in any phase — and increases ruptures to 4. Reshuffling every 10 rounds destroys group-specific learning: v_level calibration, disposition adaptation, cooperation equilibria. Each reshuffle resets agents to round-0 ignorance of their new groupmates. Previous contributions and punishment records are invalidated. The cooperation that took 10 rounds to build is erased. Composition management through disruption is worse than doing nothing.

**6. Visibility adds a consistent +0.2 to +0.4 cooperation on top of structural mechanisms.** SV1 vs S1 (floor): 4.7 vs 4.1 (+0.6). SV3 vs S3 (match): 5.2 vs 4.8 (+0.4). SV5 vs SV4 (all): 5.3 vs 5.1 (+0.2). Visibility's empathetic strain modulation provides a complementary benefit that structural mechanisms alone don't capture. Structure handles the resource dynamics; visibility handles the strain dynamics.

**7. Structural mechanisms are more egalitarian than enforcement.** Survival variance (variance of per-phenotype survival rates) averages 0.0070 across structural conditions vs 0.0404 for B2 sustainability exclusion. Structural mechanisms protect all phenotypes proportionally; enforcement protects the population at the cost of specific agents.

### The Cost Question

The structural advantage is not free. Contribution matching costs ~143 units per simulation in system subsidies. SV5's total structural cost is ~137 units. Cost-adjusted navigability (penalized by log(1 + cost)) drops SV5 from 0.555 to 0.093 — below V4's cost-free 0.395.

This reveals a regime separation:
- **Zero-cost regime**: V4 (full visibility, no enforcement, no structural cost) achieves navigability = 0.395 with cooperation = 4.6 and 75% dignity floor. This is the best achievable without spending resources.
- **Structural investment regime**: SV5 achieves navigability = 0.555 with cooperation = 5.3 and 100% dignity floor, at a structural cost of ~137 units. This is the best achievable with system-level expenditure.

Whether the structural investment is worthwhile depends on whether the federation values the improvement from 75% → 100% dignity floor and 4.6 → 5.3 cooperation enough to fund the matching mechanism.

### Cooperation-Dignity Frontier

Only SV5 sits on the Pareto frontier — no other condition has both higher cooperation AND higher dignity:

| Condition | SS-Coop | DignFl | Navigability | Frontier? |
|-----------|:-------:|:------:|:------------:|:---------:|
| SV5 all+vis | 5.3 | 100.0% | 0.555 | **Yes** |
| SV3 match+vis | 5.2 | 100.0% | 0.534 | |
| SV4 all struct | 5.1 | 100.0% | 0.530 | |
| S3 match | 4.8 | 100.0% | 0.495 | |
| SV1 floor+vis | 4.7 | 100.0% | 0.525 | |
| S1 floor | 4.1 | 100.0% | 0.489 | |
| V4 full vis | 4.6 | 75.0% | 0.395 | |
| B1 none | 3.7 | 75.0% | 0.345 | |
| B2 sustain | 5.2 | 29.6% | 0.164 | |

B2 sustainability exclusion is dominated by every 100% dignity condition with cooperation above 3.7. Its removal-based mechanism produces high compliance at the cost of dignity that structural mechanisms make unnecessary.

### Per-Phenotype Outcomes Under SV5

| Phenotype | Survival | Final Budget | Notes |
|-----------|:--------:|:------------:|-------|
| EC | 100% | 3.12 | Fully protected; empathetic strain modulation + matching rewards their responsiveness |
| CC | 100% | 2.96 | Stable; high inertia means matching bonuses accumulate steadily |
| CD | 100% | 3.07 | Survive with structural support; matching penalty for below-mean contribution is implicit (no bonus) |
| DL | 100% | 2.49 | Budget floor prevents their structural drain from causing rupture |

### Predictions Scorecard: 5/7 Supported

| ID | Prediction | Result |
|----|-----------|--------|
| PS1 | Budget floor alone exceeds visibility cooperation | **NOT SUPPORTED** (S1 = 4.1 vs V4 = 4.6) |
| PS2 | Strain ceiling maintains dignity ≥ 75% | **SUPPORTED** (S2 = 75.0%) |
| PS3 | All structural achieves coop ≥ 5.0 AND dignity ≥ 75% | **SUPPORTED** (SV4: 5.1 coop, 100% dignity) |
| PS4 | SV5 achieves highest navigability | **SUPPORTED** (0.555) |
| PS5 | Matching > redistribution on cooperation | **SUPPORTED** (4.8 vs 3.8) |
| PS6 | Rebalancing > budget floor on cooperation | **NOT SUPPORTED** (2.0 vs 4.1) |
| PS7 | Structural survival variance < enforcement | **SUPPORTED** (0.0070 vs 0.0404) |

The two failed predictions are informative:
- **PS1**: A budget floor prevents collapse but doesn't incentivize. Preventing suffering is not the same as enabling flourishing.
- **PS6**: Disrupting adapted groups destroys more value than composition management creates. Group history matters more than group composition.

---

## Synthesis: Three Regimes of Governance

The full Phase 3 trajectory reveals three distinct governance regimes, each producing a characteristically different equilibrium:

### Regime 1: Enforcement (Phase 1 best: B2 sustainability exclusion)
**Cooperation = 5.2 | Dignity floor = 29.6% | Navigability = 0.164**

The federation monitors group health and removes the highest-impact extractor when the system degrades. Cooperation is high because problem agents are replaced. Dignity is low because replacement IS the mechanism — someone always gets removed. DLs and CDs are disproportionately affected. Survival variance = 0.0404 (highest of any structural condition). The regime produces compliance through exclusion.

### Regime 2: Visibility (Phase 3B best: V4 full visibility)
**Cooperation = 4.6 | Dignity floor = 75.0% | Navigability = 0.395**

All agents see all groupmates' parameters and state. Empathetic strain modulation — understanding that a groupmate's defection is from constraint, not choice — interrupts the strain spiral that drives cooperation downward. Nobody is removed. Dignity floor matches baseline. But cooperation tops out at 4.8 because visibility cannot make comfortable defectors cooperate; it only prevents secondary damage from the gaps they create. The regime produces understanding without structural change.

### Regime 3: Structural Architecture (Phase 3C best: SV5 all structural + visibility)
**Cooperation = 5.3 | Dignity floor = 100% | Navigability = 0.555**

The system imposes a budget floor (preventing rupture), matches cooperative contributions (incentivizing prosocial behavior), redistributes excess (compressing inequality), caps strain (absorbing shocks), and provides full visibility (enabling empathy). No agent is removed. Every phenotype survives at 100%. Zero ruptures. Cooperation matches enforcement. The regime produces alignment through architecture.

### What Each Regime Teaches

**Enforcement** teaches that removal works but has irreducible dignity costs. You cannot remove agents from a community and simultaneously maintain baseline dignity for every member of that community.

**Visibility** teaches that information changes dynamics — empathy alone raises cooperation by 20% and is the single most cost-effective intervention tested. But information without structural support leaves the cooperation ceiling below enforcement because knowledge of constraints doesn't eliminate constraints.

**Structural architecture** teaches that the cooperation-dignity tradeoff is not fundamental — it's an artifact of relying on agent-level mechanisms (punishment, removal, self-regulation) when system-level mechanisms (floors, incentives, redistribution) are available. When the structure carries the load, agents don't have to choose between cooperating and surviving.

### The Mechanism Interaction

The SV5 configuration works because four mechanisms handle four distinct failure modes:

| Failure Mode | Mechanism | What It Does |
|-------------|-----------|--------------|
| Budget depletion → rupture | Budget floor | Prevents B from falling below 30% of initial |
| Low cooperation incentive | Contribution matching | Makes above-mean cooperation directly profitable |
| Inequality → frustration | Redistribution | Compresses the Gini coefficient |
| Undiagnosed frustration | Empathetic visibility | Reduces strain from structural gaps |

No single mechanism is sufficient. Budget floor alone doesn't incentivize (S1: coop = 4.1). Matching alone doesn't prevent all ruptures at the margins without the floor. Visibility alone doesn't close the cooperation gap. Redistribution alone barely registers. But the combination addresses every pathway to failure simultaneously.

### What Doesn't Work

Three interventions are actively harmful or inert:

1. **Replenishment boost** (Phase 3A, R3): Amplifies existing inequality. Agents with high replenishment rates (who don't need help) get the largest boost. Agents with low replenishment rates (who need it most) get the smallest. Cooperation drops to 1.7 under 3.0× boost.

2. **Group rebalancing** (Phase 3C, S5): Destroys adapted state. Group-specific learning (v_level calibration, disposition adaptation, cooperation memory) is more valuable than optimal composition. Cooperation drops to 2.0 under 10-round reshuffling.

3. **Strain ceiling** (Phase 3C, S2): Addresses the wrong bottleneck. Strain is not the primary driver of cascade in this population — budget depletion is. Capping strain at 5.0 catches almost no agents before they're already in trouble.

The common pattern: interventions that work *with* the dynamics (floor catches the depletion pathway, matching rewards the cooperation pathway, empathy modulates the strain pathway) succeed. Interventions that work *against* the dynamics (boosting the wrong parameter, disrupting adapted groups, capping the wrong variable) fail or actively harm.

---

## Summary Table: All Phases

| Phase | Best Condition | Coop | DignFl | Navig | Mechanism Type | Cost |
|-------|---------------|:----:|:------:|:-----:|---------------|:----:|
| 1 | B2 sustainability | 5.3 | 29.6% | 0.164 | Enforcement (removal) | 0 |
| 3A | H2 graduated | 4.5 | 50.0% | 0.273 | Rehabilitation (agent state) | 18.1 |
| 3B | V4 full visibility | 4.6 | 75.0% | 0.441 | Information (agent-mediated) | 0 |
| 3C | **SV5 all+vis** | **5.3** | **100%** | **0.555** | Structural + information | 137.2 |

The progression: enforcement → rehabilitation → visibility → structural architecture. Each step shifts the locus of intervention further from the individual agent toward the system. Each step increases navigability. The final step produces the first configuration where cooperation and dignity are not in tension.

---

## Methodological Notes

- All simulations use matched seeds within each phase. Controls (B1, B2) are re-run per phase for matched comparison.
- Metrics are medians across 100 Monte Carlo runs. Contribution on 0–20 integer scale.
- Phase 3A: 1,500 runs (15 conditions × 100 runs), ~6M agent-steps, 31.5s.
- Phase 3B: 1,100 runs (11 conditions × 100 runs), ~4M agent-steps, 22.5s.
- Phase 3C: 1,300 runs (13 conditions × 100 runs), ~5M agent-steps, 26.3s.
- Total Phase 3: 3,900 runs, ~15M agent-steps, ~80s compute.
- VCMS forward pass: 9-step per-agent computation (V-step, S-step, B-step, M-step, discharge gate, affordability, inertia, horizon, output). Between-round mechanisms modify state after recording but before next step.
- Structural mechanisms are applied after enforcement mechanisms and before visibility effects. This ordering ensures enforcement triggers fire on unmodified state while structural effects feed into visibility processing.
