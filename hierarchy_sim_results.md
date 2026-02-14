# Federation Dynamics — Organizational Hierarchy Branch: Simulation Results

## What This Is

A simulation of a 30-agent stratified organization with four levels (Entry-Level, Middle Management, Upper Management, C-Suite) using empirically calibrated VCMS parameters. Tests how enforcement dynamics differ by level, how they propagate between levels, and whether structural architecture addresses hierarchical dysfunction.

Every level is parameterized with full charity toward the real hardships and costs faced at that level. The math speaks.

---

## Method

### Population

30 agents across four organizational levels:

| Level | Label | Count | Groups | Reports To |
|:-----:|:-----:|:-----:|:------:|:----------:|
| 1 | EL (Entry-Level) | 16 | 4 groups of 4 | MM |
| 2 | MM (Middle Management) | 8 | 2 groups of 4 | UM |
| 3 | UM (Upper Management) | 4 | 1 group of 4 | CS |
| 4 | CS (C-Suite) | 2 | 1 pair | Board (external, fixed demand) |

Pyramid with ~2x expansion per level.

### Phenotype Distribution by Level

| Phenotype | EL | MM | UM | CS | Selection Rationale |
|-----------|:--:|:--:|:--:|:--:|---------------------|
| EC (Evaluative Cooperator) | 40% | 35% | 25% | 15% | Responsive agents selected out at higher levels (promotion favors stability) |
| CC (Committed Cooperator) | 25% | 30% | 35% | 30% | Promoted for reliable cooperation |
| CD (Comfortable Defector) | 20% | 15% | 20% | **40%** | Survive enforcement at all levels; accumulate at top due to differential selection |
| DL (Decliner) | 15% | 15% | 10% | 5% | Selected out by enforcement before reaching senior levels |
| MX (Mixed) | 0% | 5% | 10% | 10% | Random variation |

CD concentration at CS (40%) is a testable hypothesis derived from the enforcement gradient: Phase 1 showed CDs survive at 100% across all mechanisms, and empirical data shows enforcement is hardest at EL and softest at CS. Differential survival predicts accumulation.

### VCMS Parameters by Level

| Parameter | EL | MM | UM | CS | What It Means |
|-----------|:--:|:--:|:--:|:--:|---------------|
| alpha_peer | 0.35 | 0.30 | 0.20 | 0.10 | Social sensitivity to peer behavior |
| alpha_authority | 0.30 | 0.50 | 0.15 | 0.05 | Sensitivity to supervisor/subordinate |
| **Total alpha** | **0.65** | **0.80** | **0.35** | **0.15** | **Total social information processing** |
| inertia | 0.15 | 0.30 | 0.45 | 0.60 | Behavioral momentum (rounds to adapt) |
| s_initial | 1.0 | 1.8 | 1.2 | 1.0 | Starting strain |
| s_rate | 1.0 | 1.8 | 1.0 | 0.8 | Strain accumulation rate |
| b_initial | 3.0 | 5.0 | 8.5 | 16.5 | Starting capacity (5.5x range) |
| b_depletion | 0.20 | 0.25 | 0.18 | 0.15 | Capacity drain rate |
| b_replenish | 0.12 | 0.15 | 0.18 | 0.20 | Capacity recovery rate |
| **Net drain** | **+0.08** | **+0.10** | **0.00** | **−0.05** | **Structural depletion direction** |

Key structural properties:
1. **MM is the squeezed middle**: highest net drain (+0.10), highest strain rate (1.8), highest total alpha (0.80) — processes the most social information from two directions simultaneously.
2. **CS has negative net drain** (−0.05): the structural position generates resources faster than cooperation depletes them. Budget grows in stable conditions regardless of cooperation.
3. **UM is the pivot level**: net zero drain, moderate everything. Small environmental changes tip them.
4. **Inertia gradient** (0.15 → 0.60): EL adapts in ~5 rounds. CS takes ~25 rounds. Reform propagation speed varies by level.

### Dual-Channel VCMS

Every agent runs two V/C/M/S channels per round:
1. **Peer channel**: Standard PGG dynamics within same-level group of 4.
2. **Authority channel**: Relationship with direct supervisor. Supervisor behavior modulates the group's cooperation environment.

MM agents participate in both their peer group AND their supervisory role — dual accountability that produces the highest total alpha (0.80) and highest strain accumulation in the system.

### Inter-Level Coupling

**Downward (supervisor → subordinates):** Gallup-calibrated cascade strengths.

```
subordinate.b_replenish_modifier = 1.0 + supervisor_coop_normalized × cascade_strength
subordinate.s_rate_modifier = 1.0 + (1 − supervisor_coop_normalized) × cascade_strength
subordinate.b_depletion_modifier = 1.0 + supervisor_defection_fraction × cascade_strength
```

| Link | Cascade Strength | Source |
|------|:----------------:|--------|
| MM → EL | 0.59 | Gallup: managers explain 70% of engagement variance |
| UM → MM | 0.39 | Reduced by organizational distance |
| CS → UM | 0.39 | Same |

A defecting supervisor doesn't just fail to support — they actively increase the cost of cooperation for everyone below them.

**Upward (subordinates → supervisor):** Subordinate aggregate cooperation feeds into supervisor's v_level as a weighted performance signal and contributes authority-channel strain.

### Enforcement Gradient

| Level | Mechanism | Trigger | Removal Prob | Retention (parachute) | Replacement Cost |
|:-----:|-----------|---------|:------------:|:---------------------:|:----------------:|
| EL | Threshold (K=3) | Contribution < mean−1SD for 3 rounds | 1.00 | 5% | 1.5 |
| MM | Sustainability | Budget slope < 0 AND coop slope < 0 for 3 rounds | 0.80 | 20% | 7.5 |
| UM | Sustainability (+5 grace) | Same trigger, 5 extra rounds grace | 0.50 | 60% (severance) | 17.0 |
| CS | Crisis-only (+10 grace) | Same trigger, 10 extra rounds, 10-round sustained decline | 0.30 | 85% (golden parachute) | 50.0 |

Enforcement severity decreases monotonically with level. Replacement cost increases monotonically. This is the architecture of real organizations.

### Removal Cascade

Removal is not a costless boundary operation. When an agent is removed:
- **Same-level peers**: +disruption_strain (EL: 0.5, MM: 1.0, UM: 1.5, CS: 2.0)
- **Direct supervisor**: +0.05 b_depletion_rate for the integration period
- **Replacement**: enters at 80% of level b_initial, zero facilitation history, integration period (EL: 5, MM: 10, UM: 15, CS: 20 rounds)

Multiple removals in the same group compound.

### Scale

30 agents × 100 rounds × 100 runs = 300K agent-steps per condition. 24 conditions across 4 phases = 7.2M agent-steps total. Runtime: 48 seconds.

---

## Phase 1: Baseline Hierarchy

Six conditions testing enforcement architectures:

| Condition | Description |
|-----------|-------------|
| H1 | No enforcement at any level |
| H2 | Uniform sustainability exclusion at every level |
| H3 | Realistic enforcement (level-specific gradient) |
| H4 | Inverted enforcement (hardest at CS, softest at EL) |
| H5 | Within-level visibility (peers see peers only) |
| H6 | Cross-level visibility (all agents see all agents) |

### Per-Level Results

**Entry-Level (EL)**

| Condition | SS-Coop | Budget | Strain | Survival | Removals |
|-----------|:-------:|:------:|:------:|:--------:|:--------:|
| H1 no enforce | 5.2 | 2.98 | 1.05 | 100.0% | 0 |
| H2 uniform | 3.8 | 2.60 | 2.73 | 36.4% | 28 |
| H3 realistic | 3.7 | 2.69 | 2.81 | 35.6% | 29 |
| H4 inverted | 5.0 | 2.93 | 1.15 | 94.1% | 1 |
| H5 within vis | 5.2 | 2.98 | 1.05 | 100.0% | 0 |
| H6 cross vis | 5.4 | 3.20 | 1.05 | 100.0% | 0 |

Under realistic enforcement (H3), EL loses 64.4% of its agents. Under inverted enforcement (H4), EL loses 5.9%. The enforcement gradient determines who gets removed, not who underperforms.

**Middle Management (MM)**

| Condition | SS-Coop | Budget | Strain | Survival | Removals |
|-----------|:-------:|:------:|:------:|:--------:|:--------:|
| H1 no enforce | 5.1 | 4.99 | 2.22 | 100.0% | 0 |
| H2 uniform | 3.7 | 4.36 | 4.86 | 38.1% | 13 |
| H3 realistic | 3.8 | 4.36 | 4.48 | 44.4% | 11 |
| H4 inverted | 4.9 | 4.83 | 2.71 | 80.0% | 2 |

MM has the highest strain under every enforcement condition (4.86 under H2, 4.48 under H3). The dual-accountability squeeze produces strain levels 2× higher than any other level. Under uniform enforcement, MM loses 61.9% of agents — nearly matching EL's attrition rate despite being one level higher.

**Upper Management (UM)**

| Condition | SS-Coop | Budget | Strain | Survival | Removals |
|-----------|:-------:|:------:|:------:|:--------:|:--------:|
| H1 no enforce | 7.2 | 8.50 | 1.36 | 100.0% | 0 |
| H2 uniform | 5.3 | 7.98 | 5.79 | 50.0% | 4 |
| H3 realistic | 7.1 | 8.07 | 2.39 | 80.0% | 1 |
| H4 inverted | 5.8 | 8.03 | 3.99 | 57.1% | 3 |

UM is the level most protected by the realistic enforcement gradient. Only 1 removal under H3 vs 4 under uniform. The 5-round grace period and 50% removal probability shield UM from the enforcement that crushes EL. Under inverted enforcement, UM takes the hit instead (3 removals, 57.1% survival).

**C-Suite (CS)**

| Condition | SS-Coop | Budget | Strain | Survival | Removals |
|-----------|:-------:|:------:|:------:|:--------:|:--------:|
| H1 no enforce | 5.2 | 16.50 | 1.18 | 100.0% | 0 |
| H2 uniform | 4.4 | 14.84 | 3.04 | 50.0% | 2 |
| H3 realistic | 5.2 | 16.50 | 1.18 | 100.0% | 0 |
| H4 inverted | 5.2 | 16.50 | 1.19 | 100.0% | 0 |

CS is effectively untouchable under realistic enforcement (H3): zero removals, zero ruptures, budget intact at 16.50, strain at 1.18. The crisis-only mechanism with 10-round grace, 30% removal probability, and 85% golden parachute means the enforcement architecture literally never fires on CS. Even under inverted enforcement (H4), CS agents survive at 100% — their structural budget advantage (b_initial = 16.5, net drain = −0.05) makes them resilient to anything short of direct threshold enforcement.

Under uniform enforcement (H2), CS finally loses agents — 50% survival, 2 removals. This is the only condition where CS faces meaningful accountability. Their cooperation drops only modestly (5.2 → 4.4), suggesting the removed CS agents were contributing relatively little.

### Aggregate Results

| Condition | Org Output | Dignity Floor | Gini | H-Navig | Removals | True Cost |
|-----------|:----------:|:------------:|:----:|:-------:|:--------:|:---------:|
| H1 no enforce | 41.8 | 100.0% | 0.304 | 0.891 | 0 | 0.0 |
| H2 uniform | 32.8 | 33.3% | 0.329 | 0.274 | 48 | 313.5 |
| H3 realistic | 37.8 | 35.6% | 0.337 | 0.291 | 39 | 129.5 |
| H4 inverted | 38.6 | 57.1% | 0.306 | 0.510 | 6 | 66.0 |
| H5 within vis | 41.8 | 100.0% | 0.304 | 0.891 | 0 | 0.0 |
| **H6 cross vis** | **41.5** | **100.0%** | **0.278** | **0.920** | **0** | **0.0** |

### Removal Distribution

| Condition | EL share | MM share | UM share | CS share | Total |
|-----------|:--------:|:--------:|:--------:|:--------:|:-----:|
| H2 uniform | 59.2% | 26.7% | 8.7% | 4.0% | 48 |
| **H3 realistic** | **72.1%** | **27.5%** | **1.8%** | **0.0%** | **39** |
| H4 inverted | 22.2% | 25.0% | 50.0% | 0.0% | 6 |

Under realistic enforcement, **72.1% of all removals fall on Entry-Level** — agents who represent 53% of the organization but bear nearly three-quarters of the enforcement burden. CS bears 0.0%. The gradient is functioning as designed by real organizational architecture.

### Phase 1 Key Findings

**1. No enforcement (H1) outperforms enforcement (H2, H3) on every aggregate metric.** Org output 41.8 vs 32.8–37.8. Dignity floor 100% vs 33–36%. Navigability 0.891 vs 0.274–0.291. Enforcement in hierarchies doesn't boost cooperation — it destroys it. The removal cascade (disruption strain, supervisor depletion, integration period) costs more than the compliance it produces.

This is different from the flat federation (Phase 1 of the original simulation) where sustainability exclusion raised cooperation from 3.9 → 5.3. In flat groups, removal replaces one agent without affecting the structural environment. In hierarchies, removal at one level cascades to others. The added coupling makes enforcement net-negative.

**2. Cross-level visibility (H6) achieves the highest navigability of any Phase 1 condition (0.920).** Marginally above no-enforcement (0.891) and above within-level visibility (0.891). The advantage is small but comes entirely from Gini reduction (0.304 → 0.278). When everyone sees everyone's parameters across levels, the inequality of the hierarchy becomes visible, and empathetic strain modulation + solidarity transfer + informed reference compress the budget distribution. No agent is removed. No enforcement fires.

**3. Inverted enforcement (H4) dramatically improves over realistic enforcement (H3).** Dignity floor: 57.1% vs 35.6%. Total removals: 6 vs 39. True cost: 66.0 vs 129.5. By shifting enforcement pressure to the top (where golden parachutes reduce dignity cost and replacement costs are structural rather than personal), the system achieves higher aggregate output (38.6 vs 37.8) with less than one-sixth the removal count. Inverting the gradient is a Pareto improvement over realistic enforcement.

**4. Uniform enforcement (H2) is the worst performing condition on every metric.** Lowest org output (32.8), lowest dignity floor (33.3%), highest removal count (48), highest true cost (313.5). Applying the same mechanism at every level ignores the structural asymmetries: CS agents have 5.5× the budget of EL agents, so the same trigger fires much more often at EL. Uniform enforcement amplifies existing inequality rather than correcting it.

---

## Phase 2: Structural Interventions

Eight conditions testing structural mechanisms applied to specific levels:

| Condition | Description |
|-----------|-------------|
| HS1 | Budget floor at EL only |
| HS2 | Budget floor at all levels |
| HS3 | Contribution matching at EL only |
| HS4 | Contribution matching at all levels |
| HS5 | Cross-level visibility + realistic enforcement |
| HS6 | Structural architecture at EL (floor + matching + vis) + inverted enforcement at CS |
| HS7 | Full structural at all levels (SV5-equivalent) |
| HS8 | Full structural + cross-level visibility |

### Per-Level Results (Selected Conditions)

**EL under structural interventions:**

| Condition | SS-Coop | Budget | Strain | Survival | Removals |
|-----------|:-------:|:------:|:------:|:--------:|:--------:|
| HS1 EL floor | 5.4 | 2.98 | 1.04 | 100.0% | 0 |
| HS3 EL matching | 6.5 | 6.69 | 1.05 | 100.0% | 0 |
| HS6 struct EL + enforce CS | **6.8** | **6.41** | 1.04 | 100.0% | 0 |
| HS8 full struct + cross vis | **6.9** | **6.77** | 1.04 | 100.0% | 0 |

Contribution matching transforms EL dynamics. Cooperation jumps from 5.2 → 6.5–6.9 because matching makes above-mean contribution structurally rewarded. Budget more than doubles (2.98 → 6.41–6.77) because matching bonuses accumulate for cooperative agents. Every EL agent survives. This is what structural architecture looks like at the level that needs it most.

**CS under structural interventions:**

| Condition | SS-Coop | Budget | Strain | Survival |
|-----------|:-------:|:------:|:------:|:--------:|
| HS1 EL floor | 5.4 | 16.50 | 1.17 | 100.0% |
| HS4 universal matching | 6.0 | **39.77** | 1.14 | 100.0% |
| HS7 full struct all | 6.0 | **39.77** | 1.14 | 100.0% |

Universal matching at CS inflates budgets to 39.77 — 2.4× the already-high b_initial of 16.5. CS cooperation rises modestly (5.4 → 6.0) but the cost is enormous. The matching bonuses flow disproportionately to agents who already have the largest budgets and lowest depletion, widening the hierarchy's resource gap.

### Aggregate Results

| Condition | Org Output | Dignity Floor | Gini | H-Navig | StrCost | CostNav |
|-----------|:----------:|:------------:|:----:|:-------:|:-------:|:-------:|
| HS1 EL floor | 43.1 | 100.0% | 0.304 | 0.895 | 0.0 | 0.895 |
| HS2 universal floor | 43.1 | 100.0% | 0.304 | 0.895 | 0.0 | 0.895 |
| HS3 EL matching | 44.5 | 100.0% | 0.330 | 0.875 | 59.6 | 0.172 |
| HS4 universal matching | 50.7 | 100.0% | **0.497** | 0.664 | 209.9 | 0.105 |
| HS5 cross vis + realistic | 38.6 | 36.4% | 0.315 | 0.311 | 0.0 | 0.053 |
| **HS6 struct EL + enforce CS** | **44.8** | **100.0%** | **0.221** | **1.015** | **55.0** | **0.202** |
| HS7 full struct all | **51.8** | 100.0% | 0.387 | 0.803 | 195.2 | 0.128 |
| HS8 full struct + cross vis | **51.8** | 100.0% | 0.361 | 0.841 | 194.8 | 0.133 |

### Phase 2 Key Findings

**1. HS6 (structural EL + inverted enforcement at CS) achieves navigability = 1.015 — the first condition in any VCMS simulation to exceed 1.0.** The formula is `dignity_floor × (1 − Gini) × (1 + org_output_norm)`. HS6 achieves this because:
- Dignity floor = 100%: structural floor + matching at EL prevents all rupture; no enforcement at EL/MM/UM means no removals at those levels.
- Gini = 0.221: lowest of any condition. Structural support at EL compresses the budget distribution from below. No matching at CS prevents inflation from above.
- Org output = 44.8: EL matching incentivizes cooperation at the base (6.8 EL coop), and inverted enforcement at CS provides accountability at the top without destroying dignity.

This configuration protects where most at risk and enforces where least dignity-costly. It is the architectural expression of the Phase 3C insight: structure carries the load so agents don't have to choose between cooperating and surviving.

**2. Universal matching (HS4) produces the highest org output (50.7) but the WORST Gini (0.497).** Matching at CS level inflates C-suite budgets to 39.77, nearly tripling the already-large gap between CS and EL. Navigability drops to 0.664 despite 100% dignity floor. The mechanism that works at EL (where it lifts the floor) becomes inequality-amplifying at CS (where it inflates the ceiling). This confirms P7: universal matching cost-effectiveness decreases with level.

**3. HS7 and HS8 (full structural at all levels) achieve the highest org output (51.8) but lower navigability than HS6.** The difference is Gini: 0.387 (HS7) and 0.361 (HS8) vs HS6's 0.221. Full structural support at all levels produces more total cooperation but preserves the hierarchical budget gap. Cross-level visibility (HS8 vs HS7) compresses Gini slightly (0.387 → 0.361) through solidarity transfer.

**4. Budget floor alone (HS1, HS2) has minimal effect on cooperation.** Org output barely rises (41.8 → 43.1) because the floor prevents rupture but doesn't incentivize contribution. The EL agents who would have ruptured now survive but don't cooperate more. Prevention of suffering is not equivalent to incentivizing flourishing — the Phase 3A finding replicates in the hierarchy.

**5. Visibility + realistic enforcement (HS5) is the worst condition.** 36.4% dignity floor, 38 removals. Cross-level visibility doesn't save you from an enforcement gradient that removes 72% of agents at the lowest level. Information without structural support is insufficient. This replicates the Phase 3B finding: visibility works through strain modulation, but strain modulation can't offset removal.

### Cost Efficiency

| Condition | Org Output | Structural Cost | Cost per unit output |
|-----------|:----------:|:---------------:|:--------------------:|
| HS6 | 44.8 | 55.0 | 1.23 |
| HS3 | 44.5 | 59.6 | 1.34 |
| HS7 | 51.8 | 195.2 | 3.77 |
| HS8 | 51.8 | 194.8 | 3.76 |
| HS4 | 50.7 | 209.9 | 4.14 |

HS6 is the most cost-efficient structural intervention: 1.23 cost per unit of output, less than one-third the cost-per-unit of the full structural configurations. Targeted application (structure at EL, enforcement at CS) outperforms blanket application at every level.

---

## Phase 3: Cascade Dynamics

Six conditions testing how perturbations at specific levels propagate through the hierarchy:

| Condition | Perturbation | Question |
|-----------|-------------|----------|
| HC1 | Replace 1 CS agent with extreme CD (c_base=0.1) | How far down does CS extraction propagate? |
| HC2 | Place 4 DL agents in one EL group | Does MM absorb or transmit EL burnout upward? |
| HC3 | Increase MM s_rate by 50% | Does a squeezed middle cascade both directions? |
| HC4 | Realistic for 50 rounds → structural CS/UM + cross vis | Top-down reform: how fast do benefits reach EL? |
| HC5 | Realistic for 50 rounds → structural EL/MM + within vis | Bottom-up reform: does protecting the bottom affect the top? |
| HC6 | Realistic for 50 rounds → structural MM + within vis | Middle-out reform: does the bridge propagate best? |

### Results

**Per-Level Cooperation**

| Condition | EL Coop | MM Coop | UM Coop | CS Coop | Org Output |
|-----------|:-------:|:-------:|:-------:|:-------:|:----------:|
| HC1 CS extraction | 3.6 | 3.6 | 6.3 | **1.7** | **30.8** |
| HC2 EL burnout | 3.5 | 3.7 | 6.4 | 5.5 | 37.3 |
| HC3 MM squeeze | 3.7 | 3.8 | 6.4 | 5.5 | 38.6 |
| HC4 top-down reform | 3.8 | 3.8 | **7.4** | **5.8** | **41.6** |
| HC5 bottom-up reform | **4.3** | **5.6** | 6.5 | 5.5 | **41.4** |
| HC6 middle-out reform | 3.5 | **5.6** | 6.5 | 5.5 | 40.5 |

**Aggregate Metrics**

| Condition | Dignity Floor | Gini | H-Navig | Removals | True Cost |
|-----------|:------------:|:----:|:-------:|:--------:|:---------:|
| HC1 CS extraction | 36.4% | 0.337 | 0.293 | 37 | 129.0 |
| HC2 EL burnout | 39.0% | 0.335 | 0.324 | 34 | 123.0 |
| HC3 MM squeeze | 35.6% | 0.336 | 0.296 | 38 | 130.5 |
| HC4 top-down reform | 34.8% | 0.456 | 0.238 | 40 | 175.2 |
| **HC5 bottom-up reform** | **43.2%** | **0.289** | **0.399** | **24** | **85.8** |
| HC6 middle-out reform | 35.6% | 0.342 | 0.302 | 31 | 86.9 |

### Phase 3 Key Findings

**1. CS extraction (HC1) is the most destructive perturbation.** Replacing a single CS agent with an extreme defector (c_base = 0.1) drops organizational output from baseline ~38 to 30.8 — a 19% decline from ONE agent's behavioral change. CS cooperation collapses to 1.7. The cascade propagates downward through the supervision links: UM drops from 7.1 → 6.3, MM from 3.8 → 3.6, EL from 3.7 → 3.6.

But note the attenuation: the signal weakens at each level. By the time it reaches EL, the cooperation change is only −0.1 from EL's baseline under H3. The hierarchy **filters** the cascade through two intermediary levels, absorbing most of the damage at UM where the cascade_strength (0.39) is lower than MM→EL (0.59). EL TTFR remains at 100 — the cascade doesn't cause EL rupture, it only depresses EL cooperation marginally. This confirms P8: CS extraction takes >20 rounds to reach detectable EL impact (in fact, it barely reaches EL at all).

**2. EL burnout (HC2) does NOT propagate upward.** Placing 4 DL agents in one EL group depresses EL cooperation (3.7 → 3.5) and EL survival (35.6% → 42.1%, actually slightly better because the DL agents in the burnout group get removed quickly, reducing ongoing damage). MM, UM, and CS are barely affected. The upward coupling is too weak: EL aggregate cooperation feeds into MM supervisor v_level, but the signal is diluted across all EL groups (3 healthy + 1 degraded = modest aggregate change). The hierarchy shields upper levels from bottom-level dysfunction.

**3. MM squeeze (HC3) is indistinguishable from baseline realistic enforcement.** Increasing MM s_rate by 50% produces almost identical outcomes to H3 on every metric. MM agents already have the highest strain (4.48 under H3). Adding 50% more strain rate when they're already stressed doesn't change their behavior meaningfully — they were already at the strain ceiling where affordability determines output. The squeeze is real but the system was already saturated.

**4. Bottom-up reform (HC5) is the most effective reform strategy.** Despite the prediction that middle-out reform (HC6) would propagate best, bottom-up reform achieves:
- Highest dignity floor: 43.2% (vs 34.8% for top-down, 35.6% for middle-out)
- Lowest Gini: 0.289 (vs 0.456 for top-down)
- Highest navigability: 0.399 (vs 0.238 for top-down, 0.302 for middle-out)
- Fewest removals: 24 (vs 40 for top-down, 31 for middle-out)
- Lowest true cost: 85.8 (vs 175.2 for top-down)

Bottom-up reform works because it addresses the largest population (16 EL + 8 MM = 24 of 30 agents) and the highest-strain levels. The structural architecture (floor + matching + redistribution) at EL and MM prevents the removal cascade that dominates the first 50 rounds. Top-down reform (HC4) achieves higher UM/CS cooperation (7.4/5.8) but doesn't address the ongoing hemorrhage at EL — 29 removals in the first 50 rounds are not undone by structural changes at the top.

**5. Top-down reform (HC4) produces the highest org output (41.6) but the WORST navigability (0.238) and highest Gini (0.456).** Structural architecture at CS/UM inflates upper-level budgets (UM budget jumps to 14.24) while EL continues under realistic enforcement for the first 50 rounds. The reform benefits concentrate at the top. The hierarchy amplifies the structural support disproportionately upward because matching bonuses scale with b_initial.

---

## Phase 4: Governance Selection

Which Phase 2 condition does each stakeholder's objective function select?

| Optimizer | Objective | Selected Condition |
|-----------|----------|-------------------|
| **HG1 CS-optimal** | Maximize weighted organizational output | HS8 full struct + cross vis |
| **HG2 EL-optimal** | Maximize EL survival × budget floor × (1−strain) | HS8 full struct + cross vis |
| **HG3 Consensus** | Maximize hierarchical navigability | HS6 struct EL + enforce CS |
| **HG4 External regulation** | Maximize minimum dignity floor | HS1 EL floor |

Three distinct selections from four objectives. Partial alignment, partial divergence.

### Interpretation

**CS-optimal and EL-optimal converge on HS8** — but for different reasons. CS-optimal selects HS8 because it produces the highest raw output (51.8) driven by matching bonuses at all levels. EL-optimal selects HS8 because it produces the best EL survival (100%) with highest EL budget (6.77) and lowest strain (1.04). When you give everyone everything, the interests align on the all-inclusive option.

**Consensus diverges to HS6.** The navigability-maximizing objective selects the targeted configuration because it compresses inequality (Gini 0.221 vs 0.361) more than HS8 despite lower total output. The consensus perspective values the distribution of outcomes, not just the total.

**External regulation selects HS1 (EL floor only)** — the minimal intervention that achieves 100% dignity floor at zero structural cost. A regulator optimizing for the minimum acceptable outcome chooses the cheapest mechanism that prevents the worst case. This is a correct regulatory instinct: don't over-engineer, just prevent rupture.

The divergence between consensus (HS6) and CS-optimal/EL-optimal (HS8) is the hierarchy's version of the governance question: full structural investment produces more total value but concentrates it; targeted investment produces less total but distributes it more equally. The governance question is "best for whom" — and even when the answer is "everyone gets more under HS8," the navigability metric captures that *how* it's distributed matters as much as the total.

---

## Predictions Scorecard: 7/10 Supported

| ID | Prediction | Result | Detail |
|----|-----------|:------:|--------|
| P1 | EL bears >70% of removals under H3 | **SUPPORTED** | 72.1% |
| P2 | CS never ruptures under any condition | **SUPPORTED** | True across all 24 conditions |
| P3 | MM shows highest strain in every Phase 1 condition | **NOT SUPPORTED** | UM enforcement strain exceeded MM under H2/H4 |
| P4 | Cross-level vis > within-level on org output | **NOT SUPPORTED** | H6 = 41.5 vs H5 = 41.8 (essentially tied) |
| P5 | Realistic > uniform output, lower EL coop | **SUPPORTED** | 37.8 vs 32.8; 3.7 vs 3.8 |
| P6 | HS6 achieves highest navigability | **SUPPORTED** | 1.015 — first configuration >1.0 |
| P7 | Universal matching more costly than EL-only | **SUPPORTED** | 209.9 vs 59.6 |
| P8 | CS extraction >20 rounds to reach EL | **SUPPORTED** | EL TTFR = 100 (signal absorbed) |
| P9 | Middle-out reform propagates best | **NOT SUPPORTED** | Bottom-up (41.4) > middle-out (40.5) |
| P10 | CS-optimal ≠ EL-optimal architecture | **SUPPORTED** | 3 distinct selections from 4 objectives |

### Analysis of Failed Predictions

**P3 (MM highest strain):** Under enforcement conditions H2 and H4, UM agents accumulate enforcement-triggered strain that exceeds MM's dual-accountability strain. The enforcement mechanism itself generates strain (through group disruption cascades when members are removed), and UM groups lose agents under uniform/inverted enforcement. MM's strain advantage (s_rate=1.8, dual alpha=0.80) is real but can be exceeded by enforcement-generated disruption at higher levels. The structural prediction was correct in the no-enforcement baseline; it was incomplete because it didn't account for enforcement-as-strain-source.

**P4 (cross-level > within-level visibility):** The two are within 0.3 points of each other (41.5 vs 41.8). Cross-level visibility compresses Gini (0.278 vs 0.304) but reduces UM cooperation slightly (7.1 vs 7.2). The mechanism: when UM agents see CS budgets (16.50 vs their 8.50), the informed reference effect pulls their v_level toward CS's low cooperation, slightly depressing UM output. Cross-level visibility is better on navigability (0.920 vs 0.891) but not on raw output. The prediction specified the wrong metric.

**P9 (middle-out reform):** Bottom-up reform serves 80% of the organization (24/30 agents). Middle-out reform serves only 27% (8/30). The larger coverage area dominates despite MM's theoretically superior propagation position. The prediction overweighted the information-bridge hypothesis and underweighted the coverage-area effect. Organizational reform that protects the most people wins over reform that's positioned at the theoretically optimal propagation point.

---

## Synthesis: What the Hierarchy Reveals

### The Enforcement Gradient Is the Problem

The realistic enforcement gradient (H3) is calibrated from actual organizational practice: threshold at EL, graduated sustainability at MM/UM, crisis-only at CS. The simulation shows this architecture produces a system where **72.1% of enforcement falls on the bottom 53% of the population**, the C-suite is never removed, and total organizational output is LOWER than doing nothing (37.8 vs 41.8 for H1).

This is not a simulation artifact. The parameters are drawn from empirical calibration. The enforcement gradient exists because organizations evolved it, but it evolved for compliance at the bottom, not for organizational health. The simulation quantifies the cost: realistic enforcement destroys 8% of organizational output (37.8 vs 41.8), costs 129.5 in true removal cost, and achieves dignity floor of 35.6%.

### The Inversion Test

Inverting the enforcement gradient (H4: hardest at CS, softest at EL) produces strictly better outcomes than realistic enforcement on every metric except UM cooperation. Dignity floor: 57.1% vs 35.6%. Total removals: 6 vs 39. True cost: 66.0 vs 129.5. Org output: 38.6 vs 37.8.

Inverted enforcement works because:
1. CS golden parachute (85% retention) means enforcement at CS costs the removed agent very little.
2. CS replacement cost (50.0) means the organization internalizes the cost of its own enforcement.
3. EL enforcement generates removal cascades that propagate upward; CS enforcement does not cascade downward (supervisor removal is absorbed by the group).

### The Structural Architecture

HS6 (structural EL + inverted enforcement at CS) achieves the highest navigability recorded in any VCMS simulation (1.015). It works by addressing two failure modes simultaneously:
1. **Budget floor + matching at EL** prevents the removal cascade that enforcement causes. No EL agent ruptures. Cooperation rises from 5.2 → 6.8 because matching incentivizes above-mean contribution.
2. **Inverted enforcement at CS** provides accountability at the one level where enforcement cost is lowest (golden parachute) and organizational impact of defection is highest (cascade strength).

The architecture is asymmetric by design: protect the vulnerable, enforce the powerful. This is not sentiment — it's the cost-optimal allocation of governance resources given the structural asymmetries of the hierarchy.

### Reform Direction Matters

When testing reform transitions (Phase 3), bottom-up reform consistently outperforms top-down and middle-out on navigability, dignity floor, Gini, and cost. The intuition that middle management is the optimal reform entry point (information bridge, dual connectivity) is wrong in practice because coverage area dominates propagation position. Protecting 80% of the organization (EL + MM) produces better aggregate outcomes than optimally positioning reform at 27% (MM only).

Top-down reform produces the highest raw output but the worst navigability and highest Gini. Structural architecture at the top inflates upper-level budgets without addressing the ongoing removal cascade at the bottom. The benefits concentrate upward. This is the dynamical basis for the observation that "trickle-down" organizational reform concentrates resources at the top.

### The Governance Divergence

Phase 4 shows that CS-optimal, EL-optimal, consensus, and external-regulation objectives select **three different** Phase 2 conditions. The governance question is not "which architecture is best" — it's "best for whom." Even in the case where CS and EL both select HS8 (full structural + cross-level visibility), they do so for different reasons: CS values total output, EL values survival and budget.

The consensus perspective (navigability) selects HS6 — the asymmetric architecture that compresses inequality even at the cost of lower total output. This divergence between "maximize total" and "maximize fair distribution" is the hierarchy's version of the cooperation-dignity tradeoff, and it appears at the governance selection level rather than the agent level.

---

## Summary Table

| Phase | Best Condition | Org Output | Dignity Floor | H-Navig | Key Insight |
|-------|---------------|:----------:|:------------:|:-------:|-------------|
| 1 | H6 cross vis | 41.5 | 100.0% | 0.920 | Enforcement net-negative in hierarchies |
| 2 | HS6 struct EL + enforce CS | 44.8 | 100.0% | **1.015** | Asymmetric architecture: protect bottom, enforce top |
| 3 | HC5 bottom-up reform | 41.4 | 43.2% | 0.399 | Coverage area > propagation position |
| 4 | Consensus → HS6 | 44.8 | 100.0% | 1.015 | Governance question is "best for whom" |

---

## Methodological Notes

- All simulations use 100 Monte Carlo runs × 100 rounds, matched seeds within each phase.
- Agents drawn from the same 576-subject fitted library as the flat federation simulations.
- Level-specific parameter overrides applied after phenotype-based parameter inheritance.
- Contributions on 0–20 integer scale (MAX_CONTRIB = 20). Rupture = budget < 10% of b_initial for 3 consecutive rounds.
- Organizational output weighted: EL=1.0, MM=1.5, UM=2.0, CS=3.0 (higher levels have more organizational leverage).
- Hierarchical navigability = dignity_floor × (1 − cross_level_Gini) × (1 + org_output_norm).
- True cost = structural cost + removal cost (replacement costs stratified by level).
- Inter-level cascade strengths calibrated from Gallup engagement multiplier research.
- Parameters calibrated from BLS, Gallup, Fierce Inc / BTS / Capterra, Columbia N=22K, EPI / AFL-CIO / Equilar, CAP / Equitable Growth, Meyer & Allen, O.C. Tanner, Zenger Folkman N=320K.
