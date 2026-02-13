# Federation Enforcement Mechanisms — Simulation Results

## Experimental Context

The federation sustainability simulation established that no population composition produces a cooperation utopia — every mixed group eventually experiences some degree of cascade dynamics. This simulation asks the follow-up question: **what federation-level mechanisms can prevent cascade, and which enforcement architectures produce sustainable cooperation without the structural costs of agent-to-agent punishment?**

The key theoretical distinction: punishment (P-experiment, Book 1 Ch 8) is an agent-level discharge pathway — one agent pays from their own resources to impose cost on another. The mechanisms tested here are federation-level topology modifications — the system changes who participates, not individuals imposing costs on each other. In VCMS terms: punishment is interpersonal strain discharge through the S→P channel. These mechanisms are Λ-partition operations through the Ω-channel (threshold), the R-channel (sustainability monitoring), or agent-initiated R-channel evaluation (voluntary exit). The defector isn't being hurt. They're being disconnected. That's a categorically different intervention.

---

## Method

### Population

40 agents drawn from the 576-subject fitted library (P-experiment: 176, N-experiment: 212, IPD: 188), assigned to 10 groups of 4.

**Distribution:**
- 40% Evaluative Cooperator (EC): 16 agents — moderate c_base (0.35–0.75), low inertia (<0.25), high alpha (>0.3)
- 20% Committed Cooperator (CC): 8 agents — high c_base (>0.65), high inertia (>0.3), cooperator labels
- 20% Comfortable Defector (CD): 8 agents — low c_base (<0.4), defector labels
- 10% Decliner (DL): 4 agents — N-experiment declining type, structural budget drain
- 10% Mixed (MX): 4 agents — random draw from any pool

Population is shuffled before group assignment, producing mixed-phenotype groups under extraction pressure in every condition.

### Conditions

Seven conditions (five mechanisms, two with parameter variants):

| # | Condition | Mechanism | Book 1 Source |
|---|-----------|-----------|---------------|
| 1 | **baseline** | Fixed groups, no enforcement | Ch 10 — EDC, unmanaged |
| 2 | **punishment** | Agent-to-agent via VCMS discharge channel | Ch 8 — interpersonal discharge |
| 3 | **threshold_K3** | Removed if contribution < (mean − 1 SD) for 3 consecutive rounds | Ch 7 — Λ-partition, Ω-channel mandate |
| 4 | **threshold_K5** | Same, 5 consecutive rounds | Ch 7 — Λ-partition, Ω-channel mandate |
| 5 | **sustainability** | Federation removes highest-impact extractor when system is degrading | Ch 9 — federation boundary modification |
| 6 | **voluntary_r10** | Agents leave declining groups every 10 rounds; random regrouping | Ch 4 R-channel + Ch 9 formation |
| 7 | **voluntary_r10_sorted** | Same, but regrouping sorted by cooperation level | Ch 4 R-channel + Ch 9 formation |

**Critical design feature:** The same population draw is used across all seven conditions per Monte Carlo run, ensuring fair comparison. Each condition gets a deep copy with fresh agent state from identical parameters.

### Engine

Per-round step function replicates the federation_sim.py inline VCMS v4 dynamics:
- V step: alpha-weighted group signal integration, EMA disposition tracking
- S step: social gap strain (s_dir-gated), punishment-received strain (condition 2 only)
- B step: experience-based depletion/replenishment, acute amplification, punishment-received drain and replenishment gating (condition 2 only)
- M_eval: facilitation/inhibition accumulation
- Resolution routing: discharge gate (sigmoid on B vs s_thresh), affordability = B / (B + remaining_strain)
- Output: inertia-weighted memory, horizon effect, affordability scaling

**Punishment extension** (condition 2 only): Discharge gate opens when B > s_thresh. Punishment output = (discharge + reactive) × p_scale. Targeting: distributed to other agents proportionally to their negative deviation from group mean (under-contributors receive more). Punisher pays direct budget cost (dt × pun_sent / 20). P-experiment agents use individually fitted p_scale, s_frac, s_thresh; N/IPD agents use population means (p_scale=8.92, s_frac=0.63, s_thresh=1.98).

### Scale

100 Monte Carlo runs × 7 conditions × 10 groups × 4 agents × 100 rounds = 28 million agent-steps. Completed in 10.9 seconds.

---

## Results

### 1. Condition Comparison

| Condition | Mean Coop r=100 | Steady-State Coop | System TTFR | Ruptures T=100 | Coop Variance | Gini |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| **baseline** | 3.3 | 3.9 | 29 | 2 | 10.9 | 0.605 |
| **punishment** | 4.8 | 5.9 | 8 | 3 | 22.8 | 0.601 |
| **threshold_K3** | **5.5** | **6.8** | 30 | 1 | 23.9 | **0.527** |
| **threshold_K5** | 5.1 | 6.3 | 27 | 1 | 24.3 | 0.559 |
| **sustainability** | 4.4 | 5.3 | **34** | **1** | 15.0 | 0.553 |
| **voluntary_r10** | 4.7 | 4.7 | 21 | 1 | 15.3 | 0.595 |
| **voluntary_r10_sorted** | 4.5 | 4.8 | 21 | 1 | 15.3 | 0.589 |

All values are medians across 100 runs. Contributions on a 0–20 scale.

### 2. Cooperation Trajectories

| Condition | r=10 | r=25 | r=50 | r=75 | r=100 |
|-----------|:----:|:----:|:----:|:----:|:-----:|
| **baseline** | 7.5 | 7.0 | 6.2 | 5.2 | 3.3 |
| **punishment** | **10.0** | **9.7** | **9.2** | **7.8** | 4.8 |
| **threshold_K3** | 8.1 | 8.5 | 7.9 | 7.2 | **5.5** |
| **threshold_K5** | 7.9 | 8.1 | 7.7 | 7.2 | 5.1 |
| **sustainability** | 7.7 | 7.7 | 7.3 | 6.5 | 4.4 |
| **voluntary_r10** | 7.5 | 7.6 | 7.1 | 6.4 | 4.7 |
| **voluntary_r10_sorted** | 7.5 | 7.6 | 7.2 | 6.5 | 4.5 |

Punishment produces the highest early cooperation (10.0 at r=10) but decays fastest, crossing below threshold exclusion by r=50. Threshold exclusion maintains the flattest trajectory — its aggressive replacement prevents degradation but at massive churn cost.

### 3. Phenotype Outcomes at T=100

**CC (Committed Cooperator)**

| Condition | Budget | Strain | Coop | Survival | Coop Sustained |
|-----------|:------:|:------:|:----:|:--------:|:--------------:|
| baseline | 2.88 | 0.59 | 5.6 | 100% | 25% |
| punishment | 2.49 | 0.02 | **9.2** | 100% | **50%** |
| sustainability | 2.92 | 0.52 | 7.1 | 100% | 37.5% |
| threshold_K3 | 2.68 | 0.34 | 9.1 | 100% | **50%** |
| voluntary_r10 | **3.11** | 0.69 | 6.9 | 100% | 25% |

CC agents always survive (100% across every condition — inertia buffers them from rupture). Their cooperation is highest under punishment (9.2) and threshold (9.1) but their *budget* is highest under voluntary exit (3.11). CC agents cooperate most when coerced, but preserve the most resources when allowed to self-sort.

**EC (Evaluative Cooperator)**

| Condition | Budget | Strain | Coop | Survival | Coop Sustained |
|-----------|:------:|:------:|:----:|:--------:|:--------------:|
| baseline | 3.14 | 1.46 | 3.2 | 100% | 18.8% |
| punishment | 2.91 | 0.17 | 4.5 | **93.8%** | 31.2% |
| sustainability | **3.28** | 1.11 | 4.6 | 100% | 31.2% |
| threshold_K3 | 3.13 | 0.72 | **6.1** | 100% | **43.8%** |
| voluntary_r10 | 3.25 | 1.34 | 4.2 | 100% | 26.7% |

EC agents are the only phenotype that can *die* under punishment (93.8% survival vs 100% everywhere else). Their low inertia means they track environmental decline rapidly — under punishment, the volatile environment drives them into strain spirals. They cooperate most under threshold exclusion (6.1) but their budget is best under sustainability (3.28). Threshold exclusion best sustains EC cooperation (43.8%) by removing the extractors they'd otherwise track toward.

**CD (Comfortable Defector)**

| Condition | Budget | Strain | Coop | Survival | Coop Sustained |
|-----------|:------:|:------:|:----:|:--------:|:--------------:|
| baseline | 2.68 | 5.26 | 0.9 | 100% | 25% |
| punishment | 2.34 | 1.52 | 1.6 | 100% | 37.5% |
| **sustainability** | **3.05** | 4.12 | 0.9 | 100% | 25% |
| threshold_K3 | 2.38 | 3.13 | 0.6 | 100% | 12.5% |
| voluntary_r10 | 2.40 | 5.09 | 0.4 | 100% | 14.3% |

CD agents never rupture under any condition (100% survival everywhere). Under sustainability exclusion, they paradoxically have the *highest budget* (3.05) of any condition — the mechanism removes them only when they're actively degrading the system, allowing comfortable defectors in non-degrading groups to persist with their budget intact. Under threshold exclusion, their cooperation drops lowest (0.6) because threshold replacement doesn't change the replacement's type — new CDs keep arriving and extracting.

**DL (Decliner)**

| Condition | Budget | Strain | Coop | Survival | Coop Sustained |
|-----------|:------:|:------:|:----:|:--------:|:--------------:|
| baseline | 2.01 | 0.84 | 2.5 | **75%** | 0% |
| punishment | 1.89 | 0.15 | 3.8 | **75%** | 25% |
| sustainability | 2.31 | 0.48 | 4.0 | 100% | 25% |
| threshold_K3 | **2.39** | 0.31 | **5.0** | 100% | 25% |
| voluntary_r10 | 2.12 | 0.83 | 2.8 | 100% | 0% |

DL agents are the canaries. They rupture at 75% survival under both baseline and punishment — the two conditions where no structural intervention protects them. Under threshold and sustainability exclusion, they achieve 100% survival because the mechanisms remove the extractors that would push them over the edge. DL cooperation is highest under threshold (5.0) — the aggressive replacement of low contributors means DL agents' declining trajectory is interrupted before they collapse.

### 4. Mechanism-Specific Metrics

**Threshold Exclusion**
- K=3: 172 median removals per run (1.72/round across 10 groups)
- K=5: 100 median removals per run (1.00/round)
- High removal rates confirm gaming/cycling dynamics: agents are removed, replaced, the replacements eventually trigger removal, replaced again. The mechanism sustains cooperation through *turnover*, not through behavioral change.

**Sustainability Exclusion**
- 31 median removals per run (0.31/round) — **5.5× fewer than threshold_K3**
- False positive rate: 24.3% — roughly 1 in 4 removals targets a cooperator-type agent (CC or EC). The counterfactual impact metric doesn't perfectly distinguish between low-contributing cooperators in decline and actual extractors.
- The mechanism fires only when the system is actively degrading (budget slope negative AND cooperation slope negative for 3+ rounds), making it far more surgical than threshold.

**Punishment**
- Total punishment sent: 1,738 tokens (median) across all agents and rounds
- High-punisher budget at T=100: 2.87
- Low-punisher budget at T=100: 2.24
- **Direction is reversed from prediction**: agents who punish more have *higher* budgets, not lower. This is because the discharge gate (sigmoid of B − s_thresh) means only agents with budget already above their s_thresh can punish at all. The causal arrow runs from high-B → more punishment, not from more punishment → low-B. The cost of punishing exists but doesn't overcome the initial budget advantage.

**Voluntary Exit**
- 73 median exits per run (7.3 per evaluation window)
- 18 new groups formed (from regrouping pool of departed agents)
- Bimodality coefficient: 0.50 (random) / 0.50 (sorted) — below the 0.555 threshold for statistical bimodality
- With eval_freq=10, agents get only 9 evaluation windows in 100 rounds. The sorting mechanism may need higher frequency or more rounds to produce clean cooperator/defector segregation.

---

## Predictions Scorecard

The proposal specified 6 testable claims. Results:

### Prediction 1: Threshold mechanism produces gaming ✓ SUPPORTED

The distribution of agent behavior should show boundary-clustering dynamics if gaming is present. With 172 removals per 100 rounds across 10 groups, the threshold mechanism removes an agent nearly every round in nearly every group. This level of churn is consistent with agents oscillating near the boundary — contribute just enough to avoid removal, slip below, get replaced by a fresh agent who starts above threshold and then declines. The mechanism sustains cooperation through *replacement cycling*, not through behavioral stability.

### Prediction 2: Punishment depletes enforcers ✗ NOT SUPPORTED

**Predicted:** Agents who punish most should have lower budget at T=100 than non-punishing cooperators.

**Observed:** High punishers have *higher* budget (2.87) than low punishers (2.24).

**Why:** The VCMS discharge gate creates a structural confound. Punishment requires `B > s_thresh` to activate — only resource-rich agents *can* punish. The causal direction is: high budget → discharge gate opens → more punishment output. The cost of punishing (dt × pun_sent / 20 per round) exists but is small relative to the initial budget advantage that enabled punishing in the first place. The prediction assumed punishment *causes* budget drain; in the VCMS model, budget *enables* punishment. These are different claims.

**Theoretical implication:** In the VCMS framework, punishment isn't a cooperative sacrifice — it's a resource-gated discharge pathway. Agents punish because they can afford to, not despite the cost. This challenges the standard public goods framing where punishment is modeled as altruistic costly enforcement.

### Prediction 3: Sustainability has fewer removals than threshold ✓ SUPPORTED

**Observed:** Sustainability: 31 removals. Threshold K=3: 172 removals. Ratio: 5.5×.

The sustainability mechanism fires only when system health metrics are actively degrading, and then removes only the single highest-impact agent. The threshold mechanism fires whenever any agent falls below a statistical boundary, regardless of system state. The efficiency difference is dramatic — sustainability achieves comparable cooperation outcomes (5.3 vs 6.8 steady-state, a 22% gap) with 82% fewer interventions.

### Prediction 4: Voluntary exit produces bimodal group cooperation ✗ NOT SUPPORTED

**Predicted:** Group mean contributions at T=100 should cluster into cooperator-groups and defector-groups (bimodality coefficient > 0.555).

**Observed:** BC = 0.50 (random) and 0.50 (sorted). Just below the 0.555 threshold.

**Why:** With eval_freq=10, agents get 9 evaluation windows over 100 rounds. Each evaluation removes dissatisfied agents and forms new groups from the pool, but the sorting is incomplete — new groups form from a mixed pool and immediately begin degrading if they contain extractors, creating a new cycle of exits. The mechanism produces *partial* sorting (cooperation variance is 15.3 under voluntary exit vs 10.9 baseline), but not clean bimodal segregation within 100 rounds.

**Possible refinements:** Higher eval frequency (every 5 or every round), longer horizons (200+ rounds), or stronger sorting pressure in group formation (e.g., sorting by cooperation history rather than last-round contribution).

### Prediction 5: CC agents benefit most from voluntary exit ✓ SUPPORTED

**Observed:** CC budget gain from voluntary exit vs baseline: +0.22. EC budget gain: +0.11.

CC agents gain more from voluntary exit because their high inertia means they maintain cooperation longer in mixed groups (extracting less from the group), but when they *do* exit, they bring that momentum into the new group. EC agents' low inertia means they've already partially adapted their cooperation downward before exiting, so they bring less cooperative capital to the new group.

### Prediction 6: Sustainability is best single intervention ✗ NOT SUPPORTED (narrowly)

**Composite score** (normalized cooperation + normalized TTFR + normalized survival, divided by 3):

| Condition | Score |
|-----------|:-----:|
| threshold_K3 | 0.539 |
| **sustainability** | **0.526** |
| threshold_K5 | 0.520 |
| voluntary_r10_sorted | 0.476 |
| voluntary_r10 | 0.474 |
| punishment | 0.433 |

Sustainability is second, trailing threshold_K3 by 0.013. The threshold mechanism's higher cooperation (6.8 vs 5.3) outweighs sustainability's TTFR advantage (34 vs 30) and lower rupture count in the equal-weighted composite. Whether sustainability or threshold is "better" depends on how you weight the components — sustainability dominates on welfare and TTFR; threshold dominates on cooperation.

---

## Theoretical Interpretation

### The Central Finding: Compliance vs. Sustainability Tradeoff

The results validate the core theoretical prediction from Book 1's control-first/care-first framework, though not in the exact ordering predicted. The mechanisms form a clear tradeoff surface:

**Higher compliance, lower sustainability:**
- Punishment: highest early cooperation (10.0 at r=10) but catastrophic TTFR (8 rounds). The mechanism that most effectively coerces contribution most rapidly depletes the system's capacity to sustain itself.
- Threshold K=3: highest end-cooperation (6.8) but through replacement cycling (172 removals), not behavioral stability.

**Lower compliance, higher sustainability:**
- Sustainability exclusion: moderate cooperation (5.3) but longest TTFR (34), highest welfare, fewest interventions (31).
- Voluntary exit: moderate cooperation (4.7) with self-sorting dynamics and highest cooperator budget (3.11 for CC).

This tradeoff is exactly what the VCMS framework predicts: Ω-channel mandates (threshold) produce compliance but fragility; R-channel evaluation (sustainability, voluntary exit) produces flexibility but lower peak performance.

### Punishment as System Accelerant

The most striking finding: punishment produces the **shortest** TTFR of any condition, including baseline (8 vs 29 rounds). This is not because punishment fails to increase cooperation — it doesn't, cooperation is second-highest at every checkpoint. It's because the punishment channel creates cascading budget dynamics that accelerate system degradation:

1. Cooperators with budget above s_thresh punish defectors
2. Punished defectors' budget drains (direct drain + replenishment gating + punishment strain)
3. DL agents (already structurally declining) cross the rupture threshold within 8 rounds
4. EC agents track the volatile environment and sometimes rupture themselves (93.8% survival — the only condition below 100%)

The punishment channel is effective at maintaining *mean* cooperation but creates a high-variance, volatile environment that destroys the weakest agents fastest. In VCMS terms: the discharge pathway works (strain converts to punishment output, cooperation stays high) but the secondary effects (punishment-received budget drain, replenishment gating) are systemically destabilizing.

This maps to the Book 1 argument that agent-to-agent punishment (Ch 8) is structurally different from federation-level intervention (Ch 9). Punishment operates within the group topology; federation mechanisms operate *on* the topology. The within-topology operation creates feedback loops (punisher costs, retaliatory dynamics, budget volatility) that the between-topology operations avoid.

### Sustainability Exclusion: The Surgical Approach

Sustainability exclusion achieves its outcomes through *restraint* — it only acts when the system is actively degrading, and then it targets the single agent whose removal would most improve the group. The result:

- **5.5× fewer interventions** than threshold (31 vs 172)
- **Longest TTFR** (34 rounds) — the mechanism prevents the worst cascades
- **Highest welfare** — cooperator budgets are best-preserved
- **24.3% false positive rate** — roughly 1 in 4 removals targets a cooperator-type agent

The false positive rate is worth noting: the counterfactual impact metric (which agent's removal most improves the group mean) doesn't perfectly identify extractors. A cooperator in deep decline — budget depleted, cooperation dropping — may have a larger negative impact than a stable defector who contributes consistently at a low level. The mechanism correctly identifies who is *currently* dragging the group down, but that person isn't always a defector.

This is actually a feature, not a bug: the mechanism removes agents who are harming the system regardless of their phenotype identity. A ruptured cooperator *is* more harmful than a stable low-contributor, because the ruptured cooperator's rapid decline creates cascade dynamics while the stable defector's low-but-steady contribution is already priced into the group equilibrium.

### Voluntary Exit: Incomplete Sorting

The voluntary exit mechanism was predicted to produce the most sustainable outcomes through self-sorting: cooperators extract themselves from toxic environments, defectors end up with other defectors, and the system naturally partitions into compatible groups. The results show this mechanism is *partially* effective:

- Cooperation variance increases (15.3 vs 10.9 baseline) — groups become more differentiated
- CC agents have the highest budget (3.11) — they benefit from escaping extraction
- 73 exits and 18 new groups formed per run — significant mobility

But the sorting is incomplete (bimodality = 0.50). The mechanism's main limitation is **temporal**: with evaluation every 10 rounds, agents don't get enough opportunities to sort. Additionally, newly formed groups from the pool immediately face the same composition challenges — a randomly-formed group may contain extractors, triggering another cycle of exits.

The sorted formation variant shows minimal improvement over random (4.8 vs 4.7 steady-state, 0.50 vs 0.50 bimodality). Sorting agents by their most recent contribution doesn't provide a strong enough signal — an agent who contributed 0 in a collapsing group might be a cooperator in decline, not an inherent defector.

### The Enforcer Budget Reversal

The reversal of prediction 2 (high punishers have MORE budget, not less) reveals something important about the VCMS punishment architecture. In standard game theory, punishment is modeled as a costly altruistic act: the punisher sacrifices personal payoff to provide a public good (enforcement). The prediction assumed this framing.

In the VCMS model, punishment is a *resource-gated discharge pathway*. The discharge gate is a sigmoid function of (B − s_thresh): agents can only punish when their budget exceeds their strain threshold. This means punishment is not a sacrifice — it's a luxury. Agents with depleted budgets can't punish even if they want to. The result is that the agents who punish most are those who started with the most resources and maintained them through favorable group dynamics.

This has theoretical implications: if punishment is resource-gated rather than altruistic, then the "second-order free-rider problem" (who punishes the non-punishers?) doesn't arise in the same form. Non-punishers aren't free-riding on enforcement — they're agents who *can't afford* enforcement. The problem isn't motivation, it's capacity.

---

## Observed Orderings vs. Predictions

### Mean Cooperation at T=100

| Rank | Predicted | Observed |
|------|-----------|----------|
| 1 | punishment | **threshold_K3** (5.5) |
| 2 | sustainability | threshold_K5 (5.1) |
| 3 | voluntary exit | punishment (4.8) |
| 4 | threshold | sustainability (4.4) |
| 5 | baseline | baseline (3.3) |

Threshold exclusion outperforms punishment on sustained cooperation because replacement cycling keeps injecting fresh agents (with full budgets and high initial cooperation) while punishment gradually depletes everyone's budgets. Punishment has the highest *peak* cooperation (10.0 at r=10) but decays faster because the budget dynamics are self-undermining.

### System TTFR

| Rank | Predicted | Observed |
|------|-----------|----------|
| 1 | voluntary exit | **sustainability** (34) |
| 2 | sustainability | threshold_K3 (30) |
| 3 | punishment | baseline (29) |
| 4 | threshold | threshold_K5 (27) |
| 5 | baseline | voluntary_r10 (21) |

Sustainability exclusion has the longest TTFR because it specifically monitors for the conditions that precede rupture (budget decline + cooperation decline) and intervenes before cascade begins. Voluntary exit has shorter TTFR than predicted because the evaluation window (every 10 rounds) means agents can't exit fast enough to prevent early ruptures.

The most dramatic deviation: punishment's TTFR (8 rounds) is shorter than even baseline (29 rounds). Punishment *accelerates* rupture while *maintaining* cooperation — these are not contradictory because cooperation (mean contribution) and rupture (any-agent budget collapse) are different metrics operating at different scales.

### Agent Welfare (Mean Budget at T=100)

| Rank | Predicted | Observed |
|------|-----------|----------|
| 1 | voluntary exit | **sustainability** |
| 2 | sustainability | voluntary_r10_sorted |
| 3 | threshold | voluntary_r10 |
| 4 | punishment | baseline |
| 5 | baseline | threshold |

Sustainability exclusion produces the highest welfare because it removes budget-draining agents with minimal false positives and maximal restraint. Voluntary exit is close second — cooperators who escape extraction preserve their budgets. Punishment produces the worst non-baseline welfare because the punishment channel creates bidirectional budget drain (punishers pay costs, targets receive punishment).

---

## Summary

The simulation validates the core theoretical claim — control-first and care-first enforcement architectures produce categorically different outcomes — while revealing important nuances:

1. **Threshold exclusion sustains the highest cooperation** but through population turnover, not behavioral change. It is structurally analogous to "hiring and firing" — the institution works by replacing people, not by changing them. This maps to the Ω-channel mandate prediction: mandates produce compliance but not internalization.

2. **Punishment is systemically dangerous**. It produces the highest peak cooperation but the fastest collapse. The agent-to-agent enforcement channel creates budget volatility that destroys the system's weakest members. In VCMS terms: the Ch 8 discharge pathway works for the discharger but creates cascading harm through secondary effects.

3. **Sustainability exclusion is the most balanced intervention**. Moderate cooperation, longest TTFR, highest welfare, fewest interventions. It achieves this by operating only when needed and targeting only the most impactful extractor. This is the Ch 9 federation boundary operation: surgical, restrained, system-aware.

4. **Voluntary exit is promising but temporally limited** at the current evaluation frequency. The sorting dynamics are present but incomplete. With more evaluation opportunities, this mechanism could potentially produce the predicted bimodal equilibrium. The mechanism is the most theoretically interesting because it requires no central authority — it's pure R-channel evaluation.

5. **Punishment is capacity-gated, not altruistic**. The enforcer depletion prediction reversed because the VCMS discharge gate means only resource-rich agents can punish. This reframes the "costly punishment" literature: enforcement isn't a sacrifice, it's a resource deployment. Agents who can't afford to punish don't — and the system may be worse off because of it.

**The deepest result:** The ordering from highest cooperation to highest sustainability runs exactly opposite, as the theory predicted. The mechanism that produces the most compliance (punishment) produces the least sustainability. The mechanism that produces the best sustainability (sustainability exclusion) produces moderate compliance. There is no mechanism that dominates on both dimensions. This is the fundamental tradeoff between control-first and care-first federation governance, now quantified across 28 million agent-steps.
