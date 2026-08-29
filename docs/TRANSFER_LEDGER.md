# Transfer Ledger

**The claim this repository is built to test:** closing an adversarial red-team
loop is not, by itself, evidence that the loop makes a fraud detector better on
real fraud. Whether it helps or hurts depends on the *fidelity* of the attack
generator, and low-fidelity augmentation can measurably reduce real-fraud
recall while making every dashboard number look better.

This document explains the experiment, how to read its output, and where the
result stops being valid.

---

## 1. Why this experiment exists

The standard red-team demo has this shape:

1. Generate synthetic attacks.
2. Find the ones that evade the detector.
3. Retrain the detector on those escapes.
4. Show that recall on the synthetic attacks went up.

Step 4 is the problem. Recall measured **on the generator's own output** is a
measure of how well the detector learned the generator. It is not a measure of
how well it learned fraud. If the generator produces transactions with correct
marginal distributions but impossible joint structure -- a card-not-present
amount attached to a chip entry mode at an hour the cardholder never transacts
-- then the detector is being taught to recognise an artefact.

The failure mode is documented, including by systems that report it honestly.
One comparable submission published a generator with a C2ST AUC of **0.9801**
(1.0 = trivially distinguishable from real data) and, after hardening on those
attacks, a **3.8 point loss of recall on real fraud** and a 3.6 point loss of
PR-AUC, while synthetic-attack recall rose 44 points. Its own conclusion was
that closing the gap requires a learned generative model.

That is exactly the gap this experiment measures and this repository closes.

---

## 2. Design

One independent variable: **the attack generator.** Everything else is pinned.

| Arm | Generator | What it represents |
|-----|-----------|--------------------|
| **A0** | none | the unaugmented baseline detector |
| **A1** | `independent_marginal` | rule, template and parametric attack families: correct marginals, joint structure destroyed |
| **A2** | `gaussian_copula` | marginals **and** learned rank dependence between fields |

Held constant across all three arms:

- the same real training rows (90 real fraud rows -- labelled fraud is always scarce)
- the same augmentation budget (120 synthetic rows, held near the real-fraud count
  so synthetic data *augments* the real rows rather than swamping them)
- the same detector family and hyperparameters
- the same calibration protocol and target false-positive rate (1.00%)
- the same held-out real-fraud test set
- three seeds, with nonparametric bootstrap confidence intervals on every number

Because the budget is fixed, no generator can win by producing more data. It
can only win by producing *better* data.

### The generators

`backend/fidelity/copula.py` implements both arms behind one interface, so the
comparison is not confounded by implementation differences.

- **`IndependentMarginalSynthesizer`** fits an empirical marginal per field and
  samples each field independently. Every column is individually realistic.
  Every row is jointly implausible.
- **`GaussianCopulaSynthesizer`** maps each field to a uniform through its
  empirical CDF, then to a Gaussian latent through the normal quantile
  function, estimates the latent correlation matrix, projects it to the nearest
  positive semi-definite correlation matrix, samples from the multivariate
  normal, and maps back through the marginals. Mixed continuous and categorical
  fields are handled by interval jitter within each category's probability mass,
  with categories ordered by frequency so the latent correlation carries a
  consistent ordinal meaning rather than an alphabetical accident.
  It additionally honours the two **functional constraints** every real payload
  satisfies: `(hour_sin, hour_cos)` is renormalised onto the unit circle, and
  `mcc_group` is *derived* from the sampled `mcc_num` instead of drawn
  independently. The control arm deliberately violates both -- a row whose
  encoded hour is off-circle, or whose merchant category contradicts its own MCC,
  is not merely unusual, it is impossible. That gap **is** the fidelity variable.

The difference between the two arms is precisely the joint structure -- rank
dependence plus these hard functional relationships -- which is what makes a
fraudulent transaction look coherent rather than merely unusual.

---

## 3. What gets measured

### Transfer (`artifacts/transfer_ledger.json`)

Per arm, on **held-out real fraud**:

- recall at a false-positive rate pinned to 1.00% on a **disjoint** legitimate validation split
- realised false-positive rate on the test split
- ROC-AUC and PR-AUC
- precision at laboratory prevalence *and* at ~1.3% production prevalence
- `delta_recall_vs_baseline` -- the headline quantity

### Fidelity (`artifacts/fidelity_report.json`)

Five measures per generator, reported together:

| Measure | Ideal | What it detects |
|---------|-------|-----------------|
| C2ST AUC | 0.50 | joint separability -- the strictest test |
| mean JSD | 0 | marginal agreement -- the easiest test |
| mean TVD | 0 | categorical agreement |
| correlation Frobenius diff | 0 | rank-dependence structure |
| TSTR ratio | 1.00 | usefulness for training a detector that must work on real fraud |

A generator that passes only JSD and TVD is matching histograms. Publishing all
five is the point: the marginal measures are the ones a weak generator passes,
and quoting them alone is how a weak generator gets called realistic.

**Acceptance gate:** copula C2ST AUC <= 0.80. The observed value is published
whether or not the gate is cleared. A fidelity lab that reports only its passes
is marketing.

### Economics (`artifacts/economics.json`)

Every operating point priced in INR, including the term most fraud demos omit:
the **insult cost** of wrongly declining a legitimate payment, modelled as lost
interchange margin plus support contact cost plus probability-weighted churn
against customer lifetime value. At a 1% false-positive rate on high-volume
authorisation traffic this term dominates the ledger: **52.8%** of all cost
incurred at the measured operating point, against 20.7% for the fraud that got
through. All rates are explicit, stated as assumptions, and overridable in
`evidence/economics.CostModel`.

---

## 4. Result as measured

Numbers below are the three-seed means emitted by `make reproduce`; the
authoritative copies live in `artifacts/transfer_ledger.json`.

| | A0 baseline | A1 independent | A2 copula | published comparator |
|---|---|---|---|---|
| C2ST AUC (0.50 ideal) | -- | 0.970 | **0.881** | 0.980 |
| Frobenius corr. diff (0 ideal) | -- | 2.32 | **1.16** | -- |
| mean JSD (0 ideal) | -- | 0.038 | 0.038 | -- |
| recall on held-out real fraud | 0.9989 | 0.9844 | 0.9933 | -- |
| **Δrecall vs baseline** | -- | **−0.014** | **−0.006** | −0.038 |

Three things follow, and the third is the one that matters.

**The fidelity contrast is real and it is joint-structure-only.** The copula is
0.089 AUC closer to indistinguishable and halves the rank-dependence error,
while *mean JSD is identical to three decimal places*. The two arms match on
marginals and diverge only on the joint -- which is exactly the variable the
design pins everything else to isolate.

**Fidelity orders transfer harm, monotonically and in every seed.** The copula's
recall penalty is roughly a third of the control's, and it is the smaller penalty
in all three seeds individually, not merely on average. Both arms improve on the
published parametric comparator's −3.8 points.

**Augmentation did not turn positive, and we report that instead of engineering
around it.** The unaugmented baseline already recalls 0.9989 of held-out real
fraud, so there is no headroom for synthetic data to win -- a ceiling of this
corpus, not a property of copulas. Manufacturing a positive Δrecall from here
would mean handicapping A0 until augmentation could beat it, which would make the
number a presentation artefact rather than a measurement.

The transferable finding is therefore the **ordering**, not the sign. Fidelity is
measurable *before* deployment -- C2ST and Frobenius need synthetic rows and
held-out real rows, no incident labels and no retrained detector -- and it ranks
transfer harm *after*. That is what makes a pre-registered **fidelity gate**
operationally useful to an issuer: it rejects a red-team generator that would
degrade a live detector, before the generator is ever allowed near one. On this
corpus the gate's own threshold (C2ST <= 0.80) is **not** cleared by either arm,
including our own best generator. That is published here rather than relaxed.

---

## 5. How to read the headline figure

`docs/transfer_ledger.png`, left panel: generator C2ST AUC on the x-axis
against change in real-fraud recall on the y-axis.

- Above the dashed line, red-teaming **helped** the real detector.
- Below it, red-teaming **hurt** the real detector while its synthetic-attack
  metrics improved.
- The grey X is the published comparator (C2ST 0.980, real-fraud recall
  -3.8 pts), included as an external reference point, not as our measurement.

Right panel: recall per arm with bootstrap 95% CIs, at a false-positive rate
held identical across arms, so the bars are comparable.

---

## 6. Reproduce

```bash
make install
make reproduce          # calibration + fidelity + transfer

# or individually
python backend/experiments/run_calibration_audit.py
python backend/experiments/run_fidelity.py
python backend/experiments/run_transfer_ablation.py
```

Every artifact carries a provenance block: git sha, seeds, Python version,
platform and the exact command. `artifacts/claim_ledger.json` maps each public
claim to the artifact field that supports it, the derivation, and its boundary.
The UI at `/evidence` renders that ledger directly and shows *missing* rather
than a placeholder when an artifact has not been generated.

---

## 7. Boundaries

Stated here rather than in a footnote, because a fraud result without its
boundaries is how systems get deployed badly.

1. **"Real fraud" means held-out fraud from this repository's topology-aware
   payment environment, not issuer production data.** The experiment establishes
   a *relationship* between generator fidelity and transfer. It does not claim
   an absolute recall figure for live card traffic.
2. **The baseline is at a ceiling, so the *sign* of transfer is not testable on
   this corpus.** A0 recalls 0.9989 of held-out real fraud unaugmented. No
   augmentation can show a meaningful gain against that, so this experiment can
   only rank how much different generators *cost*, not demonstrate that a
   high-fidelity one pays. Showing positive transfer needs a corpus where the
   real-only baseline genuinely struggles; that is the next experiment, not a
   result claimed here.
3. **Prevalence is an assumption, not a measurement.** The model does not change
   when the base rate does; the precision an analyst experiences changes
   enormously. That is why the sweep is published instead of a single number.
4. **A Gaussian copula captures rank dependence only.** It does not capture
   higher-order interactions, sequence or session structure, or graph-level
   collusion topology. Higher fidelity is available and not claimed here.
5. **The augmentation budget is fixed.** Results may differ at other budgets;
   that variable was closed deliberately to isolate fidelity.
6. **Three seeds is a small sample.** Every figure ships with a bootstrap
   interval so the reader can see how small.
7. **INR cost rates are order-of-magnitude assumptions.** Disagree with them and
   recompute -- they are one dataclass.
