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

- the same real training rows
- the same augmentation budget (750 synthetic rows, matching published comparators)
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
  fields are handled by interval jitter within each category's probability mass.

The difference between the two arms is precisely the rank-dependence structure,
which is what makes a fraudulent transaction look coherent rather than merely
unusual.

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
authorisation traffic this term dominates the ledger. All rates are explicit,
stated as assumptions, and overridable in `evidence/economics.CostModel`.

---

## 4. How to read the headline figure

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

## 5. Reproduce

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

## 6. Boundaries

Stated here rather than in a footnote, because a fraud result without its
boundaries is how systems get deployed badly.

1. **"Real fraud" means held-out fraud from this repository's topology-aware
   payment environment, not issuer production data.** The experiment establishes
   a *relationship* between generator fidelity and transfer. It does not claim
   an absolute recall figure for live card traffic.
2. **Prevalence is an assumption, not a measurement.** The model does not change
   when the base rate does; the precision an analyst experiences changes
   enormously. That is why the sweep is published instead of a single number.
3. **A Gaussian copula captures rank dependence only.** It does not capture
   higher-order interactions, sequence or session structure, or graph-level
   collusion topology. Higher fidelity is available and not claimed here.
4. **The augmentation budget is fixed.** Results may differ at other budgets;
   that variable was closed deliberately to isolate fidelity.
5. **Three seeds is a small sample.** Every figure ships with a bootstrap
   interval so the reader can see how small.
6. **INR cost rates are order-of-magnitude assumptions.** Disagree with them and
   recompute -- they are one dataclass.
