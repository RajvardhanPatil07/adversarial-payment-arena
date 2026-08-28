"""Dependency-free test suite for the evidence layer.

Run:  python backend/tests/run_tests.py

No pytest. Every test asserts a property that would break silently if the
implementation regressed: metric identities, ordering relationships, and known
edge cases. Four assertions in this file were wrong when first written -- all
four were errors in the TEST, not in the code under test. Where that happened
the corrected reasoning is recorded in a comment, because a test whose threshold
was tuned until it passed is worse than no test.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import pandas as pd

from evidence.actions import (
    APPROVE,
    COOLING_OFF,
    DECLINE,
    STEP_UP,
    ActionCostModel,
    choose_action,
    evaluate_policy,
    sweep_policies,
)
from evidence.privacy import (
    attribute_inference,
    duplicate_share,
    membership_inference_auc,
    nearest_distances,
    privacy_audit,
)
from evidence.thresholds import (
    bootstrap_mean_ci,
    pin_threshold_at_fpr,
    precision_at_prevalence,
    rate_at_or_above,
)
from fidelity.behavior import (
    BEHAVIOURAL_FEATURES,
    ROW_FEATURES,
    build_features,
    matrix,
)
from fidelity.c2st_plus import c2st_report, sliced_c2st
from fidelity.divergence import (
    composite_similarity,
    compare_numeric,
    jsd_numeric,
    ks_statistic,
    normalised_wasserstein1,
    tvd_categorical,
)
from fidelity.fixtures import (
    simulate_legit,
    simulate_real_fraud,
    synth_joint_behavioural,
    synth_marginal,
)
from fidelity.graphstats import component_sizes, fanout, graph_fidelity_report, reuse_rate
from fidelity.temporal import (
    burstiness,
    campaign_durations,
    inter_arrival_times,
    night_share,
    sequence_lengths,
    temporal_fidelity_report,
    velocity_counts,
)
from ml.forest import RandomForestBinary, oof_scores, roc_auc, stratified_folds

PASSED = []


def test(fn):
    """Run immediately; abort the whole suite on the first failure."""
    name = fn.__name__
    try:
        fn()
    except Exception:
        print(f"FAILED: {name}")
        traceback.print_exc()
        print(f"\n{len(PASSED)} tests passed before the failure.")
        sys.exit(1)
    PASSED.append(name)
    print(f"  ok  {name}")
    return fn


def _stream(kind: str, n_cust: int = 50, seed: int = 1) -> pd.DataFrame:
    if kind == "legit":
        return simulate_legit(n_cust * 12, seed=seed)
    return simulate_real_fraud(n_cust * 4, seed=seed)


# --------------------------------------------------------------------------
# ml.forest
# --------------------------------------------------------------------------
@test
def test_roc_auc_edges():
    assert roc_auc([0, 1], [0.1, 0.9]) == 1.0
    assert roc_auc([0, 1], [0.9, 0.1]) == 0.0
    # all ties must give exactly chance, via mid-rank handling
    assert abs(roc_auc([0, 1, 0, 1], [0.5] * 4) - 0.5) < 1e-12
    # single-class input is undefined, not 0.5
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    # invariant to monotone rescaling
    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
    assert abs(roc_auc(y, s) - roc_auc(y, [x * 10 + 3 for x in s])) < 1e-12


@test
def test_forest_learns_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = (((X[:, 0] * X[:, 1]) > 0) & (X[:, 2] > -0.4)).astype(int)
    model = RandomForestBinary(n_estimators=40, seed=0).fit(X[:450], y[:450])
    auc = roc_auc(y[450:], model.predict_proba(X[450:]))
    assert auc > 0.85, auc
    imp = model.feature_importances_
    assert abs(imp.sum() - 1.0) < 1e-9
    # CORRECTED ASSERTION: the label is (X0*X1>0) & (X2>-0.4). An axis-aligned
    # split on feature 2 legitimately beats an XOR pair on impurity decrease, so
    # argmax == 2 is CORRECT, not a bug. The real property is that the three
    # informative features dominate the three noise features.
    assert imp[[0, 1, 2]].min() > imp[[3, 4, 5]].max(), imp
    assert imp[[0, 1, 2]].sum() > 0.70, imp


@test
def test_forest_null_is_chance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    y = rng.integers(0, 2, size=400)
    scores, _ = oof_scores(X, y, n_estimators=30, seed=1)
    auc = roc_auc(y, scores)
    assert 0.35 < auc < 0.65, auc


@test
def test_stratified_folds_partition_exactly_once():
    y = np.array([0] * 90 + [1] * 12)
    folds = stratified_folds(y, k=3, seed=0)
    allidx = np.concatenate(folds)
    assert np.array_equal(np.sort(allidx), np.arange(y.size))
    for f in folds:
        assert set(np.unique(y[f]).tolist()) == {0, 1}


# --------------------------------------------------------------------------
# fidelity.divergence
# --------------------------------------------------------------------------
@test
def test_divergence_identities_and_separation():
    rng = np.random.default_rng(2)
    a = rng.normal(size=3000)
    same_dist = rng.normal(size=3000)
    far = rng.normal(loc=6.0, size=3000)
    assert ks_statistic(a, a) == 0.0
    assert jsd_numeric(a, a) == 0.0
    assert tvd_categorical(list("aabbcc"), list("aabbcc")) == 0.0
    assert ks_statistic(a, far) > 0.95
    assert ks_statistic(a, same_dist) < 0.10
    # CORRECTED ASSERTION: pooled-IQR normalisation of a BIMODAL mixture makes
    # the scale track the separation itself, so a 6-sigma shift normalises to
    # about 1.0 rather than to 6. The original `> 2.0` was my error, not the
    # metric's. The honest property is separation vs identity, plus a documented
    # caveat in divergence.py.
    assert normalised_wasserstein1(a, far) > 0.8
    assert normalised_wasserstein1(a, same_dist) < 0.1
    rec = compare_numeric("x", a, far)
    assert rec["similarity_transform"] == "1 - KS"
    assert rec["similarity"] < 0.05
    assert composite_similarity([rec]) == rec["similarity"]
    assert composite_similarity([]) is None


# --------------------------------------------------------------------------
# fidelity.temporal
# --------------------------------------------------------------------------
@test
def test_burstiness_orders_processes_correctly():
    rng = np.random.default_rng(3)

    def frame(gaps):
        return pd.DataFrame(
            {"customer": "C", "ts": np.cumsum(gaps), "amount": 100.0}
        )

    regular = frame(np.full(400, 60.0))
    poisson = frame(rng.exponential(60.0, size=400))
    clustered = frame(
        np.where(rng.uniform(size=400) < 0.85, rng.uniform(1, 5, 400), rng.uniform(3000, 9000, 400))
    )
    b_reg = float(burstiness(regular)[0])
    b_poi = float(burstiness(poisson)[0])
    b_clu = float(burstiness(clustered)[0])
    # calibrated bands, verified by printing actual values before asserting
    assert b_reg < -0.999, b_reg
    assert -0.25 < b_poi < 0.25, b_poi
    assert b_clu > 0.30, b_clu
    assert b_clu > b_poi > b_reg


@test
def test_temporal_helpers():
    df = pd.DataFrame(
        {
            "customer": ["A", "A", "A", "B", "B"],
            "ts": [0.0, 100.0, 200.0, 0.0, 10_000.0],
            "amount": [1.0] * 5,
        }
    )
    assert np.allclose(np.sort(inter_arrival_times(df)), [100.0, 100.0, 10_000.0])
    assert sorted(sequence_lengths(df).tolist()) == [2.0, 3.0]
    assert sorted(campaign_durations(df).tolist()) == [200.0, 10_000.0]
    v = velocity_counts(df, 150.0)
    assert v.shape == (5,)
    assert v.max() >= 1.0
    # every timestamp here is inside hour 0..3 of a day
    assert night_share(df) == 1.0


@test
def test_temporal_report_separates_processes():
    real = _stream("fraud", seed=5)
    marg = synth_marginal(real, len(real), seed=7)
    joint = synth_joint_behavioural(real, len(real), seed=7)
    r_marg = temporal_fidelity_report(real, marg)
    r_joint = temporal_fidelity_report(real, joint)
    assert r_marg["composite_similarity"] is not None
    assert r_joint["composite_similarity"] is not None
    # the whole thesis: the joint generator must win on TEMPORAL fidelity
    assert r_joint["composite_similarity"] > r_marg["composite_similarity"], (
        r_joint["composite_similarity"],
        r_marg["composite_similarity"],
    )
    names = {m["measure"] for m in r_marg["measures"]}
    assert "burstiness_by_device" in names


# --------------------------------------------------------------------------
# fidelity.graphstats
# --------------------------------------------------------------------------
@test
def test_graph_stats_detect_collapsed_topology():
    ring = pd.DataFrame(
        {
            "customer": [f"C{i%5}" for i in range(40)],
            "device": ["DEV_SHARED"] * 40,
            "ip": ["IP_SHARED"] * 40,
            "merchant": [f"M{i%2}" for i in range(40)],
            "ts": np.arange(40.0),
            "amount": 100.0,
        }
    )
    isolated = pd.DataFrame(
        {
            "customer": [f"C{i}" for i in range(40)],
            "device": [f"D{i}" for i in range(40)],
            "ip": [f"I{i}" for i in range(40)],
            "merchant": [f"M{i}" for i in range(40)],
            "ts": np.arange(40.0),
            "amount": 100.0,
        }
    )
    assert reuse_rate(ring, "device", "customer") == 1.0
    assert reuse_rate(isolated, "device", "customer") == 0.0
    assert fanout(ring, "device", "customer")[0] == 5.0
    # one shared component vs 40 isolated ones
    assert component_sizes(ring).max() > component_sizes(isolated).max()
    assert component_sizes(isolated).size == 40
    rep = graph_fidelity_report(ring, isolated)
    assert rep["composite_similarity"] is not None
    assert rep["real_largest_component"] > rep["synth_largest_component"]


# --------------------------------------------------------------------------
# fidelity.fixtures
# --------------------------------------------------------------------------
@test
def test_fixture_structure():
    legit = simulate_legit(2000, seed=11)
    fraud = simulate_real_fraud(300, seed=23)
    for df in (legit, fraud):
        for col in ("customer", "device", "ip", "merchant", "mcc", "amount", "ts", "entry_mode"):
            assert col in df.columns, col
        assert df["ts"].is_monotonic_increasing
        assert (df["amount"] > 0).all()
    assert set(legit["label"]) == {0}
    assert set(fraud["label"]) == {1}
    # fraud must be the higher-value, more concentrated stream
    assert float(fraud["amount"].median()) > float(legit["amount"].median())
    assert float(burstiness(fraud, "device").mean()) > float(burstiness(legit, "device").mean())


@test
def test_generators_differ_in_the_intended_way():
    real = simulate_real_fraud(300, seed=23)
    marg = synth_marginal(real, 600, seed=101)
    joint = synth_joint_behavioural(real, 600, seed=101)
    assert len(marg) == 600 and len(joint) == 600
    # marginal generator invents a fresh entity per row: no reuse at all
    assert reuse_rate(marg, "device", "customer") == 0.0
    # joint generator rebuilds ring topology
    assert reuse_rate(joint, "device", "customer") > 0.0
    # neither may copy real rows verbatim
    real_keys = set(map(tuple, np.round(real[["amount", "ts"]].to_numpy(float), 6)))
    for synth in (marg, joint):
        keys = set(map(tuple, np.round(synth[["amount", "ts"]].to_numpy(float), 6)))
        assert len(keys & real_keys) == 0
    # marginal keeps amount marginals close (that is its whole trick)
    assert ks_statistic(real["amount"], marg["amount"]) < 0.15


@test
def test_feature_builder_is_finite_and_complete():
    df = build_features(simulate_real_fraud(200, seed=23))
    for col in BEHAVIOURAL_FEATURES:
        assert col in df.columns, col
        assert np.isfinite(df[col].to_numpy(dtype=float)).all(), col
    assert len(ROW_FEATURES) == 6
    assert len(BEHAVIOURAL_FEATURES) == 14
    assert set(ROW_FEATURES).issubset(set(BEHAVIOURAL_FEATURES))
    assert matrix(df, ROW_FEATURES).shape == (len(df), 6)
    try:
        matrix(df, ["nope"])
    except ValueError as e:
        assert "features not built" in str(e)
    else:
        raise AssertionError("matrix must reject unknown feature names")


# --------------------------------------------------------------------------
# fidelity.c2st_plus
# --------------------------------------------------------------------------
@test
def test_c2st_cannot_separate_two_halves_of_one_sample():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(500, 5))
    rep = c2st_report(X[:250], X[250:], [f"f{i}" for i in range(5)], n_permutations=0)
    assert 0.35 < rep["c2st_auc"] < 0.65, rep["c2st_auc"]
    assert rep["passes_gate"] is True
    assert rep["ci95"]["lo"] is not None and rep["ci95"]["lo"] <= rep["c2st_auc"]
    assert rep["ci95"]["hi"] >= rep["c2st_auc"]


@test
def test_c2st_exposes_the_marginal_generator():
    real = simulate_real_fraud(300, seed=23)
    marg = build_features(synth_marginal(real, 400, seed=101))
    realf = build_features(real)
    rep = c2st_report(
        matrix(realf, BEHAVIOURAL_FEATURES),
        matrix(marg, BEHAVIOURAL_FEATURES),
        list(BEHAVIOURAL_FEATURES),
        n_permutations=4,
    )
    # a generator with no entity structure is trivially separable on graph features
    assert rep["c2st_auc"] > 0.80, rep["c2st_auc"]
    assert rep["passes_gate"] is False
    top = {d["feature"] for d in rep["most_discriminative_features"]}
    assert top & {"device_customer_count", "ip_customer_count", "merchant_customer_count"}, top
    assert rep["permutation_null"]["n_permutations"] == 4
    assert rep["permutation_null"]["mean"] < rep["c2st_auc"]


@test
def test_sliced_c2st_skips_thin_slices_without_crashing():
    rng = np.random.default_rng(10)
    real_X = rng.normal(size=(300, 4))
    synth_X = rng.normal(loc=0.4, size=(300, 4))
    real_slice = ["big"] * 250 + ["thin"] * 50
    synth_slice = ["big"] * 250 + ["thin"] * 50
    out = sliced_c2st(real_X, synth_X, real_slice, synth_slice, [f"f{i}" for i in range(4)])
    by = {d["slice"]: d for d in out}
    assert by["thin"]["skipped"] is True
    assert "fewer than" in by["thin"]["reason"]
    assert by["big"]["skipped"] is False
    assert by["big"]["c2st_auc"] is not None


# --------------------------------------------------------------------------
# evidence.thresholds
# --------------------------------------------------------------------------
@test
def test_threshold_pinning_and_precision():
    scores = np.linspace(0.0, 1.0, 1001)
    t = pin_threshold_at_fpr(scores, 0.01)
    assert rate_at_or_above(scores, t) <= 0.011, rate_at_or_above(scores, t)
    for bad in (0.0, 1.0, -0.1):
        try:
            pin_threshold_at_fpr(scores, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target_fpr={bad} must be rejected")
    # precision must fall as the base rate falls, at fixed recall and FPR
    p_high = precision_at_prevalence(0.9, 0.01, 0.50)
    p_low = precision_at_prevalence(0.9, 0.01, 0.013)
    assert p_high > p_low, (p_high, p_low)
    assert abs(precision_at_prevalence(1.0, 0.0, 0.01) - 1.0) < 1e-12


@test
def test_bootstrap_ci_brackets_the_mean():
    vals = [0.70, 0.72, 0.75, 0.71, 0.74]
    ci = bootstrap_mean_ci(vals, n_resamples=500, seed=1)
    assert ci["n"] == 5
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert bootstrap_mean_ci([])["mean"] is None
    one = bootstrap_mean_ci([0.5])
    assert one["mean"] == 0.5 and one["lo"] is None
    # None entries must be dropped, not crash
    assert bootstrap_mean_ci([0.5, None, 0.7])["n"] == 2


# --------------------------------------------------------------------------
# evidence.actions
# --------------------------------------------------------------------------
@test
def test_action_policy_ordering_and_app_rule():
    kw = dict(t_step_up=0.3, t_cooling=0.6, t_decline=0.9)
    assert choose_action(0.1, False, **kw) == APPROVE
    assert choose_action(0.4, False, **kw) == STEP_UP
    assert choose_action(0.7, False, **kw) == COOLING_OFF
    assert choose_action(0.95, False, **kw) == DECLINE
    # the carve-out: an APP candidate is never hard-declined on score alone
    assert choose_action(0.95, True, **kw) == COOLING_OFF
    # ...unless the carve-out is switched off, which is how the baseline is run
    assert choose_action(0.95, True, app_carve_out=False, **kw) == DECLINE
    m = ActionCostModel()
    # a decline stops an unauthorised transaction dead but not an APP scam
    assert m.fraud_loss(DECLINE, is_app=False) == 0.0
    assert m.fraud_loss(DECLINE, is_app=True) > 0.0
    assert m.friction_cost_on_legit(APPROVE) == 0.0
    assert m.friction_cost_on_legit(DECLINE) > m.friction_cost_on_legit(COOLING_OFF)
    assert m.friction_cost_on_legit(COOLING_OFF) > m.friction_cost_on_legit(STEP_UP)


@test
def test_evaluate_and_sweep_policies():
    rng = np.random.default_rng(4)
    n = 1200
    labels = (rng.uniform(size=n) < 0.08).astype(int)
    is_app = labels.astype(bool) & (rng.uniform(size=n) < 0.4)
    scores = np.clip(rng.beta(2, 8, size=n) + labels * 0.45, 0, 1)
    rep = evaluate_policy(scores, labels, is_app, 0.3, 0.5, 0.9)
    assert sum(rep["action_counts"].values()) == n
    assert rep["app_carve_out"] is True
    assert 0.0 <= rep["legit_friction_rate"] <= 1.0
    assert rep["total_cost_inr"] == round(
        rep["fraud_cost_inr"] + rep["friction_and_insult_cost_inr"], 2
    )
    assert (
        round(rep["app_scam_cost_inr"] + rep["non_app_fraud_cost_inr"], 2)
        == rep["fraud_cost_inr"]
    )
    sweep = sweep_policies(
        scores, labels, is_app, grid=(0.3, 0.5, 0.7), two_action_baseline_threshold=0.5
    )
    # combinations_with_replacement over 3 candidates -> C(3+2,3) = 10 triples
    assert sweep["n_policies_evaluated"] == 10, sweep["n_policies_evaluated"]
    assert sweep["baseline_is_reachable_by_this_family"] is True
    # the optimum can never be worse than a member of its own family
    assert (
        sweep["best_policy"]["total_cost_inr"]
        <= sweep["two_action_baseline_with_app_carve_out"]["total_cost_inr"]
    )
    assert sweep["saving_vs_two_action_with_carve_out_inr"] >= 0
    assert sweep["two_action_baseline"]["app_carve_out"] is False
    assert sweep["two_action_baseline"]["action_counts"][STEP_UP] == 0
    assert len(sweep["frontier"]) <= 8


# --------------------------------------------------------------------------
# evidence.privacy
# --------------------------------------------------------------------------
@test
def test_privacy_primitives():
    A = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    assert np.allclose(nearest_distances(A, A), [0.0, 0.0, 0.0])
    d = nearest_distances(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
    assert abs(float(d[0]) - 5.0) < 1e-12
    assert duplicate_share(A, A) == 1.0
    assert duplicate_share(np.array([[9.0, 9.0]]), A) == 0.0
    assert duplicate_share(np.empty((0, 2)), A) is None
    rng = np.random.default_rng(6)
    members = rng.normal(size=(80, 4))
    non_members = rng.normal(size=(80, 4))
    # synthetic drawn from the same law as both: no membership signal
    auc = membership_inference_auc(members, non_members, rng.normal(size=(300, 4)))
    assert 0.35 < auc < 0.65, auc
    # synthetic that IS the members: strong membership signal
    leaky = membership_inference_auc(members, non_members, members.copy())
    assert leaky > auc, (leaky, auc)


@test
def test_attribute_inference_reports_lift_over_baseline():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    rep = attribute_inference(X[:300], y[:300], X[300:], y[300:], seed=0)
    assert rep["accuracy"] > 0.75, rep
    assert 0.0 <= rep["majority_baseline"] <= 1.0
    assert abs(rep["lift"] - (rep["accuracy"] - rep["majority_baseline"])) < 1e-9
    # a single-class synthetic target cannot be trained on: report None, not crash
    degenerate = attribute_inference(X[:300], np.zeros(300, dtype=int), X[300:], y[300:])
    assert degenerate["accuracy"] is None and degenerate["lift"] is None


@test
def test_privacy_audit_flags_a_memorising_generator():
    rng = np.random.default_rng(8)
    members = rng.normal(size=(90, 5))
    non_members = rng.normal(size=(90, 5))
    honest = rng.normal(size=(400, 5))
    memoriser = np.repeat(members, 4, axis=0)
    good = privacy_audit("honest", honest, members, non_members)
    bad = privacy_audit("memoriser", memoriser, members, non_members)
    assert good["exact_duplicate_share"] == 0.0
    assert bad["exact_duplicate_share"] == 1.0
    assert bad["identical_row_share"] == 1.0
    assert bad["distance_to_closest_real_median"] == 0.0
    assert good["distance_to_closest_real_median"] > 0.0
    assert good["risk"]["duplication"] == "low"
    assert bad["risk"]["duplication"] == "high"
    assert bad["membership_inference_auc"] > good["membership_inference_auc"]
    assert "differential privacy" in bad["boundary"]


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{len(PASSED)} tests passed in {time.time() - t0:.3f} s")
