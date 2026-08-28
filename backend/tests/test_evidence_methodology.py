import numpy as np
import pandas as pd

from evidence.actions import evaluate_policy, sweep_policies
from evidence.calibration import fpr_at_threshold, pin_threshold_at_fpr as legacy_pin
from evidence.thresholds import pin_threshold_at_fpr, rate_at_or_above
from fidelity.behavior import build_features
from fidelity.fixtures import _within_device_gap_pool


def _row(customer, device, ts):
    return {
        "customer": customer,
        "device": device,
        "ip": f"IP_{customer}",
        "merchant": "M",
        "mcc": 5411,
        "amount": 100.0,
        "ts": float(ts),
        "entry_mode": "CHIP",
        "label": 0,
        "attack_id": "LEGIT",
    }


def test_tied_scores_never_exceed_target_fpr():
    scores = np.asarray([0.1] * 95 + [0.9] * 5)
    tau = pin_threshold_at_fpr(scores, 0.01)
    assert rate_at_or_above(scores, tau) <= 0.01
    legacy_tau = legacy_pin(scores, 0.01)
    assert fpr_at_threshold(scores, legacy_tau) <= 0.01


def test_causal_entity_counts_do_not_see_future_edges():
    base = pd.DataFrame([_row("C1", "D", 1), _row("C2", "D", 2)])
    extended = pd.concat([base, pd.DataFrame([_row("C3", "D", 3)])], ignore_index=True)
    first = build_features(base)
    second = build_features(extended)
    assert first.loc[0, "device_customer_count"] == 1
    assert first.loc[1, "device_customer_count"] == 2
    assert second.loc[0, "device_customer_count"] == first.loc[0, "device_customer_count"]
    assert second.loc[1, "device_customer_count"] == first.loc[1, "device_customer_count"]


def test_app_truth_is_not_used_as_policy_candidate():
    result = evaluate_policy(
        scores=[0.9],
        labels=[1],
        app_candidates=[False],
        app_truth=[True],
        t_step_up=0.5,
        t_cooling=0.5,
        t_decline=0.5,
        app_carve_out=True,
    )
    assert result["action_counts"]["DECLINE"] == 1
    assert result["action_counts"]["COOLING_OFF"] == 0


def test_policy_family_contains_true_two_action_baseline():
    result = sweep_policies(
        scores=[0.1, 0.9],
        labels=[0, 1],
        app_candidates=[False, True],
        app_truth=[False, True],
        grid=(0.5,),
        two_action_baseline_threshold=0.5,
    )
    assert result["baseline_is_reachable_by_this_family"] is True
    assert result["saving_vs_two_action_inr"] >= 0.0


def test_behavioural_gap_pool_never_crosses_devices():
    source = pd.DataFrame(
        [
            _row("C1", "D1", 0),
            _row("C1", "D1", 10),
            _row("C2", "D2", 10_000),
            _row("C2", "D2", 10_010),
        ]
    )
    gaps = _within_device_gap_pool(source)
    assert set(gaps.tolist()) == {10.0}
