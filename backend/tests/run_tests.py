"""Small dependency-light regression runner for the evidence methodology fixes."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from tests.test_evidence_methodology import (
    test_app_truth_is_not_used_as_policy_candidate,
    test_behavioural_gap_pool_never_crosses_devices,
    test_causal_entity_counts_do_not_see_future_edges,
    test_policy_family_contains_true_two_action_baseline,
    test_tied_scores_never_exceed_target_fpr,
)


def main() -> None:
    tests = [
        test_tied_scores_never_exceed_target_fpr,
        test_causal_entity_counts_do_not_see_future_edges,
        test_app_truth_is_not_used_as_policy_candidate,
        test_policy_family_contains_true_two_action_baseline,
        test_behavioural_gap_pool_never_crosses_devices,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} methodology regression tests passed")


if __name__ == "__main__":
    main()
