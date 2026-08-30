"""No-key attacker feedback-adaptation tests."""

import json
from types import SimpleNamespace

from agents.feedback_adapter import FeedbackAwareOfflineAdapter


class _Base:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            constraints=SimpleNamespace(amount_band=(100.0, 200.0))
        )

    def create(self, **kwargs):
        del kwargs
        move = {
            "reasoning": "baseline move",
            "payload": {"amount": 150.0},
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(move)))]
        )


def _move(response) -> dict:
    return json.loads(response.choices[0].message.content)


def test_offline_adapter_reduces_ticket_after_friction():
    adapter = FeedbackAwareOfflineAdapter(_Base())
    response = adapter.create(
        response_format={"type": "json_schema"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Defender outcome for transaction 2: STEP_UP. "
                    "Observable signal family: behavioural velocity. Adapt the next synthetic move."
                ),
            }
        ],
    )
    move = _move(response)
    assert move["payload"]["amount"] == 135.0
    assert "behavioural velocity" in move["reasoning"]
    assert "reduced ticket size" in move["reasoning"]


def test_offline_adapter_reinforces_approved_strategy_within_band():
    adapter = FeedbackAwareOfflineAdapter(_Base())
    response = adapter.create(
        response_format={"type": "json_schema"},
        messages=[
            {
                "role": "user",
                "content": "Defender outcome for transaction 2: APPROVE. Adapt the next synthetic move.",
            }
        ],
    )
    move = _move(response)
    assert move["payload"]["amount"] == 157.5
    assert "reinforced" in move["reasoning"]


def test_offline_adapter_does_not_mutate_without_feedback():
    adapter = FeedbackAwareOfflineAdapter(_Base())
    response = adapter.create(
        response_format={"type": "json_schema"},
        messages=[{"role": "user", "content": "Generate a payment."}],
    )
    move = _move(response)
    assert move["payload"]["amount"] == 150.0
    assert move["reasoning"] == "baseline move"
