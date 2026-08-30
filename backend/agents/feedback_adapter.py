"""Feedback-aware adapter for the deterministic no-key attacker.

The real LLM attacker already reads defender feedback because ``pump_campaign``
appends it to the normal conversation. The deterministic CI/demo client does not
interpret messages, so this wrapper gives it two missing capabilities:

* vector-specific synthetic behaviour for every mapped AttackSpec; and
* coarse adaptation to the previous Blue-Team outcome.

All mutations remain inside the active AttackSpec and PaymentMessage schema.
"""

from __future__ import annotations

import json

from agents.synthetic_vector_policy import SyntheticVectorPolicy


class FeedbackAwareOfflineAdapter:
    """Wrap the private deterministic attacker protocol without importing it."""

    def __init__(self, base: object) -> None:
        self.base = base
        self.policy: SyntheticVectorPolicy | None = None
        spec = getattr(base, "spec", None)
        env = getattr(base, "env", None)
        rng = getattr(base, "rng", None)
        if spec is not None and env is not None and rng is not None:
            try:
                candidate = SyntheticVectorPolicy(spec, env, rng)
                if candidate.mapped:
                    self.policy = candidate
            except (AttributeError, KeyError, TypeError, ValueError):
                # Keep the adapter compatible with minimal test doubles and any
                # future provider-like fallback that lacks the arena internals.
                self.policy = None

    @staticmethod
    def _latest_feedback(messages: list[dict]) -> str | None:
        for message in reversed(messages):
            text = str(message.get("content", ""))
            if message.get("role") == "user" and text.startswith("Defender outcome for transaction"):
                return text
        return None

    def create(self, **kwargs: object) -> object:
        response = self.base.create(**kwargs)
        if kwargs.get("response_format") is None:
            return response

        message = response.choices[0].message
        raw = getattr(message, "content", None)
        if not raw:
            return response

        try:
            move = json.loads(raw)
            payload = move["payload"]

            if self.policy is not None:
                # The private baseline deliberately emits one incoherent MCC on
                # the first structured move so PlausibilityGate coaching is
                # exercised. Preserve that single fixture while making every
                # other move vector-specific.
                preserve_gate_fault = getattr(self.base, "calls", None) == 2
                self.policy.apply(move, preserve_gate_fault=preserve_gate_fault)
                payload = move["payload"]

            messages = list(kwargs.get("messages") or [])
            feedback = self._latest_feedback(messages)
            if feedback:
                amount = float(payload["amount"])
                constraints = self.base.spec.constraints
                lo, hi = constraints.amount_band

                # Reinforce an approved strategy slightly; after friction, lower
                # the ticket while remaining inside the same declared economic
                # envelope. Cross-family adaptation is handled separately by the
                # CampaignStrategist.
                if ": APPROVE." in feedback:
                    adjusted = min(float(hi), amount * 1.05)
                    action = "reinforced the previously accepted ticket size"
                else:
                    adjusted = max(float(lo), amount * 0.90)
                    action = "reduced ticket size within the declared attack band"
                payload["amount"] = round(adjusted, 2)

                signal = "defender outcome"
                marker = "Observable signal family:"
                if marker in feedback:
                    signal = feedback.split(marker, 1)[1].split(".", 1)[0].strip()
                move["reasoning"] = (
                    f"Adaptive replay after {signal}: {action}. "
                    + str(move.get("reasoning", ""))
                )

            message.content = json.dumps(move)
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # The fallback's normal Pydantic parser remains the final authority;
            # if provider-like response shape changes, fail open to the original
            # deterministic move rather than breaking the demo path.
            return response
        return response


def make_feedback_aware_if_offline(agent: object) -> bool:
    """Wrap an AttackerAgent's deterministic private client exactly once."""
    client = getattr(agent, "client", None)
    if client is None or isinstance(client, FeedbackAwareOfflineAdapter):
        return isinstance(client, FeedbackAwareOfflineAdapter)
    if client.__class__.__name__ != "_OfflineHeuristicAttacker":
        return False
    agent.client = FeedbackAwareOfflineAdapter(client)
    return True


__all__ = ["FeedbackAwareOfflineAdapter", "make_feedback_aware_if_offline"]
