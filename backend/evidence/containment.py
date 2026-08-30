"""Live campaign containment metrics.

Traditional TPR/FPR says whether a detector was right per transaction. During an
adaptive attack campaign, judges and risk teams also care about *how long the
system remained exposed*. This tracker measures that without changing decision
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CampaignContainment:
    spec_id: str
    transactions: int = 0
    total_attack_value: float = 0.0
    approved_value: float = 0.0
    approved_count: int = 0
    friction_count: int = 0
    decline_count: int = 0
    first_friction_txn: int | None = None
    first_decline_txn: int | None = None
    first_emerging_threat_txn: int | None = None
    first_emerging_threat_id: str | None = None
    value_before_first_friction: float = 0.0
    value_before_first_decline: float = 0.0

    def observe(self, record: dict, txn_index: int | None = None) -> None:
        """Fold one attack-campaign decision into exposure counters."""
        index = int(txn_index or (self.transactions + 1))
        amount = float(record.get("amount", 0.0))
        decision = str(record.get("decision", "APPROVE"))

        self.transactions += 1
        self.total_attack_value += amount

        if self.first_friction_txn is None:
            if decision == "APPROVE":
                self.value_before_first_friction += amount
            else:
                self.first_friction_txn = index

        if self.first_decline_txn is None:
            if decision != "DECLINE":
                self.value_before_first_decline += amount
            else:
                self.first_decline_txn = index

        if decision == "APPROVE":
            self.approved_count += 1
            self.approved_value += amount
        else:
            self.friction_count += 1
            if decision == "DECLINE":
                self.decline_count += 1

    def mark_emerging_threat(self, fingerprint: dict, txn_index: int | None = None) -> None:
        if self.first_emerging_threat_txn is not None:
            return
        self.first_emerging_threat_txn = int(txn_index or self.transactions)
        self.first_emerging_threat_id = str(fingerprint.get("threat_id", "UNKNOWN"))

    def summary(self) -> dict:
        n = max(self.transactions, 1)
        return {
            "spec_id": self.spec_id,
            "transactions_scored": self.transactions,
            "total_attack_value": round(self.total_attack_value, 2),
            "approved_escapes": self.approved_count,
            "escape_rate": round(self.approved_count / n, 4),
            "approved_escape_value": round(self.approved_value, 2),
            "friction_rate": round(self.friction_count / n, 4),
            "decline_rate": round(self.decline_count / n, 4),
            "transactions_to_first_friction": self.first_friction_txn,
            "transactions_to_first_decline": self.first_decline_txn,
            "transactions_to_emerging_threat": self.first_emerging_threat_txn,
            "first_emerging_threat_id": self.first_emerging_threat_id,
            "fraud_value_before_first_friction": round(self.value_before_first_friction, 2),
            "fraud_value_before_first_decline": round(self.value_before_first_decline, 2),
            "contained_fraction": round(1.0 - self.approved_count / n, 4),
        }


__all__ = ["CampaignContainment"]
