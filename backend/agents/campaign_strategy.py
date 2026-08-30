"""Campaign-level strategy controller for multidimensional red-team evaluation.

The controller does not invent payment fields or bypass the AttackSpec. It only
chooses whether the synthetic attacker should continue the current family,
mutate within its declared envelope, pivot to a connected family, or abandon a
campaign whose simulated economics have deteriorated.

The purpose is defensive evaluation: force the Blue Team to face an adversary
that can change *families* instead of replaying one script forever.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from agents.fraud_portfolio import PORTFOLIO, AttackVectorProfile, profile_for_spec


class StrategyAction(str, Enum):
    CONTINUE = "continue"
    MUTATE = "mutate"
    PIVOT = "pivot"
    ABANDON = "abandon"


@dataclass(slots=True)
class VectorStats:
    attempts: int = 0
    approvals: int = 0
    stepups: int = 0
    declines: int = 0
    manual_reviews: int = 0
    approved_value: float = 0.0
    blocked_value: float = 0.0
    signal_families: Counter[str] = field(default_factory=Counter)

    @property
    def friction_count(self) -> int:
        return self.stepups + self.declines + self.manual_reviews

    @property
    def approval_rate(self) -> float:
        return self.approvals / max(self.attempts, 1)

    @property
    def friction_rate(self) -> float:
        return self.friction_count / max(self.attempts, 1)

    @property
    def utility_score(self) -> float:
        """Dimensionless synthetic utility used only for portfolio exploration."""
        if self.attempts == 0:
            return 0.0
        outcome_score = (
            1.0 * self.approvals
            - 0.45 * self.stepups
            - 1.00 * self.declines
            - 0.70 * self.manual_reviews
        ) / self.attempts
        value_term = self.approved_value / max(self.approved_value + self.blocked_value, 1.0)
        return round(0.75 * outcome_score + 0.25 * value_term, 4)

    def to_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "approvals": self.approvals,
            "stepups": self.stepups,
            "declines": self.declines,
            "manual_reviews": self.manual_reviews,
            "approval_rate": round(self.approval_rate, 4),
            "friction_rate": round(self.friction_rate, 4),
            "approved_value": round(self.approved_value, 2),
            "blocked_value": round(self.blocked_value, 2),
            "utility_score": self.utility_score,
            "signal_families": dict(self.signal_families),
        }


@dataclass(slots=True)
class StrategyDecision:
    action: StrategyAction
    current_spec_id: str
    next_spec_id: str | None
    reason: str
    mutation_axes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "current_spec_id": self.current_spec_id,
            "next_spec_id": self.next_spec_id,
            "reason": self.reason,
            "mutation_axes": list(self.mutation_axes),
        }


@dataclass(slots=True)
class CampaignMemory:
    initial_spec_id: str
    current_spec_id: str
    max_vectors: int = 5
    total_attempts: int = 0
    total_approved_value: float = 0.0
    consecutive_approvals: int = 0
    consecutive_friction: int = 0
    visited: list[str] = field(default_factory=list)
    vector_stats: dict[str, VectorStats] = field(default_factory=dict)
    transitions: list[dict] = field(default_factory=list)
    latest_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.initial_spec_id not in self.visited:
            self.visited.append(self.initial_spec_id)
        self.vector_stats.setdefault(self.initial_spec_id, VectorStats())

    def stats_for(self, spec_id: str) -> VectorStats:
        return self.vector_stats.setdefault(spec_id, VectorStats())

    @property
    def unique_vectors(self) -> int:
        return len(set(self.visited))

    def observe(
        self,
        spec_id: str,
        decision: str,
        amount: float,
        signal_families: list[str] | tuple[str, ...] = (),
    ) -> None:
        stats = self.stats_for(spec_id)
        stats.attempts += 1
        self.total_attempts += 1
        self.latest_signals = tuple(signal_families)
        for signal in signal_families:
            stats.signal_families[str(signal)] += 1

        decision = str(decision).upper()
        if decision == "APPROVE":
            stats.approvals += 1
            stats.approved_value += float(amount)
            self.total_approved_value += float(amount)
            self.consecutive_approvals += 1
            self.consecutive_friction = 0
        else:
            stats.blocked_value += float(amount)
            self.consecutive_approvals = 0
            self.consecutive_friction += 1
            if decision == "DECLINE":
                stats.declines += 1
            elif decision == "MANUAL_REVIEW":
                stats.manual_reviews += 1
            else:
                stats.stepups += 1

    def transition(self, decision: StrategyDecision) -> None:
        row = decision.to_dict()
        row["after_attempt"] = self.total_attempts
        self.transitions.append(row)
        if decision.next_spec_id and decision.next_spec_id != self.current_spec_id:
            self.current_spec_id = decision.next_spec_id
            if decision.next_spec_id not in self.visited:
                self.visited.append(decision.next_spec_id)
            self.vector_stats.setdefault(decision.next_spec_id, VectorStats())
            self.consecutive_approvals = 0
            self.consecutive_friction = 0

    def snapshot(self) -> dict:
        return {
            "initial_spec_id": self.initial_spec_id,
            "current_spec_id": self.current_spec_id,
            "total_attempts": self.total_attempts,
            "total_approved_value": round(self.total_approved_value, 2),
            "consecutive_approvals": self.consecutive_approvals,
            "consecutive_friction": self.consecutive_friction,
            "visited": list(self.visited),
            "unique_vectors": self.unique_vectors,
            "latest_signals": list(self.latest_signals),
            "vector_stats": {
                spec_id: stats.to_dict()
                for spec_id, stats in sorted(self.vector_stats.items())
            },
            "transitions": list(self.transitions),
        }


class CampaignStrategist:
    """Deterministic portfolio explorer driven by Blue-Team outcomes."""

    def __init__(
        self,
        initial_spec_id: str,
        *,
        max_vectors: int = 5,
        pivot_after_friction: int = 3,
        abandon_after_friction: int = 8,
    ) -> None:
        if initial_spec_id not in PORTFOLIO:
            raise KeyError(f"unknown portfolio vector {initial_spec_id!r}")
        if max_vectors < 1:
            raise ValueError("max_vectors must be >= 1")
        self.memory = CampaignMemory(
            initial_spec_id=initial_spec_id,
            current_spec_id=initial_spec_id,
            max_vectors=max_vectors,
        )
        self.pivot_after_friction = max(2, int(pivot_after_friction))
        self.abandon_after_friction = max(
            self.pivot_after_friction + 1, int(abandon_after_friction)
        )

    def observe(
        self,
        spec_id: str,
        record: dict,
        *,
        signal_families: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.memory.observe(
            spec_id,
            str(record.get("decision", "STEP_UP")),
            float(record.get("amount", 0.0)),
            signal_families,
        )

    def _candidate_score(
        self,
        current: AttackVectorProfile,
        candidate: AttackVectorProfile,
    ) -> tuple[float, str]:
        visited = candidate.spec_id in self.memory.visited
        stats = self.memory.vector_stats.get(candidate.spec_id)
        prior_utility = stats.utility_score if stats and stats.attempts else 0.0
        diversity = current.genome.distance(candidate.genome)
        novelty_bonus = 1.25 if not visited else 0.0
        # Prefer a connected vector that changes several fraud dimensions. This
        # models campaign exploration without encoding Blue-Team thresholds or
        # real-world bypass recipes.
        score = novelty_bonus + diversity + 0.35 * prior_utility
        return score, candidate.spec_id

    def _best_transition(self, current: AttackVectorProfile) -> str | None:
        candidates = [PORTFOLIO[target] for target in current.transitions]
        if not candidates:
            return None
        ranked = sorted(
            (self._candidate_score(current, candidate) for candidate in candidates),
            key=lambda row: (-row[0], row[1]),
        )
        return ranked[0][1]

    def decide(self) -> StrategyDecision:
        current_id = self.memory.current_spec_id
        current = profile_for_spec(current_id)
        stats = self.memory.stats_for(current_id)

        if self.memory.consecutive_friction >= self.abandon_after_friction:
            return StrategyDecision(
                StrategyAction.ABANDON,
                current_id,
                None,
                "sustained friction made the synthetic campaign uneconomic",
            )

        if (
            self.memory.consecutive_approvals >= 2
            and stats.attempts >= 2
            and stats.approval_rate >= 0.60
        ):
            return StrategyDecision(
                StrategyAction.CONTINUE,
                current_id,
                current_id,
                "the current vector is still producing approvals",
            )

        pivot_pressure = (
            self.memory.consecutive_friction >= self.pivot_after_friction
            or (stats.attempts >= 4 and stats.friction_rate >= 0.75)
        )
        if pivot_pressure:
            if self.memory.unique_vectors >= self.memory.max_vectors:
                return StrategyDecision(
                    StrategyAction.ABANDON,
                    current_id,
                    None,
                    "the configured portfolio exploration budget is exhausted",
                )
            next_id = self._best_transition(current)
            if next_id is None or next_id == current_id:
                return StrategyDecision(
                    StrategyAction.ABANDON,
                    current_id,
                    None,
                    "no connected synthetic vector remains to explore",
                )
            return StrategyDecision(
                StrategyAction.PIVOT,
                current_id,
                next_id,
                "repeated Blue-Team friction triggered a connected-vector pivot",
            )

        if self.memory.consecutive_friction > 0:
            return StrategyDecision(
                StrategyAction.MUTATE,
                current_id,
                current_id,
                "friction increased, so the campaign will vary one declared dimension before pivoting",
                mutation_axes=current.mutation_axes[:2],
            )

        return StrategyDecision(
            StrategyAction.CONTINUE,
            current_id,
            current_id,
            "insufficient evidence to change vectors",
        )

    def apply(self, decision: StrategyDecision) -> StrategyDecision:
        self.memory.transition(decision)
        return decision

    def memory_prompt(self) -> str:
        """Compact, coarse campaign memory suitable for a newly activated LLM vector."""
        rows = []
        for spec_id in self.memory.visited:
            stats = self.memory.vector_stats.get(spec_id, VectorStats())
            rows.append(
                f"{spec_id}: attempts={stats.attempts}, approvals={stats.approvals}, "
                f"friction={stats.friction_count}, utility={stats.utility_score:.3f}"
            )
        return (
            "Synthetic campaign memory from prior vectors (coarse outcomes only): "
            + "; ".join(rows)
            + ". Continue strictly inside the newly selected AttackSpec and PaymentMessage schema."
        )

    def mutation_prompt(self, decision: StrategyDecision) -> str:
        axes = ", ".join(decision.mutation_axes) or "one declared campaign dimension"
        return (
            "Synthetic campaign controller: keep the current attack family, but vary "
            f"{axes} within the AttackSpec envelope. Do not invent new fields or leave "
            "the declared rail, resource, merchant verticals, or amount band."
        )

    def snapshot(self) -> dict:
        return self.memory.snapshot()


__all__ = [
    "StrategyAction",
    "VectorStats",
    "StrategyDecision",
    "CampaignMemory",
    "CampaignStrategist",
]
