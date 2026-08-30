"""End-to-end multi-vector campaign orchestration.

This runner deliberately depends on the existing single-vector ``pump_campaign``
through dependency injection instead of reimplementing defense logic. Each
segment therefore travels through the exact same Plausibility Gate, Blue-Team
models, threat miner, graph, containment and feedback path as a normal campaign.

The only new responsibility here is campaign-level strategy: keep using a
vector, mutate within it, pivot to a connected vector, or stop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from agents.attacker import AttackerAgent
from agents.campaign_strategy import CampaignStrategist, StrategyAction
from agents.feedback_adapter import make_feedback_aware_if_offline
from agents.fraud_portfolio import profile_for_spec
from schemas.attack import AttackSpec, load_attack_spec

Emit = Callable[[dict], Awaitable[None]]
SinglePump = Callable[..., Awaitable[None]]


def _signals_from_reasons(reasons) -> list[str]:
    families: list[str] = []
    for reason in reasons or ():
        text = str(reason)
        if text.startswith("ring_detected"):
            families.append("shared-infrastructure topology")
        elif text.startswith("velocity>"):
            families.append("behavioural velocity")
        elif "novelty" in text:
            families.append("out-of-distribution behaviour")
    return families or ["no dominant defense signal"]


def _agent_for(
    agents: dict[str, AttackerAgent],
    spec: AttackSpec,
    stack,
    sleep_s: float,
    strategist: CampaignStrategist,
) -> AttackerAgent:
    agent = agents.get(spec.spec_id)
    if agent is not None:
        return agent
    agent = AttackerAgent(spec, stack.env, sleep_between_calls_s=sleep_s)
    make_feedback_aware_if_offline(agent)
    if strategist.memory.total_attempts:
        agent._user_say(strategist.memory_prompt())
    agents[spec.spec_id] = agent
    return agent


async def pump_portfolio_campaign(
    stack,
    initial_spec: AttackSpec,
    campaign_size: int,
    emit: Emit,
    *,
    single_pump: SinglePump,
    specs_dir: str | Path,
    feedback_mode: str = "gray",
    sleep_s: float = 0.0,
    segment_size: int = 4,
    max_vectors: int = 5,
) -> dict:
    """Run one continuous campaign that may cross existing AttackSpecs.

    ``campaign_size`` is a budget of transaction *slots*, matching the existing
    AttackerAgent contract. A segment stays within one AttackSpec; strategy is
    reconsidered only at segment boundaries so the Blue Team sees coherent
    behavior rather than random per-row family switching.
    """
    campaign_size = max(1, int(campaign_size))
    segment_size = max(1, min(int(segment_size), campaign_size))
    max_vectors = max(1, int(max_vectors))
    specs_dir = Path(specs_dir)

    strategist = CampaignStrategist(
        initial_spec.spec_id,
        max_vectors=max_vectors,
    )
    agents: dict[str, AttackerAgent] = {}
    current_spec = initial_spec
    remaining = campaign_size
    global_offset = 0
    segment_number = 0
    stopped_reason = "campaign_size_exhausted"

    await emit({
        "type": "portfolio_campaign_start",
        "data": {
            "initial_spec": initial_spec.spec_id,
            "campaign_size": campaign_size,
            "segment_size": segment_size,
            "max_vectors": max_vectors,
        },
    })

    while remaining > 0:
        segment_number += 1
        this_segment = min(segment_size, remaining)
        active_spec_id = current_spec.spec_id
        profile = profile_for_spec(active_spec_id)
        agent = _agent_for(agents, current_spec, stack, sleep_s, strategist)

        await emit({
            "type": "vector_segment_start",
            "data": {
                "segment": segment_number,
                "spec_id": active_spec_id,
                "attack_file": profile.attack_file,
                "genome": profile.to_dict()["genome"],
                "slots": this_segment,
                "global_slot_start": global_offset + 1,
            },
        })

        async def segment_emit(event: dict) -> None:
            transformed = dict(event)
            event_type = transformed.get("type")

            # A portfolio campaign owns the top-level lifecycle. Convert each
            # inner agent lifecycle into segment-scoped telemetry.
            if event_type == "campaign_start":
                return
            if event_type == "campaign_summary":
                await emit({
                    "type": "vector_segment_summary",
                    "vector_spec": active_spec_id,
                    "segment": segment_number,
                    "data": transformed.get("data", {}),
                })
                return
            if event_type == "containment_summary":
                await emit({
                    "type": "vector_containment_summary",
                    "vector_spec": active_spec_id,
                    "segment": segment_number,
                    "data": transformed.get("data", {}),
                })
                return

            local_index = transformed.get("txn_index")
            if isinstance(local_index, int):
                transformed["txn_index"] = global_offset + local_index
            transformed["vector_spec"] = active_spec_id
            transformed["segment"] = segment_number

            if event_type == "defense_decision":
                record = {
                    "decision": transformed.get("decision"),
                    "amount": transformed.get("amount", 0.0),
                }
                strategist.observe(
                    active_spec_id,
                    record,
                    signal_families=_signals_from_reasons(transformed.get("reasons")),
                )

            await emit(transformed)

        await single_pump(
            stack,
            agent,
            this_segment,
            segment_emit,
            feedback_mode=feedback_mode,
        )

        global_offset += this_segment
        remaining -= this_segment

        strategy_decision = strategist.decide()
        strategist.apply(strategy_decision)
        await emit({
            "type": "strategy_transition",
            "segment": segment_number,
            "data": strategy_decision.to_dict(),
        })

        if strategy_decision.action is StrategyAction.ABANDON:
            stopped_reason = strategy_decision.reason
            break

        if strategy_decision.action is StrategyAction.PIVOT:
            next_profile = profile_for_spec(strategy_decision.next_spec_id or active_spec_id)
            next_path = specs_dir / next_profile.attack_file
            current_spec = load_attack_spec(next_path)
            continue

        if strategy_decision.action is StrategyAction.MUTATE:
            agent._user_say(strategist.mutation_prompt(strategy_decision))

    snapshot = strategist.snapshot()
    result = {
        "portfolio_mode": True,
        "initial_spec": initial_spec.spec_id,
        "slots_budgeted": campaign_size,
        "slots_consumed": global_offset,
        "stopped_reason": stopped_reason,
        "strategy": snapshot,
    }
    await emit({"type": "portfolio_campaign_summary", "data": result})
    return result


__all__ = ["pump_portfolio_campaign"]
