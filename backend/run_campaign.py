"""
CLI campaign harness — run the full adversarial loop WITHOUT the frontend.

Single-vector mode (unchanged):
    python run_campaign.py --attack attack_1 --size 50

Multi-vector portfolio mode:
    python run_campaign.py --attack attack_4 --size 40 --portfolio --fast

Every event is printed as JSONL. Portfolio mode keeps the existing attack ->
defend -> feedback loop but lets a campaign-level strategist continue, mutate,
pivot among connected AttackSpecs, or abandon based on Blue-Team outcomes.

Uses OPENROUTER_API_KEY when present; otherwise falls back to the deterministic
no-key attacker. --fast zeroes the 2s provider pacing (offline/testing only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.attacker import RATE_LIMIT_SLEEP_S, AttackerAgent  # noqa: E402
from agents.fraud_portfolio import portfolio_snapshot  # noqa: E402
from agents.portfolio_runner import pump_portfolio_campaign  # noqa: E402
from main import ArenaStack, pump_campaign, resolve_attack_path  # noqa: E402
from schemas.attack import load_attack_spec  # noqa: E402


async def _run(
    attack: str,
    size: int,
    sleep_s: float,
    *,
    portfolio: bool,
    segment_size: int,
    max_vectors: int,
    feedback_mode: str,
) -> None:
    stack = ArenaStack()
    path = resolve_attack_path(attack)
    spec = load_attack_spec(path)
    print(json.dumps({
        "type": "campaign_accepted",
        "spec": spec.spec_id,
        "size": size,
        "attack_file": path.name,
        "portfolio_mode": portfolio,
        "feedback_mode": feedback_mode,
        "models": {
            "xgb": stack.engine.scorer.model_source,
            "iforest": stack.engine.novelty.model_source,
        },
    }), flush=True)

    if portfolio:
        snapshot = portfolio_snapshot()
        print(json.dumps({
            "type": "red_team_portfolio",
            "data": {
                "model": snapshot["model"],
                "dimensions": snapshot["dimensions"],
                "vector_count": snapshot["vector_count"],
            },
        }), flush=True)

    async def emit(evt: dict) -> None:
        try:
            print(json.dumps(evt, default=str), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"type": "error", "data": f"emit failed: {exc}"}), flush=True)

    if portfolio:
        await pump_portfolio_campaign(
            stack,
            spec,
            size,
            emit,
            single_pump=pump_campaign,
            specs_dir=BACKEND_ROOT / "attack_specs",
            feedback_mode=feedback_mode,
            sleep_s=sleep_s,
            segment_size=segment_size,
            max_vectors=max_vectors,
        )
    else:
        agent = AttackerAgent(spec, stack.env, sleep_between_calls_s=sleep_s)
        await pump_campaign(
            stack,
            agent,
            size,
            emit,
            feedback_mode=feedback_mode,
        )

    print(json.dumps({"type": "final_costs", **stack.cost_summary()}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an attacker campaign against the defense stack")
    ap.add_argument("--attack", default="attack_1", help="initial attack id / yaml stem")
    ap.add_argument("--size", type=int, default=50, help="campaign size in transaction slots")
    ap.add_argument("--fast", action="store_true", help="zero the LLM rate-limit sleep")
    ap.add_argument(
        "--feedback-mode",
        choices=("black", "gray", "white"),
        default="gray",
        help="defender information exposed back to the synthetic attacker",
    )
    ap.add_argument(
        "--portfolio",
        action="store_true",
        help="enable the multidimensional campaign strategist and cross-vector pivots",
    )
    ap.add_argument(
        "--segment-size",
        type=int,
        default=4,
        help="transaction slots per vector before strategy is reconsidered",
    )
    ap.add_argument(
        "--max-vectors",
        type=int,
        default=5,
        help="maximum distinct vectors a portfolio campaign may activate",
    )
    args = ap.parse_args()

    sleep_s = 0.0 if args.fast else RATE_LIMIT_SLEEP_S
    asyncio.run(_run(
        args.attack,
        max(1, args.size),
        sleep_s,
        portfolio=bool(args.portfolio),
        segment_size=max(1, args.segment_size),
        max_vectors=max(1, args.max_vectors),
        feedback_mode=args.feedback_mode,
    ))


if __name__ == "__main__":
    main()
