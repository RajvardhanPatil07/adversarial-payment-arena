"""
CLI campaign harness — run the full adversarial loop WITHOUT the frontend.

    python run_campaign.py --attack attack_1 --size 50
    python run_campaign.py --attack attack_2_synthetic_mule_ring --size 25 --fast

Every event (agent thoughts, payloads, gate verdicts, defense decisions, cost
updates, graph deltas) is printed to stdout as one JSON object per line —
pipe it into jq, tee it to a file, or just watch the fight in the terminal.

Uses OPENROUTER_API_KEY when present (stealth/ox-alpha); otherwise falls
back to the deterministic offline fraudster. --fast zeroes the 2s rate-limit
sleep (offline/testing only).
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
from main import ArenaStack, pump_campaign, resolve_attack_path  # noqa: E402
from schemas.attack import load_attack_spec  # noqa: E402


async def _run(attack: str, size: int, sleep_s: float) -> None:
    stack = ArenaStack()
    path = resolve_attack_path(attack)
    spec = load_attack_spec(path)
    print(json.dumps({
        "type": "campaign_accepted", "spec": spec.spec_id,
        "size": size, "attack_file": path.name,
        "models": {"xgb": stack.engine.scorer.model_source,
                   "iforest": stack.engine.novelty.model_source},
    }), flush=True)

    async def emit(evt: dict) -> None:  # pump awaits its sink
        # Never let a serialization hiccup kill the campaign stream
        try:
            print(json.dumps(evt, default=str), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"type": "error", "data": f"emit failed: {exc}"}), flush=True)

    agent = AttackerAgent(spec, stack.env, sleep_between_calls_s=sleep_s)
    await pump_campaign(stack, agent, size, emit)
    print(json.dumps({"type": "final_costs", **stack.cost_summary()}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an attacker campaign against the defense stack")
    ap.add_argument("--attack", default="attack_1", help="attack id / yaml stem (e.g. attack_1)")
    ap.add_argument("--size", type=int, default=50, help="campaign size in transactions")
    ap.add_argument("--fast", action="store_true", help="zero the LLM rate-limit sleep")
    args = ap.parse_args()

    sleep_s = 0.0 if args.fast else RATE_LIMIT_SLEEP_S
    asyncio.run(_run(args.attack, max(1, args.size), sleep_s))


if __name__ == "__main__":
    main()
