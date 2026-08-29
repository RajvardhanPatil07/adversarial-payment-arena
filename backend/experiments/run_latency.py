"""
LATENCY -- measured, not estimated.

Why this exists
---------------
`docs/FEASIBILITY.md` previously carried a latency BUDGET: a table of
architectural estimates ("~5-15 ms for tree-ensemble scoring"). An estimate is
a design intention. In an authorisation path, latency is a hard constraint --
an inline risk score that misses its slot is not a slow score, it is no score --
so the number has to be measured on the actual code path.

This benchmark drives `DecisionEngine.decide()` -- the full four-layer stack,
exactly as the WebSocket server calls it -- over a realistic mixed stream and
reports the percentile distribution that an issuer would actually care about.

What is included in each measurement
------------------------------------
Everything on the blocking path:
  * feature assembly, including sliding-window velocity counters
  * supervised XGBoost scoring
  * unsupervised Isolation Forest novelty scoring
  * entity-graph ring check
  * threshold comparison, reason-code emission
  * observe-after-score state folding

What is NOT included, and why
-----------------------------
Network transport, ISO 8583/20022 parsing, and the issuer's own decision logic.
Those belong to the host, not to this component. Reporting them would inflate
the number with work this repository does not do.

Reproduce
---------
    python backend/experiments/run_latency.py
    make latency
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.corpus_builder import build_corpus  # noqa: E402
from defense.decision import DecisionEngine  # noqa: E402
from evidence.artifacts import ARTIFACTS_DIR, ClaimLedger, write_artifact  # noqa: E402

DOCS_DIR = BACKEND_ROOT.parent / "docs"
CHART_PATH = DOCS_DIR / "latency.png"

SEED = 11
N_LEGIT_TRAIN = 3000
N_LEGIT_MEASURE = 4000
WARMUP = 300
COMMAND = "python backend/experiments/run_latency.py"

# The inline authorisation budget this component must fit inside.
INLINE_BUDGET_MS = 100.0

TRAIN_COUNTS = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": 90,
    "ATTACK_2_SYNTHETIC_MULE_RING": 90,
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": 90,
    "ATTACK_5_APP_SCAM_PERSONALISED": 90,
    "ATTACK_6_VPA_RENTAL_MULE": 90,
}
MEASURE_COUNTS = {
    "ATTACK_2_SYNTHETIC_MULE_RING": 120,   # graph layer does real work here
    "ATTACK_6_VPA_RENTAL_MULE": 120,       # fan-in: widest graph neighbourhood
    "ATTACK_7_SYNCHRONISED_BURST_CASHOUT": 120,
    "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING": 120,
}


def _percentiles(samples_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(samples_ms, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": round(float(arr.mean()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p90_ms": round(float(np.percentile(arr, 90)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "p999_ms": round(float(np.percentile(arr, 99.9)), 4),
        "max_ms": round(float(arr.max()), 4),
        "stdev_ms": round(float(statistics.pstdev(arr.tolist())), 4),
    }


def main() -> Dict[str, object]:
    t0 = time.time()
    print("building corpus and training the stack ...")
    train = build_corpus(n_legit=N_LEGIT_TRAIN, attack_counts=TRAIN_COUNTS, seed=SEED)
    engine = DecisionEngine(environment=train["env"])
    engine.train(train["rows"])

    measure = build_corpus(
        n_legit=N_LEGIT_MEASURE, attack_counts=MEASURE_COUNTS, seed=SEED + 700
    )
    engine_m = DecisionEngine(
        environment=measure["env"], scorer=engine.scorer, novelty=engine.novelty
    )
    stream = sorted(measure["rows"], key=lambda r: r["payload"]["timestamp"])
    print(f"measuring decide() over {len(stream)} transactions (warmup {WARMUP}) ...")

    samples: List[float] = []
    by_class: Dict[str, List[float]] = {"legit": [], "attack": []}
    # Graph state grows as the stream advances, so latency is also reported by
    # stream position: a cold graph is not the same cost as a warm one.
    by_decile: Dict[int, List[float]] = {i: [] for i in range(10)}

    n = len(stream)
    for i, row in enumerate(stream):
        payload = row["payload"]
        t = time.perf_counter()
        engine_m.decide(payload)
        dt_ms = (time.perf_counter() - t) * 1000.0

        if i < WARMUP:
            continue  # JIT/cache warmup excluded, and stated
        samples.append(dt_ms)
        by_class["legit" if row["label"] == 0 else "attack"].append(dt_ms)
        by_decile[min(int(10 * i / n), 9)].append(dt_ms)

    overall = _percentiles(samples)
    per_class = {k: _percentiles(v) for k, v in by_class.items() if v}
    per_decile = {
        f"decile_{k}": _percentiles(v) for k, v in sorted(by_decile.items()) if v
    }

    headroom = INLINE_BUDGET_MS - overall["p99_ms"]
    verdict = (
        f"p99 {overall['p99_ms']:.2f} ms against a {INLINE_BUDGET_MS:.0f} ms inline "
        f"authorisation budget: {headroom:.1f} ms of headroom "
        f"({100.0 * overall['p99_ms'] / INLINE_BUDGET_MS:.1f}% of budget consumed)."
    )

    # ---- chart ---------------------------------------------------------- #
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.5, 5.6))

    arr = np.asarray(samples)
    clip = float(np.percentile(arr, 99.5))
    ax.hist(arr[arr <= clip], bins=70, color="#2563eb", alpha=0.85)
    for label, key, color in (
        ("p50", "p50_ms", "#059669"),
        ("p95", "p95_ms", "#f59e0b"),
        ("p99", "p99_ms", "#dc2626"),
    ):
        ax.axvline(overall[key], color=color, lw=1.9, ls="--",
                   label=f"{label} = {overall[key]:.2f} ms")
    ax.set_xlabel("end-to-end DecisionEngine.decide() latency (ms)")
    ax.set_ylabel("transactions")
    ax.set_title(
        "Measured inline decision latency, full four-layer stack\n"
        f"n = {overall['n']:,} scored transactions (warmup excluded)",
        fontsize=11.4, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    labels = list(per_decile.keys())
    p50s = [per_decile[k]["p50_ms"] for k in labels]
    p99s = [per_decile[k]["p99_ms"] for k in labels]
    xs = np.arange(len(labels))
    bx.plot(xs, p50s, marker="o", color="#059669", lw=2, label="p50")
    bx.plot(xs, p99s, marker="s", color="#dc2626", lw=2, label="p99")
    bx.axhline(INLINE_BUDGET_MS, color="#111827", lw=1.6, ls="--",
               label=f"{INLINE_BUDGET_MS:.0f} ms inline budget")
    bx.set_xticks(xs)
    bx.set_xticklabels([f"D{i}" for i in range(len(labels))], fontsize=9)
    bx.set_xlabel("stream position (decile) — entity graph grows left to right")
    bx.set_ylabel("latency (ms)")
    bx.set_yscale("log")
    bx.set_title(
        "Latency vs graph size: does the topology layer\ndegrade as the entity graph fills up?",
        fontsize=11.4, fontweight="bold",
    )
    bx.legend(fontsize=9)
    bx.grid(alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    write_artifact(
        "latency",
        {
            "experiment": "measured_inline_decision_latency",
            "question": "Does the full four-layer decision stack fit inside an inline authorisation budget?",
            "protocol": {
                "seed": SEED,
                "measured_call": "DecisionEngine.decide() -- the exact call the WebSocket server makes",
                "n_scored": overall["n"],
                "warmup_excluded": WARMUP,
                "stream_composition": {
                    "legit": len(by_class["legit"]),
                    "attack": len(by_class["attack"]),
                },
                "inline_budget_ms": INLINE_BUDGET_MS,
                "timer": "time.perf_counter(), per-transaction, single-threaded",
                "includes": [
                    "feature assembly incl. sliding-window velocity counters",
                    "supervised XGBoost scoring",
                    "unsupervised Isolation Forest novelty scoring",
                    "entity-graph ring check",
                    "threshold comparison and reason-code emission",
                    "observe-after-score state folding",
                ],
                "excludes": [
                    "network transport",
                    "ISO 8583 / ISO 20022 parsing",
                    "the issuer's own decision logic",
                ],
            },
            "overall": overall,
            "by_class": per_class,
            "by_stream_decile": per_decile,
            "verdict": verdict,
            "boundaries": [
                "Single-threaded, single-process, on the machine that produced the provenance stamp; a production host would differ.",
                "Cold-start warmup is excluded and the excluded count is reported.",
                "Latency is measured on the component, not on an end-to-end authorisation round trip.",
                "The entity graph grows over the stream, so the per-decile table is the honest read on scaling rather than the headline mean.",
            ],
        },
        seeds=[SEED],
        command=COMMAND,
    )

    ledger = ClaimLedger()
    path = ARTIFACTS_DIR / "claim_ledger.json"
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8")).get("claims", [])
        except Exception:
            prior = []
        seen = set()
        for entry in prior:
            key = entry.get("claim")
            if key and key not in seen:
                seen.add(key)
                ledger.entries.append(entry)
    (
        ledger.add(
            claim=f"The full four-layer decision stack scores inline at p99 {overall['p99_ms']:.2f} ms, inside a {INLINE_BUDGET_MS:.0f} ms authorisation budget.",
            artifact="latency",
            field="overall.p99_ms",
            derivation="time.perf_counter() around DecisionEngine.decide() over a mixed legit/attack stream, warmup excluded.",
            boundary="Single-threaded on one machine; excludes network transport and ISO message parsing.",
        ).write(command=COMMAND)
    )

    print(json.dumps({"overall": overall, "verdict": verdict}, indent=2))
    print(f"chart -> {CHART_PATH}")
    print(f"elapsed {time.time() - t0:.1f}s")
    return {"overall": overall, "verdict": verdict}


if __name__ == "__main__":
    main()
