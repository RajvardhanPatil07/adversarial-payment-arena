# Adversarial Payment Arena: Closed-Loop Defense Against GenAI Fraud

A full-stack, closed-loop adversarial simulation and real-time defense testbed for GenAI-driven payment fraud. Built for the **Mastercard Innovation Challenge**.

---

## Architecture Overview

The arena simulates an active fight between an autonomous **LLM Red-Team Attacker** and a **Multi-Layer Defense Decisioning Stack**, streaming live telemetry, graph mutations, and financial cost impact to a Next.js 16 SOC Dashboard over WebSockets.

```
 ┌──────────────────────────────────────────────────────────────┐
 │                  RED-TEAM ATTACKER AGENT                    │
 │        Autonomous LLM (OpenRouter / Claude / Stealth)        │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                      PLAUSIBILITY GATE                       │
 │    Economic Floors · Metadata Coherence · Rail Feasibility   │
 └──────────────────────────────┬───────────────────────────────┘
                                │ (Valid Transactions)
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                 DEFENSE DECISIONING STACK                    │
 │  1. XGBoost Velocity Model (Supervised Behavioral Scoring)   │
 │  2. Isolation Forest (Unsupervised Novelty & Anomaly Layer)  │
 │  3. NetworkX Entity Graph (Mule Ring & Topology Detection)   │
 │  4. Asymmetric Cost Matrix Optimizer ($ Block vs False Pos)  │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │               FASTAPI WEBSOCKET STREAM ENGINE                │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │              NEXT.js 16 SOC COMMAND DASHBOARD                │
 │   xyflow Entity Graph · Live Log Triage · Cost Matrix Analytics │
 └──────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

### 1. Autonomous LLM Red-Team Agent
- Dynamically crafts attacks constrained by real-world fraud economics (acquisition costs, target verticals, 3DS tolerances, POS entry modes).
- Operates online via **OpenRouter** (any model, set with `OPENROUTER_MODEL`; defaults to a free reasoning model, and uses `stealth/ox-alpha` when that slug is served to the account) or fully offline via a deterministic fallback fraudster — so the demo runs with **no API key**.
- Adaptive payload mutation based on real-time defense feedback and system telemetry.

### 2. Multi-Layer Defense Decisioning Stack
- **Supervised Velocity Scoring (XGBoost):** Evaluates sliding-window burst dynamics across cards, IPs, and merchant terminals.
- **Unsupervised Anomaly Detection (Isolation Forest):** Flag novel out-of-distribution patterns without requiring prior labels.
- **Graph Intelligence (NetworkX):** Real-time entity-resolution graph tracking shared infrastructure across unrelated accounts to uncover mule rings.
- **Asymmetric Cost Matrix:** Dynamically weighs False Negatives (fraud loss) vs False Positives (customer insult cost) to optimize net business impact.

### 3. Reproducible Headline: does generator fidelity determine transfer?

- **Research question:** when you train a fraud detector on *synthetic* red-team data, does the **fidelity** of the generator decide whether that data **helps or hurts** the detector on real fraud?
- **Design:** a pre-registered three-arm ablation (`A0` real-only baseline · `A1` independent-marginal synthetic · `A2` Gaussian-copula synthetic), everything else held constant — same real rows, same augmentation budget, same detector, thresholds pinned at **1% FPR** on a disjoint legitimate split, evaluated on held-out real fraud over seeds `[11, 23, 37]` with bootstrap CIs.
- **Result (every number below is emitted by `make reproduce` into `artifacts/`):**
  - The higher-fidelity generator is measurably more realistic: **C2ST AUC 0.954 (copula) vs 0.981 (independent)**, and rank-dependence error **0.98 vs 2.46** (Frobenius). Lower is more realistic; `0.5` C2ST would be indistinguishable-from-real.
  - Fidelity tracks transfer: the copula's recall penalty is **smaller** (Δrecall **−0.009**, CI touches 0) than the independent control's (**−0.013**). At a baseline already near-ceiling (0.996 recall), better fidelity does **less harm** — the relationship holds directionally, and the pre-registered fidelity gate is what flags a generator that would degrade a live detector *before* deployment.
  - **Economics:** at a 1.3% production base rate and 1% FPR, the stack nets **≈ ₹226M per million authorisations** — and wrongly-declined legitimate payments are the **single largest cost term (≈ 47%)**, which is exactly why the cost matrix is asymmetric.
- **No unverifiable hero numbers.** Every figure in this README is reachable from `make reproduce`, lands in `artifacts/*.json` with a provenance stamp (git SHA, seeds, command), and is mapped claim-by-claim in [`artifacts/claim_ledger.json`](artifacts/claim_ledger.json) / the `/evidence` page. A separate zero-day holdout (`make zero-day`) stress-tests an unseen attack family.

### 4. Interactive SOC Command Center
- **Next.js 16 + React 19 + Tailwind CSS v4** interface.
- **Interactive Entity Graph (@xyflow/react):** Dynamic DAG canvas highlighting compromised nodes, mule clusters, and merchant hotspots in real-time.
- **Real-Time WebSocket Ingestion:** Live streaming of attacker reasoning, plausibility checks, defense decisions, and financial cost metrics.

---

## Attack Specifications

| Attack Spec | Vector | Target | Tell / Fingerprint |
|---|---|---|---|
| **ATTACK 1** | Voice-Clone IVR ATO | CNP / High-Value Electronics | Phone-channel credential reset followed by rapid CNP drain |
| **ATTACK 2** | Synthetic Mule Ring | Contactless / Physical Tap | Shared device ID & egress IP across distinct customer profiles |
| **ATTACK 3** | Compromised Merchant Checkout | ECOM / Stored Credential | Per-card normality with abnormal customer convergence at single MID |
| **ATTACK 4 (Zero-Day)** | Automated CNP Card Testing | ECOM / Low-Value Tickets | High-cadence micro-transactions across hundreds of cards via bot egress |

---

## Project Structure

```
adversarial-payment-arena/
├── backend/
│   ├── agents/            # LLM red-team attacker agent & prompt templates
│   ├── attack_specs/      # Structured YAML attack definitions
│   ├── data/              # Synthetic generator, corpus builder & schemas
│   ├── defense/           # XGBoost, Isolation Forest, NetworkX Graph & Cost Engine
│   ├── environment/       # Payment state machine & Plausibility Gate
│   ├── evidence/          # Calibration, economics & the claim→artifact ledger
│   ├── experiments/       # Transfer-vs-fidelity ablation + zero-day holdout
│   ├── models/            # Serialized ML models (committed: xgb .json + iForest .joblib)
│   ├── schemas/           # Pydantic data schemas
│   ├── tests/             # Pytest test suite
│   ├── main.py            # FastAPI REST & WebSocket streaming server
│   └── run_campaign.py    # CLI runner for headless campaign benchmarks
├── artifacts/             # Provenance-stamped JSON evidence (via `make reproduce`)
├── docs/
│   ├── TRANSFER_LEDGER.md      # Headline fidelity-vs-transfer experiment write-up
│   ├── transfer_ledger.png     # The three-arm result figure
│   ├── ATTACK_TAXONOMY.md      # Full attack taxonomy
│   └── FEASIBILITY.md          # ISO 8583/20022 mapping & deployment path
└── frontend/
    ├── src/
    │   ├── app/           # Next.js App Router (SOC dashboard + /evidence page)
    │   ├── components/    # xyflow canvas, logs feed, cost telemetry, analyst panel
    │   └── lib/           # WebSocket client & state management
    └── package.json
```

---

## Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 20+** & **pnpm**
- *(Optional)* `OPENROUTER_API_KEY` for live LLM attacker reasoning

### 1. Backend Setup

```bash
cd backend

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run automated test suite
pytest tests/

# (Optional) retrain + re-serialize the defense models into backend/models/.
# The repo already ships trained models, so this is only needed if you change
# the corpus or feature pipeline. Run from the repo root:
#   make models

# Start FastAPI server (Port 8000)
uvicorn main:app --reload --port 8000
```

> The two serialized detectors (`backend/models/xgb_model.json`, `iforest_model.joblib`) are **committed**, so a fresh clone demos a working four-layer defense with **no build and no API key**.

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Start development server (Port 3000)
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) to view the SOC dashboard.

---

## Running Campaigns & Experiments

### Headless Campaign Execution
```bash
# Offline (no key): the mule-ring campaign visibly forms a ring and gets DECLINED.
python backend/run_campaign.py --attack attack_2_synthetic_mule_ring --size 25 --fast
python backend/run_campaign.py --attack attack_1 --size 50
```

### Regenerate the full evidence set (headline experiment + economics)
```bash
make reproduce        # calibration + fidelity + transfer ablation -> artifacts/*.json
```

### Zero-Day Holdout Experiment Benchmark
```bash
make zero-day         # or: python backend/experiments/zero_day_holdout.py
```

---

## License

MIT License. Developed for research and demonstration in payment security.

