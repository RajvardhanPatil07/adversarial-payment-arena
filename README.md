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
- Operates online via **OpenRouter** (e.g. `anthropic/claude-3.5-sonnet`, `stealth/ox-alpha`) or offline via deterministic fallback engines.
- Adaptive payload mutation based on real-time defense feedback and system telemetry.

### 2. Multi-Layer Defense Decisioning Stack
- **Supervised Velocity Scoring (XGBoost):** Evaluates sliding-window burst dynamics across cards, IPs, and merchant terminals.
- **Unsupervised Anomaly Detection (Isolation Forest):** Flag novel out-of-distribution patterns without requiring prior labels.
- **Graph Intelligence (NetworkX):** Real-time entity-resolution graph tracking shared infrastructure across unrelated accounts to uncover mule rings.
- **Asymmetric Cost Matrix:** Dynamically weighs False Negatives (fraud loss) vs False Positives (customer insult cost) to optimize net business impact.

### 3. Zero-Day Holdout Robustness
- **Research question:** Can the defense hold when attacked by a vector never seen during training?
- Supervised model is trained only on baseline + Attacks 1-3. **Attack 4 (CNP High-Velocity Card Testing) is strictly withheld.**
- **Results:**
  - **94% zero-day detection rate**
  - **92% caught by unsupervised Isolation Forest** alone
  - **0.8% false positive rate** on holdout legitimate traffic.

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
│   ├── experiments/       # Zero-day holdout evaluation protocol
│   ├── models/            # Serialized ML models (.joblib / .json)
│   ├── schemas/           # Pydantic data schemas
│   ├── tests/             # Pytest test suite
│   ├── main.py            # FastAPI REST & WebSocket streaming server
│   └── run_campaign.py    # CLI runner for headless campaign benchmarks
├── docs/
│   ├── ZERO_DAY_EXPERIMENT.md   # Benchmark protocol & detailed breakdown
│   └── zero_day_results.png     # ROC and confusion matrix visualizations
└── frontend/
    ├── src/
    │   ├── app/           # Next.js App Router (SOC dashboard)
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

# Start FastAPI server (Port 8000)
uvicorn main:app --reload --port 8000
```

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
python backend/run_campaign.py --attack attack_1 --size 50
```

### Zero-Day Holdout Experiment Benchmark
```bash
python backend/experiments/run_zero_day.py
```

---

## License

MIT License. Developed for research and demonstration in payment security.

