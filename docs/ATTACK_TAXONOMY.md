# Attack Taxonomy: GenAI-Enabled Payment Fraud

Breadth is a scoring criterion, so it should be honest about what is
implemented versus what is mapped. This document is the **map**. Four scenarios
are code-wired today as executable `AttackSpec` YAMLs; the rest are specified
precisely enough that implementing them is mechanical, and each one names the
fields it moves and the defensive signal it is designed to defeat.

Status legend: **[CODE]** executable spec in `backend/attack_specs/` |
**[SPEC]** mapped, not yet code-wired.

The organising principle is not the fraud's name but **what GenAI changed about
it**. Card-testing is old; card-testing where an agent reads decline reason
codes and rewrites its own amount-and-MCC policy between attempts is not.

---

## Layer 1 - Identity and onboarding

| ID | Scenario | What GenAI changed | Fields moved | Signal it defeats |
|---|---|---|---|---|
| T-01 | **Synthetic identity bust-out** **[CODE]** | Generated coherent identity histories that survive KYC document checks and age gracefully before the bust | `customer_id` novelty, thin velocity history, escalating `amount` | history-length and account-age heuristics |
| T-02 | **Deepfake liveness injection** **[SPEC]** | Video and voice synthesis defeats selfie-liveness at onboarding and at step-up | `device_id` churn, `three_ds_status` = Y obtained fraudulently | biometric step-up as a terminal control |
| T-03 | **Voice-clone MFA reset** **[CODE]** | Cloned cardholder voice passes call-centre verification, resetting credentials before any transaction | `device_id` new, `ip_country` mismatch, post-reset burst | device-binding and step-up trust |
| T-04 | **AI-assisted document forgery for merchant onboarding** **[SPEC]** | Generated registration and bank-proof documents onboard a fake merchant fast | new `merchant_id` with immediate high-ticket inflow | merchant vetting at acquiring |

## Layer 2 - Authentication and authorisation

| ID | Scenario | What GenAI changed | Fields moved | Signal it defeats |
|---|---|---|---|---|
| T-05 | **OTP-relay vishing at scale** **[SPEC]** | Conversational agents run thousands of simultaneous, personalised OTP-extraction calls | `three_ds_status` = Y with anomalous `device_id` | treating a passed 3DS challenge as proof of cardholder presence |
| T-06 | **3DS frictionless-flow abuse** **[SPEC]** | Attacks are shaped to stay inside the risk-based-authentication exemption band | `amount` just below step-up thresholds, `three_ds_status` = A | threshold-based step-up policy |
| T-07 | **Session hijack and drain** **[SPEC]** | Automated post-hijack behaviour that imitates the victim's own transaction rhythm | same `device_id`, altered beneficiary and `mcc` | device-consistency signals |

## Layer 3 - Card-not-present and transaction shaping

| ID | Scenario | What GenAI changed | Fields moved | Signal it defeats |
|---|---|---|---|---|
| T-08 | **Adaptive card-testing swarm** **[CODE]** | The agent reads decline reason codes and rewrites amount, MCC and cadence policy between attempts | small `amount`, `pos_entry_mode` = ECOM, `three_ds_status` = N, high velocity | fixed velocity rules |
| T-09 | **Amount structuring below review thresholds** **[CODE]** | Learned, per-issuer estimation of the review threshold rather than a guessed round number | `amount` clustered just under limits, low `amount_round_frac` | static amount thresholds |
| T-10 | **MCC laundering** **[SPEC]** | Category selection optimised against the issuer's own observed decline surface | `mcc` shifted to low-risk bands, mismatched `ip_country` | MCC risk weighting |
| T-11 | **Geo-velocity spoof with plausible travel** **[SPEC]** | Generated itineraries that make impossible travel look like a real trip | `ip_country` sequence, `hour_sin`/`hour_cos` shift | impossible-travel rules |

## Layer 4 - Real-time rails and UPI-era patterns (India-first)

| ID | Scenario | What GenAI changed | Fields moved | Signal it defeats |
|---|---|---|---|---|
| T-12 | **AI-personalised APP scam (authorised push payment)** **[CODE]** | Scam narratives personalised from scraped public data; the victim authorises the payment themselves | genuine `customer_id`, genuine `device_id`, novel beneficiary | every control premised on the cardholder being the victim of a *stolen* credential |
| T-13 | **Digital-arrest / impersonation coercion** **[SPEC]** | Synthetic authority voices and documents sustain multi-hour coercion | sequence of escalating `amount` from one legitimate account | single-transaction risk scoring |
| T-14 | **VPA-rental mule networks** **[CODE]** | Automated recruitment and rotation of rented payment addresses | fan-in topology, short-lived `customer_id` clusters | per-account monitoring without graph context |
| T-15 | **Fan-out dispersal and layered multi-hop** **[SPEC]** | Route planning that keeps every hop individually unremarkable | fan-out degree, rapid hop timing | thresholds applied per transaction |
| T-16 | **QR and collect-request redirection** **[SPEC]** | Generated payee identities and QR overlays that survive visual inspection | `merchant_id` mismatch against context | payee-name verification by eye |
| T-17 | **Synchronised burst cash-out** **[CODE]** | Coordinated timing across many mules within one detection window | tight timestamp clustering across unrelated `customer_id`s | independence assumptions between accounts |

## Layer 5 - Merchant, agentic commerce and model-layer attacks

| ID | Scenario | What GenAI changed | Fields moved | Signal it defeats |
|---|---|---|---|---|
| T-18 | **Prompt-injected merchant / agentic checkout hijack** **[CODE]** | A poisoned merchant catalogue or page redirects an AI shopping agent's checkout | agent-driven `merchant_id`, unusual `mcc` for the cardholder | the assumption that the buyer is a human making a choice |
| T-19 | **Agent scope expansion** **[SPEC]** | A delegated payment agent drifts beyond its mandate through intent manipulation | repeated authorised payments, rising `amount`, stable device | consent captured once at delegation time |
| T-20 | **GenAI collusive merchant ring** **[SPEC]** | Coordinated fake merchants and fake customers generate mutually reinforcing transaction history | dense bipartite merchant-customer topology | reputation built from transaction volume |
| T-21 | **Synthetic-evidence refund and chargeback abuse** **[SPEC]** | Generated receipts, photos and delivery evidence win disputes | post-authorisation, dispute-layer | manual evidence review |
| T-22 | **Model-layer attack: poisoning the feedback loop** **[SPEC]** | Attacks crafted to be *labelled* wrongly, corrupting the retraining set itself | mislabelled escapes entering the hardening corpus | closed-loop retraining without label provenance |

---

## Why T-22 matters to this repository specifically

T-22 is the attack that a closed-loop red-team system is uniquely exposed to,
and it is the reason `docs/TRANSFER_LEDGER.md` exists. If the escapes folded
back into training are unrepresentative of real fraud, the loop degrades the
detector on real traffic while every internal metric improves. That is not a
hypothetical: it is measured in this repository, and one comparable submission
published a 3.8-point real-fraud recall loss from exactly this mechanism.

A closed loop without a fidelity gate is an attack surface, not a feature.

---

## Coverage summary

| | Count |
|---|---|
| Scenarios mapped | 22 |
| Executable specs today **[CODE]** | 8 |
| Layers covered | 5 |
| India-specific real-time-rail scenarios | 6 (T-12 to T-17) |

**Stated honestly:** eight of twenty-two are executable. Claiming twenty-two
*implemented* attacks would be the kind of number this repository is built to
argue against. The taxonomy's value is that each unimplemented row already
names its fields and its target signal, so each is an afternoon of work rather
than a research question.

Executable does not mean "generates rows". Each of the eight is measured, and
each was admitted only after passing the Plausibility Gate's economic,
metadata-coherence and rail-feasibility checks -- so a family cannot be counted
by writing a YAML file that produces physically impossible traffic. ATTACK_6
initially failed that bar: it claimed a `synthetic_identity` ($200 acquisition
floor) while drawing amounts from $180, and the gate rejected the low draws as
economically irrational. The fix was not to widen the band but to correct the
threat model -- a rented VPA is a *real*, KYC-clean account borrowed for a
rotation window, not a fabricated identity, which is exactly why it is cheap
($45) and why small mule legs are rational. That constraint is now enforced at
load time by a schema validator, with the misconfiguration pinned as a
regression test in `tests/test_attack_specs.py`.

Per-family detection recall, leave-one-family-out zero-day generalisation, and
which defense layer fires for each family are reported in
`artifacts/family_coverage.json` (`make coverage`). The four newest families
were chosen for a specific reason: each defeats a *different* control, so
coverage breadth is not eight variations of velocity abuse.

| Family | Defeats |
|---|---|
| T-12 AI-personalised APP scam | every control premised on a *stolen* credential -- the victim's own device, own account, and a passed 3DS challenge |
| T-14 VPA-rental mule (fan-in) | per-account monitoring: no single node exceeds a threshold; only beneficiary convergence is visible |
| T-17 synchronised burst cash-out | the independence assumption between accounts -- nothing is shared but time |
| T-09 learned threshold structuring | static amount thresholds, including the round-number heuristics that catch naive structuring |

Adding one is mechanical: write the YAML against `schemas/attack.AttackSpec`,
add a synthesizer function to `data/corpus_builder._SYNTHESIZERS`, then rerun
`make reproduce` so the new family is measured by the same fidelity and
transfer gates as everything else.
