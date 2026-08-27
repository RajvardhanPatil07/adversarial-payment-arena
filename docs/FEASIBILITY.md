# Real-World Feasibility

A red-team lab is interesting. A red-team lab that could be switched on inside
an authorisation path is deployable. This document states exactly where each
component would sit in a live payment stack, what it costs in latency, what it
cannot do, and what would have to be true before anyone ran it against real
cardholders.

---

## 1. Where this sits in the authorisation path

```
cardholder -> merchant -> acquirer -> SCHEME (ISO 8583 / ISO 20022)
                                        |
                                        v
                              issuer authorisation host
                                        |
                    +-------------------+--------------------+
                    |                                        |
            inline risk scoring                       near-real-time
            (blocking, <100 ms)                       graph + case queue
                    |                                        |
            DecisionEngine                            NoveltyDetector,
            (velocity + XGBoost)                      collusion topology
```

The classifier is an **inline advisory score** attached to the authorisation
request. It does not replace the issuer's decision logic; it contributes a
feature and a reason code, which is how risk models actually enter production.

The red-team side (attack generation, evolution, hardening) runs **offline** in
a model-development environment. Nothing in this repository generates traffic
toward a live scheme, and nothing should.

---

## 2. Message field mapping

The internal `PaymentMessage` schema maps onto real scheme fields, which is
what makes the feature set portable rather than a bespoke research format.

| Internal field | ISO 8583 | ISO 20022 (pacs/card) | Notes |
|---|---|---|---|
| `transaction_id` | DE 37 retrieval reference number | `TxId` | idempotency key |
| `amount` | DE 4 amount, transaction | `TxAmt` | minor units in production |
| `currency` | DE 49 currency code | `Ccy` | ISO 4217 |
| `mcc` | DE 18 merchant category code | `MrchntCtgyCd` | 4-digit |
| `merchant_id` | DE 42 card acceptor ID | `MrchntId` | |
| `pos_entry_mode` | DE 22 POS entry mode | `POICpblties` / `CardDataNtryMd` | ECOM, CHIP, CONTACTLESS, SWIPE, CNP |
| `three_ds_status` | DE 48 / scheme AAV | `AuthntcnRslt` | Y / A / N |
| `timestamp` | DE 7 transmission date-time | `CreDtTm` | UTC |
| `ip_address`, `ip_country` | DE 48 subelements / SecureCode data | `DvcChanl`, `IPAdr` | CNP only |
| `device_id` | scheme device tokenisation ID | `DvcId` | fingerprint or token |
| `customer_id` | PAN surrogate / token | `CardTknId` | never a raw PAN |

**PANs are never handled.** Every identifier is a surrogate or a network token,
which is both the PCI-DSS-correct design and how tokenised 3DS2 traffic already
arrives.

### Where the score attaches

- **3DS2 / EMV 3-D Secure**: the score contributes to the risk-based
  authentication decision, deciding frictionless approval versus step-up
  challenge. This is the highest-value integration point, because step-up is
  a cheaper action than decline.
- **Authorisation**: the score contributes a reason code to the issuer's
  decline or approve logic.
- **Post-authorisation**: graph and novelty signals feed a case queue rather
  than a blocking decision, because collusion topology needs more than one
  transaction to be visible.

---

## 3. Latency budget

Authorisation scoring has a hard budget. Typical issuer inline risk budgets sit
under 100 ms end to end, with the model itself allocated a fraction of that.

| Stage | Where it runs | Budget |
|---|---|---|
| feature assembly (velocity counters) | inline, in-memory store | ~5 ms |
| tree-ensemble scoring | inline | ~5-15 ms |
| threshold + reason codes | inline | <1 ms |
| novelty / isolation forest | inline optional | ~5 ms |
| graph community detection | **offline / near-real-time** | seconds to minutes |
| copula attack synthesis | **offline only** | not in the request path |

The architectural decision that makes this feasible: **the expensive parts are
deliberately not inline.** Graph topology and generative synthesis are
development-time and case-management-time components. Only the tree ensemble
and the velocity counters sit in the blocking path, and both are
milliseconds-class.

Velocity counters in production would be a Redis or Aerospike keyspace with TTL
windows keyed on token, device and IP, not the in-process dictionaries used
here. That substitution changes the operational profile, not the feature
definition.

---

## 4. Thresholds, drift and monitoring

The threshold policy in this repository is the one an issuer actually needs:
**pin the operating point on a validation split, then measure it on disjoint
data, and publish the gap.** In production this becomes:

- daily re-pinning of the threshold on the previous window's legitimate traffic
  to hold the false-positive budget fixed as traffic shifts
- population stability index on the score distribution, alarming on drift
  before recall degrades
- feature-level drift monitoring on the marginals that the fidelity lab already
  measures (JSD and TVD per column)
- champion/challenger deployment, with the challenger scoring in shadow mode
  until its calibration gap and prevalence-adjusted precision are confirmed
- automatic rollback triggered by false-positive budget breach, because an
  insult-rate spike damages more customers per hour than a recall dip

---

## 5. Governance and regulatory posture (India)

- **RBI tokenisation (CoFT)**: the schema consumes tokens, never PANs, which is
  consistent with the card-on-file tokenisation mandate.
- **DPDP Act 2023**: features are behavioural and transactional. No biometric
  template, no message content, no contact data. Purpose limitation is
  satisfiable because every feature has a documented fraud-prevention purpose.
- **Data residency**: the model is stateless at inference; feature stores stay
  in-country. Nothing in the design requires cross-border movement of
  transaction data.
- **Explainability**: every decision carries reason codes, and SHAP attribution
  is available per decision for adverse-action explanation and for dispute
  handling.
- **Model risk management**: the artifact and claim ledgers are exactly what a
  model validation function asks for -- what was claimed, how it was derived,
  on what data, with what boundary, reproducible by command.
- **Adversarial safety**: attack specifications describe *transaction patterns*,
  not exploitation instructions. No component targets a real institution, and
  the generated corpora are synthetic throughout.

---

## 6. What this is not, stated plainly

1. **Not validated on issuer production data.** Every number here is measured
   on a synthetic payment environment. The transfer result is a statement about
   the relationship between generator fidelity and detector quality, not an
   absolute recall figure for live traffic.
2. **Not a complete fraud system.** No device attestation, no behavioural
   biometrics, no consortium data, no rules engine, no case management, no
   dispute workflow. It is a red-team loop plus a detector plus the evidence
   discipline to know whether the loop helped.
3. **Not proof against adaptive adversaries in the wild.** The attacker here
   optimises against a frozen defender within a bounded action space. Real
   adversaries change channel, recruit differently, and exploit operational
   process, not only transaction features.
4. **Latency figures are architectural estimates**, not measurements taken on
   issuer infrastructure under production load.
5. **The graph layer needs volume.** Collusion and mule topology are invisible
   at low transaction counts, so its value in a demo understates its value in
   production and its false-positive behaviour at scale is untested here.

The honest summary: the inline path is realistic and cheap, the offline path is
where the novelty lives, and the missing step before any pilot is validation on
real labelled traffic under a fixed false-positive budget.
