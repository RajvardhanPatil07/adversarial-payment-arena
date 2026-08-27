"""
Shared feature framing for fidelity measurement.

Both the real corpus and every synthesizer are projected into ONE common
feature frame so that fidelity metrics (C2ST, JSD, TVD, correlation diff) and
transfer metrics (TSTR) are computed on identical columns.

Design notes
------------
* Columns are derived only from the wire payload, so any source of payments
  (live corpus, copula sample, rule sample) can be framed the same way.
* Cyclical time is encoded as sin/cos so that 23:59 and 00:01 are neighbours
  rather than opposite extremes -- a joint-structure detail that independent
  marginal samplers routinely destroy.
* `amount_round_frac` captures the "round number" tell (fraudsters type 5000,
  cardholders pay 4987.35). Cheap, but high-signal.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping

import pandas as pd

NUMERIC_COLS: list[str] = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "dow",
    "mcc_num",
    "amount_round_frac",
]

CATEGORICAL_COLS: list[str] = [
    "pos_entry_mode",
    "three_ds_status",
    "ip_country",
    "mcc_group",
]

ALL_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def mcc_group(mcc: int) -> str:
    """Coarse MCC family: keeps categorical cardinality sane while still
    carrying the merchant-mix structure that fraud campaigns distort."""
    if mcc < 3000:
        return "travel_transport"
    if mcc < 5000:
        return "services"
    if mcc < 5600:
        return "retail_general"
    if mcc < 6000:
        return "retail_specialty"
    if mcc < 7000:
        return "financial"
    if mcc < 8000:
        return "business_services"
    return "professional_other"


def payload_to_features(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project one wire payload into the common fidelity frame."""
    ts = _parse_timestamp(payload["timestamp"])
    amount = float(payload["amount"])
    mcc = int(payload["mcc"])
    hour = ts.hour + ts.minute / 60.0
    fractional = round(amount - math.floor(amount), 2)

    three_ds = payload.get("3ds_status")
    if three_ds is None:
        three_ds = payload.get("three_ds_status", "N")

    return {
        "log_amount": math.log10(max(amount, 0.01)),
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "dow": float(ts.weekday()),
        "mcc_num": float(mcc),
        "amount_round_frac": 1.0 if fractional == 0.0 else 0.0,
        "pos_entry_mode": str(payload["pos_entry_mode"]),
        "three_ds_status": str(three_ds),
        "ip_country": str(payload["ip_country"]),
        "mcc_group": mcc_group(mcc),
    }


def frame_from_payloads(payloads: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    records = [payload_to_features(p) for p in payloads]
    return pd.DataFrame.from_records(records, columns=ALL_COLS)


def frame_from_rows(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """`rows` are corpus rows shaped {label, attack_id, payload, features}."""
    return frame_from_payloads(r["payload"] for r in rows)


def labels_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    return [int(r["label"]) for r in rows]


__all__ = [
    "NUMERIC_COLS",
    "CATEGORICAL_COLS",
    "ALL_COLS",
    "mcc_group",
    "payload_to_features",
    "frame_from_payloads",
    "frame_from_rows",
    "labels_from_rows",
]
