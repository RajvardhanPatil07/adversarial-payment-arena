"""Behavioral fingerprints for the deterministic no-key red-team attacker.

The live LLM receives rich AttackSpec prose and can reason about each family.
The deterministic CI/demo attacker historically generated nearly the same
transaction shape for most specs, which meant the named taxonomy was richer
than the traffic it produced. This module closes that gap.

Policies only mutate fields already present in PaymentMessage and always remain
inside the active AttackSpec's declared amount/rail/resource/vertical envelope.
Unsupported concepts (beneficiaries, mandates, settlement state) stay documented
as portfolio projection gaps rather than being smuggled into unrelated fields.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agents.fraud_portfolio import PORTFOLIO


class SyntheticVectorPolicy:
    """Stateful per-campaign policy for one existing AttackSpec."""

    def __init__(self, spec: object, env: object, rng: object) -> None:
        self.spec = spec
        self.env = env
        self.rng = rng
        self.step = 0
        self.customer_ids = sorted(env.customers.keys())
        self._stable_devices: dict[str, str] = {}
        self._stable_ips: dict[str, str] = {}
        self._merchant_cache: dict[str, list[Any]] = {}
        self._fixed_merchants: dict[str, Any] = {}
        self._counters: dict[str, int] = defaultdict(int)

    @property
    def spec_id(self) -> str:
        return str(self.spec.spec_id)

    @property
    def constraints(self):
        return self.spec.constraints

    def _customers(self, n: int = 8) -> list[str]:
        if not self.customer_ids:
            return []
        return self.customer_ids[: min(n, len(self.customer_ids))]

    def _customer(self, index: int) -> str:
        return self.customer_ids[index % len(self.customer_ids)]

    def _known_device(self, customer_id: str) -> str:
        return self.env.customers[customer_id].devices[0]

    def _stable_device(self, key: str) -> str:
        value = self._stable_devices.get(key)
        if value is None:
            value = f"DEV_RT{self.rng.getrandbits(32):08x}"
            self._stable_devices[key] = value
        return value

    def _stable_ip(self, key: str) -> str:
        value = self._stable_ips.get(key)
        if value is None:
            # Documentation-range address avoids implying a routable endpoint.
            token = 10 + (self.rng.randrange(200) % 190)
            host = 1 + self.rng.randrange(250)
            value = f"192.0.2.{(token + host) % 253 + 1}"
            self._stable_ips[key] = value
        return value

    def _customer_ip(self, customer_id: str) -> str:
        try:
            numeric = int(customer_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            numeric = abs(hash(customer_id))
        return f"198.51.100.{numeric % 253 + 1}"

    def _merchants(self, category: str | None = None) -> list[Any]:
        key = category or "*"
        if key in self._merchant_cache:
            return self._merchant_cache[key]
        allowed = set(self.constraints.target_verticals or [])
        rows = list(self.env.merchant_registry.values())
        if category is not None:
            rows = [merchant for merchant in rows if merchant.category == category]
        elif allowed:
            rows = [merchant for merchant in rows if merchant.category in allowed]
        if not rows:
            rows = list(self.env.merchant_registry.values())
        self._merchant_cache[key] = rows
        return rows

    def _merchant(self, *, category: str | None = None, index: int = 0, fixed: str | None = None):
        if fixed is not None and fixed in self._fixed_merchants:
            return self._fixed_merchants[fixed]
        rows = self._merchants(category)
        merchant = rows[index % len(rows)]
        if fixed is not None:
            self._fixed_merchants[fixed] = merchant
        return merchant

    def _set_merchant(self, payload: dict, merchant: Any, *, preserve_gate_fault: bool) -> None:
        payload["merchant_id"] = merchant.merchant_id
        if not preserve_gate_fault:
            payload["mcc"] = merchant.mcc
        payload["ip_country"] = merchant.country

    def _set_customer(self, payload: dict, customer_id: str, *, known_device: bool, device_key: str | None = None) -> None:
        payload["customer_id"] = customer_id
        if known_device:
            payload["device_id"] = self._known_device(customer_id)
        else:
            payload["device_id"] = self._stable_device(device_key or customer_id)

    def _amount(self, fraction: float, *, jitter: float = 0.0, nonround: bool = False) -> float:
        lo, hi = self.constraints.amount_band
        width = max(float(hi) - float(lo), 0.0)
        value = float(lo) + width * max(0.0, min(1.0, fraction))
        if jitter:
            value += width * jitter
        value = max(float(lo), min(float(hi), value))
        if nonround and value + 0.37 <= float(hi):
            value = float(int(value)) + 0.37
            value = max(float(lo), min(float(hi), value))
        return round(value, 2)

    def _preferred_tds(self, index: int = 0) -> str:
        options = self.constraints.preferred_three_ds
        return options[index % len(options)].value

    def _mode(self, index: int = 0) -> str:
        options = self.constraints.pos_entry_modes
        return options[index % len(options)].value

    def _baseline(self, payload: dict, *, preserve_gate_fault: bool) -> None:
        merchant = self._merchant(index=self.step)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["pos_entry_mode"] = self._mode(self.step)
        payload["3ds_status"] = self._preferred_tds(self.step)
        payload["amount"] = self._amount(0.50, jitter=((self.step % 5) - 2) * 0.02)

    def _attack_1(self, payload: dict, preserve_gate_fault: bool) -> str:
        group = (self.step - 1) // 3
        cid = self._customer(group)
        self._set_customer(payload, cid, known_device=False, device_key=f"a1:{cid}")
        payload["ip_address"] = self._stable_ip(f"a1:{cid}")
        merchant = self._merchant(index=self.step)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["amount"] = self._amount(0.48, jitter=(self.step % 3) * 0.03)
        return "session-style account-takeover sequence with a stable newly observed device"

    def _attack_2(self, payload: dict, preserve_gate_fault: bool) -> str:
        ring = self._customers(4)
        cid = ring[(self.step - 1) % len(ring)]
        self._set_customer(payload, cid, known_device=False, device_key="a2:ring")
        payload["ip_address"] = self._stable_ip("a2:ring")
        merchant = self._merchant(category="electronics", index=self.step)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["amount"] = self._amount(0.42, jitter=(self.step % 4) * 0.02)
        return "synthetic ring sequence sharing one device and egress identity across multiple customers"

    def _attack_3(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer(self.step - 1)
        self._set_customer(payload, cid, known_device=True)
        payload["ip_address"] = self._customer_ip(cid)
        merchant = self._merchant(category="ecommerce", fixed="a3:merchant")
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["3ds_status"] = "Y" if any(item.value == "Y" for item in self.constraints.preferred_three_ds) else self._preferred_tds()
        payload["amount"] = self._amount(0.55, jitter=((self.step % 3) - 1) * 0.015)
        return "merchant-centric fan-in burst across otherwise unrelated customers"

    def _attack_4(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer(self.step - 1)
        self._set_customer(payload, cid, known_device=False, device_key="a4:bot")
        payload["ip_address"] = self._stable_ip("a4:bot")
        merchant = self._merchant(category="ecommerce", index=(self.step - 1) // 3)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["amount"] = self._amount(0.08 + (self.step % 4) * 0.035, nonround=True)
        return "cross-customer low-ticket testing burst on shared automation infrastructure"

    def _attack_5(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer((self.step - 1) // 2)
        self._set_customer(payload, cid, known_device=True)
        payload["ip_address"] = self._customer_ip(cid)
        merchant = self._merchant(category="ecommerce", index=self.step)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["3ds_status"] = "Y"
        payload["amount"] = self._amount(0.62, jitter=(self.step % 3) * 0.05)
        return "victim-authorized behavioral-shift projection using the genuine bound device"

    def _attack_6(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer(self.step - 1)
        self._set_customer(payload, cid, known_device=True)
        payload["ip_address"] = self._customer_ip(cid)
        # Current PaymentMessage has no beneficiary/VPA entity. A small fixed
        # merchant pool is used only as the documented graph projection.
        merchant = self._merchant(category="ecommerce", index=(self.step - 1) % 2, fixed=f"a6:recipient:{(self.step - 1) % 2}")
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["amount"] = self._amount(0.32, jitter=(self.step % 5) * 0.04)
        return "many-sender fan-in projection into a small rotating recipient pool"

    def _attack_7(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer(self.step - 1)
        self._set_customer(payload, cid, known_device=True)
        payload["ip_address"] = self._customer_ip(cid)
        merchant = self._merchant(category="electronics", index=(self.step - 1) % 3)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["amount"] = self._amount(0.46, jitter=((self.step % 4) - 1.5) * 0.025)
        return "distributed cross-account cash-out wave designed to be visible in aggregate timing"

    def _attack_8(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer((self.step - 1) // 6)
        self._set_customer(payload, cid, known_device=False, device_key=f"a8:{cid}")
        payload["ip_address"] = self._stable_ip(f"a8:{cid}")
        merchant = self._merchant(index=(self.step - 1) % 2)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        jitter = ((self.step % 5) - 2) * 0.004
        payload["amount"] = self._amount(0.62, jitter=jitter, nonround=True)
        return "tight low-variance non-round amount band used as a distributional stress pattern"

    def _attack_9(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer((self.step - 1) // 3)
        self._set_customer(payload, cid, known_device=False, device_key=f"a9:{cid}")
        payload["ip_address"] = self._stable_ip(f"a9:{cid}")
        merchant = self._merchant(index=self.step)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["3ds_status"] = "Y"
        payload["amount"] = self._amount(0.58, jitter=(self.step % 3) * 0.05)
        return "passed-challenge transactions arriving from a newly observed device"

    def _attack_10(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer((self.step - 1) // 4)
        self._set_customer(payload, cid, known_device=False, device_key=f"a10:{cid}")
        payload["ip_address"] = self._stable_ip(f"a10:{cid}")
        merchant = self._merchant(index=self.step)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["amount"] = self._amount(0.88, jitter=((self.step % 5) - 2) * 0.008, nonround=True)
        return "narrow low-velocity exemption-band stress pattern"

    def _attack_11(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer((self.step - 1) // 8)
        self._set_customer(payload, cid, known_device=True)
        payload["ip_address"] = self._customer_ip(cid)
        categories = [category for category in ("ecommerce", "electronics") if self._merchants(category)]
        category = categories[0] if self.step <= 3 or len(categories) == 1 else categories[(self.step - 3) % len(categories)]
        merchant = self._merchant(category=category, index=self.step)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        fraction = min(0.10 + 0.10 * (self.step - 1), 0.92)
        payload["amount"] = self._amount(fraction)
        payload["3ds_status"] = "Y"
        return "trusted-device sequence with progressive amount escalation and merchant-category widening"

    def _attack_12(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer((self.step - 1) // 6)
        self._set_customer(payload, cid, known_device=False, device_key=f"a12:{cid}")
        sequence = []
        for category in ("travel", "ecommerce"):
            sequence.extend(self._merchants(category)[:3])
        if not sequence:
            sequence = self._merchants()
        merchant = sequence[(self.step - 1) % len(sequence)]
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        payload["ip_address"] = self._stable_ip(f"a12:{merchant.country}:{cid}")
        payload["amount"] = self._amount(0.30 + ((self.step - 1) % 4) * 0.12)
        return "multi-merchant travel/ecommerce sequence whose pairwise metadata stays coherent"

    def _attack_13(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer(self.step - 1)
        self._set_customer(payload, cid, known_device=True)
        payload["ip_address"] = self._customer_ip(cid)
        merchant = self._merchant(category="ecommerce", fixed="a13:shell")
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        # Short quiet ramp followed by a denser high-ticket wave.
        fraction = 0.28 if self.step <= 3 else min(0.68 + 0.04 * (self.step - 4), 0.92)
        payload["amount"] = self._amount(fraction)
        return "young-merchant ramp followed by cross-customer high-ticket concentration"

    def _attack_14(self, payload: dict, preserve_gate_fault: bool) -> str:
        cid = self._customer((self.step - 1) // 8)
        self._set_customer(payload, cid, known_device=False, device_key=f"a14:{cid}")
        payload["ip_address"] = self._stable_ip(f"a14:{cid}")
        merchant = self._merchant(index=(self.step - 1) % 2)
        self._set_merchant(payload, merchant, preserve_gate_fault=preserve_gate_fault)
        if self.step <= 4:
            fractions = (0.02, 0.12, 0.26, 0.42)
            payload["amount"] = self._amount(fractions[self.step - 1], nonround=True)
            phase = "reconnaissance"
        else:
            payload["amount"] = self._amount(0.76, jitter=((self.step % 3) - 1) * 0.03, nonround=True)
            phase = "exploitation"
        return f"two-phase {phase} sequence for testing whether the defender recognizes campaign history"

    def apply(self, move: dict, *, preserve_gate_fault: bool = False) -> dict:
        """Mutate one provider-like move in-place and return it."""
        self.step += 1
        payload = move.get("payload")
        if not isinstance(payload, dict):
            return move

        self._baseline(payload, preserve_gate_fault=preserve_gate_fault)
        handler = getattr(self, f"_attack_{self._attack_number()}", None)
        if handler is None:
            fingerprint = "generic AttackSpec-constrained synthetic campaign"
        else:
            fingerprint = handler(payload, preserve_gate_fault)

        # Resource and rail envelope stay authoritative even after a vector
        # handler. This is intentionally defensive programming around CI/demo
        # generation, not a second source of attack semantics.
        resource = self.constraints.stolen_resource
        payload["stolen_resource"] = resource.value if resource is not None else None
        allowed_modes = {mode.value for mode in self.constraints.pos_entry_modes}
        if payload.get("pos_entry_mode") not in allowed_modes:
            payload["pos_entry_mode"] = self._mode()
        allowed_tds = {status.value for status in self.constraints.preferred_three_ds}
        if payload.get("3ds_status") not in allowed_tds:
            payload["3ds_status"] = self._preferred_tds()
        lo, hi = self.constraints.amount_band
        payload["amount"] = round(max(float(lo), min(float(hi), float(payload["amount"]))), 2)

        if preserve_gate_fault:
            payload["mcc"] = 7994

        prefix = f"Synthetic vector fingerprint [{self.spec_id}]: {fingerprint}. "
        move["reasoning"] = prefix + str(move.get("reasoning", ""))
        return move

    def _attack_number(self) -> int:
        # Exact prefix split avoids ATTACK_1 matching ATTACK_10..14.
        try:
            return int(self.spec_id.split("_", 2)[1])
        except (IndexError, ValueError):
            return -1

    @property
    def mapped(self) -> bool:
        return self.spec_id in PORTFOLIO


__all__ = ["SyntheticVectorPolicy"]
