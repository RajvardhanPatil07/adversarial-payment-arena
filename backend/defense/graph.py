"""Evidence-weighted entity graph for real-time fraud-ring detection.

NetworkX is retained as the topology/UI mirror, but the authorization decision
uses a compact temporal risk state. When ``arena_graph_core`` is installed that
state is implemented in Rust; source checkouts use a semantically equivalent
Python fallback.

Evidence policy:
* shared device across unrelated customers is strong evidence;
* shared IP is weak by itself (office/campus/carrier NAT is normal);
* shared IP becomes hard evidence only when beneficiary/merchant convergence
  occurs in the same recent window;
* merchant fan-in contributes soft risk but never creates a ring on its own.

This removes the old "three people on one IP => hard decline" failure mode while
keeping the immediate third-customer shared-device signal used by mule attacks.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

import networkx as nx

try:
    from arena_graph_core import GraphRiskState as _RustGraphRiskState
except ImportError:
    _RustGraphRiskState = None


class _RecentCustomers:
    def __init__(self) -> None:
        self.events: deque[tuple[float, str]] = deque()
        self.counts: Counter[str] = Counter()

    def purge(self, now: float, window_seconds: float) -> None:
        while self.events and now - self.events[0][0] > window_seconds:
            _, customer = self.events.popleft()
            self.counts[customer] -= 1
            if self.counts[customer] <= 0:
                del self.counts[customer]

    def prospective_distinct(self, now: float, window_seconds: float, customer: str) -> int:
        self.purge(now, window_seconds)
        return len(self.counts) + int(customer not in self.counts)

    def observe(self, now: float, window_seconds: float, customer: str) -> None:
        self.purge(now, window_seconds)
        self.events.append((now, customer))
        self.counts[customer] += 1


class _PythonGraphRiskState:
    def __init__(self, window_seconds: float = 600.0) -> None:
        self.window_seconds = window_seconds
        self.devices: dict[str, _RecentCustomers] = defaultdict(_RecentCustomers)
        self.ips: dict[str, _RecentCustomers] = defaultdict(_RecentCustomers)
        self.merchants: dict[str, _RecentCustomers] = defaultdict(_RecentCustomers)

    @staticmethod
    def _device_risk(degree: int) -> float:
        if degree <= 1:
            return 0.0
        if degree == 2:
            return 0.18
        return min(0.95, 0.55 + 0.08 * (degree - 3))

    @staticmethod
    def _ip_risk(degree: int) -> float:
        if degree <= 1:
            return 0.0
        if degree <= 4:
            return 0.03 * (degree - 1)
        return min(0.35, 0.10 + 0.03 * (degree - 5))

    @staticmethod
    def _merchant_risk(degree: int) -> float:
        if degree <= 3:
            return 0.0
        if degree <= 7:
            return 0.025 * (degree - 3)
        return min(0.65, 0.20 + 0.04 * (degree - 8))

    @staticmethod
    def _fuse(parts: tuple[float, ...]) -> float:
        safe = [max(0.0, min(float(value), 1.0)) for value in parts]
        product = 1.0
        for value in safe:
            product *= 1.0 - value
        return max(0.0, min(1.0, 1.0 - product))

    def check(
        self,
        ts: float,
        customer_id: str,
        device_id: str,
        ip_address: str,
        merchant_id: str,
    ) -> tuple[float, bool, int, int, int]:
        device_degree = self.devices[device_id].prospective_distinct(
            ts, self.window_seconds, customer_id
        )
        ip_degree = self.ips[ip_address].prospective_distinct(
            ts, self.window_seconds, customer_id
        )
        merchant_degree = self.merchants[merchant_id].prospective_distinct(
            ts, self.window_seconds, customer_id
        )
        ring = device_degree >= 3 or (ip_degree >= 5 and merchant_degree >= 5)
        risk = self._fuse(
            (
                self._device_risk(device_degree),
                self._ip_risk(ip_degree),
                self._merchant_risk(merchant_degree),
            )
        )
        if ring:
            risk = max(risk, 0.72)
        return risk, ring, device_degree, ip_degree, merchant_degree

    def observe(
        self,
        ts: float,
        customer_id: str,
        device_id: str,
        ip_address: str,
        merchant_id: str,
    ) -> None:
        self.devices[device_id].observe(ts, self.window_seconds, customer_id)
        self.ips[ip_address].observe(ts, self.window_seconds, customer_id)
        self.merchants[merchant_id].observe(ts, self.window_seconds, customer_id)

    def state_sizes(self) -> tuple[int, int, int]:
        return len(self.devices), len(self.ips), len(self.merchants)


class EntityGraph:
    """Incremental graph mirror plus temporal evidence-weighted risk state."""

    def __init__(self, *, window_seconds: float = 600.0) -> None:
        self.g = nx.Graph()
        self.window_seconds = float(window_seconds)
        self._risk = (
            _RustGraphRiskState(self.window_seconds)
            if _RustGraphRiskState is not None
            else _PythonGraphRiskState(self.window_seconds)
        )
        self._logical_clock = 0.0

    @property
    def backend(self) -> str:
        return "rust" if _RustGraphRiskState is not None and not isinstance(
            self._risk, _PythonGraphRiskState
        ) else "python"

    def risk_state_sizes(self) -> dict[str, int]:
        devices, ips, merchants = self._risk.state_sizes()
        return {
            "devices": int(devices),
            "ips": int(ips),
            "merchants": int(merchants),
        }

    def _event_ts(self, wire: dict) -> float:
        raw = wire.get("timestamp")
        if raw is None:
            return self._logical_clock
        if isinstance(raw, datetime):
            value = raw
        else:
            value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()

    # ------------------------------------------------------------------ #
    # State / UI topology
    # ------------------------------------------------------------------ #

    def observe(self, wire: dict) -> list[tuple[str, str]]:
        """Fold one accepted transaction after ``check`` and return new UI edges."""
        ts = self._event_ts(wire)
        self._risk.observe(
            ts,
            str(wire["customer_id"]),
            str(wire["device_id"]),
            str(wire["ip_address"]),
            str(wire["merchant_id"]),
        )
        if wire.get("timestamp") is None:
            self._logical_clock += 1.0

        c, d, i, m = (
            f"C:{wire['customer_id']}",
            f"D:{wire['device_id']}",
            f"I:{wire['ip_address']}",
            f"M:{wire['merchant_id']}",
        )
        for node in (c, d, i, m):
            self.g.add_node(node)

        fresh: list[tuple[str, str]] = []
        for a, b in ((c, d), (c, i), (c, m)):
            if self._bump_edge(a, b):
                fresh.append((min(a, b), max(a, b)))
        return fresh

    def _bump_edge(self, a: str, b: str) -> bool:
        if self.g.has_edge(a, b):
            self.g[a][b]["weight"] += 1
            return False
        self.g.add_edge(a, b, weight=1)
        return True

    # ------------------------------------------------------------------ #
    # Ring / evidence risk
    # ------------------------------------------------------------------ #

    def _linked_customers(self, infra_node: str) -> set[str]:
        if infra_node not in self.g:
            return set()
        return {nb for nb in self.g.neighbors(infra_node) if nb.startswith("C:")}

    def _prospective_customers(self, infra_node: str, customer_node: str) -> set[str]:
        linked = self._linked_customers(infra_node)
        linked.add(customer_node)
        return linked

    def check(self, payload: dict) -> dict:
        """Prospectively score device/IP/beneficiary evidence for one payment."""
        ts = self._event_ts(payload)
        risk, ring_detected, device_degree, ip_degree, merchant_degree = self._risk.check(
            ts,
            str(payload["customer_id"]),
            str(payload["device_id"]),
            str(payload["ip_address"]),
            str(payload["merchant_id"]),
        )
        risk = round(float(risk), 3)
        device_degree = int(device_degree)
        ip_degree = int(ip_degree)
        merchant_degree = int(merchant_degree)

        c_node = f"C:{payload['customer_id']}"
        d_node = f"D:{payload['device_id']}"
        i_node = f"I:{payload['ip_address']}"
        implicated = {c_node}
        shared: list[dict] = []

        if device_degree >= 2:
            customers = self._prospective_customers(d_node, c_node)
            implicated.update(customers)
            shared.append({
                "node": d_node,
                "linked_customers": device_degree,
                "evidence": "strong_device",
            })
        if ip_degree >= 2:
            customers = self._prospective_customers(i_node, c_node)
            implicated.update(customers)
            shared.append({
                "node": i_node,
                "linked_customers": ip_degree,
                "evidence": "weak_ip",
            })

        ring_customers = {c_node}
        if device_degree >= 3:
            ring_customers.update(self._prospective_customers(d_node, c_node))
        elif ring_detected and ip_degree >= 5 and merchant_degree >= 5:
            ring_customers.update(self._prospective_customers(i_node, c_node))

        return {
            "ring_detected": bool(ring_detected),
            "ring_id": self._ring_id(ring_customers) if ring_detected else None,
            "risk_score": risk,
            "component_customers": len(ring_customers) if ring_detected else len(implicated),
            "shared_infra": sorted(shared, key=lambda item: -item["linked_customers"]),
            "evidence": {
                "device_degree_10m": device_degree,
                "ip_degree_10m": ip_degree,
                "merchant_fanin_10m": merchant_degree,
                "shared_ip_alone_is_hard_rule": False,
                "backend": self.backend,
            },
        }

    @staticmethod
    def _ring_id(customers: set[str]) -> str:
        digest = hashlib.sha1("|".join(sorted(customers)).encode()).hexdigest()[:10]
        return f"RING_{digest.upper()}"

    # ------------------------------------------------------------------ #
    # Global analytics
    # ------------------------------------------------------------------ #

    def scan_rings(self) -> list[dict]:
        """Conservative all-history scan for strong shared-device rings.

        Temporal weighted risk remains the live authority. The global graph is
        primarily a UI/case-analysis structure, so this scan intentionally
        avoids turning historic shared IPs into rings.
        """
        rings: list[dict] = []
        seen: set[str] = set()
        for node in self.g.nodes:
            if not node.startswith("D:"):
                continue
            customers = self._linked_customers(node)
            if len(customers) < 3:
                continue
            ring_id = self._ring_id(customers)
            if ring_id in seen:
                continue
            seen.add(ring_id)
            rings.append({
                "ring_id": ring_id,
                "customers": len(customers),
                "shared_infra": [
                    {
                        "node": node,
                        "linked_customers": len(customers),
                        "evidence": "strong_device",
                    }
                ],
            })
        return sorted(rings, key=lambda row: -row["customers"])


__all__ = ["EntityGraph"]
