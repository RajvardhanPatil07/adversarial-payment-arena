"""
Layer 2 — Entity graph for fraud-ring detection (NetworkX).

Nodes: customer (`C:`), device (`D:`), ip (`I:`), merchant (`M:`).
Edges: observed co-occurrence on accepted transactions (customer USED device,
customer SEEN at ip, customer SHOPPED at merchant).

Ring rule: a customer is in a ring when the current transaction would make a
device or IP directly shared by >=3 customers. Merchant edges are stored for
topology/UI but deliberately EXCLUDED from ring logic — everyone shops at the
same popular merchants, so merchant hubs must never inflate ring size or make
per-transaction checks traverse the whole graph.

The hot-path check is intentionally local: it inspects only the current
transaction's device/IP neighborhoods. Global analytics can still call
`scan_rings()`, which walks the merchant-free graph off the request path.
"""

from __future__ import annotations

import hashlib

import networkx as nx


class EntityGraph:
    """Incremental entity graph over the accepted-transaction stream."""

    def __init__(self) -> None:
        self.g = nx.Graph()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def observe(self, wire: dict) -> list[tuple[str, str]]:
        """Fold one ACCEPTED transaction into the graph. Call AFTER check().

        Returns only edges created by this observation. Repeated transactions
        merely bump edge weights and therefore return an empty list. The live
        API uses this delta directly instead of rescanning every graph edge
        after every transaction.
        """
        c, d, i, m = (
            f"C:{wire['customer_id']}",
            f"D:{wire['device_id']}",
            f"I:{wire['ip_address']}",
            f"M:{wire['merchant_id']}",
        )
        for n in (c, d, i, m):
            self.g.add_node(n)

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
    # Ring detection
    # ------------------------------------------------------------------ #

    def _linked_customers(self, infra_node: str) -> set[str]:
        if infra_node not in self.g:
            return set()
        return {nb for nb in self.g.neighbors(infra_node) if nb.startswith("C:")}

    def _prospective_customers(self, infra_node: str, customer_node: str) -> set[str]:
        """Customers linked after accepting the *current* transaction."""
        linked = self._linked_customers(infra_node)
        linked.add(customer_node)
        return linked

    def check(self, payload: dict) -> dict:
        """
        Ring screen for one payload. The current transaction is evaluated
        prospectively, so the third customer joining a shared device/IP is
        caught immediately instead of only on a later returning transaction.

        Returns
          {ring_detected, ring_id, risk_score, component_customers,
           shared_infra: [{node, linked_customers}]}

        `component_customers` is retained for API compatibility; on the live
        path it now means the directly implicated customer cohort reachable
        through this payload's device/IP, not a merchant-inflated global
        connected component.
        """
        c_node = f"C:{payload['customer_id']}"
        candidate_infra = (
            f"D:{payload['device_id']}",
            f"I:{payload['ip_address']}",
        )

        shared: list[dict] = []
        implicated = {c_node}
        soft_max = 1

        for infra_node in candidate_infra:
            linked = self._prospective_customers(infra_node, c_node)
            implicated.update(linked)
            soft_max = max(soft_max, len(linked))
            if len(linked) >= 3:
                shared.append({
                    "node": infra_node,
                    "linked_customers": len(linked),
                })

        base = {
            "ring_detected": False,
            "ring_id": None,
            "risk_score": 0.0,
            "component_customers": len(implicated),
            "shared_infra": sorted(shared, key=lambda s: -s["linked_customers"]),
        }

        if shared:
            # The ring identity is derived only from customers directly tied to
            # shared infrastructure in this transaction. Merchant hubs cannot
            # leak unrelated customers into the ID or risk score.
            ring_customers = {c_node}
            for item in shared:
                ring_customers.update(self._prospective_customers(item["node"], c_node))

            max_link = max(item["linked_customers"] for item in shared)
            base["ring_detected"] = True
            base["ring_id"] = self._ring_id(ring_customers)
            base["component_customers"] = len(ring_customers)
            base["risk_score"] = min(
                1.0,
                round(0.12 * len(ring_customers) + 0.06 * max_link, 3),
            )
        else:
            # Soft signal: this transaction would share infra with one other
            # profile (e.g. a couple sharing a tablet). Worth a little risk,
            # but it is not a ring.
            base["risk_score"] = round(min(0.4, 0.08 * max(soft_max - 1, 0)), 3)
        return base

    @staticmethod
    def _ring_id(customers: set[str]) -> str:
        digest = hashlib.sha1("|".join(sorted(customers)).encode()).hexdigest()[:10]
        return f"RING_{digest.upper()}"

    # ------------------------------------------------------------------ #
    # Global scans (UI / analytics)
    # ------------------------------------------------------------------ #

    def scan_rings(self) -> list[dict]:
        """Whole-graph ring scan over the merchant-free entity graph.

        This is deliberately off the transaction hot path. Connected
        components are deterministic here and match the documented topology:
        customers, devices and IPs only; merchant hubs are excluded.
        """
        infra = self.g.subgraph(
            [n for n in self.g.nodes if not n.startswith("M:")]
        ).copy()
        rings = []
        for community in nx.connected_components(infra):
            customers = {n for n in community if n.startswith("C:")}
            if len(customers) < 3:
                continue
            shared = [
                {"node": n, "linked_customers": len(self._linked_customers(n))}
                for n in community
                if n.startswith(("D:", "I:")) and len(self._linked_customers(n)) >= 3
            ]
            if shared:
                rings.append({
                    "ring_id": self._ring_id(customers),
                    "customers": len(customers),
                    "shared_infra": sorted(shared, key=lambda s: -s["linked_customers"]),
                })
        return rings


__all__ = ["EntityGraph"]
