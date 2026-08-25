"""
Layer 2 — Entity graph for fraud-ring detection (NetworkX).

Nodes: customer (`C:`), device (`D:`), ip (`I:`), merchant (`M:`).
Edges: observed co-occurrence on accepted transactions (customer USED device,
customer SEEN at ip, customer SHOPPED at merchant).

Ring rule: a connected component containing >=3 CUSTOMER nodes where at
least one shared-infrastructure node (device or IP) directly links >=3 of
them. Merchant edges are stored for topology/UI but deliberately EXCLUDED
from ring logic — everyone shops at Safeway; shared merchants are not
collusion. Shared devices and egress IPs are.

In production, this would be a Graph Neural Network (GNN) learning entity
embeddings over the issuer's full payment graph. We use NetworkX connected
components + label propagation to capture ~80% of the value in ~1% of the
build time — and we're honest about it.
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

    def observe(self, wire: dict) -> None:
        """Fold one ACCEPTED transaction into the graph. Call AFTER check()."""
        c, d, i, m = (
            f"C:{wire['customer_id']}",
            f"D:{wire['device_id']}",
            f"I:{wire['ip_address']}",
            f"M:{wire['merchant_id']}",
        )
        for n in (c, d, i, m):
            self.g.add_node(n)
        self._bump_edge(c, d)
        self._bump_edge(c, i)
        self._bump_edge(c, m)

    def _bump_edge(self, a: str, b: str) -> None:
        if self.g.has_edge(a, b):
            self.g[a][b]["weight"] += 1
        else:
            self.g.add_edge(a, b, weight=1)

    # ------------------------------------------------------------------ #
    # Ring detection
    # ------------------------------------------------------------------ #

    def _linked_customers(self, infra_node: str) -> set[str]:
        return {nb for nb in self.g.neighbors(infra_node) if nb.startswith("C:")}

    def check(self, payload: dict) -> dict:
        """
        Ring screen for one payload. Returns
          {ring_detected, ring_id, risk_score, component_customers,
           shared_infra: [{node, linked_customers}]}
        """
        c_node = f"C:{payload['customer_id']}"
        base = {
            "ring_detected": False,
            "ring_id": None,
            "risk_score": 0.0,
            "component_customers": 0,
            "shared_infra": [],
        }
        if c_node not in self.g:
            return base  # first-ever customer: nothing to collude with yet

        comp = nx.node_connected_component(self.g, c_node)
        customers = {n for n in comp if n.startswith("C:")}
        base["component_customers"] = len(customers)

        # CRITICAL SCOPING RULE: shared infrastructure must be DIRECTLY
        # linked to THIS customer. Co-residence in a component via merchant
        # hubs does NOT implicate anyone — everyone shops at Safeway; a mule
        # ring across town is not your ring until you share its device/IP.
        payload_linked_infra = [
            n for n in self.g.neighbors(c_node)
            if n.startswith(("D:", "I:"))
        ]
        shared = []
        for n in payload_linked_infra:
            linked = self._linked_customers(n)
            if len(linked) >= 3:
                shared.append({"node": n, "linked_customers": len(linked)})

        base["shared_infra"] = sorted(shared, key=lambda s: -s["linked_customers"])
        max_link = max((s["linked_customers"] for s in shared), default=0)

        if shared and len(customers) >= 3:
            base["ring_detected"] = True
            base["ring_id"] = self._ring_id(customers)
            base["risk_score"] = min(1.0, round(0.12 * len(customers) + 0.06 * max_link, 3))
        else:
            # soft signal: this customer's infra shared with exactly one other
            # profile (couple sharing a tablet) — worth a little risk, no ring.
            soft = max(
                (len(self._linked_customers(n)) for n in payload_linked_infra),
                default=1,
            )
            base["risk_score"] = round(min(0.4, 0.08 * max(soft - 1, 0)), 3)
        return base

    @staticmethod
    def _ring_id(customers: set[str]) -> str:
        digest = hashlib.sha1("|".join(sorted(customers)).encode()).hexdigest()[:10]
        return f"RING_{digest.upper()}"

    # ------------------------------------------------------------------ #
    # Global scans (UI / analytics)
    # ------------------------------------------------------------------ #

    def scan_rings(self) -> list[dict]:
        """
        Whole-graph ring scan via label propagation communities over the
        INFRA-ONLY subgraph (merchant nodes stripped), then the same
        >=3-customers-via-shared-infra rule per community.
        """
        infra = self.g.subgraph(
            [n for n in self.g.nodes if not n.startswith("M:")]
        ).copy()
        rings = []
        for community in nx.community.label_propagation_communities(infra):
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
