"""Graph / topology fidelity, numpy+pandas only (no networkx).

Fraud is relational: one device driving many customers, one IP fanning out
across a ring, a mule cluster that forms a single large connected component.
A row-level generator cannot reproduce any of this, because it never models
entity identity -- it invents a fresh customer/device/IP per row, which yields a
graph of isolated edges. This module measures that gap explicitly.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .divergence import compare_numeric, composite_similarity

ENTITY_COLUMNS = ("customer", "device", "ip", "merchant")


def _present(df: pd.DataFrame) -> List[str]:
    return [c for c in ENTITY_COLUMNS if c in df.columns]


def fanout(df: pd.DataFrame, source: str, target: str) -> np.ndarray:
    """Distinct `target` entities touched by each `source` entity."""
    if source not in df.columns or target not in df.columns:
        return np.asarray([], dtype=float)
    return df.groupby(source, sort=False)[target].nunique().to_numpy(dtype=float)


def reuse_rate(df: pd.DataFrame, source: str, target: str) -> float:
    """Share of `source` entities linked to more than one distinct `target`."""
    f = fanout(df, source, target)
    if f.size == 0:
        return float("nan")
    return float(np.mean(f > 1))


def degrees(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=float)
    return df.groupby(col, sort=False).size().to_numpy(dtype=float)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, a: str) -> str:
        self.parent.setdefault(a, a)
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def component_sizes(df: pd.DataFrame) -> np.ndarray:
    """Connected-component sizes of the entity graph (nodes = typed entities)."""
    cols = _present(df)
    if len(cols) < 2:
        return np.asarray([], dtype=float)
    uf = _UnionFind()
    arrays = {c: df[c].astype(str).to_numpy() for c in cols}
    anchor = arrays[cols[0]]
    for i in range(len(df)):
        base = f"{cols[0]}:{anchor[i]}"
        for c in cols[1:]:
            uf.union(base, f"{c}:{arrays[c][i]}")
    sizes: Dict[str, int] = {}
    for node in list(uf.parent):
        root = uf.find(node)
        sizes[root] = sizes.get(root, 0) + 1
    return np.asarray(sorted(sizes.values()), dtype=float)


def shared_entity_share(df: pd.DataFrame) -> Dict[str, float]:
    """Scalar topology summary: how much entity sharing exists at all."""
    out: Dict[str, float] = {}
    for src, tgt in (("device", "customer"), ("ip", "customer"), ("merchant", "customer"), ("customer", "device")):
        r = reuse_rate(df, src, tgt)
        out[f"{src}_multi_{tgt}_share"] = None if not np.isfinite(r) else round(float(r), 6)
    return out


def graph_fidelity_report(real: pd.DataFrame, synth: pd.DataFrame) -> Dict[str, object]:
    pairs: Tuple[Tuple[str, str], ...] = (
        ("device", "customer"),
        ("ip", "customer"),
        ("merchant", "customer"),
        ("customer", "device"),
    )
    records: List[Dict[str, object]] = []
    for src, tgt in pairs:
        r = fanout(real, src, tgt)
        s = fanout(synth, src, tgt)
        if r.size and s.size:
            records.append(compare_numeric(f"fanout_{src}_to_{tgt}", r, s))
    for col in _present(real):
        r, s = degrees(real, col), degrees(synth, col)
        if r.size and s.size:
            records.append(compare_numeric(f"degree_{col}", r, s))
    rc, sc = component_sizes(real), component_sizes(synth)
    if rc.size and sc.size:
        records.append(compare_numeric("component_size", np.log1p(rc), np.log1p(sc)))
    real_shares = shared_entity_share(real)
    synth_shares = shared_entity_share(synth)
    share_gap = {
        k: (
            None
            if real_shares.get(k) is None or synth_shares.get(k) is None
            else round(abs(float(real_shares[k]) - float(synth_shares[k])), 6)
        )
        for k in real_shares
    }
    return {
        "family": "graph",
        "measures": records,
        "composite_similarity": composite_similarity(records),
        "real_entity_sharing": real_shares,
        "synth_entity_sharing": synth_shares,
        "entity_sharing_absolute_gap": share_gap,
        "real_largest_component": None if rc.size == 0 else float(rc.max()),
        "synth_largest_component": None if sc.size == 0 else float(sc.max()),
        "interpretation": (
            "A generator with no notion of entity identity produces a graph of "
            "isolated edges: fanout collapses to 1, entity-sharing shares collapse "
            "to 0, and the largest connected component shrinks to the number of "
            "entity types. That is visible here even when row-level marginals match."
        ),
    }
