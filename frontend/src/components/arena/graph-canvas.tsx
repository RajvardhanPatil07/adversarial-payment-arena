"use client";

/**
 * EntityGraphCanvas — live fraud-ring topology on @xyflow/react.
 *
 * The backend sends incremental graph_update events (new edges + endpoint
 * nodes). We merge them into local xyflow state, re-run a dagre LR layout
 * on a throttle, and color nodes by entity type. Ring-ish edges (weight >= 3
 * shared contacts) render hot so mule rings visually pop.
 *
 * Client Components are still prerendered at build time, so the canvas only
 * mounts after hydration (mounted guard) — ResizeObserver-safe.
 */

import { useEffect, useRef, useSyncExternalStore } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  ReactFlow,
  type Edge,
  type Node,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import dagre from "dagre";
import "@xyflow/react/dist/style.css";

import type { GraphEdge, GraphNode } from "@/lib/arena-types";

const NODE_W = 150;
const NODE_H = 34;
const MAX_NODES = 150; // keep the canvas readable during long campaigns

// Design tokens only (SECTION 1: no inline hex). Exactly two accent hues:
// customers sit in the defender-blue family, device/ip are attacker
// infrastructure in the red family, merchant/unknown stay neutral.
const TYPE_STYLE: Record<string, { bg: string; border: string; label: string }> = {
  customer: { bg: "var(--blue-dim)", border: "var(--blue)", label: "var(--text)" },
  device: { bg: "var(--red-dim)", border: "var(--red)", label: "var(--text)" },
  ip: { bg: "var(--surface-3)", border: "var(--red)", label: "var(--text)" },
  merchant: { bg: "var(--surface-3)", border: "var(--border-hi)", label: "var(--text)" },
  unknown: { bg: "var(--surface-2)", border: "var(--border-hi)", label: "var(--text-dim)" },
};

function shortLabel(id: string): string {
  const [prefix, ...rest] = id.split(":");
  const body = rest.join(":");
  const cut = body.length > 13 ? `${body.slice(0, 13)}…` : body;
  return `${prefix}·${cut}`;
}

function layout(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 22, ranksep: 70 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of edges) g.setEdge(e.source, e.target);
  dagre.layout(g);

  return {
    nodes: nodes.map((n) => {
      const pos = g.node(n.id);
      return { ...n, position: { x: (pos?.x ?? 0) - NODE_W / 2, y: (pos?.y ?? 0) - NODE_H / 2 } };
    }),
    edges,
  };
}

function toFlowNode(n: GraphNode): Node {
  const style = TYPE_STYLE[n.type] ?? TYPE_STYLE.unknown;
  return {
    id: n.id,
    position: { x: 0, y: 0 },
    data: { label: shortLabel(n.id) },
    style: {
      background: style.bg,
      border: `1px solid ${style.border}`,
      color: style.label,
      borderRadius: 8,
      fontSize: 10,
      fontFamily: "var(--font-jetbrains-mono), monospace",
      padding: 4,
      width: NODE_W,
    },
  };
}

function toFlowEdge(e: GraphEdge): Edge {
  const hot = e.weight >= 3; // shared infra across >=3 entities == ring-ish
  return {
    id: `${e.source}->${e.target}`,
    source: e.source,
    target: e.target,
    animated: hot,
    style: {
      stroke: hot ? "var(--red)" : "var(--border)",
      strokeWidth: hot ? 2 : 1,
      opacity: hot ? 0.95 : 0.55,
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: hot ? "var(--red)" : "var(--text-faint)", width: 14, height: 14 },
  };
}

const emptySubscribe = () => () => {};

export function EntityGraphCanvas({
  nodes,
  edges,
  className,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  className?: string;
}) {
  // Hydration-safe mount detection without setState-in-effect: server snapshot
  // is always false, client snapshot true — canvas renders only post-hydration.
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );
  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState<Node>([]);
  const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const layoutTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!mounted) return;
    // Merge + cap: FIFO drop keeps the canvas focused on recent activity.
    const capped = nodes.length > MAX_NODES ? nodes.slice(nodes.length - MAX_NODES) : nodes;
    const ids = new Set(capped.map((n) => n.id));
    const cappedEdges = edges.filter((e) => ids.has(e.source) && ids.has(e.target));

    if (layoutTimer.current) clearTimeout(layoutTimer.current);
    layoutTimer.current = setTimeout(() => {
      const { nodes: laidOutNodes, edges: laidOutEdges } = layout(
        capped.map(toFlowNode),
        cappedEdges.map(toFlowEdge),
      );
      setFlowNodes(laidOutNodes);
      setFlowEdges(laidOutEdges);
    }, 250);
    return () => {
      if (layoutTimer.current) clearTimeout(layoutTimer.current);
    };
  }, [mounted, nodes, edges, setFlowNodes, setFlowEdges]);

  if (!mounted) {
    return (
      <div className={`flex items-center justify-center text-xs text-muted-foreground ${className ?? ""}`}>
        initializing canvas…
      </div>
    );
  }

  return (
    <div className={className}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        minZoom={0.15}
        maxZoom={1.6}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: false }}
        colorMode="dark"
      >
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="var(--surface-3)" />
        <Controls showInteractive={false} className="!bottom-3 !left-3" />
      </ReactFlow>
    </div>
  );
}
