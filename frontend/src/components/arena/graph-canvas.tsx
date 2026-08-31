"use client";

/**
 * EntityGraphCanvas — live fraud-ring topology on @xyflow/react.
 *
 * The backend sends incremental graph_update events (new edges + endpoint
 * nodes). We merge them into local xyflow state, then layout each connected
 * component separately with dagre (a dagre LR of one big graph collapses
 * disconnected nodes onto x=0, which combined with auto-fitView turned the
 * canvas into a microscopic stack). The frame re-fits on every update within
 * clamped zoom bounds, so entities that arrive late stay on screen.
 *
 * Client Components are still prerendered at build time, so the canvas only
 * mounts after hydration (mounted guard) — ResizeObserver-safe.
 */

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  ReactFlow,
  type Edge,
  type Node,
  type ReactFlowInstance,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import dagre from "dagre";
import "@xyflow/react/dist/style.css";

import type { GraphEdge, GraphNode } from "@/lib/arena-types";

const NODE_W = 150;
const NODE_H = 34;
const MAX_NODES = 150; // keep the canvas readable during long campaigns

// Keep the ring readable: never zoom out below 0.45 (was 0.15) so nodes and
// edges stay legible even on a busy graph.
const MIN_ZOOM_VIS = 0.45;
// Zoom-in cap for the ring-focus: too close and you lose the topology.
const MAX_ZOOM_VIS = 1.6;

// Stripe disconnected components side-by-side so each one gets its own dagre
// layout instead of collapsing onto a single x=0 column. Chain-shaped components
// are only NODE_H tall, so these gaps are world area spent on air — and world
// area is precisely what the frame runs out of at MIN_ZOOM_VIS.
const COMPONENT_STRIPE_GAP = 40;
const COMPONENT_RANK_GAP = 16;
// Breathing room kept around the laid-out world when framing it.
const FIT_PADDING = 16;

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));

const TYPE_STYLE: Record<string, { bg: string; border: string; label: string }> = {
  customer: { bg: "#0c4a6e", border: "#38bdf8", label: "#e0f2fe" },
  device: { bg: "#451a03", border: "#f59e0b", label: "#fef3c7" },
  ip: { bg: "#2e1065", border: "#a78bfa", label: "#ede9fe" },
  merchant: { bg: "#064e3b", border: "#34d399", label: "#d1fae5" },
  unknown: { bg: "#27272a", border: "#71717a", label: "#e4e4e7" },
};

function shortLabel(id: string): string {
  const [prefix, ...rest] = id.split(":");
  const body = rest.join(":");
  const cut = body.length > 13 ? `${body.slice(0, 13)}…` : body;
  return `${prefix}·${cut}`;
}

/** Tarjan-style DFS, kept tiny and dependency-free. */
function connectedComponents(
  nodeIds: string[],
  edges: GraphEdge[],
): string[][] {
  const adj = new Map<string, Set<string>>();
  for (const id of nodeIds) adj.set(id, new Set());
  for (const e of edges) {
    if (!adj.has(e.source) || !adj.has(e.target)) continue;
    adj.get(e.source)!.add(e.target);
    adj.get(e.target)!.add(e.source);
  }
  const seen = new Set<string>();
  const components: string[][] = [];
  for (const id of nodeIds) {
    if (seen.has(id)) continue;
    const stack = [id];
    const comp: string[] = [];
    while (stack.length) {
      const cur = stack.pop()!;
      if (seen.has(cur)) continue;
      seen.add(cur);
      comp.push(cur);
      for (const nbr of adj.get(cur) || []) if (!seen.has(nbr)) stack.push(nbr);
    }
    if (comp.length) components.push(comp);
  }
  // Largest component first so the ring (when present) gets the prime slot.
  components.sort((a, b) => b.length - a.length);
  return components;
}

/**
 * Layout a single component with dagre LR at the origin. Disconnected components
 * laid out individually avoid the x=0 collapse that comes from a single dagre
 * pass over a partially disconnected graph; the caller translates the result.
 */
function layoutComponent(
  componentIds: string[],
  edges: GraphEdge[],
): { nodes: Node[]; width: number; height: number } {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 22, ranksep: 70, marginx: 0, marginy: 0 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const id of componentIds) g.setNode(id, { width: NODE_W, height: NODE_H });
  for (const e of edges) {
    if (componentIds.includes(e.source) && componentIds.includes(e.target)) {
      g.setEdge(e.source, e.target);
    }
  }
  dagre.layout(g);

  let width = 0;
  let height = 0;
  const nodes: Node[] = componentIds.map((id) => {
    const pos = g.node(id);
    const x = (pos?.x ?? 0) - NODE_W / 2;
    const y = (pos?.y ?? 0) - NODE_H / 2;
    if (x + NODE_W > width) width = x + NODE_W;
    if (y + NODE_H > height) height = y + NODE_H;
    return { id, position: { x, y }, data: {}, type: "default" } as Node;
  });
  return { nodes, width, height };
}

function toFlowNode(
  n: GraphNode & { position: { x: number; y: number } },
  hot: boolean,
): Node {
  const style = TYPE_STYLE[n.type] ?? TYPE_STYLE.unknown;
  return {
    id: n.id,
    position: n.position,
    data: { label: shortLabel(n.id) },
    className: hot ? "graph-node-enter graph-node--ring" : "graph-node-enter",
    style: {
      background: style.bg,
      border: `1px solid ${style.border}`,
      color: style.label,
      borderRadius: 8,
      fontSize: 10,
      fontFamily: "var(--font-geist-mono), monospace",
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
    animated: false,
    className: hot ? "graph-edge-draw graph-edge--ring" : "graph-edge-draw",
    style: {
      stroke: hot ? "#ef4444" : "#3f3f46",
      strokeWidth: hot ? 2 : 1,
      opacity: hot ? 0.95 : 0.55,
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: hot ? "#ef4444" : "#52525b", width: 14, height: 14 },
  };
}

const emptySubscribe = () => () => {};

/** Used until the ResizeObserver reports a real box (first paint, SSR). */
const FALLBACK_FRAME = { w: 800, h: 460 };

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
  const flowRef = useRef<ReactFlowInstance<Node, Edge> | null>(null);
  const [ready, setReady] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);
  const [frame, setFrame] = useState(FALLBACK_FRAME);

  // The row width we pack to is a function of the frame's aspect, so the canvas
  // size has to be measured rather than assumed.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setFrame({ w: width, h: height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [mounted]);

  // Pre-compute layout positions per component so toFlowNode can reuse them.
  const layoutById = useMemo(() => {
    const capped = nodes.length > MAX_NODES ? nodes.slice(nodes.length - MAX_NODES) : nodes;
    const ids = capped.map((n) => n.id);
    const cappedEdges = edges.filter(
      (e) => ids.includes(e.source) && ids.includes(e.target),
    );
    const components = connectedComponents(ids, cappedEdges);
    const map = new Map<string, { x: number; y: number }>();
    // At the minimum legal zoom the frame shows frame/MIN_ZOOM_VIS world pixels;
    // wrapping a row at that width keeps every component framed instead of
    // pushing the row's tail past the card edge.
    const rowLimit = Math.max(NODE_W * 2, frame.w / MIN_ZOOM_VIS - COMPONENT_STRIPE_GAP);
    let cursorX = 0;
    let cursorY = 0;
    let rowHeight = 0;
    for (const comp of components) {
      const { nodes: laidOut, width, height } = layoutComponent(comp, cappedEdges);
      if (cursorX > 0 && cursorX + width > rowLimit) {
        cursorX = 0;
        cursorY += rowHeight + COMPONENT_RANK_GAP;
        rowHeight = 0;
      }
      for (const ln of laidOut) {
        map.set(ln.id, { x: ln.position.x + cursorX, y: ln.position.y + cursorY });
      }
      cursorX += width + COMPONENT_STRIPE_GAP;
      if (height > rowHeight) rowHeight = height;
    }
    return { positions: map, edges: cappedEdges, count: capped.length };
  }, [nodes, edges, frame]);

  useEffect(() => {
    if (!mounted) return;
    const { positions, edges: cappedEdges } = layoutById;
    const positioned = nodes
      .filter((n) => positions.has(n.id))
      .map((n) => ({ ...n, position: positions.get(n.id)! }));
    const hotNodeIds = new Set(
      cappedEdges.filter((edge) => edge.weight >= 3).flatMap((edge) => [edge.source, edge.target]),
    );

    if (layoutTimer.current) clearTimeout(layoutTimer.current);
    layoutTimer.current = setTimeout(() => {
      setFlowNodes(positioned.map((node) => toFlowNode(node, hotNodeIds.has(node.id))));
      setFlowEdges(cappedEdges.map(toFlowEdge));
    }, 250);
    return () => {
      if (layoutTimer.current) clearTimeout(layoutTimer.current);
    };
  }, [mounted, layoutById, nodes, setFlowNodes, setFlowEdges]);

  // Frame the world we just laid out. fitView is deliberately not used: it reads
  // node dimensions that React Flow measures asynchronously, so mid-stream it
  // fitted a stale box and left nodes cropped behind the card's overflow.
  const fitToLayout = useCallback(() => {
    const flow = flowRef.current;
    const box = canvasRef.current?.getBoundingClientRect();
    if (!flow || !box || layoutById.positions.size === 0) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of layoutById.positions.values()) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x + NODE_W > maxX) maxX = p.x + NODE_W;
      if (p.y + NODE_H > maxY) maxY = p.y + NODE_H;
    }
    const zoom = clamp(
      Math.min((box.width - FIT_PADDING * 2) / (maxX - minX), (box.height - FIT_PADDING * 2) / (maxY - minY)),
      MIN_ZOOM_VIS,
      MAX_ZOOM_VIS,
    );
    void flow.setViewport(
      {
        x: (box.width - (maxX - minX) * zoom) / 2 - minX * zoom,
        y: (box.height - (maxY - minY) * zoom) / 2 - minY * zoom,
        zoom,
      },
      { duration: 250 },
    );
  }, [layoutById]);

  // onInit fires exactly once, so mirror it in state to let the fit effect run
  // as soon as an instance exists and again on every layout after that.
  useEffect(() => {
    if (ready) fitToLayout();
  }, [ready, fitToLayout]);

  if (!mounted) {
    return (
      <div className={`flex items-center justify-center text-xs text-muted-foreground ${className ?? ""}`}>
        initializing canvas…
      </div>
    );
  }

  return (
    <div ref={canvasRef} className={className}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onInit={(instance) => {
          flowRef.current = instance;
          setReady(true);
        }}
        minZoom={MIN_ZOOM_VIS}
        maxZoom={MAX_ZOOM_VIS}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: false }}
        colorMode="dark"
      >
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#27272a" />
        <Controls showInteractive={false} className="!bottom-3 !left-3" />
      </ReactFlow>
    </div>
  );
}
