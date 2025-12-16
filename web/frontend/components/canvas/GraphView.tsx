'use client';

import { useEffect, useRef, useMemo, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { GraphData } from '@/lib/types';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
});

interface GraphViewProps {
  data: GraphData;
  width?: number;
}

const typeColors: Record<string, string> = {
  decision: '#3b82f6',
  todo: '#eab308',
  key_fact: '#22c55e',
  reminder: '#a855f7',
  insight: '#f97316',
};

export function GraphView({ data, width = 480 }: GraphViewProps) {
  const graphRef = useRef<any>();
  const isInitialized = useRef(false);

  // Stabilize graph data reference - only update when node/link IDs change
  const stableGraphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    return {
      nodes: data.nodes.map(node => ({ ...node })),
      links: data.links.map(link => ({ ...link })),
    };
  }, [
    // Only re-create when the set of node IDs or link structure changes
    data?.nodes.map(n => n.id).join(','),
    data?.links.map(l => `${l.source}-${l.target}`).join(','),
  ]);

  useEffect(() => {
    if (graphRef.current && !isInitialized.current) {
      graphRef.current.d3Force('charge').strength(-100);
      graphRef.current.d3Force('link').distance(50);
      isInitialized.current = true;
    }
  }, [stableGraphData]);

  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.content?.slice(0, 20) || '';
    const fontSize = 12 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;

    ctx.fillStyle = typeColors[node.type] || '#666';
    ctx.beginPath();
    ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI);
    ctx.fill();

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff';
    ctx.fillText(label, node.x, node.y + 10);
  }, []);

  if (!data || data.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p>No graph data available</p>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-background">
      <ForceGraph2D
        ref={graphRef}
        graphData={stableGraphData}
        nodeId="id"
        nodeLabel="content"
        nodeColor={(node: any) => typeColors[node.type] || '#666'}
        linkLabel="relation"
        linkColor={() => '#444'}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        nodeCanvasObject={nodeCanvasObject}
        cooldownTicks={100}
        warmupTicks={50}
        width={width - 16}
        height={600}
      />
    </div>
  );
}
