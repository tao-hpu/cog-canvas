'use client';

import { useEffect, useRef, useMemo, useCallback, useState } from 'react';
import { GraphData, GraphNode } from '@/lib/types';
import { Plus, Minus, Maximize2 } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';

interface GraphViewProps {
  data: GraphData;
  width: number;
  height: number;
}

const typeColors: Record<string, string> = {
  decision: '#3b82f6',
  todo: '#eab308',
  key_fact: '#22c55e',
  reminder: '#a855f7',
  insight: '#f97316',
};

export function GraphView({ data, width, height }: GraphViewProps) {
  const graphRef = useRef<any>();
  const isInitialized = useRef(false);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

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
      // Increase repulsion force so nodes spread out more
      graphRef.current.d3Force('charge').strength(-300);
      graphRef.current.d3Force('link').distance(100);

      // Add center force to keep graph centered
      graphRef.current.d3Force('center', null); // Remove default center

      // Initial fit to view after a short delay for layout to stabilize
      setTimeout(() => {
        if (graphRef.current) {
          graphRef.current.zoomToFit(400, 80);
        }
      }, 500);

      isInitialized.current = true;
    } else if (graphRef.current) {
      // Recalculate fit on data/size changes
      setTimeout(() => {
        if (graphRef.current) {
          graphRef.current.zoomToFit(400, 80);
        }
      }, 300);
    }
  }, [stableGraphData, width, height]);

  // Zoom handlers
  const handleZoomIn = useCallback(() => {
    console.log('=== ZOOM IN CLICKED ===');
    console.log('graphRef.current:', graphRef.current);
    if (graphRef.current) {
      const fg = graphRef.current;
      console.log('fg object:', fg);
      console.log('fg.zoom:', fg.zoom);
      console.log('typeof fg.zoom:', typeof fg.zoom);
      try {
        const currentZoom = fg.zoom();
        console.log('Current zoom level:', currentZoom);
        const newZoom = currentZoom * 1.5;
        console.log('Setting new zoom to:', newZoom);
        fg.zoom(newZoom, 300);
        console.log('Zoom set successfully');
      } catch (err) {
        console.error('Zoom error:', err);
      }
    } else {
      console.log('graphRef.current is null/undefined');
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    console.log('=== ZOOM OUT CLICKED ===');
    if (graphRef.current) {
      try {
        const fg = graphRef.current;
        const currentZoom = fg.zoom();
        console.log('Current zoom:', currentZoom);
        fg.zoom(currentZoom * 0.67, 300);
      } catch (err) {
        console.error('Zoom error:', err);
      }
    }
  }, []);

  const handleFitView = useCallback(() => {
    console.log('=== FIT VIEW CLICKED ===');
    if (graphRef.current) {
      try {
        graphRef.current.zoomToFit(400, 60);
        console.log('zoomToFit called');
      } catch (err) {
        console.error('Fit view error:', err);
      }
    }
  }, []);

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

  const linkCanvasObject = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const start = link.source;
    const end = link.target;
    if (typeof start !== 'object' || typeof end !== 'object') return;

    // Draw the line
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1 / globalScale;
    ctx.stroke();

    // Draw relation label at midpoint
    const midX = (start.x + end.x) / 2;
    const midY = (start.y + end.y) / 2;
    const fontSize = 10 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#888';
    ctx.fillText(link.relation || '', midX, midY);
  }, []);

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node as GraphNode);
  }, []);

  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  if (!data || data.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p>No graph data available</p>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-background relative">
      <ForceGraph2D
        ref={graphRef}
        graphData={stableGraphData}
        nodeId="id"
        nodeLabel="content"
        nodeColor={(node: any) => typeColors[node.type] || '#666'}
        linkLabel="relation"
        linkCanvasObject={linkCanvasObject}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        nodeCanvasObject={nodeCanvasObject}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
        cooldownTicks={100}
        warmupTicks={50}
        width={width}
        height={height}
      />

      {/* Zoom Controls */}
      <div className="absolute bottom-3 left-3 flex flex-col gap-1 z-50 pointer-events-auto">
        <button
          type="button"
          className="h-9 w-9 flex items-center justify-center bg-zinc-800 hover:bg-zinc-600 border border-zinc-500 rounded text-white cursor-pointer"
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleZoomIn();
          }}
          title="Zoom In"
        >
          <Plus className="h-5 w-5" />
        </button>
        <button
          type="button"
          className="h-9 w-9 flex items-center justify-center bg-zinc-800 hover:bg-zinc-600 border border-zinc-500 rounded text-white cursor-pointer"
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleZoomOut();
          }}
          title="Zoom Out"
        >
          <Minus className="h-5 w-5" />
        </button>
        <button
          type="button"
          className="h-9 w-9 flex items-center justify-center bg-zinc-800 hover:bg-zinc-600 border border-zinc-500 rounded text-white cursor-pointer"
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleFitView();
          }}
          title="Fit to View"
        >
          <Maximize2 className="h-5 w-5" />
        </button>
      </div>

      {/* Node Detail Panel */}
      {selectedNode && (
        <div className="absolute top-4 right-4 w-80 bg-background/95 backdrop-blur-xl border border-border/50 rounded-xl p-5 shadow-2xl z-10">
          <div className="flex items-center justify-between mb-4">
            <span
              className="text-xs font-semibold px-3 py-1.5 rounded-md text-white shadow-sm"
              style={{ backgroundColor: typeColors[selectedNode.type] || '#666' }}
            >
              {selectedNode.type.replace('_', ' ').toUpperCase()}
            </span>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-muted-foreground hover:text-foreground transition-colors text-lg font-light leading-none w-6 h-6 flex items-center justify-center rounded-md hover:bg-accent"
              aria-label="Close details"
            >
              ✕
            </button>
          </div>

          <div className="space-y-3.5">
            <div>
              <span className="text-muted-foreground text-xs font-medium uppercase tracking-wide block mb-1.5">
                Content
              </span>
              <p className="text-foreground text-sm leading-relaxed">{selectedNode.content}</p>
            </div>

            {selectedNode.quote && (
              <div>
                <span className="text-muted-foreground text-xs font-medium uppercase tracking-wide block mb-1.5">
                  Quote
                </span>
                <p className="text-foreground/90 text-sm italic leading-relaxed border-l-2 border-border/50 pl-3">
                  "{selectedNode.quote}"
                </p>
              </div>
            )}

            {selectedNode.context && (
              <div>
                <span className="text-muted-foreground text-xs font-medium uppercase tracking-wide block mb-1.5">
                  Context
                </span>
                <p className="text-foreground/80 text-sm leading-relaxed">{selectedNode.context}</p>
              </div>
            )}

            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground pt-3 border-t border-border/50">
              <span className="flex items-center gap-1.5">
                <span className="text-foreground/60">Confidence:</span>
                <span className="text-foreground">{(selectedNode.confidence * 100).toFixed(0)}%</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="text-foreground/60">Turn:</span>
                <span className="text-foreground">{selectedNode.turn_id}</span>
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
