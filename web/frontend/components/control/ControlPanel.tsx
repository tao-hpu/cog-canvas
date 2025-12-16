'use client';

import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Trash2, BarChart3 } from 'lucide-react';
import { CanvasStats } from '@/lib/types';

interface ControlPanelProps {
  cogcanvasEnabled: boolean;
  onToggleCogcanvas: () => void;
  viewMode: 'list' | 'graph';
  onViewModeChange: (mode: 'list' | 'graph') => void;
  onClearCanvas: () => void;
  stats: CanvasStats | null;
}

export function ControlPanel({
  cogcanvasEnabled,
  onToggleCogcanvas,
  viewMode,
  onViewModeChange,
  onClearCanvas,
  stats,
}: ControlPanelProps) {
  return (
    <div className="p-4 space-y-4 border-b">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Canvas</h2>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClearCanvas}
          title="Clear Canvas"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex items-center justify-between">
        <label htmlFor="cogcanvas-toggle" className="text-sm font-medium">
          CogCanvas Enabled
        </label>
        <Switch
          id="cogcanvas-toggle"
          checked={cogcanvasEnabled}
          onCheckedChange={onToggleCogcanvas}
        />
      </div>

      <Separator />

      <Tabs value={viewMode} onValueChange={(v) => onViewModeChange(v as 'list' | 'graph')}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="list">List</TabsTrigger>
          <TabsTrigger value="graph">Graph</TabsTrigger>
        </TabsList>
      </Tabs>

      {stats && (
        <>
          <Separator />
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Statistics</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg bg-muted p-2">
                <p className="text-xs text-muted-foreground">Total</p>
                <p className="text-lg font-bold">{stats.total_objects}</p>
              </div>
              <div className="rounded-lg bg-muted p-2">
                <p className="text-xs text-muted-foreground">Avg Confidence</p>
                <p className="text-lg font-bold">
                  {((stats.avg_confidence ?? 0) * 100).toFixed(0)}%
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-1">
              {stats.by_type && Object.entries(stats.by_type).map(([type, count]) => (
                <Badge key={type} variant="outline" className="text-xs">
                  {type}: {count}
                </Badge>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
