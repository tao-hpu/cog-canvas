'use client';

import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Trash2, MessageSquareX, RotateCcw, HelpCircle } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { CanvasStats } from '@/lib/types';

interface ControlPanelProps {
  cogcanvasEnabled: boolean;
  onToggleCogcanvas: () => void;
  viewMode: 'list' | 'graph' | 'help';
  onViewModeChange: (mode: 'list' | 'graph' | 'help') => void;
  onClearCanvas: () => void;
  onClearChat: () => void;
  onClearAll: () => void;
  stats: CanvasStats | null;
}

export function ControlPanel({
  cogcanvasEnabled,
  onToggleCogcanvas,
  viewMode,
  onViewModeChange,
  onClearCanvas,
  onClearChat,
  onClearAll,
  stats,
}: ControlPanelProps) {
  return (
    <div className="p-4 space-y-4 border-b">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">CogCanvas</h2>
        <div className="flex items-center gap-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClearChat}
                >
                  <MessageSquareX className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Clear chat history</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClearCanvas}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Clear canvas memory</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClearAll}
                  className="text-destructive hover:text-destructive"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Reset all (chat + canvas)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <label htmlFor="cogcanvas-toggle" className="text-sm font-medium cursor-pointer">
            CogCanvas Enabled
          </label>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-4 w-4 text-muted-foreground hover:text-foreground cursor-help transition-colors" />
              </TooltipTrigger>
              <TooltipContent className="max-w-[240px] text-xs">
                <p><span className="font-semibold text-primary">ON:</span> Extracts artifacts & retrieves context.</p>
                <p className="mt-1"><span className="font-semibold text-muted-foreground">OFF:</span> Standard chat without memory.</p>
                {stats && (
                  <p className="mt-2 text-muted-foreground">
                    <span className="font-semibold">Total Objects:</span> {stats.total_objects}<br/>
                    <span className="font-semibold">Avg Confidence:</span> {((stats.avg_confidence ?? 0) * 100).toFixed(0)}%
                  </p>
                )}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Switch
          id="cogcanvas-toggle"
          checked={cogcanvasEnabled}
          onCheckedChange={onToggleCogcanvas}
        />
      </div>

      <Tabs value={viewMode} onValueChange={(v) => onViewModeChange(v as 'list' | 'graph' | 'help')}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="list">List</TabsTrigger>
          <TabsTrigger value="graph">Graph</TabsTrigger>
          <TabsTrigger value="help">Guide</TabsTrigger>
        </TabsList>
      </Tabs>
    </div>
  );
}
