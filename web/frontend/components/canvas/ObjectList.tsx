'use client';

import { useState } from 'react';
import { CanvasObject } from '@/lib/types';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface ObjectListProps {
  objects: CanvasObject[];
}

const typeColors: Record<string, { bg: string, text: string, border: string }> = {
  decision: { bg: 'bg-blue-500', text: 'text-blue-500', border: 'border-blue-500' },
  todo: { bg: 'bg-yellow-500', text: 'text-yellow-500', border: 'border-yellow-500' },
  key_fact: { bg: 'bg-green-500', text: 'text-green-500', border: 'border-green-500' },
  reminder: { bg: 'bg-purple-500', text: 'text-purple-500', border: 'border-purple-500' },
  insight: { bg: 'bg-orange-500', text: 'text-orange-500', border: 'border-orange-500' },
};

export function ObjectList({ objects }: ObjectListProps) {
  const [activeTab, setActiveTab] = useState<string>('all');
  
  // Calculate counts for each tab
  const tabCounts = {
    all: objects.length,
    decision: objects.filter(obj => obj.type === 'decision').length,
    todo: objects.filter(obj => obj.type === 'todo').length,
    key_fact: objects.filter(obj => obj.type === 'key_fact').length,
    reminder: objects.filter(obj => obj.type === 'reminder').length,
    insight: objects.filter(obj => obj.type === 'insight').length,
  };

  if (objects.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p>No objects extracted yet. Start a conversation!</p>
      </div>
    );
  }

  // Filter based on active tab
  const filteredObjects = objects.filter((obj) => {
    if (activeTab === 'all') return true;
    return obj.type === activeTab;
  });

  // Sort: Newest first (Reverse chronological)
  const sortedObjects = [...filteredObjects].reverse();

  return (
    <div className="flex flex-col h-full bg-muted/10"> {/* Subtle background */}
      <div className="px-4 py-2 border-b bg-background z-10">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="w-full justify-start overflow-x-auto h-9 no-scrollbar bg-transparent p-0 gap-1">
            <TabsTrigger value="all" className="text-xs px-3 h-7 data-[state=active]:bg-primary/10 data-[state=active]:text-primary rounded-full border border-transparent data-[state=active]:border-primary/20">All ({tabCounts.all})</TabsTrigger>
            <TabsTrigger value="decision" className="text-xs px-3 h-7 data-[state=active]:bg-blue-500/10 data-[state=active]:text-blue-500 rounded-full border border-transparent data-[state=active]:border-blue-500/20">Dec ({tabCounts.decision})</TabsTrigger>
            <TabsTrigger value="todo" className="text-xs px-3 h-7 data-[state=active]:bg-yellow-500/10 data-[state=active]:text-yellow-500 rounded-full border border-transparent data-[state=active]:border-yellow-500/20">Todo ({tabCounts.todo})</TabsTrigger>
            <TabsTrigger value="key_fact" className="text-xs px-3 h-7 data-[state=active]:bg-green-500/10 data-[state=active]:text-green-500 rounded-full border border-transparent data-[state=active]:border-green-500/20">Fact ({tabCounts.key_fact})</TabsTrigger>
            <TabsTrigger value="reminder" className="text-xs px-3 h-7 data-[state=active]:bg-purple-500/10 data-[state=active]:text-purple-500 rounded-full border border-transparent data-[state=active]:border-purple-500/20">Rem ({tabCounts.reminder})</TabsTrigger>
            <TabsTrigger value="insight" className="text-xs px-3 h-7 data-[state=active]:bg-orange-500/10 data-[state=active]:text-orange-500 rounded-full border border-transparent data-[state=active]:border-orange-500/20">Ins ({tabCounts.insight})</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-4 chat-scrollbar">
        <div className="space-y-3 py-4">
          {sortedObjects.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground opacity-50">
               <p className="text-sm">No items found</p>
            </div>
          ) : (
            sortedObjects.map((obj) => {
               const colors = typeColors[obj.type] || typeColors.key_fact;
               return (
                <div 
                  key={obj.id} 
                  className="group relative bg-card rounded-xl p-3 shadow-sm border border-border/50 hover:shadow-md hover:border-border transition-all duration-200 animate-in fade-in slide-in-from-top-2"
                >
                  {/* Left Color Accent */}
                  <div className={`absolute left-0 top-3 bottom-3 w-1 rounded-r-full ${colors.bg} opacity-80`} />

                  <div className="pl-3">
                    {/* Header: Type on left, Turn + Confidence tags on right */}
                    <div className="flex items-center justify-between mb-1.5">
                      <span className={`text-[10px] font-bold uppercase tracking-wider ${colors.text}`}>
                        {obj.type.replace('_', ' ')}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700/50 text-zinc-400 font-mono">
                          T{obj.turn_id}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700/50 text-zinc-400 font-mono">
                          {(obj.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    {/* Content */}
                    <p className="text-[13px] text-card-foreground font-medium leading-normal mb-2">
                      {obj.content}
                    </p>

                    {/* Footer: Quote / Context */}
                    {(obj.quote || obj.context) && (
                      <div className="space-y-1.5">
                        {obj.quote && (
                          <div className="text-[11px] text-muted-foreground bg-muted/50 p-1.5 rounded border border-border/50 italic leading-relaxed">
                            <span className="opacity-30 not-italic mr-1">❝</span>
                            {obj.quote}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
