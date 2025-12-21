'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { HelpCircle, Zap, Eye, MousePointerClick } from 'lucide-react';

export function HelpView() {
  return (
    <div className="p-4 space-y-4 h-full overflow-y-auto chat-scrollbar">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-500" />
            Quick Start
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm space-y-3">
          <div className="flex gap-3">
            <div className="bg-primary/10 p-2 rounded-full h-fit">
              <Eye className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="font-medium">1. Monitor</p>
              <p className="text-muted-foreground">Chat normally. CogCanvas automatically extracts knowledge into the sidebar.</p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="bg-primary/10 p-2 rounded-full h-fit">
              <MousePointerClick className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="font-medium">2. Explore</p>
              <p className="text-muted-foreground">Switch to <strong>Graph View</strong> to see connections between decisions and tasks.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-blue-500" />
            Object Legend
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Badge className="bg-blue-500 hover:bg-blue-600 w-24 justify-center">DECISION</Badge>
            <span className="text-sm text-muted-foreground text-right">Explicit choices made</span>
          </div>
          <div className="flex items-center justify-between">
            <Badge className="bg-yellow-500 hover:bg-yellow-600 w-24 justify-center">TODO</Badge>
            <span className="text-sm text-muted-foreground text-right">Pending action items</span>
          </div>
          <div className="flex items-center justify-between">
            <Badge className="bg-green-500 hover:bg-green-600 w-24 justify-center">KEY FACT</Badge>
            <span className="text-sm text-muted-foreground text-right">Important context</span>
          </div>
          <div className="flex items-center justify-between">
            <Badge className="bg-purple-500 hover:bg-purple-600 w-24 justify-center">REMINDER</Badge>
            <span className="text-sm text-muted-foreground text-right">Constraints & prefs</span>
          </div>
          <div className="flex items-center justify-between">
            <Badge className="bg-orange-500 hover:bg-orange-600 w-24 justify-center">INSIGHT</Badge>
            <span className="text-sm text-muted-foreground text-right">Learned patterns</span>
          </div>
          <div className="flex items-center justify-between">
            <Badge className="bg-amber-700 hover:bg-amber-800 w-24 justify-center">PERSON ATTRIBUTE</Badge>
            <span className="text-sm text-muted-foreground text-right">Attributes of a person</span>
          </div>
          <div className="flex items-center justify-between">
            <Badge className="bg-cyan-500 hover:bg-cyan-600 w-24 justify-center">EVENT</Badge>
            <span className="text-sm text-muted-foreground text-right">Key actions or occurrences</span>
          </div>
          <div className="flex items-center justify-between">
            <Badge className="bg-gray-500 hover:bg-gray-600 w-24 justify-center">RELATIONSHIP</Badge>
            <span className="text-sm text-muted-foreground text-right">Connections between objects</span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="w-5 h-5 text-purple-500" />
            Graph Relationships
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm space-y-3">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 font-medium">
              <span className="text-xs font-mono bg-muted px-1 rounded">references</span>
              <span className="h-px bg-border flex-1"></span>
            </div>
            <p className="text-muted-foreground text-xs pl-2">Semantic connection or direct mention. (e.g., "Node A relates to Node B")</p>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 font-medium">
              <span className="text-xs font-mono bg-muted px-1 rounded">leads_to</span>
              <span className="h-px bg-border flex-1"></span>
            </div>
            <p className="text-muted-foreground text-xs pl-2">Causal flow: this object leads to another. (e.g., Node A → Node B)</p>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 font-medium">
              <span className="text-xs font-mono bg-muted px-1 rounded">caused_by</span>
              <span className="h-px bg-border flex-1"></span>
            </div>
            <p className="text-muted-foreground text-xs pl-2">Causal origin: this object was caused by another. (e.g., Node A ← Node B)</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <Eye className="w-5 h-5 text-green-500" /> {/* Changed icon to Eye for "Details" */}
            Node Details Explained
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm space-y-3">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 font-medium">
              <span className="text-xs font-mono bg-muted px-1 rounded">Content</span>
            </div>
            <p className="text-muted-foreground text-xs pl-2">The distilled, self-contained essence of the cognitive object. (What the AI remembers)</p>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 font-medium">
              <span className="text-xs font-mono bg-muted px-1 rounded">Quote</span>
            </div>
            <p className="text-muted-foreground text-xs pl-2">The verbatim excerpt from the original conversation that grounds this object. (Proof of existence)</p>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 font-medium">
              <span className="text-xs font-mono bg-muted px-1 rounded">Context</span>
            </div>
            <p className="text-muted-foreground text-xs pl-2">The Extraction LLM's explanation of why this object was extracted. (AI's reasoning/motivation)</p>
          </div>
        </CardContent>
      </Card>
      
      <div className="text-xs text-center text-muted-foreground pt-4 pb-2">
        <a
          href="https://github.com/tao-hpu/cog-canvas"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-primary transition-colors underline"
        >
          GitHub: tao-hpu/cog-canvas
        </a>
      </div>
    </div>
  );
}
