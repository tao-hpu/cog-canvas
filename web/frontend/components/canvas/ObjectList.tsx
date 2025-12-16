'use client';

import { CanvasObject } from '@/lib/types';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

interface ObjectListProps {
  objects: CanvasObject[];
}

const typeColors: Record<string, string> = {
  decision: 'bg-blue-500',
  todo: 'bg-yellow-500',
  key_fact: 'bg-green-500',
  reminder: 'bg-purple-500',
  insight: 'bg-orange-500',
};

export function ObjectList({ objects }: ObjectListProps) {
  if (objects.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p>No objects extracted yet. Start a conversation!</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-3 p-4">
        {objects.map((obj) => (
          <Card key={obj.id} className="overflow-hidden">
            <CardHeader className="p-3 pb-2">
              <div className="flex items-center justify-between">
                <Badge className={typeColors[obj.type]}>
                  {obj.type.replace('_', ' ').toUpperCase()}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {(obj.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            </CardHeader>
            <CardContent className="p-3 pt-0 space-y-2">
              <p className="text-sm font-medium">{obj.content}</p>
              {obj.quote && (
                <>
                  <Separator />
                  <p className="text-xs text-muted-foreground italic">
                    "{obj.quote}"
                  </p>
                </>
              )}
              {obj.context && (
                <p className="text-xs text-muted-foreground">
                  Context: {obj.context}
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </ScrollArea>
  );
}
