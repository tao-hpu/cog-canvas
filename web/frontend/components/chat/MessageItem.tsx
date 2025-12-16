'use client';

import { Message } from '@/lib/types';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { User, Bot, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState } from 'react';

interface MessageItemProps {
  message: Message;
  isExtracting?: boolean;
}

const typeColors: Record<string, { bg: string, text: string }> = {
  decision: { bg: 'bg-blue-500/10', text: 'text-blue-500' },
  todo: { bg: 'bg-yellow-500/10', text: 'text-yellow-500' },
  key_fact: { bg: 'bg-green-500/10', text: 'text-green-500' },
  reminder: { bg: 'bg-purple-500/10', text: 'text-purple-500' },
  insight: { bg: 'bg-orange-500/10', text: 'text-orange-500' },
};

export function MessageItem({ message, isExtracting }: MessageItemProps) {
  const isUser = message.role === 'user';
  const [isRetrievalExpanded, setIsRetrievalExpanded] = useState(false);
  const [isExtractionExpanded, setIsExtractionExpanded] = useState(false);

  return (
    <div
      className={cn(
        'flex gap-3',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      {!isUser && (
        <div className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground",
          isExtracting && "animate-pulse"
        )}>
          <Bot className="h-4 w-4" />
        </div>
      )}

      <div className={cn('flex flex-col gap-2 max-w-[75%]', isUser && 'items-end')}>
        <div
          className={cn(
            'rounded-lg px-4 py-2',
            isUser
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-foreground',
            isExtracting && 'animate-pulse'
          )}
        >
          <div className="prose prose-sm dark:prose-invert max-w-none break-words prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-table:my-2 prose-pre:my-2 prose-pre:bg-zinc-800 prose-pre:text-zinc-100 prose-code:text-pink-500 prose-code:before:content-none prose-code:after:content-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        </div>

        {/* Retrieved Objects - Show before response */}
        {message.retrievedObjects && message.retrievedObjects.length > 0 && (
          <div className="w-full rounded-md border border-border/50 bg-muted/30 overflow-hidden">
            <button
              onClick={() => setIsRetrievalExpanded(!isRetrievalExpanded)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:bg-muted/50 transition-colors cursor-pointer"
            >
              <span className="flex items-center gap-2">
                {isRetrievalExpanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                <span className="font-medium">
                  Retrieved {message.retrievedObjects.length} object{message.retrievedObjects.length !== 1 ? 's' : ''} from Canvas
                </span>
              </span>
              <span className="text-[10px] opacity-60">
                {isRetrievalExpanded ? 'Click to collapse' : 'Click to expand'}
              </span>
            </button>

            {isRetrievalExpanded && (
              <div className="px-3 pb-3 space-y-2">
                {message.retrievedObjects.map((obj) => {
                  const colors = typeColors[obj.type] || typeColors.key_fact;
                  return (
                    <div
                      key={obj.id}
                      className={cn(
                        'rounded-md p-2 border border-border/30',
                        colors.bg
                      )}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className={cn('text-[10px] font-semibold uppercase tracking-wide', colors.text)}>
                          {obj.type.replace('_', ' ')}
                        </span>
                        <div className="flex items-center gap-1">
                          <span className="text-[9px] px-1 py-0.5 rounded bg-background/50 text-muted-foreground font-mono">
                            T{obj.turn_id}
                          </span>
                          <span className="text-[9px] px-1 py-0.5 rounded bg-background/50 text-muted-foreground font-mono">
                            {(obj.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <p className="text-[11px] text-foreground/90 leading-relaxed">
                        {obj.content}
                      </p>
                      {obj.quote && (
                        <p className="text-[10px] text-muted-foreground italic mt-1 pl-2 border-l-2 border-border/50">
                          {obj.quote}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Extracting indicator */}
        {isExtracting && (
          <div className="flex items-center gap-2 text-xs text-yellow-500">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>Extracting knowledge...</span>
          </div>
        )}

        {/* Extracted Objects - Show after response */}
        {message.extractedObjects && message.extractedObjects.length > 0 && (
          <div className="w-full rounded-md border border-green-500/30 bg-green-500/5 overflow-hidden">
            <button
              onClick={() => setIsExtractionExpanded(!isExtractionExpanded)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:bg-green-500/10 transition-colors cursor-pointer"
            >
              <span className="flex items-center gap-2">
                {isExtractionExpanded ? (
                  <ChevronDown className="h-3 w-3 text-green-600" />
                ) : (
                  <ChevronRight className="h-3 w-3 text-green-600" />
                )}
                <span className="font-medium text-green-600">
                  Extracted {message.extractedObjects.length} new object{message.extractedObjects.length !== 1 ? 's' : ''}
                </span>
              </span>
              <span className="text-[10px] opacity-60">
                {isExtractionExpanded ? 'Click to collapse' : 'Click to expand'}
              </span>
            </button>

            {isExtractionExpanded && (
              <div className="px-3 pb-3 flex flex-wrap gap-2">
                {message.extractedObjects.map((obj) => {
                  const colors = typeColors[obj.type] || typeColors.key_fact;
                  const truncatedContent = obj.content.length > 30
                    ? obj.content.substring(0, 30) + '...'
                    : obj.content;
                  return (
                    <div
                      key={obj.id}
                      className={cn(
                        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 border',
                        colors.bg,
                        'border-green-500/20'
                      )}
                      title={obj.content}
                    >
                      <span className={cn('text-[10px] font-semibold uppercase tracking-wide', colors.text)}>
                        {obj.type.replace('_', ' ')}:
                      </span>
                      <span className="text-[10px] text-foreground/80">
                        {truncatedContent}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}
