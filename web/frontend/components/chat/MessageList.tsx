'use client';

import { useRef, useEffect } from 'react';
import { Message } from '@/lib/types';
import { MessageItem } from './MessageItem';
import { MessageSquare, Sparkles, Dices } from 'lucide-react';

interface MessageListProps {
  messages: Message[];
  isExtracting?: boolean;
  onDiceClick?: () => void;
}

export function MessageList({ messages, isExtracting, onDiceClick }: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastMsgContent = messages[messages.length - 1]?.content;

  // Auto-scroll on new messages or content updates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length, lastMsgContent]);

  return (
    <div
      ref={scrollRef}
      className="h-full overflow-y-auto px-4 py-6 chat-scrollbar"
    >
      <div className="space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
            <div className="relative mb-6">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center">
                <MessageSquare className="w-10 h-10 text-primary/60" />
              </div>
              <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-yellow-500/20 flex items-center justify-center">
                <Sparkles className="w-3 h-3 text-yellow-500" />
              </div>
            </div>
            <h3 className="text-xl font-semibold text-foreground mb-2">Start a Conversation</h3>
            <p className="text-muted-foreground text-sm max-w-[280px] mb-4">
              Chat naturally and CogCanvas will automatically extract knowledge into a structured graph.
            </p>
            <button
              onClick={onDiceClick}
              className="flex items-center gap-2 text-xs text-muted-foreground/60 bg-muted/50 px-3 py-1.5 rounded-full hover:bg-muted hover:text-muted-foreground transition-colors cursor-pointer"
            >
              <Dices className="w-3 h-3" />
              <span>Click here for a demo message</span>
            </button>
          </div>
        ) : (
          messages.map((message, index) => (
            <MessageItem
              key={message.id}
              message={message}
              isExtracting={isExtracting && index === messages.length - 1 && message.role === 'assistant'}
            />
          ))
        )}
      </div>
    </div>
  );
}
