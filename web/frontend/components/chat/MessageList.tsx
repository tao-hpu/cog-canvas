'use client';

import { useRef, useEffect } from 'react';
import { Message } from '@/lib/types';
import { MessageItem } from './MessageItem';

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
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
      className="h-full overflow-y-auto px-4 py-6"
    >
      <div className="space-y-6">
        {messages.length === 0 ? (
          <div className="text-center text-muted-foreground py-12">
            Start a conversation...
          </div>
        ) : (
          messages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))
        )}
      </div>
    </div>
  );
}
