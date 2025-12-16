'use client';

import { useCallback, useEffect, useState, useRef } from 'react';
import { useChat } from '@/hooks/useChat';
import { useCanvas } from '@/hooks/useCanvas';
import { MessageList } from '@/components/chat/MessageList';
import { ChatInput } from '@/components/chat/ChatInput';
import { ObjectList } from '@/components/canvas/ObjectList';
import { GraphView } from '@/components/canvas/GraphView';
import { ControlPanel } from '@/components/control/ControlPanel';
import {
  sendMessage,
  getCanvasObjects,
  getCanvasGraph,
  getCanvasStats,
  clearCanvas as clearCanvasAPI,
} from '@/lib/api';
import { CanvasObject } from '@/lib/types';

export default function Home() {
  // Resizable sidebar
  const [sidebarWidth, setSidebarWidth] = useState(480);
  const isResizing = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isResizing.current = true;
    startX.current = e.clientX;
    startWidth.current = sidebarWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [sidebarWidth]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      const delta = startX.current - e.clientX;
      const newWidth = Math.min(Math.max(startWidth.current + delta, 320), 800);
      setSidebarWidth(newWidth);
    };

    const handleMouseUp = () => {
      isResizing.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);
  const {
    messages,
    isLoading,
    sessionId,
    addMessage,
    appendToLast,
    updateLastMessage,
    setLoading,
  } = useChat();

  const {
    objects,
    graphData,
    stats,
    cogcanvasEnabled,
    viewMode,
    setObjects,
    setGraphData,
    setStats,
    toggleCogcanvas,
    setViewMode,
    clearAll,
  } = useCanvas();

  // Fetch canvas data periodically
  useEffect(() => {
    if (!cogcanvasEnabled) return;

    const fetchCanvasData = async () => {
      try {
        const [objectsData, graphDataRes, statsData] = await Promise.all([
          getCanvasObjects(sessionId),
          getCanvasGraph(sessionId),
          getCanvasStats(sessionId),
        ]);

        setObjects(objectsData);
        setGraphData(graphDataRes);
        setStats(statsData);
      } catch (error) {
        console.error('Error fetching canvas data:', error);
      }
    };

    fetchCanvasData();
    const interval = setInterval(fetchCanvasData, 5000); // Refresh every 5s

    return () => clearInterval(interval);
  }, [cogcanvasEnabled, sessionId, setObjects, setGraphData, setStats]);

  const handleSendMessage = useCallback(
    async (message: string) => {
      // Add user message
      addMessage({
        id: Date.now().toString(),
        role: 'user',
        content: message,
      });

      setLoading(true);

      try {
        const stream = await sendMessage(message, sessionId);
        const reader = stream.getReader();
        const decoder = new TextDecoder();

        // Add empty assistant message
        addMessage({
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: '',
        });

        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');

          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);

              if (data === '[DONE]') {
                continue;
              }

              try {
                const parsed = JSON.parse(data);

                if (parsed.type === 'token') {
                  appendToLast(parsed.content);
                } else if (parsed.type === 'done') {
                  // Response complete, content already accumulated
                  console.log('Stream complete');
                } else if (parsed.type === 'extraction') {
                  // Handle extracted canvas objects
                  const extractedObjects = parsed.objects as CanvasObject[];
                  updateLastMessage({ extractedObjects });
                  if (cogcanvasEnabled) {
                    // Refresh canvas data
                    const [objectsData, graphDataRes, statsData] =
                      await Promise.all([
                        getCanvasObjects(sessionId),
                        getCanvasGraph(sessionId),
                        getCanvasStats(sessionId),
                      ]);
                    setObjects(objectsData);
                    setGraphData(graphDataRes);
                    setStats(statsData);
                  }
                } else if (parsed.type === 'error') {
                  console.error('Server error:', parsed.error);
                  appendToLast(`\n\nError: ${parsed.error}`);
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            }
          }
        }
      } catch (error) {
        console.error('Error sending message:', error);
        addMessage({
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: 'Sorry, there was an error processing your message.',
        });
      } finally {
        setLoading(false);
      }
    },
    [
      sessionId,
      cogcanvasEnabled,
      addMessage,
      appendToLast,
      updateLastMessage,
      setLoading,
      setObjects,
      setGraphData,
      setStats,
    ]
  );

  const handleClearCanvas = useCallback(async () => {
    try {
      await clearCanvasAPI(sessionId);
      clearAll();
    } catch (error) {
      console.error('Error clearing canvas:', error);
    }
  }, [sessionId, clearAll]);

  return (
    <main className="flex h-screen">
      {/* Left: Chat Area */}
      <div className="flex-1 min-w-[400px] flex flex-col">
        <div className="flex-1 overflow-hidden">
          <MessageList messages={messages} />
        </div>
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
      </div>

      {/* Resize Handle */}
      <div
        className="w-1 bg-border hover:bg-primary/50 cursor-col-resize transition-colors flex-shrink-0"
        onMouseDown={handleMouseDown}
      />

      {/* Right: Canvas Sidebar */}
      <div
        className="flex flex-col flex-shrink-0 border-l"
        style={{ width: sidebarWidth }}
      >
        <ControlPanel
          cogcanvasEnabled={cogcanvasEnabled}
          onToggleCogcanvas={toggleCogcanvas}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          onClearCanvas={handleClearCanvas}
          stats={stats}
        />

        <div className="flex-1 overflow-hidden">
          {viewMode === 'list' ? (
            <ObjectList objects={objects} />
          ) : (
            graphData && <GraphView data={graphData} width={sidebarWidth} />
          )}
        </div>
      </div>
    </main>
  );
}
