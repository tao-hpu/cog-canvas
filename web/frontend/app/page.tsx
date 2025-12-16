'use client';

import { useCallback, useEffect, useState, useRef } from 'react';
import { useChat } from '@/hooks/useChat';
import { useCanvas } from '@/hooks/useCanvas';
import { MessageList } from '@/components/chat/MessageList';
import { ChatInput, getRandomTestMessage } from '@/components/chat/ChatInput';
import { ObjectList } from '@/components/canvas/ObjectList';
import { GraphView } from '@/components/canvas/GraphView';
import { HelpView } from '@/components/canvas/HelpView';
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
  const [sidebarWidth, setSidebarWidth] = useState(560);
  const isResizing = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [graphHeight, setGraphHeight] = useState(0);

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

  // Track extraction phase (between 'done' and 'extraction' events)
  const [isExtracting, setIsExtracting] = useState(false);

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

  // Fetch canvas data and handle graph resizing
  useEffect(() => {
    if (!cogcanvasEnabled) return;

    // --- Canvas Data Fetching ---
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
    const interval = setInterval(fetchCanvasData, 10000); // Refresh every 10s

    // --- ResizeObserver for Graph Height ---
    let observer: ResizeObserver;
    if (graphContainerRef.current) {
      observer = new ResizeObserver((entries) => {
        if (entries[0]) {
          setGraphHeight(entries[0].contentRect.height);
        }
      });
      observer.observe(graphContainerRef.current);
    }

    return () => {
      clearInterval(interval);
      if (graphContainerRef.current && observer) {
        observer.unobserve(graphContainerRef.current);
      }
    };
  }, [cogcanvasEnabled, sessionId, setObjects, setGraphData, setStats, viewMode]); // Added viewMode to dependencies

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
        const stream = await sendMessage(message, sessionId, cogcanvasEnabled);
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
              const data = line.slice(6).trim();

              // Skip empty data lines or [DONE] marker
              if (!data || data === '[DONE]') {
                continue;
              }

              try {
                const parsed = JSON.parse(data);

                // Log non-token events for debugging
                if (parsed.type !== 'token') {
                  console.log('SSE event:', parsed.type, parsed);
                }

                if (parsed.type === 'token') {
                  appendToLast(parsed.content);
                } else if (parsed.type === 'retrieval') {
                  // Handle retrieved context objects
                  console.log('Retrieval results received:', parsed.count);
                  const retrievedObjects = parsed.objects as CanvasObject[];
                  updateLastMessage({ retrievedObjects });
                } else if (parsed.type === 'extracting') {
                  // Extraction phase started
                  console.log('Extraction started...');
                  setLoading(false);
                  setIsExtracting(true);
                } else if (parsed.type === 'extraction') {
                  // Handle extracted canvas objects
                  console.log('Extraction results received');
                  const extractedObjects = parsed.objects as CanvasObject[];
                  updateLastMessage({ extractedObjects });
                } else if (parsed.type === 'done') {
                  // All complete
                  console.log('All complete');
                  setLoading(false);
                  setIsExtracting(false);
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
        setIsExtracting(false);
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

  const handleDiceClick = useCallback(() => {
    const testMessage = getRandomTestMessage();
    handleSendMessage(testMessage);
  }, [handleSendMessage]);

  return (
    <main className="flex h-screen">
      {/* Left: Chat Area */}
      <div className="flex-1 min-w-[400px] flex flex-col">
        <div className="flex-1 overflow-hidden">
          <MessageList messages={messages} isExtracting={isExtracting} onDiceClick={handleDiceClick} />
        </div>
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} isExtracting={isExtracting} />
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

        <div className="flex-1 overflow-hidden" ref={graphContainerRef}>
          {viewMode === 'help' ? (
            <HelpView />
          ) : viewMode === 'list' ? (
            <ObjectList objects={objects} />
          ) : (viewMode === 'graph' && graphData && graphHeight > 0) ? (
            <GraphView data={graphData} width={sidebarWidth} height={graphHeight} />
          ) : (
            null // Render nothing or a loading state if graphHeight is 0 or not graph view
          )}
        </div>
      </div>
    </main>
  );
}