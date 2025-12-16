'use client';

import { useState, FormEvent } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Send, Dices, Loader2 } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  isExtracting?: boolean;
}

// Pre-configured test messages for testing CogCanvas extraction features
// All messages revolve around the same project "TaskFlow" (a team collaboration platform)
const TEST_MESSAGES = [
  // === Group 1: Project kickoff & tech stack ===
  "TaskFlow project is officially kicking off. Our goal is to build a lightweight team collaboration platform with core features including a task board, real-time collaboration, and a team calendar.",

  "For the tech stack, I've decided to go with Next.js + TypeScript for the frontend, FastAPI for the backend, and PostgreSQL for the database. The main reasons are team familiarity and the solid balance of performance and dev productivity.",

  "Sarah has finished the UI mockups for TaskFlow. She's going with a dark theme overall and suggests using color-coded labels for task priority: red for urgent, yellow for important, blue for normal.",

  // === Group 2: Development issues ===
  "Ran into a tricky issue today: TaskFlow's board drag-and-drop has serious performance problems on Safari. Very laggy. My initial guess is that React is re-rendering too frequently.",

  "Regarding yesterday's Safari drag performance issue, I found the root cause: every drag was triggering a full board re-render. The fix is to use React.memo on TaskCard components and useMemo to cache the board data.",

  "The Safari issue is fixed now. FPS went from 15 to a stable 60. I also applied similar optimizations to the calendar view, and the overall smoothness improved significantly.",

  // === Group 3: Real-time collaboration feature ===
  "Mike asked whether we should use WebSocket for real-time collaboration. I'm leaning toward WebSocket because: 1) task state needs instant sync, 2) we'll want real-time comments later, 3) Server-Sent Events only supports one-way communication.",

  "The WebSocket technical approach is finalized: Socket.io for the client library, FastAPI's native WebSocket support on the backend. Message format will be JSON with event_type, payload, and timestamp fields.",

  "First version of real-time collaboration is done. Now when multiple people edit tasks simultaneously, state syncs within seconds. But I found an edge case: two people editing the same task title causes conflicts. We need to add an optimistic locking mechanism.",

  // === Group 4: Testing and deployment ===
  `TaskFlow progress update:

Completed features:
- Task board (create, edit, drag-and-drop, archive)
- User auth (login, registration, OAuth)
- Real-time collaboration (WebSocket state sync)
- Basic permission management (admin, member, guest)

Testing status:
- Unit test coverage at 78%
- E2E tests cover all core flows
- Performance testing: single board supports 500 concurrent users

Open items:
- Mobile responsiveness not done yet
- Need to add data export functionality`,

  "TaskFlow staging environment is deployed at staging.taskflow.dev. Please help test it out, focusing on: 1) Is drag-and-drop smooth? 2) Is real-time sync responsive? 3) Does the login flow work correctly? File issues if you find anything.",

  // === Group 5: Post-launch feedback and iteration ===
  "TaskFlow has been live for a week now. User feedback so far: 1) Want keyboard shortcuts, 2) Task search isn't powerful enough, 3) Want to toggle between board view and list view. Logging these for the next sprint.",

  "For the keyboard shortcuts feature request, I've drafted a proposal: Ctrl+N to create task, Ctrl+K for quick search, arrow keys to navigate tasks, Enter to open details, Esc to close modals. Took inspiration from Notion and Linear.",

  "Some users reported that TaskFlow works poorly on slow networks—task operations often fail. I suggest adding an offline queue mechanism: store operations in local IndexedDB first, then auto-sync when connectivity returns. That way it still works offline.",
];

export function getRandomTestMessage(): string {
  const randomIndex = Math.floor(Math.random() * TEST_MESSAGES.length);
  return TEST_MESSAGES[randomIndex];
}

export function ChatInput({ onSendMessage, disabled, isExtracting }: ChatInputProps) {
  const [input, setInput] = useState('');

  console.log('ChatInput render - isExtracting:', isExtracting);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleTestMessage = () => {
    if (!disabled) {
      const testMessage = getRandomTestMessage();
      onSendMessage(testMessage);
    }
  };

  const isDisabled = disabled || isExtracting;

  return (
    <div className="border-t">
      <form onSubmit={handleSubmit} className="flex gap-2 p-4">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isDisabled}
          className="flex-1"
        />
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="outline"
                onClick={handleTestMessage}
                disabled={isDisabled}
              >
                <Dices className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Send random test message</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="submit"
                disabled={isDisabled || !input.trim()}
                className={isExtracting ? "bg-yellow-500 hover:bg-yellow-600 text-black" : ""}
              >
                {disabled ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : isExtracting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{isExtracting ? "Extracting knowledge..." : "Send message"}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </form>
    </div>
  );
}
