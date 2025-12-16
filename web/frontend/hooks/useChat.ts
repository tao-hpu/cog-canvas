import { create } from 'zustand';
import { Message } from '@/lib/types';

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  sessionId: string;
  addMessage: (msg: Message) => void;
  appendToLast: (content: string) => void;
  updateLastMessage: (updates: Partial<Message>) => void;
  setLoading: (loading: boolean) => void;
  clearMessages: () => void;
}

export const useChat = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,
  sessionId: typeof window !== 'undefined'
    ? (sessionStorage.getItem('sessionId') || crypto.randomUUID())
    : crypto.randomUUID(),

  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),

  appendToLast: (content) =>
    set((state) => {
      if (state.messages.length === 0) return state;
      const messages = [...state.messages];
      const lastMsg = messages[messages.length - 1];
      // Create new object to trigger React re-render
      messages[messages.length - 1] = {
        ...lastMsg,
        content: lastMsg.content + content,
      };
      return { messages };
    }),

  updateLastMessage: (updates) =>
    set((state) => {
      const messages = [...state.messages];
      const lastMsg = messages[messages.length - 1];
      if (lastMsg) {
        Object.assign(lastMsg, updates);
      }
      return { messages };
    }),

  setLoading: (loading) => set({ isLoading: loading }),

  clearMessages: () => set({ messages: [] }),
}));

// Persist session ID
if (typeof window !== 'undefined') {
  useChat.subscribe((state) => {
    sessionStorage.setItem('sessionId', state.sessionId);
  });
}
