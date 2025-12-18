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
  setSessionId: (sessionId: string) => void;
}

// Get initial session ID
const getInitialSessionId = () => {
  if (typeof window === 'undefined') return crypto.randomUUID();
  return sessionStorage.getItem('sessionId') || crypto.randomUUID();
};

const initialSessionId = getInitialSessionId();

export const useChat = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,
  sessionId: initialSessionId,

  addMessage: (msg) =>
    set((state) => ({
      messages: [...state.messages, msg],
    })),

  appendToLast: (content) =>
    set((state) => {
      if (state.messages.length === 0) return state;
      const messages = [...state.messages];
      const lastMsg = messages[messages.length - 1];
      if (!lastMsg) return state;
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
        messages[messages.length - 1] = {
          ...lastMsg,
          ...updates,
        };
      }
      return { messages };
    }),

  setLoading: (loading) => set({ isLoading: loading }),

  clearMessages: () => set({ messages: [] }),

  setSessionId: (sessionId) =>
    set(() => {
      console.log(`Switched to session ${sessionId}`);
      return { sessionId, messages: [] };
    }),
}));

// Persist session ID
if (typeof window !== 'undefined') {
  useChat.subscribe((state) => {
    sessionStorage.setItem('sessionId', state.sessionId);
  });
}
