import { create } from 'zustand';
import { Message } from '@/lib/types';

// Maximum number of messages to keep in localStorage
const MAX_MESSAGES = 100;

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

// Helper function to get storage key for messages
const getMessagesStorageKey = (sessionId: string) => `chat_messages_${sessionId}`;

// Helper function to load messages from localStorage
const loadMessagesFromStorage = (sessionId: string): Message[] => {
  if (typeof window === 'undefined') return [];

  try {
    const key = getMessagesStorageKey(sessionId);
    const stored = localStorage.getItem(key);
    if (stored) {
      const messages = JSON.parse(stored) as Message[];
      console.log(`Loaded ${messages.length} messages from localStorage for session ${sessionId}`);
      return messages;
    }
  } catch (error) {
    console.error('Error loading messages from localStorage:', error);
  }

  return [];
};

// Helper function to save messages to localStorage
const saveMessagesToStorage = (sessionId: string, messages: Message[]) => {
  if (typeof window === 'undefined') return;

  try {
    const key = getMessagesStorageKey(sessionId);
    // Limit the number of messages to prevent localStorage overflow
    const messagesToSave = messages.slice(-MAX_MESSAGES);
    localStorage.setItem(key, JSON.stringify(messagesToSave));
    console.log(`Saved ${messagesToSave.length} messages to localStorage for session ${sessionId}`);
  } catch (error) {
    console.error('Error saving messages to localStorage:', error);
    // If quota exceeded, try to clear old messages
    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
      console.warn('localStorage quota exceeded, clearing old messages...');
      try {
        // Keep only the last 50 messages and try again
        const reducedMessages = messages.slice(-50);
        const key = getMessagesStorageKey(sessionId);
        localStorage.setItem(key, JSON.stringify(reducedMessages));
      } catch (retryError) {
        console.error('Failed to save even reduced messages:', retryError);
      }
    }
  }
};

// Get initial session ID and load messages
const getInitialSessionId = () => {
  if (typeof window === 'undefined') return crypto.randomUUID();
  return sessionStorage.getItem('sessionId') || crypto.randomUUID();
};

const initialSessionId = getInitialSessionId();
const initialMessages = loadMessagesFromStorage(initialSessionId);

export const useChat = create<ChatState>((set, get) => ({
  messages: initialMessages,
  isLoading: false,
  sessionId: initialSessionId,

  addMessage: (msg) =>
    set((state) => {
      const messages = [...state.messages, msg];
      saveMessagesToStorage(state.sessionId, messages);
      return { messages };
    }),

  appendToLast: (content) =>
    set((state) => {
      if (state.messages.length === 0) return state;
      const messages = [...state.messages];
      const lastMsg = messages[messages.length - 1];
      if (!lastMsg) return state;
      // Create new object to trigger React re-render
      messages[messages.length - 1] = {
        ...lastMsg,
        content: lastMsg.content + content,
      };
      saveMessagesToStorage(state.sessionId, messages);
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
      saveMessagesToStorage(state.sessionId, messages);
      return { messages };
    }),

  setLoading: (loading) => set({ isLoading: loading }),

  clearMessages: () =>
    set((state) => {
      const messages: Message[] = [];
      saveMessagesToStorage(state.sessionId, messages);
      return { messages };
    }),

  setSessionId: (sessionId) =>
    set(() => {
      // Load messages for the new session
      const messages = loadMessagesFromStorage(sessionId);
      console.log(`Switched to session ${sessionId} with ${messages.length} messages`);
      return { sessionId, messages };
    }),
}));

// Persist session ID
if (typeof window !== 'undefined') {
  useChat.subscribe((state) => {
    sessionStorage.setItem('sessionId', state.sessionId);
  });
}
