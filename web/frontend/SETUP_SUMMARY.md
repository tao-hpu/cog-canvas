# CogCanvas Frontend - Setup Summary

## Project Initialized Successfully

The Next.js frontend for CogCanvas has been fully set up with all required components and configurations.

## Installation Complete

### Core Technologies
- **Framework**: Next.js 16.0.10 (App Router)
- **Package Manager**: pnpm
- **TypeScript**: 5.9.3 with strict mode
- **Styling**: Tailwind CSS v4.1.18
- **UI Components**: shadcn/ui (New York style, Zinc color)
- **State Management**: Zustand 5.0.9
- **Graph Visualization**: react-force-graph-2d 1.29.0

### Project Structure

```
/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/web/frontend/
├── app/
│   ├── layout.tsx              # Root layout with dark theme
│   ├── page.tsx                # Main chat interface (3-column layout)
│   └── globals.css             # Tailwind v4 theme configuration
├── components/
│   ├── chat/
│   │   ├── ChatInput.tsx       # Message input with send button
│   │   ├── MessageItem.tsx     # Individual message bubble
│   │   └── MessageList.tsx     # Scrollable message container
│   ├── canvas/
│   │   ├── ObjectList.tsx      # List view of extracted objects
│   │   └── GraphView.tsx       # Force-directed graph visualization
│   ├── control/
│   │   └── ControlPanel.tsx    # Canvas controls & statistics
│   └── ui/                     # shadcn/ui components (8 components)
│       ├── button.tsx
│       ├── card.tsx
│       ├── input.tsx
│       ├── scroll-area.tsx
│       ├── tabs.tsx
│       ├── badge.tsx
│       ├── switch.tsx
│       └── separator.tsx
├── hooks/
│   ├── useChat.ts              # Chat state management (Zustand)
│   └── useCanvas.ts            # Canvas state management (Zustand)
├── lib/
│   ├── api.ts                  # API client for backend
│   ├── types.ts                # TypeScript type definitions
│   └── utils.ts                # Utility functions (cn helper)
├── components.json             # shadcn/ui configuration
├── next.config.js              # Next.js configuration
├── postcss.config.js           # PostCSS with Tailwind v4
├── tailwind.config.ts          # Tailwind CSS v4 config
├── tsconfig.json               # TypeScript strict mode config
├── package.json                # Dependencies & scripts
└── README.md                   # Project documentation
```

## Key Features Implemented

### 1. Chat Interface (`app/page.tsx`)
- Real-time streaming chat with SSE
- Message history with auto-scroll
- User/Assistant message bubbles
- Loading states
- Extracted objects display in messages

### 2. Canvas Sidebar
- **List View**: Type-colored cards with object details
- **Graph View**: Interactive force-directed graph
- Statistics dashboard
- CogCanvas enable/disable toggle
- View mode switcher (List/Graph)
- Clear canvas functionality

### 3. API Integration (`lib/api.ts`)
All endpoints configured for `http://localhost:3701`:
- `POST /api/chat` - Send messages (streaming)
- `GET /api/canvas/objects` - Get all canvas objects
- `GET /api/canvas/graph` - Get graph data
- `GET /api/canvas/stats` - Get statistics
- `POST /api/canvas/clear` - Clear canvas

### 4. State Management
- **useChat**: Messages, loading state, session ID
- **useCanvas**: Objects, graph data, stats, view mode
- Auto-refresh canvas data every 5 seconds
- Session persistence in sessionStorage

### 5. Type Safety
Comprehensive TypeScript types in `lib/types.ts`:
- `ObjectType`: decision | todo | key_fact | reminder | insight
- `CanvasObject`: Full object structure
- `Message`: Chat message with extracted objects
- `GraphData`: Nodes and links for visualization
- `CanvasStats`: Statistics interface

## Configuration

### Ports
- **Frontend Dev Server**: 3700 (`pnpm dev`)
- **Backend API**: 3701 (configured in lib/api.ts)

### TypeScript Config
Strict mode enabled:
- `noImplicitAny: true`
- `strictNullChecks: true`
- `noUncheckedIndexedAccess: true`
- `target: ES2022`

### Styling
- Dark theme by default (in layout.tsx)
- Tailwind v4 with @theme directive
- CSS variables for all colors
- Zinc color palette
- New York style components

## Known Issues

### Turbopack Path Encoding Issue
The production build (`pnpm build`) currently fails due to a Turbopack bug with Chinese characters in the file path. This is a known Next.js/Turbopack issue:

```
byte index 18 is not a char boundary; it is inside '士' (bytes 17..20)
```

**Workarounds**:
1. **Development**: Use `pnpm dev` (works fine, uses Turbopack for dev)
2. **Production**:
   - Option A: Move project to path without Chinese characters
   - Option B: Wait for Turbopack fix in future Next.js release
   - Option C: Use webpack by setting `turbopack: false` in next.config.js

The dev server works perfectly, so you can develop and test the application without issues.

## Quick Start

```bash
# Navigate to frontend directory
cd /Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/web/frontend

# Install dependencies (already done)
pnpm install

# Start development server
pnpm dev

# Visit http://localhost:3700
```

## Next Steps

1. **Start backend**: Ensure backend is running on port 3701
2. **Test frontend**: Open http://localhost:3700
3. **Start chatting**: Type messages and see objects extracted in real-time
4. **Toggle views**: Switch between List and Graph views
5. **Monitor stats**: Watch statistics update as you chat

## API Session Flow

1. Frontend generates a UUID session ID on first load
2. Session ID stored in sessionStorage
3. All API requests include `X-Session-ID` header
4. Backend maintains separate canvas state per session
5. Canvas data refreshes every 5 seconds when enabled

## Component Communication

```
User Input → ChatInput → page.tsx → API (sendMessage)
                                   ↓
                              SSE Stream
                                   ↓
                        updateLastMessage
                                   ↓
                         Refresh Canvas Data
                                   ↓
                    ┌────────────────────────┐
                    ↓                        ↓
              ObjectList                GraphView
```

## Files Summary

- **21 TypeScript files** (components, hooks, lib, app)
- **8 shadcn/ui components** (button, card, input, etc.)
- **3 chat components** (MessageList, MessageItem, ChatInput)
- **2 canvas components** (ObjectList, GraphView)
- **1 control component** (ControlPanel)
- **2 state hooks** (useChat, useCanvas)
- **3 lib files** (api, types, utils)

All components are fully typed with TypeScript and follow Next.js 14 best practices.

## Development Notes

- All components use 'use client' directive (client-side rendering)
- Streaming responses handled via Server-Sent Events
- Force graph uses dynamic import to avoid SSR issues
- Auto-scroll on new messages
- Responsive design (chat flex, sidebar fixed 96 width)
- Dark theme optimized for extended use

---

**Project Status**: ✅ Frontend Initialized & Ready for Development

**Dev Server**: Run `pnpm dev` to start on port 3700
**Backend Required**: Ensure backend runs on port 3701
