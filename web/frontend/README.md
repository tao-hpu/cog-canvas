# CogCanvas Frontend

Interactive web interface for CogCanvas - a cognitive canvas for structured information extraction from conversations.

## Features

- Real-time chat interface with streaming responses
- Canvas sidebar for viewing extracted objects
- Two view modes: List view and Graph visualization
- Dark theme optimized UI
- TypeScript with strict mode
- Built with Next.js 14 App Router

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **UI Library**: shadcn/ui (New York style)
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Graph Visualization**: react-force-graph-2d
- **Package Manager**: pnpm

## Prerequisites

- Node.js 18+
- pnpm (install with: `npm install -g pnpm`)
- Backend API running on http://localhost:3701

## Installation

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
```

The app will be available at http://localhost:3700

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout with dark theme
│   ├── page.tsx           # Main chat interface
│   └── globals.css        # Global styles with CSS variables
├── components/
│   ├── chat/              # Chat components
│   │   ├── MessageList.tsx
│   │   ├── MessageItem.tsx
│   │   └── ChatInput.tsx
│   ├── canvas/            # Canvas components
│   │   ├── ObjectList.tsx
│   │   └── GraphView.tsx
│   ├── control/           # Control panel
│   │   └── ControlPanel.tsx
│   └── ui/                # shadcn/ui components
├── hooks/                 # React hooks
│   ├── useChat.ts         # Chat state management
│   └── useCanvas.ts       # Canvas state management
├── lib/
│   ├── api.ts             # API client functions
│   ├── types.ts           # TypeScript type definitions
│   └── utils.ts           # Utility functions
└── components.json        # shadcn/ui configuration
```

## Available Scripts

- `pnpm dev` - Start development server on port 3700
- `pnpm build` - Build for production
- `pnpm start` - Start production server
- `pnpm lint` - Run ESLint

## Configuration

### Port Configuration
The dev server runs on port 3700 (configured in package.json)

### API Endpoint
Backend API is configured to http://localhost:3701 in `lib/api.ts`

### Session Management
Session IDs are automatically generated and stored in sessionStorage

## Component Overview

### Chat Components
- **MessageList**: Scrollable message container with auto-scroll
- **MessageItem**: Individual message bubble with extracted object badges
- **ChatInput**: Message input field with send button

### Canvas Components
- **ObjectList**: List view of extracted canvas objects with type badges
- **GraphView**: Force-directed graph visualization of object relationships

### Control Components
- **ControlPanel**: Canvas controls including enable/disable toggle, view mode switcher, statistics, and clear button

## State Management

### Chat State (useChat)
- Messages array
- Loading state
- Session ID management
- Message operations (add, append, update)

### Canvas State (useCanvas)
- Canvas objects
- Graph data
- Statistics
- CogCanvas enable/disable
- View mode (list/graph)

## API Integration

All API calls are in `lib/api.ts`:
- `sendMessage` - Send chat message (returns streaming response)
- `getCanvasObjects` - Fetch all canvas objects
- `getCanvasGraph` - Fetch graph data
- `getCanvasStats` - Fetch canvas statistics
- `clearCanvas` - Clear all canvas objects

## Styling

Uses Tailwind CSS with shadcn/ui design system:
- New York style components
- Zinc color palette
- CSS variables for theming
- Dark theme by default

## Type Safety

Full TypeScript support with strict mode enabled:
- `noImplicitAny: true`
- `strictNullChecks: true`
- `noUncheckedIndexedAccess: true`

## Development Notes

- The frontend automatically polls canvas data every 5 seconds when CogCanvas is enabled
- Streaming responses are handled via Server-Sent Events (SSE)
- Session persistence uses browser sessionStorage
- All components are client-side rendered ('use client' directive)
