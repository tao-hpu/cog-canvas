# CogCanvas Frontend - Quick Start Guide

## Start Development

```bash
cd /Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/web/frontend
pnpm dev
```

Frontend will be available at: **http://localhost:3700**

## Prerequisites

1. Backend API must be running on **http://localhost:3701**
2. Node.js 18+ installed
3. pnpm installed (`npm install -g pnpm`)

## Project Overview

```
Frontend (3700) ←→ Backend (3701)
     │
     ├── Chat Interface (Left)
     │   ├── Message List
     │   └── Input Field
     │
     └── Canvas Sidebar (Right)
         ├── Control Panel
         │   ├── CogCanvas Toggle
         │   ├── View Mode (List/Graph)
         │   ├── Statistics
         │   └── Clear Button
         │
         └── Canvas View
             ├── List View (default)
             └── Graph View
```

## Key Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 16.0.10 | Framework |
| React | 19.2.3 | UI Library |
| TypeScript | 5.9.3 | Type Safety |
| Tailwind CSS | 4.1.18 | Styling |
| Zustand | 5.0.9 | State Management |
| shadcn/ui | latest | UI Components |
| react-force-graph-2d | 1.29.0 | Graph Visualization |

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx       → Root layout (dark theme)
│   ├── page.tsx         → Main chat page
│   └── globals.css      → Global styles
├── components/
│   ├── chat/            → Chat UI (3 components)
│   ├── canvas/          → Canvas UI (2 components)
│   ├── control/         → Control panel (1 component)
│   └── ui/              → shadcn components (8 components)
├── hooks/
│   ├── useChat.ts       → Chat state
│   └── useCanvas.ts     → Canvas state
└── lib/
    ├── api.ts           → API client
    ├── types.ts         → TypeScript types
    └── utils.ts         → Utilities
```

## API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Send message (streaming) |
| `/api/canvas/objects` | GET | Get all canvas objects |
| `/api/canvas/graph` | GET | Get graph data |
| `/api/canvas/stats` | GET | Get statistics |
| `/api/canvas/clear` | POST | Clear canvas |

All requests include `X-Session-ID` header for session management.

## State Management

### Chat State (useChat)
- `messages` - Array of chat messages
- `isLoading` - Loading indicator
- `sessionId` - Session UUID
- `addMessage()` - Add new message
- `appendToLast()` - Append to last message
- `setLoading()` - Set loading state

### Canvas State (useCanvas)
- `objects` - Array of canvas objects
- `graphData` - Graph visualization data
- `stats` - Canvas statistics
- `cogcanvasEnabled` - CogCanvas on/off
- `viewMode` - 'list' or 'graph'
- `toggleCogcanvas()` - Toggle enabled state
- `setViewMode()` - Change view mode
- `clearAll()` - Clear all canvas data

## Object Types

```typescript
type ObjectType =
  | 'decision'    // Blue
  | 'todo'        // Yellow
  | 'key_fact'    // Green
  | 'reminder'    // Purple
  | 'insight'     // Orange
```

## Development Workflow

1. **Start Dev Server**
   ```bash
   pnpm dev
   ```

2. **Make Changes**
   - Edit components in `components/`
   - Update state in `hooks/`
   - Modify API calls in `lib/api.ts`
   - Add new types in `lib/types.ts`

3. **Hot Reload**
   - Changes auto-reload in browser
   - No need to restart server

4. **Add UI Components**
   ```bash
   pnpm dlx shadcn@latest add [component-name]
   ```

## Useful Commands

```bash
# Install dependencies
pnpm install

# Start dev server (port 3700)
pnpm dev

# Run linter
pnpm lint

# Add shadcn component
pnpm dlx shadcn@latest add [component]

# Update dependencies
pnpm update
```

## Common Tasks

### Add a New Canvas Object Type

1. Update `lib/types.ts`:
   ```typescript
   export type ObjectType = ... | 'new_type';
   ```

2. Add color in components:
   ```typescript
   const typeColors = {
     // ...
     new_type: 'bg-color-500',
   };
   ```

### Add a New API Endpoint

1. Add function in `lib/api.ts`:
   ```typescript
   export async function newEndpoint(sessionId: string) {
     const res = await fetch(`${API_BASE}/api/new`, {
       headers: { 'X-Session-ID': sessionId },
     });
     return res.json();
   }
   ```

2. Use in component:
   ```typescript
   import { newEndpoint } from '@/lib/api';

   const data = await newEndpoint(sessionId);
   ```

### Add a New Component

1. Create file in `components/`:
   ```typescript
   'use client';

   export function MyComponent() {
     return <div>Content</div>;
   }
   ```

2. Import and use:
   ```typescript
   import { MyComponent } from '@/components/MyComponent';
   ```

## Debugging

### Check if Backend is Running
```bash
curl http://localhost:3701/api/canvas/objects \
  -H "X-Session-ID: test"
```

### View Session ID
Open browser console:
```javascript
sessionStorage.getItem('sessionId')
```

### Clear Session
```javascript
sessionStorage.clear()
location.reload()
```

## Styling Tips

- Use Tailwind utility classes
- Dark theme colors via CSS variables
- shadcn components are pre-styled
- Use `cn()` helper to merge classes:
  ```typescript
  import { cn } from '@/lib/utils';

  <div className={cn('base-class', condition && 'conditional-class')} />
  ```

## Port Reference

- **3700** - Frontend (Next.js)
- **3701** - Backend (FastAPI)

## Environment

- Development server uses Turbopack
- Hot Module Replacement (HMR) enabled
- TypeScript strict mode
- React strict mode
- Dark theme by default

## Troubleshooting

**Port already in use:**
```bash
lsof -ti:3700 | xargs kill
pnpm dev
```

**Module not found:**
```bash
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

**Type errors:**
Check `tsconfig.json` for strict mode settings

**Style not applying:**
Restart dev server to rebuild Tailwind

## Resources

- [Next.js Docs](https://nextjs.org/docs)
- [shadcn/ui](https://ui.shadcn.com)
- [Tailwind CSS](https://tailwindcss.com)
- [Zustand](https://zustand-demo.pmnd.rs)

---

**Quick Start**: `cd frontend && pnpm dev` → http://localhost:3700
