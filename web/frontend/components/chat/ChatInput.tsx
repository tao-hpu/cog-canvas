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

// Pre-configured test messages from Next.js RFC #77740 (Deployment Adapters API)
// Source: https://github.com/vercel/next.js/discussions/77740
// Real multi-stakeholder technical discussion: Vercel, Netlify, Deno Deploy, Cloudflare, etc.
const TEST_MESSAGES = [
  // === Turn 1: Original RFC post ===
  `RFC: Deployment Adapters API

We're introducing a Deployment Adapters API to enable easier deployment across platforms. Vercel will use the same adapter API as every other partner.

Key Pain Points:
1. Background Work Tracking - Currently requires reverse-engineering request lifecycles
2. Config Modification - Providers must patch next.config or use undocumented env vars
3. Manifest Reliance - Undocumented manifests create fragility
4. Full Server Dependency - Entrypoints require loading entire next-server, causing cold boot issues

Build Output Changes:
- Node.js: handler(req, res, ctx) returning Promise<void>
- Edge: handler(req, ctx) returning Promise<Response>
- waitUntil callback signals background task completion

Target: Alpha in Next.js 16 this summer.`,

  // === Turn 2: Deprecation question ===
  `Can you clarify what happens to minimalMode? Also, what about Vercel-specific features like x-matched-path and x-now-route-matches? Will these become documented or refactored as generic capabilities?`,

  // === Turn 3: Routing concern ===
  `If there are separate entrypoints, does this mean adapters need to implement custom routing logic? The routing rules have historically been very lightly documented, causing lots of reverse-engineering overhead. This is a critical concern for platform providers.`,

  // === Turn 4: Deno Deploy feedback ===
  `From Deno Deploy's perspective: I'd prefer a singular entrypoint over multiple ones. Also interested in CDN cache integration hooks - currently these are flagged by minimal mode. Deno's serverless architecture would benefit from a unified entry.`,

  // === Turn 5: OpenNext questions ===
  `Three questions about the RFC:
1. Will Vercel develop adapters directly?
2. Will you incorporate OpenNext work?
3. What about image optimization provider flexibility?`,

  // === Turn 6: PPR platform lock-in concern ===
  `Partial Prerendering (PPR) is a critical missing feature for non-Vercel platforms. I propose a Progressive Rendering Format standard for CDN-friendly PPR implementation. Currently only Vercel can properly leverage PPR. This creates platform lock-in.`,

  // === Turn 7: Comprehensive feedback on gaps ===
  `Comprehensive feedback on the RFC gaps:
- Missing middleware matcher documentation
- Underspecified pathname format (suggests URLPattern standard)
- Insufficient routing behavior specification
- Unclear fallbackID handling for dynamic routes
- Ambiguous IMAGE type pathname mapping

Critical Question: Will adapters need to implement full end-to-end routing? This has historically been the biggest barrier for platform providers.`,

  // === Turn 8: Vercel detailed response ===
  `Vercel's detailed response to the concerns:
- maxDuration/expiration/revalidate will be documented
- fallbackID always references STATIC_FILE
- allowQuery helps generate stable ISR cache keys
- next-server remains but much slimmer; adapters route at CDN/edge level
- No backport to 14.x (requires big refactors)
- Node.js signature kept matching IncomingMessage/ServerResponse for compatibility
- Considering a community adapter namespace
- Undocumented private APIs will be removed with documented alternatives and lead time`,

  // === Turn 9: Beta docs announcement ===
  `Beta documentation is now available at nextjs.org/docs/beta/app/api-reference/config/next-config-js/adapterPath. This marks a significant milestone toward the alpha release in Next.js 16.`,

  // === Turn 10: Community feature requests ===
  `Community requests summary:
- Need adapter-level HTTP header customization (e.g., managing unsupported stale-while-revalidate)
- Request customizable image optimization caching locations
- Advocated for Docker/environment variable friendliness
- Suggested optional lifecycle hooks (onPreBuild, onPostOutput) for CI/CD integration`,

  // === Turn 11: Status update ===
  `Current RFC Status Summary (Oct 2025):
- Beta documentation is live
- Adapters API in alpha as of Next.js 16
- Official adapter implementations pending for Netlify, Cloudflare, AWS
- Release timeline: several months away
- Community builders exploring early adoption

Remaining TODOs:
- Complete routing specification documentation
- Finalize PPR support for non-Vercel platforms
- Ship official adapters for major platforms`,
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
