import type { CanvasObject, CanvasStats, GraphData } from "./types";

const API_BASE = "http://localhost:3701";

function withSession(path: string, sessionId: string): string {
  const url = new URL(path, API_BASE);
  url.searchParams.set("session_id", sessionId);
  return url.toString();
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function sendMessage(
  message: string,
  sessionId: string,
  cogcanvasEnabled: boolean,
): Promise<ReadableStream<Uint8Array>> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      cogcanvas_enabled: cogcanvasEnabled,
    }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
  }
  return response.body;
}

export async function getCanvasObjects(sessionId: string): Promise<CanvasObject[]> {
  const response = await fetch(withSession("/api/canvas/objects", sessionId));
  return readJson<CanvasObject[]>(response);
}

export async function getCanvasGraph(sessionId: string): Promise<GraphData> {
  const response = await fetch(withSession("/api/canvas/graph", sessionId));
  return readJson<GraphData>(response);
}

export async function getCanvasStats(sessionId: string): Promise<CanvasStats> {
  const response = await fetch(withSession("/api/canvas/stats", sessionId));
  return readJson<CanvasStats>(response);
}

export async function clearCanvas(sessionId: string): Promise<void> {
  const response = await fetch(withSession("/api/canvas/clear", sessionId), {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Clear request failed: ${response.status}`);
  }
}
