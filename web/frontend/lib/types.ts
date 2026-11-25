export interface CanvasObject {
  id: string;
  type: string;
  content: string;
  context?: string;
  confidence: number;
  turn_id: number;
  quote?: string;
  source?: string;
  references_text?: string[];
  references?: string[];
  referenced_by?: string[];
  leads_to?: string[];
  caused_by?: string[];
  timestamp?: number;
  span?: [number, number] | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  retrievedObjects?: CanvasObject[];
  extractedObjects?: CanvasObject[];
}

export interface GraphNode {
  id: string;
  name?: string;
  type: string;
  content: string;
  context?: string;
  confidence: number;
  turn_id: number;
  quote?: string;
  timestamp?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  relation?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface CanvasStats {
  total_objects: number;
  turn_count: number;
  by_type: Record<string, number>;
  avg_confidence: number;
}
