import { create } from 'zustand';
import { CanvasObject, GraphData, CanvasStats } from '@/lib/types';

interface CanvasState {
  objects: CanvasObject[];
  graphData: GraphData | null;
  stats: CanvasStats | null;
  cogcanvasEnabled: boolean;
  viewMode: 'list' | 'graph' | 'help';
  setObjects: (objects: CanvasObject[]) => void;
  addObjects: (objects: CanvasObject[]) => void;
  setGraphData: (data: GraphData) => void;
  setStats: (stats: CanvasStats) => void;
  toggleCogcanvas: () => void;
  setViewMode: (mode: 'list' | 'graph' | 'help') => void;
  clearAll: () => void;
}

export const useCanvas = create<CanvasState>((set) => ({
  objects: [],
  graphData: null,
  stats: null,
  cogcanvasEnabled: true,
  viewMode: 'list',

  setObjects: (objects) => set({ objects }),

  addObjects: (newObjects) =>
    set((state) => ({
      objects: [...state.objects, ...newObjects],
    })),

  setGraphData: (data) => set({ graphData: data }),

  setStats: (stats) => set({ stats }),

  toggleCogcanvas: () =>
    set((state) => ({ cogcanvasEnabled: !state.cogcanvasEnabled })),

  setViewMode: (mode) => set({ viewMode: mode }),

  clearAll: () =>
    set({
      objects: [],
      graphData: null,
      stats: null,
    }),
}));
