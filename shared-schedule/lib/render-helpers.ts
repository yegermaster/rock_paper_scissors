import { colorForCategory } from "./colors";
import type { Occurrence } from "./types";

// Untimed events (todos) render at a fixed end-of-day slot instead of a
// separate all-day strip, per product decision.
export const DEFAULT_UNTIMED_MINUTES = 22 * 60 + 30; // 22:30
export const DEFAULT_UNTIMED_DURATION = 30;
export const DEFAULT_EVENT_DURATION = 60; // events with a start but no stated end

export const GRID_START_MINUTES = 6 * 60; // 6:00
export const GRID_END_MINUTES = 23 * 60; // 23:00

export function displayStartMinutes(occ: Occurrence): number {
  if (!occ.event.start_time) return DEFAULT_UNTIMED_MINUTES;
  const [h, m] = occ.event.start_time.split(":").map(Number);
  return h * 60 + m;
}

export function displayDuration(occ: Occurrence): number {
  if (!occ.event.start_time) return DEFAULT_UNTIMED_DURATION;
  return occ.event.duration_minutes || DEFAULT_EVENT_DURATION;
}

export function minutesToLabel(minutes: number): string {
  const h24 = Math.floor(minutes / 60);
  const m = minutes % 60;
  const period = h24 >= 12 ? "PM" : "AM";
  const h12 = h24 % 12 === 0 ? 12 : h24 % 12;
  return m === 0 ? `${h12} ${period}` : `${h12}:${String(m).padStart(2, "0")} ${period}`;
}

export function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

export function chipColors(category: string) {
  return colorForCategory(category);
}

export type LaidOutBlock = {
  occ: Occurrence;
  top: number; // px from grid top
  height: number; // px
  colIndex: number;
  numCols: number;
};

/** Greedy overlap-column layout: not globally optimal packing, but simple
 * and visually correct for the low event-density this app expects. */
export function layoutDayColumn(occs: Occurrence[], hourHeightPx: number): LaidOutBlock[] {
  const items = occs
    .map((occ) => {
      const start = Math.max(displayStartMinutes(occ), GRID_START_MINUTES);
      const rawEnd = displayStartMinutes(occ) + displayDuration(occ);
      const end = Math.min(Math.max(rawEnd, start + 15), GRID_END_MINUTES);
      return { occ, start, end };
    })
    .sort((a, b) => a.start - b.start || b.end - a.end);

  const columnEnds: number[] = [];
  const placed: (LaidOutBlock & { start: number; end: number })[] = [];

  for (const item of items) {
    let colIndex = columnEnds.findIndex((end) => end <= item.start);
    if (colIndex === -1) {
      colIndex = columnEnds.length;
      columnEnds.push(item.end);
    } else {
      columnEnds[colIndex] = item.end;
    }
    placed.push({
      occ: item.occ,
      start: item.start,
      end: item.end,
      top: ((item.start - GRID_START_MINUTES) / 60) * hourHeightPx,
      height: Math.max(((item.end - item.start) / 60) * hourHeightPx, 14),
      colIndex,
      numCols: 1, // filled in below
    });
  }

  const numCols = Math.max(columnEnds.length, 1);
  return placed.map((p) => ({ ...p, numCols }));
}
