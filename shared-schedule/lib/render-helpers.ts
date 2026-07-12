import type { Occurrence } from "./types";

// Untimed events (todos) render at a fixed end-of-day slot instead of a
// separate all-day strip, per product decision.
export const DEFAULT_UNTIMED_MINUTES = 22 * 60 + 30; // 22:30
export const DEFAULT_UNTIMED_DURATION = 30;
export const DEFAULT_EVENT_DURATION = 60; // events with a start but no stated end

// Full 24h so night shifts / late activity (23:00-07:00) are visible —
// previously clipped to a 6:00-23:00 "daytime" window.
export const GRID_START_MINUTES = 0; // 00:00
export const GRID_END_MINUTES = 24 * 60; // 24:00 (midnight, end of day)

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
  return `${String(h24).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

export function occurrenceKey(occ: Occurrence): string {
  return occ.event.id + occ.date;
}

/** "Apart" occurrences (person !== "both") whose time ranges overlap on the
 * same day — the couple is double-booked with two different things. Keyed
 * by occurrenceKey() so callers can flag both blocks involved. */
export function computeOverlapKeys(dayOccs: Occurrence[]): Set<string> {
  const overlapping = new Set<string>();
  const apart = dayOccs.filter((o) => o.event.person !== "both");
  for (let i = 0; i < apart.length; i++) {
    for (let j = i + 1; j < apart.length; j++) {
      const a = apart[i];
      const b = apart[j];
      if (a.event.person === b.event.person) continue; // same person double-booked with themself isn't this kind of conflict
      const aStart = displayStartMinutes(a);
      const aEnd = aStart + displayDuration(a);
      const bStart = displayStartMinutes(b);
      const bEnd = bStart + displayDuration(b);
      if (aStart < bEnd && bStart < aEnd) {
        overlapping.add(occurrenceKey(a));
        overlapping.add(occurrenceKey(b));
      }
    }
  }
  return overlapping;
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
      // Floor raised (14 -> 26) so the larger event-block text still fits
      // legibly even for a 15-minute-long occurrence.
      height: Math.max(((item.end - item.start) / 60) * hourHeightPx, 26),
      colIndex,
      numCols: 1, // filled in below
    });
  }

  const numCols = Math.max(columnEnds.length, 1);
  return placed.map((p) => ({ ...p, numCols }));
}
