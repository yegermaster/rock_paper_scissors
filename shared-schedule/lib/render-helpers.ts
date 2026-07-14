import type { Occurrence } from "./types";

// Untimed events (todos) render at a fixed end-of-day slot instead of a
// separate all-day strip, per product decision.
export const DEFAULT_UNTIMED_MINUTES = 22 * 60 + 30; // 22:30
export const DEFAULT_UNTIMED_DURATION = 30;
export const DEFAULT_EVENT_DURATION = 60; // events with a start but no stated end

// The week grid's vertical range is computed per-render (see
// computeGridRange below) rather than fixed: always showing all 24 hours
// made a typical, mostly-daytime week awkwardly tall with hours of empty
// space, while clipping to a fixed daytime window hid real night activity.
// The absolute day boundary is still needed for "does this event cross
// midnight" checks regardless of the currently-visible range.
export const ABSOLUTE_DAY_START = 0; // 00:00
export const ABSOLUTE_DAY_END = 24 * 60; // 24:00 (midnight, end of day)

const DEFAULT_GRID_START_MINUTES = 7 * 60; // 07:00
const DEFAULT_GRID_END_MINUTES = 22 * 60; // 22:00

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

/** The week grid's visible hour range for one render: a sane default
 * (07:00-22:00) widened to fit any occurrence that falls outside it (e.g.
 * a 06:00 flight or a 23:00 night shift), rounded to whole hours. Spans
 * render as their own bar above the grid, so they don't affect this. */
export function computeGridRange(occurrences: Occurrence[]): { start: number; end: number } {
  let start = DEFAULT_GRID_START_MINUTES;
  let end = DEFAULT_GRID_END_MINUTES;
  for (const occ of occurrences) {
    if (occ.isMultiDaySpan) continue;
    const s = displayStartMinutes(occ);
    const e = Math.min(s + displayDuration(occ), ABSOLUTE_DAY_END);
    if (s < start) start = Math.floor(s / 60) * 60;
    if (e > end) end = Math.min(ABSOLUTE_DAY_END, Math.ceil(e / 60) * 60);
  }
  return { start: Math.max(ABSOLUTE_DAY_START, start), end: Math.min(ABSOLUTE_DAY_END, end) };
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
export function layoutDayColumn(
  occs: Occurrence[],
  hourHeightPx: number,
  gridStart: number,
  gridEnd: number
): LaidOutBlock[] {
  const items = occs
    .map((occ) => {
      const start = Math.max(displayStartMinutes(occ), gridStart);
      const rawEnd = displayStartMinutes(occ) + displayDuration(occ);
      const end = Math.min(Math.max(rawEnd, start + 15), gridEnd);
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
      top: ((item.start - gridStart) / 60) * hourHeightPx,
      // Floor raised so the block always has room for its category band +
      // title + time at the current (large) font sizes, even for a very
      // short occurrence — a squeezed block that clips its own text is
      // worse than one that's a bit taller than its exact duration.
      height: Math.max(((item.end - item.start) / 60) * hourHeightPx, 76),
      colIndex,
      numCols: 1, // filled in below
    });
  }

  const numCols = Math.max(columnEnds.length, 1);
  return placed.map((p) => ({ ...p, numCols }));
}
