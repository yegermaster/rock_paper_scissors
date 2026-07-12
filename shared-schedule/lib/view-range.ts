import { addDays, daysInMonth, startOfMonth, startOfWeek, startOfYear } from "./dates";
import type { ViewKind } from "./types";

// Shared by the image renderer and the manual events list API so both
// agree on exactly which dates a given view/anchor covers.
export function rangeFor(view: ViewKind, anchor: string): { start: string; end: string } {
  if (view === "week") {
    const start = startOfWeek(anchor);
    return { start, end: addDays(start, 6) };
  }
  if (view === "month") {
    const start = startOfMonth(anchor);
    const [y, m] = start.split("-").map(Number);
    return { start, end: `${y}-${String(m).padStart(2, "0")}-${daysInMonth(y, m)}` };
  }
  const start = startOfYear(anchor);
  const y = Number(start.slice(0, 4));
  return { start, end: `${y}-12-31` };
}
