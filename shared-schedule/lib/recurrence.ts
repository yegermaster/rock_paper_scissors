import { addDays, diffDays, dayOfWeek, startOfWeek, toDate } from "./dates";
import type { EventRecord, Occurrence } from "./types";

function matchesRecurrence(event: EventRecord, ymd: string): boolean {
  if (ymd < event.start_date) return false;
  if (event.recurrence_end_date && ymd > event.recurrence_end_date) return false;

  const interval = event.recurrence_interval ?? 1;

  if (event.recurrence_frequency === "daily") {
    const diff = diffDays(event.start_date, ymd);
    return diff >= 0 && diff % interval === 0;
  }

  if (event.recurrence_frequency === "weekly") {
    const days = event.recurrence_days_of_week ?? [dayOfWeek(event.start_date)];
    if (!days.includes(dayOfWeek(ymd))) return false;
    const weeksDiff = diffDays(startOfWeek(event.start_date), startOfWeek(ymd)) / 7;
    return weeksDiff >= 0 && weeksDiff % interval === 0;
  }

  if (event.recurrence_frequency === "monthly") {
    const base = toDate(event.start_date);
    const candidate = toDate(ymd);
    if (base.getUTCDate() !== candidate.getUTCDate()) return false;
    const monthsDiff =
      (candidate.getUTCFullYear() - base.getUTCFullYear()) * 12 +
      (candidate.getUTCMonth() - base.getUTCMonth());
    return monthsDiff >= 0 && monthsDiff % interval === 0;
  }

  return false;
}

/**
 * Expands events (recurring and multi-day) into one Occurrence per calendar
 * day they touch within [rangeStart, rangeEnd], so day-grid views can just
 * group occurrences by `date` without re-deriving recurrence/span logic.
 */
export function expandOccurrences(
  events: EventRecord[],
  rangeStart: string,
  rangeEnd: string
): Occurrence[] {
  const occurrences: Occurrence[] = [];

  for (const event of events) {
    if (event.recurrence_frequency) {
      let cursor = rangeStart;
      while (cursor <= rangeEnd) {
        if (matchesRecurrence(event, cursor)) {
          occurrences.push({
            event,
            date: cursor,
            isMultiDaySpan: false,
            spanStart: cursor,
            spanEnd: cursor,
          });
        }
        cursor = addDays(cursor, 1);
      }
      continue;
    }

    // Non-recurring: may still be a multi-day span.
    const spanStart = event.start_date > rangeStart ? event.start_date : rangeStart;
    const spanEnd = event.end_date < rangeEnd ? event.end_date : rangeEnd;
    if (spanStart > spanEnd) continue;

    let cursor = spanStart;
    while (cursor <= spanEnd) {
      occurrences.push({
        event,
        date: cursor,
        isMultiDaySpan: event.start_date !== event.end_date,
        spanStart: event.start_date,
        spanEnd: event.end_date,
      });
      cursor = addDays(cursor, 1);
    }
  }

  return occurrences;
}
