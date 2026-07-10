import { WEEKDAY_LABELS, addDays, dayOfWeek } from "../dates";
import { hebrewDateShort } from "../hebrew-date";
import {
  GRID_START_MINUTES,
  GRID_END_MINUTES,
  chipColors,
  layoutDayColumn,
  minutesToLabel,
  truncate,
} from "../render-helpers";
import type { Occurrence } from "../types";

const WIDTH = 1900;
const GUTTER = 84;
const DAY_WIDTH = (WIDTH - GUTTER) / 7;
const HOUR_HEIGHT = 56;
const GRID_HEIGHT = ((GRID_END_MINUTES - GRID_START_MINUTES) / 60) * HOUR_HEIGHT;
const HEADER_HEIGHT = 108;
const SPAN_ROW_HEIGHT = 42;
const MAX_SPAN_ROWS = 2;

// Hebrew calendar reads right-to-left: Sunday renders on the right edge
// (adjacent to the hour gutter, which itself moves to the right), Saturday
// on the left. `colX` maps a chronological day index (0=Sun..6=Sat) to its
// physical x-position for absolutely-positioned elements (span bars).
function colX(chronoIndex: number): number {
  return (6 - chronoIndex) * DAY_WIDTH;
}

export function renderWeekView(
  weekStart: string,
  occurrences: Occurrence[],
  today: string
) {
  const days = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));
  const displayDays = [...days].reverse(); // Sat..Sun, for right-to-left flex order

  const byDay = new Map<string, Occurrence[]>();
  for (const d of days) byDay.set(d, []);

  // Multi-day spans render as a dedicated bar (one entry per event, not per
  // day); everything else goes into its day column.
  const spans: Occurrence[] = [];
  const seenSpanIds = new Set<string>();
  for (const occ of occurrences) {
    if (occ.isMultiDaySpan) {
      if (!seenSpanIds.has(occ.event.id)) {
        seenSpanIds.add(occ.event.id);
        spans.push(occ);
      }
      continue;
    }
    byDay.get(occ.date)?.push(occ);
  }

  const spanRows: Occurrence[][] = [];
  for (const span of spans) {
    const startIdx = Math.max(days.indexOf(span.spanStart), 0);
    let placedRow = spanRows.find((row) =>
      row.every((r) => {
        const rStart = days.indexOf(r.spanStart);
        const rEnd = days.indexOf(r.spanEnd);
        const sEnd = days.indexOf(span.spanEnd);
        return sEnd < rStart || startIdx > rEnd;
      })
    );
    if (!placedRow) {
      placedRow = [];
      spanRows.push(placedRow);
    }
    placedRow.push(span);
  }
  const visibleSpanRows = spanRows.slice(0, MAX_SPAN_ROWS);
  const spanAreaHeight = visibleSpanRows.length * SPAN_ROW_HEIGHT;

  const hourMarks: number[] = [];
  for (let m = GRID_START_MINUTES; m <= GRID_END_MINUTES; m += 60) hourMarks.push(m);

  const height = HEADER_HEIGHT + spanAreaHeight + GRID_HEIGHT + 20;

  const node = (
    <div
      style={{
        width: WIDTH,
        height,
        display: "flex",
        flexDirection: "column",
        backgroundColor: "#ffffff",
        fontFamily: "Noto Sans Hebrew",
      }}
    >
      {/* Header row: day cells (Sat..Sun, left-to-right), gutter last (right edge) */}
      <div style={{ display: "flex", width: WIDTH, height: HEADER_HEIGHT }}>
        {displayDays.map((d) => {
          const isToday = d === today;
          return (
            <div
              key={d}
              style={{
                width: DAY_WIDTH,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: isToday ? "#eef2ff" : "#ffffff",
                borderBottom: "2px solid #e5e7eb",
              }}
            >
              <div style={{ fontSize: 19, color: "#6b7280", display: "flex" }}>
                {WEEKDAY_LABELS[dayOfWeek(d)]}
              </div>
              <div
                style={{
                  fontSize: 32,
                  fontWeight: 700,
                  color: isToday ? "#4f46e5" : "#111827",
                  display: "flex",
                }}
              >
                {Number(d.slice(-2))}
              </div>
              <div style={{ fontSize: 15, color: "#9ca3af", display: "flex" }}>
                {hebrewDateShort(d)}
              </div>
            </div>
          );
        })}
        <div style={{ width: GUTTER, display: "flex" }} />
      </div>

      {/* Multi-day span bars */}
      <div style={{ display: "flex", flexDirection: "column", width: WIDTH }}>
        {visibleSpanRows.map((row, rowIdx) => (
          <div key={rowIdx} style={{ display: "flex", width: WIDTH, height: SPAN_ROW_HEIGHT, position: "relative" }}>
            {row.map((span) => {
              const startIdx = Math.max(days.indexOf(span.spanStart), 0);
              const endIdx = Math.min(
                days.indexOf(span.spanEnd) === -1 ? days.length - 1 : days.indexOf(span.spanEnd),
                days.length - 1
              );
              const colors = chipColors(span.event.category);
              // Physical left edge = the later (rightward, since RTL) chrono day's column start.
              const left = colX(endIdx);
              const width = (endIdx - startIdx + 1) * DAY_WIDTH - 4;
              return (
                <div
                  key={span.event.id}
                  style={{
                    position: "absolute",
                    left: left + 3,
                    width,
                    height: SPAN_ROW_HEIGHT - 8,
                    backgroundColor: colors.bg,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 8,
                    display: "flex",
                    alignItems: "center",
                    paddingRight: 12,
                    fontSize: 17,
                    fontWeight: 600,
                    color: colors.text,
                  }}
                >
                  {truncate(span.event.title, 40)}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Time grid */}
      <div style={{ display: "flex", width: WIDTH, height: GRID_HEIGHT, position: "relative" }}>
        {/* Day columns, Sat..Sun left-to-right */}
        {displayDays.map((d) => {
          const laidOut = layoutDayColumn(byDay.get(d) ?? [], HOUR_HEIGHT);
          const isToday = d === today;
          return (
            <div
              key={d}
              style={{
                width: DAY_WIDTH,
                height: GRID_HEIGHT,
                position: "relative",
                display: "flex",
                backgroundColor: isToday ? "#fafaff" : "#ffffff",
                borderLeft: "1px solid #f1f5f9",
              }}
            >
              {laidOut.map((block) => {
                const colors = chipColors(block.occ.event.category);
                const blockWidth = DAY_WIDTH / block.numCols;
                return (
                  <div
                    key={block.occ.event.id}
                    style={{
                      position: "absolute",
                      top: block.top,
                      left: block.colIndex * blockWidth + 2,
                      width: blockWidth - 4,
                      height: block.height - 3,
                      backgroundColor: colors.bg,
                      border: `1px solid ${colors.border}`,
                      borderRadius: 6,
                      display: "flex",
                      flexDirection: "column",
                      overflow: "hidden",
                      padding: 4,
                      fontSize: 14,
                      color: colors.text,
                    }}
                  >
                    <div style={{ fontWeight: 700, display: "flex" }}>
                      {truncate(block.occ.event.title, 18)}
                    </div>
                    {block.occ.event.person && (
                      <div style={{ fontSize: 12, opacity: 0.8, display: "flex" }}>
                        {block.occ.event.person}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}

        {/* Hour gutter, right edge */}
        <div style={{ width: GUTTER, display: "flex", flexDirection: "column" }}>
          {hourMarks.map((m) => (
            <div
              key={m}
              style={{
                height: HOUR_HEIGHT,
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "flex-start",
                paddingLeft: 8,
                fontSize: 14,
                color: "#9ca3af",
              }}
            >
              {minutesToLabel(m)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return { node, width: WIDTH, height };
}
