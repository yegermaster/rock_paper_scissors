import { WEEKDAY_LABELS, addDays, dayOfWeek } from "../dates";
import { hebrewDateShort } from "../hebrew-date";
import { he } from "../bidi";
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
const HEADER_HEIGHT = 120;
const SPAN_ROW_HEIGHT = 44;
const MAX_SPAN_ROWS = 2;
const ACCENT = "#4f46e5";

// Hebrew calendar convention: the week flows right-to-left — Sunday is the
// RIGHTMOST column (next to the hour gutter on the right edge), Saturday
// leftmost. `colX` maps a chronological day index (0=Sun..6=Sat) to the
// physical left-x of its column, for absolutely-positioned span bars.
function colX(chronoIndex: number): number {
  return (6 - chronoIndex) * DAY_WIDTH;
}

function isWeekend(ymd: string): boolean {
  const dow = dayOfWeek(ymd);
  return dow === 5 || dow === 6; // Fri/Sat — the Israeli weekend
}

export function renderWeekView(
  weekStart: string,
  occurrences: Occurrence[],
  today: string
) {
  const days = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));
  const displayDays = [...days].reverse(); // Sat..Sun in physical left-to-right order

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

  const height = HEADER_HEIGHT + spanAreaHeight + GRID_HEIGHT + 24;

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
      {/* Header: day cells Sat..Sun (physical LTR), hour gutter at right edge */}
      <div
        style={{
          display: "flex",
          width: WIDTH,
          height: HEADER_HEIGHT,
          backgroundImage: `linear-gradient(135deg, ${ACCENT}, #7c3aed)`,
        }}
      >
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
              }}
            >
              <div style={{ fontSize: 19, color: "rgba(255,255,255,0.75)", display: "flex" }}>
                {he(WEEKDAY_LABELS[dayOfWeek(d)])}
              </div>
              <div
                style={{
                  fontSize: 34,
                  fontWeight: 800,
                  color: "#ffffff",
                  display: "flex",
                  backgroundColor: isToday ? "rgba(255,255,255,0.25)" : "transparent",
                  borderRadius: 999,
                  width: 56,
                  height: 56,
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {Number(d.slice(-2))}
              </div>
              <div style={{ fontSize: 14, color: "rgba(255,255,255,0.7)", display: "flex" }}>
                {he(hebrewDateShort(d))}
              </div>
            </div>
          );
        })}
        <div style={{ width: GUTTER, display: "flex" }} />
      </div>

      {/* Multi-day span bars */}
      <div style={{ display: "flex", flexDirection: "column", width: WIDTH, backgroundColor: "#fafaff" }}>
        {visibleSpanRows.map((row, rowIdx) => (
          <div key={rowIdx} style={{ display: "flex", width: WIDTH, height: SPAN_ROW_HEIGHT, position: "relative" }}>
            {row.map((span) => {
              const startIdx = Math.max(days.indexOf(span.spanStart), 0);
              const endIdx = Math.min(
                days.indexOf(span.spanEnd) === -1 ? days.length - 1 : days.indexOf(span.spanEnd),
                days.length - 1
              );
              const colors = chipColors(span.event.category);
              // RTL grid: the chronologically-later day sits further LEFT.
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
                    borderRadius: 10,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end",
                    paddingRight: 14,
                    fontSize: 18,
                    fontWeight: 700,
                    color: colors.text,
                  }}
                >
                  {he(truncate(span.event.title, 40))}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Time grid: day columns Sat..Sun, hour gutter at right edge */}
      <div style={{ display: "flex", width: WIDTH, height: GRID_HEIGHT, position: "relative" }}>
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
                backgroundColor: isToday ? "#eef2ff" : isWeekend(d) ? "#f8fafc" : "#ffffff",
                borderLeft: "1px solid #eef2f7",
              }}
            >
              {/* Hour gridlines */}
              {hourMarks.map((m, i) =>
                i === 0 ? null : (
                  <div
                    key={m}
                    style={{
                      position: "absolute",
                      top: i * HOUR_HEIGHT,
                      left: 0,
                      width: DAY_WIDTH,
                      height: 1,
                      backgroundColor: "#eef2f7",
                      display: "flex",
                    }}
                  />
                )
              )}
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
                      borderRadius: 8,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "flex-end",
                      overflow: "hidden",
                      padding: 8,
                      fontSize: 16,
                      color: colors.text,
                    }}
                  >
                    <div style={{ fontWeight: 800, display: "flex" }}>
                      {he(truncate(block.occ.event.title, 18))}
                    </div>
                    {block.occ.event.start_time && (
                      <div style={{ fontSize: 13, opacity: 0.9, display: "flex" }}>
                        {minutesToLabel(
                          Number(block.occ.event.start_time.slice(0, 2)) * 60 +
                            Number(block.occ.event.start_time.slice(3, 5))
                        )}
                      </div>
                    )}
                    {block.occ.event.person && (
                      <div style={{ fontSize: 13, opacity: 0.9, display: "flex" }}>
                        {he(block.occ.event.person)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}

        {/* Hour gutter, right edge */}
        <div style={{ width: GUTTER, display: "flex", flexDirection: "column", borderLeft: "1px solid #eef2f7" }}>
          {hourMarks.map((m) => (
            <div
              key={m}
              style={{
                height: HOUR_HEIGHT,
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "center",
                paddingTop: 4,
                fontSize: 14,
                fontWeight: 600,
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
