import { WEEKDAY_LABELS, addDays, dayOfWeek } from "../dates";
import { hebrewDateShort } from "../hebrew-date";
import { he } from "../bidi";
import { THEME } from "../theme";
import { renderLegends, LEGENDS_HEIGHT } from "../legend";
import { normalizePerson, personFill } from "../people";
import { categoryAccentColor } from "../category-color";
import {
  ABSOLUTE_DAY_END,
  computeGridRange,
  computeOverlapKeys,
  displayDuration,
  displayStartMinutes,
  layoutDayColumn,
  minutesToLabel,
  occurrenceKey,
  truncate,
} from "../render-helpers";
import type { Occurrence } from "../types";

// A calm, friendly yellow for the overlap notice — not the same alarming
// amber used for errors elsewhere.
const OVERLAP_YELLOW = "#fde047";

const WIDTH = 1900;
const GUTTER = 100;
const DAY_WIDTH = (WIDTH - GUTTER) / 7;
const HOUR_HEIGHT = 84;
const HEADER_HEIGHT = 116;
const SPAN_ROW_HEIGHT = 56;
const MAX_SPAN_ROWS = 2;
// Fixed-height category bar at the top of every event block — a real zone
// (not a proportional sliver), so it stays a consistent, legible size no
// matter how tall the block itself gets for a long event.
const HEADER_BAR_HEIGHT = 46;

// Hebrew calendar convention: the week flows right-to-left — Sunday is the
// RIGHTMOST column, Saturday leftmost. The hour gutter sits at the LEFT
// edge. `colX` maps a chronological day index (0=Sun..6=Sat) to the
// physical left-x of its column (gutter width excluded — added by caller).
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

  // Shrink-to-fit hour range: a default daytime window (07:00-22:00),
  // widened only when an actual occurrence falls outside it (an early
  // flight, a night shift) — showing all 24 hours unconditionally made a
  // typical, mostly-daytime week awkwardly tall for no reason.
  const { start: gridStart, end: gridEnd } = computeGridRange(occurrences);
  const GRID_HEIGHT = ((gridEnd - gridStart) / 60) * HOUR_HEIGHT;

  // Exclusive of gridEnd: one mark per row, matching GRID_HEIGHT exactly
  // (an inclusive loop here would add a phantom extra label row).
  const hourMarks: number[] = [];
  for (let m = gridStart; m < gridEnd; m += 60) hourMarks.push(m);

  const height = HEADER_HEIGHT + spanAreaHeight + GRID_HEIGHT + LEGENDS_HEIGHT + 8;

  const node = (
    <div
      style={{
        width: WIDTH,
        height,
        display: "flex",
        flexDirection: "column",
        backgroundColor: THEME.bg,
        fontFamily: "Noto Sans Hebrew",
      }}
    >
      {/* Header: gutter spacer at left edge, then day cells Sat..Sun */}
      <div
        style={{
          display: "flex",
          width: WIDTH,
          height: HEADER_HEIGHT,
          borderBottom: `1px solid ${THEME.border}`,
        }}
      >
        <div style={{ width: GUTTER, display: "flex" }} />
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
              <div style={{ fontSize: 24, color: THEME.textMuted, display: "flex" }}>
                {he(WEEKDAY_LABELS[dayOfWeek(d)])}
              </div>
              <div
                style={{
                  fontSize: 40,
                  fontWeight: 800,
                  color: isToday ? "#ffffff" : THEME.text,
                  display: "flex",
                  backgroundColor: isToday ? THEME.accent : "transparent",
                  borderRadius: 999,
                  width: 66,
                  height: 66,
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {Number(d.slice(-2))}
              </div>
              <div style={{ fontSize: 17, color: THEME.textFaint, display: "flex" }}>
                {he(hebrewDateShort(d))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Multi-day span bars */}
      <div style={{ display: "flex", flexDirection: "column", width: WIDTH, backgroundColor: THEME.panelAlt }}>
        {visibleSpanRows.map((row, rowIdx) => (
          <div key={rowIdx} style={{ display: "flex", width: WIDTH, height: SPAN_ROW_HEIGHT, position: "relative" }}>
            {row.map((span) => {
              const startIdx = Math.max(days.indexOf(span.spanStart), 0);
              const endIdx = Math.min(
                days.indexOf(span.spanEnd) === -1 ? days.length - 1 : days.indexOf(span.spanEnd),
                days.length - 1
              );
              const fill = personFill(normalizePerson(span.event.person));
              const accent = categoryAccentColor(span.event.category);
              // RTL grid: the chronologically-later day sits further LEFT.
              const left = GUTTER + colX(endIdx);
              const width = (endIdx - startIdx + 1) * DAY_WIDTH - 4;
              return (
                <div
                  key={span.event.id}
                  style={{
                    position: "absolute",
                    left: left + 3,
                    width,
                    height: SPAN_ROW_HEIGHT - 8,
                    display: "flex",
                    flexDirection: "column",
                    borderRadius: 10,
                    overflow: "hidden",
                  }}
                >
                  {/* Category band: a solid strip (not a thin border) with
                      its own dark separator, so it stays visible even when
                      the category color's hue is close to the person
                      color's (pastel blue "work" on a blue Itamar block,
                      for example). */}
                  <div style={{ height: 10, width: "100%", backgroundColor: accent, borderBottom: "2px solid rgba(0,0,0,0.35)", display: "flex", flexShrink: 0 }} />
                  <div
                    style={{
                      flex: 1,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "flex-end",
                      backgroundColor: fill.backgroundColor,
                      backgroundImage: fill.backgroundImage,
                      paddingRight: 16,
                      fontSize: 24,
                      fontWeight: 700,
                      color: fill.text,
                    }}
                  >
                    {he(truncate(span.event.title, 34))}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Time grid: hour gutter at left edge, then day columns Sat..Sun */}
      <div style={{ display: "flex", width: WIDTH, height: GRID_HEIGHT, position: "relative" }}>
        <div style={{ width: GUTTER, display: "flex", flexDirection: "column", borderRight: `1px solid ${THEME.gridLine}` }}>
          {hourMarks.map((m) => (
            <div
              key={m}
              style={{
                height: HOUR_HEIGHT,
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "center",
                paddingTop: 4,
                fontSize: 20,
                fontWeight: 600,
                color: THEME.textFaint,
              }}
            >
              {minutesToLabel(m)}
            </div>
          ))}
        </div>

        {displayDays.map((d) => {
          const dayOccs = byDay.get(d) ?? [];
          const laidOut = layoutDayColumn(dayOccs, HOUR_HEIGHT, gridStart, gridEnd);
          const overlapKeys = computeOverlapKeys(dayOccs);
          const isToday = d === today;
          return (
            <div
              key={d}
              style={{
                width: DAY_WIDTH,
                height: GRID_HEIGHT,
                position: "relative",
                display: "flex",
                backgroundColor: isToday ? THEME.accentSoft : isWeekend(d) ? THEME.weekendTint : "transparent",
                borderLeft: `1px solid ${THEME.gridLine}`,
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
                      backgroundColor: THEME.gridLine,
                      display: "flex",
                    }}
                  />
                )
              )}
              {laidOut.map((block) => {
                const fill = personFill(normalizePerson(block.occ.event.person));
                const isOverlap = overlapKeys.has(occurrenceKey(block.occ));
                const blockWidth = DAY_WIDTH / block.numCols;
                // A block whose duration runs past midnight gets clamped to
                // the day's bottom edge (occurrences are per-calendar-day).
                // Folded into the time text itself so it's never lost when
                // the visible sliver is only an hour tall.
                const spillsPastMidnight =
                  displayStartMinutes(block.occ) + displayDuration(block.occ) > ABSOLUTE_DAY_END;
                const accent = categoryAccentColor(block.occ.event.category);
                // Narrower blocks (two occurrences sharing a column) need a
                // shorter title or the bold 26px text wraps and overflows.
                const maxTitleChars = Math.max(4, Math.min(15, Math.floor((blockWidth - 44) / 14)));
                const timeLabel = block.occ.event.start_time
                  ? minutesToLabel(
                      Number(block.occ.event.start_time.slice(0, 2)) * 60 +
                        Number(block.occ.event.start_time.slice(3, 5))
                    )
                  : null;
                return (
                  <div
                    key={block.occ.event.id}
                    style={{
                      position: "absolute",
                      top: block.top,
                      left: block.colIndex * blockWidth + 2,
                      width: blockWidth - 4,
                      height: block.height - 3,
                      display: "flex",
                      flexDirection: "column",
                      borderRadius: 8,
                      border: isOverlap ? `4px solid ${OVERLAP_YELLOW}` : "1px solid rgba(255,255,255,0.15)",
                      overflow: "hidden",
                    }}
                  >
                    {/* Category header bar: a real, generously-sized zone
                        (not a hairline accent) carrying the time — stays
                        clearly legible on its own regardless of how tall
                        the person-color body below ends up being for a
                        long event, and gives the time a dedicated,
                        high-contrast spot instead of fading into the fill. */}
                    <div
                      style={{
                        height: HEADER_BAR_HEIGHT,
                        width: "100%",
                        flexShrink: 0,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "flex-end",
                        backgroundColor: accent,
                        borderBottom: "2px solid rgba(0,0,0,0.35)",
                        padding: "0 10px",
                        fontSize: 20,
                        fontWeight: 800,
                        color: "#1a2332",
                      }}
                    >
                      {timeLabel && he(spillsPastMidnight ? `${timeLabel} …` : timeLabel)}
                    </div>
                    <div
                      style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-end",
                        justifyContent: "center",
                        overflow: "hidden",
                        backgroundColor: fill.backgroundColor,
                        backgroundImage: fill.backgroundImage,
                        padding: 10,
                        fontSize: 26,
                        fontWeight: 800,
                        color: fill.text,
                      }}
                    >
                      <div style={{ display: "flex", whiteSpace: "nowrap" }}>
                        {he(truncate(block.occ.event.title, maxTitleChars))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      {renderLegends(WIDTH)}
    </div>
  );

  return { node, width: WIDTH, height };
}
