import { WEEKDAY_LABELS, dayOfWeek, daysInMonth, MONTH_LABELS } from "../dates";
import { hebrewDateShort } from "../hebrew-date";
import { he, heWrap } from "../bidi";
import { THEME } from "../theme";
import { renderLegends, LEGENDS_HEIGHT } from "../legend";
import { normalizePerson, personFill } from "../people";
import { categoryAccentColor } from "../category-color";
import { computeOverlapKeys, displayStartMinutes, occurrenceKey } from "../render-helpers";
import type { Occurrence } from "../types";

// Matches the calm-yellow overlap notice used in the week view.
const OVERLAP_YELLOW = "#fde047";

// Widened to match the week view — gives each chip enough room that most
// real titles fit on one line at a generous length instead of the old
// 10-12 character cutoff.
const WIDTH = 2600;
const HEADER_HEIGHT = 96;
const WEEKDAY_ROW_HEIGHT = 54;
const MAX_CHIPS_PER_CELL = 3;
const CELL_HEIGHT = 280;
// Full title up to this many characters — a chip wraps to a 2nd line for
// a long title rather than cutting off early (capped at 2, not the week
// view's 4, so the month grid stays a compact overview).
const MAX_TITLE_CHARS = 30;
const CHIP_LINE_HEIGHT = 24;

// Hebrew calendar convention: weeks flow right-to-left — Sunday is the
// rightmost cell of each row. Reverse each row's physical order.
function toRtlRow<T>(row: T[]): T[] {
  return [...row].reverse();
}

export function renderMonthView(monthStart: string, occurrences: Occurrence[], today: string) {
  const [year, month] = monthStart.split("-").map(Number);
  const totalDays = daysInMonth(year, month);
  const firstWeekday = dayOfWeek(monthStart);
  const totalCells = Math.ceil((firstWeekday + totalDays) / 7) * 7;
  const cellWidth = WIDTH / 7;

  const byDay = new Map<string, Occurrence[]>();
  for (const occ of occurrences) {
    if (!byDay.has(occ.date)) byDay.set(occ.date, []);
    byDay.get(occ.date)!.push(occ);
  }
  for (const list of byDay.values()) {
    list.sort((a, b) => displayStartMinutes(a) - displayStartMinutes(b));
  }

  const cells: { date: string | null; inMonth: boolean }[] = [];
  for (let i = 0; i < totalCells; i++) {
    const dayNum = i - firstWeekday + 1;
    if (dayNum < 1 || dayNum > totalDays) {
      cells.push({ date: null, inMonth: false });
    } else {
      const date = `${year}-${String(month).padStart(2, "0")}-${String(dayNum).padStart(2, "0")}`;
      cells.push({ date, inMonth: true });
    }
  }
  const rows: (typeof cells)[] = [];
  for (let i = 0; i < cells.length; i += 7) rows.push(toRtlRow(cells.slice(i, i + 7)));

  const height = HEADER_HEIGHT + WEEKDAY_ROW_HEIGHT + rows.length * CELL_HEIGHT + LEGENDS_HEIGHT;

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
      <div
        style={{
          height: HEADER_HEIGHT,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 38,
          fontWeight: 800,
          color: THEME.text,
          borderBottom: `1px solid ${THEME.border}`,
        }}
      >
        {he(`${MONTH_LABELS[month - 1]} ${year}`)}
      </div>

      <div style={{ display: "flex", width: WIDTH, height: WEEKDAY_ROW_HEIGHT, backgroundColor: THEME.panelAlt }}>
        {toRtlRow(WEEKDAY_LABELS).map((label, i) => (
          <div
            key={i}
            style={{
              width: cellWidth,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 20,
              fontWeight: 700,
              color: THEME.textMuted,
            }}
          >
            {he(label)}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", flexDirection: "column", width: WIDTH }}>
        {rows.map((row, rowIdx) => (
          <div key={rowIdx} style={{ display: "flex", width: WIDTH, height: CELL_HEIGHT }}>
            {row.map((cell, cellIdx) => {
              if (!cell.date) {
                return (
                  <div
                    key={cellIdx}
                    style={{
                      width: cellWidth,
                      height: CELL_HEIGHT,
                      display: "flex",
                      backgroundColor: THEME.panelAlt,
                      border: `1px solid ${THEME.border}`,
                    }}
                  />
                );
              }
              const dayNum = Number(cell.date.slice(-2));
              const isToday = cell.date === today;
              const dow = dayOfWeek(cell.date);
              const isWeekend = dow === 5 || dow === 6;
              const dayOccs = byDay.get(cell.date) ?? [];
              const visible = dayOccs.slice(0, MAX_CHIPS_PER_CELL);
              const overflow = dayOccs.length - visible.length;

              return (
                <div
                  key={cellIdx}
                  style={{
                    width: cellWidth,
                    height: CELL_HEIGHT,
                    display: "flex",
                    flexDirection: "column",
                    border: `1px solid ${THEME.border}`,
                    backgroundColor: isToday ? THEME.accentSoft : isWeekend ? THEME.weekendTint : "transparent",
                    padding: 8,
                  }}
                >
                  {/* Day number on the RIGHT (RTL reading start), Hebrew date left */}
                  <div style={{ display: "flex", flexDirection: "row-reverse", alignItems: "center", justifyContent: "space-between" }}>
                    <div
                      style={{
                        fontSize: 25,
                        fontWeight: isToday ? 800 : 500,
                        color: isToday ? "#ffffff" : THEME.text,
                        display: "flex",
                        backgroundColor: isToday ? THEME.accent : "transparent",
                        width: 44,
                        height: 44,
                        borderRadius: 999,
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {dayNum}
                    </div>
                    <div style={{ fontSize: 16, color: THEME.textFaint, display: "flex" }}>
                      {he(hebrewDateShort(cell.date))}
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", marginTop: 6 }}>
                    {(() => {
                      const overlapKeys = computeOverlapKeys(dayOccs);
                      return visible.map((occ) => {
                        const fill = personFill(normalizePerson(occ.event.person));
                        const isOverlap = overlapKeys.has(occurrenceKey(occ));
                        const isSpanContinuation = occ.isMultiDaySpan && occ.date !== occ.spanStart;
                        const accent = categoryAccentColor(occ.event.category);
                        // Chip usable width minus cell padding, the dot,
                        // its margin, and the chip's own padding.
                        const usableWidth = cellWidth - 16 - 13 - 7 - 18;
                        const charsPerLine = Math.max(10, Math.floor(usableWidth / 12.5));
                        const lines = heWrap(occ.event.title, charsPerLine, MAX_TITLE_CHARS).slice(0, 2);
                        return (
                          <div
                            key={occ.event.id + occ.date}
                            style={{
                              display: "flex",
                              alignItems: "flex-start",
                              justifyContent: "flex-end",
                              fontSize: 20,
                              fontWeight: 700,
                              backgroundColor: fill.backgroundColor,
                              backgroundImage: fill.backgroundImage,
                              color: fill.text,
                              borderRadius: 6,
                              border: isOverlap ? `3px solid ${OVERLAP_YELLOW}` : "1px solid rgba(255,255,255,0.15)",
                              padding: "5px 9px",
                              marginBottom: 5,
                              overflow: "hidden",
                            }}
                          >
                            {/* Category dot: a solid ring-outlined swatch
                                (not a thin edge border) so it stays visible
                                even when the category color's hue is close
                                to the person color's. */}
                            <div
                              style={{
                                width: 13,
                                height: 13,
                                borderRadius: 999,
                                backgroundColor: accent,
                                border: "2px solid rgba(0,0,0,0.4)",
                                marginLeft: 7,
                                marginTop: 5,
                                flexShrink: 0,
                                display: "flex",
                              }}
                            />
                            <div style={{ display: "flex", flexDirection: "column" }}>
                              {lines.map((line, i) => (
                                <div key={i} style={{ display: "flex", whiteSpace: "nowrap", height: CHIP_LINE_HEIGHT }}>
                                  {i === lines.length - 1 && isSpanContinuation ? `${line} ←` : line}
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      });
                    })()}
                    {overflow > 0 && (
                      <div style={{ display: "flex", justifyContent: "flex-end", fontSize: 17, fontWeight: 600, color: THEME.textFaint }}>
                        {he(`ועוד ${overflow}`)}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {renderLegends(WIDTH)}
    </div>
  );

  return { node, width: WIDTH, height };
}
