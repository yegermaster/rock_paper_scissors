import { WEEKDAY_LABELS, addDays, dayOfWeek, daysInMonth, MONTH_LABELS } from "../dates";
import { chipColors, displayStartMinutes, truncate } from "../render-helpers";
import type { Occurrence } from "../types";

const WIDTH = 1400;
const HEADER_HEIGHT = 60;
const WEEKDAY_ROW_HEIGHT = 30;
const MAX_CHIPS_PER_CELL = 3;
const CELL_HEIGHT = 130;

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
  for (let i = 0; i < cells.length; i += 7) rows.push(cells.slice(i, i + 7));

  return (
    <div
      style={{
        width: WIDTH,
        height: HEADER_HEIGHT + WEEKDAY_ROW_HEIGHT + rows.length * CELL_HEIGHT,
        display: "flex",
        flexDirection: "column",
        backgroundColor: "#ffffff",
        fontFamily: "sans-serif",
      }}
    >
      <div
        style={{
          height: HEADER_HEIGHT,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 28,
          fontWeight: 700,
          color: "#111827",
        }}
      >
        {MONTH_LABELS[month - 1]} {year}
      </div>

      <div style={{ display: "flex", width: WIDTH, height: WEEKDAY_ROW_HEIGHT }}>
        {WEEKDAY_LABELS.map((label) => (
          <div
            key={label}
            style={{
              width: cellWidth,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 12,
              color: "#6b7280",
            }}
          >
            {label}
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
                      backgroundColor: "#fafafa",
                      border: "1px solid #f1f5f9",
                    }}
                  />
                );
              }
              const dayNum = Number(cell.date.slice(-2));
              const isToday = cell.date === today;
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
                    border: "1px solid #f1f5f9",
                    backgroundColor: isToday ? "#eef2ff" : "#ffffff",
                    padding: 4,
                  }}
                >
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: isToday ? 700 : 400,
                      color: isToday ? "#4f46e5" : "#374151",
                      display: "flex",
                    }}
                  >
                    {dayNum}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", marginTop: 2 }}>
                    {visible.map((occ) => {
                      const colors = chipColors(occ.event.category);
                      const isSpanContinuation = occ.isMultiDaySpan && occ.date !== occ.spanStart;
                      return (
                        <div
                          key={occ.event.id + occ.date}
                          style={{
                            display: "flex",
                            fontSize: 10,
                            backgroundColor: colors.bg,
                            color: colors.text,
                            border: `1px solid ${colors.border}`,
                            borderRadius: 3,
                            padding: "1px 3px",
                            marginBottom: 2,
                            overflow: "hidden",
                          }}
                        >
                          {isSpanContinuation ? "→ " + truncate(occ.event.title, 12) : truncate(occ.event.title, 14)}
                        </div>
                      );
                    })}
                    {overflow > 0 && (
                      <div style={{ display: "flex", fontSize: 10, color: "#9ca3af" }}>
                        +{overflow} more
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
