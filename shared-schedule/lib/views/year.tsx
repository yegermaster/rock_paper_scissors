import { dayOfWeek, daysInMonth, MONTH_LABELS } from "../dates";
import { he } from "../bidi";
import { chipColors } from "../render-helpers";
import type { Occurrence } from "../types";

const WIDTH = 1900;
const HEADER_HEIGHT = 110;
const COLS = 4;
const ROWS = 3;
const MINI_WIDTH = WIDTH / COLS;
const MINI_HEIGHT = 340;
const DOT_SIZE = 7;
const ACCENT = "#4f46e5";

export function renderYearView(yearStart: string, occurrences: Occurrence[], today: string) {
  const year = Number(yearStart.slice(0, 4));

  const byDate = new Map<string, Occurrence[]>();
  for (const occ of occurrences) {
    if (!byDate.has(occ.date)) byDate.set(occ.date, []);
    byDate.get(occ.date)!.push(occ);
  }

  const height = HEADER_HEIGHT + ROWS * MINI_HEIGHT;

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
      <div
        style={{
          height: HEADER_HEIGHT,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 44,
          fontWeight: 800,
          color: "#ffffff",
          backgroundImage: `linear-gradient(135deg, ${ACCENT}, #7c3aed)`,
        }}
      >
        {year}
      </div>

      <div style={{ display: "flex", flexDirection: "column", width: WIDTH }}>
        {Array.from({ length: ROWS }, (_, rowIdx) => (
          <div key={rowIdx} style={{ display: "flex", width: WIDTH, height: MINI_HEIGHT }}>
            {Array.from({ length: COLS }, (_, colIdx) => {
              // RTL: January sits at the RIGHT edge of the first row.
              const monthIdx = rowIdx * COLS + (COLS - 1 - colIdx); // 0-11
              const month1 = monthIdx + 1;
              const totalDays = daysInMonth(year, month1);
              const monthStart = `${year}-${String(month1).padStart(2, "0")}-01`;
              const firstWeekday = dayOfWeek(monthStart);
              const totalCells = Math.ceil((firstWeekday + totalDays) / 7) * 7;
              const dayCellSize = (MINI_WIDTH - 16) / 7;
              // Months spanning 6 week-rows must still fit inside MINI_HEIGHT.
              const rowHeight = 44;

              const cells: (number | null)[] = [];
              for (let i = 0; i < totalCells; i++) {
                const dayNum = i - firstWeekday + 1;
                cells.push(dayNum < 1 || dayNum > totalDays ? null : dayNum);
              }
              const miniRows: (number | null)[][] = [];
              for (let i = 0; i < cells.length; i += 7) miniRows.push(cells.slice(i, i + 7));

              return (
                <div
                  key={colIdx}
                  style={{
                    width: MINI_WIDTH,
                    height: MINI_HEIGHT,
                    display: "flex",
                    flexDirection: "column",
                    padding: 12,
                    border: "1px solid #f1f5f9",
                  }}
                >
                  <div style={{ fontSize: 20, fontWeight: 800, color: ACCENT, display: "flex", justifyContent: "flex-end", marginBottom: 6 }}>
                    {he(MONTH_LABELS[monthIdx])}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column" }}>
                    {miniRows.map((mrow, mri) => (
                      <div key={mri} style={{ display: "flex" }}>
                        {[...mrow].reverse().map((dayNum, di) => {
                          if (dayNum === null) {
                            return <div key={di} style={{ width: dayCellSize, height: rowHeight, display: "flex" }} />;
                          }
                          const date = `${year}-${String(month1).padStart(2, "0")}-${String(dayNum).padStart(2, "0")}`;
                          const isToday = date === today;
                          const dayOccs = byDate.get(date) ?? [];
                          const dotCategories = [...new Set(dayOccs.map((o) => o.event.category))].slice(0, 3);

                          return (
                            <div
                              key={di}
                              style={{
                                width: dayCellSize,
                                height: rowHeight,
                                display: "flex",
                                flexDirection: "column",
                                alignItems: "center",
                                justifyContent: "center",
                              }}
                            >
                              <div
                                style={{
                                  fontSize: 13,
                                  fontWeight: isToday ? 700 : 500,
                                  color: isToday ? "#ffffff" : "#4b5563",
                                  backgroundColor: isToday ? ACCENT : "transparent",
                                  borderRadius: 999,
                                  width: 20,
                                  height: 20,
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                }}
                              >
                                {dayNum}
                              </div>
                              <div style={{ display: "flex", marginTop: 2 }}>
                                {dotCategories.map((cat) => (
                                  <div
                                    key={cat}
                                    style={{
                                      width: DOT_SIZE,
                                      height: DOT_SIZE,
                                      borderRadius: 999,
                                      backgroundColor: chipColors(cat).border,
                                      marginLeft: 1,
                                      display: "flex",
                                    }}
                                  />
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );

  return { node, width: WIDTH, height };
}
