import { colorForCategory } from "./colors";
import { he } from "./bidi";
import { THEME } from "./theme";
import type { Occurrence } from "./types";

export const LEGEND_HEIGHT = 56;

export function categoriesIn(occurrences: Occurrence[]): string[] {
  return [...new Set(occurrences.map((o) => o.event.category))].sort();
}

export function renderLegend(categories: string[], width: number) {
  if (categories.length === 0) return null;
  return (
    <div
      style={{
        display: "flex",
        width,
        height: LEGEND_HEIGHT,
        alignItems: "center",
        justifyContent: "center",
        borderTop: `1px solid ${THEME.border}`,
      }}
    >
      {categories.map((cat) => (
        <div key={cat} style={{ display: "flex", alignItems: "center", marginLeft: 26 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: 999,
              backgroundColor: colorForCategory(cat).bg,
              display: "flex",
              marginLeft: 8,
            }}
          />
          <div style={{ fontSize: 14, fontWeight: 600, color: THEME.textMuted, display: "flex" }}>{he(cat)}</div>
        </div>
      ))}
    </div>
  );
}
