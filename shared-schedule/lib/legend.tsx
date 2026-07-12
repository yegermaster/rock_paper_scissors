import { PEOPLE_LABELS, personFill, type Person } from "./people";
import { THEME } from "./theme";

export const LEGEND_HEIGHT = 56;

const ALL_PEOPLE: Person[] = ["itamar", "hadas", "both"];

export function renderPeopleLegend(width: number) {
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
      {ALL_PEOPLE.map((person) => {
        const fill = personFill(person);
        return (
          <div key={person} style={{ display: "flex", alignItems: "center", marginLeft: 26 }}>
            <div
              style={{
                width: 14,
                height: 14,
                borderRadius: 999,
                marginLeft: 8,
                display: "flex",
                ...(fill.backgroundColor ? { backgroundColor: fill.backgroundColor } : { backgroundImage: fill.backgroundImage }),
              }}
            />
            <div style={{ fontSize: 14, fontWeight: 600, color: THEME.textMuted, display: "flex" }}>
              {PEOPLE_LABELS[person]}
            </div>
          </div>
        );
      })}
    </div>
  );
}
