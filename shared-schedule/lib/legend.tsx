import { he } from "./bidi";
import { PEOPLE_LABELS, personFill, type Person } from "./people";
import { CATEGORY_COLORS, CATEGORY_LABELS, type EventCategoryBucket } from "./category-color";
import { THEME } from "./theme";

const ROW_HEIGHT = 56;
export const LEGENDS_HEIGHT = ROW_HEIGHT * 2;

const ALL_PEOPLE: Person[] = ["itamar", "hadas", "both"];
const ALL_CATEGORIES: EventCategoryBucket[] = ["work", "study", "sport", "leisure", "chores", "important"];

function legendRow(
  items: { key: string; label: string; backgroundColor?: string; backgroundImage?: string }[],
  width: number,
  withTopBorder: boolean
) {
  return (
    <div
      style={{
        display: "flex",
        width,
        height: ROW_HEIGHT,
        alignItems: "center",
        justifyContent: "center",
        borderTop: withTopBorder ? `1px solid ${THEME.border}` : "none",
      }}
    >
      {items.map((item) => (
        <div key={item.key} style={{ display: "flex", alignItems: "center", marginLeft: 24 }}>
          <div
            style={{
              width: 14,
              height: 14,
              borderRadius: 999,
              marginLeft: 8,
              display: "flex",
              backgroundColor: item.backgroundColor,
              backgroundImage: item.backgroundImage,
            }}
          />
          <div style={{ fontSize: 16, fontWeight: 600, color: THEME.textMuted, display: "flex" }}>
            {he(item.label)}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Two legend rows: who it's for (person color) and what kind of thing it
 * is (category accent color) — matches the two independent color signals
 * used on every event block. */
export function renderLegends(width: number) {
  const peopleItems = ALL_PEOPLE.map((person) => {
    const fill = personFill(person);
    return { key: person, label: PEOPLE_LABELS[person], backgroundColor: fill.backgroundColor, backgroundImage: fill.backgroundImage };
  });
  const categoryItems = ALL_CATEGORIES.map((cat) => ({
    key: cat,
    label: CATEGORY_LABELS[cat],
    backgroundColor: CATEGORY_COLORS[cat],
    backgroundImage: `linear-gradient(${CATEGORY_COLORS[cat]}, ${CATEGORY_COLORS[cat]})`,
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", width }}>
      {legendRow(peopleItems, width, true)}
      {legendRow(categoryItems, width, false)}
    </div>
  );
}
