// Fixed pair of people this calendar belongs to. Every event is tagged to
// exactly one of these — never freeform, never optional.
export type Person = "itamar" | "hadas" | "both";

export const PEOPLE_LABELS: Record<Person, string> = {
  itamar: "איתמר",
  hadas: "הדס",
  both: "שניהם",
};

const BLUE = "#3b82f6";
const BLUE_DARK = "#1d4ed8";
const PINK = "#ec4899";
const PINK_DARK = "#be185d";

/** Falls back to "both" for legacy/invalid data instead of crashing render. */
export function normalizePerson(value: string | null | undefined): Person {
  if (value === "itamar" || value === "hadas" || value === "both") return value;
  return "both";
}

export function personFill(person: Person): {
  backgroundColor?: string;
  backgroundImage?: string;
  border: string;
  text: string;
} {
  if (person === "itamar") return { backgroundColor: BLUE, border: BLUE_DARK, text: "#ffffff" };
  if (person === "hadas") return { backgroundColor: PINK, border: PINK_DARK, text: "#ffffff" };
  // "both" — a diagonal blue/pink split reads as "together" without
  // needing an arbitrary third color.
  return {
    backgroundImage: `linear-gradient(135deg, ${BLUE} 50%, ${PINK} 50%)`,
    border: "#a855f7",
    text: "#ffffff",
  };
}
