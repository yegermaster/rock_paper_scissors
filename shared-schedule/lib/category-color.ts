// A small accent color per event category, shown as a thin stripe on top
// of the person-based fill (blue/pink/gradient) so events also read as
// "what kind of thing is this" at a glance. Categories are open-ended free
// text, so we bucket common keywords into six broad, low-eye-strain
// pastel categories (palette chosen for calm long-session readability,
// not saturated/loud colors).
//
// Rendered as flat CSS shapes (never text/emoji): the fonts loaded for the
// calendar image only cover Hebrew + Latin glyphs, so any emoji character
// would render as a missing-glyph box under Satori.
export type EventCategoryBucket = "work" | "study" | "sport" | "leisure" | "chores" | "important";

export const CATEGORY_LABELS: Record<EventCategoryBucket, string> = {
  work: "עבודה",
  study: "לימודים",
  sport: "ספורט",
  leisure: "פנאי וחברים",
  chores: "משימות בית",
  important: "אירועים חשובים",
};

export const CATEGORY_COLORS: Record<EventCategoryBucket, string> = {
  work: "#A9CCE3", // pastel blue — cool, easy on the eyes for work items
  study: "#D7BDE2", // light purple — distinct from work, avoids confusion
  sport: "#F5CBA7", // peach orange — warm/energetic but low-saturation
  leisure: "#A3E4D7", // mint green — soothing, separates downtime from tasks
  chores: "#D5D8DC", // blue-gray — neutral, low-priority routine tasks
  important: "#F5B7B1", // delicate red-pink — noticeable without being alarming
};

const CATEGORY_PATTERNS: { pattern: RegExp; category: EventCategoryBucket }[] = [
  { pattern: /עבודה|משמרת|משרד|פגיש|work|shift|meeting/i, category: "work" },
  { pattern: /לימוד|אוניברסיט|מבחן|קורס|study|exam|school|course/i, category: "study" },
  { pattern: /ספורט|ריצה|כושר|אימון|שחיה|יוגה|ג'יטסו|gym|run|sport|yoga/i, category: "sport" },
  { pattern: /רופא|תור|בריאות|שיניים|doctor|health|dentist|חשוב|important/i, category: "important" },
  { pattern: /משימ|סידור|todo|task|errand|קניות|סופר|shopping|בית|chores/i, category: "chores" },
  // Anything else (dinner, hobbies, trips, family, friends...) reads as
  // unstructured downtime — bucketed as "leisure and friends" by default.
];

export function categoryBucketOf(category: string): EventCategoryBucket {
  const normalized = category.trim().toLowerCase();
  const match = CATEGORY_PATTERNS.find((c) => c.pattern.test(normalized));
  return match ? match.category : "leisure";
}

export function categoryAccentColor(category: string): string {
  return CATEGORY_COLORS[categoryBucketOf(category)];
}
