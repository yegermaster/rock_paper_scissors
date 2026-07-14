// A color per event category, shown as its own header zone (week view) or
// a dot swatch (month view) so events also read as "what kind of thing is
// this" at a glance. Categories are open-ended free text, so we bucket
// common keywords into six broad categories.
//
// Deliberately vivid/saturated ("Tailwind 500-600" range), not pastel —
// an earlier pastel palette proved too washed-out to read at a glance,
// especially at small sizes. Each hue is also chosen to stay visually
// distinct from the two person colors (a fairly saturated blue #3b82f6
// and pink #ec4899) so a category zone never blends into the person zone
// next to it.
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
  work: "#0ea5e9", // vivid sky-blue — distinct from Itamar's more indigo blue
  study: "#a855f7", // vivid violet
  sport: "#f97316", // vivid orange
  leisure: "#22c55e", // vivid green
  chores: "#64748b", // slate — still a real color, just the calmest of the six
  important: "#ef4444", // vivid red — distinct from Hadas's more magenta pink
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
