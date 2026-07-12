// A small accent color per event category, shown as a thin stripe on top
// of the person-based fill (blue/pink/gradient) so events also read as
// "what kind of thing is this" at a glance. Categories are open-ended free
// text, so we match common keywords — same idea as the old category-color
// scheme, just layered as an accent instead of being the main fill.
//
// Rendered as flat CSS shapes (never text/emoji): the fonts loaded for the
// calendar image only cover Hebrew + Latin glyphs, so any emoji character
// would render as a missing-glyph box under Satori.
const CATEGORY_HUES: { pattern: RegExp; hue: number }[] = [
  { pattern: /עבודה|משמרת|משרד|פגיש|work|shift|meeting/i, hue: 210 }, // sky
  { pattern: /ארוח|אוכל|מסעד|דינר|בישול|קפה|dinner|lunch|food|coffee/i, hue: 27 }, // orange/tan
  { pattern: /ספורט|ריצה|כושר|אימון|שחיה|יוגה|ג'יטסו|gym|run|sport|yoga/i, hue: 152 }, // green
  { pattern: /ריקוד|חוג|שיעור|מוזיקה|אמנות|תחביב|dance|class|hobby|music/i, hue: 43 }, // gold
  { pattern: /לימוד|אוניברסיט|מבחן|קורס|study|exam|school|course/i, hue: 265 }, // violet
  { pattern: /טיול|חופש|נסיעה|טיסה|מלון|trip|vacation|travel|hotel|flight/i, hue: 174 }, // teal
  { pattern: /רופא|תור|בריאות|שיניים|doctor|health|dentist/i, hue: 355 }, // deep red
  { pattern: /קניות|סופר|shopping/i, hue: 320 }, // magenta
  { pattern: /משפחה|יום הולדת|מסיב|family|birthday|party/i, hue: 45 }, // gold
  { pattern: /חבר|social|friend/i, hue: 15 }, // warm orange-red
  { pattern: /משימ|סידור|todo|task|errand/i, hue: 200 }, // steel blue
];

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

export function categoryAccentColor(category: string): string {
  const normalized = category.trim().toLowerCase();
  const semantic = CATEGORY_HUES.find((c) => c.pattern.test(normalized));
  const hue = semantic ? semantic.hue : hashString(normalized) % 360;
  return `hsl(${hue}, 85%, 68%)`;
}
