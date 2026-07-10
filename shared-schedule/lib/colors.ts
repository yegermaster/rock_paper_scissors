// Category colors: semantic first, hash fallback second.
//
// Categories are open-ended free text (mostly Hebrew, from Gemini), so we
// match common keywords to intuitive colors — work is blue, food is orange,
// sport is green, etc. Anything unrecognized gets a stable hash-picked hue
// so the same category always renders the same color.

const SEMANTIC_HUES: { pattern: RegExp; hue: number }[] = [
  { pattern: /עבודה|משמרת|משרד|פגיש|work|shift|meeting/i, hue: 221 }, // blue
  { pattern: /ארוח|אוכל|מסעד|דינר|בישול|קפה|dinner|lunch|food/i, hue: 27 }, // orange
  { pattern: /ספורט|ריצה|כושר|אימון|שחיה|יוגה|gym|run|sport/i, hue: 145 }, // green
  { pattern: /ריקוד|חוג|שיעור|מוזיקה|אמנות|dance|class|hobby/i, hue: 272 }, // purple
  { pattern: /טיול|חופש|נסיעה|טיסה|trip|vacation|travel/i, hue: 174 }, // teal
  { pattern: /רופא|תור|בריאות|שיניים|doctor|health|dentist/i, hue: 2 }, // red
  { pattern: /קניות|סופר|shopping/i, hue: 330 }, // pink
  { pattern: /לימוד|אוניברסיט|מבחן|קורס|study|exam|school/i, hue: 245 }, // indigo
  { pattern: /משפחה|חבר|יום הולדת|מסיב|family|friend|birthday|party/i, hue: 45 }, // gold
  { pattern: /משימ|סידור|todo|task|errand/i, hue: 200 }, // sky
];

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

// Bold, saturated solid-color chips (white text) instead of pale tints —
// reads as a real app, not a spreadsheet.
export function colorForCategory(category: string): { bg: string; border: string; text: string } {
  const normalized = category.trim().toLowerCase();
  const semantic = SEMANTIC_HUES.find((s) => s.pattern.test(normalized));
  const hue = semantic ? semantic.hue : hashString(normalized) % 360;
  return {
    bg: `hsl(${hue}, 72%, 46%)`,
    border: `hsl(${hue}, 72%, 34%)`,
    text: `#ffffff`,
  };
}
