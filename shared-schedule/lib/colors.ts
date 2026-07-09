// Deterministic color per category (open-ended free text categories, so we
// can't use a fixed lookup table — hash the string into a stable hue instead).
// Fixed saturation/lightness keeps every category readable and visually
// consistent even though the hue varies.

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

export function colorForCategory(category: string): { bg: string; border: string; text: string } {
  const hue = hashString(category.trim().toLowerCase()) % 360;
  return {
    bg: `hsl(${hue}, 65%, 88%)`,
    border: `hsl(${hue}, 55%, 55%)`,
    text: `hsl(${hue}, 55%, 22%)`,
  };
}
