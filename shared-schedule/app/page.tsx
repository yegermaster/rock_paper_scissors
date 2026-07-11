"use client";

import { useState } from "react";
import { addDays, addMonths, addYears, startOfWeek, MONTH_LABELS } from "@/lib/dates";
import type { ViewKind } from "@/lib/types";

const VIEW_LABELS: Record<ViewKind, string> = { week: "שבוע", month: "חודש", year: "שנה" };

const THEME = {
  bg: "#111827",
  panel: "#161c2c",
  panelAlt: "#1a2133",
  border: "rgba(255,255,255,0.09)",
  text: "#f1f5f9",
  textMuted: "#94a3b8",
  textFaint: "#64748b",
  accent: "#818cf8",
};

function todayYMD(): string {
  return new Date().toISOString().slice(0, 10);
}

function shiftAnchor(view: ViewKind, anchor: string, direction: 1 | -1): string {
  if (view === "week") return addDays(anchor, 7 * direction);
  if (view === "month") return addMonths(anchor, 1 * direction);
  return addYears(anchor, 1 * direction);
}

function subtitleFor(view: ViewKind, anchor: string): string {
  if (view === "week") {
    const start = startOfWeek(anchor);
    const end = addDays(start, 6);
    const [, sm, sd] = start.split("-").map(Number);
    const [, em, ed] = end.split("-").map(Number);
    return sm === em
      ? `שבוע של ${sd}–${ed} ${MONTH_LABELS[sm - 1]}`
      : `שבוע של ${sd} ${MONTH_LABELS[sm - 1]} – ${ed} ${MONTH_LABELS[em - 1]}`;
  }
  if (view === "month") {
    const [y, m] = anchor.split("-").map(Number);
    return `${MONTH_LABELS[m - 1]} ${y}`;
  }
  return anchor.slice(0, 4);
}

export default function Page() {
  const [view, setView] = useState<ViewKind>("week");
  const [anchor, setAnchor] = useState(todayYMD());
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [context, setContext] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imgVersion, setImgVersion] = useState(0);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setLoading(true);
    setInput("");
    try {
      const res = await fetch("/api/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, context }),
      });
      const data = await res.json();

      if (data.needsClarification) {
        setPendingQuestion(data.message);
        setContext(data.context ?? null);
        setStatus(null);
      } else if (data.ok) {
        setStatus(data.message);
        setPendingQuestion(null);
        setContext(null);
        setImgVersion((v) => v + 1);
      } else {
        setStatus(data.message ?? "משהו השתבש.");
        setPendingQuestion(null);
        setContext(null);
      }
    } catch {
      setStatus("לא הצלחתי להתחבר לשרת — נסה שוב.");
    } finally {
      setLoading(false);
    }
  }

  const imgSrc = `/api/calendar-image?view=${view}&date=${anchor}&v=${imgVersion}`;
  const downloadName = `לוח-שנה-${VIEW_LABELS[view]}-${anchor}.png`;

  return (
    <main
      dir="rtl"
      lang="he"
      style={{
        maxWidth: 1600,
        margin: "0 auto",
        padding: "28px 24px 64px",
        display: "flex",
        flexDirection: "column",
        gap: 18,
        minHeight: "100vh",
        backgroundColor: THEME.bg,
        color: THEME.text,
      }}
    >
      <div style={{ fontSize: 16, fontWeight: 500, color: THEME.textMuted }}>לוח שנה משותף</div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 14 }}>
        <div style={{ display: "flex", gap: 8 }}>
          {(["week", "month", "year"] as ViewKind[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                padding: "11px 24px",
                borderRadius: 10,
                border: `1px solid ${view === v ? THEME.accent : THEME.border}`,
                background: view === v ? THEME.accent : THEME.panel,
                color: "#ffffff",
                fontWeight: 700,
                fontSize: 16,
                cursor: "pointer",
              }}
            >
              {VIEW_LABELS[v]}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button onClick={() => setAnchor((a) => shiftAnchor(view, a, -1))} style={navBtnStyle}>
            ← הקודם
          </button>
          <button onClick={() => setAnchor(todayYMD())} style={navBtnStyle}>
            היום
          </button>
          <button onClick={() => setAnchor((a) => shiftAnchor(view, a, 1))} style={navBtnStyle}>
            הבא →
          </button>
          <a href={imgSrc} download={downloadName} style={downloadBtnStyle}>
            ⬇ הורדה כתמונה
          </a>
        </div>
      </div>

      <div style={{ fontSize: 14, color: THEME.textFaint }}>{subtitleFor(view, anchor)}</div>

      <div style={{ background: THEME.panel, borderRadius: 14, overflow: "hidden", border: `1px solid ${THEME.border}` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img key={imgSrc} src={imgSrc} alt={`לוח שנה — ${VIEW_LABELS[view]}`} style={{ width: "100%", display: "block" }} />
      </div>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {pendingQuestion && (
          <div style={{ background: "#3a2a12", border: "1px solid #7c4a12", borderRadius: 10, padding: "14px 18px", color: "#fbbf6b", fontSize: 16 }}>
            {pendingQuestion}
          </div>
        )}
        {status && !pendingQuestion && (
          <div style={{ color: "#86efac", fontSize: 15 }}>{status}</div>
        )}
        <div style={{ display: "flex", gap: 12 }}>
          <input
            dir="rtl"
            lang="he"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={pendingQuestion ? "השב כאן…" : "לדוגמה: ארוחת ערב בשישי בשבע, או שיעור ריקוד כל יום שני בשש"}
            style={{
              flex: 1,
              padding: "16px 18px",
              borderRadius: 12,
              border: `1px solid ${THEME.border}`,
              background: THEME.panel,
              color: THEME.text,
              fontSize: 17,
              outline: "none",
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "16px 30px",
              borderRadius: 12,
              border: "none",
              background: THEME.accent,
              color: "#ffffff",
              fontWeight: 700,
              fontSize: 17,
              cursor: loading ? "default" : "pointer",
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "…" : "שלח"}
          </button>
        </div>
      </form>
    </main>
  );
}

const navBtnStyle: React.CSSProperties = {
  padding: "11px 16px",
  borderRadius: 10,
  border: `1px solid ${THEME.border}`,
  background: THEME.panel,
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 15,
  color: THEME.text,
};

const downloadBtnStyle: React.CSSProperties = {
  padding: "11px 16px",
  borderRadius: 10,
  border: `1px solid ${THEME.border}`,
  background: THEME.panel,
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 15,
  color: THEME.text,
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
};
