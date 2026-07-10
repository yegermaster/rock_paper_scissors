"use client";

import { useState } from "react";
import { addDays, addMonths, addYears } from "@/lib/dates";
import type { ViewKind } from "@/lib/types";

const VIEW_LABELS: Record<ViewKind, string> = { week: "שבוע", month: "חודש", year: "שנה" };

function todayYMD(): string {
  return new Date().toISOString().slice(0, 10);
}

function shiftAnchor(view: ViewKind, anchor: string, direction: 1 | -1): string {
  if (view === "week") return addDays(anchor, 7 * direction);
  if (view === "month") return addMonths(anchor, 1 * direction);
  return addYears(anchor, 1 * direction);
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
        padding: "32px 24px 64px",
        display: "flex",
        flexDirection: "column",
        gap: 24,
      }}
    >
      <h1 style={{ fontSize: 30, fontWeight: 800, margin: 0, color: "#111827" }}>לוח שנה משותף</h1>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
        <div style={{ display: "flex", gap: 10 }}>
          {(["week", "month", "year"] as ViewKind[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                padding: "14px 28px",
                borderRadius: 12,
                border: "1px solid #d1d5db",
                background: view === v ? "#4f46e5" : "#fff",
                color: view === v ? "#fff" : "#111827",
                fontWeight: 700,
                fontSize: 18,
                cursor: "pointer",
              }}
            >
              {VIEW_LABELS[v]}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
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

      <div style={{ background: "#fff", borderRadius: 16, overflow: "hidden", boxShadow: "0 2px 10px rgba(0,0,0,0.12)" }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img key={imgSrc} src={imgSrc} alt={`לוח שנה — ${VIEW_LABELS[view]}`} style={{ width: "100%", display: "block" }} />
      </div>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {pendingQuestion && (
          <div style={{ background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: 10, padding: "14px 18px", color: "#9a3412", fontSize: 17 }}>
            {pendingQuestion}
          </div>
        )}
        {status && !pendingQuestion && (
          <div style={{ color: "#166534", fontSize: 17 }}>{status}</div>
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
              padding: "18px 20px",
              borderRadius: 12,
              border: "1px solid #d1d5db",
              fontSize: 18,
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "18px 32px",
              borderRadius: 12,
              border: "none",
              background: "#4f46e5",
              color: "#fff",
              fontWeight: 700,
              fontSize: 18,
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
  padding: "12px 18px",
  borderRadius: 12,
  border: "1px solid #d1d5db",
  background: "#fff",
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 16,
  color: "#111827",
};

const downloadBtnStyle: React.CSSProperties = {
  padding: "12px 18px",
  borderRadius: 12,
  border: "1px solid #d1d5db",
  background: "#fff",
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 16,
  color: "#111827",
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
};
