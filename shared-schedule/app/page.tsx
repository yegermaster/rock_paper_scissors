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

  return (
    <main
      dir="rtl"
      style={{
        maxWidth: 1440,
        margin: "0 auto",
        padding: "24px 16px 48px",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div style={{ display: "flex", gap: 6 }}>
          {(["week", "month", "year"] as ViewKind[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                padding: "8px 16px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: view === v ? "#4f46e5" : "#fff",
                color: view === v ? "#fff" : "#111827",
                fontWeight: 600,
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
        </div>
      </div>

      <div style={{ background: "#fff", borderRadius: 12, overflow: "hidden", boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img key={imgSrc} src={imgSrc} alt={`לוח שנה — ${VIEW_LABELS[view]}`} style={{ width: "100%", display: "block" }} />
      </div>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {pendingQuestion && (
          <div style={{ background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: 8, padding: "10px 14px", color: "#9a3412" }}>
            {pendingQuestion}
          </div>
        )}
        {status && !pendingQuestion && (
          <div style={{ color: "#166534", fontSize: 14 }}>{status}</div>
        )}
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={pendingQuestion ? "השב כאן…" : "לדוגמה: ארוחת ערב בשישי בשבע, או שיעור ריקוד כל יום שני בשש"}
            style={{
              flex: 1,
              padding: "12px 14px",
              borderRadius: 8,
              border: "1px solid #d1d5db",
              fontSize: 15,
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "12px 20px",
              borderRadius: 8,
              border: "none",
              background: "#4f46e5",
              color: "#fff",
              fontWeight: 600,
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
  padding: "8px 14px",
  borderRadius: 8,
  border: "1px solid #d1d5db",
  background: "#fff",
  cursor: "pointer",
  fontWeight: 500,
};
