"use client";

import { useEffect, useState } from "react";
import { addDays, addMonths, addYears, startOfWeek, MONTH_LABELS } from "@/lib/dates";
import { THEME } from "@/lib/theme";
import type { ViewKind, EventListItem, PendingAction } from "@/lib/types";
import EventForm, { emptyFormValues, formValuesFromItem, toEventPayload, type EventFormValues } from "./EventForm";
import EventList from "./EventList";

const VIEW_LABELS: Record<ViewKind, string> = { week: "שבוע", month: "חודש", year: "שנה" };

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
  const [pendingConfirmation, setPendingConfirmation] = useState<string | null>(null);
  const [context, setContext] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);
  const [loading, setLoading] = useState(false);
  const [imgVersion, setImgVersion] = useState(0);

  const [events, setEvents] = useState<EventListItem[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [formMode, setFormMode] = useState<"closed" | "add" | "edit">("closed");
  const [editingItem, setEditingItem] = useState<EventListItem | null>(null);
  const [savingForm, setSavingForm] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setEventsLoading(true);
    fetch(`/api/events?view=${view}&date=${anchor}`)
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) setEvents(data.items || []);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setEventsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [view, anchor, imgVersion]);

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
        body: JSON.stringify({ text, context, pendingAction, view, anchor }),
      });
      const data = await res.json();

      if (data.needsConfirmation) {
        setPendingConfirmation(data.message);
        setPendingAction(data.pendingAction ?? null);
        setPendingQuestion(null);
        setContext(null);
        setStatus(null);
      } else if (data.needsClarification) {
        setPendingQuestion(data.message);
        setContext(data.context ?? null);
        setPendingConfirmation(null);
        setPendingAction(null);
        setStatus(null);
      } else if (data.ok) {
        setStatus(data.message);
        setPendingQuestion(null);
        setPendingConfirmation(null);
        setPendingAction(null);
        setContext(null);
        if (data.changed) setImgVersion((v) => v + 1);
      } else {
        setStatus(data.message ?? "משהו השתבש.");
        setPendingQuestion(null);
        setPendingConfirmation(null);
        setPendingAction(null);
        setContext(null);
      }
    } catch {
      setStatus("לא הצלחתי להתחבר לשרת — נסה שוב.");
    } finally {
      setLoading(false);
    }
  }

  async function handleFormSave(values: EventFormValues) {
    setSavingForm(true);
    try {
      const url = editingItem ? `/api/events/${editingItem.id}` : "/api/events";
      const method = editingItem ? "PATCH" : "POST";
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(toEventPayload(values)),
      });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || "לא הצלחתי לשמור.");
        return;
      }
      setFormMode("closed");
      setEditingItem(null);
      setStatus(editingItem ? "האירוע עודכן ✅" : "האירוע נוסף ✅");
      setImgVersion((v) => v + 1);
    } catch {
      setStatus("שגיאת רשת — נסה שוב.");
    } finally {
      setSavingForm(false);
    }
  }

  async function handleDelete(item: EventListItem) {
    const warning = item.isRecurring
      ? "זהו אירוע חוזר — מחיקה תבטל את כל הסדרה, לא רק את המופע הזה. למחוק בכל זאת?"
      : "למחוק את האירוע?";
    if (!confirm(warning)) return;

    setDeletingId(item.id);
    try {
      const res = await fetch(`/api/events/${item.id}`, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || "לא הצלחתי למחוק.");
        return;
      }
      setStatus("האירוע נמחק ✅");
      setImgVersion((v) => v + 1);
    } catch {
      setStatus("שגיאת רשת — נסה שוב.");
    } finally {
      setDeletingId(null);
    }
  }

  const imgSrc = `/api/calendar-image?view=${view}&date=${anchor}&v=${imgVersion}`;
  const downloadName = `לוח-שנה-${VIEW_LABELS[view]}-${anchor}.png`;
  const banner = pendingConfirmation ?? pendingQuestion;
  const placeholder = pendingConfirmation ? "כן / לא, או תקן פרטים…" : pendingQuestion ? "השב כאן…" : "לדוגמה: ארוחת ערב בשישי בשבע, או שיעור ריקוד כל יום שני בשש";

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
        {banner && (
          <div
            style={{
              background: pendingConfirmation ? "#1e2340" : "#3a2a12",
              border: `1px solid ${pendingConfirmation ? THEME.accent : "#7c4a12"}`,
              borderRadius: 10,
              padding: "14px 18px",
              color: pendingConfirmation ? "#c7d2fe" : "#fbbf6b",
              fontSize: 16,
              whiteSpace: "pre-line",
            }}
          >
            {banner}
          </div>
        )}
        {status && !banner && <div style={{ color: "#86efac", fontSize: 15 }}>{status}</div>}
        <div style={{ display: "flex", gap: 12 }}>
          <input
            dir="rtl"
            lang="he"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={placeholder}
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

      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 16, fontWeight: 700, color: THEME.text }}>אירועים בטווח הזה</div>
          {formMode === "closed" && (
            <button
              onClick={() => {
                setEditingItem(null);
                setFormMode("add");
              }}
              style={addBtnStyle}
            >
              + הוספה ידנית
            </button>
          )}
        </div>

        {formMode !== "closed" && (
          <EventForm
            initial={editingItem ? formValuesFromItem(editingItem) : emptyFormValues(anchor)}
            isEditing={formMode === "edit"}
            saving={savingForm}
            onCancel={() => {
              setFormMode("closed");
              setEditingItem(null);
            }}
            onSave={handleFormSave}
          />
        )}

        <EventList
          items={events}
          loading={eventsLoading}
          deletingId={deletingId}
          onEdit={(item) => {
            setEditingItem(item);
            setFormMode("edit");
          }}
          onDelete={handleDelete}
        />
      </div>
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

const addBtnStyle: React.CSSProperties = {
  padding: "9px 18px",
  borderRadius: 10,
  border: `1px solid ${THEME.accent}`,
  background: "transparent",
  color: THEME.accent,
  cursor: "pointer",
  fontWeight: 700,
  fontSize: 14,
};
