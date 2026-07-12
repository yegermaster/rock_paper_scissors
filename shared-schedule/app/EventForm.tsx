"use client";

import { useState } from "react";
import { THEME } from "@/lib/theme";
import { dayOfWeek } from "@/lib/dates";
import type { EventListItem } from "@/lib/types";

export type EventFormValues = {
  title: string;
  category: string;
  person: string;
  start_date: string;
  start_time: string; // "" = none
  duration_minutes: number;
  recurring: boolean;
  frequency: "daily" | "weekly" | "monthly";
  interval: number;
  end_date: string; // "" = none
};

export function emptyFormValues(defaultDate: string): EventFormValues {
  return {
    title: "",
    category: "",
    person: "",
    start_date: defaultDate,
    start_time: "",
    duration_minutes: 60,
    recurring: false,
    frequency: "weekly",
    interval: 1,
    end_date: "",
  };
}

export function formValuesFromItem(item: EventListItem): EventFormValues {
  return {
    title: item.title,
    category: item.category,
    person: item.person || "",
    start_date: item.occurrenceDate,
    start_time: item.startTime || "",
    duration_minutes: item.durationMinutes,
    recurring: item.isRecurring,
    frequency: item.recurrenceFrequency || "weekly",
    interval: item.recurrenceInterval || 1,
    end_date: item.recurrenceEndDate || "",
  };
}

/** Payload shape the /api/events endpoints expect. */
export function toEventPayload(values: EventFormValues) {
  return {
    title: values.title,
    category: values.category || null,
    person: values.person || null,
    start_date: values.start_date,
    end_date: values.start_date,
    start_time: values.start_time || null,
    duration_minutes: values.duration_minutes,
    recurrence: values.recurring
      ? {
          frequency: values.frequency,
          interval: values.interval,
          days_of_week: values.frequency === "weekly" ? [dayOfWeek(values.start_date)] : null,
          end_date: values.end_date || null,
        }
      : null,
  };
}

export default function EventForm({
  initial,
  isEditing,
  onCancel,
  onSave,
  saving,
}: {
  initial: EventFormValues;
  isEditing: boolean;
  onCancel: () => void;
  onSave: (values: EventFormValues) => void;
  saving: boolean;
}) {
  const [values, setValues] = useState<EventFormValues>(initial);
  const set = <K extends keyof EventFormValues>(key: K, val: EventFormValues[K]) =>
    setValues((v) => ({ ...v, [key]: val }));

  const canSave = values.title.trim().length > 0 && values.start_date.length > 0 && !saving;

  return (
    <div
      style={{
        background: THEME.panel,
        border: `1px solid ${THEME.border}`,
        borderRadius: 12,
        padding: 18,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div style={{ fontSize: 15, fontWeight: 700, color: THEME.text }}>
        {isEditing ? "עריכת אירוע" : "הוספת אירוע"}
      </div>
      {isEditing && initial.recurring && (
        <div style={{ fontSize: 13, color: THEME.textFaint }}>
          ⚠️ זהו אירוע חוזר — שינויים כאן חלים על כל הסדרה.
        </div>
      )}

      <input dir="rtl" placeholder="כותרת" value={values.title} onChange={(e) => set("title", e.target.value)} style={inputStyle} />

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <input type="date" value={values.start_date} onChange={(e) => set("start_date", e.target.value)} style={{ ...inputStyle, flex: 1, minWidth: 140 }} />
        <input type="time" value={values.start_time} onChange={(e) => set("start_time", e.target.value)} style={{ ...inputStyle, flex: 1, minWidth: 110 }} />
        <input dir="rtl" placeholder="קטגוריה (לא חובה)" value={values.category} onChange={(e) => set("category", e.target.value)} style={{ ...inputStyle, flex: 1, minWidth: 140 }} />
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 8, color: THEME.textMuted, fontSize: 14, cursor: "pointer" }}>
        <input type="checkbox" checked={values.recurring} onChange={(e) => set("recurring", e.target.checked)} />
        אירוע חוזר
      </label>

      {values.recurring && (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <select value={values.frequency} onChange={(e) => set("frequency", e.target.value as EventFormValues["frequency"])} style={inputStyle}>
            <option value="daily">כל יום</option>
            <option value="weekly">כל שבוע</option>
            <option value="monthly">כל חודש</option>
          </select>
          <input type="date" value={values.end_date} onChange={(e) => set("end_date", e.target.value)} style={inputStyle} title="תאריך סיום (לא חובה)" />
        </div>
      )}

      <div style={{ display: "flex", gap: 10 }}>
        <button disabled={!canSave} onClick={() => onSave(values)} style={{ ...primaryBtnStyle, opacity: canSave ? 1 : 0.6 }}>
          {saving ? "שומר…" : "שמירה"}
        </button>
        <button onClick={onCancel} style={secondaryBtnStyle}>
          ביטול
        </button>
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: "10px 12px",
  borderRadius: 8,
  border: `1px solid ${THEME.border}`,
  background: THEME.bg,
  color: THEME.text,
  fontSize: 15,
  outline: "none",
};

const primaryBtnStyle: React.CSSProperties = {
  padding: "10px 22px",
  borderRadius: 8,
  border: "none",
  background: THEME.accent,
  color: "#ffffff",
  fontWeight: 700,
  cursor: "pointer",
  fontSize: 15,
};

const secondaryBtnStyle: React.CSSProperties = {
  padding: "10px 22px",
  borderRadius: 8,
  border: `1px solid ${THEME.border}`,
  background: "transparent",
  color: THEME.textMuted,
  cursor: "pointer",
  fontSize: 15,
};
