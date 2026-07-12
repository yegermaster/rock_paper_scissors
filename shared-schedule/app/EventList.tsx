"use client";

import { THEME } from "@/lib/theme";
import type { EventListItem } from "@/lib/types";

export default function EventList({
  items,
  loading,
  onEdit,
  onDelete,
  deletingId,
}: {
  items: EventListItem[];
  loading: boolean;
  onEdit: (item: EventListItem) => void;
  onDelete: (item: EventListItem) => void;
  deletingId: string | null;
}) {
  if (loading && items.length === 0) {
    return <div style={{ color: THEME.textFaint, fontSize: 14, padding: "8px 4px" }}>טוען…</div>;
  }
  if (items.length === 0) {
    return <div style={{ color: THEME.textFaint, fontSize: 14, padding: "8px 4px" }}>אין אירועים בטווח הזה.</div>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {items.map((item) => (
        <div
          key={item.id + item.occurrenceDate}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            background: THEME.panel,
            border: `1px solid ${THEME.border}`,
            borderRadius: 10,
            padding: "10px 14px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
            <div style={{ color: THEME.textFaint, fontSize: 13, whiteSpace: "nowrap" }}>
              {item.occurrenceDate.slice(8, 10)}/{item.occurrenceDate.slice(5, 7)}
              {item.startTime ? ` · ${item.startTime}` : ""}
            </div>
            <div
              style={{
                color: THEME.text,
                fontSize: 15,
                fontWeight: 600,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                maxWidth: 260,
              }}
            >
              {item.title}
            </div>
            {item.isRecurring && (
              <span style={{ color: THEME.accent, fontSize: 13 }} title="אירוע חוזר — עריכה/מחיקה חלה על כל הסדרה">
                🔁
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
            <button onClick={() => onEdit(item)} style={iconBtnStyle} title="עריכה" aria-label="עריכה">
              ✎
            </button>
            <button
              onClick={() => onDelete(item)}
              disabled={deletingId === item.id}
              style={{ ...iconBtnStyle, opacity: deletingId === item.id ? 0.5 : 1 }}
              title="מחיקה"
              aria-label="מחיקה"
            >
              ✕
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

const iconBtnStyle: React.CSSProperties = {
  width: 32,
  height: 32,
  borderRadius: 8,
  border: `1px solid ${THEME.border}`,
  background: "transparent",
  color: THEME.textMuted,
  cursor: "pointer",
  fontSize: 15,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};
