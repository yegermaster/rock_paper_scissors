import { createClient, SupabaseClient } from "@supabase/supabase-js";

// Server-only client. Uses the service-role key, which bypasses RLS —
// never import this from client components. Access to the app itself is
// gated by the SHARED_SECRET link (see middleware.ts), not Supabase auth.
//
// Cached at module scope: a warm serverless instance reuses the same
// client (and its connection pool) across requests instead of paying
// client-construction cost every time.
let cached: SupabaseClient | null = null;

export function getSupabaseServerClient() {
  if (cached) return cached;

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !key) {
    throw new Error(
      "Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars."
    );
  }

  cached = createClient(url, key, {
    auth: { persistSession: false },
  });
  return cached;
}
