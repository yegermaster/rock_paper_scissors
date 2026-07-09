import { createClient } from "@supabase/supabase-js";

// Server-only client. Uses the service-role key, which bypasses RLS —
// never import this from client components. Access to the app itself is
// gated by the SHARED_SECRET link (see middleware.ts), not Supabase auth.
export function getSupabaseServerClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !key) {
    throw new Error(
      "Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars."
    );
  }

  return createClient(url, key, {
    auth: { persistSession: false },
  });
}
