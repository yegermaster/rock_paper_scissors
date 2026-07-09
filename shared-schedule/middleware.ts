import { NextRequest, NextResponse } from "next/server";

const COOKIE_NAME = "ss_key";

export function middleware(req: NextRequest) {
  const secret = process.env.SHARED_SECRET;
  if (!secret) {
    // Misconfigured deployment — fail closed rather than open.
    return new NextResponse("Server misconfigured: SHARED_SECRET is not set.", { status: 500 });
  }

  const urlKey = req.nextUrl.searchParams.get("key");
  const cookieKey = req.cookies.get(COOKIE_NAME)?.value;

  if (urlKey === secret) {
    const res = NextResponse.next();
    // Strip the key from the visible URL, remember it via cookie instead.
    if (req.nextUrl.searchParams.has("key")) {
      const cleanUrl = req.nextUrl.clone();
      cleanUrl.searchParams.delete("key");
      const redirect = NextResponse.redirect(cleanUrl);
      redirect.cookies.set(COOKIE_NAME, secret, {
        httpOnly: true,
        sameSite: "lax",
        maxAge: 60 * 60 * 24 * 365,
      });
      return redirect;
    }
    return res;
  }

  if (cookieKey === secret) {
    return NextResponse.next();
  }

  return new NextResponse("Not authorized. Use the shared link with ?key=...", { status: 401 });
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
