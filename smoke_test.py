"""
Pre-deploy smoke test for the advisor - free (no Anthropic calls), ~20 s.

Imports the app in-process with a throwaway test account and exercises the routes a
deploy could break: login gate, card lookups (incl. fuzzy/ambiguous names), deck
parsing, autocomplete, and the UI page. Exit code 0 = safe to deploy.

    python smoke_test.py [--ui advisor_ui.dev.html]
"""
import argparse
import ast
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY_FILES = ["advisor_app.py", "mtg_tools.py", "role_index.py", "mtg_mcp.py", "accounts.py"]
UI_MARKERS = ["function send(", "function renderDeck(", "function initChats(", "initChats();",
              "function deckForChat(", "const UI_VERSION = '__UI_VERSION__'", "function openIntake(", "</html>"]

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default="advisor_ui.html", help="UI file to validate/serve")
    args = ap.parse_args()
    t0 = time.time()

    print("Syntax")
    for f in PY_FILES:
        try:
            ast.parse((REPO / f).read_text(encoding="utf-8"))
            check(f, True)
        except SyntaxError as e:
            check(f, False, f"line {e.lineno}: {e.msg}")
    if failures:
        return

    # Throwaway account, site code, secret and accounts file; load_dotenv never overrides these.
    import tempfile
    os.environ["ADVISOR_USERS"] = "smoketest:smoke-pass"
    os.environ["ADVISOR_PASSWORD"] = ""
    os.environ["ADVISOR_SITE_CODE"] = "smoke-code"
    os.environ["ADVISOR_SESSION_SECRET"] = "smoke-test-secret"
    os.environ["ADVISOR_ACCOUNTS_FILE"] = str(Path(tempfile.mkdtemp()) / "accounts.json")
    import advisor_app as a
    from fastapi.testclient import TestClient

    ui = REPO / args.ui
    a.UI_FILE = ui
    html = ui.read_text(encoding="utf-8")
    print(f"UI ({args.ui})")
    for m in UI_MARKERS:
        check(f"contains {m!r}", m in html)

    c = TestClient(a.app, base_url="https://smoke")
    print("Routes")
    check("healthz", c.get("/healthz").status_code == 200)
    check("anon API -> 401", c.get("/card?name=Sol Ring").status_code == 401)
    r = c.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    check("anon page -> login redirect", r.status_code == 303 and r.headers.get("location") == "/login")
    check("login page", c.get("/login").status_code == 200)
    r = c.post("/login", data={"username": "smoketest", "password": "smoke-pass"}, follow_redirects=False)
    check("login", r.status_code == 303 and r.headers.get("location") == "/")
    r = c.get("/")
    check("UI served", r.status_code == 200 and "function send(" in r.text)
    ver = c.get("/version").json().get("ui", "")
    check("UI version stamped", "__UI_VERSION__" not in r.text and f"'{ver}'" in r.text)
    check("/me", c.get("/me").json().get("user") == "smoketest")

    print("Sign-up")
    s = TestClient(a.app, base_url="https://smoke")
    check("signup page", s.get("/signup").status_code == 200)
    form = {"username": "Friend_1", "password": "hunter22", "confirm": "hunter22", "code": "nope"}
    r = s.post("/signup", data=form, follow_redirects=False)
    check("wrong site code rejected", r.headers.get("location") == "/signup?e=code")
    r = s.post("/signup", data={**form, "code": "smoke-code"}, follow_redirects=False)
    check("signup with site code", r.headers.get("location") == "/" and s.get("/me").json().get("user") == "Friend_1")
    s2 = TestClient(a.app, base_url="https://smoke")
    r = s2.post("/signup", data={**form, "username": "friend_1", "code": "smoke-code"}, follow_redirects=False)
    check("duplicate username (case-insensitive) rejected", r.headers.get("location") == "/signup?e=taken")
    r = s2.post("/login", data={"username": "FRIEND_1", "password": "hunter22"}, follow_redirects=False)
    check("login with new account", r.headers.get("location") == "/" and s2.get("/me").json().get("user") == "Friend_1")
    a.accounts.remove("Friend_1")
    check("removed account is logged out", s2.get("/me").status_code == 401)

    print("Cards (live Scryfall)")
    r = c.get("/card?name=Sol Ring")
    check("/card exact", r.status_code == 200 and r.json().get("name") == "Sol Ring")
    r = c.get("/deck/card?name=morcant")
    check("/deck/card ambiguous name", r.status_code == 200 and r.json().get("name") == "High Perfect Morcant",
          r.text[:80])
    r = c.get("/card/search?q=krenko")
    check("/card/search", r.status_code == 200 and "Krenko, Mob Boss" in r.json().get("names", []))
    r = c.post("/deck/parse", json={"text": "1 Sol Ring\n1 Glasspool Mimic // Glasspool Shore\n2 Island"})
    d = r.json()
    check("/deck/parse (incl. MDFC)", r.status_code == 200 and len(d.get("cards", [])) == 3 and not d.get("not_found"),
          str(d.get("not_found")))

    arch = ("1x Krenko, Mob Boss (m13) 138 [Commander{top}] ^Sleeved,#fb00e5^\n"
            "1x Sol Ring (c21) 263 *F* [Ramp] ^Sleeved^\n"
            "1x Skullclamp (c20) 252 [Draw]\n"
            "1x Umbral Mantle (shm) 267 [Maybeboard]")
    d = c.post("/deck/parse", json={"text": arch}).json()
    check("/deck/parse Archidekt export (commander, maybeboard)",
          d.get("commander") == ["Krenko, Mob Boss"] and d.get("skipped") == 1 and len(d.get("cards", [])) == 3,
          str({k: d.get(k) for k in ("commander", "skipped", "not_found")}))
    d = c.post("/deck/parse", json={"text": "1x Agent of the Iron Throne [Commander]\n"
                                            "1x Wilson, Refined Grizzly [Commander]\n1x Sol Ring"}).json()
    elig = {x["name"]: x.get("can_command") for x in d.get("cards", [])}
    check("commander eligibility via Scryfall (Background ok, Sol Ring not)",
          elig.get("Agent of the Iron Throne") is True and elig.get("Wilson, Refined Grizzly") is True
          and elig.get("Sol Ring") is False, str(elig))
    many = "\n".join(f"1x Card {i} (set) {i} [Other]" for i in range(16))
    check("'1x' pastes route to the review model", a.pick_model(many)[0] == a.REVIEW_MODEL)

    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # an import/runtime crash is a failed smoke test
        check("unexpected error", False, repr(e))
    if failures:
        print(f"SMOKE TEST FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("SMOKE TEST PASSED")
