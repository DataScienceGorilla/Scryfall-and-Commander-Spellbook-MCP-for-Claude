"""
Self-service advisor accounts, stored locally in accounts.json (gitignored).

Friends sign up with a username + password + the SITE CODE (ADVISOR_SITE_CODE in .env),
so only people you've given the code to can get in, and the activity log shows who's who.
Passwords are hashed with scrypt; the file is rewritten atomically.

CLI (run from the repo):
    python accounts.py list
    python accounts.py remove <username>
    python accounts.py reset <username> <new-password>
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
from datetime import datetime
from pathlib import Path

ACCOUNTS_FILE = Path(os.getenv("ADVISOR_ACCOUNTS_FILE") or Path(__file__).parent / "accounts.json")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,24}$")
MIN_PASSWORD = 6
_lock = threading.Lock()


def _load() -> dict:
    if not ACCOUNTS_FILE.exists():
        return {}
    return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    tmp = ACCOUNTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(ACCOUNTS_FILE)


def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    return hmac.compare_digest(_hash(password, bytes.fromhex(salt_hex)), stored)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def exists(username: str) -> bool:
    return (username or "").lower() in _load()


def display_name(username: str) -> str | None:
    """The account's username as the player typed it at sign-up (None if no such account)."""
    acct = _load().get((username or "").lower())
    return acct["username"] if acct else None


class AccountError(Exception):
    """A sign-up problem to show the player; .code is used in the page's ?e= parameter."""
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def create(username: str, password: str, reserved: set[str] = frozenset()) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise AccountError("badname")
    if len(password or "") < MIN_PASSWORD:
        raise AccountError("weak")
    with _lock:
        data = _load()
        key = username.lower()
        if key in data or key in {r.lower() for r in reserved}:
            raise AccountError("taken")
        data[key] = {"username": username, "pw": _hash(password), "created": now(), "last_login": None}
        _save(data)
    return username


def check(username: str, password: str) -> str | None:
    """Canonical username if the password is right, else None (constant-ish time)."""
    acct = _load().get((username or "").strip().lower())
    ok = _verify(password or "", acct["pw"] if acct else _hash("dummy-password"))
    if not (acct and ok):
        return None
    with _lock:
        data = _load()
        data[acct["username"].lower()]["last_login"] = now()
        _save(data)
    return acct["username"]


def remove(username: str) -> bool:
    with _lock:
        data = _load()
        if data.pop((username or "").lower(), None) is None:
            return False
        _save(data)
    return True


def reset_password(username: str, password: str) -> bool:
    if len(password or "") < MIN_PASSWORD:
        raise AccountError("weak")
    with _lock:
        data = _load()
        acct = data.get((username or "").lower())
        if not acct:
            return False
        acct["pw"] = _hash(password)
        _save(data)
    return True


def _cli(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "list"
    if cmd == "list":
        data = _load()
        if not data:
            print("No accounts yet.")
        for a in sorted(data.values(), key=lambda a: a["created"]):
            print(f"{a['username']:<24} created {a['created']}   last login {a['last_login'] or '-'}")
        return 0
    if cmd == "remove" and len(argv) == 3:
        print("Removed." if remove(argv[2]) else "No such account.")
        return 0
    if cmd == "reset" and len(argv) == 4:
        print("Password reset." if reset_password(argv[2], argv[3]) else "No such account.")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
