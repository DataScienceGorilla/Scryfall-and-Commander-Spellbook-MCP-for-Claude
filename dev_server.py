"""
Dev instance of the advisor, alongside the live one: port 8001, login OFF, and it
serves advisor_ui.dev.html (gitignored) if present, so UI work never touches the
live site. Promote with: copy advisor_ui.dev.html -> advisor_ui.html.

    python dev_server.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))
# Set-but-empty so load_dotenv (which never overrides) can't turn the login back on.
os.environ["ADVISOR_PASSWORD"] = ""
os.environ["ADVISOR_USERS"] = ""
os.environ["ADVISOR_SITE_CODE"] = ""  # the sign-up site code turns the login on too
os.environ["ADVISOR_SESSION_SECRET"] = "dev-only"

import uvicorn  # noqa: E402
import advisor_app  # noqa: E402

dev_ui = REPO / "advisor_ui.dev.html"
if dev_ui.exists():
    advisor_app.UI_FILE = dev_ui

if __name__ == "__main__":
    uvicorn.run(advisor_app.app, host="127.0.0.1", port=8001)
