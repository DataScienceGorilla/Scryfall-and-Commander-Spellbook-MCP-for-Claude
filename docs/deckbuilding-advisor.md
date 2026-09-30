# Deckbuilding Advisor — Architecture & Handoff

A local, self-hosted web app that gives **tailored MTG Commander deckbuilding advice** with a
Scryfall card canvas. It's built on the same tool layer as the MCP server / Discord bot in this
repo. This doc is the single source of truth for how it works and how to keep building it.

> **Status (2026-09-30):** the advisor is fully working, behind a login page, and kept alive on the
> user's PC by a supervisor + logon task. The current build-out is the **Deck Workbench** (turning it
> into an agentic deckbuilder) — see [Deck Workbench](#8-deck-workbench-roadmap). Phases 1–2
> (deck object + full manual editor) are done; phases 3–4 are next. Pending: switching the tunnel to
> the user's domain **brewbot.link** (needs their Cloudflare tunnel token in `.env`).

---

## 1. Run it

```bash
# from the repo root, main Python env (NOT a venv — deps are in requirements.txt)
python -m uvicorn advisor_app:app --host 127.0.0.1 --port 8000
# open http://localhost:8000
```
- **`.claude/launch.json`** has an `advisor` config (`autoPort: false`, port 8000 — the UI fetches
  relative paths, so the port must be stable for the tunnel).
- **Day-to-day: the supervisor.** `run_advisor.ps1` keeps uvicorn + the Cloudflare tunnel alive:
  every 30 s it probes `GET /healthz` (restarting a dead/hung server, killing whatever holds the
  port) and restarts `cloudflared` if it exits. Logs to `supervisor.log`; the current public URL is
  written to `advisor_url.txt`. Single-instance (named mutex).
  - Run once, detached: `Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","run_advisor.ps1"`
  - **Autostart at logon + survive reboots:** `install_autostart.ps1` registers the per-user
    scheduled task **"MTG Advisor"** (restarts on failure; `-Uninstall` removes it). The PC must be
    awake — set sleep to "Never" on AC power for true always-on.
- **Restarting after backend edits:** the server does NOT auto-reload. Kill the uvicorn process
  (`Get-NetTCPConnection -LocalPort 8000` → owning pid); the supervisor restarts it within ~30 s.
  Editing `advisor_ui.html` / `advisor_login.html` needs no restart — they're read per request.

### Auth (login page + session cookie)
**Enforced only when accounts are configured** (local use stays open otherwise). Accounts:
`ADVISOR_USER` / `ADVISOR_PASSWORD` and/or `ADVISOR_USERS="alice:pw1,bob:pw2"` in `.env`.
- `GET /login` serves `advisor_login.html`; `POST /login` checks the password (constant-time) and
  sets a signed `advisor_session` cookie (Starlette `SessionMiddleware`, 30 days, HttpOnly, Secure,
  SameSite=Lax). `GET /logout` clears it; `GET /me` returns the user (drives the Sign-out button).
- Unauthed page loads redirect to `/login`; unauthed API calls get `401` JSON, and the UI's `fetch`
  wrapper sends the browser to `/login` (chats are in localStorage, nothing is lost).
- Brute-force brake: 5 failed logins per client IP (`CF-Connecting-IP` behind the tunnel) → 15 min
  lockout, plus a 1 s delay per failure. Logins are recorded in `advisor_activity.log`.
- Signing key: `ADVISOR_SESSION_SECRET` in `.env`, else auto-generated into `.advisor_secret`
  (gitignored) so restarts don't log people out. Delete that file to force everyone to re-login.
- `GET /healthz` is the only unauthenticated route.

### Public sharing (Cloudflare tunnel)
The supervisor runs `cloudflared.exe` (portable binary, gitignored).
- **Default: quick tunnel** — random `https://<words>.trycloudflare.com`, **new URL every time
  cloudflared restarts** (check `advisor_url.txt`).
- **Stable URL: named tunnel** — put `ADVISOR_TUNNEL_TOKEN=<token>` (and `ADVISOR_PUBLIC_URL=https://…`)
  in `.env`; the supervisor then runs `cloudflared tunnel run --token`. Requires a Cloudflare account
  and a domain on Cloudflare; create the tunnel in the Zero Trust dashboard pointing at
  `http://localhost:8000`. After changing the token, restart the task
  (`Stop-ScheduledTask "MTG Advisor"; Start-ScheduledTask "MTG Advisor"`) — on start the supervisor
  kills any cloudflared of the wrong kind (quick vs named) and launches the right one.

### Moving to a server (prepared, not deployed)
Hosting decision (2026-09-30): **run on the user's PC for now**; the app is packaged so a move to a
~$5–7/mo VPS (Hetzner recommended) is quick.
- `Dockerfile` (python:3.12-slim, non-root, ONNX embedder baked in, healthcheck) installs only
  `requirements-advisor.txt` (~350 MB, **no PyTorch**). Verified: the app boots and serves every
  route from a clean venv with just those deps. The image itself hasn't been built (no Docker on the PC).
- `docker-compose.yml` runs the app + `cloudflared` (sharing the app's network namespace so the
  tunnel's `localhost:8000` target works unchanged). RAG DBs, `role_index.json` and the activity log
  are bind-mounted, not baked in. Set `ADVISOR_SESSION_SECRET` in the server `.env`.
- Cutover = copy data + `.env`, `docker compose up -d`, then stop the PC's task (`install_autostart.ps1
  -Uninstall`) — **never run the same tunnel token in two places** (Cloudflare load-balances between them).
- **Embeddings:** all code uses chromadb's `DefaultEmbeddingFunction` (ONNX all-MiniLM-L6-v2).
  Verified identical to the old sentence-transformers vectors (cosine 1.0, identical top-8 hits on
  both collections), so existing DBs didn't need re-ingesting. `chromadb` is pinned to 0.5.3 in the
  slim requirements to match the on-disk DB format.

### `.env` keys
`ANTHROPIC_API_KEY` (required), `ADVISOR_USER` / `ADVISOR_PASSWORD` / `ADVISOR_USERS` (login),
optional `ADVISOR_SESSION_SECRET`, `ADVISOR_TUNNEL_TOKEN`, `ADVISOR_PUBLIC_URL`, plus
`DISCORD_BOT_TOKEN`, `APIFY_TOKEN` used by other components. `.env` is gitignored.

---

## 2. Architecture

```
Browser (advisor_ui.html, single-file vanilla-JS SPA)
  │  localStorage = source of truth for chats + deck objects (per chat)
  │  POST /chat {session_id, message, history, }   ← sends FULL transcript each turn
  ▼
FastAPI (advisor_app.py)
  │  agent_stream(): Claude tool-use loop, SSE to the browser
  ├─ tools from mtg_tools.py (Scryfall, Spellbook, rules RAG, theory RAG)
  ├─ role_index.py  (otag concrete-role lookup)
  ├─ prompt caching + model tiering
  └─ Anthropic API (AsyncAnthropic)
```

**Key design decision: the browser is the source of truth.** Chat history and the deck object live
in `localStorage` (per chat) and are sent to the backend each request. The backend `SESSIONS` dict
is only a fallback. This makes memory survive server restarts (the "forgetting" fix) — the model
always sees exactly what the user sees.

### Endpoints (`advisor_app.py`)
| Route | Purpose |
|---|---|
| `GET /` | Serves `advisor_ui.html` (read per request → HTML edits need no restart). Auth-gated. |
| `POST /chat` | SSE stream. Body `{session_id, message, history}`. Runs `agent_stream`. |
| `POST /reset` | Clears a session id from `SESSIONS`. |
| `GET /card?name=` | **Cached** Scryfall proxy — resolves a card to slim JSON (image, color identity, `game_changer`, concrete `roles`). The canvas uses this so 100-card decks don't trip Scryfall's rate limit. |
| `POST /deck/parse` | Parses a pasted decklist into **structured deck cards** (see [Deck object](#deck-object-schema)). |
| `GET /deck/card?name=` | One card in the deck-card shape (fuzzy name, cached) — the editor's add-card. |
| `GET /card/search?q=` | Card-name autocomplete (Scryfall autocomplete, cached, ≥2 chars). |
| `GET /login` `POST /login` `GET /logout` `GET /me` | Login page + session (see Auth). |
| `GET /healthz` | Unauthenticated liveness probe (supervisor / Docker healthcheck). |

### `agent_stream` (the tool-use loop)
- Streams `status` (tool calls), `identity`, `text`, `warning`, `done`, `error` SSE events.
- Buffers the final answer and **validates color identity** before showing it; regenerates up to
  `MAX_IDENTITY_RETRIES` times, then strips off-color lines as a backstop.
- `MAX_ITERATIONS = 12` tool turns; if exhausted it does a **graceful finish** — one final
  tools-off call that synthesizes an answer from what it gathered (never "stopped after too many
  steps" with nothing).
- Captures `commander_identity` from tool args (authoritative for the color-identity guardrail).

---

## 3. Model config, caching, tiering (cost levers)

- **Model tiering** (`pick_model`): scans the **whole conversation** — if a decklist or deck URL has
  appeared anywhere, the session uses **`claude-sonnet-5`** (full reasoning); otherwise
  **`claude-haiku-4-5`** for deck-free trivia. (Earlier bug: routing on only the latest message made
  long deck chats drop to Haiku and "get stupid" — fixed by scanning the whole convo.)
- **Thinking params**: Sonnet 5 uses adaptive thinking + effort, passed via **`extra_body`**
  (`{"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`) because the installed SDK
  (`anthropic` 0.75) doesn't type these params. **Haiku 4.5 does NOT support these** → its
  `extra_body` is `{}`. ⚠️ If you bump the SDK or model, re-check this against the `claude-api` skill.
- **Prompt caching**: `_cached_system()` caches system+tools (1h TTL); `_cache_last()` puts a cache
  breakpoint on the last message so the whole conversation prefix (incl. big deck details) reads from
  cache (~0.1×) instead of being re-billed. This is the dominant cost saver — a follow-up dropped
  from ~$0.20–0.70 to ~$0.003.
- **Activity log** (`advisor_activity.log`, gitignored): one line per query with model, token
  in/cache_read/cache_write/out, cost estimate, and tools used. `tail -f` it to watch usage.

---

## 4. The system prompt (guardrails, in `advisor_app.py` `SYSTEM_PROMPT`)

Built up from real battle-testing. Major sections:
- **USING YOUR TOOLS (judgment, not a pipeline)** — match effort to the ask: full review vs targeted
  question vs follow-up. Don't re-fetch what you already have. Guardrails apply *when relevant*, not
  as mandatory steps.
- **Verify before claiming** — never describe a card from memory; fetch its text. Never invent combos
  (report only what `spellbook_find_combos_in_decklist` returns).
- **COLOR IDENTITY** — recommendations must be in-identity; enforced by tool-level `id<=` scoping +
  the regenerate gate + a `%%IDENTITY:XX%%` marker + frontend flagging.
- **CARD EVALUATION** — judge by a card's PRIMARY ability and the RESOURCE it uses; read scope
  precisely (you-control vs each-player; counterspells = Target Interaction); read the whole card.
- **RECOMMENDATION QUALITY** — real+legal isn't enough: weigh redundancy vs the commander's own
  engine, on-theme ≠ upgrade, stay on the wincon, justify only with cards actually in the list.
- **BRACKET DISCIPLINE** — per-bracket game-changer caps AND turn expectations AND rules (no MLD /
  chained extra turns / 2-card combos, B3 "no 2-card combo before turn 6"); respect intentional
  omissions of famous staples.
- **DEEP CUTS** — surface 1–2 obscure gems (high EDHREC rank) that fit the engine map.
- **FULL DECK REVIEW OUTPUT** — diagnose first: How it wins → **the Engine Map** (enablers →
  payoffs → force multipliers → engine → threats/finishers) → structural read (real numbers) →
  failure modes → recommendations.
- **Playbook** (`playbooks/unified.md`) appended — distilled deckbuilding lenses from creator content.

---

## 5. Data assets & the role taxonomy

- **Rules RAG** — `mtg_comprehensive_rules` Chroma collection (`mtg_rules_data/`, gitignored). Built
  by `rules_ingestion.py`, which auto-discovers the latest Comprehensive Rules and self-updates
  (monthly Windows scheduled task). Tool: `mtg_rules_search`.
- **Theory RAG** — `mtg_deckbuilding_theory` Chroma collection (`mtg_theory_data/`, gitignored;
  embeddings `all-MiniLM-L6-v2`). Sources: YouTube transcripts (`theory_ingestion.py`, via Apify) +
  **Rebel Lily's Commander Template Academy** articles (`ingest_academy.py`). Tool: `deckbuilding_search`.
  `distill_theory.py` map-reduces the corpus into `playbooks/`.
- **Concrete-role index** — `role_index.py` builds `role_index.json` (gitignored) from **Scryfall
  oracle-tags (otags)**: card → concrete roles. Refreshes every 30 days (`--force` to rebuild).
- **Card-role taxonomy (the user's 15 roles).** This is the backbone of the workbench:
  - **Concrete (text-derivable) → otags**: Ramp, Draw, Tutor, Recursion, Target Interaction
    (spot-removal + counterspells), Mass Interaction (board wipes), Protection, Stax
    (`otag:tax` + a curated list — no `otag:stax` exists).
  - **Contextual (whole-deck judgment) → the LLM's Engine Map**: Enabler, Payoff, Force Multiplier,
    Threats, Alternate Wincon/Finisher, Misc Value (residual).
  - **Combo → the Spellbook detector** (structural axis; a card can be a Payoff *and* a combo piece).
  - History: we tried fine-tuning a local classifier (`laya` / ModernBERT) on the user's Archidekt
    tags; it peaked ~73% then overfit, and **otags beat it** on the concrete roles (100% where the
    model was weakest). The fine-tune was abandoned; otags + LLM is the shipped design. Definitions,
    the certified gold set, and this rationale live in the memory note `card-role-taxonomy`.

---

## 6. Frontend (`advisor_ui.html`, single file)

- **Multi-chat sidebar**, persisted in `localStorage` (`STORE_KEY = mtg_advisor_chats_v1`). Each chat:
  `{id, title, sessionId, conversation:[{role,text}], deck, updatedAt}`.
- **Streaming** (`send()`): binds each stream to the **origin chat** so switching tabs mid-stream
  never misfiles the answer; only writes to the DOM when that chat is on screen.
- **Card canvas** has two areas: `deckView` (the editable deck, when there is one) above
  `mentionView` (chat `[[Card Name]]` mentions → `requestCard()` → `/card` proxy; hover preview +
  click lightbox). With no deck, mentions group by concrete role (`ROLE_ORDER`); with a deck, mentions
  not in it collect in one **"Mentioned in chat · not in deck"** group with a **+ Add** button, and
  mentions already in the deck don't duplicate (clicking the chip flashes the deck tile).
- **Tiles** keep the card aspect-ratio (`63/88`) so full cards show (a prior bug cropped them).

---

## 7. Deck object (the workbench foundation)

### Deck object schema
Per chat, in `localStorage`, sent to the backend with each request. Populated by `POST /deck/parse`.
```js
deck = {
  commander: [],          // names, max 2 (partners) — set in the editor; not auto-detected from pastes
  bracket: null,          // 1–5 — set in the editor
  not_found: [...],       // names /deck/parse couldn't resolve
  cards: [{
    name, qty, scryfall_uri, type_line, cmc, mana_cost,
    color_identity: [...],
    image,                // normal-size art
    game_changer: bool,   // WotC game-changers list (Scryfall flag)
    is_land: bool,
    any_qty: bool,        // singleton-exempt (basic land / "any number of cards named")
    can_command: bool,    // legendary creature or "can be your commander"
    roles: [...],         // CONCRETE roles from the otag index
    role: null,           // user/AI role override (any DECK_ROLES name) — set by the editor
    tags: [],             // user tags — set by the editor
  }, ...]
}
```

### How it works today (Phase 1)
- Pasting a decklist in chat calls `looksLikeDeck()` → `loadDeckFromText()` → `POST /deck/parse` →
  sets `deck` → `renderDeck()`. Runs in parallel with the chat review.
- `renderDeck()` shows a **composition header** (card/land counts, color identity, game-changer
  count, role breakdown) and lays cards into role groups (Commander/Ramp/…/Stax/Other/Lands), with
  qty badges and a game-changer highlight. Uses images from `/deck/parse` (no per-card fetch).
- Persisted per chat; re-rendered on chat switch/reload.

### Manual editor (Phase 2)
All client-side against `deck`; every mutation is `pushUndo()` → mutate → `deckChanged()` (persist
via `saveState`, full `renderDeck()`, `regroupMentions()`, sync the chat's identity flagging).
- **Toolbar** (built once per deck, synced each render): add-card search with autocomplete
  (`/card/search`, arrow keys + Enter), commander picker (legal commanders in the deck; partners shown
  as "A + B"), bracket picker (B1–B5), Undo, Copy list (Moxfield-style text, commander first).
- **Stats + warnings:** `n/100` cards, lands, avg MV (nonland), identity, game changers vs the bracket
  cap (`GC_CAP`: B1–2 = 0, B3 = 3); warning chips for count ≠ 100, GC over cap, off-color (vs
  commander identity), not singleton (`anyQty` exempts basics / "any number of cards named"), no or
  illegal commander, unresolved paste names.
- **Groups** follow the full taxonomy (`DECK_ROLES`): `cardGroup(c)` = Commander if in
  `deck.commander`, else the user's `c.role` override, else Lands / first concrete role / Other.
  Sorted by MV then name inside each group.
- **Tiles:** hover −/+/× (hover-capable devices only), qty badge, `#tag` badge, commander outline.
  **Drag** a tile onto any group (empty groups appear as drop zones while dragging) to set its role;
  dropping on its natural group clears the override; dropping on Commander makes it (co-)commander.
- **Card editor modal** (click a tile; the phone-friendly path): quantity stepper, role picker
  ("Auto (X)" + all roles), comma-separated tags, set/unset commander, remove, Scryfall link.
- **Undo:** 40-deep per-chat stack (reset on chat switch); toolbar button, Ctrl+Z (outside inputs),
  and the toast after add/remove. A pasted decklist replacing the cards is also undoable.
- **Old decks** (saved before `can_command`/`any_qty` existed) fall back to type-line checks.
- The AI still doesn't *see* manual edits — it reads chat history only. Sending `deck` to `/chat`
  is the first step of Phase 3.

---

## 8. Deck Workbench roadmap

The agentic deckbuilder, built in shippable phases (user-approved scope):

- **Phase 1 — deck object + populate + render** ✅ *done.* (`POST /deck/parse`, `deck` state,
  `renderDeck`.)
- **Phase 2 — full manual editor** ✅ *done (2026-09-30).* See [Manual editor](#manual-editor-phase-2).
  Possible follow-ups: auto-detect the commander from a pasted list's commander section, a mana-curve
  chart, maybeboard/sideboard.
- **Phase 3 — AI edits via accept/reject.** The AI proposes discrete cuts/adds (a structured
  mechanism — a dedicated tool the model calls, or a new SSE event type carrying proposed changes);
  the UI renders each as an **accept/reject** card; accepting mutates the `deck` object. **The deck
  stays the user's — nothing changes without their click.** The AI should read the current `deck`
  object as authoritative state (send it in the request alongside `history`).
- **Phase 4 — intake wizard.** A guided Q&A (in-app) that pins down the deck's goal / gameplan /
  bracket / constraints, then feeds that as focused context so every suggestion is anchored.

**Design guidance for phases 3–4:** send the `deck` object to `/chat` (like `history`) so the AI
reasons over structured state, not re-parsed text. Keep changes proposal-based (accept/reject) — the
user was explicit about staying in control. The contextual roles (Engine Map) are the AI's job; the
concrete roles come free from the otag index.

---

## 9. Dev & test notes / gotchas

- **UI dev loop:** `python dev_server.py` runs a second instance on **:8001** with login off, serving
  `advisor_ui.dev.html` (gitignored) if it exists. Build there, then promote by copying it over
  `advisor_ui.html` (no restart needed for HTML; backend changes need the live server restarted).
- **Testing auth-gated features:** use `fastapi.testclient.TestClient(app, base_url="https://testserver")`
  (https so the Secure cookie sticks) with a throwaway `ADVISOR_USERS` account set in the env before
  importing `advisor_app`.
- **Restart the server** after backend edits (no auto-reload — kill it, the supervisor restarts it).
  Refresh the browser for HTML edits.
- **Windows console encoding**: scripts reconfigure stdout to UTF-8 (`errors="replace"`) — cp1252
  crashes on card glyphs/emoji.
- **Scraping the Academy**: `commandertemplate.com` is a Cloudflare-protected Next.js SPA — plain
  fetches 403 / return an empty shell. Content was captured via the in-app browser and embedded in
  `ingest_academy.py`.
- **Disk**: keep an eye on the `D:` drive; a local ML venv (the abandoned laya fine-tune) was removed
  to reclaim ~6 GB.
- **Conventions**: work on `main`, keep `feat/deckbuilding-advisor` synced to it; end commit messages
  with the Claude Code co-author line.

---

## 10. File map

| File | Role |
|---|---|
| `advisor_app.py` | FastAPI advisor: routes, `agent_stream`, system prompt, caching, tiering, `/deck/parse`, `/card`. |
| `advisor_ui.html` | Single-file SPA: chat, sidebar, canvas, deck object + render. |
| `advisor_login.html` | Sign-in page (served by `GET /login`). |
| `dev_server.py` | Dev instance on :8001 (login off, serves `advisor_ui.dev.html`). |
| `Dockerfile` / `docker-compose.yml` / `requirements-advisor.txt` | Server packaging (see "Moving to a server"). |
| `run_advisor.ps1` / `install_autostart.ps1` | Supervisor that keeps server + tunnel alive / registers it as a logon task. |
| `mtg_tools.py` | Shared tool layer (Scryfall, Spellbook, rules/theory RAG, decklist details) + `TOOLS`/`TOOL_FUNCTIONS`. Used by advisor, Discord bot, MCP server. |
| `role_index.py` | Builds/serves the otag concrete-role index (`role_index.json`). |
| `rules_ingestion.py` | Auto-updating Comprehensive Rules → rules RAG. |
| `theory_ingestion.py` / `theory_sources.py` | YouTube theory transcripts → theory RAG. |
| `ingest_academy.py` | Rebel Lily Academy articles → theory RAG. |
| `distill_theory.py` | Map-reduce the theory corpus into `playbooks/`. |
| `mtg_mcp.py` / `discord_bot.py` | The MCP server and Discord judge-bot (same tool layer). |
