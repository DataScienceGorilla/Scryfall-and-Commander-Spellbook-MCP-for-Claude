# Deckbuilding Advisor — Architecture & Handoff

A local, self-hosted web app that gives **tailored MTG Commander deckbuilding advice** with a
Scryfall card canvas. It's built on the same tool layer as the MCP server / Discord bot in this
repo. This doc is the single source of truth for how it works and how to keep building it.

> **Status (2026-09-30):** the advisor is fully working and tunnel-shareable. The current build-out
> is the **Deck Workbench** (turning it into an agentic deckbuilder) — see [Deck Workbench](#deck-workbench-roadmap).
> Phase 1 (deck object + populate + render) is done; phases 2–4 are next.

---

## 1. Run it

```bash
# from the repo root, main Python env (NOT a venv — deps are in requirements.txt)
python -m uvicorn advisor_app:app --host 127.0.0.1 --port 8000
# open http://localhost:8000
```
- **`.claude/launch.json`** has an `advisor` config (`autoPort: false`, port 8000 — the UI fetches
  relative paths, so the port must be stable for the tunnel).
- **Detached / survives-Claude-session** (how it's run day to day) — a background process, not the
  preview server, because preview servers die on session boundaries:
  ```powershell
  Start-Process -WindowStyle Hidden -WorkingDirectory <repo> -FilePath <python> `
    -ArgumentList "-m","uvicorn","advisor_app:app","--host","127.0.0.1","--port","8000" `
    -RedirectStandardOutput "advisor_server.out.log" -RedirectStandardError "advisor_server.err.log"
  ```
  Detached processes survive session resets but **not a PC reboot** (no startup task yet).
- **The server does NOT auto-reload.** After editing `advisor_app.py` / `mtg_tools.py` / `role_index.py`,
  restart it. Editing `advisor_ui.html` needs no restart — `GET /` reads the file per request, so
  just refresh the browser.

### Auth (optional password gate)
HTTP Basic on all routes, **enforced only when `ADVISOR_PASSWORD` is set** in `.env` (local use
stays open when unset). For tunnel sharing set `ADVISOR_USER` / `ADVISOR_PASSWORD`. Browsers cache
the credentials and resend them on same-origin `/chat`, `/card`, `/deck/parse`.

### Public sharing (Cloudflare tunnel)
`cloudflared.exe` (portable binary, gitignored) run detached against `http://localhost:8000` gives a
random `https://<words>.trycloudflare.com` URL. Quick tunnels get a **new URL each restart** and
aren't reboot-persistent. A named tunnel (free Cloudflare account) would give a stable URL — not set
up yet.

### `.env` keys
`ANTHROPIC_API_KEY` (required), `ADVISOR_USER` / `ADVISOR_PASSWORD` (gate), plus `DISCORD_BOT_TOKEN`,
`APIFY_TOKEN` used by other components. `.env` is gitignored.

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
- **Card canvas** — cards grouped by role in labeled sections (`ROLE_ORDER`). Two feeders:
  1. Chat `[[Card Name]]` mentions → `requestCard()` → `/card` proxy (hover preview + click lightbox).
  2. The **deck object** → `renderDeck()` (see below).
- **Tiles** keep the card aspect-ratio (`63/88`) so full cards show (a prior bug cropped them).

---

## 7. Deck object (the workbench foundation)

### Deck object schema
Per chat, in `localStorage`, sent to the backend with each request. Populated by `POST /deck/parse`.
```js
deck = {
  commander: [],          // names (set by user/AI; not auto-detected yet)
  bracket: null,          // 1–5 (not set yet)
  not_found: [...],       // names /deck/parse couldn't resolve
  cards: [{
    name, qty, scryfall_uri, type_line, cmc, mana_cost,
    color_identity: [...],
    image,                // normal-size art
    game_changer: bool,   // WotC game-changers list (Scryfall flag)
    is_land: bool,
    roles: [...],         // CONCRETE roles from the otag index
    role: null,           // CONTEXTUAL role — assigned by AI/user later
    tags: [],             // user tags
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

---

## 8. Deck Workbench roadmap

The agentic deckbuilder, built in shippable phases (user-approved scope):

- **Phase 1 — deck object + populate + render** ✅ *done.* (`POST /deck/parse`, `deck` state,
  `renderDeck`.)
- **Phase 2 — full manual editor.** Search-to-add cards, remove / quantity steppers, set commander &
  bracket, per-card role/tag editing, drag between role groups, live-updating composition. All
  client-side against the `deck` object; add-card can reuse `/card` for resolution.
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

- **Testing the UI in a browser + the auth quirk:** a page loaded with credentials in the URL
  (`http://user:pass@host`) **rejects relative `fetch`** ("credentials in URL"). To test features
  that fetch (`/card`, `/deck/parse`): navigate once *with* creds to cache Basic auth, then navigate
  to the **bare** URL so the document URL is clean and relative fetches work.
- **Restart the detached server** after backend edits (no auto-reload). Refresh the browser for HTML
  edits.
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
| `mtg_tools.py` | Shared tool layer (Scryfall, Spellbook, rules/theory RAG, decklist details) + `TOOLS`/`TOOL_FUNCTIONS`. Used by advisor, Discord bot, MCP server. |
| `role_index.py` | Builds/serves the otag concrete-role index (`role_index.json`). |
| `rules_ingestion.py` | Auto-updating Comprehensive Rules → rules RAG. |
| `theory_ingestion.py` / `theory_sources.py` | YouTube theory transcripts → theory RAG. |
| `ingest_academy.py` | Rebel Lily Academy articles → theory RAG. |
| `distill_theory.py` | Map-reduce the theory corpus into `playbooks/`. |
| `mtg_mcp.py` / `discord_bot.py` | The MCP server and Discord judge-bot (same tool layer). |
