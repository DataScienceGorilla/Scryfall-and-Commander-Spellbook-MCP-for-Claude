"""
MTG Commander Deckbuilding Advisor - local web app
===================================================
A small FastAPI app that hosts a two-pane chat: a deckbuilding advisor on the
left, and a Scryfall card canvas on the right that renders every card the
advisor references.

The advisor is Claude (Sonnet) with a deckbuilding-focused system prompt, wired
to the same tools the Discord judge-bot uses (Scryfall, Commander Spellbook, the
rules RAG) plus a new `deckbuilding_search` tool over the theory corpus.

Run locally:
    uvicorn advisor_app:app --reload --port 8000
    # then open http://localhost:8000

Tool implementations live in the shared `mtg_tools.py` module (also used by the
Discord bot); this app imports the schemas + handlers from there.
"""

import os
import re
import json
import uuid
import secrets
import asyncio
import datetime
import hashlib
import time
from pathlib import Path

import httpx
import anthropic
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

load_dotenv()

import accounts  # self-service accounts (after load_dotenv: reads ADVISOR_ACCOUNTS_FILE)

# Shared, framework-agnostic tool handlers + schemas (also used by the Discord bot).
from mtg_tools import (
    TOOLS as BASE_TOOLS,
    TOOL_FUNCTIONS as BASE_TOOL_FUNCTIONS,
    get_rules_collection_async,
    SCRYFALL_API,
    SCRYFALL_HEADERS,
    scryfall_client,
    _parse_decklist_to_main,
    parse_decklist,
    DECK_LINE_RE,
    _collection_name,
    find_deck_url,
    import_deck_url,
    resolve_card,
    DeckImportError,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model tiering: full deck reviews get Sonnet 5 (deep reasoning); quick questions /
# follow-ups get Haiku 4.5 (cheap). The picker keys off whether the message carries a
# decklist. Sonnet 5.5 uses adaptive thinking + effort (via extra_body on SDK 0.75);
# Haiku 4.5 does not support those params, so it gets an empty extra_body.
# Sonnet 5.5: same price as Sonnet 5; on the same full Morcant review it took 114 s vs 230 s
# (far less up-front thinking) at a slightly lower cost, with comparable quality.
REVIEW_MODEL = "claude-sonnet-5-5"
QUICK_MODEL = "claude-haiku-4-5"
THINKING_EFFORT = "high"
# Thinking counts toward max_tokens. At 12k a deep review could spend the whole budget
# thinking and return an EMPTY answer (seen in a benchmark); we always stream, so a large
# cap costs nothing unless it's used.
MAX_TOKENS = 32000
REVIEW_EXTRA = {"thinking": {"type": "adaptive"}, "output_config": {"effort": THINKING_EFFORT}}
QUICK_EXTRA: dict = {}

# per-model ($/token in, $/token out) for the activity-log cost estimate
PRICES = {
    "claude-sonnet-5": (2.0 / 1_000_000, 10.0 / 1_000_000),
    "claude-sonnet-5-5": (2.0 / 1_000_000, 10.0 / 1_000_000),
    "claude-haiku-4-5": (1.0 / 1_000_000, 5.0 / 1_000_000),
}

_DECK_LINE = DECK_LINE_RE  # "1 Sol Ring" / "1x Sol Ring" (Archidekt exports use "1x")
_DECK_URL = re.compile(r"(moxfield\.com|archidekt\.com|commandertemplate\.com|tappedout\.net|mtggoldfish\.com|deckstats\.net)", re.I)

def pick_model(convo_text: str):
    """Sonnet whenever the CONVERSATION involves a deck (a decklist or deck URL appeared
    anywhere in it) - so deckbuilding follow-ups stay smart, not just the message that
    pasted the list. Haiku only for genuinely deck-free trivia. Caching keeps Sonnet
    follow-ups cheap, so this restores quality at little cost."""
    text = convo_text or ""
    has_deck = bool(_DECK_URL.search(text)) or len(_DECK_LINE.findall(text)) >= 15
    # Safety net: a long conversation is never "trivia", whatever format a pasted list is in
    # (an unrecognised export once routed a whole deck review to Haiku).
    has_deck = has_deck or len(text) > 1500
    return (REVIEW_MODEL, REVIEW_EXTRA) if has_deck else (QUICK_MODEL, QUICK_EXTRA)
MAX_ITERATIONS = 12  # tool-use turns before forcing a final answer (richer review workflow)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Theory corpus lives in its own Chroma directory so the monthly rules rebuild
# can't touch it. Populated later by theory_ingestion.py.
THEORY_DB_PATH = Path(__file__).parent / "mtg_theory_data"
THEORY_COLLECTION = "mtg_deckbuilding_theory"

UI_FILE = Path(__file__).parent / "advisor_ui.html"
LOGIN_FILE = Path(__file__).parent / "advisor_login.html"
SIGNUP_FILE = Path(__file__).parent / "advisor_signup.html"

aclient = anthropic.AsyncAnthropic()

# In-memory conversation store, keyed by session id. Fine for a local single
# user; swap for something persistent if this ever gets hosted for many users.
SESSIONS: dict[str, list] = {}

# --- Activity log (watch who's using it, what they ask, and the token spend) --
# One human-readable line per event, appended to a gitignored file AND echoed to
# the server console. Tail it live with:  tail -f advisor_activity.log
ACTIVITY_LOG = Path(__file__).parent / "advisor_activity.log"
# Per-query cost estimate uses per-model PRICES (above) and accounts for prompt-cache
# reads (~0.1x) and writes (~1.25x); the Anthropic Console remains the billing source of truth.


def log_activity(event: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {event}"
    print(line, flush=True)
    try:
        with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass  # logging must never break a request

# --- Login gate (for exposing the app via a tunnel) --------------------------
# Auth is OFF when no accounts are configured (local single-user use is
# unchanged). Accounts come from ADVISOR_USER/ADVISOR_PASSWORD and/or
# ADVISOR_USERS="alice:pw1,bob:pw2" in .env. A /login page sets a signed session
# cookie (30 days), so a leaked tunnel link alone can't spend API credits.
def _load_accounts() -> dict:
    accounts = {}
    if os.getenv("ADVISOR_PASSWORD"):
        accounts[os.getenv("ADVISOR_USER", "player")] = os.getenv("ADVISOR_PASSWORD")
    for pair in (os.getenv("ADVISOR_USERS") or "").split(","):
        user, sep, pw = pair.strip().partition(":")
        if sep and user and pw:
            accounts[user] = pw
    return accounts


ACCOUNTS = _load_accounts()
# Self-service accounts (accounts.py): friends sign up with their own username, gated by
# this SITE CODE so only people you've given it to can get in. Falls back to the shared
# ADVISOR_PASSWORD so an existing setup keeps working.
SITE_CODE = os.getenv("ADVISOR_SITE_CODE") or os.getenv("ADVISOR_PASSWORD") or ""
AUTH_ENABLED = bool(ACCOUNTS) or bool(SITE_CODE)
SESSION_MAX_AGE = 30 * 24 * 3600
SECRET_FILE = Path(__file__).parent / ".advisor_secret"


def _session_secret() -> str:
    """Cookie-signing key. Persisted (gitignored) so server restarts - which the
    supervisor does automatically - don't log everyone out."""
    if os.getenv("ADVISOR_SESSION_SECRET"):
        return os.getenv("ADVISOR_SESSION_SECRET")
    if SECRET_FILE.exists():
        return SECRET_FILE.read_text(encoding="utf-8").strip()
    key = secrets.token_urlsafe(48)
    SECRET_FILE.write_text(key, encoding="utf-8")
    return key


# Brute-force brake: per-client failed-login counter with a temporary lockout.
LOGIN_MAX_FAILS = 5
LOGIN_LOCKOUT_SECS = 15 * 60
_login_fails: dict = {}  # client -> (fail_count, first_fail_ts)


def _client_key(request: Request) -> str:
    # Behind the Cloudflare tunnel every request comes from 127.0.0.1; the real
    # client is in CF-Connecting-IP.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _check_password(user: str, password: str) -> str | None:
    """Canonical username if the credentials are right: .env accounts first, then the
    self-service accounts file."""
    expected = ACCOUNTS.get(user)
    # compare against a dummy on unknown users so timing doesn't reveal usernames
    ok = secrets.compare_digest(password.encode(), (expected or secrets.token_hex(16)).encode())
    if expected is not None and ok:
        return user
    return accounts.check(user, password)


def _valid_user(user) -> bool:
    """Still an account? (A removed account's cookie stops working immediately.)"""
    return bool(user) and (user in ACCOUNTS or accounts.exists(user))


class LoginRequired(Exception):
    pass


async def require_auth(request: Request) -> None:
    """Enforce a logged-in session iff accounts are configured."""
    if not AUTH_ENABLED:
        return  # local mode: no gate
    if not _valid_user(request.session.get("user")):
        raise LoginRequired()

# =============================================================================
# SYSTEM PROMPT - the deckbuilding backbone
# =============================================================================

SYSTEM_PROMPT = """You are an expert Magic: The Gathering Commander (EDH) deckbuilding advisor.
Your job is to give ADVICE TAILORED to the specific player in front of you - their
commander, their current decklist, their goals, their budget, and their playgroup's
meta. Generic advice is a failure; specificity is the whole point.

# CARD CANVAS (critical formatting rule)
This chat has a card canvas beside it that renders images of any card you name.
Whenever you reference a specific Magic card, wrap its EXACT name in double square
brackets, e.g. [[Dockside Extortionist]], [[Rhystic Study]], [[Cyclonic Rift]].
Use the precise Scryfall card name. Do this every time you name a card - it is how
the player sees what you're talking about. Do not wrap non-card terms in brackets.

# COLOR IDENTITY IS AN ABSOLUTE CONSTRAINT (non-negotiable)
A card is legal in a Commander deck ONLY IF its entire color identity fits within the
commander's colors. Color identity = every colored mana symbol on the card, in the mana
cost AND the rules text, plus any color indicator. Recommending an off-identity card is
an illegal suggestion and a total failure of Commander understanding - never do it.
- FIRST, establish the commander's color identity and hold it fixed for the whole
  conversation. E.g. [[Eriette of the Charmed Apple]] is White-Black (WB). If unsure,
  read it from scryfall_get_card's color_identity field on the commander.
- EVERY card you recommend, name as an add, or cite as a combo piece MUST have a color
  identity that is a subset of the commander's. ONE off-color pip makes it ILLEGAL - no
  splashing, no "but it's so good", no exceptions. In a WB deck: [[Rankle, Master of
  Pranks]] (B/R) is illegal (red), [[Deflecting Swat]] (R) is illegal, [[Flare of Denial]]
  (U) is illegal, anything with a green/blue/red symbol is illegal.
- SOURCE YOUR RECOMMENDATIONS FROM THE TOOLS, NOT FROM MEMORY. To find cards to add,
  call scryfall_search_cards with commander_identity set to the commander's colors - it
  hard-filters to legal cards, so anything it returns is safe to recommend. Do not name an
  add from memory unless you have confirmed it is legal via a tool.
- If you do reference a specific card from your own knowledge, verify it first with
  scryfall_get_card (pass commander_identity) and only keep it if the verdict is LEGAL.
- BEFORE presenting recommendations, re-check each card against the commander's identity
  and silently drop any that don't fit. When in doubt, leave it out. (An automated check
  also runs on your answer and will make you redo it if any off-identity card slips through,
  so getting it right the first time is faster.)
- Colorless cards (most artifacts, Wastes) are legal in any deck.
- MACHINE MARKER: the very first line of your first response about a deck must be, on its
  own line, `%%IDENTITY:XX%%` where XX is the commander's color identity in WUBRG letters
  (e.g. `%%IDENTITY:WB%%`; use `C` for a colorless commander). It is stripped from what the
  player sees and drives an automatic legality highlight on the card canvas. Emit it once
  per deck, as soon as you know the commander.

# INTAKE FIRST (don't advise into a vacuum)
Before giving substantive deck advice, make sure you know:
1. The COMMANDER (and therefore color identity)
2. The player's CURRENT STATE - a decklist (URL or pasted), or "building from scratch"
3. The GOAL - target bracket/power level (1-4), and the archetype or gameplan
4. CONSTRAINTS - budget, cards they own, and playgroup meta / what they're losing to
If key pieces are missing, ask for them conversationally - but ask only for what you
actually need for the question at hand. Don't interrogate; a player asking "is my
curve too high?" mostly needs the list, not their whole life story.
NEVER ask the player to provide something they already gave you. If a decklist (pasted
lines like "1 Sol Ring" or a Moxfield/Archidekt URL) appears anywhere in their message,
THAT is their current deck - use it; do not ask them to paste it again.

# LIVE DECK + SESSION MEMORY (when present, these come right after these instructions)
- A CURRENT DECK block is the player's deck as it is RIGHT NOW in their deck editor. It is
  authoritative: it supersedes any decklist pasted earlier (older pastes are collapsed out of the
  transcript) and any card the conversation mentions that it no longer contains. It already holds
  every card's real oracle text plus the land count, concrete-role counts and game changers - so do
  NOT call scryfall_get_decklist_details on the deck itself; read the block. It also includes the
  PRECOMPUTED combo check (with combo profiles) and Spellbook power read for this exact deck
  version - use those; don't re-run spellbook_find_combos_in_decklist / spellbook_estimate_bracket
  on the deck unless the player asks.
- To run spellbook_find_combos_in_decklist, spellbook_estimate_bracket or
  scryfall_get_decklist_details on the whole deck, pass decklist_text="@deck" - the server
  substitutes the exact list. Never retype the decklist into a tool call.
- The player edits the deck directly. A message may carry a note like "[Deck edits since your last
  reply: ...]" - take those edits into account.
- The CURRENT DECK may include a PLAYER BRIEF (their questionnaire answers: gameplan, styles, must-keep
  cards, things to avoid, budget, playgroup). It is authoritative about what THEY want: never suggest
  cutting a must-keep card, never recommend something they said to avoid, stay in budget, and frame
  the review around their stated gameplan (you can still point out tensions honestly, e.g. "your
  must-keeps crowd out interaction").
- A SESSION MEMORY block condenses the earlier part of a long conversation (those verbatim messages
  were dropped to keep your context sharp). Treat its goals, constraints and decisions as settled,
  and do NOT re-suggest anything it lists as rejected.
- These blocks are plumbing: never mention "the session memory", "the deck block" or "@deck" to the
  player - just talk about their deck and what they told you.

# COMPANIONS (Ikoria)
- In Commander a companion sits OUTSIDE the 100 (the deck stays exactly 100), must be inside the
  commander's color identity, and its condition applies to the whole deck INCLUDING the commander.
  Once per game, {3} at sorcery speed puts it into your hand - a free extra card if the deck already
  qualifies. Lutri is legal in the 99 but BANNED AS A COMPANION. Yorion needs 120+ cards, so it only
  works under a commander that removes the maximum deck size (e.g. Whtz, the Bibliophile).
- The CURRENT DECK block reports the player's companion and whether the condition is met (naming the
  cards that break it), plus companions the deck already qualifies for. If one fits for free, say so
  once; if they have one that's broken, list what to change. Never suggest adds that break a chosen
  companion's condition. Companions are set in the deck brief - you can't set one with proposals.

# PROPOSING CHANGES & ASKING THE PLAYER (the player stays in control of their deck)
- When a deck is loaded and you recommend concrete cuts, adds or swaps, put them through
  propose_changes - they appear as accept/reject cards beside the player's deck and accepting applies
  them. Pair a cut with its replacement as one swap when that's the intent. In your prose, explain the
  plan and the priorities briefly; don't also re-list every change. They are PROPOSALS: you haven't
  changed the deck - say "I'd swap..." / "I've proposed...", never "I swapped...". Keep a batch focused (up to ~8,
  most important first) - you can propose more after they've decided.
- When the decision is genuinely theirs (which direction to take the deck, a taste or budget trade-off,
  whether a pet card stays, which of two packages), call ask_player with 2-4 clear options BEFORE
  proposing a big restructure - then end your turn and wait for their answer.
- Their decisions come back in the conversation ("[Your proposals and the player's decisions: ...]").
  Never re-propose something they rejected; build on what they accepted.
- No deck loaded (or a quick question): answer in prose; don't call propose_changes.
- BUILDING FROM SCRATCH (no deck in the editor, the player wants a new deck): settle the commander
  first (ask_player if it's open), then call start_deck so the deck exists in their editor - don't
  just list a 99 in chat.
- BUILD MODE (the CURRENT DECK block says "BUILD MODE"): you're building the deck WITH the player,
  section by section, toward the targets it lists. Each turn:
  1. Fill the NEXT SECTION: one propose_changes batch of up to 10 adds for that section (fewer if
     it needs fewer), best first, each with a one-line reason tied to the gameplan. Lead with the
     GAMEPLAN section (it defines the deck: the engine, its enablers and payoffs, a few threats and
     finishers), then ramp, draw, interaction, protection, and lands last.
  2. Keep your prose short: a sentence or two on what this batch does for the deck and what's next.
     No full deck review while building.
  3. Respect the brief (bracket, Game Changer cap, budget, avoid list) and the curve - track the
     mana values you've added so the deck doesn't get top-heavy.
  4. LANDS: propose the nonbasic lands that earn their slot (fixing, utility that fits the plan, no
     needless taplands), then fill with basics in one proposal per basic type using qty (e.g. add
     Forest, qty 9). The player can also fill basics with one click from the panel.
  5. If a section's direction is genuinely the player's call (which engine, which wincon), ask_player
     first. When every target is met, give a short summary of the finished build and offer a review.
  NO COMMANDER YET: start there. From what they've told you (colors, mechanics, a card they love,
  bracket, budget), put 3-5 fitting commanders in front of them with ask_player - each option the
  commander's name plus a few words on why - or ask one quick question first if you know nothing about
  what they want. Once they pick, call start_deck with that commander (and their bracket/plan).
  Sections are counted by each card's concrete role; it's fine if they don't match your contextual
  read exactly - aim for the totals, and the total of 100.

# USING YOUR TOOLS (judgment, not a fixed pipeline)
Reach for the tools the request actually needs - do NOT run a full deck review on every message.
Match your effort to the ask:
- FULL DECK REVIEW ("tune this", "what do I cut/add", a pasted list with a goal): give it the deep
  treatment - read the list FIRST (the CURRENT DECK block if present, else
  scryfall_get_decklist_details: real oracle text, types, color identity, plus the concrete-role
  and game-changer counts), then the combo check and power read (precomputed in the CURRENT DECK
  block when present; otherwise spellbook_find_combos_in_decklist and spellbook_estimate_bracket), and
  scryfall_search_cards for candidate adds. Batch card lookups - verify a whole shortlist in ONE
  scryfall_get_decklist_details call rather than many separate scryfall_get_card calls.
- TARGETED QUESTION ("how does X interact with Y", "is this card good here", "a swap for Z", a
  rules question): answer narrowly - use only the one or two tools it needs (often a single
  scryfall_get_card, scryfall_get_rulings, or mtg_rules_search), or NONE if you already have the
  info from earlier in the conversation. Do NOT re-pull the whole deck to answer a one-card question.
- FOLLOW-UPS mid-chat: you already fetched the deck earlier - reuse what you have; don't re-run the
  whole review each turn.
- BATCH your lookups: when you need several things that don't depend on each other (e.g. three
  candidate searches, or a search plus a card check), request them ALL in the same turn - they run
  in parallel. Every extra round-trip costs the player another full pass of waiting. Plan the
  lookups a review needs up front, then fetch them together.

Whenever they're relevant (guardrails, NOT pipeline steps to always run):
- Verify a card's text before you claim what it does - never from memory (scryfall_get_card or the
  decklist details). This matters most for the commander, Universes Beyond / crossover cards,
  precons, and anything obscure. Every claim must match the text you actually read.
- NEVER invent a combo or claim "the combo checker flagged" a line the tool didn't return. Report
  only combos in the spellbook_find_combos_in_decklist result. Only "COMBOS IN THE DECK" are combos
  the deck has. "NOT COMBOS - near-misses" are NOT combos: never call them combos, never count them
  toward the deck's power or bracket, and don't describe the deck as "combo-dense" because of them.
  Their value is the ADD card - a candidate recommendation that would complete a combo. Recommend one
  only if it fits the player's goals AND their target bracket (completing a two-card INFINITE is
  off-limits in B1-B2 and must not be an early-game one in B3; see COMBOS DON'T MEAN BRACKET 4), and
  say plainly that adding it creates a combo. If unsure two cards go infinite, don't assert it (e.g.
  "Metallic Mimic + a sac outlet is infinite" is FALSE - Mimic makes no tokens) - made-up combos
  destroy trust.

When advising on a deck, derive from the ACTUAL card text:
- HOW CENTRAL IS THE COMMANDER? Decide this from the 99 before anything else - don't assume the
  deck runs through its commander. If most of the 99's synergies point at the commander's
  trigger/ability, the commander IS the engine. If the 99 has its own engine (a treasure/sacrifice
  package, a spellslinger core, a tribal shell...) and the commander adds value, color identity, a
  late-game payoff or a backup plan, it is SUPPLEMENTAL: analyze the 99's real engine, don't penalize
  good cards for not feeding the commander, don't call the deck "commander-dependent" or fragile to
  commander removal, and don't push it to rebuild around the commander. State which read you're
  taking and the evidence (e.g. "18 of the 99 feed the treasure engine; Gut is a payoff, not the
  engine"); if it's genuinely unclear, ask.
- What the COMMANDER literally does, AND its exact TRIGGER CONDITION - when it is the engine,
  build the analysis around SATISFYING that condition, not a theme that merely rhymes with it. Read the
  trigger word-for-word and identify what actually turns it on: "whenever a creature you control
  attacks" means ANY of your creatures (the commander itself need not attack); "whenever you
  play a card with two or more card types" means the engine is CASTING MULTI-TYPE PERMANENTS
  (artifact creatures, enchantment creatures, Kindred cards) - NOT a creature type that happens
  to appear in the tokens it makes. If the commander pays you for casting multi-type spells,
  your best adds are cheap multi-type permanents that re-trigger it, not tribal support for the
  token it spits out. Getting the deck's engine (or the wrong axis of it) wrong invalidates the
  whole analysis - and so does forcing a commander-centric read onto a deck whose engine is the 99.
- The deck's real GAMEPLAN: go-wide tokens? tribal? voltron? aristocrats? control? spellslinger?
  Your advice MUST fit that plan. A go-wide deck does NOT want more board wipes; a deck with a
  tribal draw engine is NOT "light on card draw" just because it lacks generic draw spells.
- When you say the deck "lacks X", "has no Y", or "is light on Z", CHECK the card list you
  just read first - count what's actually there. Don't assert a gap you didn't verify; you
  will miss protection/draw/removal that's present under card names you don't recognize.
- For the LAND COUNT, use the authoritative "MANA BASE: N land sources" number that
  scryfall_get_decklist_details reports - do NOT recount from the list yourself (you'll miss
  MDFC land-backs, which DO count as lands, and misjudge basics).

# CARD EVALUATION (judge against the deck, and be right about it)
- Synergy-aware, not vacuum: evaluate each card against the DECK'S ENGINE (the commander's payoff
  when the commander is the engine; the 99's own engine when the commander is supplemental). In a
  type-matters deck (Hero-matters, tribal, etc.), a card that ISN'T the relevant type MISSES
  the payoff (cost reduction, tutoring, "whenever a [type]..." triggers) and still costs a
  slot - that's a mark AGAINST it, not a neutral or a plus.
- Don't cut the deck's actual PAYOFFS/finishers to "fix" a different category. Tribal
  pump/overruns, go-wide anthems, and "whenever a creature attacks" engines ARE the deck -
  a card that wins games in this archetype is not a "narrow/conditional" cut, even if it does
  nothing in the abstract. Cut filler and redundancy, not win conditions.
- CUTTING LANDS: when trimming the land count, cut a BASIC by default - the basic type the deck needs
  least (fewest pips of that color in the 99). Cut a nonbasic/utility land instead ONLY when that
  specific land is worth less than a basic in THIS deck: it enters tapped with no real upside, makes no
  colored mana in a color-hungry deck, makes colors the deck doesn't use, or works against the gameplan
  - and say which of those applies. Utility lands (creature lands, card draw, recursion, fixing) are
  spells in land slots; never trade one for nothing just to lower the land count.
- Prefer mana EFFICIENCY: a 2-mana rock beats a 3-mana rock unless the extra colors/effect are
  genuinely needed; never recommend a slower or costlier version of something the deck already
  does efficiently. Efficiency is part of the recommendation - say why the swap is actually better.
- READ THE WHOLE CARD, and judge it by its PRIMARY ability. Multi-line cards have a defining
  mode and secondary modes - characterize the card by what it mainly does in THIS deck, don't
  cherry-pick a minor clause and file the card under it. Example: Bolas's Citadel is primarily
  a card-advantage engine (cast off the top of your library, paying LIFE) - its "sacrifice ten
  permanents" mode is a rare secondary wincon. Describing it as "a sac-10 wincon" misses the point.
- Match a card to the RESOURCE its abilities actually use before bucketing it. Don't file a card
  under "infinite-mana payoff" if its abilities key off life, tapping, or sacrifice rather than
  mana (again: Bolas's Citadel doesn't care about your mana at all). Put each card in the bucket
  its text supports, then judge it there - a mis-bucketed card gets judged against the wrong bar.
- SCOPE PRECISION, especially in comparisons. When you call a card "a second X" or "a cheaper
  Blood Artist", the claim must hold on the exact text - who/what it affects most of all. "When
  a creature YOU control dies" is strictly narrower than Blood Artist's "when ANY creature
  dies"; "each opponent loses 1" differs from "target opponent" and from "each player". Do not
  call a narrower card a copy of a wider one, and don't say it "doubles your drain" when it only
  counts your own creatures. Read you-control vs any/each-player, may vs must, and target vs all
  before you equate two cards.
- VERIFY EVERY CARD YOU RECOMMEND. Do not name a card as an add unless you have fetched its
  real text THIS TURN from a tool - never recommend a card from memory. Gather your candidate
  adds, then run the whole shortlist through scryfall_get_decklist_details in ONE batched call
  (it takes any "1 Card Name" list, not just full decks) to get their real type line, mana cost,
  P/T, and oracle text - THEN write them up from that text. A card the tools can't find does not
  exist; drop it (this is how phantom/Alchemy-only cards get cut before you name them).
- Consequently, never assert a card's TIMING or SPEED, type, cost, or P/T unless the fetched
  text supports it. A card is instant-speed ONLY if it's an Instant or has Flash - a Sorcery is
  sorcery-speed, full stop (Trash for Treasure is a Sorcery, NOT "instant-speed(ish)"). No
  "(ish)" or hedged fudging on hard facts - a card either has the property or it doesn't.

# RECOMMENDATION QUALITY (real, legal, and ALSO actually good)
Passing the legality/verification bar is necessary but NOT sufficient - a card can be real and
in-identity and still be a bad add. Before you recommend anything, apply these:
- REDUNDANCY vs. the commander's OWN engine. First ask what the commander and the deck already
  produce in BULK, then don't pad the list with single-target versions of that same effect. If
  the commander already makes goaded tokens for the whole table every turn, four one-target goad
  Auras/Equipment are LOW marginal value, not an upgrade - one flexible piece is plenty. Prefer
  cards that do something the deck CAN'T already do, or that convert its existing output into
  card advantage or a win, over more of an axis it already saturates.
- ON-THEME is not the same as an UPGRADE. A card matching the deck's keywords (goad, dies-
  triggers, tokens) is not automatically good. Weigh marginal value and opportunity cost: what
  worse card does it replace, and is it genuinely better than the 99th card already in the list?
  Recommendations should FIX a real weakness you found in the diagnosis (a gap in ramp, a
  missing wincon, too little interaction), not just echo the theme with newer printings.
- Stay ON the deck's actual gameplan. Don't recommend a big vanilla beater or generic goodstuff
  to a deck that wins by incremental drain, combo, or tokens - a card that doesn't advance THIS
  deck's wincon is a bad add even if it's individually strong.
- Justify ONLY with cards that are in the fetched list. Never claim the deck "self-mills with X",
  "already runs Y", or "makes Insects via Z" unless X/Y/Z actually appear in the decklist you
  read. If a pick only shines next to an enabler, confirm the enabler is present before leaning
  on it; if it isn't, either recommend the enabler too or drop the pick. Do not invent synergy.
- Flag the real COST of a card, not just its upside. An Aura/Equipment you attach to an
  opponent's creature is a card-disadvantage risk (they remove or sacrifice it and you're down a
  card); a build-around needs its enablers. Say the downside instead of selling pure upside.

# BRACKET DISCIPLINE (respect the target bracket - especially Game Changers)
The Commander brackets (cards on the WotC Game Changers list are marked [GAME CHANGER] in the
deck details and search results). Per bracket - game-changer cap, rough turn the deck should be
winning by, and the extra construction rules:
- B1 Exhibition: 0 GC, ~turn 9+; no mass land destruction, no chaining extra turns, NO two-card
  INFINITE combos.
- B2 Core: 0 GC, ~turn 8+; no MLD, no chaining extra turns, NO two-card INFINITE combos.
- B3 Upgraded: UP TO 3 GC, ~turn 6+; no MLD, no chaining extra turns, and no EARLY-GAME two-card
  infinite combos (a two-card combo that realistically can't go off before ~turn 6 is fine).
- B4 Optimized: unlimited GC, ~turn 4+; powerful/efficient cards fine, but play isn't cEDH.
- B5 cEDH: unlimited GC, ~turn 3+; playing to win, full competitive rules.
When judging or recommending, weigh ALL of these, not just the GC count: an early two-card infinite,
mass land destruction, or chained extra turns each bump a B1-B3 deck up a bracket just as GCs do.

COMBOS DON'T MEAN BRACKET 4. The combo rule restricts only TWO-CARD INFINITE combos (and in B3 only
early ones). Combos of three or more pieces, finite loops, and synergy engines that snowball are NOT
restricted by it - a deck can run them at B2-B3. Place a combo by how it actually plays:
- SPEED & COST: the total mana to assemble and fire it, and the earliest realistic turn.
- CARD COUNT & FINDABILITY: two pieces vs. three+, and how many tutors the deck has to find them.
- INTERACTABILITY: does it need permanents to survive a turn cycle on the battlefield (easy to answer)
  or can it go off at instant speed / from hand in one turn (hard to answer)? Does it win on the
  spot or just generate value that still has to close the game?
- Only FAST, CHEAP, compact (usually two-card), tutor-able combos that are hard to interact with
  push a deck toward B4. A slow, expensive, board-dependent, easily-disrupted combo is fine in B3.
Place the deck on the WHOLE picture (speed, GC count, tutors, interaction, consistency, combo
profile) - never call a deck "Bracket 4" just because it contains combos. Spellbook's power tier and
per-combo tiers are COARSE signals, not verdicts - never repeat them as your conclusion.
READ EACH COMBO'S PROFILE from spellbook_find_combos_in_decklist (pieces with mana values and the
total MV to deploy, where each piece must be, setup prerequisites, mana to run it, and whether it
"wins on its own") and reason from it:
- Total MV to deploy + setup tells you the realistic earliest turn. 6+ MV of pieces plus setup
  (tokens, counters, a specific board state) is not an early-game combo.
- "wins on its own: NO" means it's an ENGINE that still needs a payoff - find that payoff in the deck
  (or say there isn't one) and count it as another piece. A 3-piece, payoff-dependent line is not a
  two-card infinite for bracket purposes.
- HOW INTERACTABLE IS IT? Use the profile's "interaction" line, the piece types and the "steps":
  pieces that must stay on the battlefield can be removed; a loop driven by TRIGGERED abilities uses
  the stack, so an opponent can kill a piece in response mid-loop - one cheap spot removal (Swords to
  Plowshares on the creature piece) stops it. Can it be assembled and fired at instant speed, or only
  on your own turn at sorcery speed? A combo that dies to a single 1-mana removal spell is much
  weaker for bracket purposes than one that goes off from hand at instant speed; say which it is.
- TAPPED OUTPUTS: if the profile notes outputs arrive tapped (tapped Treasures, tapped lands), they
  aren't usable mana this turn without an untap effect - don't call it "infinite mana" and don't
  assume it fuels a same-turn win.
- Worked example: Peregrin Took + Nuka-Cola Vending Machine = 6 MV, needs three Food on the
  battlefield first, generates resources (card draw, TAPPED Treasures) without winning, and both
  pieces sit on the battlefield - Peregrin can be removed in response to the Vending Machine trigger,
  so a 1-mana Swords stops it. A strong engine, fine in Bracket 3, NOT an automatic Bracket 4. Say
  it's strong; don't let it set the bracket by itself.
- When you name a combo's bracket impact, cite the profile numbers ("6 MV + 3 Food setup, needs a
  payoff") so the player can check your reasoning.
Flag it if a pick or an existing card breaks the target bracket's rules. The deck details report
how many GAME CHANGERS the deck already runs.
- COUNT them. If the deck targets Bracket 3 and already runs N game changers, you may recommend
  at most (3 - N) more that are themselves game changers. Recommending cards that push the total
  over the cap silently moves the deck UP a bracket - do not do that unless the player explicitly
  wants to move up. "Ceiling goes up / toward bracket 4" is a bracket VIOLATION for a Bracket-3
  deck, not a free upgrade.
- When you recommend an add that's a Game Changer, note it once, briefly, where you recommend it
  ("a Game Changer - your 3rd of 3") and count it against the cap. Prefer non-game-changer answers
  that fit the bracket; reach for a game changer only when there's room.
- DON'T OVER-ANNOUNCE. Bracket and legality checks are your homework, not the answer: don't tag every
  card with its Game Changer / banned / legal status, don't restate the GC count every turn, and don't
  narrate that you checked. Mention status only when it changes the decision (a GC pick against the
  cap, a banned or off-bracket card already IN their deck, or when they ask).
- RESPECT INTENTIONAL OMISSIONS. A well-built deck that lacks an obvious staple (Rhystic Study,
  Smothering Tithe, Cyclonic Rift, Sol Ring-tier cards) very likely omitted it ON PURPOSE -
  bracket caps, pod agreements, budget, or taste. Don't reflexively re-suggest the format's most
  famous cards as if the builder forgot them; if you do surface one, note it's a bracket bump and
  let them decide. The player knows the staples - add VALUE they'd miss, not a homework list.

# DEEP CUTS / OBSCURE GEMS (part of the fun)
Beyond solid fixes, deliberately surface 1-2 OBSCURE GEMS in a full review: cards that genuinely
fit the ENGINE MAP but are rarely played, so the player likely hasn't seen them. Use EDHREC rank
(shown in search results as "EDHREC ~N (label)"; higher N = more obscure - "niche" or "deep cut").
- To find them: run a scoped scryfall_search_cards on a specific engine-map need (e.g. a payoff
  for the exact thing the deck does), pass a LARGER limit (~15-20) so the obscure tail is visible,
  and pick from the "niche"/"deep cut" end - a card that fits, not obscure for its own sake.
- Verify its text like any recommendation, and LABEL it as a deep cut ("rarely played, but...")
  with the specific reason it works HERE. One or two real gems beats a pile of staples.
- This complements the rule above: instead of re-pitching famous cards, dig for the hidden ones.

# TOOL ECONOMY (spend calls where they earn their keep)
Use as many or as few tools as the request needs - a rules question might be one call, a full
review a handful. Don't pad, don't re-fetch what you already have, and BATCH (verify a shortlist
in one scryfall_get_decklist_details call, not a dozen scryfall_get_card calls). Reason from the
details you already fetched rather than re-verifying cards you've read. At most one
deckbuilding_search call. The one thing you must never skimp on: don't name a RECOMMENDED add you
haven't fetched this turn. Once you have what the answer needs, stop calling tools and write it.

# DECKBUILDING FRAMEWORK (the backbone - apply, don't recite)
A functional 99-card Commander deck is roughly:
- ~36-38 lands (adjust for average mana value and ramp count)
- ~10-12 ramp / mana rocks / dorks
- ~10-12 card advantage / draw engines
- ~8-12 targeted interaction (spot removal, counters, protection)
- ~3-5 board wipes
- the REMAINDER (~30-35) is your theme, synergy pieces, and win conditions
These are starting ratios, not laws - a heavy-ramp deck wants fewer lands, a spellslinger
deck wants more draw, a low-curve aggro deck wants fewer wipes. Reason from the deck's
actual gameplan.

Think in "units of value": every non-land card should advance the gameplan, and lands,
ramp, and draw exist to reliably deploy those units. Prize CONSISTENCY over ceiling -
a deck that does its thing every game beats one with a higher top end it rarely assembles.
Favor REDUNDANCY (multiple cards that do the same key job) so the deck doesn't hinge on
drawing one piece. When you suggest an add, also consider what it CUTS - decks are
zero-sum at 100 cards. Recommend cuts, not just adds.

# HOW TO USE YOUR TOOLS
- deckbuilding_search: search the theory corpus (transcripts of respected Commander
  deckbuilders) for their SPECIFIC takes, worked examples, and nuance. You already carry
  their distilled thinking in the playbook below, so use this tool for DEPTH and CITATION:
  for a substantive strategy or theory question ("how should I approach X", "is Y worth
  running", "why does my deck do Z but not win"), CONSULT IT ONCE to pull a relevant
  creator take, and attribute that creator inline when it sharpens your answer. Reach for
  it whenever a concrete creator take, example, or number would make the advice better -
  not for trivial lookups the playbook already answers. (Respect the no-false-citation
  rule: only name a source you actually retrieved.)
- spellbook_find_combos_in_decklist: analyze a pasted/linked decklist for combos.
- spellbook_estimate_bracket: gauge a deck's power level / bracket.
- spellbook_search_combos / spellbook_find_combos_for_cards: find combo lines to add.
- scryfall_search_cards: find candidate cards by criteria (use id: for color identity!).
  e.g. id:mardu t:creature o:"whenever" cmc<=3
- scryfall_get_card: verify EXACT oracle text before you rely on how a card works.
- mtg_rules_search / scryfall_get_rulings: confirm interactions when advice hinges on them.
  scryfall_get_rulings returns the card's OFFICIAL rulings, which spell out exactly these
  corner cases - USE IT before you assert how a tricky card behaves.

Always VERIFY card text with scryfall_get_card before making a claim about what a card
does - never trust your memory of oracle text. When recommending cards for a commander,
respect COLOR IDENTITY strictly (use Scryfall id: searches scoped to the commander's colors).

# RULES INTERACTIONS (don't get the interaction wrong)
When a card's value depends on how it INTERACTS with the rules or with the deck - copying,
tokens, the legend rule, replacement/state-based effects, triggers, targeting restrictions,
"may" vs "must", timing/priority - reason it through carefully, and when you're not certain
call scryfall_get_rulings on that card (its official rulings usually cover the exact case) or
mtg_rules_search. Don't sell a payoff whose engine doesn't actually work as described. In
particular, watch for classic ANTI-SYNERGIES the card text alone doesn't warn you about:
- Myriad / token-copy effects (Blade of Selves, Helm of the Host) on a LEGENDARY creature:
  the copies are legendary too, so the legend rule makes you keep ONE and put the rest in the
  graveyard immediately - you get NO extra attackers. (ETB/dies triggers of the tokens DO
  still fire, so it can be intentional in an aristocrats/ETB shell - but it is NOT a "swing at
  every opponent" payoff on a legend. Say which effect the player is actually buying.)
- Copy effects generally only copy printed characteristics (not counters, Auras/Equipment, or
  non-copy buffs). "Enters tapped and attacking" tokens were never declared as attackers, so
  "whenever a creature attacks" abilities don't retrigger off them.
- Life-total / group changes, "each opponent" vs "target opponent" vs "each player", and
  symmetric effects that help the table as much as you - check who it actually affects.
When in doubt about an interaction, verify it rather than guessing - a wrong interaction makes
the whole recommendation worthless and erodes trust.

# STYLE
Be concrete and opinionated, but explain the "why" so the player learns the principle,
not just the pick. Tie every recommendation back to THEIR commander, gameplan, bracket,
and budget.

Sourcing rule (important): NEVER announce that you're about to cite something or "check
what the sources say" - that promise-without-delivery reads as broken. Either attribute a
specific claim inline as you make it - e.g. "(Salubrious Snail, 'EDH Doesn't Have
Archetypes')" - or just make the point plainly with no mention of sourcing. Only name a
source when you ACTUALLY retrieved it via deckbuilding_search and are attributing a specific
idea to it; synthesize/paraphrase in your own words (at most a brief quote), never long
verbatim passages. It is completely fine to answer from your own knowledge with no citation
- just don't claim a citation you aren't giving. Don't narrate your tool use.

Keep it readable, but for a full deck review DEPTH beats brevity - the player wants a
thorough, correct diagnosis over a quick cut/add list, and is fine with the extra time.

# FULL DECK REVIEW OUTPUT (only when they actually want a full review - diagnose first)
This structure is for a genuine full review, NOT for a targeted question or a quick follow-up
(answer those directly and briefly). When you ARE doing a full review, don't jump to a cut/add
list - structure it as:
1. HOW IT WINS & PLAYS OUT - the real gameplan: where the engine actually lives (the commander,
   or the 99 with the commander as a supplemental piece - say which), the best-case line, and what
   typical turns look like. Show you understand the deck.
2. THE ENGINE MAP (the core of the read - this is what interpreting a deck actually means):
   lay out, holistically, what the deck WANTS TO DO, what ENABLES that, and what BENEFITS from
   it. Concretely, assign the CONTEXTUAL roles by reasoning over the whole list:
   - ENABLERS - cards that let the deck do its thing (make the tokens, fill the yard, go wide).
   - PAYOFFS - cards that turn that activity into advantage or a win (the other side of the coin).
   - FORCE MULTIPLIERS - doublers / turbochargers that amplify the ENABLERS themselves (not the
     output): Doubling Season, extra combats.
   - ENGINE - the 2-3 card combination(s) that generate repeatable value together (name the
     pieces; this is combinatorial, so read it from the whole list).
   - THREATS / FINISHERS - the clocks and the cards you expect to close the game.
   The CONCRETE roles (Ramp/Draw/Target & Mass Interaction/Recursion/Tutor/Protection/Stax) are
   already counted for you in the deck details (from Scryfall's tags) - trust those numbers and
   spend your reasoning on the contextual map above, which is the part only whole-deck judgment
   can do. A card can be both (a Ramp card that's also your key Enabler).
3. STRUCTURAL READ - use the composition data (authoritative land count, the CONCRETE ROLE
   counts, creature/legendary density) to say what's healthy vs. stretched. Cite real numbers,
   not vibes - e.g. "6 pieces of targeted interaction but only 1 board wipe," using the counts given.
4. FAILURE MODES - the specific games where it stumbles and WHY - that's the thing to fix.
5. RECOMMENDATIONS - cuts and adds, each tied to a point above with the reasoning (why this
   card, why it beats what it replaces, what it does for the plan). These are the CONCLUSION of
   the analysis, not the whole thing. Before cutting a card, consider its synergy with the
   deck's density (a legendary-matters rock in a legendary-heavy deck is not "redundant ramp").
Go deep and be specific; a rich, correct read is the goal, not speed.
- Depth belongs in the ANALYSIS, not in bookkeeping. Keep mechanical accounting - legality
  confirmations, "you have one open slot," card-count math - to a single crisp line, then move
  on. Never walk through step-by-step count arithmetic (command zone vs. library totals, "98 to
  98") in prose; state the conclusion ("this swap is color-legal and leaves you one slot to
  fill") and spend your words on the actual read and the pick. Don't open with paragraphs of
  throat-clearing before the substance."""


# Append the distilled creator "how to think" playbook (if present). It's kept in
# a file so it can be re-distilled/edited without touching code. These are lenses
# to reason WITH, not a procedure to follow.
_PLAYBOOK_PATH = Path(__file__).parent / "playbooks" / "unified.md"
if _PLAYBOOK_PATH.exists():
    SYSTEM_PROMPT += (
        "\n\n# HOW TO THINK (distilled from expert Commander deckbuilders)\n"
        "Reason WITH the playbook below. These are LENSES to weigh with judgment, never a "
        "checklist - diagnose what THIS deck actually needs and go deep on the few lenses that "
        "matter, ignoring the rest. Attribute a distinctive idea to its creator when it helps.\n\n"
        + _PLAYBOOK_PATH.read_text(encoding="utf-8")
    )

# Re-assert the machine-critical formatting rules AFTER the playbook so recency keeps
# them salient (the long playbook otherwise drowns out the top-of-prompt rules).
SYSTEM_PROMPT += (
    "\n\n# NON-NEGOTIABLE FORMATTING (overrides any habit from the playbook above)\n"
    "1. Wrap EVERY specific Magic card name in [[double brackets]] - e.g. [[Sneak Attack]], "
    "[[Ram Through]], [[Ghalta, Stampede Tyrant]]. NEVER use bold or plain text for a card "
    "name; the [[ ]] markers are what render it on the card canvas, so bolding a card instead "
    "silently breaks the UI.\n"
    "2. Emit the `%%IDENTITY:XX%%` marker as the very first line of your first response about a "
    "deck.\n"
    "3. Do NOT narrate your tool use or 'think out loud' about refining searches - just make "
    "the calls silently and write the finished answer."
)

# =============================================================================
# THEORY CORPUS (deckbuilding_search tool) - lazy loaded, mirrors rules pattern
# =============================================================================

_theory_collection = None
_theory_loading = False


def _load_theory_collection_sync():
    """Blocking load of the theory collection; run in a thread pool."""
    global _theory_collection
    if not THEORY_DB_PATH.exists():
        return None
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=str(THEORY_DB_PATH))
        embedding_func = embedding_functions.DefaultEmbeddingFunction()  # ONNX all-MiniLM-L6-v2: same vectors, no PyTorch
        _theory_collection = client.get_collection(
            name=THEORY_COLLECTION,
            embedding_function=embedding_func,
        )
        return _theory_collection
    except Exception as e:
        print(f"Warning: could not load theory collection: {e}", flush=True)
        return None


async def get_theory_collection_async():
    """Async loader for the theory collection (heavy load runs off the event loop)."""
    global _theory_collection, _theory_loading
    if _theory_collection is not None:
        return _theory_collection
    if not THEORY_DB_PATH.exists():
        return None
    if _theory_loading:
        while _theory_loading and _theory_collection is None:
            await asyncio.sleep(0.5)
        return _theory_collection
    _theory_loading = True
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_theory_collection_sync)
        return _theory_collection
    finally:
        _theory_loading = False


async def deckbuilding_search(query: str, num_results: int = 6) -> str:
    """
    Search the deckbuilding theory corpus (video transcripts + articles) for
    relevant principles and archetype guidance. Returns chunks with source
    attribution so the advisor can cite them.
    """
    collection = await get_theory_collection_async()
    if collection is None:
        return (
            "The deckbuilding theory corpus hasn't been built yet (run "
            "theory_ingestion.py to populate it). For now, rely on the deckbuilding "
            "framework in your instructions plus the Scryfall and Spellbook tools."
        )

    num_results = max(1, min(int(num_results or 6), 12))

    def _query():
        return collection.query(query_texts=[query], n_results=num_results)

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _query)
    except Exception as e:
        return f"Error searching theory corpus: {e}"

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return f"No relevant deckbuilding theory found for: {query}"

    lines = [f"**Deckbuilding theory for:** {query}\n"]
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        title = meta.get("title", "Unknown source")
        author = meta.get("author", "")
        url = meta.get("url", "")
        src_type = meta.get("type", "")
        header = f"### {title}"
        if author:
            header += f" - {author}"
        if src_type:
            header += f" ({src_type})"
        lines.append(header)
        lines.append(doc)
        if url:
            lines.append(f"Source: {url}")
        lines.append("")
    return "\n".join(lines)


# Extend the bot's tools with the deckbuilding search tool.
DECKBUILDING_SEARCH_SCHEMA = {
    "name": "deckbuilding_search",
    "description": (
        "Search the Commander deckbuilding THEORY corpus (transcripts of deckbuilding "
        "videos and articles) for principles, archetype guides, ratios, and nuanced "
        "strategic takes. Use this to ground deck advice and cite sources. This is for "
        "STRATEGY/THEORY, not card data (use Scryfall) or rules (use mtg_rules_search)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language deckbuilding question or topic, "
                               "e.g. 'how many lands for a low-curve aggro deck' or "
                               "'aristocrats sacrifice engine redundancy'.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of theory chunks to return (1-12).",
                "default": 6,
            },
        },
        "required": ["query"],
    },
}

TOOLS = list(BASE_TOOLS) + [DECKBUILDING_SEARCH_SCHEMA]
TOOL_FUNCTIONS = {**BASE_TOOL_FUNCTIONS, "deckbuilding_search": deckbuilding_search}

# =============================================================================
# AGENT STREAMING LOOP
# =============================================================================


# Cloudflare drops a proxied response that sends nothing for ~100 s, and the final answer is
# buffered for the color-identity check - so a long review looked like a "connection error".
# An SSE comment every few seconds keeps the tunnel open; the page ignores comment frames.
HEARTBEAT_SECS = 15

# Stream the answer to the browser as it's written (first words minutes sooner on a full
# review). The off-color guardrail then runs AFTER: an off-identity pick gets struck out via a
# `replace` event + warning instead of a silent regenerate. ADVISOR_STREAM=0 in .env restores
# the buffered validate-then-show behavior.
STREAM_ANSWERS = os.getenv("ADVISOR_STREAM", "1") != "0"
STREAM_FLUSH_CHARS = 80
STREAM_FLUSH_SECS = 0.25
HEARTBEAT = ": keepalive\n\n"


async def _with_heartbeat(source):
    """Yield items from an async iterator, plus None whenever HEARTBEAT_SECS pass with nothing
    - a model thinking silently can go minutes without a single stream event."""
    it = source.__aiter__()
    nxt = asyncio.ensure_future(it.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({nxt}, timeout=HEARTBEAT_SECS)
            if not done:
                yield None
                continue
            try:
                item = nxt.result()
            except StopAsyncIteration:
                return
            yield item
            nxt = asyncio.ensure_future(it.__anext__())
    finally:
        if not nxt.done():
            nxt.cancel()


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_CARD_RE = re.compile(r"\[\[([^\]]+)\]\]")
_IDENT_RE = re.compile(r"%%IDENTITY:([WUBRGC]+)%%", re.I)


async def _check_off_color(answer_text: str, identity_override: str | None = None):
    """
    Deterministic color-identity guardrail. Uses identity_override when given
    (captured authoritatively from the model's commander_identity tool args),
    else falls back to the declared %%IDENTITY:XX%% marker. Verifies every
    [[card]] against Scryfall's color_identity. Returns a list of off-identity
    cards (illegal in this deck), or None if no identity is known at all.
    """
    if identity_override:
        allowed = set(identity_override.upper().replace("C", ""))
    else:
        m = _IDENT_RE.search(answer_text)
        if not m:
            return None
        allowed = set(m.group(1).upper().replace("C", ""))
    names = list(dict.fromkeys(_CARD_RE.findall(answer_text)))  # unique, in order
    if not names:
        return []
    # one batched lookup for the whole answer, shared with the card panel's cache
    await _batch_resolve(names)
    off = []
    for name in names:
        slim = CARD_IMG_CACHE.get(name.lower())
        if not slim:
            continue
        ci = set(slim.get("color_identity") or [])
        if not ci.issubset(allowed):
            off.append({"name": name, "identity": "".join(sorted(ci)) or "C"})
    return off


def _strip_off_color_lines(text: str, off_names: list) -> str:
    """
    Last-resort deterministic backstop: drop any line that recommends an
    off-identity card (matched by its [[Name]] marker), so an illegal card can
    never appear in the shown answer even if the model refuses to comply.
    """
    tokens = [f"[[{n}]]".lower() for n in off_names]
    kept = [ln for ln in text.split("\n") if not any(t in ln.lower() for t in tokens)]
    return "\n".join(kept)


def _short_input(tool_input: dict) -> str:
    """A compact, human-readable summary of a tool call's arguments."""
    try:
        s = json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        s = str(tool_input)
    return s if len(s) <= 160 else s[:157] + "..."


MAX_IDENTITY_RETRIES = 2  # regenerate the answer this many times if off-color cards slip in


# The Commander ban list and the Game Changers list, fetched from Scryfall at startup and once a
# day after. Appended to the static system prompt (changes ~never, so it stays cached).
FORMAT_LISTS_TEXT = ""


async def _load_format_lists() -> None:
    global FORMAT_LISTS_TEXT

    async def names(q: str) -> list[str] | None:
        out, page = [], 1
        while page <= 3:
            data = await _scryfall_get("/cards/search", {"q": q, "order": "name", "page": page})
            if data is None:
                return None
            out += [c["name"] for c in data.get("data") or []]
            if not data.get("has_more"):
                return out
            page += 1
        return out

    banned, gcs = await names("banned:commander"), await names("is:gamechanger")
    if not banned or not gcs:
        print("Format lists: Scryfall fetch failed - keeping the previous lists.", flush=True)
        return
    FORMAT_LISTS_TEXT = (
        "\n\n# FORMAT LISTS (current, from Scryfall)\n"
        f"BANNED IN COMMANDER ({len(banned)}) - never recommend, propose or build with these: "
        + "; ".join(banned) + ".\n"
        f"GAME CHANGERS ({len(gcs)}, the official list that sets bracket caps): " + "; ".join(gcs) + ".\n"
        "Use these lists silently: simply don't suggest banned cards, and count Game Changers against the "
        "bracket. Don't narrate checks (\"none of these are banned\", \"checking game-changer status\").")
    print(f"Format lists loaded: {len(banned)} banned, {len(gcs)} game changers.", flush=True)


def _cached_system(memory_block: str = "", deck_block: str = ""):
    """System prompt + optional SESSION MEMORY + CURRENT DECK, each its own cache breakpoint,
    ordered most- to least-stable (static prompt -> memory, which changes every ~N turns ->
    deck, which changes on edits). An edit re-caches only what follows it."""
    blocks = [{"type": "text", "text": SYSTEM_PROMPT + FORMAT_LISTS_TEXT,
               "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    for text in (memory_block, deck_block):
        if text:
            blocks.append({"type": "text", "text": text, "cache_control": {"type": "ephemeral"}})
    return blocks


# --- Live deck context (workbench phase 3) -----------------------------------
DECK_REF = "@deck"  # the model passes this as decklist_text; the server substitutes the list
DECKLIST_TOOLS = {"scryfall_get_decklist_details", "spellbook_find_combos_in_decklist",
                  "spellbook_estimate_bracket"}
_DECK_DETAILS_CACHE: dict[str, str] = {}  # deck text -> scryfall_get_decklist_details output

# --- Interactive tools (workbench phase 3): proposals + questions the player decides on ---
# These don't fetch data; they put UI in front of the player. The server validates each
# proposal (legal, in color identity, not a must-keep cut...) before it is shown.
UI_TOOLS = [
    {
        "name": "propose_changes",
        "description": (
            "Show the player concrete deck edits as accept/reject cards next to their live deck. Use this "
            "whenever you recommend specific cuts, adds or swaps for the deck in the editor (instead of only "
            "listing them in prose). Each change is an add, a cut, or a swap (both). Give a short reason per "
            "change. The server rejects illegal/off-color adds, cuts of cards not in the deck, and cuts of the "
            "player's must-keep cards - you'll be told which. Accepting applies the change to their deck."),
        "input_schema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array", "minItems": 1, "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "add": {"type": "string", "description": "Exact card name to add (omit for a pure cut)."},
                            "cut": {"type": "string", "description": "Exact name of a card in the deck to cut (omit for a pure add)."},
                            "qty": {"type": "integer", "minimum": 1, "maximum": 40,
                                    "description": "Copies to add - only for basic lands (and other any-number cards)."},
                            "reason": {"type": "string", "description": "One or two sentences: why, in this deck."},
                        },
                        "required": ["reason"],
                    },
                },
            },
            "required": ["changes"],
        },
    },
    {
        "name": "ask_player",
        "description": (
            "Ask the player a question with 2-6 short answer buttons when you need their decision to go on "
            "(direction, trade-offs, which of two packages, whether to keep a pet card). After calling this, "
            "end your turn with at most a sentence or two - their tap on an answer arrives as their next message."),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
                "allow_multiple": {"type": "boolean", "description": "Let them pick several options."},
            },
            "required": ["question", "options"],
        },
    },
]
UI_TOOLS.append({
    "name": "start_deck",
    "description": (
        "The player is building a NEW deck and there is no deck in their editor yet: once the commander is "
        "settled, call this to offer creating the deck in the editor. It shows a 'Start this deck' card with "
        "the commander; when they create it, the deck (just the commander) appears as CURRENT DECK on their "
        "next message and you fill it in with propose_changes adds. End your turn after calling it."),
    "input_schema": {
        "type": "object",
        "properties": {
            "commander": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2,
                          "description": "Exact commander name(s) - two only for partners / a Background."},
            "bracket": {"type": "integer", "minimum": 1, "maximum": 5,
                        "description": "Target bracket, if the player has said."},
            "plan": {"type": "string", "description": "One or two sentences: the gameplan they've described."},
        },
        "required": ["commander"],
    },
})
UI_TOOL_NAMES = {t["name"] for t in UI_TOOLS}


def _deck_card_index(deck: dict | None) -> dict[str, str]:
    """lower(name) and lower(front face) -> canonical deck card name."""
    idx = {}
    for c in (deck or {}).get("cards") or []:
        n = c.get("name") or ""
        idx[n.lower()] = n
        idx[_collection_name(n).lower()] = n
    return idx


async def _validate_proposals(changes: list, deck: dict | None, identity: str | None) -> tuple[list, list]:
    """(shown, refused). shown items carry display data for the page; refused carry reasons
    for the model. Cards to add are resolved (fuzzy) and must be commander-legal, in color
    identity and not already in the deck; cuts must be in the deck and not must-keep."""
    idx = _deck_card_index(deck)
    keep = {k.lower() for k in ((deck or {}).get("intake") or {}).get("keep") or []}
    allowed = set(identity or "") - {"C"} if identity else None
    shown, refused = [], []
    for ch in (changes or [])[:10]:
        add_name, cut_name = (ch.get("add") or "").strip(), (ch.get("cut") or "").strip()
        reason = " ".join(str(ch.get("reason") or "").split())[:400]
        if not add_name and not cut_name:
            continue
        item = {"id": uuid.uuid4().hex[:8], "reason": reason, "status": "pending"}
        problem = None
        if cut_name:
            canon = idx.get(cut_name.lower()) or idx.get(_collection_name(cut_name).lower())
            if not canon:
                problem = f"cut '{cut_name}': not in the deck"
            elif canon.lower() in keep:
                problem = f"cut '{canon}': it's on the player's MUST KEEP list"
            else:
                item["cut"] = canon
        if not problem and add_name:
            card = await _resolve_full_card(add_name)
            if not card:
                problem = f"add '{add_name}': no such card"
            else:
                slim = _slim_deck_card(card, 1)
                ci = set(card.get("color_identity") or [])
                if (card.get("legalities") or {}).get("commander") != "legal":
                    problem = f"add '{slim['name']}': not legal in Commander"
                elif allowed is not None and not ci <= allowed:
                    problem = f"add '{slim['name']}': color identity {''.join(sorted(ci))} is outside {identity}"
                elif slim["name"].lower() in idx and not slim.get("any_qty"):
                    problem = f"add '{slim['name']}': already in the deck"
                else:
                    item["add"] = slim["name"]
                    try:
                        q = int(ch.get("qty") or 1)
                    except (TypeError, ValueError):
                        q = 1
                    if q > 1 and slim.get("any_qty"):
                        item["qty"] = min(q, 40)
                    item["addCard"] = {k: slim.get(k) for k in ("name", "image", "image_back", "type_line", "mana_cost",
                                                                "color_identity", "game_changer", "scryfall_uri")}
        if problem:
            refused.append(problem)
        else:
            shown.append(item)
    return shown, refused


async def _start_deck_offer(inp: dict, deck: dict | None) -> tuple[str, dict | None]:
    """(tool result for the model, start_deck event for the page or None). Commanders are
    resolved and must be able to lead a deck (Scryfall is:commander / Background)."""
    if isinstance(deck, dict) and deck.get("cards"):
        return ("A deck is already loaded in the editor - build on it with propose_changes instead.", None)
    names = [" ".join(str(n).split())[:120] for n in (inp.get("commander") or []) if str(n).strip()][:2]
    if not names:
        return ("Give the commander's name.", None)
    cards, problems = [], []
    for n in names:
        full = await _resolve_full_card(n)
        if not full:
            problems.append(f"'{n}': no such card")
            continue
        if (full.get("legalities") or {}).get("commander") not in ("legal", None):
            problems.append(f"'{full.get('name')}': not legal in Commander")
            continue
        cards.append(_slim_deck_card(full, 1))
    if cards:
        await _apply_commander_eligibility(cards)
        problems += [f"'{c['name']}': can't be a commander" for c in cards if not c.get("can_command")]
        cards = [c for c in cards if c.get("can_command")]
    if problems or not cards:
        return ("NOT shown: " + "; ".join(problems or ["no commander"]) + ". Fix the commander and try again.", None)
    try:
        bracket = int(inp.get("bracket")) if inp.get("bracket") is not None else None
    except (TypeError, ValueError):
        bracket = None
    offer = {"id": uuid.uuid4().hex[:8], "commander": cards,
             "bracket": bracket if bracket in (1, 2, 3, 4, 5) else None,
             "plan": " ".join(str(inp.get("plan") or "").split())[:400], "status": "pending"}
    return ("The 'Start this deck' card is on screen. End your turn now (a sentence or two at most); "
            "once they create the deck it arrives as CURRENT DECK and you fill it with propose_changes.", offer)


def _deck_to_text(deck: dict) -> str:
    """Deterministic '1 Name' list (commander(s) first, then A-Z) - stable bytes keep the
    deck block cacheable across turns."""
    cmdrs = {n.lower() for n in deck.get("commander") or []}
    cards = [c for c in deck.get("cards") or [] if c.get("name")]
    head = sorted((c for c in cards if c["name"].lower() in cmdrs), key=lambda c: c["name"])
    rest = sorted((c for c in cards if c["name"].lower() not in cmdrs), key=lambda c: c["name"])
    return "\n".join(f"{int(c.get('qty') or 1)} {c['name']}" for c in head + rest)


def _deck_identity(deck: dict) -> str | None:
    cmdrs = {n.lower() for n in deck.get("commander") or []}
    if not cmdrs:
        return None
    letters = set()
    for c in deck.get("cards") or []:
        if (c.get("name") or "").lower() in cmdrs:
            letters.update(c.get("color_identity") or [])
    return "".join(x for x in "WUBRG" if x in letters) or "C"


def _brief_lines(intake) -> list[str]:
    """The player's deck-brief answers (intake questionnaire) as PLAYER BRIEF lines."""
    if not isinstance(intake, dict):
        return []
    clip = lambda v, n=400: " ".join(str(v).split())[:n]
    items = lambda v: ", ".join(clip(x, 80) for x in (v or [])[:40] if x)
    rows = [("Gameplan / how they want to win", clip(intake.get("plan") or "")),
            ("Play styles", items(intake.get("styles"))),
            ("MUST KEEP (never suggest cutting)", items(intake.get("keep"))),
            ("AVOID (don't recommend)", "; ".join(x for x in (items(intake.get("avoid")),
                                                              clip(intake.get("avoidNote") or "")) if x)),
            ("Budget", clip(intake.get("budget") or "", 120)),
            ("Playgroup / other notes", clip(intake.get("notes") or ""))]
    rows = [f"- {k}: {v}" for k, v in rows if v]
    if not rows:
        return []
    return ["", "PLAYER BRIEF (their answers to the deck questionnaire - these are the player's stated goals "
            "and limits: build your read and every recommendation around them):"] + rows


# --- Guided build from scratch: section targets + progress ----------------------------------
# Section targets follow the player's framework (Rebel Lily): ~38 lands and a 24-slot ramp+draw
# budget split by commander mana value, plus interaction and protection; the rest is the
# gameplan (engine, payoffs, threats, finishers). The page computes the same numbers.
BUILD_SECTIONS = [("gameplan", "Gameplan (engine, payoffs, threats, finishers)"), ("ramp", "Ramp"),
                  ("draw", "Card draw"), ("interaction", "Interaction (spot + mass)"),
                  ("protection", "Protection"), ("lands", "Lands")]


def _default_build_targets(deck: dict) -> dict:
    cmdrs = {n.lower() for n in deck.get("commander") or []}
    mvs = [float(c.get("cmc") or 0) for c in deck.get("cards") or [] if (c.get("name") or "").lower() in cmdrs]
    mv = max(mvs) if mvs else 4
    ramp, draw = (14, 10) if mv >= 5 else (12, 12) if mv >= 4 else (10, 14) if mv >= 3 else (8, 16)
    t = {"lands": 38, "ramp": ramp, "draw": draw, "interaction": 10, "protection": 3}
    t["gameplan"] = max(0, 100 - max(1, len(cmdrs)) - sum(t.values()))
    return t


_SECTION_OF = {"Lands": "lands", "Ramp": "ramp", "Draw": "draw", "Target Interaction": "interaction",
               "Mass Interaction": "interaction", "Protection": "protection"}


def _build_counts(deck: dict) -> dict:
    """Each non-commander card counts once, by the group it sits in on the player's canvas: their
    role if they set one, else Lands for lands, else its first concrete role, else Other. Groups
    outside the five named sections are gameplan. Mirrors cardGroup/buildCounts in the page."""
    cmdrs = {n.lower() for n in deck.get("commander") or []}
    n = dict.fromkeys(("gameplan", "ramp", "draw", "interaction", "protection", "lands"), 0)
    for c in deck.get("cards") or []:
        if (c.get("name") or "").lower() in cmdrs:
            continue
        group = c.get("role") or ("Lands" if c.get("is_land") else ((c.get("roles") or ["Other"])[0]))
        n[_SECTION_OF.get(group, "gameplan")] += int(c.get("qty") or 1)
    return n


def _build_lines(deck: dict) -> list[str]:
    build = deck.get("build")
    if not isinstance(build, dict) or build.get("done"):
        return []
    own = {k: int(v) for k, v in (build.get("targets") or {}).items()
           if k in dict(BUILD_SECTIONS) and str(v).isdigit()}
    targets = {**_default_build_targets(deck), **own}
    if "gameplan" not in own:  # the gameplan takes whatever the player's other targets leave of 100
        targets["gameplan"] = max(0, 100 - max(1, len(deck.get("commander") or []))
                                  - sum(v for k, v in targets.items() if k != "gameplan"))
    have = _build_counts(deck)
    total = sum(int(c.get("qty") or 1) for c in deck.get("cards") or [])
    rows, short = [], []
    for key, label in BUILD_SECTIONS:
        gap = targets[key] - have[key]
        rows.append(f"- {label}: {have[key]}/{targets[key]}" + (f" (need {gap})" if gap > 0 else " (done)"))
        if gap > 0:
            short.append(key)
    nxt = next((k for k in short if k != "lands"), "lands" if short else None)
    return ["", "BUILD MODE - guided build from scratch (the player is building this deck with you; "
            f"{total}/100 cards so far). Section targets (have/target):", *rows,
            f"NEXT SECTION TO FILL: {dict(BUILD_SECTIONS)[nxt] if nxt else 'none - all targets met; '}"
            + ("" if nxt else "review the whole list, then suggest the player finish the build."),
            "Work one section per batch (see BUILD MODE in your instructions)."]


# --- Ikoria companions + curve/type stats ------------------------------------------------------
import companions as companions_mod


async def _full_cards(entries: list) -> list:
    """[(name, qty)] -> [(full Scryfall card, qty)] via the shared batch cache (unresolved dropped)."""
    await _batch_resolve([n for n, _ in entries])
    out = []
    for n, q in entries:
        full = FULL_CARD_CACHE.get(n.lower())
        if full is None:
            slim = CARD_IMG_CACHE.get(n.lower())
            full = FULL_CARD_CACHE.get((slim or {}).get("name", "").lower()) if slim else None
        if full:
            out.append((full, q))
    return out


async def _companion_report(cards: list, commanders: list, companion: str | None) -> dict:
    """{chosen: check | None, identity_ok, options: [{name, condition, ok, n_bad, bad}]} for the
    deck (commander included). Options = the Ikoria companions inside the commander's colors."""
    full = await _full_cards(cards)
    cmdr = {n.lower() for n in commanders}
    ident = set()
    for c, _ in full:
        if c["name"].lower() in cmdr:
            ident |= set(c.get("color_identity") or [])
    comp_full = await _full_cards([(n, 1) for n in companions_mod.CONDITIONS])
    by_name = {c["name"]: c for c, _ in comp_full}
    options = []
    if cmdr:
        for name in companions_mod.CONDITIONS:
            card = by_name.get(name)
            if not card or not set(card.get("color_identity") or []) <= ident:
                continue
            r = companions_mod.check(name, full, cmdr)
            # a Yorion that's merely short of 120 cards is a fixable goal, not a dead option
            fixable = name.startswith("Yorion") and r["note"].startswith("needs")
            options.append({"name": name, "condition": r["condition"], "ok": r["ok"],
                            "note": "" if fixable else r["note"], "short": r["note"] if fixable else "",
                            "n_bad": len(r["violations"]), "bad": r["violations"][:12],
                            "image": (card.get("image_uris") or {}).get("normal")})
    chosen = None
    identity_ok = True
    if companion:
        chosen = companions_mod.check(companion, full, cmdr)
        card = by_name.get(chosen["name"])
        identity_ok = bool(card) and (not cmdr or set(card.get("color_identity") or []) <= ident)
    return {"chosen": chosen, "identity_ok": identity_ok, "options": options}


async def _companion_and_stats_lines(deck: dict) -> list[str]:
    entries = [(c["name"], int(c.get("qty") or 1)) for c in deck.get("cards") or [] if c.get("name")]
    cmdrs = deck.get("commander") or []
    companion = (deck.get("companion") or {}).get("name") if isinstance(deck.get("companion"), dict) \
        else deck.get("companion")
    lines = []
    try:
        full = await _full_cards(entries)
        st = companions_mod.deck_stats(full, {n.lower() for n in cmdrs})
        curve = " ".join(f"{'7+' if b == 7 else b}:{v['creature'] + v['other']}" for b, v in sorted(st["curve"].items()))
        types = ", ".join(f"{t} {n}" for t, n in sorted(st["types"].items(), key=lambda x: -x[1]))
        lines += ["", f"MANA CURVE (nonland cards by mana value, commander excluded): {curve}",
                  f"CARD TYPES (one main type per card, commander excluded): {types}"]
        rep = await _companion_report(entries, cmdrs, companion)
    except Exception:
        return lines
    ch = rep["chosen"]
    if ch:
        if ch["note"]:
            status = ch["note"]
        elif not rep["identity_ok"]:
            status = "ILLEGAL - outside the commander's color identity"
        elif ch["ok"]:
            status = "condition MET"
        else:
            status = (f"condition NOT MET - {len(ch['violations'])} card(s) break it: "
                      + ", ".join(ch["violations"][:15]))
        lines += ["", f"COMPANION (outside the 100): {ch['name']} - {ch['condition']} - {status}"]
    fits = [o["name"] for o in rep["options"] if o["ok"]]
    close = [f"{o['name']} ({o['n_bad']} break it)" for o in rep["options"]
             if not o["ok"] and not o["note"] and 0 < o["n_bad"] <= 8]
    if (fits or close) and not (ch and ch["ok"]):
        lines.append("Companion options in these colors: "
                     + "; ".join([f"{n} - deck already qualifies" for n in fits] + close))
    return lines


async def _deck_context(deck) -> tuple[str, str, str | None]:
    """(CURRENT DECK block, canonical deck text, commander identity) - empty when no deck."""
    if not isinstance(deck, dict) or not deck.get("cards"):
        return "", "", None
    deck_text = _deck_to_text(deck)
    if deck_text not in _DECK_DETAILS_CACHE:
        # Card details + the combo check + Spellbook's bracket read are deterministic lookups:
        # run them together here (once per deck version) instead of spending model turns on them.
        details, combos, bracket_read = await asyncio.gather(
            TOOL_FUNCTIONS["scryfall_get_decklist_details"](decklist_text=deck_text),
            TOOL_FUNCTIONS["spellbook_find_combos_in_decklist"](decklist_text=deck_text, limit=12),
            TOOL_FUNCTIONS["spellbook_estimate_bracket"](decklist_text=deck_text))
        failed = any(str(x).startswith(("Error", "{'error'")) for x in (details, combos, bracket_read))
        if str(details).startswith("Error"):
            details = "(Card details unavailable right now - use scryfall_get_decklist_details with @deck.)"
        analysis = ["", "PRECOMPUTED FOR THIS DECK VERSION (already run - don't call these tools for the deck "
                    "itself; re-run them with @deck only if the player asks you to double-check):",
                    "## Combo check (spellbook_find_combos_in_decklist)", str(combos),
                    "## Spellbook power read (spellbook_estimate_bracket)", str(bracket_read)]
        details = str(details) + "\n" + "\n".join(analysis)
        if not failed:
            _DECK_DETAILS_CACHE[deck_text] = details
    else:
        details = _DECK_DETAILS_CACHE[deck_text]
    cards = deck.get("cards") or []
    cmdrs = deck.get("commander") or []
    bracket = deck.get("bracket")
    total = sum(int(c.get("qty") or 1) for c in cards)
    lines = [
        "# CURRENT DECK (live from the player's deck editor - authoritative, see instructions)",
        f"Commander: {' + '.join(cmdrs) if cmdrs else 'NOT SET in the editor - infer it from the conversation or ask'}",
        f"Target bracket: {bracket if bracket else 'not set'} | Cards: {total}",
    ]
    overrides = [f"{c['name']} -> {c['role']}" for c in cards if c.get("role")]
    if overrides:
        lines.append("Player-assigned roles (their read of the deck - respect it): " + "; ".join(sorted(overrides)))
    tagged = [f"{c['name']} #{' #'.join(c['tags'])}" for c in cards if c.get("tags")]
    if tagged:
        lines.append("Player tags: " + "; ".join(sorted(tagged)))
    lines += _brief_lines(deck.get("intake"))
    lines += await _companion_and_stats_lines(deck)
    lines += _build_lines(deck)
    lines += ["", "Decklist:", deck_text, "", details]
    return "\n".join(lines), deck_text, _deck_identity(deck)


def _decisions_summary(m: dict) -> str:
    """What this assistant turn put in front of the player and what they decided - the
    tool calls themselves aren't in the history, so the model would otherwise forget."""
    out = []
    props = [p for p in (m.get("proposals") or []) if isinstance(p, dict)][:12]
    if props:
        mark = {"accepted": "ACCEPTED", "rejected": "REJECTED",
                "skipped": "SKIPPED (they moved on without deciding - it's not in the deck)"}
        rows = []
        for p in props:
            what = " / ".join(x for x in (f"cut {p.get('cut')}" if p.get("cut") else "",
                                          (f"add {p.get('qty')}x {p.get('add')}" if (p.get("qty") or 1) > 1
                                           else f"add {p.get('add')}") if p.get("add") else "") if x)
            why = " ".join(str(p.get("note") or "").split())[:240]
            rows.append(f"{what} -> {mark.get(p.get('status'), 'undecided')}"
                        + (f" (player's reason: \"{why}\")" if why else ""))
        out.append("[Your proposals and the player's decisions: " + "; ".join(rows)
                   + ". Don't re-propose rejected ones, and learn from their reasons - they tell you what "
                   "this player values.]")
    sd = m.get("startDeck")
    if isinstance(sd, dict) and sd.get("commander"):
        names = " + ".join(str(c.get("name") if isinstance(c, dict) else c) for c in sd["commander"])[:200]
        st = {"created": "they CREATED it - it's now their CURRENT DECK",
              "declined": "they declined for now",
              "skipped": "they moved on without creating it"}.get(sd.get("status"), "not decided yet")
        out.append(f"[You offered to start a deck with {names} as commander - {st}.]")
    q = m.get("question")
    if isinstance(q, dict) and q.get("question"):
        out.append(f"[You asked: \"{str(q['question'])[:200]}\" - options: {', '.join(map(str, q.get('options') or []))[:300]}"
                   + (f" - they answered by typing: {str(q.get('answer'))[:200]}]" if q.get("answer") and q.get("typed")
                      else f" - they answered: {str(q.get('answer'))[:200]}]" if q.get("answer")
                      else " - they skipped it]" if q.get("skipped") else "]"))
    return ("\n\n" + "\n".join(out)) if out else ""


def _collapse_pasted_decklist(text: str) -> str:
    """With a live deck present, an old pasted list is stale and competes with it (e.g. a
    paste that still had the maybeboard). Replace the list lines with a pointer."""
    if len(_DECK_LINE.findall(text)) < 15:
        return text
    kept = [ln for ln in text.splitlines() if not _DECK_LINE.match(ln)]
    kept.append("[decklist pasted here - superseded by the CURRENT DECK block]")
    return "\n".join(ln for ln in kept if ln.strip())


# --- Session memory: condense the old part of long chats ---------------------
MEMORY_TRIGGER_CHARS = 100_000  # verbatim transcript size (~25k tokens) that triggers condensing
MEMORY_KEEP_RECENT = 6          # the last N messages always stay verbatim

MEMORY_PROMPT = """You maintain the SESSION MEMORY for a Commander (EDH) deckbuilding chat between a player and an AI advisor. Older messages are about to be removed from the advisor's context, so this memory is ALL the advisor will know about them. Merge the existing memory (if any) with the conversation excerpt into ONE updated memory.

Write these sections (omit a section only if there is truly nothing for it):
## Player & deck goals - commander, gameplan, win conditions, target bracket, budget, playgroup/meta notes
## Constraints & preferences - hard rules the player set (pet cards to keep, cards they don't own, styles they dislike, bracket rules to respect)
## Decisions made - changes the player accepted or made, with the one-line reason
## Rejected suggestions - cards/ideas the player turned down, with their reason (the advisor must not re-suggest these)
## Key findings - concrete analysis results worth keeping: bracket estimate, combos found, land/ramp/draw counts, structural problems identified
## Open threads - questions still unanswered, things the player said they'd come back to

Rules: keep card names EXACT. Keep numbers exact. Only record what the excerpt/memory actually says - never invent. Drop pleasantries, tool chatter, and anything superseded by a later decision. Be dense: short bullets, no prose padding. Output only the memory."""


async def _condense_memory(prev_memory: str, excerpt: list[dict]) -> str:
    transcript = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in excerpt)
    msg = (f"EXISTING MEMORY:\n{prev_memory or '(none yet)'}\n\n"
           f"CONVERSATION EXCERPT TO FOLD IN:\n{transcript}")
    resp = await aclient.messages.create(
        model=REVIEW_MODEL, max_tokens=4000, system=MEMORY_PROMPT,
        messages=[{"role": "user", "content": msg}],
        extra_body={"output_config": {"effort": "medium"}},
    )
    u = resp.usage
    pin, pout = PRICES[REVIEW_MODEL]
    log_activity(f"MEMORY condensed {len(excerpt)} msgs in={u.input_tokens} out={u.output_tokens} "
                 f"~${u.input_tokens * pin + u.output_tokens * pout:.4f}")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _cache_last(messages: list) -> list:
    """Return messages with a cache breakpoint on the last block, so the whole conversation
    prefix (system + tools + prior turns, incl. big deck details) is read from cache on the
    next call instead of re-billed. messages[-1] is always a user message at call time."""
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        nc = [dict(b) if isinstance(b, dict) else b for b in content]
        nc[-1] = {**nc[-1], "cache_control": {"type": "ephemeral"}}
        last["content"] = nc
    else:
        return messages
    out[-1] = last
    return out


async def agent_stream(session_id: str, messages: list, model: str = REVIEW_MODEL, extra: dict = None,
                       system: list | None = None, deck_text: str = "", deck_identity: str | None = None,
                       deck: dict | None = None):
    if extra is None:
        extra = REVIEW_EXTRA
    if system is None:
        system = _cached_system()
    # Cards already IN the deck may be discussed even if off-color ("Beastmaster Ascension
    # is illegal here, cut it") - the guardrail only polices new recommendations.
    in_deck = set()
    for ln in deck_text.splitlines():
        name = ln.split(" ", 1)[-1].strip().lower()
        in_deck.update({name, name.split(" // ")[0]})

    def _not_in_deck(off):
        if not off:
            return off
        off = [c for c in off if (c.get("name") or "").lower() not in in_deck
               and (c.get("name") or "").lower().split(" // ")[0] not in in_deck]
        return off or None
    """
    Drives the Claude tool-use loop. Tool-call turns stream status live; the FINAL
    answer is buffered and validated for color-identity legality BEFORE it is shown,
    and regenerated if any off-identity card slipped in - so an illegal recommendation
    never reaches the user.
    """
    yield _sse("session", {"session_id": session_id})
    identity_retries = 0
    # Authoritative commander identity: from the deck editor when a commander is set there,
    # else captured from the model's tool args below.
    if deck_identity:
        yield _sse("identity", {"identity": deck_identity})
    usage_in = usage_out = usage_cr = usage_cw = 0  # uncached in / out / cache-read / cache-write
    tools_used: list[str] = []
    kept_parts: list[str] = []  # answer text written before a question/proposal (kept, see below)
    t_start = time.monotonic()
    timeline: list[str] = []  # per turn: "model <secs>s/<out tokens>" [+ "<n> tools <secs>s"]

    def _tally(u):
        nonlocal usage_in, usage_out, usage_cr, usage_cw
        if not u:
            return
        usage_in += getattr(u, "input_tokens", 0) or 0
        usage_cr += getattr(u, "cache_read_input_tokens", 0) or 0
        usage_cw += getattr(u, "cache_creation_input_tokens", 0) or 0
        usage_out += getattr(u, "output_tokens", 0) or 0

    try:
        for _ in range(MAX_ITERATIONS + MAX_IDENTITY_RETRIES):
            turn_parts = []
            t_turn = time.monotonic()
            async with aclient.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system,   # cached: prompt (+ memory + live deck)
                tools=TOOLS + UI_TOOLS,
                messages=_cache_last(messages),  # cache the conversation prefix
                extra_body=extra,
            ) as stream:
                pending, last_flush, streamed = "", time.monotonic(), False
                async for event in _with_heartbeat(stream):
                    if event is None:
                        yield HEARTBEAT
                    elif (
                        event.type == "content_block_delta"
                        and getattr(event.delta, "type", None) == "text_delta"
                    ):
                        turn_parts.append(event.delta.text)
                        if STREAM_ANSWERS:  # batch deltas so the page re-renders a few times/sec
                            pending += event.delta.text
                            if len(pending) >= STREAM_FLUSH_CHARS or time.monotonic() - last_flush > STREAM_FLUSH_SECS:
                                yield _sse("text", {"text": pending})
                                pending, last_flush, streamed = "", time.monotonic(), True
                if pending:
                    yield _sse("text", {"text": pending})
                    streamed = True
                final = await stream.get_final_message()

            _tally(getattr(final, "usage", None))
            timeline.append(f"model {time.monotonic() - t_turn:.0f}s/"
                            f"{getattr(getattr(final, 'usage', None), 'output_tokens', 0) or 0}")

            messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason == "tool_use":
                # Suppress the model's mid-process narration on tool-call turns
                # (the "let me refine this search" chatter) - the tool trace shows
                # what's happening; only the final answer turn streams to the user.

                tool_blocks = [b for b in final.content if b.type == "tool_use"]
                turn_text = "".join(turn_parts).strip()
                if any(b.name in UI_TOOL_NAMES for b in tool_blocks):
                    # Text written before a question/proposal IS the answer (analysis, then
                    # "which way?") - keep it, don't treat it as tool chatter.
                    if turn_text:
                        kept_parts.append(turn_text)
                        if streamed:
                            yield _sse("text", {"text": "\n\n"})  # separates it from what follows
                elif streamed:  # "let me look that up" before a data lookup - take it back,
                    # but only back to the answer text already kept from earlier turns
                    keep = "\n\n".join(kept_parts)
                    yield _sse("reset", {"keep": keep + "\n\n" if keep else ""})
                tools_used.extend(b.name for b in tool_blocks)
                for block in tool_blocks:
                    # Capture the commander identity the model passes to its tools -
                    # authoritative for the color-identity guardrail (doesn't depend on
                    # the %%IDENTITY%% marker, which the model sometimes forgets).
                    ci = (block.input or {}).get("commander_identity")
                    if ci and not deck_identity:
                        deck_identity = ci
                        yield _sse("identity", {"identity": ci})
                    if block.name not in UI_TOOL_NAMES:
                        yield _sse("status", {"tool": block.name, "input": _short_input(block.input)})

                # Interactive tools put UI in front of the player (and can't run in the
                # parallel pool - they emit events), so handle them first.
                ui_results = {}
                t_ui = time.monotonic()
                for block in tool_blocks:
                    if block.name == "propose_changes":
                        if not deck:
                            ui_results[block.id] = ("No deck is loaded in the player's editor, so proposals can't "
                                                    "be shown. If they're building a new deck, call start_deck "
                                                    "first; otherwise give your suggestions in prose.")
                            continue
                        shown, refused = await _validate_proposals((block.input or {}).get("changes"), deck, deck_identity)
                        if shown:
                            yield _sse("proposals", {"items": shown})
                        desc = lambda it: " / ".join(x for x in (f"cut {it['cut']}" if it.get("cut") else "",
                                                                 f"add {it['add']}" if it.get("add") else "") if x)
                        ui_results[block.id] = (
                            (f"Shown to the player as accept/reject cards: {'; '.join(desc(i) for i in shown)}. "
                             "Don't repeat them as a list - refer to them briefly; the player decides in the panel."
                             if shown else "Nothing was shown.")
                            + (f" REFUSED (not shown): {'; '.join(refused)}. Fix or drop these." if refused else ""))
                        log_activity(f"PROPOSE sid={session_id[:8]} shown={len(shown)} refused={len(refused)}"
                                     + (f" ({'; '.join(refused)})" if refused else ""))
                    elif block.name == "ask_player":
                        inp = block.input or {}
                        opts = [" ".join(str(o).split())[:120] for o in (inp.get("options") or [])][:6]
                        yield _sse("question", {"id": uuid.uuid4().hex[:8], "question": str(inp.get("question") or "")[:500],
                                                "options": opts, "multi": bool(inp.get("allow_multiple"))})
                        ui_results[block.id] = ("The question is on screen with answer buttons. End your turn now "
                                                "(a sentence or two at most) and wait for their answer.")
                    elif block.name == "start_deck":
                        ui_results[block.id], offer = await _start_deck_offer(block.input or {}, deck)
                        if offer:
                            yield _sse("start_deck", offer)
                        log_activity(f"STARTDECK sid={session_id[:8]} "
                                     + (", ".join(c["name"] for c in offer["commander"]) if offer else "refused"))

                async def _run(block):
                    if block.id in ui_results:
                        return ui_results[block.id]
                    func = TOOL_FUNCTIONS.get(block.name)
                    if func is None:
                        return f"Unknown tool: {block.name}"
                    args = dict(block.input or {})
                    # "@deck" (or no list at all) -> the exact editor list, so the model
                    # never retypes 100 lines (and can't drop cards doing it).
                    if block.name in DECKLIST_TOOLS and deck_text and (
                            (args.get("decklist_text") or "").strip() == DECK_REF
                            or not (args.get("decklist_text") or args.get("decklist_url"))):
                        args["decklist_text"] = deck_text
                        args.pop("decklist_url", None)
                    try:
                        return await func(**args)
                    except Exception as e:
                        return f"Error running {block.name}: {e}"

                if time.monotonic() - t_ui >= 1:
                    timeline[-1] += f" -> ui {time.monotonic() - t_ui:.0f}s"
                t_tools = time.monotonic()
                tools_task = asyncio.ensure_future(asyncio.gather(*(_run(b) for b in tool_blocks)))
                while not (await asyncio.wait({tools_task}, timeout=HEARTBEAT_SECS))[0]:
                    yield HEARTBEAT  # slow tools (Spellbook, big Scryfall batches) keep the tunnel open
                results = tools_task.result()
                timeline[-1] += f" -> {len(tool_blocks)} tool{'s' if len(tool_blocks) > 1 else ''} {time.monotonic() - t_tools:.0f}s"
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": b.id, "content": r}
                        for b, r in zip(tool_blocks, results)
                    ],
                })
                continue

            # ---- Final answer turn: validate color identity BEFORE showing it ----
            answer_text = "\n\n".join(kept_parts + ["".join(turn_parts)]).strip()
            streamed = streamed or bool(kept_parts and STREAM_ANSWERS)
            if final.stop_reason == "max_tokens":
                log_activity(f"TRUNCATED sid={session_id[:8]} hit max_tokens={MAX_TOKENS} "
                             f"({len(answer_text)} chars of answer)")
                if not answer_text.strip():
                    yield _sse("error", {"message": "The advisor ran out of room while thinking this through "
                                                    "and didn't get to an answer - try asking again, or narrow "
                                                    "the question."})
                    break
                answer_text += "\n\n_(Answer cut off at the length limit.)_"
                if STREAM_ANSWERS:
                    yield _sse("text", {"text": "\n\n_(Answer cut off at the length limit.)_"})
            t_check = time.monotonic()
            try:
                off = _not_in_deck(await _check_off_color(answer_text, deck_identity))
            except Exception:
                off = None
            if time.monotonic() - t_check >= 1:
                timeline.append(f"color check {time.monotonic() - t_check:.0f}s")

            if STREAM_ANSWERS:
                # Already on screen: strike any off-identity picks after the fact.
                if off:
                    off_names = [c["name"] for c in off]
                    fixed = _strip_off_color_lines(answer_text, off_names) + (
                        f"\n\n_(Removed {len(off)} off-identity card{'s' if len(off) > 1 else ''} "
                        f"from the recommendations: {', '.join(off_names)}.)_")
                    yield _sse("replace", {"text": fixed})
                    yield _sse("warning", {"cards": off})
                    log_activity(f"OFFCOLOR sid={session_id[:8]} stripped after streaming: {', '.join(off_names)}")
                elif not streamed and answer_text:
                    yield _sse("text", {"text": answer_text})
                break

            if off and identity_retries < MAX_IDENTITY_RETRIES:
                identity_retries += 1
                bad = ", ".join(f"{c['name']} ({c['identity']})" for c in off)
                yield _sse("status", {"tool": "color-identity check",
                                      "input": f"off-identity found ({bad}) - revising"})
                messages.append({"role": "user", "content": (
                    "STOP - your previous answer recommended cards that are ILLEGAL in "
                    f"this deck's color identity: {bad}. Rewrite your ENTIRE previous "
                    "answer, removing every one of those cards and replacing each with a "
                    "legal in-identity alternative (verify replacements with "
                    "scryfall_search_cards using commander_identity). Do not mention the "
                    "illegal cards at all. Keep the same %%IDENTITY%% marker and format."
                )})
                continue  # regenerate; do NOT show the bad answer

            # Clean (or out of retries) - now it's safe to show.
            if off:
                # Out of retries but still off-color: physically strip the illegal
                # recommendations, then flag what was removed.
                off_names = [c["name"] for c in off]
                answer_text = _strip_off_color_lines(answer_text, off_names)
                answer_text += (
                    f"\n\n_(Removed {len(off)} off-identity card"
                    f"{'s' if len(off) > 1 else ''} that couldn't be legally replaced: "
                    f"{', '.join(off_names)}.)_"
                )
            if answer_text:
                yield _sse("text", {"text": answer_text})
            if off:
                yield _sse("warning", {"cards": off})
            break
        else:
            # Ran out of tool-turns - don't discard the work. Force one final answer
            # with tools OFF so the model must synthesize from what it already gathered.
            messages.append({"role": "user", "content": (
                "Stop calling tools now and write your best complete answer using everything "
                "you've already gathered above.")})
            final_parts = []
            async with aclient.messages.stream(
                model=model, max_tokens=MAX_TOKENS, system=system,
                messages=_cache_last(messages), extra_body=extra,
            ) as stream:
                pending, last_flush, streamed = "", time.monotonic(), False
                async for event in _with_heartbeat(stream):
                    if event is None:
                        yield HEARTBEAT
                    elif (event.type == "content_block_delta"
                            and getattr(event.delta, "type", None) == "text_delta"):
                        final_parts.append(event.delta.text)
                        if STREAM_ANSWERS:
                            pending += event.delta.text
                            if len(pending) >= STREAM_FLUSH_CHARS or time.monotonic() - last_flush > STREAM_FLUSH_SECS:
                                yield _sse("text", {"text": pending})
                                pending, last_flush, streamed = "", time.monotonic(), True
                if pending:
                    yield _sse("text", {"text": pending})
                    streamed = True
                fmsg = await stream.get_final_message()
            _tally(getattr(fmsg, "usage", None))
            answer_text = "\n\n".join(kept_parts + ["".join(final_parts)]).strip()
            streamed = streamed or bool(kept_parts and STREAM_ANSWERS)
            try:
                off = _not_in_deck(await _check_off_color(answer_text, deck_identity))
            except Exception:
                off = None
            if off:
                off_names = [c["name"] for c in off]
                answer_text = _strip_off_color_lines(answer_text, off_names)
                answer_text += (f"\n\n_(Removed {len(off)} off-identity card"
                                f"{'s' if len(off) > 1 else ''}: {', '.join(off_names)}.)_")
            if streamed:
                if off:
                    yield _sse("replace", {"text": answer_text})
            elif answer_text:
                yield _sse("text", {"text": answer_text})
            if off:
                yield _sse("warning", {"cards": off})

        pin, pout = PRICES.get(model, PRICES[REVIEW_MODEL])
        cost = usage_in * pin + usage_cr * pin * 0.1 + usage_cw * pin * 1.25 + usage_out * pout
        tool_summary = ", ".join(f"{t}x{tools_used.count(t)}" for t in dict.fromkeys(tools_used)) or "none"
        short_model = model.replace("claude-", "")
        log_activity(
            f"ANSWER sid={session_id[:8]} [{short_model}] in={usage_in} cache_r={usage_cr} "
            f"cache_w={usage_cw} out={usage_out} ~${cost:.4f} | {time.monotonic() - t_start:.0f}s "
            f"[{' | '.join(timeline)}] | tools: {tool_summary}"
        )
        yield _sse("done", {})
    except anthropic.APIError as e:
        log_activity(f"ERROR  sid={session_id[:8]} API error: {e}")
        yield _sse("error", {"message": f"API error: {e}"})
    except asyncio.CancelledError:
        # the browser/tunnel went away mid-answer - worth knowing when someone says "it errored"
        log_activity(f"ABORT  sid={session_id[:8]} client disconnected mid-answer "
                     f"(out={usage_out} tokens so far, tools: {', '.join(tools_used) or 'none'})")
        raise
    except Exception as e:
        log_activity(f"ERROR  sid={session_id[:8]} {type(e).__name__}: {e}")
        yield _sse("error", {"message": str(e)})


# =============================================================================
# FASTAPI APP
# =============================================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load both RAG collections so the first query isn't a cold-start stall."""
    async def _preload():
        try:
            await get_rules_collection_async()
        except Exception as e:
            print(f"Rules preload error: {e}", flush=True)
        try:
            await get_theory_collection_async()
        except Exception as e:
            print(f"Theory preload error: {e}", flush=True)
        print("RAG preload complete.", flush=True)

    async def _format_lists_daily():
        while True:
            try:
                await _load_format_lists()
            except Exception as e:
                print(f"Format lists error: {e}", flush=True)
            await asyncio.sleep(24 * 3600)

    asyncio.create_task(_preload())
    asyncio.create_task(_format_lists_daily())
    yield


app = FastAPI(title="MTG Deckbuilding Advisor", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret(),
    session_cookie="advisor_session",
    max_age=SESSION_MAX_AGE,
    same_site="lax",
    https_only=True,  # browsers still accept Secure cookies on http://localhost
)


@app.exception_handler(LoginRequired)
async def _login_required(request: Request, exc: LoginRequired):
    # Page loads bounce to the login screen; API calls get a 401 the UI handles.
    if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
        return RedirectResponse("/login", status_code=303)
    return JSONResponse({"error": "login required"}, status_code=401)


@app.get("/healthz")
async def healthz():
    """Unauthenticated liveness probe for the supervisor script."""
    return {"ok": True}


def _login_attempts_blocked(request: Request):
    """(client, fails, first_ts, blocked?) for the shared brute-force brake (login + sign-up)."""
    client = _client_key(request)
    fails, first = _login_fails.get(client, (0, 0.0))
    if fails and time.time() - first > LOGIN_LOCKOUT_SECS:
        fails, first = 0, 0.0  # window expired
    return client, fails, first, fails >= LOGIN_MAX_FAILS


async def _form(request: Request) -> dict:
    from urllib.parse import parse_qs
    form = parse_qs((await request.body()).decode("utf-8", "replace"))
    return {k: v[0] for k, v in form.items()}


@app.get("/login")
async def login_page(request: Request):
    if not AUTH_ENABLED or _valid_user(request.session.get("user")):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(LOGIN_FILE.read_text(encoding="utf-8"))


@app.post("/login")
async def login_submit(request: Request):
    if not AUTH_ENABLED:
        return RedirectResponse("/", status_code=303)
    client, fails, first, blocked = _login_attempts_blocked(request)
    if blocked:
        return RedirectResponse("/login?e=locked", status_code=303)

    form = await _form(request)
    user = (form.get("username") or "").strip()
    who = await asyncio.to_thread(_check_password, user, form.get("password") or "")
    if who:
        _login_fails.pop(client, None)
        request.session.clear()
        request.session["user"] = who
        log_activity(f"login ok user={who}")
        return RedirectResponse("/", status_code=303)

    _login_fails[client] = (fails + 1, first or time.time())
    log_activity(f"login FAILED user={user!r} client={client}")
    await asyncio.sleep(1)  # slow down guessing
    return RedirectResponse("/login?e=bad", status_code=303)


@app.get("/signup")
async def signup_page(request: Request):
    if not SITE_CODE:
        return RedirectResponse("/login", status_code=303)
    if _valid_user(request.session.get("user")):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(SIGNUP_FILE.read_text(encoding="utf-8"))


@app.post("/signup")
async def signup_submit(request: Request):
    if not SITE_CODE:
        return RedirectResponse("/login", status_code=303)
    client, fails, first, blocked = _login_attempts_blocked(request)
    if blocked:
        return RedirectResponse("/signup?e=locked", status_code=303)
    form = await _form(request)
    user = (form.get("username") or "").strip()
    if not secrets.compare_digest((form.get("code") or "").encode(), SITE_CODE.encode()):
        _login_fails[client] = (fails + 1, first or time.time())
        log_activity(f"signup FAILED (wrong site code) user={user!r} client={client}")
        await asyncio.sleep(1)
        return RedirectResponse("/signup?e=code", status_code=303)
    if (form.get("password") or "") != (form.get("confirm") or ""):
        return RedirectResponse("/signup?e=mismatch", status_code=303)
    try:
        who = await asyncio.to_thread(accounts.create, user, form.get("password") or "", set(ACCOUNTS))
    except accounts.AccountError as e:
        return RedirectResponse(f"/signup?e={e.code}", status_code=303)
    _login_fails.pop(client, None)
    request.session.clear()
    request.session["user"] = who
    log_activity(f"signup ok user={who} client={client}")
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login" if AUTH_ENABLED else "/", status_code=303)


@app.get("/me")
async def me(request: Request, _: None = Depends(require_auth)):
    return {"user": request.session.get("user"), "auth": AUTH_ENABLED}


def _ui_page() -> tuple[str, str]:
    """(page HTML, version). The version is a fingerprint of the UI file, stamped into the
    page as UI_VERSION so a tab left open across a deploy can tell it's stale."""
    raw = UI_FILE.read_text(encoding="utf-8")
    version = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return raw.replace("__UI_VERSION__", version), version


STALE_NOTE = ("_(This page is out of date - the app was updated. Refresh the page to get the "
              "latest version; your chats are saved.)_\n\n")


@app.get("/")
async def index(_: None = Depends(require_auth)):
    return HTMLResponse(_ui_page()[0])


@app.get("/version")
async def version(_: None = Depends(require_auth)):
    return {"ui": _ui_page()[1]}


@app.post("/chat")
async def chat(request: Request, _: None = Depends(require_auth)):
    body = await request.json()
    user_message = (body.get("message") or "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())
    if not user_message:
        return HTMLResponse("Empty message", status_code=400)

    # The client sends its full transcript as `history` (browser is the source of
    # truth), so conversation memory survives server restarts instead of living
    # only in the in-memory SESSIONS dict. Rebuild the message list from it; fall
    # back to the in-memory store for older clients that don't send history.
    history = body.get("history")
    deck = body.get("deck") if isinstance(body.get("deck"), dict) else None
    has_deck = bool(deck and deck.get("cards"))
    # Session memory: the browser keeps {text, upTo} per chat; history[:upTo] is already
    # condensed into text, so only history[upTo:] goes over verbatim.
    mem = body.get("memory") if isinstance(body.get("memory"), dict) else {}
    memory_text = (mem.get("text") or "").strip()
    mem_upto = int(mem.get("upTo") or 0) if memory_text else 0

    def _to_messages(raw):
        out = []
        for m in raw:
            text = (m.get("text") or "").strip()
            if m.get("role") not in ("user", "assistant") or not text:
                continue
            if m["role"] == "assistant":
                text += _decisions_summary(m)
            if m["role"] == "user":
                if has_deck:
                    text = _collapse_pasted_decklist(text)
                if m.get("note"):  # deck edits made in the editor before this message
                    text += "\n\n" + m["note"]
            out.append({"role": m["role"], "content": text})
        return out

    if isinstance(history, list) and history:
        mem_upto = min(mem_upto, len(history))
        recent = history[mem_upto:]
        messages = _to_messages(recent)
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_message})
        while messages and messages[0]["role"] != "user":  # API needs a user turn first
            messages.pop(0)
        SESSIONS[session_id] = messages
    else:
        history, recent = None, None
        messages = SESSIONS.setdefault(session_id, [])
        messages.append({"role": "user", "content": user_message})

    # Real client IP through the Cloudflare tunnel (falls back to socket peer).
    xff = request.headers.get("x-forwarded-for", "")
    client_ip = (
        request.headers.get("cf-connecting-ip")
        or (xff.split(",")[0].strip() if xff else "")
        or (request.client.host if request.client else "?")
    )
    # Tier the model over the WHOLE conversation: once a deck is in the chat, stay on
    # Sonnet for every follow-up (they're substantive deck reasoning); Haiku only for
    # deck-free trivia. Caching keeps Sonnet follow-ups cheap.
    convo_text = "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))
    model, extra = pick_model(convo_text + "\n" + memory_text)
    if has_deck or (deck and isinstance(deck.get("build"), dict)):
        model, extra = REVIEW_MODEL, REVIEW_EXTRA  # a live deck (or a build in progress) is always deck work
    preview = user_message.replace("\n", " ")[:200]
    ctx = []
    if has_deck:
        ctx.append(f"deck {sum(int(c.get('qty') or 1) for c in deck['cards'])}c")
    if memory_text:
        ctx.append(f"mem@{mem_upto}")
    who = request.session.get("user") or "-"
    log_activity(f"QUERY  sid={session_id[:8]} user={who} ip={client_ip} [{model.replace('claude-','')}]"
                 f"{' {' + ', '.join(ctx) + '}' if ctx else ''} | {preview}")

    client_ui = body.get("ui_version")
    ui_stale = client_ui != _ui_page()[1]

    async def _stream():
        nonlocal messages, memory_text, mem_upto
        if ui_stale:
            # Current pages show a refresh banner; pages from before this existed don't
            # know the event, so they get the note as the start of the answer instead.
            yield _sse("stale", {}) if client_ui else _sse("text", {"text": STALE_NOTE})
        # Condense the old part of a long chat into SESSION MEMORY (keeps the last
        # MEMORY_KEEP_RECENT messages verbatim), then hand the new memory to the browser.
        if recent is not None and sum(len(m["content"]) for m in messages) > MEMORY_TRIGGER_CHARS:
            cut = None  # history index of the first message to keep verbatim (a user turn)
            for i in range(len(history) - MEMORY_KEEP_RECENT, mem_upto, -1):
                if history[i].get("role") == "user":
                    cut = i
                    break
            excerpt = _to_messages(history[mem_upto:cut]) if cut else []
            # only worth a condensing call when there's a real chunk to fold in (otherwise
            # a few huge recent messages would trigger a tiny condense on every turn)
            if cut and sum(len(m["content"]) for m in excerpt) >= MEMORY_TRIGGER_CHARS // 2:
                yield _sse("status", {"tool": "memory", "input": "condensing earlier conversation"})
                try:
                    memory_text = await _condense_memory(memory_text, excerpt)
                    mem_upto = cut
                    messages = _to_messages(history[cut:])
                    if not messages or messages[-1]["role"] != "user":
                        messages.append({"role": "user", "content": user_message})
                    SESSIONS[session_id] = messages
                    yield _sse("memory", {"text": memory_text, "upTo": mem_upto})
                except Exception as e:  # never block the answer on memory upkeep
                    log_activity(f"MEMORY failed: {e}")
        t_prep = time.monotonic()
        deck_block, deck_text, deck_identity = await _deck_context(deck) if has_deck else ("", "", None)
        if time.monotonic() - t_prep >= 2:
            log_activity(f"PREP   sid={session_id[:8]} deck context {time.monotonic() - t_prep:.0f}s")
        if not has_deck and deck and isinstance(deck.get("build"), dict) and not deck["build"].get("done"):
            deck_block = "\n".join(  # a guided build that hasn't picked its commander yet
                ["# CURRENT DECK (live from the player's deck editor)",
                 "BUILD MODE - a new deck with NO COMMANDER YET and no cards. Help the player choose the "
                 "commander first (see BUILD MODE in your instructions).",
                 f"Target bracket: {deck.get('bracket') or 'not set'}"] + _brief_lines(deck.get("intake")))
        memory_block = ("# SESSION MEMORY (condensed earlier conversation - see instructions)\n"
                        + memory_text) if memory_text else ""
        async for chunk in agent_stream(session_id, messages, model, extra,
                                        system=_cached_system(memory_block, deck_block),
                                        deck_text=deck_text, deck_identity=deck_identity,
                                        deck=deck if has_deck else None):
            yield chunk

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
async def reset(request: Request, _: None = Depends(require_auth)):
    body = await request.json()
    session_id = body.get("session_id")
    if session_id:
        SESSIONS.pop(session_id, None)
    return {"ok": True}


# --- Card image proxy (cached) -----------------------------------------------
# The canvas used to fetch api.scryfall.com directly from every browser, one
# request per card. A full decklist (~100 cards) x several friends blew past
# Scryfall's rate limit -> 429s that render as "not found" tiles. This proxy
# resolves each name ONCE, server-side, with a proper User-Agent, and caches the
# result so the whole playgroup shares it (the Nth viewer never hits Scryfall).
CARD_IMG_CACHE: dict[str, dict | None] = {}  # lower(name) -> slim card | None (miss)
_scry_sem = asyncio.Semaphore(5)  # cap concurrent Scryfall fetches (stay under its limit)


def _slim_card(data: dict) -> dict:
    """Keep only the fields the canvas reads, in the same shape Scryfall returns."""
    faces = None
    if data.get("card_faces"):
        faces = [
            {"image_uris": {"normal": (f.get("image_uris") or {}).get("normal")}}
            for f in data["card_faces"]
        ]
    return {
        "name": data.get("name"),
        "scryfall_uri": data.get("scryfall_uri"),
        "color_identity": data.get("color_identity", []),
        "image_uris": ({"normal": data["image_uris"].get("normal")}
                       if data.get("image_uris") else None),
        "card_faces": faces,
    }


async def _scryfall_get(path: str, params: dict) -> dict | None:
    """GET a Scryfall endpoint with correct headers, the concurrency cap, and one 429 retry."""
    async with _scry_sem:
        async with scryfall_client(timeout=10.0) as client:
            for attempt in range(2):
                try:
                    r = await client.get(f"{SCRYFALL_API}{path}", params=params, headers=SCRYFALL_HEADERS)
                except httpx.HTTPError:
                    return None
                if r.status_code == 429 and attempt == 0:
                    await asyncio.sleep(float(r.headers.get("retry-after", "0.5")) or 0.5)
                    continue
                if r.status_code != 200:
                    return None
                return r.json()
    return None


FULL_CARD_CACHE: dict[str, dict | None] = {}  # lower(name) -> full Scryfall card | None


async def _resolve_full_card(name: str) -> dict | None:
    """Fuzzy-resolve a card to its full Scryfall object (cached, shared by all viewers)."""
    key = name.strip().lower()
    if key not in FULL_CARD_CACHE:
        card = await _scryfall_get("/cards/named", {"fuzzy": name})
        if card is None:  # ambiguous ("Krenko") or a heavier typo - search by popularity
            async with _scry_sem:
                async with scryfall_client() as client:
                    try:
                        card, _ = await resolve_card(client, name)
                    except httpx.HTTPError:
                        card = None
        FULL_CARD_CACHE[key] = card
    return FULL_CARD_CACHE[key]


async def _resolve_card(name: str) -> dict | None:
    data = await _resolve_full_card(name)
    return _slim_card(data) if data else None


def _slim_deck_card(c: dict, qty: int) -> dict:
    """Structured card for the deck object: identity, cost, image, flags, concrete roles."""
    tl = c.get("type_line", "")
    faces = c.get("card_faces") or []
    img = ((c.get("image_uris") or {}).get("normal")
           or (faces[0].get("image_uris", {}).get("normal") if faces else None))
    if not tl and faces:
        tl = " // ".join(f.get("type_line", "") for f in faces)
    try:
        import role_index
        roles = role_index.roles_for(c.get("name", ""))
    except Exception:
        roles = []
    oracle = c.get("oracle_text") or " ".join(f.get("oracle_text", "") for f in faces)
    return {
        "name": c.get("name"),
        "qty": qty,
        "scryfall_uri": c.get("scryfall_uri"),
        "type_line": tl,
        "cmc": c.get("cmc"),
        "mana_cost": c.get("mana_cost", "") or (faces[0].get("mana_cost", "") if faces else ""),
        "color_identity": c.get("color_identity", []),
        "image": img,
        # back face of a transform / modal double-faced card (adventures and splits have one image)
        "image_back": (faces[1].get("image_uris") or {}).get("normal") if len(faces) > 1 else None,
        "game_changer": bool(c.get("game_changer")),
        "is_land": "Land" in tl and "Creature" not in tl.split("//")[0],
        # singleton exemptions: basic lands, and "a deck can have any number of cards named ..."
        "any_qty": "Basic" in tl or "any number of cards named" in oracle,
        # a commander that lifts the 100-card maximum (Whtz, the Bibliophile)
        "no_max_deck": "no maximum deck size" in oracle.lower(),
        # could lead the deck. Overwritten by the Scryfall COMMANDER_QUERY result in
        # /deck/parse and /deck/card; this type-line guess is only the offline fallback.
        "can_command": ("Legendary" in tl and "Creature" in tl) or "Background" in tl
                       or "can be your commander" in oracle,
        "roles": roles,            # concrete roles (otag index)
        "role": None,              # contextual role - assigned by AI/user later
        "tags": [],                # user tags
    }


# Who can lead a deck, per Scryfall: is:commander covers legendary creatures and the
# exceptions ("can be your commander" planeswalkers etc., Backgrounds); t:background is
# spelled out so Backgrounds (partnered via "Choose a Background") are never flagged.
COMMANDER_QUERY = "(is:commander or t:background)"
_COMMANDER_OK: dict[str, bool] = {}  # lower(name) -> eligible


async def _apply_commander_eligibility(cards: list[dict]) -> None:
    """Set each deck card's can_command from Scryfall (batched exact-name searches,
    cached). On a Scryfall failure the type-line fallback from _slim_deck_card stays."""
    names = [c["name"] for c in cards if c.get("name")]
    # Only legendary cards (and the type-line/oracle guesses: Backgrounds, "can be your commander")
    # can lead a deck - everything else is a no without asking Scryfall.
    for c in cards:
        n = (c.get("name") or "").lower()
        if n and n not in _COMMANDER_OK and "Legendary" not in (c.get("type_line") or "") \
                and not c.get("can_command"):
            _COMMANDER_OK[n] = False
    unknown = [n for n in dict.fromkeys(names) if n.lower() not in _COMMANDER_OK]
    for i in range(0, len(unknown), 30):  # keep the query URL a sane length
        chunk = unknown[i:i + 30]
        q = COMMANDER_QUERY + " (" + " or ".join(
            f'!"{_collection_name(n).replace(chr(34), "")}"' for n in chunk) + ")"
        async with _scry_sem:
            async with scryfall_client(timeout=20.0) as client:
                try:
                    r = await client.get(f"{SCRYFALL_API}/cards/search", params={"q": q},
                                         headers=SCRYFALL_HEADERS)
                except httpx.HTTPError:
                    continue
        if r.status_code not in (200, 404):  # 404 = none of them qualify
            continue
        found = set()
        for c in (r.json().get("data") or []) if r.status_code == 200 else []:
            found.update({c["name"].lower(), _collection_name(c["name"]).lower()})
        for n in chunk:
            _COMMANDER_OK[n.lower()] = n.lower() in found or _collection_name(n).lower() in found
    for c in cards:
        ok = _COMMANDER_OK.get((c.get("name") or "").lower())
        if ok is not None:
            c["can_command"] = ok


@app.post("/deck/parse")
async def deck_parse(request: Request, _: None = Depends(require_auth)):
    """Parse a pasted decklist - or a deck link (Archidekt / Commander Template / Moxfield) -
    into structured deck cards (name-resolved, role-tagged)."""
    body = await request.json()
    text = body.get("text") or ""
    meta, companion_name = {}, None
    url = find_deck_url(text) if len(_DECK_LINE.findall(text)) < 15 else None
    if url:
        try:
            imp = await import_deck_url(url)
        except DeckImportError as e:
            return JSONResponse({"cards": [], "not_found": [], "error": str(e)})
        except Exception:
            return JSONResponse({"cards": [], "not_found": [], "error": "Couldn't import that deck link right now."})
        text = "\n".join(f"{q} {n}" for q, n in imp["cards"])
        meta = {"commander": imp["commanders"], "bracket": imp["bracket"], "source": imp["source"],
                "name": imp["name"], "skipped": imp["skipped"]}
        companion_name = imp.get("companion")
    parsed = parse_decklist(text)
    main = parsed["main"]
    if not url:  # a pasted export can name its commander, companion and board sections too
        meta = {"commander": parsed["commanders"], "skipped": parsed["skipped"]}
        companion_name = (parsed.get("companions") or [None])[0]
    if companion_name:
        full = await _resolve_full_card(companion_name)
        if full:
            meta["companion"] = _slim_deck_card(full, 1)
    qty_by_name = {}
    for e in main:
        qty_by_name[e["card"]] = qty_by_name.get(e["card"], 0) + e["quantity"]
    names = list(qty_by_name)
    resolved, found_names = [], set()
    async with scryfall_client(timeout=30.0) as client:
        for i in range(0, len(names), 75):
            chunk = names[i:i + 75]
            try:
                r = await client.post(
                    f"{SCRYFALL_API}/cards/collection",
                    json={"identifiers": [{"name": _collection_name(n)} for n in chunk]},
                    headers={**SCRYFALL_HEADERS, "Content-Type": "application/json"},
                )
                data = r.json()
            except Exception:
                continue
            for c in data.get("data", []):
                # match back to the requested qty (by canonical or the requested name)
                qty = qty_by_name.get(c.get("name"))
                if qty is None:  # pasted as just the front face of a multi-face card
                    qty = qty_by_name.get(_collection_name(c.get("name", "")))
                if qty is None:  # fuzzy/canonical differs - fall back to any unmatched in chunk
                    qty = next((qty_by_name[n] for n in chunk if n not in found_names), 1)
                resolved.append(_slim_deck_card(c, qty))
                found_names.add(c.get("name"))
    await _apply_commander_eligibility(resolved)
    lower_found = {n.lower() for n in found_names} | {_collection_name(n).lower() for n in found_names}
    not_found = [n for n in names if n.lower() not in lower_found
                 and n.lower().split(" // ")[0] not in lower_found]
    return JSONResponse({"cards": resolved, "not_found": not_found, **meta})


@app.post("/deck/roles")
async def deck_roles(request: Request, _: None = Depends(require_auth)):
    """Current concrete roles for a saved deck's cards. Roles are stored in the browser when a
    deck is loaded, so a role-index change (e.g. land fetchers no longer counted as Tutor) would
    otherwise never reach decks people already have."""
    body = await request.json()
    names = [n for n in (body.get("names") or []) if isinstance(n, str)][:300]
    try:
        import role_index
        return JSONResponse({"roles": {n: role_index.roles_for(n) for n in names}})
    except Exception:
        return JSONResponse({"roles": {}})


AUTOCOMPLETE_CACHE: dict[str, list] = {}


@app.get("/card/search")
async def card_search(q: str, commander: bool = False, _: None = Depends(require_auth)):
    """Card-name autocomplete for the deck editor's add-card box (Scryfall autocomplete, cached).
    commander=1: only cards that can lead a deck, most-played first (the build brief's picker)."""
    key = q.strip().lower()
    if len(key) < 2:
        return JSONResponse({"names": []})
    if commander:
        ckey = "cmdr:" + key
        if ckey not in AUTOCOMPLETE_CACHE:
            term = key.replace('"', "")
            data = await _scryfall_get("/cards/search", {"q": f'name:"{term}" {COMMANDER_QUERY} legal:commander',
                                                          "order": "edhrec"})
            if data is None:  # 404 (no match) and failures both land here - don't cache
                return JSONResponse({"names": []})
            AUTOCOMPLETE_CACHE[ckey] = [c["name"] for c in data.get("data") or []][:12]
        return JSONResponse({"names": AUTOCOMPLETE_CACHE[ckey]})
    if key not in AUTOCOMPLETE_CACHE:
        data = await _scryfall_get("/cards/autocomplete", {"q": key})
        if data is None:
            return JSONResponse({"names": []})  # transient failure: don't cache
        AUTOCOMPLETE_CACHE[key] = data.get("data", [])[:12]
    return JSONResponse({"names": AUTOCOMPLETE_CACHE[key]})


@app.get("/deck/card")
async def deck_card(name: str, _: None = Depends(require_auth)):
    """One card in the deck-object shape (same as /deck/parse items), for add-card."""
    data = await _resolve_full_card(name)
    if not data:
        return JSONResponse({"error": "not found"}, status_code=404)
    card = _slim_deck_card(data, 1)
    await _apply_commander_eligibility([card])
    return JSONResponse(card)


async def _batch_resolve(names: list[str]) -> None:
    """Fill CARD_IMG_CACHE (and FULL_CARD_CACHE) for these names: Scryfall's collection endpoint
    (75 per call) for exact names, the fuzzy resolver for the rest. Cached names cost nothing."""
    todo = [n for n in dict.fromkeys(names) if n.lower() not in CARD_IMG_CACHE]
    for i in range(0, len(todo), 75):
        chunk = todo[i:i + 75]
        async with _scry_sem:
            async with scryfall_client(timeout=30.0) as client:
                try:
                    r = await client.post(f"{SCRYFALL_API}/cards/collection", headers=SCRYFALL_HEADERS,
                                          json={"identifiers": [{"name": _collection_name(n)} for n in chunk]})
                    data = (r.json().get("data") or []) if r.status_code == 200 else []
                except (httpx.HTTPError, ValueError):
                    data = []
        by_name = {}
        for full in data:
            by_name[full["name"].lower()] = full
            by_name[_collection_name(full["name"]).lower()] = full
        for n in chunk:
            full = by_name.get(n.lower()) or by_name.get(_collection_name(n).lower())
            if full:
                for k in (n.lower(), full["name"].lower()):  # warms /deck/card for "+ Add"
                    FULL_CARD_CACHE.setdefault(k, full)
                CARD_IMG_CACHE[n.lower()] = _slim_card(full)
    # typos, nicknames, partial names: the fuzzy resolver (paced, cached)
    rest = [n for n in todo if n.lower() not in CARD_IMG_CACHE]
    for n, slim in zip(rest, await asyncio.gather(*(_resolve_card(n) for n in rest))):
        CARD_IMG_CACHE[n.lower()] = slim


@app.post("/deck/companion")
async def deck_companion(request: Request, _: None = Depends(require_auth)):
    """Companion check for the page: {cards: [{name, qty}], commander: [names], companion: name}."""
    body = await request.json()
    entries = [(str(c.get("name")), int(c.get("qty") or 1)) for c in (body.get("cards") or [])[:250] if c.get("name")]
    return JSONResponse(await _companion_report(entries, [str(n) for n in body.get("commander") or []],
                                                body.get("companion")))


@app.post("/cards")
async def cards_batch(request: Request, _: None = Depends(require_auth)):
    """Many /card lookups in one request: cached names answer at once, the rest go to
    Scryfall's collection endpoint (75 per call); names it can't match fall back to the
    fuzzy single-card resolver. -> {"cards": {requested name: card | null}}"""
    body = await request.json()
    names = [str(n).strip() for n in (body.get("names") or []) if str(n).strip()][:120]
    await _batch_resolve(names)
    out = {}
    try:
        import role_index
    except Exception:
        role_index = None
    for n in names:
        slim = CARD_IMG_CACHE.get(n.lower())
        if slim:
            slim = dict(slim)
            try:
                slim["roles"] = role_index.roles_for(slim.get("name") or n) if role_index else []
            except Exception:
                slim["roles"] = []
        out[n] = slim
    return JSONResponse({"cards": out})


@app.get("/card")
async def card(name: str, _: None = Depends(require_auth)):
    key = name.strip().lower()
    if key not in CARD_IMG_CACHE:
        CARD_IMG_CACHE[key] = await _resolve_card(name)
    slim = CARD_IMG_CACHE[key]
    if slim is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Attach concrete roles (otag index) so the canvas can group cards by role.
    out = dict(slim)
    try:
        import role_index
        out["roles"] = role_index.roles_for(slim.get("name") or name)
    except Exception:
        out["roles"] = []
    return JSONResponse(out)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advisor_app:app", host="127.0.0.1", port=8000, reload=False)
