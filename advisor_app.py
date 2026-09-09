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
from pathlib import Path

import httpx
import anthropic
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv

load_dotenv()

# Shared, framework-agnostic tool handlers + schemas (also used by the Discord bot).
from mtg_tools import (
    TOOLS as BASE_TOOLS,
    TOOL_FUNCTIONS as BASE_TOOL_FUNCTIONS,
    get_rules_collection_async,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "claude-sonnet-5"
THINKING_EFFORT = "high"  # adaptive-thinking effort for Sonnet 5 (reason through card interactions)
MAX_TOKENS = 12000  # room for thinking + a full deck diagnosis
MAX_ITERATIONS = 6  # tool-use loop safety limit (bounds worst-case latency)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Theory corpus lives in its own Chroma directory so the monthly rules rebuild
# can't touch it. Populated later by theory_ingestion.py.
THEORY_DB_PATH = Path(__file__).parent / "mtg_theory_data"
THEORY_COLLECTION = "mtg_deckbuilding_theory"

UI_FILE = Path(__file__).parent / "advisor_ui.html"

aclient = anthropic.AsyncAnthropic()

# In-memory conversation store, keyed by session id. Fine for a local single
# user; swap for something persistent if this ever gets hosted for many users.
SESSIONS: dict[str, list] = {}

# --- Optional password gate (for exposing the app via a tunnel) --------------
# Auth is OFF when ADVISOR_PASSWORD is unset/empty (local single-user use is
# unchanged). Set ADVISOR_PASSWORD (and optionally ADVISOR_USER) in .env before
# exposing the app publicly so a leaked tunnel link alone can't spend API credits.
ADVISOR_USER = os.getenv("ADVISOR_USER", "player")
ADVISOR_PASSWORD = os.getenv("ADVISOR_PASSWORD", "")
_basic = HTTPBasic(auto_error=False)


async def require_auth(
    credentials: HTTPBasicCredentials | None = Depends(_basic),
) -> None:
    """Enforce HTTP Basic auth iff ADVISOR_PASSWORD is set. Browsers cache the
    credentials after the first prompt and resend them on same-origin /chat and
    /reset requests automatically, so no UI changes are needed."""
    if not ADVISOR_PASSWORD:
        return  # local mode: no gate
    ok = credentials is not None and secrets.compare_digest(
        credentials.username, ADVISOR_USER
    ) and secrets.compare_digest(credentials.password, ADVISOR_PASSWORD)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

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

# WHEN A DECKLIST IS PROVIDED (READ the cards first, then advise)
If the message contains a decklist, your FIRST action is:
1. scryfall_get_decklist_details (pass the pasted decklist_text) - it returns the ACTUAL
   oracle text, type, and color identity of every card. READ IT before you form any opinion.
   Do NOT guess what a card does from its name - your memory of card text is frequently
   wrong, ESPECIALLY for the commander, Universes Beyond / crossover cards, precons, and
   anything obscure. Every claim you make about a card must match the text you just read.
Then, once each:
2. spellbook_find_combos_in_decklist  3. spellbook_estimate_bracket (combos + power level).

Before advising, derive from the ACTUAL card text:
- What the COMMANDER literally does. Read its ability precisely - e.g. "whenever a creature
  you control attacks" means ANY of your creatures, so the commander itself need not attack.
  Getting the commander's engine wrong invalidates the whole analysis.
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
- Synergy-aware, not vacuum: evaluate each card against the COMMANDER'S payoff. In a
  type-matters deck (Hero-matters, tribal, etc.), a card that ISN'T the relevant type MISSES
  the payoff (cost reduction, tutoring, "whenever a [type]..." triggers) and still costs a
  slot - that's a mark AGAINST it, not a neutral or a plus.
- Don't cut the deck's actual PAYOFFS/finishers to "fix" a different category. Tribal
  pump/overruns, go-wide anthems, and "whenever a creature attacks" engines ARE the deck -
  a card that wins games in this archetype is not a "narrow/conditional" cut, even if it does
  nothing in the abstract. Cut filler and redundancy, not win conditions.
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

# TOOL BUDGET (be economical, but GROUND EVERY CLAIM)
A good budget for a full deck review is ~5 calls: scryfall_get_decklist_details on the deck
(1 - the important one, it grounds everything), spellbook_find_combos_in_decklist +
spellbook_estimate_bracket (2), optionally ONE scryfall_search_cards (scoped with
commander_identity) to find candidate upgrades (3), and a SECOND scryfall_get_decklist_details
on your shortlist of proposed adds (4) so every card you recommend is grounded in real text.
Then STOP and write the answer.
- Reason from the decklist details you fetched - you don't need to re-verify individual
  cards you already read there. Cards surfaced by scryfall_search_cards already come with text,
  so they're grounded too; the shortlist-verify call is for adds you thought of yourself.
- Do NOT make more than ONE deckbuilding_search call.
- The one thing you must NOT skimp on: never name a recommended card you haven't fetched this
  turn. Beyond that, once you've read the deck, the combo/bracket data, and your shortlist,
  you have what you need - write the answer.

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

Always VERIFY card text with scryfall_get_card before making a claim about what a card
does - never trust your memory of oracle text. When recommending cards for a commander,
respect COLOR IDENTITY strictly (use Scryfall id: searches scoped to the commander's colors).

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

# DECK REVIEW OUTPUT (diagnose first, prescribe second)
Do NOT jump straight to a cut/add list. Structure a full review as:
1. HOW IT WINS & PLAYS OUT - the real gameplan: the commander/engine's actual function, the
   best-case line, and what typical turns look like. Show you understand the deck.
2. STRUCTURAL READ - use the composition data (authoritative land count, creature/legendary
   density, removal/draw/ramp you can actually count from the card list) to say what's healthy
   vs. stretched. Cite real numbers, not vibes.
3. FAILURE MODES - the specific games where it stumbles and WHY - that's the thing to fix.
4. RECOMMENDATIONS - cuts and adds, each tied to a point above with the reasoning (why this
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
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
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

    async with httpx.AsyncClient(
        timeout=15.0, headers={"User-Agent": "mtg-advisor/1.0"}
    ) as client:
        sem = asyncio.Semaphore(6)

        async def lookup(name):
            async with sem:
                try:
                    r = await client.get(
                        "https://api.scryfall.com/cards/named",
                        params={"fuzzy": name},
                    )
                    if r.status_code != 200:
                        return None
                    ci = set(r.json().get("color_identity", []))
                    if not ci.issubset(allowed):
                        return {"name": name, "identity": "".join(sorted(ci)) or "C"}
                except Exception:
                    return None
                return None

        results = await asyncio.gather(*(lookup(n) for n in names))
    return [r for r in results if r]


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


async def agent_stream(session_id: str, messages: list):
    """
    Drives the Claude tool-use loop. Tool-call turns stream status live; the FINAL
    answer is buffered and validated for color-identity legality BEFORE it is shown,
    and regenerated if any off-identity card slipped in - so an illegal recommendation
    never reaches the user.
    """
    yield _sse("session", {"session_id": session_id})
    identity_retries = 0
    deck_identity = None  # authoritative commander identity, captured from tool args
    try:
        for _ in range(MAX_ITERATIONS + MAX_IDENTITY_RETRIES):
            turn_parts = []
            async with aclient.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                # Sonnet 5 adaptive thinking; passed via extra_body since SDK 0.75
                # doesn't yet type these params.
                extra_body={
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": THINKING_EFFORT},
                },
            ) as stream:
                async for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and getattr(event.delta, "type", None) == "text_delta"
                    ):
                        turn_parts.append(event.delta.text)
                final = await stream.get_final_message()

            messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason == "tool_use":
                # Suppress the model's mid-process narration on tool-call turns
                # (the "let me refine this search" chatter) - the tool trace shows
                # what's happening; only the final answer turn streams to the user.

                tool_blocks = [b for b in final.content if b.type == "tool_use"]
                for block in tool_blocks:
                    # Capture the commander identity the model passes to its tools -
                    # authoritative for the color-identity guardrail (doesn't depend on
                    # the %%IDENTITY%% marker, which the model sometimes forgets).
                    ci = (block.input or {}).get("commander_identity")
                    if ci and not deck_identity:
                        deck_identity = ci
                        yield _sse("identity", {"identity": ci})
                    yield _sse("status", {"tool": block.name, "input": _short_input(block.input)})

                async def _run(block):
                    func = TOOL_FUNCTIONS.get(block.name)
                    if func is None:
                        return f"Unknown tool: {block.name}"
                    try:
                        return await func(**block.input)
                    except Exception as e:
                        return f"Error running {block.name}: {e}"

                results = await asyncio.gather(*(_run(b) for b in tool_blocks))
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": b.id, "content": r}
                        for b, r in zip(tool_blocks, results)
                    ],
                })
                continue

            # ---- Final answer turn: validate color identity BEFORE showing it ----
            answer_text = "".join(turn_parts)
            try:
                off = await _check_off_color(answer_text, deck_identity)
            except Exception:
                off = None

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
            yield _sse("text", {"text": "\n\n_(Stopped after too many steps.)_"})

        yield _sse("done", {})
    except anthropic.APIError as e:
        yield _sse("error", {"message": f"API error: {e}"})
    except Exception as e:
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

    asyncio.create_task(_preload())
    yield


app = FastAPI(title="MTG Deckbuilding Advisor", lifespan=lifespan)


@app.get("/")
async def index(_: None = Depends(require_auth)):
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(request: Request, _: None = Depends(require_auth)):
    body = await request.json()
    user_message = (body.get("message") or "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())
    if not user_message:
        return HTMLResponse("Empty message", status_code=400)

    messages = SESSIONS.setdefault(session_id, [])
    messages.append({"role": "user", "content": user_message})

    return StreamingResponse(
        agent_stream(session_id, messages),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advisor_app:app", host="127.0.0.1", port=8000, reload=False)
