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

TECH DEBT: tool implementations are imported from discord_bot to avoid a third
copy. Next refactor is to extract a shared `mtg_tools.py` that the bot, the MCP
server (mtg_mcp.py), and this app all import.
"""

import os
import json
import uuid
import asyncio
from pathlib import Path

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

# Reuse the existing, framework-agnostic tool handlers + schemas from the bot.
from discord_bot import (
    TOOLS as BASE_TOOLS,
    TOOL_FUNCTIONS as BASE_TOOL_FUNCTIONS,
    get_rules_collection_async,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "claude-sonnet-5"
MAX_TOKENS = 6000
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

# WHEN A DECKLIST IS PROVIDED (do this first, then advise)
If the message contains a decklist, your FIRST actions are exactly these two tool calls,
using the pasted text (as decklist_text) or the URL (as decklist_url):
1. spellbook_find_combos_in_decklist - to see what combos/synergies are already present
2. spellbook_estimate_bracket - to gauge the deck's current power level
Do these ONCE. Then reason about strengths and gaps from the results plus the framework
below. You already know the 99 - do not go card-by-card verifying it.

# TOOL BUDGET (hard limit - obey this strictly; latency matters a lot)
You have a budget of about 5 tool calls for a full deck review, fewer for smaller
questions. Every tool call adds ~10-15 seconds, so a sprawling 12-call response feels
broken to the user. Spend your budget like this:
- Full deck review: spellbook_find_combos_in_decklist + spellbook_estimate_bracket (2),
  plus AT MOST ONE scryfall_search_cards (scoped with id: to the commander's colors) to
  surface candidate upgrades. That is enough. Then STOP and WRITE YOUR ANSWER.
- Do NOT call scryfall_get_card to verify cards you already know - trust your own
  knowledge for well-known cards; only look up genuinely obscure or ambiguous ones.
- Do NOT make more than ONE deckbuilding_search call.
- The moment you have combo + bracket data, you have what you need to advise. A strong
  answer now beats a slightly-more-verified answer a minute later. Bias hard toward
  answering. When in doubt, answer from your knowledge instead of calling another tool.

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
- deckbuilding_search: search the theory corpus (deckbuilding videos/articles) for
  principles, archetype guides, and nuanced takes. Use it to ground advice and to cite
  sources. (If the corpus isn't built yet, it will say so - fall back to this framework.)
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
and budget. When you draw on the theory corpus, briefly cite the source. Keep it readable -
lead with the answer, then support it."""

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


def _short_input(tool_input: dict) -> str:
    """A compact, human-readable summary of a tool call's arguments."""
    try:
        s = json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        s = str(tool_input)
    return s if len(s) <= 160 else s[:157] + "..."


async def agent_stream(session_id: str, messages: list):
    """
    Drives the Claude tool-use loop, streaming text deltas and tool-status
    events to the browser as SSE frames.
    """
    yield _sse("session", {"session_id": session_id})
    try:
        for _ in range(MAX_ITERATIONS):
            async with aclient.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "disabled"},
            ) as stream:
                async for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and getattr(event.delta, "type", None) == "text_delta"
                    ):
                        yield _sse("text", {"text": event.delta.text})
                final = await stream.get_final_message()

            # Record the assistant turn (may contain text + tool_use blocks).
            messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason != "tool_use":
                break

            # Execute every tool the model asked for (concurrently), feed results back.
            tool_blocks = [b for b in final.content if b.type == "tool_use"]
            for block in tool_blocks:
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
        else:
            yield _sse("text", {"text": "\n\n_(Stopped after too many tool steps.)_"})

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
async def index():
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(request: Request):
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
async def reset(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    if session_id:
        SESSIONS.pop(session_id, None)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advisor_app:app", host="127.0.0.1", port=8000, reload=False)
