"""
MTG tool implementations shared by the Discord bot and the deckbuilding advisor
web app (and, for helpers, the MCP server). Single source of truth for the
Scryfall / Commander Spellbook / rules-RAG tool handlers plus their Anthropic
tool schemas (TOOLS) and the name->function map (TOOL_FUNCTIONS).
"""

import asyncio
import json
import logging
import os
import re
import httpx
from pathlib import Path
from typing import Optional

# Chroma 0.5.3 + posthog>=6 logs "Failed to send telemetry event ... capture() takes
# 1 positional argument" on every client start, even with telemetry off. Harmless;
# turn telemetry off and mute that logger.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Path to the rules database (created by rules_ingestion.py)
RULES_DB_PATH = Path(__file__).parent / "mtg_rules_data"


# Scryfall API settings (same as MCP server)
SCRYFALL_API = "https://api.scryfall.com"
SCRYFALL_HEADERS = {
    "User-Agent": "MTG-Discord-Bot/1.0",
    "Accept": "application/json"
}

# Commander Spellbook API
SPELLBOOK_API = "https://backend.commanderspellbook.com"


# Rules database (lazy loaded)
_rules_collection = None
_rules_loading = False  # Prevents multiple simultaneous loads


def _load_rules_collection_sync():
    """
    Synchronous function to load the rules collection.
    This is slow (~10-20s first time) because it imports heavy libraries.
    Should be run in a thread pool to avoid blocking the event loop.
    """
    global _rules_collection
    
    if not RULES_DB_PATH.exists():
        return None
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        
        client = chromadb.PersistentClient(path=str(RULES_DB_PATH))
        embedding_func = embedding_functions.DefaultEmbeddingFunction()  # ONNX all-MiniLM-L6-v2: same vectors, no PyTorch
        _rules_collection = client.get_collection(
            name="mtg_comprehensive_rules",
            embedding_function=embedding_func
        )
        return _rules_collection
    except Exception as e:
        print(f"Warning: Could not load rules database: {e}")
        return None


async def get_rules_collection_async():
    """
    Asynchronously loads the ChromaDB collection for rules search.
    Runs the heavy import/load in a thread pool to avoid blocking Discord's heartbeat.
    """
    global _rules_collection, _rules_loading
    
    # Return cached collection if already loaded
    if _rules_collection is not None:
        return _rules_collection
    
    # Check if database exists before trying to load
    if not RULES_DB_PATH.exists():
        return None
    
    # Prevent multiple simultaneous loads
    if _rules_loading:
        # Wait for the other load to finish
        while _rules_loading and _rules_collection is None:
            await asyncio.sleep(0.5)
        return _rules_collection
    
    _rules_loading = True
    
    try:
        # Run the blocking load in a thread pool
        # This prevents it from blocking Discord's heartbeat
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_rules_collection_sync)
        return _rules_collection
    finally:
        _rules_loading = False


def get_rules_collection():
    """
    Synchronous getter - only use if collection is already loaded.
    Returns None if not yet loaded (caller should use async version).
    """
    return _rules_collection


# =============================================================================
# TOOL DEFINITIONS FOR CLAUDE
# =============================================================================

# These tell Claude what tools are available and how to use them
TOOLS = [
    {
        "name": "scryfall_search_cards",
        "description": "Search for Magic: The Gathering cards using Scryfall's search syntax. Use operators like c: (color), t: (type), o: (oracle text), cmc: (mana value), pow: (power). When finding cards to recommend for a Commander deck, ALWAYS pass commander_identity - results are then hard-filtered to that color identity so only legal cards come back.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Scryfall search query. Examples: 'c:blue t:creature', 'o:\"draw a card\" cmc<=3', 't:legendary'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (1-10)",
                    "default": 5
                },
                "commander_identity": {
                    "type": "string",
                    "description": "The commander's color identity as WUBRG letters (e.g. 'WB', 'GWU', or 'C' for colorless). When set, results are strictly limited to cards legal in that identity. Always set this when searching for cards to add to a specific deck."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "scryfall_get_card",
        "description": "Look up a specific Magic: The Gathering card by name. Supports fuzzy matching for typos. Pass commander_identity to get an explicit legal/illegal verdict for that deck's color identity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Card name to look up"
                },
                "commander_identity": {
                    "type": "string",
                    "description": "Optional. The commander's color identity as WUBRG letters (e.g. 'WB'). When set, the result states whether this card is legal in that deck."
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "scryfall_get_rulings",
        "description": "Get official rulings for a specific card OR keyword ability. If you pass a keyword like 'myriad', 'cascade', 'mobilize', etc., it will find a card with that keyword and return relevant rulings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_name": {
                    "type": "string",
                    "description": "Name of the card OR keyword ability to get rulings for (e.g., 'Lightning Bolt' or 'myriad')"
                }
            },
            "required": ["card_name"]
        }
    },
    {
        "name": "spellbook_search_combos",
        "description": "Search for Commander/EDH combos on Commander Spellbook. Find combos by card names, effects, or color identity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - card names, effects, or 'card:\"Card Name\"' syntax"
                },
                "color_identity": {
                    "type": "string",
                    "description": "Filter by color identity using WUBRG letters (e.g., 'UB' for Dimir)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max combos to return (1-10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "spellbook_find_combos_for_cards",
        "description": "Find all combos that include specific cards. Great for discovering what combos are possible with cards you own.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of card names to find combos for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max combos to return (1-10)",
                    "default": 5
                }
            },
            "required": ["cards"]
        }
    },
    {
        "name": "spellbook_find_combos_in_decklist",
        "description": "Find all combos present in a decklist. Accepts either a deck URL (Moxfield, Archidekt, etc.) or a pasted list of card names.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decklist_url": {
                    "type": "string",
                    "description": "URL to a decklist (Moxfield, Archidekt, Deckstats, TappedOut, etc.)"
                },
                "decklist_text": {
                    "type": "string",
                    "description": "Pasted decklist as text - one card per line, quantity optional (e.g., '1 Sol Ring' or just 'Sol Ring')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max combos to return (1-20)",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "scryfall_get_decklist_details",
        "description": "Fetch the ACTUAL oracle text, type line, mana cost and color identity for every card in a decklist (pasted text OR a deck link), in one batch, plus the land count, concrete-role counts and game changers. Call this FIRST when reviewing a decklist so your evaluation is grounded in what the cards really do - do NOT guess a card's function from its name, especially for crossover/Universes Beyond/precon/obscure cards where your memory is often wrong.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decklist_text": {
                    "type": "string",
                    "description": "Pasted decklist as text, one card per line (quantity optional)."
                },
                "decklist_url": {
                    "type": "string",
                    "description": "Deck link (Archidekt works; Moxfield/Commander Template links can't be fetched - ask for the pasted export)."
                }
            }
        }
    },
    {
        "name": "spellbook_estimate_bracket",
        "description": "Commander Spellbook's power read of a decklist: its own power tier (Exhibition < Core < Oddball < Powerful < Spicy < Ruthless - NOT an official bracket number), the official game changers in the deck, banned cards, mass land denial, extra turns and combo counts. Use these signals with the official bracket criteria to place the deck.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decklist_url": {
                    "type": "string",
                    "description": "URL to a decklist (Moxfield, Archidekt, etc.)"
                },
                "decklist_text": {
                    "type": "string",
                    "description": "Pasted decklist as text - one card per line"
                }
            }
        }
    },
    {
        "name": "mtg_rules_search",
        "description": "Search the MTG Comprehensive Rules using semantic search. Ask rules questions in natural language. ALWAYS use this tool first when answering rules questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Rules question in natural language (e.g., 'How does summoning sickness work?')"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of relevant rules to return (1-10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def _identity_letters(commander_identity):
    """Normalize a WUBRG identity string to a set of uppercase letters ('' = colorless)."""
    if not commander_identity:
        return None
    return set(commander_identity.upper().replace("C", ""))


def _pop_label(rank) -> str:
    """Rough popularity bucket from EDHREC rank (lower = more played)."""
    if not rank:
        return "unranked"
    if rank <= 300:
        return "staple"
    if rank <= 2000:
        return "common"
    if rank <= 6000:
        return "niche"
    return "deep cut"


async def scryfall_search_cards(query: str, limit: int = 5, commander_identity: str = None) -> str:
    """
    Search for cards on Scryfall.

    When commander_identity is provided (e.g. "WB"), the search is HARD-scoped to
    that Commander color identity: the query is constrained with Scryfall's id<=
    operator AND every result is post-filtered so no card outside the identity can
    ever be returned. Always pass commander_identity when finding cards for a deck.
    """
    allowed = _identity_letters(commander_identity)
    q = query
    if allowed is not None:
        scope = f"id<={''.join(sorted(allowed)).lower()}" if allowed else "id:c"
        # legal:commander excludes Alchemy/digital-only cards (e.g. "A-" rebalances)
        # and Commander-banned cards, so recommendations are real, paper-legal cards.
        q = f"({query}) {scope} legal:commander"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{SCRYFALL_API}/cards/search",
                params={"q": q, "order": "edhrec"},  # popular -> obscure; lets us spot deep cuts
                headers=SCRYFALL_HEADERS,
                timeout=30.0
            )
            if response.status_code == 404:
                return "No cards found matching that search."
            response.raise_for_status()
            data = response.json()

            cards = data.get("data", [])
            # Belt-and-suspenders: guarantee color-identity legality on our side too.
            if allowed is not None:
                cards = [c for c in cards if set(c.get("color_identity", [])).issubset(allowed)]
            total = data.get("total_cards", len(cards))
            cards = cards[:limit]

            if not cards:
                scope_note = f" within color identity {commander_identity.upper()}" if allowed is not None else ""
                return f"No cards found matching that search{scope_note}."

            header = f"Found {total} cards (showing {len(cards)})"
            if allowed is not None:
                header += f", all legal in a {commander_identity.upper()} deck"
            lines = [header + ":"]
            for card in cards:
                name = card.get("name", "Unknown")
                mana = card.get("mana_cost", "")
                type_line = card.get("type_line", "")
                ci = "".join(card.get("color_identity", [])) or "C"
                gc = " [GAME CHANGER]" if card.get("game_changer") else ""
                rank = card.get("edhrec_rank")
                pop = f" · EDHREC ~{rank} ({_pop_label(rank)})" if rank else " · EDHREC unranked"
                lines.append(f"**{name}** {mana} - {type_line} [id:{ci}]{gc}{pop}")

            return "\n".join(lines)

        except httpx.HTTPStatusError as e:
            return f"Search error: {e.response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"


async def resolve_card(client: httpx.AsyncClient, name: str) -> tuple[Optional[dict], list[str]]:
    """Best card for a partial / misspelled / nickname-ish name, plus other plausible
    matches. Scryfall's fuzzy lookup handles typos ("sol rin") but refuses AMBIGUOUS
    names ("Morcant", "Krenko" match several cards) - fall back to a name search
    ranked by EDHREC popularity, then to autocomplete for heavier typos."""
    name = (name or "").strip()
    if not name:
        return None, []
    get = lambda path, params: client.get(f"{SCRYFALL_API}{path}", params=params,
                                          headers=SCRYFALL_HEADERS, timeout=30.0)
    r = await get("/cards/named", {"fuzzy": name})
    if r.status_code == 200:
        return r.json(), []
    phrase = name.replace('"', "")
    r = await get("/cards/search", {"q": f'name:"{phrase}" game:paper', "order": "edhrec", "unique": "cards"})
    if r.status_code == 200 and r.json().get("data"):
        data = r.json()["data"]
        return data[0], [c["name"] for c in data[1:8]]
    # Autocomplete matches word prefixes, so a typo mid-word ("craterhof") misses -
    # retry with the query trimmed back a few characters ("craterho" -> Craterhoof).
    names = []
    for cut in range(0, 4):
        q = name[:len(name) - cut] if cut else name
        if len(q) < 4:
            break
        r = await get("/cards/autocomplete", {"q": q})
        names = r.json().get("data", []) if r.status_code == 200 else []
        if names:
            break
    if names:
        r = await get("/cards/named", {"exact": names[0]})
        if r.status_code == 200:
            return r.json(), names[1:8]
    return None, []


async def scryfall_get_card(name: str, commander_identity: str = None) -> str:
    """
    Look up a specific card by name. When commander_identity is provided, the
    result includes an explicit color-identity legality verdict for that deck.
    """
    async with httpx.AsyncClient() as client:
        try:
            card, others = await resolve_card(client, name)
            if card is None:
                return f"Could not find card: {name}"

            lines = []
            if others:
                lines.append(
                    f"(\"{name}\" matches several cards - showing the most-played, {card.get('name')}. "
                    f"Other matches: {', '.join(others)}. If the player meant one of those, "
                    "look it up by its full name.)\n")
            
            # Check if this is a dual-faced card (DFC)
            # DFCs have a "card_faces" array instead of top-level oracle_text, mana_cost, etc.
            if "card_faces" in card:
                # Process each face of the card separately
                for i, face in enumerate(card["card_faces"]):
                    if i > 0:
                        lines.append("\n---\n")  # Separator between faces
                    
                    # Get face-specific attributes
                    face_name = face.get("name", "Unknown")
                    mana_cost = face.get("mana_cost", "")
                    lines.append(f"**{face_name}** {mana_cost}")
                    
                    type_line = face.get("type_line", "")
                    if type_line:
                        lines.append(f"*{type_line}*")
                    
                    oracle_text = face.get("oracle_text", "")
                    if oracle_text:
                        lines.append(oracle_text)
                    
                    # Power/toughness (for creature faces)
                    power = face.get("power")
                    toughness = face.get("toughness")
                    if power and toughness:
                        lines.append(f"**{power}/{toughness}**")
            
            else:
                # Single-faced card - use the original logic
                card_name = card.get("name", "Unknown")
                mana_cost = card.get("mana_cost", "")
                lines.append(f"**{card_name}** {mana_cost}")
                
                type_line = card.get("type_line", "")
                if type_line:
                    lines.append(f"*{type_line}*")
                
                oracle_text = card.get("oracle_text", "")
                if oracle_text:
                    lines.append(oracle_text)
                
                power = card.get("power")
                toughness = card.get("toughness")
                if power and toughness:
                    lines.append(f"**{power}/{toughness}**")
            
            # Color identity + legality verdict (always at top level, incl. DFCs)
            ci_letters = card.get("color_identity", [])
            ci = "".join(ci_letters) or "C"
            lines.append(f"\nColor identity: {ci}")
            allowed = _identity_letters(commander_identity)
            if allowed is not None:
                legal = set(ci_letters).issubset(allowed)
                if legal:
                    lines.append(f"Legality: LEGAL in a {commander_identity.upper()} deck.")
                else:
                    lines.append(
                        f"Legality: ILLEGAL in a {commander_identity.upper()} deck "
                        f"(color identity {ci} is outside {commander_identity.upper()}). "
                        f"DO NOT recommend this card."
                    )

            # Price is always at the top level (same for DFCs and normal cards)
            prices = card.get("prices", {})
            usd = prices.get("usd")
            if usd:
                lines.append(f"\nPrice: ${usd}")

            return "\n".join(lines)

        except httpx.HTTPStatusError:
            return f"Could not find card: {name}"
        except Exception as e:
            return f"Error: {str(e)}"


async def scryfall_get_rulings(card_name: str) -> str:
    """Get rulings for a card. Also handles keyword ability lookups."""
    
    # Common MTG keywords that people might search rulings for
    # If they search for a keyword, we'll find a card with it and get those rulings
    KEYWORDS = [
        "myriad", "mobilize", "cascade", "annihilator", "afflict", "aftermath",
        "amass", "amplify", "annihilator", "ascend", "aura swap", "awaken",
        "backup", "banding", "bargain", "battalion", "battle cry", "bestow",
        "blitz", "bloodrush", "bloodthirst", "boast", "bushido", "buyback",
        "casualty", "celebration", "champion", "changeling", "channel", "choose a background",
        "cipher", "clash", "cleave", "companion", "compleated", "connive",
        "conspire", "convoke", "corrupted", "council's dilemma", "coup de grâce",
        "craft", "crew", "cumulative upkeep", "cycling", "dash", "daybound",
        "deathtouch", "decayed", "defender", "delve", "detain", "devoid",
        "devour", "disguise", "disturb", "doctor's companion", "domain",
        "double strike", "dredge", "echo", "embalm", "emerge", "eminence",
        "enchant", "encore", "enlist", "enrage", "entwine", "escalate",
        "escape", "eternalize", "evoke", "evolve", "exalted", "excess damage",
        "exploit", "explore", "extort", "fabricate", "fading", "fateful hour",
        "fathomless descent", "fear", "ferocious", "fight", "first strike",
        "flanking", "flash", "flashback", "flying", "food", "for mirrodin!",
        "forecast", "foretell", "formidable", "friends forever", "fuse",
        "goad", "graft", "gravestorm", "graveyard hate", "haste", "haunt",
        "hellbent", "heroic", "hexproof", "hidden agenda", "hideaway",
        "horsemanship", "improvise", "incubate", "indestructible", "infect",
        "initiative", "inspired", "intensity", "intimidate", "investigate",
        "jump-start", "kicker", "landfall", "landwalk", "learn", "level up",
        "lifelink", "living weapon", "madness", "magecraft", "manifest",
        "meld", "melee", "menace", "mentor", "metalcraft", "mill", "miracle",
        "modular", "monstrosity", "morbid", "morph", "mutate", "ninjutsu",
        "offering", "offspring", "outlast", "overload", "pack tactics", "paradox",
        "parley", "partner", "persist", "phasing", "pilot", "plainscycling",
        "plot", "populate", "proliferate", "protection", "provoke", "prowess",
        "prowl", "radiance", "raid", "rampage", "ravenous", "reach", "rebound",
        "reconfigure", "recover", "reinforce", "renown", "replicate", "retrace",
        "revolt", "riot", "saddle", "scavenge", "scry", "shadow", "shroud",
        "skulk", "soulbond", "soulshift", "spectacle", "spell mastery",
        "splice", "split second", "squad", "storm", "strive", "sunburst",
        "support", "surge", "surveil", "suspend", "swampcycling", "threshold",
        "totem armor", "toxic", "trample", "training", "transfigure", "transform",
        "transmute", "treasure", "tribute", "undaunted", "undying", "unearth",
        "unleash", "vanishing", "vigilance", "ward", "wither"
    ]
    
    search_term = card_name.lower().strip()
    is_keyword = search_term in KEYWORDS
    
    async with httpx.AsyncClient() as client:
        try:
            # If it's a keyword, search for a card with that keyword first
            if is_keyword:
                # Search for a card with this keyword in its oracle text
                # Prefer cards that have the keyword as a main mechanic
                search_response = await client.get(
                    f"{SCRYFALL_API}/cards/search",
                    params={"q": f"o:{search_term}", "order": "edhrec"},
                    headers=SCRYFALL_HEADERS,
                    timeout=30.0
                )
                
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    cards = search_data.get("data", [])
                    
                    if cards:
                        # Use the first (most popular) card with this keyword
                        card = cards[0]
                        card_id = card.get("id")
                        actual_name = card.get("name")
                        
                        # Get rulings for this card
                        response = await client.get(
                            f"{SCRYFALL_API}/cards/{card_id}/rulings",
                            headers=SCRYFALL_HEADERS,
                            timeout=30.0
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        rulings = data.get("data", [])
                        if not rulings:
                            return f"No rulings found for the keyword '{search_term}' (searched via {actual_name})."
                        
                        # Filter to rulings that mention the keyword
                        keyword_rulings = [r for r in rulings if search_term in r.get("comment", "").lower()]
                        
                        # If we found keyword-specific rulings, show those; otherwise show all
                        rulings_to_show = keyword_rulings if keyword_rulings else rulings
                        
                        lines = [f"Rulings for **{search_term}** (via {actual_name}):"]
                        for ruling in rulings_to_show[:8]:  # Limit for Discord
                            comment = ruling.get("comment", "")
                            lines.append(f"• {comment}")
                        
                        if len(rulings_to_show) > 8:
                            lines.append(f"*...and {len(rulings_to_show) - 8} more rulings*")
                        
                        return "\n".join(lines)
                    else:
                        return f"Could not find any cards with the keyword '{search_term}'."
                else:
                    # Fall through to regular card lookup
                    pass
            
            # Regular card lookup (not a keyword)
            response = await client.get(
                f"{SCRYFALL_API}/cards/named",
                params={"fuzzy": card_name},
                headers=SCRYFALL_HEADERS,
                timeout=30.0
            )
            response.raise_for_status()
            card = response.json()
            card_id = card.get("id")
            actual_name = card.get("name")
            
            # Now get rulings
            response = await client.get(
                f"{SCRYFALL_API}/cards/{card_id}/rulings",
                headers=SCRYFALL_HEADERS,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            rulings = data.get("data", [])
            if not rulings:
                return f"No rulings found for {actual_name}."
            
            lines = [f"Rulings for **{actual_name}**:"]
            for ruling in rulings[:8]:  # Limit to 8 for Discord
                comment = ruling.get("comment", "")
                lines.append(f"• {comment}")
            
            if len(rulings) > 8:
                lines.append(f"*...and {len(rulings) - 8} more rulings*")
            
            return "\n".join(lines)
            
        except httpx.HTTPStatusError:
            if is_keyword:
                return f"Could not find rulings for the keyword '{search_term}'. Try searching for a specific card with this ability."
            return f"Could not find card: {card_name}"
        except Exception as e:
            return f"Error getting rulings: {str(e)}"


async def spellbook_search_combos(query: str, color_identity: Optional[str] = None, limit: int = 5) -> str:
    """Search for combos on Commander Spellbook."""
    async with httpx.AsyncClient() as client:
        try:
            params = {"q": query, "limit": limit}
            if color_identity:
                params["id"] = color_identity.upper()
            
            response = await client.get(
                f"{SPELLBOOK_API}/variants",
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            combos = data.get("results", [])[:limit]
            
            if not combos:
                return "No combos found matching that search."
            
            lines = [f"Found {len(combos)} combos:"]
            for combo in combos:
                # Get card names
                uses = combo.get("uses", [])
                card_names = [u.get("card", {}).get("name", "?") for u in uses]
                cards_str = " + ".join(card_names[:4])  # Limit card names shown
                if len(card_names) > 4:
                    cards_str += f" + {len(card_names) - 4} more"
                
                # Get results
                produces = combo.get("produces", [])
                results = [p.get("feature", {}).get("name", "") for p in produces[:2]]
                results_str = ", ".join(results) if results else "combo"
                
                lines.append(f"• {cards_str} → {results_str}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error searching combos: {str(e)}"


async def spellbook_find_combos_for_cards(cards: list, limit: int = 5) -> str:
    """Find combos containing specific cards."""
    # Build query with all card names
    card_queries = [f'card:"{card}"' for card in cards]
    combined_query = " OR ".join(card_queries)
    
    return await spellbook_search_combos(combined_query, limit=limit)


# A decklist line: "1 Sol Ring", "1x Sol Ring", "12x Plains" (quantity first).
DECK_LINE_RE = re.compile(r"^\s*\d+\s*[xX]?\s+\S", re.M)

# Section headers and per-card categories that mean "not in the 99".
_SECTION_COMMANDER = {"commander", "commanders", "companion"}
_SECTION_MAIN = {"deck", "main", "mainboard", "main deck", "the 99"}
_SECTION_SKIP = {"sideboard", "maybeboard", "maybe", "considering", "tokens", "token", "attractions",
                 "stickers", "contraptions"}
_LINE_RE = re.compile(r"^\s*(\d+)\s*[xX]?\s+(.+?)\s*$")


def parse_decklist(text: str) -> dict:
    """Parse a pasted decklist in any common export format into
    {"main": [{"card", "quantity"}] (the playable deck INCLUDING commanders),
     "commanders": [names], "skipped": n (maybe/side-board cards left out)}.

    Understands: plain "1 Sol Ring" / "1x Sol Ring"; set/collector suffixes "(C21) 263";
    section headers ("Commander", "// Sideboard", "MAYBEBOARD:", "Deck (99)"); and
    Archidekt's text export, where each line carries its categories and tags:
    "1x Agent of the Iron Throne (clb) 107 [Commander{top}] ^Sleeved,#fb00e5^"."""
    main, commanders, skipped = [], [], 0
    section = "main"
    # Bare names (no quantity) only count in a names-only list or under a Commander
    # header - otherwise the chat text around a pasted list would become "cards".
    names_only = not DECK_LINE_RE.search(text or "")
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            header = re.sub(r"\(\d+\)|[:/#*\-]", " ", line).strip().lower()
            header = " ".join(header.split())
            if header in _SECTION_COMMANDER:
                section = "commander"
            elif header in _SECTION_SKIP:
                section = "skip"
            elif header in _SECTION_MAIN:
                section = "main"
            elif (names_only or section == "commander") and len(line) < 60 and not re.search(r"https?://", line):
                # a bare card name (quantity 1) - old behavior kept for "Sol Ring" lines
                name = re.split(r"\s\(|\s\[|\s\^", line)[0].strip()
                if name and section != "skip" and not header.endswith("board"):
                    main.append({"card": name, "quantity": 1})
                    if section == "commander":
                        commanders.append(name)
            continue
        qty, rest = int(m.group(1)), m.group(2)
        cats = []
        for c in re.findall(r"\[([^\]]*)\]", rest):  # Archidekt categories
            cats += [re.sub(r"\{.*?\}", "", x).strip().lower() for x in c.split(",")]
        name = re.sub(r"\^[^^]*\^", "", rest)          # ^tags^
        name = re.sub(r"\[[^\]]*\]", "", name)          # [categories]
        name = re.sub(r"\*[A-Za-z]+\*", "", name)       # *F* foil / *E* etched markers
        name = re.split(r"\s\(", name, 1)[0].strip()    # (set) collector#
        if not name:
            continue
        if section == "skip" or any(c in _SECTION_SKIP for c in cats):
            skipped += qty
            continue
        if section == "commander" or "commander" in cats:
            commanders.append(name)
        main.append({"card": name, "quantity": qty})
    return {"main": main, "commanders": commanders, "skipped": skipped}


def _parse_decklist_to_main(text: str) -> list[dict]:
    """The playable deck (incl. commanders) as Spellbook 'main' entries:
    [{"card": name, "quantity": n}, ...]. Maybe/side boards are left out."""
    return parse_decklist(text)["main"]


def _collection_name(name: str) -> str:
    """Name to send to Scryfall /cards/collection. It rejects multi-face cards by their
    full "Front // Back" name (MDFCs, split cards, adventures) - silently dropping them
    from a deck - but resolves them by the front face."""
    return name.split(" // ")[0].strip()


_ZONES = {"B": "battlefield", "H": "hand", "G": "graveyard", "E": "exile", "L": "library", "C": "command zone"}
_WIN_WORDS = ("win the game", "lose the game", "loses the game", "wins the game")


async def _card_facts(names: list[str]) -> dict[str, dict]:
    """lower(name) -> {"mv": mana value, "type": front-face type line}, one batched Scryfall
    lookup (front-face names)."""
    out: dict[str, dict] = {}
    names = list(dict.fromkeys(n for n in names if n))
    async with httpx.AsyncClient(headers=SCRYFALL_HEADERS, timeout=30.0) as client:
        for i in range(0, len(names), 75):
            try:
                r = await client.post(f"{SCRYFALL_API}/cards/collection",
                                      json={"identifiers": [{"name": _collection_name(n)} for n in names[i:i + 75]]})
                data = r.json().get("data", []) if r.status_code == 200 else []
            except Exception:
                data = []
            for c in data:
                tl = c.get("type_line") or ((c.get("card_faces") or [{}])[0].get("type_line", ""))
                facts = {"mv": c.get("cmc"), "type": tl.split(" // ")[0]}
                out[c["name"].lower()] = facts
                out[_collection_name(c["name"]).lower()] = facts
    return out


_PERMANENT_TYPES = ("Creature", "Artifact", "Enchantment", "Planeswalker", "Land", "Battle")


def _combo_profile(combo: dict, facts: dict[str, dict]) -> str:
    """How heavy a combo really is: pieces with mana values/types and total, where each must
    be, how it can be interacted with, setup prerequisites, mana to run it, whether it wins on
    its own, and the step-by-step. The advisor judges bracket weight from this - Spellbook's
    one-letter tier is too coarse on its own."""
    uses = [u for u in combo.get("uses") or [] if isinstance(u, dict) and u.get("card", {}).get("name")]
    pieces, total, unknown = [], 0.0, False
    on_board: list[str] = []  # permanent types that must be on the battlefield
    for u in uses:
        n = u["card"]["name"]
        f = facts.get(n.lower(), facts.get(_collection_name(n).lower())) or {}
        v, tl = f.get("mv"), f.get("type", "")
        main_types = "/".join(t for t in ("Creature", "Artifact", "Enchantment", "Planeswalker", "Land",
                                          "Battle", "Instant", "Sorcery") if t in tl)
        zone_codes = u.get("zoneLocations") or ["B"]
        zones = "/".join(_ZONES.get(z, z) for z in zone_codes)
        state = u.get("battlefieldCardState") or ""
        extra = ", ".join(x for x in (main_types, zones if zones != "battlefield" else "", state) if x)
        pieces.append(f"{n} (MV {v:g}{', ' + extra if extra else ''})" if v is not None
                      else f"{n}{' (' + extra + ')' if extra else ''}")
        if v is None:
            unknown = True
        else:
            total += v
        if zone_codes == ["B"] and any(t in tl for t in _PERMANENT_TYPES):
            on_board.append(main_types.lower() or "permanent")
    templates = [f"{r.get('quantity', 1)}x {r['template']['name']}" if r.get("quantity", 1) > 1 else r["template"]["name"]
                 for r in combo.get("requires") or [] if isinstance(r, dict) and r.get("template")]
    produces = [(p.get("feature") or {}).get("name") or "" for p in combo.get("produces") or [] if isinstance(p, dict)]
    wins = any(w in p.lower() for p in produces for w in _WIN_WORDS)
    prereq = " ".join(x.strip() for x in (combo.get("easyPrerequisites") or "", combo.get("notablePrerequisites") or "") if x and x.strip())
    parts = [f"{len(uses) + len(templates)} pieces: {', '.join(pieces)}"
             + (f" + any {', '.join(templates)}" if templates else "")
             + (f" = {total:g}{'+' if unknown else ''} MV to deploy" if pieces else "")]
    if prereq:
        parts.append(f"setup: {' '.join(prereq.split())[:220]}")
    if combo.get("manaNeeded"):
        parts.append(f"mana to run: {combo['manaNeeded']}")
    parts.append("wins on its own: YES" if wins else "wins on its own: NO - needs a separate payoff to close the game")
    if on_board:
        kinds = ", ".join(f"{on_board.count(k)} {k}" for k in dict.fromkeys(on_board))
        parts.append(f"interaction: {len(on_board)} permanent(s) must stay on the battlefield ({kinds}) - "
                     "removal on any one stops it, including in response to a trigger mid-loop")
    else:
        parts.append("interaction: no piece has to sit on the battlefield - hard to answer with removal")
    tapped = [p for p in produces if "tapped" in p.lower()]
    if tapped:
        parts.append(f"note: outputs arrive TAPPED ({', '.join(tapped)}) - tapped Treasure/lands can't make "
                     "mana this turn without an untap effect")
    if combo.get("bracketTag"):
        parts.append(f"Spellbook tier {combo['bracketTag']} (coarse - judge from the profile)")
    steps = " ".join((combo.get("description") or "").split())
    out = "\n    " + " | ".join(parts)
    if steps:
        out += f"\n    steps: {steps[:420]}{'…' if len(steps) > 420 else ''}"
    return out


def _format_combo(combo: dict) -> str:
    """One-line summary of a Spellbook combo: cards → what it produces."""
    uses = combo.get("uses", []) or []
    names = [u.get("card", {}).get("name") for u in uses if isinstance(u, dict)]
    names = [n for n in names if n]
    cards_str = " + ".join(names[:5]) if names else "combo"
    if len(names) > 5:
        cards_str += f" +{len(names) - 5} more"
    produces = combo.get("produces", []) or []
    prod = []
    for p in produces[:3]:
        if isinstance(p, dict):
            feat = p.get("feature") or {}
            prod.append(feat.get("name") or p.get("name") or "")
    prod_str = ", ".join(x for x in prod if x)
    return cards_str + (f" -> {prod_str}" if prod_str else "")


# --- Deck URL import (Archidekt / Commander Template / Moxfield) ---------------
DECK_URL_RE = re.compile(
    r"https?://(?:www\.)?(archidekt\.com/(?:api/)?decks/(?P<arch>\d+)"
    r"|moxfield\.com/decks/(?P<mox>[\w-]+)"
    r"|commandertemplate\.com/(?:decks|precons)/(?P<ct>[\w-]+))", re.I)
IMPORT_HEADERS = {"User-Agent": "MTG-Deckbuilding-Advisor/1.0 (personal deck import)"}


class DeckImportError(Exception):
    """A deck link that can't be imported; the message is shown to the player."""


def find_deck_url(text: str):
    m = DECK_URL_RE.search(text or "")
    return m.group(0) if m else None


async def import_deck_url(url: str) -> dict:
    """Fetch a public deck from its link. Returns {source, name, commanders:[names],
    cards:[(qty, name)] (the MAIN deck incl. commanders; maybe/side boards excluded),
    skipped (maybeboard/sideboard card count), bracket (int|None)}."""
    m = DECK_URL_RE.search(url or "")
    if not m:
        raise DeckImportError("Not a supported deck link (Archidekt, Moxfield, Commander Template).")
    async with httpx.AsyncClient(headers=IMPORT_HEADERS, timeout=30.0, follow_redirects=True) as client:
        if m.group("arch"):
            r = await client.get(f"https://archidekt.com/api/decks/{m.group('arch')}/")
            if r.status_code != 200:
                raise DeckImportError("Couldn't read that Archidekt deck - is it public?")
            d = r.json()
            included = {c["name"]: c.get("includedInDeck", True) for c in d.get("categories", [])}
            premier = {c["name"] for c in d.get("categories", []) if c.get("isPremier")}
            cards, commanders, skipped = [], [], 0
            for c in d.get("cards", []):
                if c.get("deletedAt"):
                    continue
                name = c["card"]["oracleCard"]["name"]
                cats = c.get("categories") or []
                if cats and not included.get(cats[0], True):  # primary category = its board
                    skipped += c.get("quantity", 1)
                    continue
                if premier & set(cats):
                    commanders.append(name)
                cards.append((c.get("quantity", 1), name))
            return {"source": "Archidekt", "name": d.get("name"), "commanders": commanders,
                    "cards": cards, "skipped": skipped, "bracket": d.get("edhBracket")}

        if m.group("mox"):
            r = await client.get(f"https://api2.moxfield.com/v3/decks/all/{m.group('mox')}")
            if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
                raise DeckImportError(
                    "Moxfield blocks automated access to decks. In Moxfield use "
                    "Export -> Copy as plain text, and paste the list here instead.")
            d = r.json()
            boards = d.get("boards", {})
            cards, commanders = [], []
            for b in ("commanders", "companions", "mainboard"):
                for c in (boards.get(b, {}).get("cards") or {}).values():
                    cards.append((c.get("quantity", 1), c["card"]["name"]))
                    if b == "commanders":
                        commanders.append(c["card"]["name"])
            skipped = sum(boards.get(b, {}).get("count", 0) for b in ("maybeboard", "sideboard"))
            return {"source": "Moxfield", "name": d.get("name"), "commanders": commanders,
                    "cards": cards, "skipped": skipped, "bracket": None}

        # Commander Template: a Next.js page that embeds the full deck object in its
        # flight data (self.__next_f.push chunks) - no public API needed.
        r = await client.get(url)
        if r.status_code == 403:
            raise DeckImportError(
                "Commander Template blocks automated access to its pages. Copy the decklist "
                "out of Commander Template and paste it here instead.")
        if r.status_code != 200:
            raise DeckImportError("Couldn't open that Commander Template deck - is it public?")
        chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', r.text, re.S)
        flight = "".join(json.loads('"' + c + '"') for c in chunks)
        deck = None
        dec = json.JSONDecoder()
        k = flight.find('"selectedCommanders"')
        while k != -1 and deck is None:
            start = flight.rfind('{"id":"', 0, k)
            try:
                obj, _ = dec.raw_decode(flight, start)
                if "deckInstances" in obj:
                    deck = obj
            except ValueError:
                pass
            k = flight.find('"selectedCommanders"', k + 1)
        if deck is None:
            raise DeckImportError("Couldn't find a deck on that Commander Template page - it may be private.")
        counts: dict[str, int] = {}
        for inst in deck.get("deckInstances") or []:
            n = (inst.get("cardData") or {}).get("name")
            if n:
                counts[n] = counts.get(n, 0) + 1
        commanders = [c["name"] for c in deck.get("selectedCommanders") or [] if c.get("name")]
        if (deck.get("selectedCompanion") or {}).get("name"):
            counts[deck["selectedCompanion"]["name"]] = 1
        cards = [(1, n) for n in commanders] + [(q, n) for n, q in counts.items()]
        bracket = deck.get("deckBracket")
        return {"source": "Commander Template", "name": deck.get("name"), "commanders": commanders,
                "cards": cards, "skipped": len(deck.get("maybeboardInstances") or []),
                "bracket": bracket if isinstance(bracket, int) else None}


async def _decklist_to_main(client, decklist_url, decklist_text):
    """Resolve a decklist (pasted text or URL) into the 'main' payload list."""
    if decklist_text:
        return _parse_decklist_to_main(decklist_text)
    try:
        imported = await import_deck_url(decklist_url)
    except DeckImportError as e:
        raise ValueError(str(e))
    return [{"card": n, "quantity": q} for q, n in imported["cards"]]


async def scryfall_get_decklist_details(decklist_text: str = None, decklist_url: str = None) -> str:
    """
    Fetch the ACTUAL oracle text, type, mana cost and color identity for every card
    in a pasted decklist, in one batch (Scryfall /cards/collection). Lets the advisor
    reason from what cards really do instead of guessing from names.
    """
    if not decklist_text and decklist_url:
        try:
            imported = await import_deck_url(decklist_url)
        except DeckImportError as e:
            return f"I couldn't import that deck URL: {e}"
        decklist_text = "\n".join(f"{q} {n}" for q, n in imported["cards"])
    if not decklist_text:
        return "Provide the decklist as pasted text (one card per line) or a deck link to read its cards."
    main = _parse_decklist_to_main(decklist_text)
    names, seen = [], set()
    for e in main:
        k = e["card"].lower()
        if k not in seen:
            seen.add(k)
            names.append(e["card"])
    if not names:
        return "Couldn't parse any cards from that decklist."

    cards, not_found = [], []
    async with httpx.AsyncClient(headers=SCRYFALL_HEADERS, timeout=30.0) as client:
        for i in range(0, len(names), 75):  # Scryfall collection endpoint caps at 75
            batch = [{"name": _collection_name(n)} for n in names[i:i + 75]]
            try:
                resp = await client.post(f"{SCRYFALL_API}/cards/collection", json={"identifiers": batch})
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return f"Error fetching card details: {e}"
            cards.extend(data.get("data", []))
            not_found.extend(nf.get("name", "?") for nf in data.get("not_found", []))
            await asyncio.sleep(0.1)

    # Quantities (per pasted name), robust to MDFC front/full-name mismatches.
    qty = {}
    for e in main:
        qty[e["card"].lower()] = qty.get(e["card"].lower(), 0) + int(e.get("quantity", 1))

    def qty_for(c):
        keys = [c.get("name", "").lower(), c.get("name", "").split("//")[0].strip().lower()]
        keys += [(f.get("name", "") or "").lower() for f in (c.get("card_faces") or [])]
        for k in keys:
            if k in qty:
                return qty[k]
        return 1

    def type_lines(c):
        tls = [c.get("type_line", "")]
        tls += [f.get("type_line", "") for f in (c.get("card_faces") or [])]
        return [t.lower() for t in tls if t]

    # Deterministic composition counts (qty-weighted): lands (incl. MDFC land-backs),
    # creatures, legendaries, and a type breakdown - so synergy density is visible
    # (e.g. a legendary-heavy deck makes "ramp only for legendary spells" premium).
    land_count, mdfc_lands = 0, []
    creature_count = legendary_count = 0
    type_counts = {}
    for c in cards:
        tls = type_lines(c)
        n = qty_for(c)
        joined = " ".join(tls)
        if any("land" in t for t in tls):
            land_count += n
            top = c.get("type_line", "").lower()
            if "//" in top and not top.strip().startswith("land"):
                mdfc_lands.append(c.get("name", "?"))
        if "creature" in joined:
            creature_count += n
        if "legendary" in joined:
            legendary_count += n
        for t in ("creature", "instant", "sorcery", "artifact", "enchantment", "planeswalker", "battle"):
            if t in joined:
                type_counts[t] = type_counts.get(t, 0) + n

    def fmt(c):
        name = c.get("name", "?")
        ci = "".join(c.get("color_identity", [])) or "C"
        tl = c.get("type_line", "")
        if c.get("card_faces") and not c.get("oracle_text"):
            faces = c["card_faces"]
            cost = faces[0].get("mana_cost", "")
            ot = " // ".join(f.get("oracle_text", "") for f in faces)
        else:
            cost = c.get("mana_cost", "")
            ot = c.get("oracle_text", "")
        pt = f" [{c.get('power')}/{c.get('toughness')}]" if c.get("power") is not None else ""
        gc = " [GAME CHANGER]" if c.get("game_changer") else ""
        return f"[{ci}] {name}{gc} {cost} - {tl}{pt}: {' '.join(ot.split())}"

    composition = f"MANA BASE: {land_count} land sources (authoritative count - use this, don't recount)."
    if mdfc_lands:
        composition += (f" Includes {len(mdfc_lands)} MDFC/flex land(s) that count as lands: "
                        f"{', '.join(mdfc_lands[:8])}.")
    breakdown = ", ".join(f"{t}s {type_counts[t]}" for t in
                          ("creature", "instant", "sorcery", "artifact", "enchantment", "planeswalker", "battle")
                          if type_counts.get(t))
    composition += (f"\nCOMPOSITION (qty-weighted): {creature_count} creatures "
                    f"({legendary_count} legendary permanents) | {breakdown}. "
                    "Use this density to judge synergy - e.g. a legendary-heavy deck makes "
                    "'ramp/effects that only work for legendary spells' premium, not redundant.")

    # CONCRETE role composition from the Scryfall otag index (deterministic skeleton).
    # Contextual roles (Enabler/Payoff/Force Multiplier/Engine Piece/Threats/Alt Wincon/
    # Misc Value) are NOT here - the advisor assigns those holistically from the gameplan.
    try:
        import role_index
        nonland_qty = {}
        for c in cards:
            if any("land" in t for t in type_lines(c)):
                continue
            nm = c.get("name", "")
            nonland_qty[nm] = nonland_qty.get(nm, 0) + qty_for(c)
        rc = role_index.deck_concrete_roles(list(nonland_qty))
        counts = {}
        for nm, rs in rc["per_card"].items():
            for r in rs:
                counts[r] = counts.get(r, 0) + nonland_qty.get(nm, 1)
        n_ctx = sum(1 for nm, rs in rc["per_card"].items() if not rs)
        if counts:
            concrete_line = ", ".join(f"{r} {counts[r]}" for r in sorted(counts, key=lambda k: -counts[k]))
            composition += (
                "\nCONCRETE ROLES (Scryfall otags - authoritative counts, use these): " + concrete_line + ". "
                f"The other ~{n_ctx} nonland cards carry CONTEXTUAL roles you must assign yourself from "
                "the deck's gameplan (Enabler / Payoff / Force Multiplier / Engine Piece / Threats / "
                "Alternate Wincon / Misc Value) - a card can hold a concrete role AND a contextual one.")
    except Exception:
        pass

    # Game changers in the deck (official Commander bracket list, via Scryfall's flag).
    # Bracket 2 allows 0, Bracket 3 allows up to 3, Bracket 4-5 unlimited - the advisor
    # must respect this when recommending, since GC adds change the deck's bracket.
    gc_cards = sorted(c.get("name", "?") for c in cards if c.get("game_changer"))
    if gc_cards:
        composition += (f"\nGAME CHANGERS in deck: {len(gc_cards)} ({', '.join(gc_cards)}). "
                        "Bracket limits: B2=0, B3=up to 3, B4-5=unlimited. Count these against the "
                        "target bracket before recommending any card also marked [GAME CHANGER].")
    else:
        composition += ("\nGAME CHANGERS in deck: 0. (Bracket limits: B2=0, B3=up to 3, "
                        "B4-5=unlimited - stay within the target bracket when recommending.)")

    lines = [composition,
             f"\nActual card details for {len(cards)}/{len(names)} cards (color identity in [brackets]):\n"]
    lines += [fmt(c) for c in sorted(cards, key=lambda x: x.get("name", ""))]
    if not_found:
        lines.append(f"\nNot resolved (check exact names): {', '.join(not_found[:25])}")
    return "\n".join(lines)


async def spellbook_find_combos_in_decklist(
    decklist_url: str = None,
    decklist_text: str = None,
    limit: int = 10
) -> str:
    """
    Find all combos present in a decklist.
    Can accept either a URL to a deck or pasted card list.
    """
    if not decklist_text and not decklist_url:
        return "Please paste a decklist (one card per line, e.g. '1 Sol Ring')."
    async with httpx.AsyncClient() as client:
        try:
            try:
                main = await _decklist_to_main(client, decklist_url, decklist_text)
            except Exception as e:
                return (f"I couldn't import that deck URL: {e} "
                        "Otherwise, paste the decklist text (one card per line).")
            if not main:
                return "Couldn't parse any cards from that decklist."

            response = await client.post(
                f"{SPELLBOOK_API}/find-my-combos/",
                json={"main": main},
                timeout=60.0,  # Can be slow for large decklists
            )
            response.raise_for_status()
            results = response.json().get("results", {})
            included = results.get("included", []) or []
            almost = results.get("almostIncluded", []) or []
            identity = results.get("identity", "")

            if not included and not almost:
                return f"No combos found in this deck ({len(main)} cards, color identity {identity})."

            shown_almost = almost[:max(0, limit - len(included))]
            mv = await _card_facts([u.get("card", {}).get("name") for c in included[:limit] + shown_almost
                                          for u in c.get("uses") or [] if isinstance(u, dict)])

            lines = [f"Analyzed {sum(e.get('quantity', 1) for e in main)} cards (color identity {identity})."]
            lines.append(f"\n**COMBOS IN THE DECK ({len(included)})** - every piece is already in the list. "
                         "Each has a profile (pieces + mana values, setup, whether it wins on its own): judge "
                         "its bracket weight from that, not from Spellbook's tier.")
            for combo in included[:limit]:
                lines.append(f"• {_format_combo(combo)}{_combo_profile(combo, mv)}")
            if not included:
                lines.append("• none")

            remaining = max(0, limit - len(included))
            if almost and remaining > 0:
                have = set()
                for e in main:
                    n = e["card"].lower()
                    have.update({n, n.split(" // ")[0]})
                lines.append(
                    f"\n**NOT COMBOS - near-misses ({len(almost)})**. The deck does NOT have these combos; each is "
                    "missing the card(s) marked ADD. Those missing cards are candidate recommendations (they'd "
                    "complete a combo) - but check the target bracket first: completing a two-card INFINITE is off-limits "
                    "in B1-B2 and must not be early-game in B3; 3+ card combos aren't restricted.")
                for combo in shown_almost:
                    names = [u.get("card", {}).get("name") for u in combo.get("uses") or [] if isinstance(u, dict)]
                    names = [n for n in names if n]
                    missing = [n for n in names if n.lower() not in have and n.lower().split(" // ")[0] not in have]
                    held = [n for n in names if n not in missing]
                    needs = [r.get("template", {}).get("name") for r in combo.get("requires") or [] if isinstance(r, dict)]
                    needs = [n for n in needs if n]
                    prod = [(p.get("feature") or {}).get("name") for p in (combo.get("produces") or [])[:3] if isinstance(p, dict)]
                    prod = ", ".join(x for x in prod if x)
                    tier = combo.get("bracketTag")
                    lines.append(
                        f"• ADD {' + '.join(missing) or '?'}"
                        + (f" (with {', '.join(held)} already in deck)" if held else "")
                        + (f" + any {', '.join(needs)}" if needs else "")
                        + (f" -> {prod}" if prod else "")
                        + _combo_profile(combo, mv))

            return "\n".join(lines)

        except httpx.HTTPStatusError as e:
            return f"Error analyzing decklist: {e.response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"


async def mtg_rules_search(query: str, num_results: int = 5) -> str:
    """Search the Comprehensive Rules."""
    # Use async loader to avoid blocking the event loop
    collection = await get_rules_collection_async()
    
    if collection is None:
        return "Rules database not available. Run rules_ingestion.py to set it up."
    
    try:
        # Run the query in a thread pool too, since ChromaDB can be slow
        loop = asyncio.get_event_loop()
        
        def do_query():
            return collection.query(
                query_texts=[query],
                n_results=num_results
            )
        
        results = await loop.run_in_executor(None, do_query)
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return "No relevant rules found."
        
        lines = []
        for doc, meta in zip(documents, metadatas):
            rule_num = meta.get("rule_number", "?")
            # Truncate long rules for Discord, but keep more context
            text = doc[:500] + "..." if len(doc) > 500 else doc
            lines.append(f"**{rule_num}**: {text}")
        
        return "\n\n".join(lines)
        
    except Exception as e:
        return f"Error searching rules: {str(e)}"


async def spellbook_estimate_bracket(
    decklist_url: str = None,
    decklist_text: str = None
) -> str:
    """
    Estimate the Commander bracket (power level) for a decklist.
    
    Bracket levels:
    - Bracket 1: Casual - No two-card infinite combos
    - Bracket 2: Precon-appropriate / Oddball - Simple combos, fair play
    - Bracket 3: Powerful / Spicy - Strong combos, optimized
    - Bracket 4: Ruthless / cEDH - Competitive, fast combos
    """
    if not decklist_text and not decklist_url:
        return "Please paste a decklist to estimate its bracket."
    async with httpx.AsyncClient() as client:
        try:
            try:
                main = await _decklist_to_main(client, decklist_url, decklist_text)
            except Exception as e:
                return (f"I couldn't import that deck URL: {e} "
                        "Otherwise, paste the decklist text (one card per line).")
            if not main:
                return "Couldn't parse any cards from that decklist."

            response = await client.post(
                f"{SPELLBOOK_API}/estimate-bracket/",
                json={"main": main},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            tag = data.get("bracketTag", "?")
            cards = [c for c in (data.get("cards") or []) if isinstance(c, dict)]
            combos = data.get("combos", []) or []

            # Spellbook's OWN power tiers (per its API schema) - not official bracket
            # numbers, so report the tier by name and let the advisor place the deck.
            tag_names = {"E": "Exhibition", "C": "Core", "O": "Oddball", "P": "Powerful",
                         "S": "Spicy", "R": "Ruthless", "B": "Banned (contains a banned card)"}

            def _flagged(key):
                return [n for n in (c.get("card", {}).get("name") for c in cards if c.get(key)) if n]

            # Per-card flags (they mirror the official bracket criteria) + combo flags.
            gc_names = _flagged("gameChanger")
            banned = _flagged("banned")
            mld_cards = _flagged("massLandDenial")
            extra_turn_cards = _flagged("extraTurn")
            two_card = [c for c in combos if c.get("definitelyTwoCard") or c.get("arguablyTwoCard")]
            mld = bool(mld_cards) or any(c.get("massLandDenial") for c in combos)
            extra_turns = bool(extra_turn_cards) or any(c.get("extraTurn") or c.get("skipTurns") for c in combos)
            locks = any(c.get("lock") or c.get("controlAllOpponents") for c in combos)

            total = sum(e.get("quantity", 1) for e in main)
            lines = [f"**Commander Spellbook power tier: {tag} - {tag_names.get(tag, 'unknown')}** ({total} cards). "
                     "Spellbook's own scale (Exhibition < Core < Oddball < Powerful < Spicy < Ruthless) - "
                     "NOT an official bracket number."]
            lines.append(f"Game changers (official list): {len(gc_names)}"
                         + (f" - {', '.join(gc_names)}" if gc_names else ""))
            if banned:
                lines.append(f"BANNED in Commander: {', '.join(banned)}")

            lines.append(f"Combos detected: {len(combos)} (two-card: {len(two_card)})")
            flags = []
            if mld:
                flags.append("mass land denial" + (f" ({', '.join(mld_cards)})" if mld_cards else ""))
            if extra_turns:
                flags.append("extra turns" + (f" ({', '.join(extra_turn_cards)})" if extra_turn_cards else ""))
            if locks:
                flags.append("lock / control-all-opponents")
            if flags:
                lines.append("Bracket-raising elements present: " + ", ".join(flags))

            lines.append("\n(Weigh these signals against the official bracket criteria to place the deck.)")
            return "\n".join(lines)

        except httpx.HTTPStatusError as e:
            return f"Error estimating bracket: {e.response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"


# Map tool names to functions
TOOL_FUNCTIONS = {
    "scryfall_search_cards": scryfall_search_cards,
    "scryfall_get_card": scryfall_get_card,
    "scryfall_get_rulings": scryfall_get_rulings,
    "spellbook_search_combos": spellbook_search_combos,
    "spellbook_find_combos_for_cards": spellbook_find_combos_for_cards,
    "spellbook_find_combos_in_decklist": spellbook_find_combos_in_decklist,
    "spellbook_estimate_bracket": spellbook_estimate_bracket,
    "scryfall_get_decklist_details": scryfall_get_decklist_details,
    "mtg_rules_search": mtg_rules_search,
}
