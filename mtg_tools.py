"""
MTG tool implementations shared by the Discord bot and the deckbuilding advisor
web app (and, for helpers, the MCP server). Single source of truth for the
Scryfall / Commander Spellbook / rules-RAG tool handlers plus their Anthropic
tool schemas (TOOLS) and the name->function map (TOOL_FUNCTIONS).
"""

import asyncio
import json
import httpx
from pathlib import Path
from typing import Optional

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
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
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
        "description": "Fetch the ACTUAL oracle text, type line, mana cost and color identity for every card in a pasted decklist, in one batch. Call this FIRST when reviewing a decklist so your evaluation is grounded in what the cards really do - do NOT guess a card's function from its name, especially for crossover/Universes Beyond/precon/obscure cards where your memory is often wrong.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decklist_text": {
                    "type": "string",
                    "description": "Pasted decklist as text, one card per line (quantity optional)."
                }
            },
            "required": ["decklist_text"]
        }
    },
    {
        "name": "spellbook_estimate_bracket",
        "description": "Estimate the Commander bracket (power level 1-4) for a decklist based on its combos. Bracket 1 = Casual, Bracket 2 = Precon-appropriate, Bracket 3 = Powerful, Bracket 4 = Ruthless/cEDH.",
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
                params={"q": q},
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
                lines.append(f"**{name}** {mana} - {type_line} [id:{ci}]")

            return "\n".join(lines)

        except httpx.HTTPStatusError as e:
            return f"Search error: {e.response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"


async def scryfall_get_card(name: str, commander_identity: str = None) -> str:
    """
    Look up a specific card by name. When commander_identity is provided, the
    result includes an explicit color-identity legality verdict for that deck.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{SCRYFALL_API}/cards/named",
                params={"fuzzy": name},
                headers=SCRYFALL_HEADERS,
                timeout=30.0
            )
            response.raise_for_status()
            card = response.json()
            
            lines = []
            
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


def _parse_decklist_to_main(text: str) -> list[dict]:
    """
    Parse a pasted decklist into the Commander Spellbook 'main' format:
    [{"card": name, "quantity": n}, ...].

    Handles lines like "1 Sol Ring", "12 Plains", "Sol Ring", and strips
    trailing set/collector annotations like " (C21) 263". Section headers
    (Commander, Deck, Mainboard, etc.) are skipped.
    """
    main = []
    skip = {"commander", "deck", "mainboard", "sideboard", "companion", "maybeboard"}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().rstrip(":") in skip:
            continue
        qty = 1
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].rstrip("xX").isdigit():
            qty = int(parts[0].rstrip("xX"))
            name = parts[1].strip()
        else:
            name = line
        # Drop trailing set-code / collector-number annotations, e.g. "Sol Ring (C21) 263"
        if " (" in name:
            name = name.split(" (", 1)[0].strip()
        if name:
            main.append({"card": name, "quantity": qty})
    return main


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


async def _decklist_to_main(client, decklist_url, decklist_text):
    """Resolve a decklist (pasted text or URL) into the 'main' payload list."""
    if decklist_text:
        return _parse_decklist_to_main(decklist_text)
    # URL import via Spellbook (best effort; endpoint may be unavailable)
    resp = await client.post(
        f"{SPELLBOOK_API}/card-list-from-url/",
        json={"url": decklist_url},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"card": (c.get("name") if isinstance(c, dict) else str(c)), "quantity": 1}
        for c in data.get("cards", [])
    ]


async def scryfall_get_decklist_details(decklist_text: str = None) -> str:
    """
    Fetch the ACTUAL oracle text, type, mana cost and color identity for every card
    in a pasted decklist, in one batch (Scryfall /cards/collection). Lets the advisor
    reason from what cards really do instead of guessing from names.
    """
    if not decklist_text:
        return "Provide the decklist as pasted text (one card per line) to read its cards."
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
            batch = [{"name": n} for n in names[i:i + 75]]
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
        return f"[{ci}] {name} {cost} - {tl}{pt}: {' '.join(ot.split())}"

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
            except Exception:
                return ("I couldn't import that deck URL. Please paste the decklist "
                        "text instead (one card per line).")
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

            lines = [f"Analyzed {len(main)} cards (color identity {identity})."]
            if included:
                lines.append(f"\n**Complete combos already in the deck ({len(included)}):**")
                for combo in included[:limit]:
                    lines.append(f"• {_format_combo(combo)}")

            remaining = max(0, limit - len(included))
            if almost and remaining > 0:
                lines.append(f"\n**Almost there (missing 1-2 pieces) ({len(almost)}):**")
                for combo in almost[:remaining]:
                    lines.append(f"• {_format_combo(combo)}")

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
            except Exception:
                return ("I couldn't import that deck URL. Please paste the decklist "
                        "text instead (one card per line).")
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
            gc_cards = data.get("cards", []) or []
            combos = data.get("combos", []) or []

            # Spellbook single-letter bracket tags -> official bracket names
            tag_names = {
                "E": "1 - Exhibition",
                "C": "2 - Core",
                "U": "3 - Upgraded",
                "O": "4 - Optimized / cEDH",
            }

            # Evidence pulled straight from the combo flags (these mirror the
            # official bracket criteria, so the advisor can place the deck).
            two_card = [c for c in combos if c.get("definitelyTwoCard") or c.get("arguablyTwoCard")]
            mld = any(c.get("massLandDenial") for c in combos)
            extra_turns = any(c.get("extraTurn") or c.get("skipTurns") for c in combos)
            locks = any(c.get("lock") or c.get("controlAllOpponents") for c in combos)

            lines = [f"**Spellbook bracket estimate: {tag_names.get(tag, tag)}** ({len(main)} cards)"]

            gc_names = [c.get("card", {}).get("name") for c in gc_cards if isinstance(c, dict)]
            gc_names = [n for n in gc_names if n]
            lines.append(f"Game-changer / notable cards: {len(gc_names)}")
            if gc_names:
                lines.append("  " + ", ".join(gc_names[:15]))

            lines.append(f"Combos detected: {len(combos)} (two-card infinite-style: {len(two_card)})")
            flags = []
            if mld:
                flags.append("mass land denial")
            if extra_turns:
                flags.append("extra turns")
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
