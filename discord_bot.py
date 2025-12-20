"""
MTG Discord Bot
===============
A Discord bot that uses Claude AI to answer Magic: The Gathering questions.
Claude can search cards on Scryfall, find combos on Commander Spellbook,
and look up rules from the Comprehensive Rules.

Setup:
1. pip install -r requirements-discord.txt
2. Create a Discord bot at https://discord.com/developers/applications
3. Get a Claude API key from https://console.anthropic.com
4. Set environment variables (see below)
5. python discord_bot.py

Environment Variables:
    DISCORD_BOT_TOKEN - Your Discord bot token
    ANTHROPIC_API_KEY - Your Claude API key

Discord Bot Permissions Needed:
    - Read Messages/View Channels
    - Send Messages
    - Read Message History
"""

import os
import asyncio
import json
from typing import Optional
import discord
from discord.ext import commands
import anthropic
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Looks for .env in the same directory as this script
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Bot responds to @mentions or messages starting with this prefix
COMMAND_PREFIX = "!mtg "

# Which Claude model to use (claude-sonnet-4-20250514 is fast and capable)
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Maximum tokens for Claude's response
MAX_TOKENS = 1024

# Path to the rules database (same as MCP server)
RULES_DB_PATH = Path(__file__).parent / "mtg_rules_data"

# System prompt that tells Claude how to behave
SYSTEM_PROMPT = """You are an expert Magic: The Gathering judge assistant in a Discord server. 
You help players with card lookups, combo searches, rules questions, and deck power level analysis.

You have access to these tools:
- scryfall_search_cards: Search for MTG cards using Scryfall syntax
- scryfall_get_card: Look up a specific card by name
- scryfall_get_rulings: Get official rulings for a card
- spellbook_search_combos: Search for Commander/EDH combos
- spellbook_find_combos_for_cards: Find combos containing specific cards
- spellbook_find_combos_in_decklist: Analyze a full decklist (URL or pasted) for combos
- spellbook_estimate_bracket: Estimate the Commander bracket (power level 1-4) for a deck
- mtg_rules_search: Search the Comprehensive Rules (if database is available)

CRITICAL - ALWAYS GATHER SUFFICIENT CONTEXT:
You MUST use multiple tools before answering rules questions. A single tool call is rarely enough, you have 12 before the response is stopped.

For ANY rules question or card interaction:
1. Call mtg_rules_search with the core mechanic/question
2. Call scryfall_get_card for EACH card mentioned
3. Call scryfall_get_rulings for EACH card mentioned
4. If a keyword ability is involved, also call scryfall_get_rulings with the keyword name

DO NOT answer a rules question unless sufficient context has been gathered.

Example - "Does Blade of Selves cause additional attack triggers?":
- scryfall_get_card("Blade of Selves")
- scryfall_get_rulings("Blade of Selves")
- scryfall_get_rulings("equip")
- mtg_rules_search("myriad tokens attacking declared attackers")
- scryfall_get_rulings("myriad")

Example - "How does Panharmonicon work with Solemn Simulacrum?":
- mtg_rules_search("enters the battlefield triggered abilities")
- scryfall_get_card("Panharmonicon")
- scryfall_get_card("Solemn Simulacrum")  
- scryfall_get_rulings("Panharmonicon")
- scryfall_get_rulings("Solemn Simulacrum")

COMMANDER BRACKET SYSTEM (official WotC system):
- Bracket 1: Exhibition
    Players expect:
    Decks to prioritize a goal, theme, or idea over power 
    Rules around card legality or viable commanders to have some flexibility depending on the pod 
    Win conditions to be highly thematic or substandard 
    Gameplay to be an opportunity to show off creations 
    At least nine turns before a win or loss.
    No mass land denial
    Only thematic game changers (if any)

- Bracket 2: Core
    Players expect:
    Decks to be unoptimized and straightforward, with some cards chosen to maximize creativity and/or entertainment 
    Win conditions to be incremental, telegraphed on the board, and disruptable 
    Gameplay to be low pressure with an emphasis on social interaction 
    Gameplay to be proactive and considerate, letting each deck showcase its plan 
    At least eight turns before a win or loss.
    No mass land denial, chaining extra turns, or two card infinite combos
    No game changers

- Bracket 3: Upgraded
    Players expect:
    Decks to be powered up with strong synergy and high card quality; they can effectively disrupt opponents 
    Game Changers that are likely to be value engines and game-ending spells 
    Win conditions that can be deployed in one big turn from hand, usually because of steadily accrued resources 
    Gameplay to feature many proactive and reactive plays 
    At least six turns before a win or loss.
    No mass land denial, chaining extra turns, or early two card infinite combos
    Up to 3 game changers

- Bracket 4: Optimized / cEDH
    Players expect:
    Decks to be lethal, consistent, and fast, designed to take people down as fast as possible 
    Game Changers that are likely to be fast mana, snowballing resource engines, free disruption, and tutors 
    Win conditions to vary but be efficient and instantaneous 
    Gameplay to be explosive and powerful, featuring huge threats and efficient disruption to match 
    Anything goes
    No game changers restrictions

Key bracket factors:
- You can query scryfall with is:gamechanger to identify game changers
- Two-card infinite combos push decks toward Bracket 3-4
- Mass land destruction, extra turns, and stax effects raise bracket
- Combo piece count and combo speed matter
- Fast mana (Mana Crypt, Mox Diamond, etc.) raises bracket

IMPORTANT - When answering rules questions:
1. Look up any specific cards mentioned with scryfall_get_card to see exact oracle text first
2. Follow up by searching the rules with mtg_rules_search first
3. Consider if a line of play requires an impossible game state, like a creature existing with 0 or less toughness, or negative life total without a replacement effect
4. Ensure targets are legal for effects, for instance you cannot equip a creature with zero toughness as it dies before becoming targetable
5. Some triggers can only occur during specific phases or steps of a turn, like attack triggers requiring declared attackers (508.3a), check those are valid triggers
6. Take your time analyzing timings and interactions carefully, it matters exactly when things happen
7. Check scryfall_get_rulings for official clarifications on those cards, put weight behind ALL official rulings, even if they dont seem relevant at first
8. Cite rule numbers in your answer when possible

CRITICAL SBA handling:
State based actions DO NOT go on the stack and must be checked after EACH object resolves on the stack.
This means: 
- If a spell or ability reduces a creature to 0 toughness, like skullclamp, it dies BEFORE you can respond with that creature's abilities
- You cannot "respond" to a creature dying from SBAs - there is no priority window between the SBA check and the death


KEY MTG RULES PRINCIPLES (use these to verify your answers):
- Abilities on the stack exist independently of their source (killing a creature doesn't counter its ability)
- "Dies" and "leaves the battlefield" triggers see the game state right BEFORE the event
- Colorless is NOT a color - cards can't "share a color" if they're both colorless
- Summoning sickness checks if YOU'VE controlled the creature since your turn began, not when it ETB'd
- Replacement effects modify events as they happen - they don't use the stack
- State-based actions don't use the stack and can't be responded to
- "As [this] enters" and "enters with" are replacement effects, not triggered abilities
- Activated abilities are written as "[cost]: [effect]" - the colon is the giveaway
- Triggered abilities start with "when", "whenever", or "at"
- Each activation of an ability that grants an ability STACKS (e.g., activating a manland's ability twice gives two instances of any granted triggered abilities)
- The stack resolves top-down, but state-based actions are checked after EACH object resolves
- Equip effects are sorcery speed unless otherwise specified, and go on the stack
- "Target" is a magic word - if a spell/ability doesn't say "target", it doesn't target


Keep responses concise since this is Discord - aim for under 2000 characters.
Use markdown formatting sparingly. Don't use headers (##) in Discord.
When showing card info, focus on the most relevant details.
"""



# =============================================================================
# API CLIENTS AND SETUP
# =============================================================================

# Scryfall API settings (same as MCP server)
SCRYFALL_API = "https://api.scryfall.com"
SCRYFALL_HEADERS = {
    "User-Agent": "MTG-Discord-Bot/1.0",
    "Accept": "application/json"
}

# Commander Spellbook API
SPELLBOOK_API = "https://backend.commanderspellbook.com"

# Initialize the Anthropic client (reads ANTHROPIC_API_KEY from environment)
claude_client = anthropic.Anthropic()

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
        "description": "Search for Magic: The Gathering cards using Scryfall's search syntax. Use operators like c: (color), t: (type), o: (oracle text), cmc: (mana value), pow: (power), id: (color identity for Commander).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Scryfall search query. Examples: 'c:blue t:creature', 'o:\"draw a card\" cmc<=3', 'id:simic t:legendary'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (1-10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "scryfall_get_card",
        "description": "Look up a specific Magic: The Gathering card by name. Supports fuzzy matching for typos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Card name to look up"
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

async def scryfall_search_cards(query: str, limit: int = 5) -> str:
    """Search for cards on Scryfall."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{SCRYFALL_API}/cards/search",
                params={"q": query},
                headers=SCRYFALL_HEADERS,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            cards = data.get("data", [])[:limit]
            total = data.get("total_cards", len(cards))
            
            if not cards:
                return "No cards found matching that search."
            
            # Format results concisely for Discord
            lines = [f"Found {total} cards (showing {len(cards)}):"]
            for card in cards:
                name = card.get("name", "Unknown")
                mana = card.get("mana_cost", "")
                type_line = card.get("type_line", "")
                lines.append(f"**{name}** {mana} - {type_line}")
            
            return "\n".join(lines)
            
        except httpx.HTTPStatusError as e:
            return f"Search error: {e.response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"


async def scryfall_get_card(name: str) -> str:
    """Look up a specific card by name."""
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
            
            # Format card info for Discord
            lines = []
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
            
            # Price
            prices = card.get("prices", {})
            usd = prices.get("usd")
            if usd:
                lines.append(f"Price: ${usd}")
            
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


async def spellbook_find_combos_in_decklist(
    decklist_url: str = None, 
    decklist_text: str = None, 
    limit: int = 10
) -> str:
    """
    Find all combos present in a decklist.
    Can accept either a URL to a deck or pasted card list.
    """
    async with httpx.AsyncClient() as client:
        try:
            cards = []
            
            # Option 1: Get cards from a deck URL
            if decklist_url:
                response = await client.post(
                    f"{SPELLBOOK_API}/card-list-from-url/",
                    json={"url": decklist_url},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract card names from the response
                card_list = data.get("cards", [])
                for card in card_list:
                    if isinstance(card, dict):
                        cards.append(card.get("name", card.get("card", "")))
                    else:
                        cards.append(str(card))
            
            # Option 2: Parse pasted decklist text
            elif decklist_text:
                response = await client.post(
                    f"{SPELLBOOK_API}/card-list-from-text/",
                    json={"text": decklist_text},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                card_list = data.get("cards", [])
                for card in card_list:
                    if isinstance(card, dict):
                        cards.append(card.get("name", card.get("card", "")))
                    else:
                        cards.append(str(card))
            
            else:
                return "Please provide either a decklist URL or pasted card list."
            
            if not cards:
                return "Couldn't extract any cards from that decklist."
            
            # Now find combos using the find-my-combos endpoint
            response = await client.post(
                f"{SPELLBOOK_API}/find-my-combos/",
                json={"cards": cards},
                timeout=60.0  # Can be slow for large decklists
            )
            response.raise_for_status()
            data = response.json()
            
            # The response contains 'results' with 'included' combos
            results = data.get("results", {})
            included = results.get("included", [])
            almost = results.get("almost_included", [])
            
            if not included and not almost:
                return f"No combos found in this deck ({len(cards)} cards analyzed)."
            
            lines = [f"Analyzed {len(cards)} cards:"]
            
            # Show fully included combos first
            if included:
                lines.append(f"\n**Complete combos in deck ({len(included)}):**")
                for combo in included[:limit]:
                    uses = combo.get("uses", [])
                    card_names = [u.get("card", {}).get("name", "?") for u in uses]
                    cards_str = " + ".join(card_names[:4])
                    if len(card_names) > 4:
                        cards_str += f" +{len(card_names)-4} more"
                    
                    produces = combo.get("produces", [])
                    results_list = [p.get("feature", {}).get("name", "") for p in produces[:2]]
                    results_str = ", ".join(results_list) if results_list else "combo"
                    
                    lines.append(f"• {cards_str} → {results_str}")
            
            # Show "almost" combos (missing 1 card) if room
            remaining = limit - len(included)
            if almost and remaining > 0:
                lines.append(f"\n**Almost complete (missing 1 card):**")
                for combo in almost[:remaining]:
                    uses = combo.get("uses", [])
                    card_names = [u.get("card", {}).get("name", "?") for u in uses]
                    cards_str = " + ".join(card_names[:4])
                    
                    # Try to identify the missing card
                    missing = combo.get("missing", [])
                    if missing:
                        missing_name = missing[0].get("card", {}).get("name", "?")
                        lines.append(f"• {cards_str} (needs: {missing_name})")
                    else:
                        lines.append(f"• {cards_str}")
            
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
    async with httpx.AsyncClient() as client:
        try:
            cards = []
            
            # Get cards from URL or text
            if decklist_url:
                response = await client.post(
                    f"{SPELLBOOK_API}/card-list-from-url/",
                    json={"url": decklist_url},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                card_list = data.get("cards", [])
                for card in card_list:
                    if isinstance(card, dict):
                        cards.append(card.get("name", card.get("card", "")))
                    else:
                        cards.append(str(card))
                        
            elif decklist_text:
                response = await client.post(
                    f"{SPELLBOOK_API}/card-list-from-text/",
                    json={"text": decklist_text},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                card_list = data.get("cards", [])
                for card in card_list:
                    if isinstance(card, dict):
                        cards.append(card.get("name", card.get("card", "")))
                    else:
                        cards.append(str(card))
            else:
                return "Please provide either a decklist URL or pasted card list."
            
            if not cards:
                return "Couldn't extract any cards from that decklist."
            
            # Call the bracket estimation endpoint
            response = await client.post(
                f"{SPELLBOOK_API}/estimate-bracket/",
                json={"cards": cards},
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse the bracket response
            bracket = data.get("bracket", "Unknown")
            
            # Get combo breakdown by bracket
            combos_by_bracket = data.get("combos_by_bracket", {})
            two_card_combos = data.get("two_card_combos", [])
            
            # Build the response
            lines = [f"**Estimated Bracket: {bracket}**"]
            lines.append(f"Cards analyzed: {len(cards)}")
            
            # Bracket descriptions
            bracket_desc = {
                "1": "Exhibition - Thematic, creative, 9+ turns expected",
                "2": "Core - Unoptimized, social, no two-card infinites",
                "3": "Upgraded - Strong synergy, up to 3 game changers", 
                "4": "Optimized/cEDH - Lethal, consistent, anything goes"
            }
            
            bracket_num = str(bracket).split()[0] if bracket else "?"
            if bracket_num in bracket_desc:
                lines.append(f"*{bracket_desc[bracket_num]}*")
            
            # Show two-card combos if any (these heavily influence bracket)
            if two_card_combos:
                lines.append(f"\n**Two-card combos found ({len(two_card_combos)}):**")
                for combo in two_card_combos[:5]:
                    cards_in_combo = combo.get("uses", [])
                    card_names = [c.get("card", {}).get("name", "?") for c in cards_in_combo]
                    combo_bracket = combo.get("bracket", "?")
                    lines.append(f"• {' + '.join(card_names)} (Bracket {combo_bracket})")
                
                if len(two_card_combos) > 5:
                    lines.append(f"  ...and {len(two_card_combos) - 5} more")
            
            # Show combo count by bracket level
            if combos_by_bracket:
                lines.append("\n**Combos by bracket level:**")
                for b_level, combos in sorted(combos_by_bracket.items()):
                    count = len(combos) if isinstance(combos, list) else combos
                    lines.append(f"• Bracket {b_level}: {count} combos")
            
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
    "mtg_rules_search": mtg_rules_search,
}


# =============================================================================
# CLAUDE INTERACTION
# =============================================================================

async def ask_claude(user_message: str) -> str:
    """
    Send a message to Claude and handle any tool calls.
    
    This implements a tool-use loop:
    1. Send the user's message to Claude
    2. If Claude wants to use a tool, execute it and send results back
    3. Repeat until Claude gives a final text response
    """
    messages = [{"role": "user", "content": user_message}]
    
    # Loop to handle multiple tool calls if needed
    max_iterations = 12  # Safety limit
    for _ in range(max_iterations):
        # Call Claude
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Find all tool use blocks in the response
            tool_results = []
            assistant_content = response.content
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id
                    
                    # Execute the tool
                    print(f"Executing tool: {tool_name} with {tool_input}")
                    
                    if tool_name in TOOL_FUNCTIONS:
                        func = TOOL_FUNCTIONS[tool_name]
                        # Call the async function with the provided arguments
                        result = await func(**tool_input)
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result
                    })
            
            # Add assistant's response and tool results to messages
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        
        else:
            # Claude is done - extract the text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            
            return "I couldn't generate a response."
    
    return "I got stuck in a loop trying to answer. Please try rephrasing your question."


# =============================================================================
# DISCORD BOT
# =============================================================================

# Set up Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)


@bot.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""
    print(f"Bot is ready! Logged in as {bot.user}")
    print(f"Invite URL: https://discord.com/api/oauth2/authorize?client_id={bot.user.id}&permissions=274877908992&scope=bot")
    
    # Pre-load the rules database in the background
    # This prevents the first rules query from blocking for 10-20 seconds
    if RULES_DB_PATH.exists():
        print("Pre-loading rules database in background...")
        asyncio.create_task(preload_rules_database())


async def preload_rules_database():
    """Background task to pre-load the rules database on startup."""
    try:
        collection = await get_rules_collection_async()
        if collection:
            print("Rules database loaded successfully!")
        else:
            print("Rules database not found or failed to load.")
    except Exception as e:
        print(f"Error pre-loading rules database: {e}")


@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages."""
    # Don't respond to ourselves
    if message.author == bot.user:
        return
    
    # Check if the bot was mentioned or the message starts with the prefix
    should_respond = False
    user_content = message.content
    
    # Check for @mention
    if bot.user.mentioned_in(message):
        should_respond = True
        # Remove the mention from the message
        user_content = message.content.replace(f"<@{bot.user.id}>", "").strip()
        user_content = user_content.replace(f"<@!{bot.user.id}>", "").strip()
    
    # Check for command prefix
    elif message.content.startswith(COMMAND_PREFIX):
        should_respond = True
        user_content = message.content[len(COMMAND_PREFIX):].strip()
    
    if not should_respond or not user_content:
        return
    
    # Show typing indicator while processing
    async with message.channel.typing():
        try:
            # Get response from Claude
            response = await ask_claude(user_content)
            
            # Discord has a 2000 character limit
            if len(response) > 2000:
                # Split into multiple messages if needed
                chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                for chunk in chunks:
                    await message.reply(chunk)
            else:
                await message.reply(response)
                
        except anthropic.APIError as e:
            await message.reply(f"Sorry, I encountered an API error: {str(e)}")
        except Exception as e:
            print(f"Error: {e}")
            await message.reply("Sorry, something went wrong while processing your question.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    # Check for required environment variables
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not discord_token:
        print("Error: DISCORD_BOT_TOKEN environment variable not set")
        print("\nTo set it:")
        print("  Windows: set DISCORD_BOT_TOKEN=your_token_here")
        print("  Linux/Mac: export DISCORD_BOT_TOKEN=your_token_here")
        return
    
    if not anthropic_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  Windows: set ANTHROPIC_API_KEY=your_key_here")
        print("  Linux/Mac: export ANTHROPIC_API_KEY=your_key_here")
        return
    
    print("Starting MTG Discord Bot...")
    print(f"Command prefix: {COMMAND_PREFIX}")
    print("The bot also responds to @mentions")
    
    # Check if rules database exists
    if RULES_DB_PATH.exists():
        print("Rules database: Found ✓")
    else:
        print("Rules database: Not found (run rules_ingestion.py to enable rules search)")
    
    print()
    
    # Run the bot
    bot.run(discord_token)


if __name__ == "__main__":
    main()