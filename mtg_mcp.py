"""
MTG MCP Server
==============
An MCP server that provides tools for querying Magic: The Gathering data
from Scryfall (card database), Commander Spellbook (combo database),
and the MTG Comprehensive Rules (via local RAG).

This server enables Claude to help with:
- Card lookups and searches using Scryfall's powerful syntax
- Finding combos for Commander/EDH decks
- Checking card rulings and legality
- Answering rules questions using the Comprehensive Rules

Setup:
1. pip install -r requirements.txt
2. python rules_ingestion.py  (to download and index the rules)
3. Add to Claude Desktop config
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List
from enum import Enum
from pathlib import Path
import httpx
import json
import asyncio

# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

mcp = FastMCP("mtg_mcp")

# API base URLs
SCRYFALL_API = "https://api.scryfall.com"
SPELLBOOK_API = "https://backend.commanderspellbook.com"

# Required headers for Scryfall (they require User-Agent)
SCRYFALL_HEADERS = {
    "User-Agent": "MTG-MCP-Server/1.0",
    "Accept": "application/json"
}

# Path to the rules database (created by rules_ingestion.py)
RULES_DB_PATH = Path(__file__).parent / "mtg_rules_data"


# =============================================================================
# RULES DATABASE SETUP (Lazy Loading)
# =============================================================================

# We use lazy loading so the server starts fast even if the DB isn't ready
_rules_collection = None

def get_rules_collection():
    """
    Lazily loads the ChromaDB collection for rules search.
    
    Returns None if the database hasn't been created yet.
    This allows the server to run even without the rules database.
    """
    global _rules_collection
    
    # Return cached collection if we already loaded it
    if _rules_collection is not None:
        return _rules_collection
    
    # Check if the database exists
    if not RULES_DB_PATH.exists():
        return None
    
    try:
        # Import ChromaDB only when needed
        import chromadb
        from chromadb.utils import embedding_functions
        
        # Connect to the existing database
        client = chromadb.PersistentClient(path=str(RULES_DB_PATH))
        
        # Set up the same embedding function used during ingestion
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get the collection
        _rules_collection = client.get_collection(
            name="mtg_comprehensive_rules",
            embedding_function=embedding_func
        )
        
        return _rules_collection
        
    except Exception as e:
        # If anything goes wrong, just return None
        # The tool will show an appropriate error message
        print(f"Warning: Could not load rules database: {e}", flush=True)
        return None


# =============================================================================
# SHARED UTILITIES
# =============================================================================

async def make_scryfall_request(endpoint: str, params: dict = None) -> dict:
    """
    Makes a request to the Scryfall API with proper headers and rate limiting.
    
    Scryfall asks for 50-100ms between requests, so we add a small delay.
    This helper centralizes error handling for all Scryfall calls.
    """
    async with httpx.AsyncClient() as client:
        try:
            # Build the full URL
            url = f"{SCRYFALL_API}{endpoint}"
            
            # Make the request with required headers
            response = await client.get(
                url,
                params=params,
                headers=SCRYFALL_HEADERS,
                timeout=30.0
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            # Return structured error info
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": _parse_scryfall_error(e.response)
            }
        except httpx.TimeoutException:
            return {"error": True, "message": "Request timed out. Please try again."}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {str(e)}"}


async def make_spellbook_request(endpoint: str, params: dict = None) -> dict:
    """
    Makes a request to the Commander Spellbook API.
    
    This helper centralizes error handling for all Spellbook calls.
    """
    async with httpx.AsyncClient() as client:
        try:
            url = f"{SPELLBOOK_API}{endpoint}"
            
            response = await client.get(
                url,
                params=params,
                timeout=30.0
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": f"API error: {e.response.status_code}"
            }
        except httpx.TimeoutException:
            return {"error": True, "message": "Request timed out. Please try again."}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {str(e)}"}


async def make_spellbook_post(endpoint: str, data: dict) -> dict:
    """
    Makes a POST request to Commander Spellbook API.
    
    Some endpoints like find-my-combos require POST with JSON body.
    """
    async with httpx.AsyncClient() as client:
        try:
            url = f"{SPELLBOOK_API}{endpoint}"
            
            response = await client.post(
                url,
                json=data,
                timeout=60.0  # Longer timeout for combo searches
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": f"API error: {e.response.status_code}"
            }
        except httpx.TimeoutException:
            return {"error": True, "message": "Request timed out. Please try again."}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {str(e)}"}


def _parse_scryfall_error(response) -> str:
    """
    Extracts a human-readable error message from Scryfall error responses.
    """
    try:
        data = response.json()
        # Scryfall returns errors with 'details' field
        return data.get("details", f"HTTP {response.status_code}")
    except:
        return f"HTTP {response.status_code}"


def format_card_markdown(card: dict) -> str:
    """
    Formats a Scryfall card object as readable markdown.
    
    This helper extracts the most useful info and presents it clearly.
    """
    lines = []
    
    # Card name and mana cost
    name = card.get("name", "Unknown")
    mana_cost = card.get("mana_cost", "")
    lines.append(f"## {name} {mana_cost}")
    
    # Type line
    type_line = card.get("type_line", "")
    if type_line:
        lines.append(f"**{type_line}**")
    
    # Oracle text (the rules text on the card)
    oracle_text = card.get("oracle_text", "")
    if oracle_text:
        lines.append(f"\n{oracle_text}")
    
    # Power/Toughness for creatures
    power = card.get("power")
    toughness = card.get("toughness")
    if power and toughness:
        lines.append(f"\n**P/T:** {power}/{toughness}")
    
    # Loyalty for planeswalkers
    loyalty = card.get("loyalty")
    if loyalty:
        lines.append(f"\n**Starting Loyalty:** {loyalty}")
    
    # Set and rarity
    set_name = card.get("set_name", "")
    rarity = card.get("rarity", "").capitalize()
    if set_name:
        lines.append(f"\n*{set_name} ({rarity})*")
    
    # Legalities (just show Commander since that's our focus)
    legalities = card.get("legalities", {})
    commander_legal = legalities.get("commander", "unknown")
    lines.append(f"\n**Commander Legal:** {commander_legal}")
    
    # Price info (USD)
    prices = card.get("prices", {})
    usd = prices.get("usd")
    usd_foil = prices.get("usd_foil")
    if usd or usd_foil:
        price_parts = []
        if usd:
            price_parts.append(f"${usd}")
        if usd_foil:
            price_parts.append(f"${usd_foil} foil")
        lines.append(f"**Price:** {' / '.join(price_parts)}")
    
    # Scryfall URL for more details
    scryfall_uri = card.get("scryfall_uri", "")
    if scryfall_uri:
        lines.append(f"\n[View on Scryfall]({scryfall_uri})")
    
    return "\n".join(lines)


def format_combo_markdown(combo: dict) -> str:
    """
    Formats a Commander Spellbook combo as readable markdown.
    """
    lines = []
    
    # Combo ID for reference
    combo_id = combo.get("id", "unknown")
    lines.append(f"## Combo #{combo_id}")
    
    # Cards involved
    uses = combo.get("uses", [])
    if uses:
        card_names = [u.get("card", {}).get("name", "Unknown") for u in uses]
        lines.append(f"\n**Cards:** {', '.join(card_names)}")
    
    # Color identity
    identity = combo.get("identity", "")
    if identity:
        lines.append(f"**Color Identity:** {identity}")
    
    # Prerequisites (what you need to set up)
    requires = combo.get("requires", [])
    if requires:
        lines.append("\n**Prerequisites:**")
        for req in requires:
            template = req.get("template", {})
            name = template.get("name", "")
            if name:
                lines.append(f"- {name}")
    
    # Steps to execute
    description = combo.get("description", "")
    if description:
        lines.append(f"\n**Steps:**\n{description}")
    
    # Results (what the combo does)
    produces = combo.get("produces", [])
    if produces:
        lines.append("\n**Results:**")
        for prod in produces:
            feature = prod.get("feature", {})
            name = feature.get("name", "")
            if name:
                lines.append(f"- {name}")
    
    # Bracket/power level
    bracket = combo.get("bracket", "")
    if bracket:
        lines.append(f"\n**Bracket:** {bracket}")
    
    # Link to Commander Spellbook
    lines.append(f"\n[View on Commander Spellbook](https://commanderspellbook.com/combo/{combo_id})")
    
    return "\n".join(lines)


# =============================================================================
# INPUT MODELS (Pydantic models for input validation)
# =============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class ScryfallSearchInput(BaseModel):
    """
    Input for searching cards on Scryfall.
    
    The query uses Scryfall's search syntax which is very powerful.
    Examples:
    - 'c:blue t:creature' = blue creatures
    - 'o:draw cmc<=2' = cards with 'draw' in text, CMC 2 or less
    - 'id:simic t:legendary' = Simic identity legendary cards
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    query: str = Field(
        ...,
        description=(
            "Search query using Scryfall syntax. Examples: "
            "'c:green t:creature pow>=5' (green creatures with 5+ power), "
            "'o:\"draw a card\" id:izzet' (Izzet cards with draw effects), "
            "'t:legendary t:creature id:simic' (Simic legendary creatures)"
        ),
        min_length=1,
        max_length=500
    )
    
    limit: int = Field(
        default=10,
        description="Maximum number of results to return (1-50)",
        ge=1,
        le=50
    )
    
    order: Optional[str] = Field(
        default=None,
        description=(
            "Sort order: 'name', 'released', 'set', 'rarity', 'color', "
            "'usd', 'cmc', 'power', 'toughness', 'edhrec' (by EDHREC rank)"
        )
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )


class ScryfallNamedInput(BaseModel):
    """Input for looking up a specific card by name."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(
        ...,
        description="Card name to look up (e.g., 'Lightning Bolt', 'Rhystic Study')",
        min_length=1,
        max_length=200
    )
    
    fuzzy: bool = Field(
        default=True,
        description=(
            "If True, allows fuzzy matching for typos/partial names. "
            "If False, requires exact name match."
        )
    )
    
    set_code: Optional[str] = Field(
        default=None,
        description="Optional 3-letter set code to get a specific printing (e.g., 'mh2', 'cmr')"
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )


class ScryfallRandomInput(BaseModel):
    """Input for getting a random card."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    query: Optional[str] = Field(
        default=None,
        description=(
            "Optional Scryfall query to filter random selection. "
            "Example: 't:creature c:red' for a random red creature"
        ),
        max_length=500
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )


class ScryfallRulingsInput(BaseModel):
    """Input for getting card rulings."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    card_name: str = Field(
        ...,
        description="Name of the card to get rulings for",
        min_length=1,
        max_length=200
    )


class SpellbookSearchInput(BaseModel):
    """
    Input for searching combos on Commander Spellbook.
    
    You can search by card names, color identity, or combo results.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    query: str = Field(
        ...,
        description=(
            "Search query. Can include card names, effects, or use their syntax: "
            "'card:\"Thassa's Oracle\"' for combos with that card, "
            "'result:infinite' for infinite combos"
        ),
        min_length=1,
        max_length=500
    )
    
    color_identity: Optional[str] = Field(
        default=None,
        description=(
            "Filter by color identity using WUBRG letters. "
            "Examples: 'UB' for Dimir, 'GUR' for Temur, 'WUBRG' for 5-color"
        )
    )
    
    limit: int = Field(
        default=10,
        description="Maximum number of combos to return (1-50)",
        ge=1,
        le=50
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )


class SpellbookFindCombosInput(BaseModel):
    """
    Input for finding combos that use specific cards.
    
    Provide a list of card names and find all combos that can be made with them.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    cards: List[str] = Field(
        ...,
        description=(
            "List of card names to find combos for. "
            "Example: ['Thassa\\'s Oracle', 'Demonic Consultation']"
        ),
        min_length=1,
        max_length=100
    )
    
    limit: int = Field(
        default=10,
        description="Maximum number of combos to return (1-50)",
        ge=1,
        le=50
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )
    
    @field_validator('cards')
    @classmethod
    def validate_cards(cls, v: List[str]) -> List[str]:
        """Ensure card names are not empty strings."""
        return [card.strip() for card in v if card.strip()]


class SpellbookGetComboInput(BaseModel):
    """Input for getting details of a specific combo by ID."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    combo_id: str = Field(
        ...,
        description="The Commander Spellbook combo ID",
        min_length=1
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )


class SpellbookDecklistInput(BaseModel):
    """
    Input for finding combos in a decklist.
    
    Accepts either a URL to a deck or pasted card names.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    decklist_url: Optional[str] = Field(
        default=None,
        description="URL to a decklist (Moxfield, Archidekt, Deckstats, TappedOut, etc.)"
    )
    
    decklist_text: Optional[str] = Field(
        default=None,
        description="Pasted decklist - one card per line, quantity optional (e.g., '1 Sol Ring' or just 'Sol Ring')"
    )
    
    limit: int = Field(
        default=10,
        description="Maximum number of combos to return (1-20)",
        ge=1,
        le=20
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )


class SpellbookBracketInput(BaseModel):
    """
    Input for estimating a deck's Commander bracket (power level).
    
    Accepts either a URL to a deck or pasted card names.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    decklist_url: Optional[str] = Field(
        default=None,
        description="URL to a decklist (Moxfield, Archidekt, Deckstats, TappedOut, etc.)"
    )
    
    decklist_text: Optional[str] = Field(
        default=None,
        description="Pasted decklist - one card per line, quantity optional"
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for raw data"
    )


class RulesSearchInput(BaseModel):
    """
    Input for searching the MTG Comprehensive Rules.
    
    Uses semantic search to find relevant rules for your question.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    query: str = Field(
        ...,
        description=(
            "Your rules question in natural language. Examples: "
            "'When can I cast instants?', "
            "'How does summoning sickness work?', "
            "'What happens when a creature has 0 toughness?'"
        ),
        min_length=3,
        max_length=500
    )
    
    num_results: int = Field(
        default=5,
        description="Number of relevant rules to return (1-15)",
        ge=1,
        le=15
    )


# =============================================================================
# SCRYFALL TOOLS
# =============================================================================

@mcp.tool(
    name="scryfall_search_cards",
    annotations={
        "title": "Search MTG Cards on Scryfall",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def scryfall_search_cards(params: ScryfallSearchInput) -> str:
    """
    Search for Magic: The Gathering cards using Scryfall's powerful search syntax.
    
    This tool queries Scryfall's card database which contains every MTG card ever printed.
    The search syntax is very flexible - you can filter by color, type, text, CMC, and more.
    
    Common search operators:
    - c: or color: = card color (c:blue, c:UR for blue/red)
    - id: or identity: = color identity for Commander (id:simic)
    - t: or type: = card type (t:creature, t:instant)
    - o: or oracle: = oracle text contains (o:"draw a card")
    - cmc: or mv: = mana value (cmc<=3, cmc=5)
    - pow: and tou: = power/toughness (pow>=4)
    - r: or rarity: = rarity (r:mythic)
    - is:commander = can be a commander
    
    Args:
        params (ScryfallSearchInput): Search parameters including:
            - query (str): Scryfall search syntax query
            - limit (int): Max results to return (1-50)
            - order (str): Sort order (name, cmc, edhrec, etc.)
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: Formatted card results or JSON data
    """
    # Build the query parameters
    api_params = {"q": params.query}
    
    if params.order:
        api_params["order"] = params.order
    
    # Make the API request
    result = await make_scryfall_request("/cards/search", api_params)
    
    # Handle errors
    if result.get("error"):
        return f"**Error:** {result.get('message', 'Unknown error')}"
    
    # Extract the cards from the response
    cards = result.get("data", [])
    total = result.get("total_cards", len(cards))
    
    # Limit to requested number
    cards = cards[:params.limit]
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"total": total, "cards": cards}, indent=2)
    
    # Markdown format
    lines = [f"**Found {total} cards** (showing {len(cards)})\n"]
    
    for card in cards:
        lines.append(format_card_markdown(card))
        lines.append("\n---\n")
    
    return "\n".join(lines)


@mcp.tool(
    name="scryfall_get_card",
    annotations={
        "title": "Get MTG Card by Name",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def scryfall_get_card(params: ScryfallNamedInput) -> str:
    """
    Look up a specific Magic: The Gathering card by name.
    
    This is faster than searching when you know the exact card name.
    Supports fuzzy matching for typos or partial names.
    
    Args:
        params (ScryfallNamedInput): Lookup parameters including:
            - name (str): Card name to look up
            - fuzzy (bool): Allow fuzzy/partial matching
            - set_code (str): Optional set code for specific printing
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: Card details or JSON data
    """
    # Build query parameters
    # Use 'fuzzy' for approximate matching, 'exact' for precise
    param_key = "fuzzy" if params.fuzzy else "exact"
    api_params = {param_key: params.name}
    
    if params.set_code:
        api_params["set"] = params.set_code
    
    # Make the API request
    result = await make_scryfall_request("/cards/named", api_params)
    
    # Handle errors
    if result.get("error"):
        return f"**Error:** {result.get('message', 'Card not found. Try a different name or enable fuzzy matching.')}"
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    return format_card_markdown(result)


@mcp.tool(
    name="scryfall_random_card",
    annotations={
        "title": "Get Random MTG Card",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,  # Returns different results each time
        "openWorldHint": True
    }
)
async def scryfall_random_card(params: ScryfallRandomInput) -> str:
    """
    Get a random Magic: The Gathering card.
    
    Optionally filter the random selection with a Scryfall query.
    Great for discovery, challenges, or inspiration.
    
    Args:
        params (ScryfallRandomInput): Parameters including:
            - query (str): Optional Scryfall query to filter
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: Random card details or JSON data
    """
    # Build query parameters
    api_params = {}
    if params.query:
        api_params["q"] = params.query
    
    # Make the API request
    result = await make_scryfall_request("/cards/random", api_params)
    
    # Handle errors
    if result.get("error"):
        return f"**Error:** {result.get('message', 'Could not get random card')}"
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    return format_card_markdown(result)


@mcp.tool(
    name="scryfall_get_rulings",
    annotations={
        "title": "Get MTG Card Rulings",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def scryfall_get_rulings(params: ScryfallRulingsInput) -> str:
    """
    Get official rulings for a Magic: The Gathering card.
    
    Rulings are official clarifications from Wizards of the Coast about
    how a card works. Useful for understanding complex interactions.
    
    Args:
        params (ScryfallRulingsInput): Parameters including:
            - card_name (str): Name of the card to get rulings for
    
    Returns:
        str: List of rulings with dates
    """
    # First, we need to get the card to find its Scryfall ID
    card_result = await make_scryfall_request("/cards/named", {"fuzzy": params.card_name})
    
    if card_result.get("error"):
        return f"**Error:** Could not find card '{params.card_name}'"
    
    card_id = card_result.get("id")
    card_name = card_result.get("name")
    
    # Now get the rulings for this card
    rulings_result = await make_scryfall_request(f"/cards/{card_id}/rulings")
    
    if rulings_result.get("error"):
        return f"**Error:** {rulings_result.get('message', 'Could not get rulings')}"
    
    rulings = rulings_result.get("data", [])
    
    if not rulings:
        return f"**No rulings found for {card_name}.**\n\nThis card has no official rulings or clarifications."
    
    # Format the rulings
    lines = [f"## Rulings for {card_name}\n"]
    
    for ruling in rulings:
        date = ruling.get("published_at", "Unknown date")
        comment = ruling.get("comment", "")
        source = ruling.get("source", "wotc")
        
        # Format the source nicely
        source_label = "Wizards of the Coast" if source == "wotc" else source.upper()
        
        lines.append(f"**{date}** ({source_label})")
        lines.append(f"> {comment}\n")
    
    return "\n".join(lines)


# =============================================================================
# COMMANDER SPELLBOOK TOOLS
# =============================================================================

@mcp.tool(
    name="spellbook_search_combos",
    annotations={
        "title": "Search EDH Combos",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def spellbook_search_combos(params: SpellbookSearchInput) -> str:
    """
    Search for Commander/EDH combos on Commander Spellbook.
    
    Find combos by card names, effects, or color identity.
    Great for discovering new synergies for your decks.
    
    Args:
        params (SpellbookSearchInput): Search parameters including:
            - query (str): Search text (card names, effects, etc.)
            - color_identity (str): Filter by colors (e.g., 'UB', 'GUR')
            - limit (int): Max results to return (1-50)
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: List of combos or JSON data
    """
    # Build the search parameters
    # Commander Spellbook uses 'q' for the query parameter
    api_params = {"q": params.query, "limit": params.limit}
    
    if params.color_identity:
        # Format the color identity (needs to be uppercase)
        api_params["id"] = params.color_identity.upper()
    
    # Make the API request
    result = await make_spellbook_request("/variants", api_params)
    
    # Handle errors
    if result.get("error"):
        return f"**Error:** {result.get('message', 'Search failed')}"
    
    # The API returns a 'results' list
    combos = result.get("results", result) if isinstance(result, dict) else result
    
    # Handle case where result is a list directly
    if isinstance(combos, list):
        pass
    elif isinstance(combos, dict):
        combos = combos.get("results", [])
    else:
        combos = []
    
    combos = combos[:params.limit]
    
    if not combos:
        return "**No combos found.** Try a different search query or broader color identity."
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(combos, indent=2)
    
    # Markdown format
    lines = [f"**Found {len(combos)} combos**\n"]
    
    for combo in combos:
        lines.append(format_combo_markdown(combo))
        lines.append("\n---\n")
    
    return "\n".join(lines)


@mcp.tool(
    name="spellbook_find_combos_for_cards",
    annotations={
        "title": "Find Combos for Specific Cards",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def spellbook_find_combos_for_cards(params: SpellbookFindCombosInput) -> str:
    """
    Find all combos that can be made with a specific set of cards.
    
    This is perfect for checking what combos exist in your deck
    or discovering new combos when adding cards.
    
    Args:
        params (SpellbookFindCombosInput): Parameters including:
            - cards (List[str]): List of card names
            - limit (int): Max combos to return (1-50)
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: Combos found with those cards or JSON data
    """
    # Build a search query with all card names
    # We search for combos containing any of these cards
    card_queries = [f'card:"{card}"' for card in params.cards]
    combined_query = " OR ".join(card_queries)
    
    # Make the API request
    api_params = {"q": combined_query, "limit": params.limit}
    result = await make_spellbook_request("/variants", api_params)
    
    # Handle errors
    if result.get("error"):
        return f"**Error:** {result.get('message', 'Search failed')}"
    
    # Extract combos
    combos = result.get("results", result) if isinstance(result, dict) else result
    
    if isinstance(combos, list):
        pass
    elif isinstance(combos, dict):
        combos = combos.get("results", [])
    else:
        combos = []
    
    combos = combos[:params.limit]
    
    if not combos:
        card_list = ", ".join(params.cards)
        return f"**No combos found** containing these cards: {card_list}\n\nThese cards may not have any documented combos, or try different card names."
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(combos, indent=2)
    
    # Markdown format
    card_list = ", ".join(params.cards)
    lines = [f"**Combos containing:** {card_list}\n"]
    lines.append(f"**Found {len(combos)} combos**\n")
    
    for combo in combos:
        lines.append(format_combo_markdown(combo))
        lines.append("\n---\n")
    
    return "\n".join(lines)


@mcp.tool(
    name="spellbook_get_combo",
    annotations={
        "title": "Get Combo Details by ID",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def spellbook_get_combo(params: SpellbookGetComboInput) -> str:
    """
    Get detailed information about a specific combo by its ID.
    
    Use this when you have a combo ID and want full details including
    prerequisites, steps, and results.
    
    Args:
        params (SpellbookGetComboInput): Parameters including:
            - combo_id (str): The Commander Spellbook combo ID
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: Full combo details or JSON data
    """
    # Get the specific combo by ID
    result = await make_spellbook_request(f"/variants/{params.combo_id}")
    
    # Handle errors
    if result.get("error"):
        return f"**Error:** Could not find combo with ID '{params.combo_id}'"
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    return format_combo_markdown(result)


@mcp.tool(
    name="spellbook_find_combos_in_decklist",
    annotations={
        "title": "Find Combos in Decklist",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def spellbook_find_combos_in_decklist(params: SpellbookDecklistInput) -> str:
    """
    Find all combos present in a decklist.
    
    Can accept either a URL to a deck (Moxfield, Archidekt, etc.)
    or a pasted list of card names.
    
    Args:
        params (SpellbookDecklistInput): Parameters including:
            - decklist_url (str): URL to a decklist
            - decklist_text (str): Pasted card list
            - limit (int): Max combos to return (1-20)
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: Combos found in the decklist
    """
    cards = []
    
    # Option 1: Get cards from a deck URL
    if params.decklist_url:
        result = await make_spellbook_post(
            "/card-list-from-url/",
            {"url": params.decklist_url}
        )
        
        if result.get("error"):
            return f"**Error:** Could not fetch decklist from URL. Make sure it's a valid Moxfield, Archidekt, or similar link."
        
        card_list = result.get("cards", [])
        for card in card_list:
            if isinstance(card, dict):
                cards.append(card.get("name", card.get("card", "")))
            else:
                cards.append(str(card))
    
    # Option 2: Parse pasted decklist text
    elif params.decklist_text:
        result = await make_spellbook_post(
            "/card-list-from-text/",
            {"text": params.decklist_text}
        )
        
        if result.get("error"):
            return f"**Error:** Could not parse the decklist text."
        
        card_list = result.get("cards", [])
        for card in card_list:
            if isinstance(card, dict):
                cards.append(card.get("name", card.get("card", "")))
            else:
                cards.append(str(card))
    
    else:
        return "**Error:** Please provide either a decklist URL or pasted card list."
    
    if not cards:
        return "**Error:** Couldn't extract any cards from that decklist."
    
    # Now find combos using the find-my-combos endpoint
    result = await make_spellbook_post("/find-my-combos/", {"cards": cards})
    
    if result.get("error"):
        return f"**Error:** {result.get('message', 'Could not analyze decklist')}"
    
    # The response contains 'results' with 'included' combos
    results = result.get("results", {})
    included = results.get("included", [])
    almost = results.get("almost_included", [])
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "cards_analyzed": len(cards),
            "complete_combos": included[:params.limit],
            "almost_complete": almost[:params.limit]
        }, indent=2)
    
    if not included and not almost:
        return f"**No combos found** in this deck ({len(cards)} cards analyzed)."
    
    lines = [f"**Analyzed {len(cards)} cards**\n"]
    
    # Show fully included combos first
    if included:
        lines.append(f"## Complete Combos ({len(included)} found)\n")
        for combo in included[:params.limit]:
            lines.append(format_combo_markdown(combo))
            lines.append("\n---\n")
    
    # Show "almost" combos (missing 1 card)
    remaining = params.limit - len(included)
    if almost and remaining > 0:
        lines.append(f"\n## Almost Complete (missing 1 card)\n")
        for combo in almost[:remaining]:
            # Get the missing card name
            missing = combo.get("missing", [])
            if missing:
                missing_name = missing[0].get("card", {}).get("name", "Unknown")
                lines.append(f"**Missing:** {missing_name}\n")
            lines.append(format_combo_markdown(combo))
            lines.append("\n---\n")
    
    return "\n".join(lines)


@mcp.tool(
    name="spellbook_estimate_bracket",
    annotations={
        "title": "Estimate Commander Bracket",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def spellbook_estimate_bracket(params: SpellbookBracketInput) -> str:
    """
    Estimate the Commander bracket (power level 1-4) for a decklist.
    
    The bracket system is the official WotC Commander power level system:
    - Bracket 1: Exhibition - Thematic/creative, 9+ turns, no MLD
    - Bracket 2: Core - Unoptimized, social, no two-card infinites or game changers
    - Bracket 3: Upgraded - Strong synergy, up to 3 game changers, 6+ turns
    - Bracket 4: Optimized/cEDH - Anything goes, fast and lethal
    
    Args:
        params (SpellbookBracketInput): Parameters including:
            - decklist_url (str): URL to a decklist
            - decklist_text (str): Pasted card list
            - response_format (str): 'markdown' or 'json'
    
    Returns:
        str: Bracket estimation with combo breakdown
    """
    cards = []
    
    # Option 1: Get cards from a deck URL
    if params.decklist_url:
        result = await make_spellbook_post(
            "/card-list-from-url/",
            {"url": params.decklist_url}
        )
        
        if result.get("error"):
            return f"**Error:** Could not fetch decklist from URL."
        
        card_list = result.get("cards", [])
        for card in card_list:
            if isinstance(card, dict):
                cards.append(card.get("name", card.get("card", "")))
            else:
                cards.append(str(card))
    
    # Option 2: Parse pasted decklist text
    elif params.decklist_text:
        result = await make_spellbook_post(
            "/card-list-from-text/",
            {"text": params.decklist_text}
        )
        
        if result.get("error"):
            return f"**Error:** Could not parse the decklist text."
        
        card_list = result.get("cards", [])
        for card in card_list:
            if isinstance(card, dict):
                cards.append(card.get("name", card.get("card", "")))
            else:
                cards.append(str(card))
    
    else:
        return "**Error:** Please provide either a decklist URL or pasted card list."
    
    if not cards:
        return "**Error:** Couldn't extract any cards from that decklist."
    
    # Call the bracket estimation endpoint
    result = await make_spellbook_post("/estimate-bracket/", {"cards": cards})
    
    if result.get("error"):
        return f"**Error:** {result.get('message', 'Could not estimate bracket')}"
    
    # Parse the response
    bracket = result.get("bracket", "Unknown")
    combos_by_bracket = result.get("combos_by_bracket", {})
    two_card_combos = result.get("two_card_combos", [])
    
    # Format based on requested output
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "bracket": bracket,
            "cards_analyzed": len(cards),
            "two_card_combos": two_card_combos,
            "combos_by_bracket": combos_by_bracket
        }, indent=2)
    
    # Bracket descriptions
    bracket_desc = {
        "1": "Exhibition - Thematic, creative, 9+ turns expected",
        "2": "Core - Unoptimized, social, no two-card infinites",
        "3": "Upgraded - Strong synergy, up to 3 game changers",
        "4": "Optimized/cEDH - Lethal, consistent, anything goes"
    }
    
    lines = [f"## Bracket Estimation: **{bracket}**"]
    lines.append(f"*Cards analyzed: {len(cards)}*\n")
    
    bracket_num = str(bracket).split()[0] if bracket else "?"
    if bracket_num in bracket_desc:
        lines.append(f"**{bracket_desc[bracket_num]}**\n")
    
    # Show two-card combos (these heavily influence bracket)
    if two_card_combos:
        lines.append(f"### Two-Card Combos ({len(two_card_combos)} found)")
        lines.append("*These have the biggest impact on bracket level*\n")
        for combo in two_card_combos[:8]:
            cards_in_combo = combo.get("uses", [])
            card_names = [c.get("card", {}).get("name", "?") for c in cards_in_combo]
            combo_bracket = combo.get("bracket", "?")
            
            produces = combo.get("produces", [])
            results = [p.get("feature", {}).get("name", "") for p in produces[:2]]
            results_str = f" → {', '.join(results)}" if results else ""
            
            lines.append(f"- **{' + '.join(card_names)}** (B{combo_bracket}){results_str}")
        
        if len(two_card_combos) > 8:
            lines.append(f"\n*...and {len(two_card_combos) - 8} more two-card combos*")
    
    # Show combo count by bracket
    if combos_by_bracket:
        lines.append("\n### All Combos by Bracket Level")
        for b_level, combos in sorted(combos_by_bracket.items()):
            count = len(combos) if isinstance(combos, list) else combos
            lines.append(f"- Bracket {b_level}: {count} combos")
    
    return "\n".join(lines)


# =============================================================================
# COMPREHENSIVE RULES TOOL
# =============================================================================

@mcp.tool(
    name="mtg_rules_search",
    annotations={
        "title": "Search MTG Comprehensive Rules",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False  # Local database, not external
    }
)
async def mtg_rules_search(params: RulesSearchInput) -> str:
    """
    Search the MTG Comprehensive Rules using semantic search.
    
    This tool finds relevant rules from the official Comprehensive Rules
    document based on your question. Great for understanding game mechanics,
    resolving disputes, and learning edge cases.
    
    The search uses AI embeddings to find semantically relevant rules,
    not just keyword matching. Ask questions in natural language!
    
    Args:
        params (RulesSearchInput): Search parameters including:
            - query (str): Your rules question in natural language
            - num_results (int): Number of rules to return (1-15)
    
    Returns:
        str: Relevant rules with their rule numbers
    
    Note: Run rules_ingestion.py first to download and index the rules.
    """
    # Try to get the rules collection
    collection = get_rules_collection()
    
    # If no database, provide helpful error message
    if collection is None:
        return (
            "**Rules database not found.**\n\n"
            "To use this tool, you need to run the ingestion script first:\n"
            "```\n"
            "python rules_ingestion.py\n"
            "```\n"
            "This will download the Comprehensive Rules and create the search index."
        )
    
    try:
        # Query the collection for relevant rules
        # ChromaDB will convert the query to an embedding and find similar rules
        results = collection.query(
            query_texts=[params.query],
            n_results=params.num_results
        )
        
        # Extract the results
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        if not documents:
            return f"**No relevant rules found for:** {params.query}\n\nTry rephrasing your question."
        
        # Format the results
        lines = [f"**Relevant rules for:** {params.query}\n"]
        
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            rule_number = meta.get("rule_number", "Unknown")
            
            # Convert distance to a simple relevance indicator
            # Lower distance = more relevant
            # ChromaDB uses L2 distance by default
            relevance = "●●●" if dist < 0.5 else "●●○" if dist < 1.0 else "●○○"
            
            lines.append(f"### {rule_number} {relevance}")
            lines.append(f"{doc}\n")
        
        lines.append("---")
        lines.append("*Relevance: ●●● = High, ●●○ = Medium, ●○○ = Lower*")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"**Error searching rules:** {str(e)}"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the MCP server using stdio transport (for local use)
    mcp.run()