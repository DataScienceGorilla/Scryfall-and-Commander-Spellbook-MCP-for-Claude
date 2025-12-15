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
SYSTEM_PROMPT = """You are an expert Magic: The Gathering assistant in a Discord server. 
You help players with card lookups, combo searches, and rules questions.

You have access to these tools:
- scryfall_search_cards: Search for MTG cards using Scryfall syntax
- scryfall_get_card: Look up a specific card by name
- scryfall_get_rulings: Get official rulings for a card
- spellbook_search_combos: Search for Commander/EDH combos
- spellbook_find_combos_for_cards: Find combos containing specific cards
- mtg_rules_search: Search the Comprehensive Rules (if database is available)

Keep responses concise since this is Discord - aim for under 2000 characters.
Use markdown formatting sparingly. Don't use headers (##) in Discord.
When showing card info, focus on the most relevant details.
If a user asks about rules interactions, search the rules AND consider the specific cards involved.
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


def get_rules_collection():
    """Lazily loads the ChromaDB collection for rules search."""
    global _rules_collection
    
    if _rules_collection is not None:
        return _rules_collection
    
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
        "description": "Get official rulings and clarifications for a specific card from Wizards of the Coast.",
        "input_schema": {
            "type": "object",
            "properties": {
                "card_name": {
                    "type": "string",
                    "description": "Name of the card to get rulings for"
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
        "name": "mtg_rules_search",
        "description": "Search the MTG Comprehensive Rules using semantic search. Ask rules questions in natural language.",
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
                    "default": 3
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
    """Get rulings for a card."""
    async with httpx.AsyncClient() as client:
        try:
            # First get the card to find its ID
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
            for ruling in rulings[:5]:  # Limit to 5 for Discord
                comment = ruling.get("comment", "")
                lines.append(f"• {comment}")
            
            return "\n".join(lines)
            
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


async def mtg_rules_search(query: str, num_results: int = 3) -> str:
    """Search the Comprehensive Rules."""
    collection = get_rules_collection()
    
    if collection is None:
        return "Rules database not available. Run rules_ingestion.py to set it up."
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=num_results
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return "No relevant rules found."
        
        lines = []
        for doc, meta in zip(documents, metadatas):
            rule_num = meta.get("rule_number", "?")
            # Truncate long rules for Discord
            text = doc[:300] + "..." if len(doc) > 300 else doc
            lines.append(f"**{rule_num}**: {text}")
        
        return "\n\n".join(lines)
        
    except Exception as e:
        return f"Error searching rules: {str(e)}"


# Map tool names to functions
TOOL_FUNCTIONS = {
    "scryfall_search_cards": scryfall_search_cards,
    "scryfall_get_card": scryfall_get_card,
    "scryfall_get_rulings": scryfall_get_rulings,
    "spellbook_search_combos": spellbook_search_combos,
    "spellbook_find_combos_for_cards": spellbook_find_combos_for_cards,
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
    max_iterations = 5  # Safety limit
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