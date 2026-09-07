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

# Which Claude model to use (claude-sonnet-5 is fast and capable)
CLAUDE_MODEL = "claude-sonnet-5"

# Maximum tokens for Claude's response
MAX_TOKENS = 1024


from mtg_tools import (
    RULES_DB_PATH,
    TOOLS,
    TOOL_FUNCTIONS,
    get_rules_collection_async,
    get_rules_collection,
)

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

CRITICAL Layers rules for continuous effects:
The order in which you apply the layers is as follows.

1. Copy Effects
Copy effects like Clone or Mirrorweave.

2. Control-changing Effects
Control changing effects like Agent of Treachery or Control Magic.

3. Text-changing Effects
Text changing effects like Sleight of Mind or the overload mechanic (see Mizzium Mortars).

4. Type-changing Effects
Type changing effects like Blood Moon or Arcane Adaptation.

5. Color-changing Effects
Color changing effects like Snakeform or Painter's Servant.

6. Effects that Add or Remove Abilities
Effects that add or remove abilities like Humility or Akroma's Memorial.

7. Effects That Change Power or Toughness
There are a lot of different kinds and they're applied in the following order:

a) Characteristic-defining abilities (also known as CDAs) like those on Tarmogoyf or Necrogoyf. Basically any ability that determines the value of a * on that creature's power/toughness.
b) Effects that set “base” power and toughness like Ensoul Artifact or Witness Protection.
c) Effects and counters that modify power and/or toughness like Giant Growth, Dead Weight or +1/+1 or -1/-1 counters.
d) Effects that switch power and toughness like Inside Out or Twisted Image.

NOTE: Two continuous effects in the same layer are applied in timestamp order.

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
- "As [this] enters" and "enters with" are replacement effects, not triggered abilities
- Activated abilities are written as "[cost]: [effect]" - the colon is the giveaway
- Triggered abilities start with "when", "whenever", or "at"
- Each activation of an ability that grants an ability STACKS (e.g., activating a manland's ability twice gives two instances of any granted triggered abilities)
- The stack resolves top-down, but state-based actions are checked after EACH object resolves
- Equip effects are sorcery speed unless otherwise specified, and go on the stack
- "Target" is a magic word - if a spell/ability doesn't say "target", it doesn't target

IMPORTANT - When answering rules questions:
1. Look up any specific cards mentioned with scryfall_get_card to see exact oracle text first
2. Follow up by searching the rules with mtg_rules_search first
3. Consider if a line of play requires an impossible game state, like a creature existing with 0 or less toughness, or negative life total without a replacement effect
4. Ensure targets are legal for effects, for instance you cannot equip a creature with zero toughness as it dies before becoming targetable
5. Some triggers can only occur during specific phases or steps of a turn, like attack triggers requiring declared attackers (508.3a), check those are valid triggers
6. Take your time analyzing timings and interactions carefully, it matters exactly when things happen
7. Check scryfall_get_rulings for official clarifications on those cards, put weight behind ALL official rulings, even if they dont seem relevant at first
8. Cite rule numbers in your answer when possible

Format your response as:
<analysis>
[Think through the rules, interactions, and edge cases here]
</analysis>

<ruling>
[Final answer only]
</ruling>


Keep responses concise since this is Discord - aim for under 2000 characters.
Use markdown formatting sparingly. Don't use headers (##) in Discord.
When showing card info, focus on the most relevant details.
"""



# =============================================================================
# API CLIENTS AND SETUP
# =============================================================================

# Initialize the Anthropic client (reads ANTHROPIC_API_KEY from environment)
claude_client = anthropic.Anthropic()



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
