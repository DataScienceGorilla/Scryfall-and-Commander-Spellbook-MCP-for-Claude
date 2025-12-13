# MTG MCP Server

An MCP (Model Context Protocol) server that integrates **Scryfall** (card database) and **Commander Spellbook** (combo database) for Magic: The Gathering.

## Features

### Scryfall Tools
- **scryfall_search_cards**: Search for cards using Scryfall's powerful syntax
- **scryfall_get_card**: Look up a specific card by name (fuzzy or exact)
- **scryfall_random_card**: Get a random card, optionally filtered
- **scryfall_get_rulings**: Get official rulings/clarifications for a card

### Commander Spellbook Tools
- **spellbook_search_combos**: Search for EDH/Commander combos
- **spellbook_find_combos_for_cards**: Find combos containing specific cards
- **spellbook_get_combo**: Get detailed info about a specific combo

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python mtg_mcp.py
```

## Claude Desktop Configuration

Add this to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mtg": {
      "command": "python",
      "args": ["/path/to/mtg_mcp.py"]
    }
  }
}
```

Replace `/path/to/mtg_mcp.py` with the actual path to the server file.

## Example Usage

### Scryfall Search Syntax

The search tool uses Scryfall's powerful syntax. Here are some examples:

| Query | Description |
|-------|-------------|
| `c:blue t:creature` | Blue creatures |
| `id:simic t:legendary` | Simic-identity legendary cards |
| `o:"draw a card" cmc<=3` | Cards with "draw a card" text, CMC 3 or less |
| `t:creature pow>=5` | Creatures with 5+ power |
| `is:commander id:golgari` | Golgari-legal commanders |
| `r:mythic set:mh2` | Mythic rares from Modern Horizons 2 |

### Finding Combos

Search for combos by:
- Card names: `card:"Thassa's Oracle"`
- Effects/results: `result:infinite`
- Color identity: Use the `color_identity` parameter

## API References

- [Scryfall API Documentation](https://scryfall.com/docs/api)
- [Commander Spellbook](https://commanderspellbook.com/)

## Rate Limits

- **Scryfall**: 10 requests/second (the server adds automatic delays)
- **Commander Spellbook**: No documented rate limits, but be reasonable

## License

MIT License - Use freely for your MTG projects!
