"""
System prompt v2 ("coach") - the same knowledge and mechanics as v1, reorganized into four layers so
the advisor reads as a deckbuilding advisor helping the player build, not a chat that builds for them:

  1. STANCE     - who the advisor is and how it relates to the player (prompts/v2/stance.md)
  2. KNOWLEDGE  - the framework (prompts/v2/framework.md, matching the Template panel) plus v1's
                  card evaluation, bracket, combo, rules and review sections and the distilled playbook
  3. SITUATION  - how to read the live deck, memory and profile blocks (v1) and the per-turn
                  [SITUATION] block (prompts/v2/situation.md; built by situation_block below)
  4. MECHANICS  - the canvas, color identity, proposals / questions, tools and formatting (v1)

v2 is ASSEMBLED FROM v1's own sections (matched by heading), so edits to v1's rules carry over; only
the opening, INTAKE, STYLE and DECKBUILDING FRAMEWORK sections are replaced. A/B: /chat takes
"prompt_variant" ("v1" | "v2"); ADVISOR_PROMPT_DEFAULT sets the default (v1).
"""
import re
from pathlib import Path

_DIR = Path(__file__).parent / "prompts" / "v2"

# v1 sections each layer reuses, by heading prefix, in the order they appear in v2
KNOWLEDGE = ["# CARD EVALUATION", "# RECOMMENDATION QUALITY", "# BRACKET DISCIPLINE", "# DEEP CUTS",
             "# RULES INTERACTIONS", "# RULEBREAKER COMMANDERS", "# COMPANIONS", "# FULL DECK REVIEW OUTPUT"]
SITUATION = ["# LIVE DECK + SESSION MEMORY"]
MECHANICS = ["# CARD CANVAS", "# COLOR IDENTITY", "# PROPOSING CHANGES", "# USING YOUR TOOLS",
             "# TOOL ECONOMY", "# HOW TO USE YOUR TOOLS"]
REPLACED = ["# INTAKE FIRST", "# STYLE", "# DECKBUILDING FRAMEWORK"]  # rewritten in prompts/v2

# v1's STYLE carried the citation rule; it belongs with the mechanics now
SOURCING = """# SOURCING
NEVER announce that you're about to cite something or "check what the sources say" - that
promise-without-delivery reads as broken. Either attribute a specific claim inline as you make it -
e.g. "(Salubrious Snail, 'EDH Doesn't Have Archetypes')" - or just make the point plainly. Only name a
source when you ACTUALLY retrieved it via deckbuilding_search; paraphrase in your own words (at most a
brief quote). Answering from your own knowledge with no citation is fine. Don't narrate your tool use.
For a full deck review DEPTH beats brevity; everywhere else, keep it as short as the question allows."""


def _read(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8").strip()


def _sections(v1: str) -> tuple[dict[str, str], str, str]:
    """v1 -> ({heading prefix: section text}, playbook chunk, closing formatting rules)."""
    pb_at = v1.find("\n\n# HOW TO THINK")
    end_at = v1.find("\n\n# NON-NEGOTIABLE")
    body = v1[:pb_at if pb_at >= 0 else end_at if end_at >= 0 else len(v1)]
    playbook = v1[pb_at:end_at].strip() if pb_at >= 0 and end_at > pb_at else ""
    closing = v1[end_at:].strip() if end_at >= 0 else ""
    found = {}
    for part in re.split(r"\n(?=# [A-Z])", body):
        head = part.split("\n", 1)[0]
        for key in KNOWLEDGE + SITUATION + MECHANICS + REPLACED:
            if head.startswith(key):
                found[key] = part.strip()
    return found, playbook, closing


def build(v1: str) -> str:
    """The v2 system prompt, assembled from v1's sections and the prompts/v2 files."""
    s, playbook, closing = _sections(v1)
    missing = [k for k in KNOWLEDGE + SITUATION + MECHANICS if k not in s]
    if missing:  # a renamed v1 heading must not silently drop a rule from v2
        raise RuntimeError(f"prompt_v2: v1 sections not found: {missing}")
    layer = lambda title: f"\n\n{'=' * 8} {title} {'=' * 8}"
    parts = [_read("stance.md"),
             layer("PART 2 - WHAT YOU KNOW"), _read("framework.md"), *[s[k] for k in KNOWLEDGE]]
    if playbook:
        parts.append(playbook)
    parts += [layer("PART 3 - READING THE SITUATION"), s[SITUATION[0]], _read("situation.md"),
              layer("PART 4 - MECHANICS (how the app works - follow exactly)"),
              *[s[k] for k in MECHANICS], SOURCING]
    if closing:
        parts.append(closing)
    return "\n\n".join(p for p in parts if p)


# --- per-turn situation -----------------------------------------------------------------------
_STAGE_HINT = {
    "none": "Answer the question; if they're starting a deck, help them get one into the editor.",
    "commander": "Help them find a commander that fits what they enjoy - two or three options and what "
                 "each plays like - rather than a list of popular picks.",
    "early": "Help the deck find its identity: how it wins, the engine, the first package. Directions "
             "and a few anchor cards, not lists of staples.",
    "filling": "Point at the gap that would hurt most in real games (template shortfalls, curve holes) and "
               "offer a few cards that fit what's already there.",
    "final": "Close the biggest remaining gap; mind the curve.",
    "tuning": "They're tuning a full deck: favor swaps (a cut paired with each add) and explain the tradeoff.",
    "cutting": "Help them cut: crowded curve slots first, then interaction, then the secondary utility; "
               "never lands. Ask which cards are pets before cutting them.",
}


def _num(x, d=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def situation_block(deck: dict | None, template: dict | None, note: str = "") -> str:
    """The [SITUATION] block for the player's latest message: build stage, what changed since the last
    reply (the editor's own note), and the Template panel's numbers as the client computed them."""
    deck = deck if isinstance(deck, dict) else {}
    cards = [c for c in deck.get("cards") or [] if isinstance(c, dict)]
    total = sum(max(1, _num(c.get("qty"), 1)) for c in cards)
    build = deck.get("build") if isinstance(deck.get("build"), dict) else None
    building = bool(build and not build.get("done"))
    cmdrs = [n for n in deck.get("commander") or [] if isinstance(n, str)]
    no_max = any(c.get("no_max_deck") for c in cards if c.get("name") in cmdrs)
    if not cards and not building:
        stage, line = "none", "No deck in the editor."
    elif not cmdrs:
        stage, line = "commander", f"Choosing a commander ({total} cards so far)."
    elif total < 45:
        stage, line = "early", f"Early brew - {total}/100: the core and the gameplan are taking shape."
    elif total < 95:
        stage, line = "filling", f"Filling out - {total}/100."
    elif total < 100:
        stage, line = "final", f"Final stretch - {total}/100, {100 - total} to go."
    elif total == 100 or no_max:
        stage, line = "tuning", f"Complete at {total} - tuning."
    else:
        stage, line = "cutting", f"Over by {total - 100} ({total}/100) - cutting down."
    if building:
        line += " Guided build mode is on."
    out = ["[SITUATION - from the player's deck editor this turn; the player doesn't see this]",
           f"- Stage: {line}"]

    if (note or "").strip():  # the note itself already rides on their message - point at it, don't repeat it
        out.append("- Since your last reply they edited the deck - the bracketed note on their message lists "
                   "what they added, cut or saved. React to it.")
    elif cards:
        out.append("- No deck edits since your last reply.")

    t = template if isinstance(template, dict) else None
    if t and cards:
        want, have = t.get("targets") or {}, t.get("have") or {}
        rows = []
        for k, label in (("lands", "Lands"), ("ramp", "Ramp"), ("draw", "Draw"), ("interaction", "Interaction")):
            if k not in want:
                continue
            h, w = _num(have.get(k)), _num(want.get(k))
            gap = w - h
            rows.append(f"{label} {h}/{w}" + (f" ({gap} short)" if gap > 0 else f" ({-gap} over)" if gap < -2 else ""))
        if rows:
            out.append(f"- Template (operational threshold {_num(t.get('threshold'), 4)}): " + " · ".join(rows))
        curve = t.get("curve") or {}
        now, sug = curve.get("now") or [], curve.get("want") or []
        if len(now) == 8 and len(sug) == 8:
            marks = []
            for i, (n, w) in enumerate(zip(now, sug)):
                n, w = _num(n), _num(w)
                if abs(w - n) >= 2:
                    marks.append(f"MV {'7+' if i == 7 else i}: {n} vs ~{w} ({'crowded' if n > w else 'thin'})")
            if marks:
                out.append("- Curve vs suggested: " + "; ".join(marks))
    out.append(f"- What usually helps most here: {_STAGE_HINT[stage]}")
    return "\n".join(out)
