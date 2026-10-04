"""Ikoria companions in Commander: deckbuilding conditions checked against full Scryfall cards.

In Commander a companion sits outside the 100 (deck size stays exactly 100), must be within the
commander's color identity, and its condition applies to the whole starting deck INCLUDING the
commander(s). Putting it into your hand costs {3} (sorcery speed).
- Yorion needs 20 cards over the minimum: impossible at exactly 100, possible only under a commander
  that removes the maximum deck size (e.g. Whtz, the Bibliophile - "no maximum deck size"), with 120+.
- Lutri is legal in the 99 but BANNED AS A COMPANION in Commander.

Cards are judged by the face they have in the library: the front face of transform / modal
double-faced cards; split and adventure cards use their full mana value.
"""
import re

PERMANENT_TYPES = {"Artifact", "Creature", "Enchantment", "Land", "Planeswalker", "Battle"}
CARD_TYPES = PERMANENT_TYPES | {"Instant", "Sorcery", "Kindred", "Tribal"}
KAHEERA_TYPES = {"Cat", "Elemental", "Nightmare", "Dinosaur", "Beast"}
# keyword abilities that are activated abilities (Zirda); "...cycling" covers landcycling etc.
ACTIVATED_KEYWORDS = {"equip", "ninjutsu", "commander ninjutsu", "unearth", "crew", "reconfigure",
                      "level up", "outlast", "scavenge", "embalm", "eternalize", "fortify", "transmute",
                      "encore", "bloodrush", "craft", "station", "saddle", "forecast", "auto"}
_FRONT_ONLY = {"transform", "modal_dfc", "flip", "meld", "reversible_card"}


def _front(card: dict) -> dict:
    """The characteristics the card has in a library."""
    faces = card.get("card_faces") or []
    if faces and card.get("layout") in _FRONT_ONLY:
        f = faces[0]
        return {"type_line": f.get("type_line", ""), "oracle": f.get("oracle_text", ""),
                "costs": [f.get("mana_cost", "")], "cmc": card.get("cmc") or 0,
                "keywords": card.get("keywords") or []}
    if faces:  # split / adventure: both halves are on the card
        return {"type_line": card.get("type_line") or " // ".join(f.get("type_line", "") for f in faces),
                "oracle": "\n".join(f.get("oracle_text", "") for f in faces),
                "costs": [f.get("mana_cost", "") for f in faces], "cmc": card.get("cmc") or 0,
                "keywords": card.get("keywords") or []}
    return {"type_line": card.get("type_line", ""), "oracle": card.get("oracle_text", ""),
            "costs": [card.get("mana_cost", "")], "cmc": card.get("cmc") or 0,
            "keywords": card.get("keywords") or []}


def _types(front: dict) -> set:
    main = front["type_line"].split(" — ")[0]
    return {w for w in re.split(r"[\s/]+", main) if w in CARD_TYPES}


def _subtypes(front: dict) -> set:
    parts = front["type_line"].split(" — ", 1)
    return set(parts[1].split()) if len(parts) > 1 else set()


def _is_land(front: dict) -> bool:
    return "Land" in _types(front)


def _is_permanent(front: dict) -> bool:
    return bool(_types(front) & PERMANENT_TYPES)


def _has_activated(front: dict) -> bool:
    """Approximate: a "cost: effect" line not inside quotes (granted abilities don't count), an
    activated keyword, or a land with a mana ability (basics' is in reminder text)."""
    if _is_land(front) and "{T}" in front["oracle"]:
        return True
    if any(k.lower() in ACTIVATED_KEYWORDS or k.lower().endswith("cycling") for k in front["keywords"]):
        return True
    for line in front["oracle"].split("\n"):
        line = re.sub(r'"[^"]*"', "", re.sub(r"\([^)]*\)", "", line))
        if ":" in line and not line.lower().startswith(("companion", "choose")):
            return True
    return False


def _repeats_symbol(front: dict) -> bool:
    for cost in front["costs"]:
        syms = re.findall(r"\{[^}]+\}", cost or "")
        if len(syms) != len(set(syms)):
            return True
    return False


# name -> (short condition, per-card check: True = the card is fine). Umori and Yorion are deck-level.
CONDITIONS = {
    "Gyruda, Doom of Depths": ("only cards with even mana values",
                               lambda f: int(f["cmc"]) % 2 == 0),
    "Jegantha, the Wellspring": ("no card has more than one of the same mana symbol in its cost",
                                 lambda f: not _repeats_symbol(f)),
    "Kaheera, the Orphanguard": ("every creature is a Cat, Elemental, Nightmare, Dinosaur or Beast",
                                 lambda f: "Creature" not in _types(f) or bool(_subtypes(f) & KAHEERA_TYPES)
                                 or any(k.lower() == "changeling" for k in f["keywords"])),
    "Keruga, the Macrosage": ("only lands and cards with mana value 3 or more",
                              lambda f: _is_land(f) or f["cmc"] >= 3),
    "Lurrus of the Dream-Den": ("every permanent card has mana value 2 or less",
                                lambda f: not _is_permanent(f) or f["cmc"] <= 2),
    "Lutri, the Spellchaser": ("every nonland card has a different name", None),
    "Obosh, the Preypiercer": ("only lands and cards with odd mana values",
                               lambda f: _is_land(f) or int(f["cmc"]) % 2 == 1),
    "Umori, the Collector": ("every nonland card shares a card type", None),
    "Yorion, Sky Nomad": ("at least 20 cards over the minimum deck size", None),
    "Zirda, the Dawnwaker": ("every permanent card has an activated ability (checked approximately)",
                             lambda f: not _is_permanent(f) or _has_activated(f)),
}


def no_max_deck_size(card: dict) -> bool:
    """A commander whose text lifts the 100-card maximum (Whtz, the Bibliophile)."""
    text = card.get("oracle_text") or " ".join(f.get("oracle_text", "") for f in card.get("card_faces") or [])
    return "no maximum deck size" in text.lower()


def check(companion: str, cards: list, commanders: set = frozenset()) -> dict:
    """cards: [(full Scryfall card, qty)] for the whole starting deck INCLUDING the commander(s);
    commanders: lowercased commander names. -> {name, condition, ok, violations: [names], note}."""
    name = next((n for n in CONDITIONS if n.lower() == (companion or "").lower()), None)
    if not name:
        return {"name": companion, "condition": "", "ok": False, "violations": [],
                "note": "not an Ikoria companion"}
    cond, fn = CONDITIONS[name]
    if name.startswith("Lutri"):
        return {"name": name, "condition": cond, "ok": False, "violations": [],
                "note": "banned as a companion in Commander (it can still be in the 99)"}
    if name.startswith("Yorion"):
        if not any(no_max_deck_size(c) for c, _ in cards if c.get("name", "").lower() in commanders):
            return {"name": name, "condition": cond, "ok": False, "violations": [],
                    "note": "impossible in Commander - decks are exactly 100 cards (unless your commander "
                            "removes the maximum deck size)"}
        size = sum(q for _, q in cards)
        cond = "at least 120 cards (your commander removes the 100-card maximum)"
        return {"name": name, "condition": cond, "ok": size >= 120, "violations": [],
                "note": "" if size >= 120 else f"needs at least 120 cards - the deck has {size}"}
    fronts = [(c, q, _front(c)) for c, q in cards]
    if name.startswith("Lutri"):
        bad = [c["name"] for c, q, f in fronts if q > 1 and not _is_land(f)]
    elif name.startswith("Umori"):
        nonland = [(c, f) for c, q, f in fronts if not _is_land(f)]
        shared = set.intersection(*(_types(f) for _, f in nonland)) if nonland else set()
        if shared:
            bad = []
        else:  # best type to aim for = the most common one; the rest break it
            counts = {}
            for _, f in nonland:
                for t in _types(f):
                    counts[t] = counts.get(t, 0) + 1
            best = max(counts, key=counts.get) if counts else None
            bad = [c["name"] for c, f in nonland if best not in _types(f)]
            cond = f"{cond} (closest: {best})" if best else cond
    else:
        bad = [c["name"] for c, q, f in fronts if not fn(f)]
    return {"name": name, "condition": cond, "ok": not bad, "violations": bad, "note": ""}


TYPE_ORDER = ["Land", "Creature", "Planeswalker", "Battle", "Instant", "Sorcery", "Artifact", "Enchantment"]


def main_type(card: dict) -> str:
    """One type per card for the type breakdown (front face; artifact creature = Creature,
    artifact land = Land). The page uses the same order."""
    t = _types(_front(card))
    return next((x for x in TYPE_ORDER if x in t), "Other")


def deck_stats(cards: list, commanders: set) -> dict:
    """cards: [(full card, qty)]. Curve = nonland cards, commander(s) excluded, 7+ bucketed;
    split by creature / noncreature. Types = every card except the commander(s)."""
    curve, types = {}, {}
    for c, q in cards:
        if (c.get("name") or "").lower() in commanders:
            continue
        mt = main_type(c)
        types[mt] = types.get(mt, 0) + q
        if mt == "Land":
            continue
        b = min(int(_front(c)["cmc"]), 7)
        key = "creature" if "Creature" in _types(_front(c)) else "other"
        curve.setdefault(b, {"creature": 0, "other": 0})[key] += q
    return {"curve": curve, "types": types}
