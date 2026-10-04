"""Rulebreaker commanders (Mystery Booster Commander Edition). Not legal in sanctioned Commander -
played by agreement ("Rule 0"). Each relaxes a deckbuilding rule:

- Whtz, the Bibliophile: no maximum deck size (see companions.no_max_deck_size - Yorion).
- The rest let certain cards ignore the commander's color identity; all but Grizzlegom also allow
  "any basic land cards". Tolabow lets instants/sorceries add ONE color of your choice: the deck's
  most common extra color on its instants/sorceries is taken as the choice.

`exempt()` answers "is this off-identity card allowed anyway?" for a card dict carrying
type_line / cmc / color_identity (a full Scryfall card or the app's slim deck card). The page
mirrors these rules in rbExempt().
"""

RULEBREAKERS = {
    "Grizzlegom, Hurloon Hero": "any land cards",
    "Maular, the Next Evolution": "creature cards with mana value 7 or greater of any color identity, and any basic lands",
    "Seluma, Light of Aysen": "Angel cards of any color identity, and any basic lands",
    "The Everforger": "artifact creature and Equipment cards of any color identity, and any basic lands",
    "The Unluckiest Planeswalker": "Aura cards of any color identity, and any basic lands",
    "Tolabow, Loch Rascal": "instants and sorceries may include one extra color of your choice, and any basic lands",
    "Valko Indorian": "Phyrexian cards of any color identity, and any basic lands",
    "Whtz, the Bibliophile": "no maximum deck size",
}


def _front_type(card: dict) -> str:
    return (card.get("type_line") or "").split(" // ")[0]


def _words(card: dict) -> set:
    return set(_front_type(card).replace("—", " ").split())


def active(commanders: list) -> list:
    """The rulebreakers among these commander names (canonical names)."""
    lower = {n.lower() for n in commanders or []}
    return [n for n in RULEBREAKERS if n.lower() in lower]


def tolabow_color(deck_cards: list, identity: set) -> str | None:
    """Tolabow's chosen extra color = the most common off-identity color on the deck's
    instants and sorceries (None if there are none)."""
    counts = {}
    for c in deck_cards or []:
        w = _words(c)
        if "Instant" in w or "Sorcery" in w:
            for x in set(c.get("color_identity") or []) - identity:
                counts[x] = counts.get(x, 0) + int(c.get("qty") or 1)
    return max(counts, key=counts.get) if counts else None


def exempt(card: dict, commanders: list, identity: set, deck_cards: list = None) -> bool:
    """True if this card is allowed outside `identity` because of a rulebreaker commander."""
    rbs = active(commanders)
    if not rbs:
        return False
    w = _words(card)
    basic = "Basic" in w and "Land" in w
    for rb in rbs:
        if rb.startswith("Grizzlegom") and "Land" in w:
            return True
        if rb.startswith("Whtz"):
            continue
        if basic:
            return True
        if rb.startswith("Maular") and "Creature" in w and float(card.get("cmc") or 0) >= 7:
            return True
        if rb.startswith("Seluma") and "Angel" in w:
            return True
        if rb == "The Everforger" and (("Artifact" in w and "Creature" in w) or "Equipment" in w):
            return True
        if rb == "The Unluckiest Planeswalker" and "Aura" in w:
            return True
        if rb.startswith("Valko") and "Phyrexian" in w:
            return True
        if rb.startswith("Tolabow") and ("Instant" in w or "Sorcery" in w):
            extra = set(card.get("color_identity") or []) - identity
            chosen = tolabow_color(deck_cards, identity)
            if len(extra) <= 1 and (chosen is None or extra <= {chosen}):
                return True
    return False


def context_lines(commanders: list, deck_cards: list, identity: set) -> list[str]:
    rbs = active(commanders)
    if not rbs:
        return []
    out = ["", "RULEBREAKER COMMANDER (Mystery Booster Commander Edition - not legal in sanctioned Commander; "
               "played by agreement): " + "; ".join(f"{n}: {RULEBREAKERS[n]}" for n in rbs) + "."]
    if any(n.startswith("Tolabow") for n in rbs):
        c = tolabow_color(deck_cards, identity)
        out.append(f"Tolabow's extra color for instants/sorceries: {c or 'not chosen yet (no off-color instants/sorceries so far)'}.")
    return out
