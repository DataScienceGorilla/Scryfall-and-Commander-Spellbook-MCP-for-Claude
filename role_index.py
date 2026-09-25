"""
Concrete card-role index from Scryfall oracle-tags (otags)
==========================================================
The advisor's ROLE COMPOSITION has two tiers:
  * CONCRETE roles (this module) - Ramp / Draw / Target Interaction / Mass Interaction /
    Recursion / Tutor / Protection / Stax - are a function of the card's text, so we read
    them from Scryfall's human-curated otags (validated ~86-100% recall vs. hand-tags,
    with full card-pool coverage and zero ML). Built once into a cached card->roles index
    and refreshed periodically, like the rules RAG.
  * CONTEXTUAL roles (Enabler / Payoff / Force Multiplier / Engine Piece / Threats /
    Alternate Wincon / Misc Value) depend on the whole deck, not the single card, so the
    advisor's LLM assigns those holistically at review time. Combo comes from Spellbook.

Refresh:  python role_index.py            (skips if current)
          python role_index.py --force    (rebuild)
"""
import json
import time
import sys
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import date

INDEX_PATH = Path(__file__).parent / "role_index.json"
STALE_DAYS = 30

# Concrete role -> Scryfall otag query (validated against the user's certified gold).
ROLE_OTAG = {
    "Ramp": "otag:ramp",
    "Draw": "otag:card-advantage",
    "Target Interaction": "(otag:spot-removal or otag:counterspell)",
    "Mass Interaction": "otag:board-wipe",
    "Recursion": "otag:recursion",
    "Tutor": "otag:tutor",
    "Protection": "otag:protection",
}

# Stax has no otag; seed with a curated list + otag:tax (taxing effects).
STAX_OTAG = "otag:tax"
STAX_CURATED = [
    "Winter Orb", "Static Orb", "Rule of Law", "Archon of Emeria", "Thalia, Guardian of Thraben",
    "Drannith Magistrate", "Sphere of Resistance", "Collector Ouphe", "Blood Moon", "Back to Basics",
    "Stony Silence", "Null Rod", "Cursed Totem", "Aven Mindcensor", "Opposition Agent",
    "Grand Abolisher", "Deafening Silence", "Root Maze", "Damping Sphere", "Ethersworn Canonist",
    "Trinisphere", "Thorn of Amethyst", "Lodestone Golem", "Kataki, War's Wage", "Magus of the Moon",
    "Hushbringer", "Notion Thief", "Narset, Parter of Veils", "Torpor Orb", "Elesh Norn, Grand Cenobite",
]

_HEADERS = {"User-Agent": "MTGAdvisor/1.0", "Accept": "application/json"}


def _get(url: str, tries: int = 6):
    """GET with 429/backoff retry (Scryfall throttles bursts). Returns parsed JSON or None."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS)) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After", "1")) or 1.0)
                continue
            if e.code == 404:
                return {"data": [], "next_page": None}  # empty tag
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return None


def _fetch_set(query: str) -> set[str]:
    """All card names (lowercased, front-face too) matching a Scryfall query."""
    names: set[str] = set()
    url = "https://api.scryfall.com/cards/search?q=" + urllib.parse.quote(query) + "&unique=cards"
    while url:
        d = _get(url)
        if d is None:
            raise RuntimeError(f"Scryfall fetch failed for: {query}")
        for c in d.get("data", []):
            nm = c["name"].lower()
            names.add(nm)
            if " // " in nm:
                names.add(nm.split(" // ")[0])
        url = d.get("next_page")
        time.sleep(0.15)
    return names


def build_index(force: bool = False) -> dict:
    """Fetch otag sets and invert to {card_lower: [roles]}. Cached to INDEX_PATH."""
    if INDEX_PATH.exists() and not force:
        meta = json.loads(INDEX_PATH.read_text(encoding="utf-8")).get("_meta", {})
        built = meta.get("built")
        if built and (date.today() - date.fromisoformat(built)).days < STALE_DAYS:
            print(f"role_index is current (built {built}); use --force to rebuild.")
            return load_role_index()

    card_roles: dict[str, set[str]] = {}
    role_sizes = {}
    for role, q in ROLE_OTAG.items():
        s = _fetch_set(q)
        role_sizes[role] = len(s)
        for nm in s:
            card_roles.setdefault(nm, set()).add(role)
        print(f"  {role:<22} {len(s)}", flush=True)
    # Stax: otag:tax + curated
    stax = _fetch_set(STAX_OTAG) | {n.lower() for n in STAX_CURATED}
    role_sizes["Stax"] = len(stax)
    for nm in stax:
        card_roles.setdefault(nm, set()).add("Stax")
    print(f"  {'Stax':<22} {len(stax)}", flush=True)

    out = {"_meta": {"built": date.today().isoformat(), "role_sizes": role_sizes,
                     "roles": list(ROLE_OTAG) + ["Stax"]},
           "cards": {k: sorted(v) for k, v in card_roles.items()}}
    INDEX_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(card_roles)} cards -> {INDEX_PATH}")
    return out


_CACHE = None
def load_role_index() -> dict:
    global _CACHE
    if _CACHE is None:
        if not INDEX_PATH.exists():
            return build_index()
        _CACHE = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return _CACHE


def roles_for(card_name: str) -> list[str]:
    """Concrete roles for a single card (empty if none / not indexed)."""
    idx = load_role_index()["cards"]
    nm = card_name.lower()
    return idx.get(nm) or idx.get(nm.split(" // ")[0], [])


def deck_concrete_roles(names: list[str]) -> dict:
    """Given card names (respect quantities by repeating), return {counts, per_card}."""
    idx = load_role_index()["cards"]
    counts: dict[str, int] = {}
    per_card: dict[str, list[str]] = {}
    for n in names:
        nm = n.lower()
        rs = idx.get(nm) or idx.get(nm.split(" // ")[0], [])
        per_card[n] = rs
        for r in rs:
            counts[r] = counts.get(r, 0) + 1
    return {"counts": counts, "per_card": per_card}


if __name__ == "__main__":
    build_index(force="--force" in sys.argv)
