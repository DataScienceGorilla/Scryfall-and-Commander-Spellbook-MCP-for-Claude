"""
Distill a creator's deckbuilding philosophy from the theory corpus into a
PLAYBOOK the advisor can use as operating instructions.

Stages (map-reduce summarization, NOT model distillation):
  1. MAP  - per video, a cheap model extracts principles/lenses (with the WHY)
            and concrete worked examples. Parallelized + cached (resumable).
  2. REDUCE - a stronger model synthesizes those notes into a per-creator
            playbook written as lenses-with-rationale + worked examples
            (deliberately NOT a rigid step-by-step procedure).
  3. SYNTHESIZE (--synthesize) - merge all per-creator playbooks into one unified
            playbook, noting consensus and disagreements.

Usage:
    python distill_theory.py "Salubrious Snail"
    python distill_theory.py "The Command Zone" --cap-videos 120
    python distill_theory.py --synthesize
"""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

import anthropic

DATA_DIR = Path(__file__).parent / "mtg_theory_data"
PLAYBOOK_DIR = Path(__file__).parent / "playbooks"
CACHE_DIR = DATA_DIR / "distill_cache"
COLLECTION = "mtg_deckbuilding_theory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAP_MODEL = "claude-haiku-4-5-20251001"   # cheap/fast for per-video extraction
REDUCE_MODEL = "claude-sonnet-5"          # quality for synthesis
MAP_WORKERS = 5                           # concurrent map calls

client = anthropic.Anthropic()


def reassemble_transcripts(source_name: str) -> list[dict]:
    """Rebuild per-video transcripts from the corpus chunks for one source."""
    import chromadb
    from chromadb.utils import embedding_functions
    c = chromadb.PersistentClient(path=str(DATA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    col = c.get_collection(COLLECTION, embedding_function=ef)
    got = col.get(where={"source": source_name}, include=["documents", "metadatas"])

    by_video = {}
    for _id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        vid, _, idx = _id.rpartition(":")
        try:
            idx = int(idx)
        except ValueError:
            idx = 0
        entry = by_video.setdefault(vid, {"title": meta.get("title", vid), "chunks": []})
        entry["chunks"].append((idx, doc))

    videos = []
    for vid, entry in by_video.items():
        entry["chunks"].sort(key=lambda t: t[0])
        text = " ".join(doc for _, doc in entry["chunks"])
        videos.append({"video_id": vid, "title": entry["title"], "text": text})
    return videos


def _ask(model, prompt, max_tokens):
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except anthropic.APIError as e:
            wait = 5 * (attempt + 1)
            print(f"    API error ({str(e)[:80]}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("repeated API failures")


MAP_PROMPT = """You are analyzing a transcript from {creator}, a Commander (EDH) deckbuilding \
content creator. Pull out ONLY what is actually present in this transcript:

1. PRINCIPLES / LENSES: deckbuilding principles, heuristics, or mental models they express \
- each with the WHY (the reasoning behind it), in their spirit.
2. WORKED EXAMPLES: any concrete "here's a situation/deck/card, here's what I'd do and why" \
reasoning - specific decisions, tradeoffs, cuts, or adds and the thinking behind them.

If this video isn't about deckbuilding theory (pure gameplay, news, unboxing), reply exactly: \
NO_THEORY

Keep it terse. Title: {title}

TRANSCRIPT:
{text}"""

REDUCE_PROMPT = """You are distilling the deckbuilding philosophy of {creator} from notes taken \
across {n} of their videos. Synthesize a PLAYBOOK that will be handed to an AI Commander \
deckbuilding advisor as operating instructions, so the advisor reasons the way {creator} does.

CRITICAL - avoid rigidity: do NOT write a step-by-step procedure or checklist. Good \
deckbuilding is judgment, not an algorithm. Write it so the advisor treats these as LENSES to \
weigh with discretion - applying the ones that matter for the situation and ignoring the rest.

Structure:
- HOW {creator} THINKS: 1-2 sentences on their overall philosophy/temperament.
- LENSES (weigh with judgment, not a checklist): each key principle/mental model, EACH WITH \
its rationale (the WHY). Make explicit that not every lens applies to every deck.
- RED FLAGS: things they'd call out as mistakes, with why.
- WORKED EXAMPLES: 2-4 concrete examples showing them applying judgment SITUATIONALLY - \
different problems foregrounding different lenses, showing restraint (not running every lens).
- A closing line reminding the advisor to diagnose what actually matters and go deep there.

Keep it tight and operational (~700-1000 words).

NOTES FROM THEIR VIDEOS:
{notes}"""

UNIFIED_PROMPT = """You are merging {n} per-creator Commander deckbuilding playbooks into ONE \
unified playbook for an AI deckbuilding advisor. Each was distilled from a respected creator's \
videos. Produce a single coherent playbook the advisor reasons with.

Requirements:
- KEEP the anti-rigid framing: these are LENSES to weigh with judgment, never a checklist/procedure.
- MERGE overlapping lenses into crisp shared principles; where creators genuinely DISAGREE or \
emphasize different things, note it briefly (that nuance is valuable).
- Attribute distinctive ideas to the creator when it adds credibility (e.g. "Rebel Lily's \
unit-of-value framing", "Salubrious Snail's floor/ceiling test").
- MANDATORY: a "WORKED EXAMPLES" section with 4-6 concrete examples (drawn from across the \
creators' playbooks) that show SITUATIONAL judgment - each example a specific deck/card problem \
where ONE or TWO lenses dominate and the rest are explicitly set aside. These examples are the \
most important part: they teach the advisor to diagnose and go deep rather than run a checklist. \
Do not omit or shorten this section.
- End with a reminder to diagnose what THIS deck needs and go deep on the few lenses that matter.

Keep it operational and readable (~1600-2200 words), and DO include the worked-examples section. \
This goes directly into a system prompt.

PER-CREATOR PLAYBOOKS:
{playbooks}"""


def _slug(name):
    return name.lower().replace(" ", "_").replace("/", "-")


def map_one(source_name, slug, v):
    cache = CACHE_DIR / f"{slug}__{v['video_id']}.txt"
    if cache.exists():
        return v, cache.read_text(encoding="utf-8")
    prompt = MAP_PROMPT.format(creator=source_name, title=v["title"], text=v["text"][:24000])
    note = _ask(MAP_MODEL, prompt, 1400).strip()
    cache.write_text(note, encoding="utf-8")
    return v, note


def distill_creator(source_name, cap_videos=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(source_name)

    print(f"\n### Distilling {source_name!r} ###", flush=True)
    videos = reassemble_transcripts(source_name)
    if cap_videos:
        videos = videos[:cap_videos]
    print(f"  {len(videos)} videos (map with {MAP_WORKERS} workers)", flush=True)

    notes, done = [], 0
    with ThreadPoolExecutor(max_workers=MAP_WORKERS) as ex:
        futures = [ex.submit(map_one, source_name, slug, v) for v in videos]
        for fut in as_completed(futures):
            v, note = fut.result()
            done += 1
            if done % 20 == 0 or done == len(videos):
                print(f"  mapped {done}/{len(videos)}", flush=True)
            if note.strip() != "NO_THEORY":
                notes.append(f"### {v['title']}\n{note}")

    print(f"  {len(notes)} usable; synthesizing playbook...", flush=True)
    combined = "\n\n".join(notes)
    playbook = _ask(REDUCE_MODEL, REDUCE_PROMPT.format(
        creator=source_name, n=len(notes), notes=combined[:180000]), 6000).strip()

    out = PLAYBOOK_DIR / f"{slug}.md"
    out.write_text(f"# {source_name} - Deckbuilding Playbook\n\n"
                   f"*Distilled from {len(notes)} videos. Lenses to weigh with judgment, not a checklist.*\n\n"
                   + playbook + "\n", encoding="utf-8")
    print(f"  wrote {out} ({len(playbook)} chars)", flush=True)


def synthesize():
    parts = []
    for p in sorted(PLAYBOOK_DIR.glob("*.md")):
        if p.name == "unified.md":
            continue
        parts.append(p.read_text(encoding="utf-8"))
    print(f"Synthesizing unified playbook from {len(parts)} creator playbooks...", flush=True)
    combined = "\n\n=====\n\n".join(parts)

    # Synthesis is non-deterministic and occasionally stops early; retry until the
    # output is actually complete (has the worked-examples section, ends on a full
    # sentence, and is long enough to contain all sections).
    unified = ""
    for attempt in range(4):
        unified = _ask(REDUCE_MODEL, UNIFIED_PROMPT.format(
            n=len(parts), playbooks=combined[:400000]), 9000).strip()
        complete = ("WORKED EXAMPLES" in unified.upper()
                    and len(unified) > 6000
                    and unified[-1] in ".!?)\"'")
        if complete:
            break
        print(f"  attempt {attempt + 1}: incomplete ({len(unified)} chars, "
              f"worked-examples={'yes' if 'WORKED EXAMPLES' in unified.upper() else 'no'}); retrying",
              flush=True)
    out = PLAYBOOK_DIR / "unified.md"
    out.write_text("# Unified Commander Deckbuilding Playbook\n\n"
                   "*Synthesized from the distilled playbooks of the corpus creators. "
                   "Lenses to weigh with judgment, not a checklist.*\n\n" + unified + "\n",
                   encoding="utf-8")
    print(f"Wrote {out} ({len(unified)} chars)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", help="Creator name to distill.")
    p.add_argument("--cap-videos", type=int, default=None, help="Max videos to map (samples the biggest channels).")
    p.add_argument("--synthesize", action="store_true", help="Merge all per-creator playbooks into unified.md.")
    args = p.parse_args()

    if args.synthesize:
        synthesize()
    elif args.source:
        distill_creator(args.source, cap_videos=args.cap_videos)
    else:
        print("Give a creator name, or --synthesize.")
