"""
MTG Commander Deckbuilding Theory Ingestion
===========================================
Builds the `mtg_deckbuilding_theory` ChromaDB collection that powers the
`deckbuilding_search` tool. For each YouTube channel in theory_sources.py it:

1. Enumerates the channel's uploads (yt-dlp, flat/fast).
2. Fetches each video's caption transcript (youtube-transcript-api).
3. Chunks the transcript and embeds it with the same local model as the rules RAG.
4. Stores chunks (with source/author/title/url metadata) in ChromaDB.

It is INCREMENTAL and resumable: a manifest tracks which videos are already done
(or have no transcript), so re-runs only fetch new videos.

Usage:
    python theory_ingestion.py                 # ingest all sources (incremental)
    python theory_ingestion.py --cap 10        # only the 10 most recent per channel
    python theory_ingestion.py --source "3/3 Elk"   # just one source
    python theory_ingestion.py --retry-missing # retry videos previously found with no transcript
    python theory_ingestion.py --force         # re-fetch/re-embed everything
"""

import os
import re
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, VideoUnplayable,
    AgeRestricted, InvalidVideoId, IpBlocked, RequestBlocked,
)

from theory_sources import SOURCES

# Errors that mean the video genuinely has no usable transcript (mark and move on).
_GENUINELY_MISSING = (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
    VideoUnplayable, AgeRestricted, InvalidVideoId,
)
# Errors that mean YouTube is throttling/blocking us (back off and retry, don't
# mark the video as missing).
_BLOCKING = (IpBlocked, RequestBlocked)


class Blocked(Exception):
    """Raised when YouTube is blocking/throttling transcript requests."""

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent / "mtg_theory_data"
COLLECTION_NAME = "mtg_deckbuilding_theory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MANIFEST_PATH = DATA_DIR / "ingested.json"

CHUNK_SIZE = 1200        # ~250-300 tokens
CHUNK_OVERLAP = 150
FETCH_DELAY = 1.2        # base politeness delay between transcript fetches (seconds)
BACKOFF_SCHEDULE = [20, 45, 90, 180]  # seconds to wait after a block, per retry

# Apify backend (transcripts fetched via Apify's proxies - no IP block, no cookies).
# Needs APIFY_TOKEN in the environment / .env. maxTotalChargeUsd caps spend so a run
# can never exceed the free monthly credit.
APIFY_ACTOR = "om_kh/youtube-transcript-api"
APIFY_MAX_USD = float(os.environ.get("APIFY_MAX_USD", "4.5"))
APIFY_BATCH = 40         # videos per Apify run

_transcript_api = YouTubeTranscriptApi()


# =============================================================================
# YOUTUBE
# =============================================================================

def enumerate_channel(channel_url: str, cap: int | None = None) -> list[dict]:
    """Return [{video_id, title, url}] for a channel's uploads (newest first)."""
    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if cap:
        opts["playlistend"] = cap
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    videos = []
    for entry in info.get("entries", []) or []:
        vid = entry.get("id")
        if not vid:
            continue
        videos.append({
            "video_id": vid,
            "title": entry.get("title") or vid,
            "url": entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
        })
    return videos


def fetch_transcript(video_id: str) -> str | None:
    """
    Fetch and flatten a video's caption transcript.

    Returns the text, or None if the video genuinely has no transcript.
    Raises Blocked if YouTube is throttling us (so the caller can back off
    instead of mislabelling the video as missing).
    """
    try:
        fetched = _transcript_api.fetch(video_id)
        text = " ".join(snippet.text for snippet in fetched).strip()
        return text or None
    except _GENUINELY_MISSING:
        return None
    except _BLOCKING as e:
        raise Blocked(str(e) or "IP blocked")
    except Exception as e:
        # Unknown/transient (HTTP error, request failed, PoToken required, ...).
        # Treat as blocking so we retry/stop rather than losing the video.
        raise Blocked(f"{type(e).__name__}: {e}")


def fetch_with_backoff(video_id: str) -> str | None:
    """fetch_transcript with retry+backoff on blocking; raises Blocked if it never clears."""
    last = None
    for wait in [0] + BACKOFF_SCHEDULE:
        if wait:
            print(f"      ...throttled; waiting {wait}s then retrying", flush=True)
            time.sleep(wait)
        try:
            return fetch_transcript(video_id)
        except Blocked as e:
            last = e
    raise Blocked(str(last))


def wait_until_unblocked(probe_video: str, interval: int, deadline: float | None) -> bool:
    """
    Poll (using probe_video) every `interval` seconds until YouTube stops blocking
    us. Returns True once unblocked, or False if the deadline passes first.
    """
    waited = 0
    while True:
        if deadline and time.time() > deadline:
            return False
        time.sleep(interval)
        waited += interval
        try:
            fetch_transcript(probe_video)  # returns text/None if NOT blocked
            print(f"      ...unblocked after ~{waited // 60} min", flush=True)
            return True
        except Blocked:
            mins = waited // 60
            print(f"      ...still blocked (waited ~{mins} min); probing again in {interval // 60} min", flush=True)


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split transcript text into overlapping, word-boundary-aware chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        # extend slightly to the next space so we don't cut mid-word
        if end < n:
            nxt = text.find(" ", end)
            if nxt != -1 and nxt - end < 40:
                end = nxt
        chunk = text[i:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


# =============================================================================
# STORAGE
# =============================================================================

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def get_collection():
    import chromadb
    from chromadb.utils import embedding_functions

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DATA_DIR))
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"description": "MTG Commander deckbuilding theory (video transcripts)"},
    )


# =============================================================================
# MAIN
# =============================================================================

def store_video(collection, manifest, source, v, text, grand):
    """Chunk + embed a video's transcript and record it in the manifest."""
    vid = v["video_id"]
    chunks = chunk_text(text) if text else []
    if not chunks:
        manifest[vid] = {"status": "no_transcript", "title": v["title"], "source": source["name"]}
        grand["no_transcript"] += 1
        print(f"  [no transcript] {v['title'][:70]}", flush=True)
        return
    ids = [f"{vid}:{i}" for i in range(len(chunks))]
    metadatas = [{
        "source": source["name"], "author": source["author"],
        "title": v["title"], "url": v["url"], "video_id": vid, "type": "video",
    } for _ in chunks]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    manifest[vid] = {"status": "ok", "title": v["title"], "source": source["name"], "chunks": len(chunks)}
    grand["transcribed"] += 1
    grand["chunks"] += len(chunks)
    print(f"  [ok {len(chunks):>3} chunks] {v['title'][:66]}", flush=True)


# ---- Apify backend (transcripts via Apify proxies; no IP block, no cookies) ----

def _item_video_id(item: dict):
    """Best-effort extraction of the 11-char YouTube id from an Apify result item."""
    for k in ("videoId", "video_id", "id"):
        val = item.get(k)
        if isinstance(val, str) and re.fullmatch(r"[A-Za-z0-9_-]{11}", val):
            return val
    for k in ("url", "video_url", "videoUrl", "link", "webpage_url"):
        u = item.get(k)
        if isinstance(u, str):
            m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", u)
            if m:
                return m.group(1)
    return None


def fetch_transcripts_apify(video_ids: list) -> dict:
    """Fetch a batch of transcripts via Apify. Returns {video_id: text|None}."""
    from apify_client import ApifyClient
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise RuntimeError("APIFY_TOKEN is not set. Add it to your .env "
                           "(Apify console -> Settings -> API tokens).")
    client = ApifyClient(token)
    run_input = {
        "videos": list(video_ids),
        "includeSegments": False,
        # YouTube blocks datacenter IPs; residential proxy is required to get through.
        "proxyConfiguration": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
        "maxTotalChargeUsd": APIFY_MAX_USD,
    }
    run = client.actor(APIFY_ACTOR).call(run_input=run_input)
    rd = run.model_dump() if hasattr(run, "model_dump") else dict(run)
    dataset_id = rd.get("default_dataset_id") or rd.get("defaultDatasetId")
    out = {}
    if not dataset_id:
        return out
    for item in client.dataset(dataset_id).iterate_items():
        vid = _item_video_id(item)
        text = (item.get("text") or "").strip() or None
        if vid:
            out[vid] = text
    return out


def run_apify(sources, cap, force, retry_missing, collection, manifest, grand):
    """Ingest via Apify: enumerate channels locally, fetch transcripts in batches."""
    for source in sources:
        name = source["name"]
        print(f"\n--- {name} (via Apify) ---", flush=True)
        try:
            videos = enumerate_channel(source["channel_url"], cap)
        except Exception as e:
            print(f"  Could not enumerate channel: {e}", flush=True)
            continue
        print(f"  {len(videos)} videos found", flush=True)

        pending = []
        for v in videos:
            grand["videos"] += 1
            prior = manifest.get(v["video_id"])
            if prior and not force:
                if prior.get("status") == "ok":
                    grand["skipped"] += 1
                    continue
                if prior.get("status") == "no_transcript" and not retry_missing:
                    grand["skipped"] += 1
                    continue
            pending.append(v)

        for i in range(0, len(pending), APIFY_BATCH):
            batch = pending[i:i + APIFY_BATCH]
            print(f"  fetching {len(batch)} transcripts via Apify...", flush=True)
            try:
                results = fetch_transcripts_apify([v["url"] for v in batch])
            except Exception as e:
                print(f"  Apify batch failed: {e}", flush=True)
                save_manifest(manifest)
                return
            for v in batch:
                store_video(collection, manifest, source, v, results.get(v["video_id"]), grand)
            save_manifest(manifest)


def _print_summary(grand, collection):
    print("\n" + "=" * 64)
    print("Done.")
    print(f"  Videos seen:        {grand['videos']}")
    print(f"  Newly transcribed:  {grand['transcribed']}  ({grand['chunks']} chunks)")
    print(f"  No transcript:      {grand['no_transcript']}")
    print(f"  Already done/skip:  {grand['skipped']}")
    try:
        print(f"  Collection size:    {collection.count()} chunks total")
    except Exception:
        pass
    print("Restart the advisor / MCP server so deckbuilding_search picks up the corpus.")
    print("=" * 64)


def main(cap=None, only_source=None, retry_missing=False, force=False,
         overnight=False, max_hours=10.0, probe_interval=600, backend="local"):
    print("=" * 64)
    print("MTG Deckbuilding Theory Ingestion" + ("  [OVERNIGHT MODE]" if overnight else ""))
    print("=" * 64, flush=True)

    collection = get_collection()
    manifest = load_manifest()

    # Overnight mode paces slower and waits out IP blocks instead of stopping.
    deadline = (time.time() + max_hours * 3600) if (overnight and max_hours) else None
    delay = 2.5 if overnight else FETCH_DELAY

    sources = SOURCES
    if only_source:
        sources = [s for s in SOURCES if s["name"].lower() == only_source.lower()]
        if not sources:
            print(f"No source named {only_source!r}. Options: {[s['name'] for s in SOURCES]}")
            return

    grand = {"videos": 0, "transcribed": 0, "no_transcript": 0, "skipped": 0, "chunks": 0}
    break_all = False

    if backend == "apify":
        run_apify(sources, cap, force, retry_missing, collection, manifest, grand)
        save_manifest(manifest)
        _print_summary(grand, collection)
        return

    for source in sources:
        name = source["name"]
        print(f"\n--- {name} ---")
        try:
            videos = enumerate_channel(source["channel_url"], cap)
        except Exception as e:
            print(f"  Could not enumerate channel: {e}")
            continue
        print(f"  {len(videos)} videos found")

        for v in videos:
            vid = v["video_id"]
            grand["videos"] += 1
            prior = manifest.get(vid)

            if prior and not force:
                if prior.get("status") == "ok":
                    grand["skipped"] += 1
                    continue
                if prior.get("status") == "no_transcript" and not retry_missing:
                    grand["skipped"] += 1
                    continue

            if overnight and deadline and time.time() > deadline:
                print("  Reached max runtime; stopping. Re-run to resume.", flush=True)
                break_all = True
                break

            text = None
            got = False
            while not got:
                try:
                    text = fetch_with_backoff(vid)
                    got = True
                except Blocked as e:
                    if not overnight:
                        print(f"\n  YouTube is blocking transcript requests right now ({e}).")
                        print("  Progress is saved. Wait ~30-60 min and re-run")
                        print("  `python theory_ingestion.py` - it resumes and skips finished videos.")
                        save_manifest(manifest)
                        break_all = True
                        break
                    # overnight: wait out the block, then retry this same video
                    print(f"  [blocked] waiting it out (probe every {probe_interval // 60} min)...", flush=True)
                    if not wait_until_unblocked(vid, probe_interval, deadline):
                        print("  Max runtime reached while blocked; stopping. Re-run to resume.", flush=True)
                        save_manifest(manifest)
                        break_all = True
                        break
            if break_all:
                break
            time.sleep(delay)

            if not text:
                manifest[vid] = {"status": "no_transcript", "title": v["title"], "source": name}
                grand["no_transcript"] += 1
                print(f"  [no transcript] {v['title'][:70]}")
                continue

            store_video(collection, manifest, source, v, text, grand)
            save_manifest(manifest)

        if break_all:
            break

    save_manifest(manifest)
    _print_summary(grand, collection)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest Commander deckbuilding theory videos.")
    p.add_argument("--cap", type=int, default=None, help="Max videos per channel (newest first).")
    p.add_argument("--source", default=None, help="Only ingest this source by name.")
    p.add_argument("--retry-missing", action="store_true", help="Retry videos previously found with no transcript.")
    p.add_argument("--force", action="store_true", help="Re-fetch and re-embed everything.")
    p.add_argument("--overnight", action="store_true",
                   help="Run slow and wait out IP blocks (poll until unblocked) instead of stopping.")
    p.add_argument("--max-hours", type=float, default=10.0, help="Max runtime in overnight mode (default 10).")
    p.add_argument("--probe-interval", type=int, default=600, help="Seconds between unblock probes (default 600).")
    p.add_argument("--backend", choices=["local", "apify"], default="local",
                   help="Transcript source: 'local' (youtube-transcript-api) or 'apify' (needs APIFY_TOKEN).")
    args = p.parse_args()
    main(cap=args.cap, only_source=args.source, retry_missing=args.retry_missing, force=args.force,
         overnight=args.overnight, max_hours=args.max_hours, probe_interval=args.probe_interval,
         backend=args.backend)
