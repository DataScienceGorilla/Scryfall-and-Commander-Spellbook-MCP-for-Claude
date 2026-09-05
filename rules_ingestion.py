"""
MTG Comprehensive Rules Ingestion Script
=========================================
Downloads the MTG Comprehensive Rules, chunks by rule number,
and stores embeddings in ChromaDB for semantic search.

Usage:
    python rules_ingestion.py

This will:
1. Download the latest Comprehensive Rules TXT file
2. Parse and chunk rules by their rule numbers (e.g., 704.5k)
3. Generate embeddings using sentence-transformers
4. Store everything in a local ChromaDB database

Run this whenever you want to update to the latest rules.
"""

import re
import os
import sys
import argparse
import httpx
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Official page that lists the current rules downloads. The script scrapes this
# to auto-discover the latest Comprehensive Rules URL, so it stays current on its
# own as new sets are released.
RULES_PAGE_URL = "https://magic.wizards.com/en/rules"

# Fallback URL used only if auto-discovery fails (e.g. the page layout changed
# or there's no network). Bump the date here as a manual safety net.
FALLBACK_RULES_URL = "https://media.wizards.com/2026/downloads/MagicCompRules%2020260819.txt"

# Directory to store the ChromaDB database
# This will be created next to the script
DATA_DIR = Path(__file__).parent / "mtg_rules_data"

# Tracks which rules version (YYYYMMDD) is currently ingested, so re-runs can
# skip the work when nothing has changed.
VERSION_FILE = DATA_DIR / "version.txt"

# ChromaDB collection name
COLLECTION_NAME = "mtg_comprehensive_rules"

# Embedding model - this one is small, fast, and works well for this use case
# It runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# RULES PARSING
# =============================================================================

def version_from_url(url: str) -> str:
    """Pulls the YYYYMMDD version stamp out of a rules URL, or 'unknown'."""
    match = re.search(r'(\d{8})\.txt', url)
    return match.group(1) if match else "unknown"


def discover_latest_url() -> tuple[str, str] | None:
    """
    Scrapes the official rules page for the latest Comprehensive Rules TXT link.

    Returns (url, version_date) on success, or None if the link can't be found
    (in which case the caller should fall back to FALLBACK_RULES_URL).
    """
    headers = {"User-Agent": "MTG-MCP-RulesIngestion/1.0"}
    try:
        resp = httpx.get(RULES_PAGE_URL, headers=headers, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
    except Exception as e:
        print(f"Auto-discovery failed to reach {RULES_PAGE_URL}: {e}")
        return None

    # Links look like .../MagicCompRules 20260819.txt — the space may be encoded
    # as %20, a '+', or a literal space depending on the page.
    match = re.search(
        r'https://media\.wizards\.com/(\d{4})/downloads/MagicCompRules(?:%20|\+|\s)*(\d{8})\.txt',
        resp.text,
    )
    if not match:
        print("Auto-discovery: no Comprehensive Rules TXT link found on the page.")
        return None

    year, date = match.group(1), match.group(2)
    url = f"https://media.wizards.com/{year}/downloads/MagicCompRules%20{date}.txt"
    return url, date


def read_installed_version() -> str | None:
    """Returns the version stamp of the currently ingested rules, if any."""
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def write_installed_version(version: str) -> None:
    """Records the version stamp of the rules we just ingested."""
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(version, encoding="utf-8")


def download_rules(url: str) -> str:
    """
    Downloads the Comprehensive Rules TXT file.
    
    Returns the full text content of the rules document.
    """
    print(f"Downloading rules from {url}...")
    
    # Need a User-Agent header or some servers reject the request
    headers = {"User-Agent": "MTG-MCP-RulesIngestion/1.0"}
    
    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    
    # The rules file is usually UTF-8, but let's be safe
    content = response.text
    print(f"Downloaded {len(content):,} characters")
    
    return content


def parse_rules(content: str) -> list[dict]:
    """
    Parses the Comprehensive Rules into individual rule chunks.
    
    Each chunk contains:
    - rule_number: The rule identifier (e.g., "704.5k", "302.6")
    - text: The full text of the rule (with section context prepended)
    - section: The major section number (e.g., "7" for state-based actions)

    The chunking strategy:
    - Each numbered rule becomes its own chunk
    - Subrules (like 704.5a, 704.5b) are kept as separate chunks
    - Section headers are prepended to give context
    - Examples within rules are kept with their parent rule
    - The glossary entries are also chunked individually
    """
    chunks = []
    
    # First, extract section headers (e.g., "704. State-Based Actions")
    # These help provide context for individual rules
    section_headers = {}
    header_pattern = re.compile(r'^(\d{3})\.\s+([A-Z][^\n]+)', re.MULTILINE)
    for match in header_pattern.finditer(content):
        section_num = match.group(1)
        section_name = match.group(2).strip()
        section_headers[section_num] = section_name

    # Pattern to match rule numbers like "100.1", "704.5k", "702.16a"
    # Rule numbers start at the beginning of a line
    rule_pattern = re.compile(
        r'^(\d{3}\.\d+[a-z]?)\s+(.+?)(?=^\d{3}\.\d+[a-z]?\s|\Z)',
        re.MULTILINE | re.DOTALL
    )
    
    # Find all rules in the main body
    for match in rule_pattern.finditer(content):
        rule_number = match.group(1)
        rule_text = match.group(2).strip()
        
        # Extract the major section (first three digits)
        section = rule_number.split('.')[0]
        
        # Clean up the text - remove excessive whitespace
        rule_text = re.sub(r'\s+', ' ', rule_text)
        
        # Skip very short rules (usually just headers)
        if len(rule_text) < 20:
            continue
        
        # Prepend section context for better semantic search
        # e.g., "State-Based Actions (704): If a creature has 0 toughness..."
        section_name = section_headers.get(section, "")
        if section_name:
            contextualized_text = f"{section_name} ({section}): {rule_text}"
        else:
            contextualized_text = rule_text

        chunks.append({
            "rule_number": rule_number,
            "text": contextualized_text,
            "section": section,
            "section_name": section_name
        })
    
    # Also parse the glossary section
    # Glossary entries look like: "Term\nDefinition..."
    glossary_start = content.find("Glossary")
    if glossary_start != -1:
        glossary_content = content[glossary_start:]
        
        # Glossary entries are separated by blank lines
        # Each entry starts with a capitalized term
        glossary_pattern = re.compile(
            r'^([A-Z][A-Za-z\s,\'-]+)\n(.+?)(?=^[A-Z][A-Za-z\s,\'-]+\n|\Z)',
            re.MULTILINE | re.DOTALL
        )
        
        for match in glossary_pattern.finditer(glossary_content):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Clean up
            definition = re.sub(r'\s+', ' ', definition)
            
            # Skip if too short or if it's not a real definition
            if len(definition) < 20 or term in ["Glossary", "Credits"]:
                continue
            
            chunks.append({
                "rule_number": f"glossary:{term}",
                "text": f"{term}: {definition}",
                "section": "glossary"
            })
    
    print(f"Parsed {len(chunks)} rule chunks")
    return chunks


# =============================================================================
# CHROMADB STORAGE
# =============================================================================

def create_database(chunks: list[dict], data_dir: Path):
    """
    Creates a ChromaDB database and stores all rule chunks with embeddings.
    
    ChromaDB handles the embedding generation automatically using
    the sentence-transformers model we specify.
    """
    print(f"Creating database in {data_dir}...")
    
    # Create the data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB with persistent storage
    # This saves the database to disk so it persists between runs
    client = chromadb.PersistentClient(path=str(data_dir))
    
    # Set up the embedding function using sentence-transformers
    # This model runs locally - no API key needed
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Delete existing collection if it exists (for clean re-ingestion)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except:
        pass
    
    # Create the collection with our embedding function
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"description": "MTG Comprehensive Rules"}
    )
    
    # Prepare the data for insertion
    # ChromaDB wants lists of ids, documents, and metadatas
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        ids.append(f"rule_{i}")
        documents.append(chunk["text"])
        metadatas.append({
            "rule_number": chunk["rule_number"],
            "section": chunk["section"],
            "section_name": chunk.get("section_name", "")
        })
    
    # Add all chunks to the collection
    # ChromaDB will automatically generate embeddings
    print(f"Generating embeddings and storing {len(chunks)} chunks...")
    print("(This may take a minute on first run as it downloads the model)")
    
    # Add in batches to show progress
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"  Processed {end}/{len(ids)} chunks")
    
    print(f"Database created successfully!")
    print(f"Location: {data_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main(force: bool = False, url_override: str | None = None):
    """
    Main entry point - discovers the latest rules, and (re)builds the database
    if a newer version is available.

    Args:
        force: Re-ingest even if the installed version already matches.
        url_override: Ingest this exact URL instead of auto-discovering.
    """
    print("=" * 60)
    print("MTG Comprehensive Rules Ingestion")
    print("=" * 60)
    print()

    # Step 0: Figure out which rules version we should have
    if url_override:
        url = url_override
        version = version_from_url(url)
        print(f"Using provided URL (version {version}).")
    else:
        discovered = discover_latest_url()
        if discovered:
            url, version = discovered
            print(f"Latest published rules: {version}")
        else:
            url = FALLBACK_RULES_URL
            version = version_from_url(url)
            print(f"Falling back to pinned URL (version {version}).")

    installed = read_installed_version()
    print(f"Currently ingested:     {installed or 'none'}")

    if installed == version and version != "unknown" and not force:
        print("\nAlready up to date. Nothing to do.")
        print("(Run with --force to rebuild anyway.)")
        return

    # Step 1: Download the rules
    try:
        content = download_rules(url)
    except Exception as e:
        print(f"Error downloading rules: {e}")
        print("\nYou can manually download the rules from:")
        print("https://magic.wizards.com/en/rules")
        print("\nThen save as 'MagicCompRules.txt' in this directory")

        # Try to load from local file as fallback
        local_file = Path(__file__).parent / "MagicCompRules.txt"
        if local_file.exists():
            print(f"\nFound local file: {local_file}")
            content = local_file.read_text(encoding='utf-8')
        else:
            return

    # Step 2: Parse into chunks
    chunks = parse_rules(content)
    
    if not chunks:
        print("Error: No rules were parsed. The file format may have changed.")
        return
    
    # Show some stats
    sections = {}
    for chunk in chunks:
        section = chunk["section"]
        sections[section] = sections.get(section, 0) + 1
    
    print("\nChunks by section:")
    for section, count in sorted(sections.items()):
        print(f"  Section {section}: {count} rules")
    
    # Step 3: Create the database
    print()
    create_database(chunks, DATA_DIR)

    # Step 4: Record which version we just ingested
    write_installed_version(version)

    print()
    print("=" * 60)
    print(f"Done! Rules database is ready (version {version}).")
    print("You can now use the mtg_rules_search tool in the MCP server.")
    print("Restart the MCP server so it picks up the fresh database.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and index the MTG Comprehensive Rules for semantic search."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest even if the installed version is already current.",
    )
    parser.add_argument(
        "--url", default=None,
        help="Ingest this exact rules TXT URL instead of auto-discovering.",
    )
    args = parser.parse_args()
    main(force=args.force, url_override=args.url)