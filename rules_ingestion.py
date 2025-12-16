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
import httpx
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# URL to the Comprehensive Rules TXT file
# Update this URL when new rules are released
RULES_URL = "https://media.wizards.com/2025/downloads/MagicCompRules%2020251114.txt"

# Directory to store the ChromaDB database
# This will be created next to the script
DATA_DIR = Path(__file__).parent / "mtg_rules_data"

# ChromaDB collection name
COLLECTION_NAME = "mtg_comprehensive_rules"

# Embedding model - this one is small, fast, and works well for this use case
# It runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# RULES PARSING
# =============================================================================

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

def main():
    """
    Main entry point - downloads rules, parses them, and creates the database.
    """
    print("=" * 60)
    print("MTG Comprehensive Rules Ingestion")
    print("=" * 60)
    print()
    
    # Step 1: Download the rules
    try:
        content = download_rules(RULES_URL)
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
    
    print()
    print("=" * 60)
    print("Done! The rules database is ready to use.")
    print("You can now use the mtg_rules_search tool in the MCP server.")
    print("=" * 60)


if __name__ == "__main__":
    main()