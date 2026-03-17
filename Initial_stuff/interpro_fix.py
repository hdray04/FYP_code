"""
DROP-IN REPLACEMENT FUNCTIONS for your existing script.

Replace fetch_interpro_entry, get_superfamily_hierarchy, and the
domain_hierarchy building block with these. Everything else in your
script (loading UniProt data, visualisation, CSV export) stays the same.

KEY FIXES:
1. Entry type is at metadata.type (a string), not type.code
2. Hierarchy walking uses metadata.relationships.parents (not metadata.hierarchy.parents)
3. The walk is now recursive — it climbs until it finds a Homologous Superfamily
   or runs out of parents
4. A curated keyword fallback handles anything the hierarchy can't resolve
"""

import requests
import time
import json
import os
from collections import Counter, defaultdict

# ── Constants ──────────────────────────────────────────────────────────────────

INTERPRO_API  = "https://www.ebi.ac.uk/interpro/api"
CACHE_FILE    = "interpro_cache.json"
RATE_LIMIT    = 0.15   # seconds between API calls

# Curated fallback: substring (lower-case) → clean structural class label
# Order matters — first match wins
KEYWORD_MAP = [
    ("homeodomain",           "Homeodomain"),
    ("homeobox",              "Homeodomain"),
    ("homedomain",            "Homeodomain"),          # typo present in InterPro
    ("winged helix",          "Winged-helix / Forkhead"),
    ("fork head",             "Winged-helix / Forkhead"),
    ("forkhead",              "Winged-helix / Forkhead"),
    ("hmg box",               "HMG box"),
    ("high mobility group",   "HMG box"),
    ("zinc finger",           "Zinc finger"),
    ("znf",                   "Zinc finger"),
    ("zf-",                   "Zinc finger"),
    ("nuclear hormone",       "Nuclear receptor"),
    ("nuclear receptor",      "Nuclear receptor"),
    ("nhr",                   "Nuclear receptor"),
    ("ets domain",            "ETS domain"),
    ("ets family",            "ETS domain"),
    ("helix-turn-helix",      "Helix-turn-helix"),
    ("hth",                   "Helix-turn-helix"),
    ("lambda repressor",      "Helix-turn-helix"),
    ("p53",                   "p53-like"),
    ("runt",                  "Runt domain"),
    ("t-box",                 "T-box"),
    ("bzip",                  "bZIP"),
    ("leucine zipper",        "bZIP"),
    ("bhlh",                  "bHLH"),
    ("helix-loop-helix",      "bHLH"),
    ("paired domain",         "Paired domain"),
    ("pax",                   "Paired domain"),
    ("gata",                  "GATA zinc finger"),
    ("cold shock",            "Cold shock domain"),
    ("sox",                   "HMG box"),               # SOX proteins use HMG
    ("integrase",             "⚠ Transposable element"),
    ("transposase",           "⚠ Transposable element"),
    ("reverse transcriptase", "⚠ Transposable element"),
    ("gag",                   "⚠ Transposable element"),
]

# ── Cache helpers ──────────────────────────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ── API fetch (replaces your fetch_interpro_entry) ────────────────────────────

def fetch_interpro_entry(domain_id, cache=None):
    """
    Fetch InterPro entry. Uses cache to avoid redundant API calls.
    Returns the full JSON dict, or None on failure.
    """
    if cache is not None and domain_id in cache:
        return cache[domain_id]  # None is a valid cached value (404 etc.)

    url = f"{INTERPRO_API}/entry/interpro/{domain_id}/?format=json"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
        elif r.status_code == 404:
            data = None
        else:
            print(f"  [WARN] {domain_id}: HTTP {r.status_code}")
            return None          # don't cache transient errors
        time.sleep(RATE_LIMIT)
        if cache is not None:
            cache[domain_id] = data
        return data
    except Exception as e:
        print(f"  [WARN] Error fetching {domain_id}: {e}")
        return None

# ── Entry-type reader ─────────────────────────────────────────────────────────

def get_entry_type(entry_data):
    """
    Return the InterPro entry type string, e.g. 'homologous_superfamily'.
    The API returns this at metadata.type — it is a plain string, not a dict.
    """
    if not entry_data:
        return None
    return entry_data.get("metadata", {}).get("type", None)

def get_entry_name(entry_data):
    if not entry_data:
        return ""
    name_field = entry_data.get("metadata", {}).get("name", {})
    # name can be a dict {"name": "...", "short": "..."} or occasionally a string
    if isinstance(name_field, dict):
        return name_field.get("name", "")
    return str(name_field)

def get_parent_ids(entry_data):
    """
    Return list of parent InterPro accessions.
    InterPro API stores these at metadata.relationships.parents
    """
    if not entry_data:
        return []
    relationships = entry_data.get("metadata", {}).get("relationships", {})
    parents = relationships.get("parents", [])
    # Each parent is a dict like {"accession": "IPR009057", "name": {...}}
    return [p["accession"] for p in parents if isinstance(p, dict) and "accession" in p]

# ── Recursive hierarchy walk (replaces get_superfamily_hierarchy) ─────────────

def find_superfamily(domain_id, cache, _visited=None):
    """
    Walk up the InterPro hierarchy from domain_id.
    Returns (superfamily_id, superfamily_name) if a homologous_superfamily
    ancestor is found, else (None, None).

    This is the key function that was missing recursive climbing in the
    original script.
    """
    if _visited is None:
        _visited = set()
    if domain_id in _visited:   # cycle guard
        return None, None
    _visited.add(domain_id)

    entry = fetch_interpro_entry(domain_id, cache)
    if entry is None:
        return None, None

    entry_type = get_entry_type(entry)

    # Found a homologous superfamily — return it
    if entry_type == "homologous_superfamily":
        return domain_id, get_entry_name(entry)

    # Climb to each parent, return first superfamily found
    for parent_id in get_parent_ids(entry):
        sf_id, sf_name = find_superfamily(parent_id, cache, _visited)
        if sf_id:
            return sf_id, sf_name

    return None, None

# ── Keyword fallback ──────────────────────────────────────────────────────────

def keyword_label(name):
    """Return structural class from name via keyword matching, or 'Unclassified'."""
    lower = name.lower()
    for kw, label in KEYWORD_MAP:
        if kw in lower:
            return label
    return "Unclassified"

# ── Master resolver ───────────────────────────────────────────────────────────

def resolve_domain_to_class(domain_id, cache):
    """
    Returns (structural_class_label, method) for a single InterPro domain.
    method is one of: 'hierarchy', 'keyword', 'unresolved'
    """
    sf_id, sf_name = find_superfamily(domain_id, cache)
    if sf_name:
        return sf_name, "hierarchy"

    # Fallback: keyword match on this entry's own name
    entry = fetch_interpro_entry(domain_id, cache)
    name  = get_entry_name(entry) if entry else domain_id
    label = keyword_label(name)
    method = "keyword" if label not in ("Unclassified",) else "unresolved"
    return label, method

# ── Replacement main processing block ─────────────────────────────────────────
# Replace everything from "# Fetch hierarchy information from InterPro API"
# down to "# Create superfamily mapping" with this block.

def build_domain_class_map(unique_domains):
    """
    For all unique_domains, return:
      domain_class  : {ipr_id -> structural_class_label}
      domain_method : {ipr_id -> resolution_method}
    """
    cache = load_cache()

    domain_class  = {}
    domain_method = {}
    failed        = []

    batch = 50
    for i, domain_id in enumerate(unique_domains):
        if i % batch == 0:
            print(f"  Processing {i+1}–{min(i+batch, len(unique_domains))} "
                  f"of {len(unique_domains)}")
            save_cache(cache)   # checkpoint every batch

        label, method = resolve_domain_to_class(domain_id, cache)
        domain_class[domain_id]  = label
        domain_method[domain_id] = method
        if method == "unresolved":
            failed.append(domain_id)

    save_cache(cache)

    # Resolution summary
    method_counts = Counter(domain_method.values())
    print("\nResolution summary:")
    for m, c in method_counts.items():
        print(f"  {m:12s}: {c:4d} ({100*c/len(unique_domains):.1f}%)")
    print(f"  Unresolved   : {len(failed)} domains")

    return domain_class, domain_method


def assign_proteins_to_classes(protein_domains, domain_class):
    """
    For each protein assign a primary structural class.
    Returns superfamily_counts (Counter) and superfamily_proteins (dict of sets)
    matching the shape your existing visualisation code expects.
    """
    superfamily_counts   = Counter()
    superfamily_proteins = defaultdict(set)
    protein_primary      = {}   # protein_id -> primary class (useful for export)

    for protein_id, domains in protein_domains.items():
        labels = [domain_class.get(d, "Unclassified") for d in domains]

        # Separate out TE-flagged
        te_flagged = [l for l in labels if l.startswith("⚠")]
        clean      = [l for l in labels
                      if not l.startswith("⚠") and l != "Unclassified"]

        # Primary class = most frequent clean label; fall back progressively
        if clean:
            primary = Counter(clean).most_common(1)[0][0]
        elif te_flagged:
            primary = "⚠ Transposable element"
        else:
            primary = "Unclassified"

        protein_primary[protein_id] = primary
        superfamily_counts[primary] += 1
        superfamily_proteins[primary].add(protein_id)

    return superfamily_counts, superfamily_proteins, protein_primary


# ── HOW TO INTEGRATE INTO YOUR EXISTING SCRIPT ────────────────────────────────
#
# 1. Paste all functions above into your script (replacing the old
#    fetch_interpro_entry and get_superfamily_hierarchy).
#
# 2. Replace the block from:
#      "# Fetch hierarchy information from InterPro API"
#    to just before:
#      "# Generate results and visualizations"
#    with:
#
#      domain_class, domain_method = build_domain_class_map(unique_domains)
#      superfamily_counts, superfamily_proteins, protein_primary = \
#          assign_proteins_to_classes(protein_domains, domain_class)
#
# 3. Your visualisation and CSV export code below that line needs one small
#    tweak — the superfamily labels are now clean strings like "Homeodomain"
#    rather than "IPR001356 (Unknown: {...})".
#    The label-simplification logic in your pie/bar chart (splitting on '(')
#    is no longer needed and can be removed.
#
# 4. To flag TE proteins for exclusion in your enrichment analysis:
#      te_proteins = {pid for pid, cls in protein_primary.items()
#                     if cls == "⚠ Transposable element"}
#
# ── QUICK STANDALONE TEST (runs without your data file) ───────────────────────

if __name__ == "__main__":
    test_domains = [
        "IPR001356",   # Homeodomain         → should resolve to Homeodomain-like sf
        "IPR001766",   # Fork head domain     → should resolve to Winged-helix sf
        "IPR000418",   # Ets domain           → should climb to ETS sf
        "IPR009071",   # HMG box              → should resolve to HMG sf
        "IPR001628",   # Zinc finger NHR      → Nuclear receptor sf
    ]

    print("Testing hierarchy resolution on 5 domains...\n")
    cache = load_cache()
    for ipr in test_domains:
        label, method = resolve_domain_to_class(ipr, cache)
        print(f"  {ipr}  →  {label}  [{method}]")
    save_cache(cache)
    print("\nDone. Check interpro_cache.json for cached results.")
