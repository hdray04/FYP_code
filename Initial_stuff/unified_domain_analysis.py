"""
Unified InterPro Domain Analysis Script
Combines robust API hierarchy resolution with comprehensive visualization.

This script:
1. Loads DNA-binding proteins from UniProt JSON
2. Uses InterPro API to resolve domains to structural superfamilies
3. Creates multiple visualizations of the results
4. Exports detailed CSV data
"""

import json
import requests
import time
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd

# ── InterPro API Configuration ──────────────────────────────────────────────

INTERPRO_API = "https://www.ebi.ac.uk/interpro/api"
CACHE_FILE = "interpro_cache.json"
RATE_LIMIT = 0.15  # seconds between API calls

# Curated fallback mapping for keyword-based classification
KEYWORD_MAP = [
    ("homeodomain", "Homeodomain"),
    ("homeobox", "Homeodomain"),
    ("homedomain", "Homeodomain"),
    ("winged helix", "Winged-helix / Forkhead"),
    ("fork head", "Winged-helix / Forkhead"),
    ("forkhead", "Winged-helix / Forkhead"),
    ("hmg box", "HMG box"),
    ("high mobility group", "HMG box"),
    ("zinc finger", "Zinc finger"),
    ("znf", "Zinc finger"),
    ("zf-", "Zinc finger"),
    ("nuclear hormone", "Nuclear receptor"),
    ("nuclear receptor", "Nuclear receptor"),
    ("nhr", "Nuclear receptor"),
    ("ets domain", "ETS domain"),
    ("ets family", "ETS domain"),
    ("helix-turn-helix", "Helix-turn-helix"),
    ("hth", "Helix-turn-helix"),
    ("lambda repressor", "Helix-turn-helix"),
    ("p53", "p53-like"),
    ("runt", "Runt domain"),
    ("t-box", "T-box"),
    ("bzip", "bZIP"),
    ("leucine zipper", "bZIP"),
    ("bhlh", "bHLH"),
    ("helix-loop-helix", "bHLH"),
    ("paired domain", "Paired domain"),
    ("pax", "Paired domain"),
    ("gata", "GATA zinc finger"),
    ("cold shock", "Cold shock domain"),
    ("sox", "HMG box"),
    ("integrase", "⚠ Transposable element"),
    ("transposase", "⚠ Transposable element"),
    ("reverse transcriptase", "⚠ Transposable element"),
    ("gag", "⚠ Transposable element"),
]

# ── Cache Management ────────────────────────────────────────────────────────

def load_cache():
    """Load cached InterPro API responses."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ── InterPro API Functions ──────────────────────────────────────────────────

def fetch_interpro_entry(domain_id, cache=None):
    """Fetch InterPro entry with caching support."""
    if cache is not None and domain_id in cache:
        return cache[domain_id]

    url = f"{INTERPRO_API}/entry/interpro/{domain_id}/?format=json"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
        elif response.status_code == 404:
            data = None
        else:
            print(f"  [WARN] {domain_id}: HTTP {response.status_code}")
            return None
        
        time.sleep(RATE_LIMIT)
        if cache is not None:
            cache[domain_id] = data
        return data
    except Exception as e:
        print(f"  [WARN] Error fetching {domain_id}: {e}")
        return None

def get_entry_type(entry_data):
    """Extract entry type from InterPro response."""
    if not entry_data:
        return None
    return entry_data.get("metadata", {}).get("type", None)

def get_entry_name(entry_data):
    """Extract entry name from InterPro response."""
    if not entry_data:
        return ""
    name_field = entry_data.get("metadata", {}).get("name", {})
    if isinstance(name_field, dict):
        return name_field.get("name", "")
    return str(name_field)

def get_parent_ids(entry_data):
    """Extract parent InterPro IDs from hierarchy."""
    if not entry_data:
        return []
    relationships = entry_data.get("metadata", {}).get("relationships", {})
    parents = relationships.get("parents", [])
    return [p["accession"] for p in parents if isinstance(p, dict) and "accession" in p]

def find_superfamily(domain_id, cache, _visited=None):
    """Recursively walk up hierarchy to find homologous superfamily."""
    if _visited is None:
        _visited = set()
    if domain_id in _visited:  # prevent cycles
        return None, None
    _visited.add(domain_id)

    entry = fetch_interpro_entry(domain_id, cache)
    if entry is None:
        return None, None

    entry_type = get_entry_type(entry)

    # Found a homologous superfamily - return it
    if entry_type == "homologous_superfamily":
        return domain_id, get_entry_name(entry)

    # Climb to parents and return first superfamily found
    for parent_id in get_parent_ids(entry):
        sf_id, sf_name = find_superfamily(parent_id, cache, _visited)
        if sf_id:
            return sf_id, sf_name

    return None, None

def keyword_label(name):
    """Classify domain by keyword matching."""
    lower = name.lower()
    for keyword, label in KEYWORD_MAP:
        if keyword in lower:
            return label
    return "Unclassified"

def resolve_domain_to_class(domain_id, cache):
    """Resolve single domain to structural class."""
    # Try hierarchy first
    sf_id, sf_name = find_superfamily(domain_id, cache)
    if sf_name:
        return sf_name, "hierarchy"

    # Fallback to keyword matching
    entry = fetch_interpro_entry(domain_id, cache)
    name = get_entry_name(entry) if entry else domain_id
    label = keyword_label(name)
    method = "keyword" if label != "Unclassified" else "unresolved"
    return label, method

# ── Domain Classification Pipeline ──────────────────────────────────────────

def build_domain_class_map(unique_domains):
    """Build mapping from domains to structural classes."""
    cache = load_cache()
    
    # Check which domains we already have data for
    cached_domains = set(cache.keys())
    missing_domains = [d for d in unique_domains if d not in cached_domains]
    
    print(f"Domain resolution status:")
    print(f"  Total domains: {len(unique_domains)}")
    print(f"  Already cached: {len(cached_domains & set(unique_domains))}")
    print(f"  Need to fetch: {len(missing_domains)}")
    
    if missing_domains:
        print(f"\nFetching data for {len(missing_domains)} new domains...")
        batch_size = 50
        for i, domain_id in enumerate(missing_domains):
            if i % batch_size == 0:
                print(f"  Processing {i+1}–{min(i+batch_size, len(missing_domains))} of {len(missing_domains)}")
                save_cache(cache)  # checkpoint
            
            # Fetch only missing domains
            fetch_interpro_entry(domain_id, cache)
        
        save_cache(cache)
        print(f"Completed API fetching. Cache now contains {len(cache)} domains.")
    else:
        print("All domains already cached - no API calls needed!")
    
    # Now resolve all domains to classes
    print("\nResolving domains to structural classes...")
    domain_class = {}
    domain_method = {}
    failed = []
    
    for domain_id in unique_domains:
        label, method = resolve_domain_to_class(domain_id, cache)
        domain_class[domain_id] = label
        domain_method[domain_id] = method
        if method == "unresolved":
            failed.append(domain_id)
    
    # Print resolution summary
    method_counts = Counter(domain_method.values())
    print("\nResolution summary:")
    for method, count in method_counts.items():
        print(f"  {method:12s}: {count:4d} ({100*count/len(unique_domains):.1f}%)")
    if failed:
        print(f"  Unresolved: {len(failed)} domains")
    
    return domain_class, domain_method

def assign_proteins_to_classes(protein_domains, domain_class):
    """Assign each protein to a primary structural class."""
    superfamily_counts = Counter()
    superfamily_proteins = defaultdict(set)
    protein_primary = {}
    
    for protein_id, domains in protein_domains.items():
        labels = [domain_class.get(d, "Unclassified") for d in domains]
        
        # Separate transposable elements from clean labels
        te_flagged = [l for l in labels if l.startswith("⚠")]
        clean = [l for l in labels if not l.startswith("⚠") and l != "Unclassified"]
        
        # Primary class = most frequent clean label
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

# ── Main Analysis and Visualization ─────────────────────────────────────────

def main():
    print("="*80)
    print("UNIFIED INTERPRO DOMAIN ANALYSIS")
    print("="*80)
    
    # Load UniProt data
    print("\nLoading UniProt DNA-binding protein data...")
    with open('./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json') as f:
        data = json.load(f)
    
    proteins = data['results']
    print(f"Loaded {len(proteins)} proteins")
    
    # Extract InterPro domains
    print("\nExtracting InterPro domains...")
    all_domains = []
    protein_domains = {}
    
    for protein in proteins:
        protein_id = protein['primaryAccession']
        xrefs = protein.get('uniProtKBCrossReferences', [])
        interpro_domains = [x['id'] for x in xrefs if x.get('database') == 'InterPro']
        
        protein_domains[protein_id] = interpro_domains
        all_domains.extend(interpro_domains)
    
    unique_domains = list(set(all_domains))
    domain_counts = Counter(all_domains)
    
    print(f"Found {len(unique_domains)} unique InterPro domains")
    print(f"Total domain instances: {len(all_domains)}")
    
    # Classify domains using hierarchy
    print(f"\nFetching hierarchy data from InterPro API...")
    print("This will take several minutes due to API rate limiting...")
    
    domain_class, domain_method = build_domain_class_map(unique_domains)
    superfamily_counts, superfamily_proteins, protein_primary = assign_proteins_to_classes(protein_domains, domain_class)
    
    # Analysis results
    print(f"\n{'='*80}")
    print("STRUCTURAL SUPERFAMILY ANALYSIS")
    print(f"{'='*80}")
    
    print(f"Unique structural classes found: {len(superfamily_counts)}")
    print(f"Total proteins classified: {sum(superfamily_counts.values())}")
    
    print(f"\nTop 20 Structural Classes:")
    for i, (superfamily, count) in enumerate(superfamily_counts.most_common(20), 1):
        protein_count = len(superfamily_proteins[superfamily])
        percent = (count / len(proteins)) * 100
        print(f"{i:2d}. {superfamily:40s}: {protein_count:3d} proteins ({percent:4.1f}%)")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Pie chart of top superfamilies
    top_10_superfamilies = dict(superfamily_counts.most_common(10))
    other_count = len(proteins) - sum(top_10_superfamilies.values())
    if other_count > 0:
        top_10_superfamilies['Others'] = other_count
    
    ax1.pie(top_10_superfamilies.values(), labels=top_10_superfamilies.keys(), 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Top 10 Structural Classes\n(by protein count)')
    
    # 2. Horizontal bar chart
    top_15 = superfamily_counts.most_common(15)
    classes, counts = zip(*top_15)
    
    y_pos = range(len(classes))
    ax2.barh(y_pos, counts)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes, fontsize=8)
    ax2.set_xlabel('Number of Proteins')
    ax2.set_title('Top 15 Structural Classes')
    ax2.invert_yaxis()
    
    # 3. Distribution of class sizes
    class_sizes = list(superfamily_counts.values())
    ax3.hist(class_sizes, bins=20, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Proteins per Structural Class')
    ax3.set_ylabel('Number of Classes')
    ax3.set_title('Distribution of Class Sizes')
    
    # 4. Resolution method breakdown
    method_counts = Counter(domain_method.values())
    ax4.pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%')
    ax4.set_title('Domain Resolution Methods')
    
    plt.tight_layout()
    plt.savefig('unified_interpro_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'unified_interpro_analysis.png'")
    
    # Export detailed results
    print("\nExporting detailed results...")
    
    # Main results CSV
    results_df = []
    for superfamily, count in superfamily_counts.most_common():
        protein_count = len(superfamily_proteins[superfamily])
        protein_list = ';'.join(sorted(superfamily_proteins[superfamily]))
        results_df.append({
            'Structural_Class': superfamily,
            'Protein_Count': protein_count,
            'Percentage': (protein_count / len(proteins)) * 100,
            'Protein_IDs': protein_list
        })
    
    df_results = pd.DataFrame(results_df)
    df_results.to_csv('structural_classes_results.csv', index=False)
    print("Main results saved to 'structural_classes_results.csv'")
    
    # Domain mapping CSV
    domain_mapping = []
    for domain_id in unique_domains:
        freq = domain_counts[domain_id]
        struct_class = domain_class.get(domain_id, "Unclassified")
        method = domain_method.get(domain_id, "unknown")
        domain_mapping.append({
            'Domain_ID': domain_id,
            'Frequency': freq,
            'Structural_Class': struct_class,
            'Resolution_Method': method
        })
    
    df_domains = pd.DataFrame(domain_mapping)
    df_domains.to_csv('domain_classifications.csv', index=False)
    print("Domain mapping saved to 'domain_classifications.csv'")
    
    # Summary statistics
    te_proteins = {pid for pid, cls in protein_primary.items() if cls.startswith("⚠")}
    print(f"\nSummary Statistics:")
    print(f"  Total proteins analyzed: {len(proteins)}")
    print(f"  Unique domains found: {len(unique_domains)}")
    print(f"  Structural classes identified: {len(superfamily_counts)}")
    print(f"  Transposable element proteins: {len(te_proteins)} ({len(te_proteins)/len(proteins)*100:.1f}%)")
    print(f"  Unclassified proteins: {superfamily_counts.get('Unclassified', 0)}")
    
    plt.show()

if __name__ == "__main__":
    main()