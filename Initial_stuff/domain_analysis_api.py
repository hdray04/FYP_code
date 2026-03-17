import json
import requests
import time
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd

domain_class, domain_method = build_domain_class_map(unique_domains)
superfamily_counts, superfamily_proteins, protein_primary = \
    assign_proteins_to_classes(protein_domains, domain_class)

def get_superfamily_hierarchy(domain_data):
    """Extract superfamily information from InterPro entry data."""
    if not domain_data:
        return None
    
    entry_type = domain_data.get('type', {}).get('code', 'Unknown')
    name = domain_data.get('metadata', {}).get('name', 'Unknown')
    
    # Look for parent entries (superfamilies)
    parent_list = domain_data.get('metadata', {}).get('member_databases', {})
    hierarchy_info = {
        'domain_id': domain_data.get('metadata', {}).get('accession', ''),
        'name': name,
        'type': entry_type,
        'superfamily': None,
        'superfamily_name': None,
        'go_terms': []
    }
    
    # Extract GO terms for DNA-binding relevance
    go_terms = []
    metadata = domain_data.get('metadata', {})
    if metadata and 'go_terms' in metadata and metadata['go_terms'] is not None:
        for go_term in metadata['go_terms']:
            if go_term and 'identifier' in go_term:
                go_terms.append(go_term.get('identifier', ''))
    hierarchy_info['go_terms'] = go_terms
    
    # Look for parent/child relationships
    hierarchy = domain_data.get('metadata', {}).get('hierarchy', {})
    if hierarchy and isinstance(hierarchy, dict):
        if 'children' in hierarchy or 'parents' in hierarchy:
            # If this has parents, try to find the superfamily
            parents = hierarchy.get('parents', [])
            if parents and isinstance(parents, list) and len(parents) > 0:
                # Take the first parent as superfamily (could be refined)
                parent = parents[0]
                if isinstance(parent, dict):
                    hierarchy_info['superfamily'] = parent.get('accession', '')
                    hierarchy_info['superfamily_name'] = parent.get('name', '')
    
    # If this entry type is already a superfamily, use itself
    if entry_type in ['Homologous superfamily', 'Superfamily']:
        hierarchy_info['superfamily'] = hierarchy_info['domain_id']
        hierarchy_info['superfamily_name'] = name
    
    return hierarchy_info

def is_dna_binding_related(go_terms):
    """Check if GO terms suggest DNA-binding function."""
    dna_binding_go = {
        'GO:0003677',  # DNA binding
        'GO:0006355',  # transcription regulation
        'GO:0006281',  # DNA repair
        'GO:0006260',  # DNA replication
        'GO:0006325'   # chromatin organization
    }
    return any(go_id in dna_binding_go for go_id in go_terms)

# Load UniProt data
print("Loading UniProt DNA-binding protein data...")
with open('./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json') as f:
    data = json.load(f)

proteins = data['results']
print(f"Loaded {len(proteins)} proteins")

# Extract all unique InterPro domains
print("Extracting InterPro domains...")
all_domains = []
protein_domains = {}

for protein in proteins:
    protein_id = protein['primaryAccession']
    xrefs = protein.get('uniProtKBCrossReferences', [])
    interpro_domains = [x['id'] for x in xrefs if x.get('database') == 'InterPro']
    
    protein_domains[protein_id] = interpro_domains
    all_domains.extend(interpro_domains)

unique_domains = list(set(all_domains))
print(f"Found {len(unique_domains)} unique InterPro domains")

# Fetch hierarchy information from InterPro API
print(f"Fetching hierarchy data from InterPro API...")
print("This will take a few minutes due to API rate limiting...")

domain_hierarchy = {}
failed_domains = []

# Process domains in batches to avoid overwhelming the API
batch_size = 50
for i, domain_id in enumerate(unique_domains):
    if i % batch_size == 0:
        print(f"Processing domains {i+1}-{min(i+batch_size, len(unique_domains))} of {len(unique_domains)}")
    
    # Rate limiting - be respectful to the API
    time.sleep(0.1)  # 100ms delay between requests
    
    entry_data = fetch_interpro_entry(domain_id)
    if entry_data:
        hierarchy_info = get_superfamily_hierarchy(entry_data)
        
        if hierarchy_info:
            domain_hierarchy[domain_id] = hierarchy_info
        else:
            print(f"  Warning: Could not extract hierarchy for {domain_id}")
            failed_domains.append(domain_id)
    else:
        failed_domains.append(domain_id)

print(f"Successfully processed {len(domain_hierarchy)} domains")
if failed_domains:
    print(f"Failed to process {len(failed_domains)} domains: {failed_domains[:10]}...")

# Create superfamily mapping
superfamily_counts = Counter()
superfamily_proteins = defaultdict(set)
domain_to_superfamily = {}

for protein_id, domains in protein_domains.items():
    protein_superfamilies = []
    
    for domain_id in domains:
        if domain_id in domain_hierarchy:
            hierarchy_info = domain_hierarchy[domain_id]
            
            # Determine superfamily
            superfamily = hierarchy_info.get('superfamily')
            superfamily_name = hierarchy_info.get('superfamily_name', '')
            
            if superfamily:
                superfamily_label = f"{superfamily} ({superfamily_name})"
                domain_to_superfamily[domain_id] = superfamily_label
                protein_superfamilies.append(superfamily_label)
            else:
                # Fall back to domain name and type
                domain_name = hierarchy_info.get('name', '')
                domain_type = hierarchy_info.get('type', '')
                fallback_label = f"{domain_id} ({domain_type}: {domain_name})"
                domain_to_superfamily[domain_id] = fallback_label
                protein_superfamilies.append(fallback_label)
        else:
            # Domain not in our hierarchy data
            domain_to_superfamily[domain_id] = f"{domain_id} (Unknown)"
            protein_superfamilies.append(f"{domain_id} (Unknown)")
    
    # For proteins with multiple superfamilies, prioritize DNA-binding ones
    dna_binding_superfamilies = []
    for domain_id in domains:
        if domain_id in domain_hierarchy:
            hierarchy_info = domain_hierarchy[domain_id]
            go_terms = hierarchy_info.get('go_terms', [])
            if is_dna_binding_related(go_terms):
                superfamily = hierarchy_info.get('superfamily')
                superfamily_name = hierarchy_info.get('superfamily_name', '')
                if superfamily:
                    dna_binding_superfamilies.append(f"{superfamily} ({superfamily_name})")
    
    # Use DNA-binding superfamily if available, otherwise use all superfamilies
    final_superfamilies = dna_binding_superfamilies if dna_binding_superfamilies else protein_superfamilies
    
    for sfam in final_superfamilies:
        superfamily_counts[sfam] += 1
        superfamily_proteins[sfam].add(protein_id)

# Generate results and visualizations
print(f"\n{'='*80}")
print("INTERPRO SUPERFAMILY ANALYSIS (Using InterPro API Hierarchy)")
print(f"{'='*80}")

print(f"Unique superfamilies found: {len(superfamily_counts)}")
print(f"Total superfamily instances: {sum(superfamily_counts.values())}")

print(f"\nTop 20 Superfamilies:")
for i, (superfamily, count) in enumerate(superfamily_counts.most_common(20), 1):
    protein_count = len(superfamily_proteins[superfamily])
    percent = (count / sum(superfamily_counts.values())) * 100
    print(f"{i:2d}. {superfamily:60s}: {count:3d} instances, {protein_count:3d} proteins ({percent:4.1f}%)")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Pie chart of top superfamilies
top_superfamilies = dict(superfamily_counts.most_common(10))
other_count = sum(superfamily_counts.values()) - sum(top_superfamilies.values())
if other_count > 0:
    top_superfamilies['Others'] = other_count

# Simplify labels for pie chart
simplified_labels = []
for label in top_superfamilies.keys():
    if label == 'Others':
        simplified_labels.append('Others')
    elif '(' in label:
        # Extract just the descriptive name part
        name_part = label.split('(')[1].split(')')[0]
        simplified_labels.append(name_part[:25] + '...' if len(name_part) > 25 else name_part)
    else:
        simplified_labels.append(label[:25] + '...' if len(label) > 25 else label)

ax1.pie(top_superfamilies.values(), labels=simplified_labels, autopct='%1.1f%%', startangle=90)
ax1.set_title('Top 10 InterPro Superfamilies\n(Domain instances)')

# 2. Horizontal bar chart of top superfamilies by protein count
top_by_proteins = [(sfam, len(superfamily_proteins[sfam])) for sfam, _ in superfamily_counts.most_common(15)]
superfamilies, protein_counts = zip(*top_by_proteins)

# Simplify labels for bar chart
bar_labels = []
for label in superfamilies:
    if '(' in label:
        name_part = label.split('(')[1].split(')')[0]
        bar_labels.append(name_part[:30] + '...' if len(name_part) > 30 else name_part)
    else:
        bar_labels.append(label[:30] + '...' if len(label) > 30 else label)

y_pos = range(len(bar_labels))
ax2.barh(y_pos, protein_counts)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(bar_labels, fontsize=8)
ax2.set_xlabel('Number of Proteins')
ax2.set_title('Top 15 Superfamilies by Protein Count')
ax2.invert_yaxis()

# 3. Distribution of superfamily sizes
superfamily_sizes = [len(superfamily_proteins[sfam]) for sfam in superfamily_counts.keys()]
ax3.hist(superfamily_sizes, bins=20, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Number of Proteins per Superfamily')
ax3.set_ylabel('Number of Superfamilies')
ax3.set_title('Distribution of Superfamily Sizes')

# 4. Summary statistics table
summary_data = {
    'Metric': [
        'Total Proteins',
        'Proteins with InterPro domains', 
        'Unique InterPro domains',
        'Unique Superfamilies',
        'Average domains per protein',
        'Average superfamilies per protein',
        'Largest superfamily size',
        'Number of singleton superfamilies'
    ],
    'Value': [
        len(proteins),
        len([p for p in protein_domains.values() if p]),
        len(unique_domains),
        len(superfamily_counts),
        f"{sum(len(domains) for domains in protein_domains.values()) / len(proteins):.1f}",
        f"{sum(superfamily_counts.values()) / len(proteins):.1f}",
        max(len(superfamily_proteins[sfam]) for sfam in superfamily_counts.keys()) if len(superfamily_counts) > 0 else 0,
        len([sfam for sfam in superfamily_counts.keys() if len(superfamily_proteins[sfam]) == 1]) if len(superfamily_counts) > 0 else 0
    ]
}

table_data = list(zip(summary_data['Metric'], summary_data['Value']))
ax4.axis('tight')
ax4.axis('off')
table = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                  cellLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax4.set_title('Summary Statistics')

plt.tight_layout()
plt.savefig('interpro_superfamily_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save detailed results to CSV
results_df = []
for superfamily, count in superfamily_counts.most_common():
    protein_count = len(superfamily_proteins[superfamily])
    protein_list = ';'.join(sorted(superfamily_proteins[superfamily]))
    results_df.append({
        'Superfamily': superfamily,
        'Domain_Instances': count,
        'Protein_Count': protein_count,
        'Percentage': (count / sum(superfamily_counts.values())) * 100,
        'Protein_IDs': protein_list
    })

df = pd.DataFrame(results_df)
df.to_csv('interpro_superfamily_results.csv', index=False)
print(f"\nDetailed results saved to 'interpro_superfamily_results.csv'")
print(f"Visualization saved to 'interpro_superfamily_analysis.png'")