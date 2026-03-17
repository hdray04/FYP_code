import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# InterPro domain to broad category mapping
DOMAIN_CATEGORIES = {
    # Homeodomain family
    'IPR009057': 'Homeodomain',
    'IPR001356': 'Homeodomain', 
    'IPR017970': 'Homeodomain-related',
    'IPR020479': 'Homeodomain-related',
    
    # Nuclear receptor family
    'IPR001628': 'Nuclear receptor',
    'IPR000536': 'Nuclear receptor',
    'IPR001723': 'Nuclear receptor',
    'IPR013088': 'Nuclear receptor',
    
    # Basic helix-loop-helix
    'IPR001092': 'bHLH',
    'IPR011598': 'bHLH',
    
    # Zinc finger domains
    'IPR007087': 'Zinc finger',
    'IPR013087': 'Zinc finger', 
    'IPR015880': 'Zinc finger',
    'IPR000315': 'Zinc finger',
    'IPR036236': 'Zinc finger',
    'IPR001909': 'Zinc finger',
    'IPR015422': 'Zinc finger',
    
    # High Mobility Group (HMG) domains
    'IPR000910': 'HMG box',
    'IPR009071': 'HMG box',
    'IPR036910': 'HMG box',
    
    # DNA repair domains
    'IPR006165': 'DNA repair',
    'IPR004365': 'DNA repair',
    'IPR002421': 'DNA repair',
    'IPR006134': 'DNA repair',
    'IPR003583': 'DNA repair',
    
    # Replication machinery
    'IPR001766': 'DNA replication',
    'IPR030456': 'DNA replication',
    'IPR008918': 'DNA replication',
    'IPR027417': 'DNA replication',
    
    # Transcription factors
    'IPR036388': 'Transcription factor',
    'IPR036390': 'Transcription factor',
    'IPR011991': 'Transcription factor',
    'IPR001965': 'Transcription factor',
    'IPR000047': 'Transcription factor',
    'IPR003121': 'Transcription factor',
    'IPR035500': 'Transcription factor',
    
    # Chromatin remodeling/modification
    'IPR000953': 'Chromatin remodeling',
    'IPR001487': 'Chromatin remodeling',
    'IPR002119': 'Chromatin remodeling',
    'IPR016181': 'Chromatin remodeling',
    
    # DNA-binding domains (general)
    'IPR036279': 'DNA-binding domain',
    'IPR036421': 'DNA-binding domain',
    'IPR019548': 'DNA-binding domain',
    'IPR005818': 'DNA-binding domain',
}

def categorize_domain(domain_id):
    """Map InterPro domain ID to broad functional category."""
    return DOMAIN_CATEGORIES.get(domain_id, 'Other/Unknown')

# Load UniProt data
print("Loading DNA-binding protein data...")
with open('./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json') as f:
    data = json.load(f)

proteins = data['results']
total_proteins = len(proteins)

print(f"Analyzing {total_proteins} DNA-binding proteins")
print("="*60)

# Extract domains
all_domains = []
protein_domains = {}
proteins_with_domains = 0

for protein in proteins:
    protein_id = protein['primaryAccession']
    xrefs = protein.get('uniProtKBCrossReferences', [])
    interpro_domains = [x['id'] for x in xrefs if x.get('database') == 'InterPro']
    
    protein_domains[protein_id] = interpro_domains
    all_domains.extend(interpro_domains)
    
    if interpro_domains:
        proteins_with_domains += 1

# Basic stats
print(f"Proteins with domains: {proteins_with_domains}/{total_proteins} ({proteins_with_domains/total_proteins*100:.1f}%)")
print(f"Total domain instances: {len(all_domains)}")
print(f"Unique domains: {len(set(all_domains))}")

# Categorize domains
categories = [categorize_domain(domain) for domain in all_domains]
category_counts = Counter(categories)
domain_counts = Counter(all_domains)

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('DNA-Binding Protein Domain Analysis', fontsize=16, fontweight='bold')

# 1. Pie chart of broad categories
category_data = category_counts.most_common()
category_names = [cat[0] for cat in category_data]
category_values = [cat[1] for cat in category_data]

# Color scheme for pie chart
colors = sns.color_palette("husl", len(category_names))

wedges, texts, autotexts = ax1.pie(category_values, labels=category_names, autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax1.set_title('Distribution of Broad Domain Categories', fontweight='bold')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 2. Top specific domains (horizontal bar chart)
top_domains = domain_counts.most_common(15)
domain_names = [f"{d[0]}\n({categorize_domain(d[0])})" for d in top_domains]
domain_values = [d[1] for d in top_domains]

bars = ax2.barh(range(len(domain_names)), domain_values, color=sns.color_palette("viridis", len(domain_names)))
ax2.set_yticks(range(len(domain_names)))
ax2.set_yticklabels(domain_names, fontsize=8)
ax2.set_xlabel('Number of Proteins')
ax2.set_title('Top 15 Most Frequent InterPro Domains', fontweight='bold')
ax2.invert_yaxis()

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
             f'{int(width)}', ha='left', va='center', fontweight='bold')

# 3. Stacked bar chart of categories with top domains
category_domain_data = defaultdict(list)
for domain_id, count in domain_counts.most_common():
    category = categorize_domain(domain_id)
    category_domain_data[category].append((domain_id, count))

# Get top categories for stacked chart
top_categories = [cat for cat, _ in category_counts.most_common(6)]
bottom_values = [0] * len(top_categories)

# Create stacked bars for top 3 domains in each category
for i, category in enumerate(top_categories):
    top_domains_in_cat = category_domain_data[category][:3]  # Top 3 domains
    
    for j, (domain_id, count) in enumerate(top_domains_in_cat):
        color = sns.color_palette("Set3", 3)[j]
        ax3.bar(i, count, bottom=bottom_values[i], label=domain_id if i == 0 else "", 
                color=color, alpha=0.8)
        bottom_values[i] += count

ax3.set_xticks(range(len(top_categories)))
ax3.set_xticklabels(top_categories, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Number of Domain Instances')
ax3.set_title('Top Domains within Major Categories', fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 4. Domain frequency distribution
freq_distribution = Counter(domain_counts.values())
frequencies = sorted(freq_distribution.keys())
freq_counts = [freq_distribution[f] for f in frequencies]

# Log scale for better visualization
ax4.bar(frequencies, freq_counts, alpha=0.7, color='skyblue', edgecolor='navy')
ax4.set_xlabel('Domain Frequency (proteins)')
ax4.set_ylabel('Number of Domains')
ax4.set_title('Distribution of Domain Frequencies', fontweight='bold')
ax4.set_yscale('log')

# Add some statistics text
stats_text = f"Total domains: {len(set(all_domains))}\nRare domains (1 protein): {freq_distribution[1]}\nMost frequent: {max(domain_counts.values())} proteins"
ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

plt.tight_layout()
plt.savefig('domain_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

print(f"\n{'BROAD DOMAIN CATEGORIES':<30s} {'Count':<8s} {'%':<6s} {'Unique':<7s}")
print("-" * 55)
for category, count in category_counts.most_common():
    percentage = count / len(all_domains) * 100
    unique_count = len([d for d in set(all_domains) if categorize_domain(d) == category])
    print(f"{category:<30s} {count:<8d} {percentage:<6.1f} {unique_count:<7d}")

print(f"\n{'TOP SPECIFIC DOMAINS':<15s} {'Category':<20s} {'Count':<8s} {'%':<6s}")
print("-" * 55)
for domain_id, count in domain_counts.most_common(10):
    category = categorize_domain(domain_id)
    percentage = count / total_proteins * 100
    print(f"{domain_id:<15s} {category:<20s} {count:<8d} {percentage:<6.1f}")

print(f"\nDomain Analysis Complete!")
print(f"Visualization saved as 'domain_analysis_plots.png'")