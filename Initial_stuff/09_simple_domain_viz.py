import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Load UniProt data
print("Loading UniProt DNA-binding protein data...")
with open('./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json') as f:
    data = json.load(f)

proteins = data['results']
print(f"Loaded {len(proteins)} proteins")

# Extract all InterPro domains
print("Extracting InterPro domains...")
all_domains = []
protein_domains = {}

for protein in proteins:
    protein_id = protein['primaryAccession']
    xrefs = protein.get('uniProtKBCrossReferences', [])
    interpro_domains = [x['id'] for x in xrefs if x.get('database') == 'InterPro']
    
    protein_domains[protein_id] = interpro_domains
    all_domains.extend(interpro_domains)

unique_domains = set(all_domains)
domain_counts = Counter(all_domains)

print(f"Found {len(unique_domains)} unique InterPro domains")
print(f"Total domain instances: {len(all_domains)}")

# Analysis and Statistics
print(f"\n{'='*60}")
print("INTERPRO DOMAIN ANALYSIS")
print(f"{'='*60}")

print(f"Top 20 most frequent domains:")
for i, (domain, count) in enumerate(domain_counts.most_common(20), 1):
    percent = (count / len(all_domains)) * 100
    protein_count = len([p for p, domains in protein_domains.items() if domain in domains])
    print(f"{i:2d}. {domain}: {count:3d} instances ({percent:4.1f}%), {protein_count:3d} proteins")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Pie chart of top 10 domains
top_10_domains = dict(domain_counts.most_common(10))
other_count = len(all_domains) - sum(top_10_domains.values())
if other_count > 0:
    top_10_domains['Others'] = other_count

ax1.pie(top_10_domains.values(), labels=top_10_domains.keys(), autopct='%1.1f%%', 
        startangle=90)
ax1.set_title('Top 10 InterPro Domains\n(by frequency)')

# 2. Horizontal bar chart of top 15 domains
top_15_domains = domain_counts.most_common(15)
domains, counts = zip(*top_15_domains)

y_pos = range(len(domains))
ax2.barh(y_pos, counts)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(domains, fontsize=9)
ax2.set_xlabel('Frequency')
ax2.set_title('Top 15 InterPro Domains')
ax2.invert_yaxis()

# 3. Distribution of domain frequencies
freq_values = list(domain_counts.values())
ax3.hist(freq_values, bins=30, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Domain Frequency')
ax3.set_ylabel('Number of Domains')
ax3.set_title('Distribution of Domain Frequencies')
ax3.set_yscale('log')  # Log scale due to many rare domains

# 4. Protein distribution by number of domains
domains_per_protein = [len(domains) for domains in protein_domains.values()]
ax4.hist(domains_per_protein, bins=20, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Number of Domains per Protein')
ax4.set_ylabel('Number of Proteins')
ax4.set_title('Distribution: Domains per Protein')

plt.tight_layout()
plt.savefig('interpro_domain_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed statistics
print(f"\n{'='*60}")
print("DETAILED STATISTICS")
print(f"{'='*60}")

rare_domains = [domain for domain, count in domain_counts.items() if count == 1]
common_domains = [domain for domain, count in domain_counts.items() if count >= 10]

print(f"Domain frequency statistics:")
print(f"  Median frequency: {sorted(freq_values)[len(freq_values)//2]}")
print(f"  Mean frequency: {sum(freq_values)/len(freq_values):.1f}")
print(f"  Max frequency: {max(freq_values)}")
print(f"  Domains appearing ≥10 times: {len(common_domains)}")
print(f"  Domains appearing only once: {len(rare_domains)}")

print(f"\nProtein statistics:")
print(f"  Mean domains per protein: {sum(domains_per_protein)/len(domains_per_protein):.1f}")
print(f"  Protein with most domains: {max(domains_per_protein)} domains")
print(f"  Proteins with no domains: {len([x for x in domains_per_protein if x == 0])}")

# Create a simple CSV output
print(f"\nCreating domain_results.csv...")
with open('domain_results.csv', 'w') as f:
    f.write('Domain_ID,Frequency,Protein_Count,Percentage\n')
    for domain, count in domain_counts.most_common():
        protein_count = len([p for p, domains in protein_domains.items() if domain in domains])
        percentage = (count / len(all_domains)) * 100
        f.write(f'{domain},{count},{protein_count},{percentage:.2f}\n')

print("Analysis complete!")
print("- Visualization saved as 'interpro_domain_analysis.png'")
print("- Data saved as 'domain_results.csv'")