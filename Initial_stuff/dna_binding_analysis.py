import json
import csv
import statistics
from collections import Counter, defaultdict

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

# Load UniProt data (DNA-binding proteins only)
print("Loading UniProt DNA-binding protein data...")
with open('./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json') as f:
    data = json.load(f)

proteins = data['results']
total_proteins = len(proteins)

print(f"Loaded {total_proteins} DNA-binding proteins")
print("="*80)

# ============================================================================
# PART 1: InterPro Domain Analysis 
# ============================================================================

print("INTERPRO DOMAIN ANALYSIS")
print("="*80)

# Extract InterPro domains for all proteins
protein_domains = {}  # protein_id -> list of domain IDs
all_domains = []      # flat list of all domain IDs
proteins_with_interpro = 0
proteins_without_interpro = []

for protein in proteins:
    protein_id = protein['primaryAccession']
    xrefs = protein.get('uniProtKBCrossReferences', [])
    interpro_domains = [x['id'] for x in xrefs if x.get('database') == 'InterPro']
    
    protein_domains[protein_id] = interpro_domains
    all_domains.extend(interpro_domains)
    
    if interpro_domains:
        proteins_with_interpro += 1
    else:
        proteins_without_interpro.append(protein_id)

# Basic statistics
print(f"Proteins with InterPro domains:     {proteins_with_interpro} ({proteins_with_interpro/total_proteins*100:.1f}%)")
print(f"Proteins without InterPro domains:  {len(proteins_without_interpro)} ({len(proteins_without_interpro)/total_proteins*100:.1f}%)")
print(f"Total InterPro domain instances:    {len(all_domains)}")

if all_domains:
    unique_domains = set(all_domains)
    print(f"Unique InterPro domains:            {len(unique_domains)}")
    
    # Domain frequency analysis
    domain_counts = Counter(all_domains)
    avg_domains_per_protein = len(all_domains) / proteins_with_interpro if proteins_with_interpro > 0 else 0
    print(f"Average domains per protein (with domains): {avg_domains_per_protein:.1f}")
    
    # Categorize domains into broad functional groups
    category_counts = defaultdict(int)
    category_domains = defaultdict(set)
    all_categories = []
    
    for domain_id in all_domains:
        category = categorize_domain(domain_id)
        category_counts[category] += 1
        category_domains[category].add(domain_id)
        all_categories.append(category)
    
    print(f"\n" + "="*50)
    print("BROAD DOMAIN CATEGORY ANALYSIS")
    print("="*50)
    
    print(f"Broad domain categories found:      {len(category_counts)}")
    print(f"Average category instances per protein: {len(all_categories)/proteins_with_interpro:.1f}")
    
    # Most common broad categories
    print(f"\nMost frequent domain categories:")
    category_counter = Counter(all_categories)
    for category, count in category_counter.most_common():
        unique_domains_in_cat = len(category_domains[category])
        percent_proteins = (count / len(all_domains)) * 100
        print(f"  {category:25s}: {count:4d} instances ({percent_proteins:4.1f}%) - {unique_domains_in_cat:3d} unique domains")
    
    # Show specific domains within each major category
    print(f"\nDetailed breakdown of major categories:")
    for category, count in category_counter.most_common(8):  # Top 8 categories
        print(f"\n{category} ({count} instances, {len(category_domains[category])} unique domains):")
        # Count specific domains within this category
        category_domain_counts = {domain: domain_counts[domain] 
                                for domain in category_domains[category]}
        sorted_domains = sorted(category_domain_counts.items(), 
                              key=lambda x: x[1], reverse=True)
        
        for domain_id, domain_count in sorted_domains[:8]:  # Top 8 domains per category
            percent = (domain_count / total_proteins) * 100
            print(f"    {domain_id}: {domain_count:3d} proteins ({percent:4.1f}%)")
        
        if len(sorted_domains) > 8:
            print(f"    ... and {len(sorted_domains) - 8} more domains")
    
    print(f"\n" + "="*50)
    print("SPECIFIC DOMAIN ANALYSIS")
    print("="*50)
    
    # Most common specific domains
    print(f"\nMost frequent specific InterPro domains:")
    for domain_id, count in domain_counts.most_common(15):
        category = categorize_domain(domain_id)
        percent_proteins = (count / total_proteins) * 100
        print(f"  {domain_id} ({category:20s}): {count:3d} proteins ({percent_proteins:4.1f}%)")
    
    # Least common domains (occurring in only 1 protein)
    rare_domains = [domain for domain, count in domain_counts.items() if count == 1]
    print(f"\nDomains found in only 1 protein:    {len(rare_domains)}")
    if len(rare_domains) <= 10:
        print(f"  Rare domains: {', '.join(rare_domains)}")
    else:
        print(f"  Example rare domains: {', '.join(rare_domains[:10])}...")
    
    # Domain distribution statistics
    domain_frequencies = list(domain_counts.values())
    print(f"\nDomain frequency distribution:")
    print(f"  Median frequency:                  {statistics.median(domain_frequencies)}")
    print(f"  Mean frequency:                    {statistics.mean(domain_frequencies):.1f}")
    print(f"  Max frequency:                     {max(domain_frequencies)}")
    print(f"  Min frequency:                     {min(domain_frequencies)}")

else:
    print("No InterPro domains found in dataset")

# Show proteins without domains (if any)
if proteins_without_interpro:
    print(f"\nProteins without InterPro domains:")
    for protein_id in proteins_without_interpro[:10]:  # Show first 10
        print(f"  {protein_id}")
    if len(proteins_without_interpro) > 10:
        print(f"  ... and {len(proteins_without_interpro) - 10} more")

print("\n" + "="*80)

# ============================================================================
# PART 2: ClinVar Variant Mapping Analysis
# ============================================================================

print("CLINVAR VARIANT MAPPING ANALYSIS")
print("="*80)

# Create set of all protein accessions (primary + secondary)
all_protein_accessions = set()
for protein in proteins:
    all_protein_accessions.add(protein['primaryAccession'])
    all_protein_accessions.update(protein.get('secondaryAccessions', []))

print(f"Total protein accessions to match:  {len(all_protein_accessions)}")

# Load and process ClinVar data
clinvar_variants = {}  # uniprot_id -> variant_count
variant_data = []      # list of variant records for detailed analysis

clinvar_file = './Data/clinvar_variants_28_08_2025.csv'

try:
    print("Loading ClinVar variants...")
    with open(clinvar_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_clinvar_rows = 0
        mapped_variants = 0
        
        for row in reader:
            total_clinvar_rows += 1
            uniprot_id = row.get('uniprot', '').strip()
            
            if uniprot_id and uniprot_id in all_protein_accessions:
                # Count variants per protein
                if uniprot_id not in clinvar_variants:
                    clinvar_variants[uniprot_id] = 0
                clinvar_variants[uniprot_id] += 1
                mapped_variants += 1
                
                # Store variant details for analysis
                variant_data.append({
                    'uniprot': uniprot_id,
                    'type': row.get('Type', 'Unknown').strip(),
                    'significance': row.get('ClinicalSignificance', 'Unknown').strip(),
                    'review': row.get('ReviewStatus', 'Unknown').strip()
                })
    
    print(f"Processed {total_clinvar_rows} total ClinVar variants")
    print(f"Mapped {mapped_variants} variants to DNA-binding proteins")
    print(f"Unique DNA-binding proteins with variants: {len(clinvar_variants)}")

except FileNotFoundError:
    print(f"Error: ClinVar file not found: {clinvar_file}")
    print("Skipping variant analysis.")
    clinvar_variants = {}
    variant_data = []
except Exception as e:
    print(f"Error processing ClinVar data: {e}")
    clinvar_variants = {}
    variant_data = []

# Calculate protein-level variant mapping
proteins_with_variants = 0
proteins_without_variants = 0
total_variants_mapped = 0
protein_variant_counts = []

for protein in proteins:
    # Check all accessions for this protein
    protein_accessions = [protein['primaryAccession']]
    protein_accessions.extend(protein.get('secondaryAccessions', []))
    
    # Count variants for this protein
    protein_variant_count = 0
    for acc in protein_accessions:
        protein_variant_count += clinvar_variants.get(acc, 0)
    
    protein_variant_counts.append(protein_variant_count)
    total_variants_mapped += protein_variant_count
    
    if protein_variant_count > 0:
        proteins_with_variants += 1
    else:
        proteins_without_variants += 1

# Variant mapping statistics
print(f"\nProtein-level variant mapping:")
print(f"Proteins with ClinVar variants:     {proteins_with_variants} ({proteins_with_variants/total_proteins*100:.1f}%)")
print(f"Proteins with NO variants:          {proteins_without_variants} ({proteins_without_variants/total_proteins*100:.1f}%)")
print(f"Total variants mapped:              {total_variants_mapped}")

if proteins_with_variants > 0:
    avg_variants = total_variants_mapped / proteins_with_variants
    print(f"Average variants per protein (with variants): {avg_variants:.1f}")

# Variant count distribution
if protein_variant_counts:
    print(f"\nVariant count distribution:")
    print(f"  Median variants per protein:      {statistics.median(protein_variant_counts)}")
    print(f"  Mean variants per protein:        {statistics.mean(protein_variant_counts):.1f}")
    print(f"  Maximum variants in one protein:  {max(protein_variant_counts)}")
    
    # Variant count ranges
    ranges = [
        (0, 0, "0 variants"),
        (1, 1, "1 variant"), 
        (2, 5, "2-5 variants"),
        (6, 10, "6-10 variants"),
        (11, 50, "11-50 variants"),
        (51, float('inf'), "50+ variants")
    ]
    
    print(f"\nVariant count distribution:")
    for min_val, max_val, label in ranges:
        count = len([x for x in protein_variant_counts if min_val <= x <= max_val])
        percent = (count / total_proteins) * 100
        print(f"  {label:15s}: {count:4d} proteins ({percent:5.1f}%)")

print("\n" + "="*80)

# ============================================================================
# PART 3: Variant Type Analysis
# ============================================================================

if variant_data:
    print("VARIANT TYPE AND CLINICAL SIGNIFICANCE ANALYSIS")
    print("="*80)
    
    # Analyze variant types
    variant_types = Counter(v['type'] for v in variant_data)
    print(f"Variant types (total: {len(variant_data)} variants):")
    for var_type, count in variant_types.most_common():
        percent = (count / len(variant_data)) * 100
        print(f"  {var_type:40s}: {count:6d} ({percent:5.1f}%)")
    
    # Analyze clinical significance
    print(f"\nClinical significance:")
    significance_types = Counter(v['significance'] for v in variant_data)
    for sig, count in significance_types.most_common():
        percent = (count / len(variant_data)) * 100
        print(f"  {sig:40s}: {count:6d} ({percent:5.1f}%)")
    
    # Categorize pathogenicity
    pathogenic_variants = 0
    benign_variants = 0
    uncertain_variants = 0
    
    for variant in variant_data:
        sig = variant['significance'].lower()
        if 'pathogenic' in sig and 'likely pathogenic' not in sig:
            pathogenic_variants += 1
        elif 'likely pathogenic' in sig:
            pathogenic_variants += 1
        elif 'benign' in sig or 'likely benign' in sig:
            benign_variants += 1
        elif 'uncertain' in sig or 'conflicting' in sig:
            uncertain_variants += 1
    
    print(f"\nPathogenicity classification:")
    print(f"  Pathogenic/Likely pathogenic:     {pathogenic_variants:6d} ({pathogenic_variants/len(variant_data)*100:5.1f}%)")
    print(f"  Benign/Likely benign:             {benign_variants:6d} ({benign_variants/len(variant_data)*100:5.1f}%)")
    print(f"  Uncertain/Conflicting:            {uncertain_variants:6d} ({uncertain_variants/len(variant_data)*100:5.1f}%)")
    
    # Review status
    print(f"\nReview status:")
    review_status = Counter(v['review'] for v in variant_data)
    for status, count in review_status.most_common():
        percent = (count / len(variant_data)) * 100
        print(f"  {status:40s}: {count:6d} ({percent:5.1f}%)")

print("\n" + "="*80)

# ============================================================================
# PART 4: Combined Analysis - Domains vs Variants
# ============================================================================

print("DOMAIN-VARIANT RELATIONSHIP ANALYSIS")
print("="*80)

# Analyze domain enrichment in proteins with/without variants
domains_with_variants = []
domains_without_variants = []
categories_with_variants = []
categories_without_variants = []

for protein in proteins:
    protein_id = protein['primaryAccession']
    protein_accessions = [protein_id] + protein.get('secondaryAccessions', [])
    
    # Check if protein has variants
    protein_variant_count = sum(clinvar_variants.get(acc, 0) for acc in protein_accessions)
    has_variants = protein_variant_count > 0
    
    # Get domains for this protein
    domains = protein_domains.get(protein_id, [])
    
    if has_variants:
        domains_with_variants.extend(domains)
        categories_with_variants.extend([categorize_domain(d) for d in domains])
    else:
        domains_without_variants.extend(domains)
        categories_without_variants.extend([categorize_domain(d) for d in domains])

if domains_with_variants or domains_without_variants:
    domain_counts_with_variants = Counter(domains_with_variants)
    domain_counts_without_variants = Counter(domains_without_variants)
    category_counts_with_variants = Counter(categories_with_variants)
    category_counts_without_variants = Counter(categories_without_variants)
    
    print(f"Domains in proteins WITH variants:     {len(domains_with_variants)} total, {len(domain_counts_with_variants)} unique")
    print(f"Domains in proteins WITHOUT variants:  {len(domains_without_variants)} total, {len(domain_counts_without_variants)} unique")
    
    # Show category comparison
    print(f"\n" + "-"*60)
    print("BROAD CATEGORY COMPARISON: Variants vs No Variants")
    print("-"*60)
    
    all_categories_set = set(category_counts_with_variants.keys()) | set(category_counts_without_variants.keys())
    
    print(f"{'Category':<25s} {'With Variants':<15s} {'No Variants':<12s} {'Ratio':<8s}")
    print("-" * 65)
    
    for category in sorted(all_categories_set):
        with_count = category_counts_with_variants.get(category, 0)
        without_count = category_counts_without_variants.get(category, 0)
        
        if without_count > 0:
            ratio = with_count / without_count
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = "∞" if with_count > 0 else "0"
        
        print(f"{category:<25s} {with_count:<15d} {without_count:<12d} {ratio_str:<8s}")
    
    # Show most common domains in each category
    print(f"\n" + "-"*60)
    print("SPECIFIC DOMAIN COMPARISON")
    print("-"*60)
    
    print(f"\nTop domains in proteins WITH variants:")
    for domain, count in domain_counts_with_variants.most_common(10):
        category = categorize_domain(domain)
        print(f"  {domain} ({category}): {count}")
    
    print(f"\nTop domains in proteins WITHOUT variants:")
    for domain, count in domain_counts_without_variants.most_common(10):
        category = categorize_domain(domain)
        print(f"  {domain} ({category}): {count}")
    
    # Domain enrichment analysis (simple comparison)
    print(f"\nDomain enrichment analysis:")
    all_variant_domains = set(domain_counts_with_variants.keys())
    all_no_variant_domains = set(domain_counts_without_variants.keys())
    
    variant_specific = all_variant_domains - all_no_variant_domains
    no_variant_specific = all_no_variant_domains - all_variant_domains
    shared_domains = all_variant_domains & all_no_variant_domains
    
    print(f"  Domains only in variant proteins:     {len(variant_specific)}")
    print(f"  Domains only in non-variant proteins: {len(no_variant_specific)}")
    print(f"  Domains in both categories:           {len(shared_domains)}")
    
    if variant_specific:
        variant_specific_categories = [categorize_domain(d) for d in variant_specific]
        variant_category_counts = Counter(variant_specific_categories)
        print(f"  Variant-specific domain categories: {dict(variant_category_counts)}")
    
    if no_variant_specific:
        no_variant_specific_categories = [categorize_domain(d) for d in no_variant_specific]
        no_variant_category_counts = Counter(no_variant_specific_categories)
        print(f"  No-variant-specific domain categories: {dict(no_variant_category_counts)}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Summary
print(f"SUMMARY:")
print(f"Total DNA-binding proteins analyzed:    {total_proteins}")
print(f"Proteins with InterPro domains:        {proteins_with_interpro} ({proteins_with_interpro/total_proteins*100:.1f}%)")
print(f"Proteins with ClinVar variants:        {proteins_with_variants} ({proteins_with_variants/total_proteins*100:.1f}%)")
print(f"Unique InterPro domains found:         {len(set(all_domains)) if all_domains else 0}")
if all_domains:
    unique_categories = len(set([categorize_domain(d) for d in set(all_domains)]))
    print(f"Broad domain categories found:         {unique_categories}")
print(f"Total ClinVar variants mapped:         {total_variants_mapped}")
if variant_data and total_variants_mapped > 0:
    print(f"Pathogenic/Likely pathogenic variants: {pathogenic_variants} ({pathogenic_variants/len(variant_data)*100:.1f}%)")