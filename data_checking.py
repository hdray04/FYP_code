import json
import csv
import statistics

with open('./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json') as f:
    data = json.load(f)
results = data['results']

#### InterPro and Pfam Annotation Analysis ####
total = len(results)
has_interpro = 0
has_pfam = 0
has_either = 0
has_both = 0
missing_either = []

for entry in results:
    xrefs = entry.get('uniProtKBCrossReferences', [])
    dbs = set(x['database'] for x in xrefs)
    ip = 'InterPro' in dbs
    pf = 'Pfam' in dbs
    if ip:
        has_interpro += 1
    if pf:
        has_pfam += 1
    if ip or pf:
        has_either += 1
    if ip and pf:
        has_both += 1
    if not ip and not pf:
        missing_either.append(entry['primaryAccession'])

print(f'Total proteins:              {total}')
print(f'With InterPro annotations:   {has_interpro} ({100 * has_interpro / total:.1f}%)')
print(f'With Pfam annotations:       {has_pfam} ({100 * has_pfam / total:.1f}%)')
print(f'With InterPro or Pfam:       {has_either} ({100 * has_either / total:.1f}%)')
print(f'With both InterPro and Pfam: {has_both} ({100 * has_both / total:.1f}%)')
print(f'Without either:              {total - has_either} ({100 * (total - has_either) / total:.1f}%)')

if missing_either:
    print(f'\nProteins missing both InterPro and Pfam annotations:')
    for acc in missing_either:
        print(f'  {acc}')


    
##### ClinVar Analysis ####


print(f'\n{"="*60}')
print("CLINVAR VARIANT MAPPING ANALYSIS")
print(f'{"="*60}')

# Load ClinVar data
clinvar_variants = {}
variant_types_mapped = {}
clinical_significance_mapped = {}
review_status_mapped = {}
mapped_variants_data = []

# Get all protein accessions from our dataset first
all_protein_accessions = set()
for entry in results:
    all_protein_accessions.add(entry['primaryAccession'])
    all_protein_accessions.update(entry.get('secondaryAccessions', []))

clinvar_file = './Data/clinvar_variants_28_08_2025.csv'

try:
    with open(clinvar_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('OriginSimple', '').strip() != 'germline':
                continue
            if row.get('canonical_residue', '').strip() == 'Met' and row.get('residue_seq', '').strip() == '1':
                continue  # remove start-lost
            uniprot_id = row['uniprot']
            if uniprot_id and uniprot_id.strip() and uniprot_id in all_protein_accessions:  # Only count variants for our proteins
                if uniprot_id not in clinvar_variants:
                    clinvar_variants[uniprot_id] = 0
                clinvar_variants[uniprot_id] += 1
                
                # Collect variant characteristics only for mapped variants
                var_type = row.get('Type', 'Unknown').strip()
                clin_sig = row.get('ClinicalSignificance', 'Unknown').strip()
                review_stat = row.get('ReviewStatus', 'Unknown').strip()
                
                variant_types_mapped[var_type] = variant_types_mapped.get(var_type, 0) + 1
                clinical_significance_mapped[clin_sig] = clinical_significance_mapped.get(clin_sig, 0) + 1
                review_status_mapped[review_stat] = review_status_mapped.get(review_stat, 0) + 1
                
                mapped_variants_data.append({
                    'uniprot': uniprot_id,
                    'type': var_type,
                    'significance': clin_sig,
                    'review': review_stat
                })
    
    print(f'Loaded {len(clinvar_variants)} unique proteins with ClinVar variants (from our 611 protein dataset)')
    print(f'Total ClinVar variants mapped to our proteins: {len(mapped_variants_data)}')
    
    if mapped_variants_data:
        # Show variant type distribution for mapped variants only
        print(f'\n--- ClinVar Variant Types (Mapped to Our 611 Proteins) ---')
        sorted_types = sorted(variant_types_mapped.items(), key=lambda x: x[1], reverse=True)
        for var_type, count in sorted_types:
            percent = (count / len(mapped_variants_data)) * 100
            print(f'{var_type:40s}: {count:6d} ({percent:5.1f}%)')
        
        # Show clinical significance distribution for mapped variants only
        print(f'\n--- Clinical Significance (Mapped Variants Only) ---')
        sorted_significance = sorted(clinical_significance_mapped.items(), key=lambda x: x[1], reverse=True)
        for sig, count in sorted_significance:
            percent = (count / len(mapped_variants_data)) * 100
            print(f'{sig:40s}: {count:6d} ({percent:5.1f}%)')
        
        # Show review status distribution for mapped variants only
        print(f'\n--- Review Status (Mapped Variants Only) ---')
        sorted_review = sorted(review_status_mapped.items(), key=lambda x: x[1], reverse=True)
        for status, count in sorted_review:
            percent = (count / len(mapped_variants_data)) * 100
            print(f'{status:40s}: {count:6d} ({percent:5.1f}%)')
    else:
        print('No variants found that map to our protein dataset')

    # Analyze pathogenic variants and population frequencies
    print(f'\n{"="*60}')
    print("PATHOGENIC VARIANTS VS POPULATION FREQUENCY ANALYSIS")
    print(f'{"="*60}')
    
    if mapped_variants_data:
        # First, check what frequency columns are available
        print("Checking available columns in ClinVar data...")
        with open(clinvar_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            available_columns = reader.fieldnames
            
        # Look for frequency-related columns
        freq_columns = [col for col in available_columns if any(freq_term in col.lower() 
                       for freq_term in ['freq', 'gnomad', '1000g', 'af', 'allele_freq', 'maf', 'population'])]
        
        dbsnp_column = 'RS# (dbSNP)' if 'RS# (dbSNP)' in available_columns else None
        
        print(f"Frequency-related columns found: {freq_columns}")
        print(f"dbSNP column available: {'Yes' if dbsnp_column else 'No'}")
        
        # Analyze pathogenic variants
        pathogenic_variants = []
        benign_variants = []
        uncertain_variants = []
        
        for variant in mapped_variants_data:
            significance = variant['significance'].lower()
            if 'pathogenic' in significance and 'likely pathogenic' not in significance:
                pathogenic_variants.append(variant)
            elif 'likely pathogenic' in significance:
                pathogenic_variants.append(variant)
            elif 'benign' in significance or 'likely benign' in significance:
                benign_variants.append(variant)
            elif 'uncertain' in significance or 'conflicting' in significance:
                uncertain_variants.append(variant)
        
        print(f"\nVariant classification for our mapped variants:")
        print(f"Pathogenic/Likely pathogenic: {len(pathogenic_variants)} ({len(pathogenic_variants)/len(mapped_variants_data)*100:.1f}%)")
        print(f"Benign/Likely benign:        {len(benign_variants)} ({len(benign_variants)/len(mapped_variants_data)*100:.1f}%)")
        print(f"Uncertain/Conflicting:       {len(uncertain_variants)} ({len(uncertain_variants)/len(mapped_variants_data)*100:.1f}%)")
        
        
    
    # Count matches - each protein entry is one protein, regardless of how many accessions it has
    proteins_with_variants = 0
    proteins_without_variants = 0
    total_variants_mapped = 0
    
    for entry in results:
        # Check all possible accessions for this protein
        protein_accessions = [entry['primaryAccession']]
        protein_accessions.extend(entry.get('secondaryAccessions', []))
        
        # Check if any accession has ClinVar variants
        protein_has_variants = False
        protein_variant_count = 0
        
        for acc in protein_accessions:
            if acc in clinvar_variants:
                protein_has_variants = True
                protein_variant_count += clinvar_variants[acc]
        
        if protein_has_variants:
            proteins_with_variants += 1
            total_variants_mapped += protein_variant_count
        else:
            proteins_without_variants += 1
    
    # Results summary
    total_proteins = len(results)  # Count actual protein entries, not accessions
    percent_with_variants = (proteins_with_variants / total_proteins) * 100
    percent_without_variants = (proteins_without_variants / total_proteins) * 100
    
    print(f'\nClinVar Mapping Results:')
    print(f'Total proteins in dataset:           {total_proteins}')
    print(f'Proteins with ClinVar variants:      {proteins_with_variants} ({percent_with_variants:.1f}%)')
    print(f'Proteins with ZERO ClinVar variants: {proteins_without_variants} ({percent_without_variants:.1f}%)')
    print(f'Total ClinVar variants mapped:       {total_variants_mapped}')
    
    if proteins_with_variants > 0:
        avg_variants = total_variants_mapped / proteins_with_variants
        print(f'Average variants per protein (for those with variants): {avg_variants:.1f}')
    
    # Detailed statistical analysis of variant distribution
    all_variant_counts = []
    for entry in results:
        # Check all possible accessions for this protein
        protein_accessions = [entry['primaryAccession']]
        protein_accessions.extend(entry.get('secondaryAccessions', []))
        
        # Sum up variants for this protein across all its accessions
        protein_variant_count = 0
        for acc in protein_accessions:
            protein_variant_count += clinvar_variants.get(acc, 0)
        
        all_variant_counts.append(protein_variant_count)
    
    # Calculate statistics
    print(f'\n--- Variant Distribution Statistics ---')
    print(f'Median variants per protein:         {statistics.median(all_variant_counts):.1f}')
    print(f'Mean variants per protein:           {statistics.mean(all_variant_counts):.1f}')
    print(f'Standard deviation:                  {statistics.stdev(all_variant_counts) if len(all_variant_counts) > 1 else 0:.1f}')
    print(f'Minimum variants:                    {min(all_variant_counts)}')
    print(f'Maximum variants:                    {max(all_variant_counts)}')
    
    # Quartiles and percentiles
    if len(all_variant_counts) > 1:
        q1 = statistics.quantiles(all_variant_counts, n=4)[0]
        q3 = statistics.quantiles(all_variant_counts, n=4)[2]
        iqr = q3 - q1
        
        print(f'Q1 (25th percentile):                {q1:.1f}')
        print(f'Q3 (75th percentile):                {q3:.1f}')
        print(f'Interquartile Range (IQR):           {iqr:.1f}')
        
        # Outlier detection using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [count for count in all_variant_counts if count < lower_bound or count > upper_bound]
        
        print(f'Outlier bounds (1.5*IQR):           [{lower_bound:.1f}, {upper_bound:.1f}]')
        print(f'Number of outlier proteins:          {len(outliers)} ({len(outliers)/len(all_variant_counts)*100:.1f}%)')
        
        if outliers:
            outliers.sort(reverse=True)
            print(f'Outlier values (top 10):             {outliers[:10]}')
    
    # Show detailed distribution by ranges
    print(f'\n--- Detailed Variant Count Ranges ---')
    ranges = [
        (0, 0, "0 variants"),
        (1, 1, "1 variant"), 
        (2, 5, "2-5 variants"),
        (6, 10, "6-10 variants"),
        (11, 20, "11-20 variants"),
        (21, 50, "21-50 variants"),
        (51, 100, "51-100 variants"),
        (101, float('inf'), "100+ variants")
    ]
    
    for min_val, max_val, label in ranges:
        count = len([x for x in all_variant_counts if min_val <= x <= max_val])
        percent = (count / len(all_variant_counts)) * 100
        print(f'{label:20s}: {count:4d} proteins ({percent:5.1f}%)')
    
    # Show top variant counts (individual values)
    variant_counts = {}
    for count in all_variant_counts:
        variant_counts[count] = variant_counts.get(count, 0) + 1
    
    print(f'\n--- Top Individual Variant Counts ---')
    for count in sorted(variant_counts.keys())[:15]:  # Show first 15 values
        num_proteins = variant_counts[count]
        percent = (num_proteins / total_proteins) * 100
        print(f'  {count:3d} variants: {num_proteins:3d} proteins ({percent:5.1f}%)')
    
    if len(variant_counts) > 15:
        print(f'  ... and {len(variant_counts) - 15} more unique counts')


    # Analyze InterPro families for proteins with no variants
    print(f'\n{"="*60}')
    print("INTERPRO FAMILIES FOR PROTEINS WITH NO CLINVAR VARIANTS")
    print(f'{"="*60}')
    
    proteins_no_variants = []
    interpro_domains_no_variants = {}
    all_interpro_domains = {}
    proteins_with_interpro = 0
    
    for entry in results:
        # Check all possible accessions for this protein
        protein_accessions = [entry['primaryAccession']]
        protein_accessions.extend(entry.get('secondaryAccessions', []))
        
        # Sum up variants for this protein across all its accessions
        protein_variant_count = 0
        for acc in protein_accessions:
            protein_variant_count += clinvar_variants.get(acc, 0)
        
        # Get InterPro domains for this protein
        xrefs = entry.get('uniProtKBCrossReferences', [])
        interpro_domains = [x['id'] for x in xrefs if x.get('database') == 'InterPro']
        
        if interpro_domains:
            proteins_with_interpro += 1
            # Count domains overall
            for domain_id in interpro_domains:
                all_interpro_domains[domain_id] = all_interpro_domains.get(domain_id, 0) + 1
        
        # Check if protein has no variants
        if protein_variant_count == 0:
            proteins_no_variants.append(entry['primaryAccession'])
            # Count domains in proteins with no variants
            for domain_id in interpro_domains:
                interpro_domains_no_variants[domain_id] = interpro_domains_no_variants.get(domain_id, 0) + 1
    
    print(f'Found {len(proteins_no_variants)} proteins with no ClinVar variants')
    print(f'These proteins contain {len(interpro_domains_no_variants)} unique InterPro domains')
    
    if interpro_domains_no_variants:
        # Sort by frequency
        sorted_domains = sorted(interpro_domains_no_variants.items(), key=lambda x: x[1], reverse=True)
        
        print(f'\nTop InterPro families in proteins with no variants:')
        for i, (domain_id, count) in enumerate(sorted_domains[:20]):
            percent_of_no_variants = (count / len(proteins_no_variants)) * 100
            print(f'  {i+1:2d}. {domain_id}: {count:2d} proteins ({percent_of_no_variants:4.1f}%)')
        
        if len(sorted_domains) > 20:
            print(f'  ... and {len(sorted_domains) - 20} more domains')
        
        # Compare with overall domain distribution
        print(f'\nComparison with overall domain distribution:')
        print(f'Domain enrichment in zero-variant proteins:')
        
        for domain_id, count_no_variants in sorted_domains[:10]:
            overall_count = all_interpro_domains.get(domain_id, 0)
            expected_in_no_variants = (overall_count * len(proteins_no_variants)) / proteins_with_interpro if proteins_with_interpro > 0 else 0
            
            if expected_in_no_variants > 0:
                enrichment = count_no_variants / expected_in_no_variants
                print(f'  {domain_id}: {enrichment:.2f}x {"enriched" if enrichment > 1.2 else "depleted" if enrichment < 0.8 else "similar"}')
    else:
        print('No InterPro domains found in proteins with zero variants')
    

except FileNotFoundError:
    print(f'ClinVar file not found: {clinvar_file}')
    print('Please ensure the file exists and path is correct.')
except Exception as e:
    print(f'Error processing ClinVar data: {e}')


#### Evidence Source Analysis ####

print(f'\n{"="*60}')
print("EVIDENCE SOURCE ANALYSIS")
print(f'{"="*60}')

# ECO code classification
EXPERIMENTAL_CODES = {
    'ECO:0000269',  # experimental evidence used in manual assertion
    'ECO:0000314',  # direct assay evidence
    'ECO:0000315',  # mutant phenotype evidence
    'ECO:0000353',  # physical interaction evidence
    'ECO:0000316',  # genetic interaction evidence
    'ECO:0000270',  # expression pattern evidence
    'ECO:0007005',  # immunofluorescence evidence
    'ECO:0006056',  # imaging assay evidence
    'ECO:0007829',  # X-ray crystallography evidence
    'ECO:0007744',  # combinatorial/structural evidence
}
BY_SIMILARITY_CODES = {
    'ECO:0000250',  # sequence similarity (By similarity)
    'ECO:0000255',  # match to sequence model (Probable/Potential)
    'ECO:0000256',  # automated sequence model match
}
CURATED_CODES = {
    'ECO:0000305',  # curator inference
    'ECO:0000303',  # non-traceable author statement
    'ECO:0000304',  # author statement
    'ECO:0000312',  # imported information
}

def collect_evidence_counts(obj, counts):
    if isinstance(obj, dict):
        if 'evidenceCode' in obj:
            code = obj['evidenceCode']
            if code in EXPERIMENTAL_CODES:
                counts['experimental'] += 1
            elif code in BY_SIMILARITY_CODES:
                counts['by_similarity'] += 1
            elif code in CURATED_CODES:
                counts['curated'] += 1
            else:
                counts['other'] += 1
        for v in obj.values():
            collect_evidence_counts(v, counts)
    elif isinstance(obj, list):
        for item in obj:
            collect_evidence_counts(item, counts)

totals = {'experimental': 0, 'by_similarity': 0, 'curated': 0, 'other': 0}
protein_profiles = {}

for entry in results:
    acc = entry['primaryAccession']
    counts = {'experimental': 0, 'by_similarity': 0, 'curated': 0, 'other': 0}
    collect_evidence_counts(entry, counts)
    protein_profiles[acc] = counts
    for k in totals:
        totals[k] += counts[k]

grand_total = sum(totals.values())

print(f'\n--- Annotation-level breakdown ({grand_total} total evidence tags) ---')
print(f'Experimental (direct):         {totals["experimental"]:6d}  ({100*totals["experimental"]/grand_total:.1f}%)')
print(f'By similarity/probable/potential:{totals["by_similarity"]:5d}  ({100*totals["by_similarity"]/grand_total:.1f}%)')
print(f'Curated/inferred:              {totals["curated"]:6d}  ({100*totals["curated"]/grand_total:.1f}%)')
if totals['other']:
    print(f'Other/unclassified:            {totals["other"]:6d}  ({100*totals["other"]/grand_total:.1f}%)')

# Per-protein profile
exp_only = []
sim_only = []
both = []
neither = []

for acc, ev in protein_profiles.items():
    has_exp = ev['experimental'] > 0
    has_sim = ev['by_similarity'] > 0
    if has_exp and has_sim:
        both.append(acc)
    elif has_exp:
        exp_only.append(acc)
    elif has_sim:
        sim_only.append(acc)
    else:
        neither.append(acc)

print(f'\n--- Per-protein evidence profile ---')
print(f'Experimental + by similarity:  {len(both):4d} proteins ({100*len(both)/total:.1f}%)')
print(f'Experimental only:             {len(exp_only):4d} proteins ({100*len(exp_only)/total:.1f}%)')
print(f'By similarity only:            {len(sim_only):4d} proteins ({100*len(sim_only)/total:.1f}%)')
print(f'Neither (curated/other only):  {len(neither):4d} proteins ({100*len(neither)/total:.1f}%)')

# Top-level proteinExistence field
from collections import Counter as _Counter
pe_counts = _Counter(entry.get('proteinExistence') for entry in results)
print(f'\n--- proteinExistence field (top-level UniProt classification) ---')
for level, count in sorted(pe_counts.items()):
    print(f'  {level}: {count} ({100*count/total:.1f}%)')

# List proteins with ONLY similarity-based evidence
if sim_only:
    print(f'\n--- Proteins with NO experimental evidence (by similarity only): {len(sim_only)} ---')
    for acc in sorted(sim_only):
        ev = protein_profiles[acc]
        print(f'  {acc}  (by_similarity={ev["by_similarity"]}, curated={ev["curated"]})')

