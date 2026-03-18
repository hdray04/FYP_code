import json 
import pandas as pd 
UNIPROT_JSON = "data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json"
CLINVAR_CSV = "data/clinvar_variants_28_08_2025.csv"

#Extract DNA binding domain ranges from Uniprot JSON
with open(UNIPROT_JSON) as f:
    uniprot_data = json.load(f)

domain_records = []
for entry in uniprot_data["results"]:
    accession = entry["primaryAccession"]
    for feature in entry.get("features", []):
        if feature["type"] == "DNA binding":
            start = feature["location"]["start"]["value"]
            end = feature["location"]["end"]["value"]
            domain_records.append({
                "uniprot_id": accession,
                "domain_start": start,
                "domain_end": end
            })

domains = pd.DataFrame(domain_records)
print(f"{len(domains)} DNA binding domains extracted from Uniprot\n")
print(domains.head())

#Load ClinVar data 
cv = pd.read_csv(CLINVAR_CSV, low_memory=False)
cv = cv.dropna(subset=["uniprot", "residue_seq"]).copy()
cv["residue_seq"] = cv["residue_seq"].astype(int)

print(f"{len(cv)} ClinVar variants with valid UniProt ID and residue position\n")
print(cv["uniprot"].nunique(), "unique proteins in ClinVar\n")

#Merge with domain info 
classif = pd.read_csv("protein_classifications_hierarchy.csv")
classif = classif[~classif["is_te"]]  # Exclude TEs
dbp_ids = set(classif["uniprot_id"])

cv_dbp = cv[cv["uniprot"].isin(dbp_ids)].copy()
print(f"{len(cv_dbp)} ClinVar variants in DBP proteins\n")

#Map onto domains 
domains_dbp = domains[domains["uniprot_id"].isin(dbp_ids)].copy()

hits = []
for accession, prot_vars in cv_dbp.groupby("uniprot"):
    prot_domains = domains_dbp[domains_dbp["uniprot_id"] == accession]
    if prot_domains.empty:
        continue
    for _, dom in prot_domains.iterrows():
        in_domain = prot_vars[
            (prot_vars["residue_seq"] >= dom["domain_start"]) &
            (prot_vars["residue_seq"] <= dom["domain_end"])
        ]
        if in_domain.empty:
            continue
        in_domain["domain_start"] = dom["domain_start"]
        in_domain["domain_end"] = dom["domain_end"]
        hits.append(in_domain)


mapped = pd.concat(hits, ignore_index=True)
print(f"{len(mapped)} variant-domain hits\n")
print(f"{mapped['#AlleleID'].nunique()} unique variants mapped to domains\n")
print(f"{mapped['uniprot'].nunique()} unique proteins with domain-mapped variants\n")
print(f"{cv_dbp['#AlleleID'].nunique() - mapped['#AlleleID'].nunique():,} variants not mapped to any domain")

proteins_no_domain = set(cv_dbp["uniprot"]) - set(domains["uniprot_id"])
missing = cv_dbp[cv_dbp["uniprot"].isin(proteins_no_domain)]
print(f"{len(proteins_no_domain)} DBPs with no DNA binding domain annotation")
print(f"{len(missing):,} variants on those proteins")
