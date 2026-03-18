import json 
import pandas as pd 
import scipy.stats
import numpy as np
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
domains = domains.drop_duplicates(subset=["uniprot_id", "domain_start", "domain_end"]) # Remove duplicate - O15266
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
        ].copy()
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

# Find the 50 variants that appear more than once in mapped
dup_ids = mapped[mapped.duplicated("#AlleleID", keep=False)]["#AlleleID"].unique()
print(mapped[mapped["#AlleleID"].isin(dup_ids)][["#AlleleID", "uniprot", "residue_seq", "domain_start", "domain_end"]].sort_values(["#AlleleID"]))

# Recode clinical significance
def recode(sig):
    if sig in {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic",
               "Pathogenic; other", "Pathogenic; drug response",
               "Likely pathogenic; drug response",
               "Likely pathogenic/Likely risk allele"}:
        return "Pathogenic/LP"
    if sig in {"Benign", "Likely benign", "Benign/Likely benign",
               "Benign; risk factor", "Benign; other"}:
        return "Benign/LB"
    if sig in {"Uncertain significance",
               "Uncertain significance/Uncertain risk allele",
               "Uncertain significance; drug response"}:
        return "VUS"
    return "Other/Conflicting"

mapped["sig_group"] = mapped["ClinicalSignificance"].apply(recode)
cv_dbp["sig_group"] = cv_dbp["ClinicalSignificance"].apply(recode)

in_domain_ids = set(mapped["#AlleleID"])
in_sig  = mapped.drop_duplicates("#AlleleID")["sig_group"].value_counts()


out_domain = cv_dbp[~cv_dbp["#AlleleID"].isin(in_domain_ids)]
out_sig = out_domain["sig_group"].value_counts()

print("In-domain variants:")
for grp, n in in_sig.items():
    print(f"  {str(grp):<20} {int(n):>6,}  ({100*int(n)/in_sig.sum():>4.1f}%)")

print("\nOut-of-domain variants:")
for grp, n in out_sig.items():
    print(f"  {str(grp):<20} {int(n):>6,}  ({100*int(n)/out_sig.sum():>4.1f}%)")

def odds_ratio_ci_95(a, b, c, d, continuity=0.5):
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + continuity, b + continuity, c + continuity, d + continuity
    or_hat = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z  = scipy.stats.norm.ppf(0.975)
    lo = np.exp(np.log(or_hat) - z * se)
    hi = np.exp(np.log(or_hat) + z * se)
    return or_hat, lo, hi, corrected

n_in  = in_sig.sum()
n_out = out_sig.sum()

print(f"{'Sig group':<20} {'In-dom%':>8}  {'Out-dom%':>9}  {'OR':>6}  {'95% CI':>18}  {'p':>10}")
print("-" * 80)

for sig in ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]:
    a = int(in_sig.get(sig, 0))   # in domain, this sig
    b = int(out_sig.get(sig, 0))  # out of domain, this sig
    c = n_in  - a                  # in domain, other sigs
    d = n_out - b                  # out of domain, other sigs

    _, p = scipy.stats.fisher_exact([[a, b], [c, d]])
    or_hat, lo, hi, corrected = odds_ratio_ci_95(a, c, b, d)

    cc    = "†" if corrected else ""
    stars = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
    print(f"{sig:<20} {100*a/n_in:>7.1f}%  {100*b/n_out:>8.1f}%  "
          f"{or_hat:>6.2f}{cc}  [{lo:>6.2f}, {hi:>6.2f}]  {p:>10.3e}{stars}")