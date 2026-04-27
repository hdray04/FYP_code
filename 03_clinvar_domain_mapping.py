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
cv = cv[cv["OriginSimple"] == "germline"].copy()
cv = cv[~((cv["canonical_residue"] == "Met") & (cv["residue_seq"] == 1))].copy()  # remove start-lost
cv = cv.dropna(subset=["uniprot", "residue_seq"]).copy()
cv["residue_seq"] = cv["residue_seq"].astype(int)

print(f"{len(cv)} ClinVar variants with valid UniProt ID and residue position\n")
print(cv["uniprot"].nunique(), "unique proteins in ClinVar\n")

#Merge with domain info 
classif = pd.read_csv("Csvs/protein_classifications_hierarchy.csv")
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

mapped["sig_group"]   = mapped["ClinicalSignificance"].apply(recode)
cv_dbp["sig_group"]  = cv_dbp["ClinicalSignificance"].apply(recode)

mapped["feature_type"] = "DNA binding"
mapped.to_csv("Csvs/clinvar_domain_hits.csv", index=False)
print("Saved: Csvs/clinvar_domain_hits.csv\n")

in_domain_ids = set(mapped["#AlleleID"])
in_sig  = mapped.drop_duplicates("#AlleleID")["sig_group"].value_counts()

out_domain = cv_dbp[~cv_dbp["#AlleleID"].isin(in_domain_ids)]
out_sig = out_domain.drop_duplicates("#AlleleID")["sig_group"].value_counts()

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

SIG_GROUPS = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]
n_tests = len(SIG_GROUPS)
print(f"  Bonferroni correction: n_tests={n_tests}")
print(f"{'Sig group':<20} {'In-dom%':>8}  {'Out-dom%':>9}  {'OR':>6}  {'95% CI':>18}  {'p_raw':>10}  {'p_bonf':>10}")
print("-" * 95)

for sig in SIG_GROUPS:
    a = int(in_sig.get(sig, 0))   # in domain, this sig
    b = int(out_sig.get(sig, 0))  # out of domain, this sig
    c = n_in  - a                  # in domain, other sigs
    d = n_out - b                  # out of domain, other sigs

    _, p_raw = scipy.stats.fisher_exact([[a, b], [c, d]])
    or_hat, lo, hi, corrected = odds_ratio_ci_95(a, c, b, d)
    p_bonf = min(p_raw * n_tests, 1.0)

    cc    = "†" if corrected else ""
    stars = " ***" if p_bonf < 0.001 else (" **" if p_bonf < 0.01 else (" *" if p_bonf < 0.05 else ""))
    fmt_p = lambda p: f"< {np.finfo(float).tiny:.2e}" if p == 0.0 else f"{p:.3e}"
    print(f"{sig:<20} {100*a/n_in:>7.1f}%  {100*b/n_out:>8.1f}%  "
          f"{or_hat:>6.2f}{cc}  [{lo:>6.2f}, {hi:>6.2f}]  {fmt_p(p_raw):>14}  {fmt_p(p_bonf):>14}{stars}")

# ── CHECKING: boundary precision sensitivity ──────────────────────────────────
# Domain boundaries are hard integer positions from UniProt. Variants near the
# edge of a domain may be misclassified if boundaries are off by a few residues.
# Test robustness by shrinking (±3, ±5) and expanding (±3, ±5) all intervals.

def classify_with_offset(cv_dbp, domains_dbp, offset):
    """Re-run in/out classification with domain boundaries shifted by `offset`
    residues (positive = expand, negative = shrink). Returns (in_sig, out_sig)."""
    hits = []
    for accession, prot_vars in cv_dbp.groupby("uniprot"):
        prot_domains = domains_dbp[domains_dbp["uniprot_id"] == accession]
        if prot_domains.empty:
            continue
        for _, dom in prot_domains.iterrows():
            lo_bound = max(1, dom["domain_start"] - offset)
            hi_bound = dom["domain_end"] + offset
            in_dom = prot_vars[
                (prot_vars["residue_seq"] >= lo_bound) &
                (prot_vars["residue_seq"] <= hi_bound)
            ].copy()
            if in_dom.empty:
                continue
            hits.append(in_dom)
    if not hits:
        mapped_off = pd.DataFrame(columns=cv_dbp.columns)
    else:
        mapped_off = pd.concat(hits, ignore_index=True).drop_duplicates("#AlleleID")
    in_ids = set(mapped_off["#AlleleID"])
    in_s   = mapped_off["sig_group"].value_counts()
    out_s  = cv_dbp[~cv_dbp["#AlleleID"].isin(in_ids)]["sig_group"].value_counts()
    return in_s, out_s

# Baseline ORs for Pathogenic/LP (primary outcome)
def path_or(in_s, out_s):
    a = int(in_s.get("Pathogenic/LP", 0))
    b = int(out_s.get("Pathogenic/LP", 0))
    c = in_s.sum() - a
    d = out_s.sum() - b
    or_hat, _, _, _ = odds_ratio_ci_95(a, c, b, d)
    return or_hat

base_or = path_or(in_sig, out_sig)

print("\n── Boundary precision sensitivity (Pathogenic/LP OR) ──")
print(f"{'Offset':>10}  {'n_in':>8}  {'OR':>6}  {'Δ OR':>8}  {'%Δ OR':>8}")
print("-" * 50)
print(f"{'baseline':>10}  {int(in_sig.sum()):>8,}  {base_or:>6.3f}  {'—':>8}  {'—':>8}")

for offset in [-5, -3, 3, 5]:
    in_s, out_s = classify_with_offset(cv_dbp, domains_dbp, offset)
    or_off = path_or(in_s, out_s)
    delta  = or_off - base_or
    pct    = 100 * delta / base_or
    label  = f"{offset:+d} residues"
    print(f"{label:>10}  {int(in_s.sum()):>8,}  {or_off:>6.3f}  {delta:>+8.3f}  {pct:>+7.1f}%")