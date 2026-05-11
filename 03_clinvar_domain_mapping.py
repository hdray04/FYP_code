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

ONE_STAR_STATUSES = {
    "criteria provided, single submitter",
    "criteria provided, multiple submitters, no conflicts",
    "criteria provided, conflicting classifications",
    "reviewed by expert panel",
    "practice guideline",
}
cv = cv[cv["ReviewStatus"].isin(ONE_STAR_STATUSES)].copy()
print(f"After ≥1-star review filter: {len(cv):,} variants")

print(f"{len(cv)} ClinVar variants with valid UniProt ID and residue position\n")
print(cv["uniprot"].nunique(), "unique proteins in ClinVar\n")

#Merge with domain info 
classif = pd.read_csv("Csvs/protein_classifications_hierarchy.csv")
classif = classif[~classif["is_te"] & ~classif["is_noncanonical"]]  # Exclude TEs and non-canonical DBPs (e.g. FN1)
dbp_ids = set(classif["uniprot_id"])

cv_dbp = cv[cv["uniprot"].isin(dbp_ids)].copy()
print(f"{len(cv_dbp)} ClinVar variants in DBP proteins\n map to {cv_dbp['uniprot'].nunique()} unique DBP proteins\n")

#Map onto domains 
domains_dbp = domains[domains["uniprot_id"].isin(dbp_ids)].copy()

# 1. Get IN-DOMAIN variants
hits_in_domain = []
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
        in_domain["feature_type"] = "DNA binding" # Add feature_type for consistency
        hits_in_domain.append(in_domain)

# The original logic could create duplicate rows for a single variant if it hit multiple domain definitions.
# We must deduplicate by AlleleID to ensure each variant appears once.
mapped_in_domain = pd.concat(hits_in_domain, ignore_index=True).drop_duplicates(subset=["#AlleleID"])
print(f"{mapped_in_domain['#AlleleID'].nunique():,} unique variants mapped to a DNA-binding domain.")

# 2. Get OUT-OF-DOMAIN variants
in_domain_allele_ids = set(mapped_in_domain["#AlleleID"])
mapped_out_of_domain = cv_dbp[~cv_dbp["#AlleleID"].isin(in_domain_allele_ids)].copy()

# Recode clinical significance
def recode(sig):
    if sig in {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}:
        return "Pathogenic/LP"
    if sig in {"Benign", "Likely benign", "Benign/Likely benign"}:
        return "Benign/LB"
    if sig in {"Uncertain significance",
               "Uncertain significance/Uncertain risk allele",
               "Uncertain significance; drug response"}:
        return "VUS"
    return "Other/Conflicting"

mapped_in_domain["sig_group"]   = mapped_in_domain["ClinicalSignificance"].apply(recode)
mapped_out_of_domain["sig_group"] = mapped_out_of_domain["ClinicalSignificance"].apply(recode)
cv_dbp["sig_group"]  = cv_dbp["ClinicalSignificance"].apply(recode)

mapped_in_domain.to_csv("Csvs/clinvar_domain_hits.csv", index=False)
print("Saved IN-DOMAIN variants to Csvs/clinvar_domain_hits.csv")

print(f"\n{mapped_out_of_domain['#AlleleID'].nunique():,} unique variants are outside a DNA-binding domain.")
mapped_out_of_domain.to_csv("Csvs/clinvar_out_of_domain_hits.csv", index=False)
print("Saved OUT-OF-DOMAIN variants to Csvs/clinvar_out_of_domain_hits.csv\n")

in_domain_ids = set(mapped_in_domain["#AlleleID"])
in_sig  = mapped_in_domain.drop_duplicates("#AlleleID")["sig_group"].value_counts()

out_domain = mapped_out_of_domain
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

# ── Primary enrichment test: P/LP vs B/LB (in-domain vs out-of-domain) ──────
# Using only confidently classified variants as comparator groups.
# VUS and Other/Conflicting are excluded from this test.
a = int(in_sig.get("Pathogenic/LP", 0))   # P/LP in-domain
b = int(out_sig.get("Pathogenic/LP", 0))  # P/LP out-of-domain
c = int(in_sig.get("Benign/LB", 0))       # B/LB in-domain
d = int(out_sig.get("Benign/LB", 0))      # B/LB out-of-domain

_, p_raw = scipy.stats.fisher_exact([[a, b], [c, d]], alternative="greater")
or_hat, lo, hi, corrected = odds_ratio_ci_95(a, b, c, d)
fmt_p = lambda p: f"< {np.finfo(float).tiny:.2e}" if p == 0.0 else f"{p:.3e}"

print(f"\n── Primary enrichment: P/LP vs B/LB (in-domain vs out-of-domain) ──")
print(f"  P/LP:  in-domain={a:,}  out-of-domain={b:,}")
print(f"  B/LB:  in-domain={c:,}  out-of-domain={d:,}")
print(f"  OR = {or_hat:.2f}  [{lo:.2f}, {hi:.2f}]  p={fmt_p(p_raw)}")

print(f"\n── Variant counts by significance group ──")
for grp in ["Pathogenic/LP", "Benign/LB", "VUS", "Other/Conflicting"]:
    n_in_g  = int(in_sig.get(grp, 0))
    n_out_g = int(out_sig.get(grp, 0))
    print(f"  {grp:<25} in={n_in_g:>6,}  out={n_out_g:>7,}")

# ── Domain coverage vs variant density (per-protein Wilcoxon) ────────────────
# For each protein, compare the fraction of variants observed in-domain against
# the fraction of the protein sequence covered by a DNA-binding domain.
# A Wilcoxon signed-rank test on per-protein differences avoids the pooled-
# probability assumption and operates at the natural unit of the question.

prot_lengths = {
    entry["primaryAccession"]: entry["sequence"]["length"]
    for entry in uniprot_data["results"]
    if "sequence" in entry
}

in_domain_counts = (
    mapped_in_domain.drop_duplicates("#AlleleID")
    .groupby("uniprot")["#AlleleID"].count()
)

per_protein_rows = []
for accession, prot_vars in cv_dbp.groupby("uniprot"):
    prot_domains = domains_dbp[domains_dbp["uniprot_id"] == accession]
    if prot_domains.empty or accession not in prot_lengths:
        continue

    length    = prot_lengths[accession]
    intervals = sorted(zip(prot_domains["domain_start"], prot_domains["domain_end"]))
    merged    = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    domain_residues = sum(e - s + 1 for s, e in merged)
    coverage_frac   = min(domain_residues / length, 1.0)

    n_total       = prot_vars["#AlleleID"].nunique()
    n_in          = in_domain_counts.get(accession, 0)
    observed_frac = n_in / n_total

    per_protein_rows.append({
        "uniprot_id":    accession,
        "coverage_frac": coverage_frac,
        "observed_frac": observed_frac,
        "diff":          observed_frac - coverage_frac,
        "n_variants":    n_total,
        "n_in_domain":   n_in,
    })

per_protein_df = pd.DataFrame(per_protein_rows)
diffs = per_protein_df["diff"].values

wilcox_stat, wilcox_p = scipy.stats.wilcoxon(diffs, alternative="greater")
median_cov = float(np.median(per_protein_df["coverage_frac"]))
median_obs = float(np.median(per_protein_df["observed_frac"]))
median_diff = float(np.median(diffs))

observed_in_pos  = mapped_in_domain["#AlleleID"].nunique()
total_variants_pos = per_protein_df["n_variants"].sum()
observed_out_pos = total_variants_pos - observed_in_pos

print("\n── Domain coverage vs variant density (per-protein) ──")
print(f"  Proteins included:         {len(per_protein_df)}")
print(f"  Total variants:            {total_variants_pos:,}")
print(f"  In-domain variants:        {observed_in_pos:,}  ({100*observed_in_pos/total_variants_pos:.1f}%)")
print(f"  Out-of-domain variants:    {observed_out_pos:,}  ({100*observed_out_pos/total_variants_pos:.1f}%)")
print(f"\n  Median domain coverage:    {100*median_cov:.1f}%")
print(f"  Median observed fraction:  {100*median_obs:.1f}%")
print(f"  Median difference:         {100*median_diff:+.1f}pp")
print(f"  Wilcoxon signed-rank p:    {wilcox_p:.3e}  (one-sided: observed > coverage)")

# ── CHECKING: boundary precision sensitivity ──────────────────────────────────
# Domain boundaries are hard integer positions from UniProt. Variants near the
# edge of a domain may be misclassified if boundaries are off by a few residues.
# Test robustness by shrinking (±3, ±5) and expanding (±3, ±5) all intervals.

def classify_with_offset(cv_dbp, domains_dbp, offset):
    hits = []
    for accession, prot_vars in cv_dbp.groupby("uniprot"):
        prot_domains = domains_dbp[domains_dbp["uniprot_id"] == accession]
        if prot_domains.empty:
            continue
        for _, dom in prot_domains.iterrows():
            lo_bound = max(1, dom["domain_start"] - offset)
            hi_bound = dom["domain_end"] + offset
            if hi_bound < lo_bound:
                continue
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

def path_or(in_s, out_s):
    a = int(in_s.get("Pathogenic/LP", 0))
    b = int(out_s.get("Pathogenic/LP", 0))
    c = int(in_s.get("Benign/LB", 0))
    d = int(out_s.get("Benign/LB", 0))
    or_hat, _, _, _ = odds_ratio_ci_95(a, b, c, d)
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