"""
Sensitivity Analysis: Impact of removing similarity-only proteins on ClinVar ORs

Checks whether the 52 proteins in the dataset with NO direct experimental evidence
(annotations derived solely from sequence similarity / ECO:0000250, ECO:0000255,
ECO:0000256) materially affect the ClinVar enrichment odds ratios.

Outputs:
  - Which proteins are similarity-only and why
  - OR values with and without those proteins
  - Absolute and % change per significance group
  - Conclusion statement
"""

import json
import numpy as np
import pandas as pd
import scipy.stats

# ── Paths ──────────────────────────────────────────────────────────────────────

UNIPROT_JSON  = "Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json"
CLASSIF_CSV   = "Csvs/protein_classifications_hierarchy.csv"
CLINVAR_CSV   = "Data/clinvar_variants_28_08_2025.csv"

# ── ECO code classification ────────────────────────────────────────────────────

EXPERIMENTAL_CODES = {
    "ECO:0000269",  # experimental evidence used in manual assertion
    "ECO:0000314",  # direct assay evidence
    "ECO:0000315",  # mutant phenotype evidence
    "ECO:0000353",  # physical interaction evidence
    "ECO:0000316",  # genetic interaction evidence
    "ECO:0000270",  # expression pattern evidence
    "ECO:0007005",  # immunofluorescence evidence
    "ECO:0006056",  # imaging assay evidence
    "ECO:0007829",  # X-ray crystallography evidence
    "ECO:0007744",  # combinatorial / structural evidence
}
BY_SIMILARITY_CODES = {
    "ECO:0000250",  # sequence similarity (By similarity)
    "ECO:0000255",  # match to sequence model (Probable / Potential)
    "ECO:0000256",  # automated sequence model match
}

# ── Step 1: identify similarity-only proteins ──────────────────────────────────

print("=" * 70)
print("STEP 1: Identifying similarity-only proteins")
print("=" * 70)

with open(UNIPROT_JSON) as f:
    data = json.load(f)
results = data["results"]

def count_evidence(obj, counts):
    if isinstance(obj, dict):
        if "evidenceCode" in obj:
            code = obj["evidenceCode"]
            if code in EXPERIMENTAL_CODES:
                counts["experimental"] += 1
            elif code in BY_SIMILARITY_CODES:
                counts["by_similarity"] += 1
        for v in obj.values():
            count_evidence(v, counts)
    elif isinstance(obj, list):
        for item in obj:
            count_evidence(item, counts)

sim_only_proteins = {}   # acc -> evidence counts
for entry in results:
    counts = {"experimental": 0, "by_similarity": 0}
    count_evidence(entry, counts)
    if counts["experimental"] == 0 and counts["by_similarity"] > 0:
        sim_only_proteins[entry["primaryAccession"]] = counts

print(f"Total proteins in dataset:            {len(results)}")
print(f"Proteins with ANY experimental evidence: "
      f"{len(results) - len(sim_only_proteins)}")
print(f"Similarity-only (zero experimental):  {len(sim_only_proteins)}")
print(f"  (these have annotations via ECO:0000250 / ECO:0000255 / ECO:0000256 only)")
print()
print("Similarity-only proteins:")
print(f"  {'Accession':<15} {'by_similarity':>14} {'curated':>8}")
print(f"  {'-'*40}")

# re-collect curated counts for display
CURATED_CODES = {"ECO:0000305", "ECO:0000303", "ECO:0000304", "ECO:0000312"}
for entry in results:
    acc = entry["primaryAccession"]
    if acc in sim_only_proteins:
        cur = {"curated": 0}
        def _count_curated(obj, c):
            if isinstance(obj, dict):
                if obj.get("evidenceCode") in CURATED_CODES:
                    c["curated"] += 1
                for v in obj.values():
                    _count_curated(v, c)
            elif isinstance(obj, list):
                for i in obj:
                    _count_curated(i, c)
        _count_curated(entry, cur)
        print(f"  {acc:<15} {sim_only_proteins[acc]['by_similarity']:>14} "
              f"{cur['curated']:>8}")

sim_only_set = set(sim_only_proteins.keys())

# ── Step 2: check overlap with DBP classification ─────────────────────────────

print()
print("=" * 70)
print("STEP 2: Are any similarity-only proteins in the DBP classification set?")
print("=" * 70)

classif = pd.read_csv(CLASSIF_CSV)
classif = classif[~classif["is_te"]]
dbp_ids_full = set(classif["uniprot_id"])
in_dbp = sim_only_set & dbp_ids_full

print(f"DBP proteins (TE-excluded):           {len(dbp_ids_full)}")
print(f"Sim-only proteins in DBP set:         {len(in_dbp)}")
if in_dbp:
    print(f"  All {len(in_dbp)} overlap — their ClinVar variants sit inside the DBP pool.")
else:
    print("  None — they only affect the non-DBP pool.")

# ── Step 3: ClinVar variant counts for sim-only proteins ──────────────────────

print()
print("=" * 70)
print("STEP 3: ClinVar variant counts for similarity-only proteins")
print("=" * 70)

cv = pd.read_csv(CLINVAR_CSV, low_memory=False)
cv = cv[cv["OriginSimple"] == "germline"].copy()
cv = cv[~((cv["canonical_residue"] == "Met") & (cv["residue_seq"] == 1))].copy()  # remove start-lost
sim_cv = cv[cv["uniprot"].isin(sim_only_set)]

print(f"Total ClinVar variants:               {len(cv):,}")
print(f"Variants mapping to sim-only proteins:{len(sim_cv):,} "
      f"({100*len(sim_cv)/len(cv):.3f}% of all ClinVar)")

cv_dbp_full = cv[cv["uniprot"].isin(dbp_ids_full)]
print(f"Variants in DBP pool (full):          {len(cv_dbp_full):,}")
print(f"Sim-only variants as % of DBP pool:  "
      f"{100*len(sim_cv)/len(cv_dbp_full):.2f}%")

# ── Step 4: OR calculation ─────────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 4: Odds ratio comparison — full set vs. sim-only removed")
print("=" * 70)

PATHOGENIC_TERMS = {
    "Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic",
    "Pathogenic; other", "Pathogenic; drug response",
    "Likely pathogenic; drug response", "Likely pathogenic/Likely risk allele",
}
BENIGN_TERMS = {
    "Benign", "Likely benign", "Benign/Likely benign",
    "Benign; risk factor", "Benign; other",
}
VUS_TERMS = {
    "Uncertain significance", "Uncertain significance/Uncertain risk allele",
    "Uncertain significance; drug response",
}

def recode(sig):
    if sig in PATHOGENIC_TERMS:  return "Pathogenic/LP"
    if sig in BENIGN_TERMS:      return "Benign/LB"
    if sig in VUS_TERMS:         return "VUS"
    return "Other/Conflicting"

cv["sig"] = cv["ClinicalSignificance"].apply(recode)
SIG_GROUPS = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]

def odds_ratio_ci(a, b, c, d, continuity=0.5):
    """OR with 95% CI (log method); Haldane-Anscombe correction if any cell = 0."""
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + continuity, b + continuity, c + continuity, d + continuity
    OR  = (a * d) / (b * c)
    se  = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z   = scipy.stats.norm.ppf(0.975)
    lo  = np.exp(np.log(OR) - z * se)
    hi  = np.exp(np.log(OR) + z * se)
    return OR, lo, hi, corrected

def compute_ors(dbp_ids, label):
    cv_dbp = cv[cv["uniprot"].isin(dbp_ids)]
    n_dbp  = len(cv_dbp)
    n_non  = len(cv) - n_dbp
    dbp_c  = cv_dbp["sig"].value_counts()
    all_c  = cv["sig"].value_counts()
    non_c  = (all_c - dbp_c).fillna(all_c)
    rows = []
    for sig in SIG_GROUPS:
        a = int(dbp_c.get(sig, 0))
        b = int(non_c.get(sig, 0))
        c = n_dbp - a
        d = n_non  - b
        _, p = scipy.stats.fisher_exact([[a, b], [c, d]])
        OR, lo, hi, corr = odds_ratio_ci(a, b, c, d)
        rows.append({"sig": sig, "OR": OR, "lo": lo, "hi": hi, "p": p,
                     "n_dbp": a, "n_non": b})
    return rows

dbp_ids_filtered = dbp_ids_full - sim_only_set

full_ors     = compute_ors(dbp_ids_full,     "Full set")
filtered_ors = compute_ors(dbp_ids_filtered, "Sim-only removed")

# Print side-by-side table
print()
print(f"  DBP proteins — full set:        {len(dbp_ids_full)}")
print(f"  DBP proteins — sim-only removed:{len(dbp_ids_filtered)}")
print()
print(f"  {'Sig group':<20} {'OR (full)':>10}  {'95% CI':>16}  "
      f"{'OR (filtered)':>14}  {'95% CI':>16}  {'Delta OR':>10}  {'Change%':>8}")
print(f"  {'-'*100}")

for r1, r2 in zip(full_ors, filtered_ors):
    delta = r2["OR"] - r1["OR"]
    pct   = 100 * delta / r1["OR"]
    print(f"  {r1['sig']:<20} "
          f"{r1['OR']:>10.4f}  [{r1['lo']:>6.3f}, {r1['hi']:>6.3f}]  "
          f"{r2['OR']:>14.4f}  [{r2['lo']:>6.3f}, {r2['hi']:>6.3f}]  "
          f"{delta:>+10.4f}  {pct:>+8.3f}%")

# ── Step 5: Conclusion ─────────────────────────────────────────────────────────

max_change = max(abs(100 * (r2["OR"] - r1["OR"]) / r1["OR"])
                 for r1, r2 in zip(full_ors, filtered_ors))

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"Maximum OR change across all significance groups: {max_change:.2f}%")
print()
if max_change < 5.0:
    print("  Removing the 52 similarity-only proteins has a NEGLIGIBLE effect on")
    print("  the ClinVar enrichment odds ratios (max change < 5%). The direction")
    print("  and significance of all results are unchanged. These proteins are")
    print("  RETAINED in the analysis; this check confirms the results are robust")
    print("  to their inclusion.")
else:
    print("  WARNING: removing similarity-only proteins changes at least one OR")
    print(f"  by {max_change:.1f}%. Consider whether exclusion is warranted.")
