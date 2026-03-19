"""
ClinVar Variant Density — normalised by protein length
=======================================================
Produces a top-20 table of DBPs ranked by ClinVar variants per 100 amino acids
(variant density), alongside the raw count table for comparison.

Protein lengths are sourced from the local UniProt JSON.
"""

import json
import pandas as pd

CLINVAR_CSV  = "Data/clinvar_variants_28_08_2025.csv"
CLASSIF_CSV  = "Csvs/protein_classifications_hierarchy.csv"
UNIPROT_JSON = "Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json"

# ── Load protein lengths from UniProt JSON ─────────────────────────────────────

print("Loading UniProt lengths...")
with open(UNIPROT_JSON) as f:
    uniprot_data = json.load(f)

lengths = {
    entry["primaryAccession"]: entry["sequence"]["length"]
    for entry in uniprot_data["results"]
}
print(f"  {len(lengths):,} proteins with length data\n")

# ── Load classifications (TE-excluded) ────────────────────────────────────────

classif = pd.read_csv(CLASSIF_CSV)
classif = classif[~classif["is_te"]]
dbp_ids = set(classif["uniprot_id"])

# ── Load ClinVar and filter to DBPs ───────────────────────────────────────────

print("Loading ClinVar...")
cv = pd.read_csv(CLINVAR_CSV, low_memory=False)
cv = cv[cv["OriginSimple"] == "germline"].copy()
cv = cv[~((cv["canonical_residue"] == "Met") & (cv["residue_seq"] == 1))].copy()  # remove start-lost
cv_dbp = cv[cv["uniprot"].isin(dbp_ids)].copy()
print(f"  {len(cv_dbp):,} variants across {cv_dbp['uniprot'].nunique()} DBPs\n")

# ── Per-protein variant counts ─────────────────────────────────────────────────

prot_counts = (
    cv_dbp.groupby("uniprot")
    .size()
    .reset_index(name="n_variants")
    .merge(classif[["uniprot_id", "gene", "structural_class"]],
           left_on="uniprot", right_on="uniprot_id", how="left")
)

prot_counts["length"] = prot_counts["uniprot"].map(lengths)
missing = prot_counts["length"].isna().sum()
if missing:
    print(f"  Warning: {missing} proteins missing length data — excluded from density table\n")

prot_counts_len = prot_counts.dropna(subset=["length"]).copy()
prot_counts_len["length"] = prot_counts_len["length"].astype(int)

# Variants per 100 amino acids
prot_counts_len["variants_per_100aa"] = (
    prot_counts_len["n_variants"] / prot_counts_len["length"] * 100
).round(2)

# ── Build both top-20 tables ───────────────────────────────────────────────────

top20_raw = (
    prot_counts_len
    .sort_values("n_variants", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
top20_raw.index += 1

top20_density = (
    prot_counts_len
    .sort_values("variants_per_100aa", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
top20_density.index += 1

# ── Print side-by-side ─────────────────────────────────────────────────────────

def fmt_row(rank, row, value_col, value_fmt):
    gene = row["gene"] if pd.notna(row["gene"]) and row["gene"] else "—"
    cls  = row["structural_class"] if pd.notna(row["structural_class"]) else "—"
    val  = value_fmt.format(row[value_col])
    return f"{rank:>4}  {gene:<12} {row['uniprot']:<10} {row['length']:>7,}  {val:>10}  {cls}"

header = f"{'Rank':>4}  {'Gene':<12} {'UniProt':<10} {'Length':>7}  {'Value':>10}  Structural class"
divider = "-" * 100

print("=" * 100)
print("TABLE 1: Top 20 DBPs by RAW variant count")
print("=" * 100)
print(header.replace("Value", "Variants"))
print(divider)
for rank, (_, row) in enumerate(top20_raw.iterrows(), 1):
    print(fmt_row(rank, row, "n_variants", "{:,.0f}"))

print()
print("=" * 100)
print("TABLE 2: Top 20 DBPs by VARIANT DENSITY (variants per 100 amino acids)")
print("=" * 100)
print(header.replace("Value", "Var/100aa"))
print(divider)
for rank, (_, row) in enumerate(top20_density.iterrows(), 1):
    print(fmt_row(rank, row, "variants_per_100aa", "{:.2f}"))

# ── Save both tables ───────────────────────────────────────────────────────────

cols = ["uniprot", "gene", "structural_class", "length", "n_variants", "variants_per_100aa"]
top20_raw[cols].to_csv("Csvs/top20_variants_raw.csv", index=False)
top20_density[cols].to_csv("Csvs/top20_variants_density.csv", index=False)
print("\nSaved: Csvs/top20_variants_raw.csv  |  Csvs/top20_variants_density.csv")
