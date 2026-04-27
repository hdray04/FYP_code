import os
import pandas as pd

FILTERED_PATH = "Csvs/alphamissense_dbp.csv"

if os.path.exists(FILTERED_PATH):
    print("Loading existing filtered AlphaMissense data...")
    am_dbp = pd.read_csv(FILTERED_PATH)
else:
    dbp_df = pd.read_csv("Csvs/protein_classifications_hierarchy.csv", usecols=["uniprot_id"])
    dbp_ids = set(dbp_df["uniprot_id"])
    print(f"Filtering for {len(dbp_ids)} DBP UniProt accessions...")

    chunks = []
    for chunk in pd.read_csv(
        "Data/AlphaMissense_aa_substitutions.tsv.gz",
        sep="\t",
        comment="#",
        chunksize=100_000,
    ):
        filtered = chunk[chunk["uniprot_id"].isin(dbp_ids)]
        if not filtered.empty:
            chunks.append(filtered)

    am_dbp = pd.concat(chunks, ignore_index=True)
    am_dbp.to_csv(FILTERED_PATH, index=False)
    print(f"Saved {len(am_dbp):,} variants across {am_dbp['uniprot_id'].nunique()} DBPs to {FILTERED_PATH}")

dbp_df = pd.read_csv("Csvs/protein_classifications_hierarchy.csv", usecols=["uniprot_id"])
am = pd.read_csv(FILTERED_PATH, usecols=["uniprot_id", "protein_variant"])

mapped = set(am["uniprot_id"].dropna().unique())
missing_dbps = dbp_df[~dbp_df["uniprot_id"].isin(mapped)].drop_duplicates(subset=["uniprot_id"])

print(f"DBPs with no AlphaMissense variants mapped: {len(missing_dbps)}")
print(missing_dbps[["uniprot_id"]])

# Merge with ClinVar variants
cv_required = {"#AlleleID", "canonical_residue", "residue_seq", "modified_residue", "uniprot", "uniprot_id"}
cv = pd.read_csv(
    "Csvs/clinvar_domain_hits.csv",
    usecols=lambda c: c in cv_required,
)

# Support either column name used for UniProt accession
if "uniprot" not in cv.columns and "uniprot_id" in cv.columns:
    cv = cv.rename(columns={"uniprot_id": "uniprot"})

# Parse AlphaMissense protein_variant like "M1A"
am_parts = am["protein_variant"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
am["ref_aa_1"] = am_parts[0]
am["pos"] = pd.to_numeric(am_parts[1], errors="coerce")
am["alt_aa_1"] = am_parts[2]

# Convert ClinVar 3-letter residues to 1-letter
aa3_to_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V"
}
cv["ref_aa_1"] = cv["canonical_residue"].map(aa3_to_1)
cv["alt_aa_1"] = cv["modified_residue"].map(aa3_to_1)
cv["pos"] = pd.to_numeric(cv["residue_seq"], errors="coerce")

# Inner join = variants found in both datasets
merged = cv.merge(
    am,
    left_on=["uniprot", "ref_aa_1", "pos", "alt_aa_1"],
    right_on=["uniprot_id", "ref_aa_1", "pos", "alt_aa_1"],
    how="inner",
    suffixes=("_clinvar", "_am")
)

print("Matched variants:", len(merged))
print("Unique matched AlleleID:", merged["#AlleleID"].nunique())

merged_dedup = merged.drop_duplicates(subset=["#AlleleID"])
print(f"After deduplication: {len(merged_dedup):,} variants")

merged.to_csv("Csvs/clinvar_alphamissense_merged.csv", index=False)
merged_dedup.to_csv("Csvs/clinvar_alphamissense_merged_dedup.csv", index=False)