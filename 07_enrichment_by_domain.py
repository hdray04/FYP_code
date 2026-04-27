#Are pathogenic variants over-represented in certain structural
# classes of DNA-binding proteins, specifically within their DNA-binding domains?
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.lines import Line2D

domain_hits_csv = "Csvs/clinvar_domain_hits.csv"
classif_csv = "Csvs/protein_classifications_hierarchy.csv"

hits = pd.read_csv(domain_hits_csv)
classif = pd.read_csv(classif_csv)
classif = classif[~classif["is_te"]] #Remove transposable elements 

dna_hits = hits[hits["feature_type"] == "DNA binding"].copy()
print(f"  {len(dna_hits):,} variant-domain within DNA-binding domains")
dna_hits = dna_hits.drop_duplicates("#AlleleID")
print(f"  {len(dna_hits):,} unique variants within DNA-binding domains")

#Join variants with classifications to get structural class and gene information for each variant
merged = dna_hits.merge(
    classif[["uniprot_id","structural_class","gene"]],
    left_on="uniprot",
    right_on="uniprot_id",
    how="left"
)

merged = merged[merged["structural_class"] != "No structural annotation"] #Remove variants without structural class information
merged = merged[merged["structural_class"] != "⚠ Transposable element / ERV"] #Remove variants without structural class information

#Defining ClinVar significance categories/colours
sig_groups = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]
sig_colours = {
    "Pathogenic/LP": "#d62728", #Red
    "VUS": "#ff7f0e", #Orange
    "Benign/LB": "#2ca02c", #Green
    "Other/Conflicting": "#1f77b4" #Blue
}

#Odds Ratio
def odds_ratio_ci_95(a,b,c,d, continuity=0.5):
    corrected = min(a,b,c,d) == 0
    if corrected:
        a,b,c,d = a+continuity, b+continuity, c+continuity, d+continuity
    or_value = (a*d) / (b*c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = scipy.stats.norm.ppf(0.975) # 95% confidence
    lo = np.exp(np.log(or_value) - z*se)
    hi = np.exp(np.log(or_value) + z*se)
    return or_value, lo, hi, corrected

proteins_per_class = (
    classif[~classif["structural_class"].isin(["No structural annotation", "⚠ Transposable element / ERV"])]
            .groupby("structural_class")["uniprot_id"].nunique().rename("n_proteins")
)

pivot = (merged.groupby(["structural_class", "sig_group"]).size().unstack(fill_value=0))    

for col in sig_groups:
    if col not in pivot.columns:
        pivot[col] = 0

pivot["total_variants"] = pivot[sig_groups].sum(axis=1)
pivot = pivot.join(proteins_per_class)
pivot["variants_per_protein"] = pivot["total_variants"] / pivot["n_proteins"].round(1)
pivot = pivot.sort_values("total_variants", ascending=False)

#Calculate OR for each class compared to others, ratio of P to B 

total_path = pivot["Pathogenic/LP"].sum()
total_benign = pivot["Benign/LB"].sum()
test_classes = pivot[(pivot["n_proteins"] >= 2) & (pivot["total_variants"] > 0)].copy()
n_tests = len(test_classes)

results = []
for cls, row in test_classes.iterrows():
    cls_path = int(row["Pathogenic/LP"])
    cls_benign = int(row["Benign/LB"])
    other_path = int(total_path) - cls_path
    other_benign = int(total_benign) - cls_benign

    _, p_raw = scipy.stats.fisher_exact([[cls_path, cls_benign], [other_path, other_benign]],alternative ="greater",)
    or_hat, ci_lo, ci_hi, corrected = odds_ratio_ci_95(cls_path, cls_benign, other_path, other_benign)
    results.append({
        "structural_class": cls,
        "or_path_vs_benign": or_hat, 
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_raw": p_raw,
        "ci_continuity_corrected": corrected,
        "cls_path": cls_path,
        "cls_benign": cls_benign,
        "other_path": other_path,
        "other_benign": other_benign
    })

results_df = pd.DataFrame(results)
results_df["p_adj"] = (results_df["p_raw"] * n_tests).clip(upper=1.0)
results_df = results_df.sort_values("or_path_vs_benign", ascending=False)

sig_classes = results_df[results_df["p_adj"] < 0.05]["structural_class"]

print(f"Fisher's exact test results (Pathogenic vs Benign): {'Class':<50} {'OR':>6} {'CI95%':>18} {'p_raw':>10} {'p_adj':>10}")
print("-" * 100)
for _, row in results_df.iterrows():
   cc = "+" if row["ci_continuity_corrected"] else ""
   sig = "*" if row["p_adj"] < 0.05 else ""
   print(f"{row['structural_class']:<50} {row['or_path_vs_benign']:.2f} {cc} [{row['ci95_lo']:.2f}, {row['ci95_hi']:.2f}] {row['p_raw']:.2e} {row['p_adj']:.2e} {sig}")

results_df.to_csv("Csvs/clinvar_domain_class_enrichment_results.csv", index=False)

total_path2 = pivot["Pathogenic/LP"].sum()
total_benign2 = pivot["Benign/LB"].sum() - total_path2

results2 = []
for cls, row in test_classes.iterrows():
    cls_path = int(row["Pathogenic/LP"])
    cls_benign = int(row["Benign/LB"])
    cls_total = int(row["total_variants"])
    cls_nonpath = cls_total - cls_path
    other_path = int(total_path2) - cls_path
    other_benign = int(total_benign2) - cls_benign


    _, p_raw = scipy.stats.fisher_exact([[cls_path, cls_benign], [other_path, other_benign]],alternative ="greater",)
    or_hat, ci_lo, ci_hi, corrected = odds_ratio_ci_95(cls_path, cls_benign, other_path, other_benign)
    results2.append({
        "cls_path": cls_path,
        "cls_nonpath": cls_nonpath,
        "other_path": other_path,
        "other_nonpath": other_benign
    })

