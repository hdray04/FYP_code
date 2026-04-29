import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from sklearn.metrics import roc_auc_score

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

dbp_df = pd.read_csv("Csvs/protein_classifications_hierarchy.csv", usecols=["uniprot_id", "structural_class"]).drop_duplicates(subset=["uniprot_id"])
am = pd.read_csv(FILTERED_PATH, usecols=["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"])

mapped = set(am["uniprot_id"].dropna().unique())
missing_dbps = dbp_df[~dbp_df["uniprot_id"].isin(mapped)].drop_duplicates(subset=["uniprot_id"])

print(f"DBPs with no AlphaMissense variants mapped: {len(missing_dbps)}")
print(missing_dbps[["uniprot_id"]])

# Merge with ClinVar variants
cv_required = {"#AlleleID", "canonical_residue", "residue_seq", "modified_residue", "uniprot", "uniprot_id", "ClinicalSignificance", "feature_type", "domain_name", "sig_group"}
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

# Left join to keep all ClinVar variants and see what matches from AlphaMissense
merged = cv.merge(
    am,
    left_on=["uniprot", "ref_aa_1", "pos", "alt_aa_1"],
    right_on=["uniprot_id", "ref_aa_1", "pos", "alt_aa_1"],
    how="left",
    suffixes=("_clinvar", "_am")
)

# Add structural class information
merged = merged.merge(dbp_df, left_on="uniprot", right_on="uniprot_id", how="left", suffixes=("", "_dbp")).drop(columns=["uniprot_id_dbp"], errors="ignore")

# Analyze the result of the left join
total_variants = len(merged)
matched_variants = merged["am_pathogenicity"].notna().sum()
print(f"Total ClinVar variants processed: {total_variants:,}")
print(f"Variants with a match in AlphaMissense: {matched_variants:,} ({matched_variants/total_variants:.1%})")
print(f"Variants without a match: {total_variants - matched_variants:,}")

print("\nMatched variants (deduplicated by AlleleID):", merged.drop_duplicates(subset=['#AlleleID'])['am_pathogenicity'].notna().sum())

merged_dedup = merged.drop_duplicates(subset=["#AlleleID"]).copy()
print(f"After deduplication: {len(merged_dedup):,} total unique variants")

merged.to_csv("Csvs/clinvar_alphamissense_merged.csv", index=False)
merged_dedup.to_csv("Csvs/clinvar_alphamissense_merged_dedup.csv", index=False)

print(merged_dedup["sig_group"].value_counts())
print(merged_dedup["structural_class"].value_counts())

# --- Concordance ---
# For concordance analysis, filter to rows that have a match in AlphaMissense
concordance_df = merged_dedup[merged_dedup["am_pathogenicity"].notna()].copy()
print(f"\nPerforming concordance analysis on {len(concordance_df):,} variants with both ClinVar and AM data.")

sig_order = ["Pathogenic/LP", "VUS", "Benign/LB"]
am_order  = ["pathogenic", "ambiguous", "benign"]

concordance_df = concordance_df[concordance_df["sig_group"].isin(sig_order)].copy()
concordance_df["sig_group"] = pd.Categorical(concordance_df["sig_group"], categories=sig_order, ordered=True)
concordance_df["am_class"]  = pd.Categorical(concordance_df["am_class"],  categories=am_order,  ordered=True)

ct     = pd.crosstab(concordance_df["sig_group"], concordance_df["am_class"])[am_order].reindex(sig_order)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

print("\n=== Concordance counts ===")
print(ct.to_string())
print("\n=== Row % ===")
print(ct_pct.round(1).to_string())

chi2, p, dof, _ = chi2_contingency(ct)
print(f"\nChi-squared: {chi2:.1f}, df={dof}, p={p:.2e}")
print("Interpretation: tests whether AM classification is independent of ClinVar sig group.")
print("A significant result means AM class distribution differs across sig groups,")
print("i.e. AM scores are not random with respect to clinical significance.")

n_plp     = ct.loc["Pathogenic/LP", "pathogenic"];  tot_plp = ct.loc["Pathogenic/LP"].sum()
n_blb     = ct.loc["Benign/LB",     "benign"];       tot_blb = ct.loc["Benign/LB"].sum()
print(f"\nPathogenic/LP called pathogenic by AM: {n_plp}/{tot_plp} ({100*n_plp/tot_plp:.1f}%)")
print(f"Benign/LB called benign by AM:         {n_blb}/{tot_blb} ({100*n_blb/tot_blb:.1f}%)")

# --- Breakdown by structural class (n > 50) ---
# Use the concordance_df which is already filtered for matched variants
sc_n_all        = concordance_df.groupby("structural_class").size().rename("n")
sc_proteins_all = concordance_df.groupby("structural_class")["uniprot"].nunique()
valid_classes   = sc_n_all[(sc_n_all > 50) & (sc_proteins_all >= 3)].index
sc = concordance_df[concordance_df["structural_class"].isin(valid_classes)].copy()

sc_sig = (
    sc.groupby("structural_class")["sig_group"]
    .value_counts(normalize=True)
    .mul(100)
    .unstack(fill_value=0)
    .reindex(columns=sig_order, fill_value=0)
    .round(1)
)
sc_stats = sc.groupby("structural_class")["am_pathogenicity"].agg(
    median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)
)
sc_n  = sc_n_all[valid_classes]

sc_summary = sc_sig.join(sc_stats).join(sc_n).sort_values("median", ascending=False)

print(f"\n=== Structural class breakdown (n > 50, {len(sc_summary)} classes) ===")
print("Sig group % (row) + median AM pathogenicity score (IQR):")
print(sc_summary.to_string())

# --- Lollipop: median AM score by structural class ---
sc_plot = sc_summary.sort_values("median")
labels  = [f"{c}  (n={int(sc_n[c]):,})" for c in sc_plot.index]

fig, ax = plt.subplots(figsize=(9, 7))
y = range(len(sc_plot))

ax.hlines(y, 0, sc_plot["median"], color="#ccc", linewidth=1.5, zorder=1)
dot_colors = ["#d62728" if v >= 0.564 else "#2171b5" for v in sc_plot["median"]]
ax.scatter(sc_plot["median"], y, color=dot_colors, s=60, zorder=2)
ax.hlines(y, sc_plot["q1"].clip(lower=0), sc_plot["q3"].clip(upper=1), color=dot_colors, linewidth=3, alpha=0.4, zorder=1)

ax.axvline(0.564, color="#555", linestyle="--", linewidth=1)
ax.text(0.564 + 0.005, len(sc_plot) - 0.5, "Pathogenic\nthreshold", fontsize=7.5, color="#555", va="top")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Median AlphaMissense pathogenicity score (bar = IQR)")
ax.set_xlim(0, 1)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/am_mean_by_structural_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved Figures/am_mean_by_structural_class.png")

# --- Lollipop: % AM pathogenic by structural class ---
# This provides a more direct comparison to the ClinVar %P/LP plot
sc_am_pct = (
    sc.groupby("structural_class")["am_class"]
    .value_counts(normalize=True)
    .mul(100)
    .unstack(fill_value=0)
)
if "pathogenic" not in sc_am_pct.columns:
    sc_am_pct["pathogenic"] = 0
sc_am_pct = sc_am_pct.rename(columns={"pathogenic": "pct_pathogenic"})

sc_am_plot = sc_am_pct.join(sc_n).sort_values("pct_pathogenic")
labels_am  = [f"{c}  (n={int(sc_n[c]):,})" for c in sc_am_plot.index]

# Calculate overall % pathogenic for the reference line
overall_pct_pathogenic = 100 * (concordance_df["am_class"] == "pathogenic").sum() / len(concordance_df)

fig, ax = plt.subplots(figsize=(9, 7))
y_am = range(len(sc_am_plot))

ax.hlines(y_am, 0, sc_am_plot["pct_pathogenic"], color="#ccc", linewidth=1.5, zorder=1)
dot_colors_am = ["#d62728" if v > overall_pct_pathogenic else "#2171b5" for v in sc_am_plot["pct_pathogenic"]]
ax.scatter(sc_am_plot["pct_pathogenic"], y_am, color=dot_colors_am, s=60, zorder=2)

ax.axvline(overall_pct_pathogenic, color="#555", linestyle="--", linewidth=1)
ax.text(overall_pct_pathogenic + 0.5, len(sc_am_plot) - 0.5, f"Overall mean\n({overall_pct_pathogenic:.1f}%)", fontsize=7.5, color="#555", va="top")

ax.set_yticks(y_am)
ax.set_yticklabels(labels_am, fontsize=8)
ax.set_xlabel("% of variants classified as 'pathogenic' by AlphaMissense")
ax.set_xlim(0, max(sc_am_plot["pct_pathogenic"]) * 1.1)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/am_pct_pathogenic_by_structural_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved Figures/am_pct_pathogenic_by_structural_class.png")

# --- Statistics: Kruskal-Wallis + pairwise Mann-Whitney U (Bonferroni) ---
# Use the concordance_df which is already filtered for matched variants
group_data = {sg: concordance_df.loc[concordance_df["sig_group"] == sg, "am_pathogenicity"].dropna()
              for sg in sig_order}

kw_stat, kw_p = kruskal(*group_data.values())
print(f"\n=== Kruskal-Wallis (overall) ===")
print(f"H={kw_stat:.1f}, p={kw_p:.2e}")
print("Tests whether any group differs in AM score distribution.\n")

pairs = list(itertools.combinations(sig_order, 2))
n_comparisons = len(pairs)
print("=== Pairwise Mann-Whitney U (Bonferroni-corrected) ===")
mw_results = {}
for a, b in pairs:
    stat, p_raw = mannwhitneyu(group_data[a], group_data[b], alternative="two-sided")
    p_corr = min(p_raw * n_comparisons, 1.0)
    mw_results[(a, b)] = p_corr
    stars = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
    print(f"  {a} vs {b}: U={stat:.0f}, p_raw={p_raw:.2e}, p_corr={p_corr:.2e} {stars}")

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

# --- Box plot ---
import seaborn as sns
# Define the color palette as a list in the correct order for Seaborn
palette_colors = ["#d62728", "grey", "#2171b5"] 

fig, ax = plt.subplots(figsize=(8, 6))

# Boxplot
sns.boxplot(
    x="sig_group",
    y="am_pathogenicity",
    data=concordance_df,
    order=sig_order,
    palette=palette_colors,
    hue="sig_group",
    legend=False,
    showfliers=False,  # Correct parameter to hide outliers
    ax=ax,
    width=0.5,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2, color="#444"),
    capprops=dict(linewidth=1.2, color="#444"),
    boxprops=dict(alpha=0.75),
)

# Stripplot to show individual points
# Use a dictionary for the stripplot palette to ensure points match boxes
stripplot_palette = {"Pathogenic/LP": "#d62728", "VUS": "grey", "Benign/LB": "#2171b5"}
sns.stripplot(x="sig_group", y="am_pathogenicity", data=concordance_df, order=sig_order,
              palette=stripplot_palette, size=2, jitter=0.2, alpha=0.2, ax=ax)
ax.set_xlabel("") # Remove redundant x-axis label
ax.set_ylabel("AlphaMissense pathogenicity score")
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#ddd", zorder=0)
ax.set_axisbelow(True)
ax.axhline(0.564, color="#555", linestyle="--", linewidth=1)
ax.axhline(0.34,  color="#555", linestyle=":",  linewidth=1)
ax.text(2.28, 0.574, "Pathogenic threshold (0.564)", fontsize=7.5, color="#555", va="bottom")
ax.text(2.28, 0.350, "Benign threshold (0.34)",      fontsize=7.5, color="#555", va="bottom")
ax.spines[["top", "right"]].set_visible(False)

# Significance brackets (drawn above axes with clip_on=False)
bracket_y = [1.03, 1.09, 1.15]
pos = {sg: i for i, sg in enumerate(sig_order)}
for (a, b), y in zip(pairs, bracket_y):
    x1, x2 = pos[a], pos[b]
    ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y],
            color="black", linewidth=1, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.015, stars(mw_results[(a, b)]),
            ha="center", va="bottom", fontsize=10, clip_on=False)

plt.tight_layout()
plt.savefig("Figures/am_score_by_sig_group_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved Figures/am_score_by_sig_group_boxplot.png")


# --- VUS Analysis: What proportion does AM call pathogenic? ---
in_domain_vus = merged_dedup[merged_dedup["sig_group"] == "VUS"].copy()
in_domain_vus["location"] = "In DNA-binding domain"

# Build out-of-domain VUS by loading, converting, and merging with AM separately
cv_out_cols = {"#AlleleID", "canonical_residue", "residue_seq", "modified_residue", "uniprot", "sig_group"}
cv_out = pd.read_csv("Csvs/clinvar_out_of_domain_hits.csv", usecols=lambda c: c in cv_out_cols)
cv_out["ref_aa_1"] = cv_out["canonical_residue"].map(aa3_to_1)
cv_out["alt_aa_1"] = cv_out["modified_residue"].map(aa3_to_1)
cv_out["pos"] = pd.to_numeric(cv_out["residue_seq"], errors="coerce")
merged_out = cv_out.merge(
    am,
    left_on=["uniprot", "ref_aa_1", "pos", "alt_aa_1"],
    right_on=["uniprot_id", "ref_aa_1", "pos", "alt_aa_1"],
    how="left",
)
out_domain_vus = merged_out[merged_out["sig_group"] == "VUS"].drop_duplicates(subset=["#AlleleID"]).copy()
out_domain_vus["location"] = "Outside DNA-binding domain"

vus_df = pd.concat([in_domain_vus, out_domain_vus], ignore_index=True)
print(f"\n\n=== VUS Analysis ({len(vus_df):,} variants: {len(in_domain_vus):,} in-domain, {len(out_domain_vus):,} out-of-domain) ===")

# 1. In-domain vs. Out-of-domain
vus_location_summary = (
    vus_df.groupby("location")["am_class"]
    .value_counts(normalize=True)
    .mul(100)
    .unstack(fill_value=0)
    .rename(columns={"pathogenic": "pct_pathogenic"})
    .round(1)
)
vus_location_counts = vus_df.groupby("location").size().rename("n_vus")
vus_location_summary = vus_location_summary.join(vus_location_counts)

print("\n--- Proportion of VUS called 'pathogenic' by AlphaMissense ---")
print(vus_location_summary[["n_vus", "pct_pathogenic"]].to_string())

vus_ct = pd.crosstab(vus_df["location"], vus_df["am_class"])
chi2_vus, p_vus, dof_vus, _ = chi2_contingency(vus_ct)
print(f"\nChi-squared (in- vs out-of-domain AM class distribution): χ²={chi2_vus:.2f}, df={dof_vus}, p={p_vus:.2e}")


# 2. Breakdown by structural class (for in-domain VUS only)

vus_sc_summary = (
    in_domain_vus.groupby("structural_class")["am_class"]
    .value_counts(normalize=True)
    .mul(100)
    .unstack(fill_value=0)
    .rename(columns={"pathogenic": "pct_pathogenic"})
)
vus_sc_counts = in_domain_vus.groupby("structural_class").size().rename("n_vus")
vus_sc_summary = vus_sc_summary.join(vus_sc_counts)
vus_sc_summary = vus_sc_summary[vus_sc_summary["n_vus"] >= 10] # Filter for classes with at least 10 VUS
vus_sc_summary = vus_sc_summary.sort_values("pct_pathogenic", ascending=False)

print("\n--- Proportion of IN-DOMAIN VUS called 'pathogenic' by Structural Class (n>=10 VUS) ---")
print(vus_sc_summary[["n_vus", "pct_pathogenic"]].round(1).to_string())


# --- AUC per structural class ---
auc_results = []
# Filter for binary classification: P/LP vs B/LB
auc_df = concordance_df[concordance_df["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])].copy()

for s_class, group_df in auc_df.groupby("structural_class"):
    n_plp = (group_df["sig_group"] == "Pathogenic/LP").sum()
    n_blb = (group_df["sig_group"] == "Benign/LB").sum()

    # Need at least one of each class to calculate AUC
    if n_plp > 0 and n_blb > 0:
        y_true = (group_df["sig_group"] == "Pathogenic/LP").astype(int)
        y_score = group_df["am_pathogenicity"]
        
        auc = roc_auc_score(y_true, y_score)
        auc_results.append({
            "structural_class": s_class,
            "auc": auc,
            "n_plp": n_plp,
            "n_blb": n_blb,
            "n_total": n_plp + n_blb,
        })

if auc_results:
    auc_summary_df = pd.DataFrame(auc_results).sort_values("auc", ascending=False)
    print("\n=== AUC per Structural Class (AM score predicting P/LP vs B/LB) ===")
    print(auc_summary_df.to_string(index=False))
else:
    print("\nCould not compute AUC for any structural class (requires at least one P/LP and one B/LB variant per class).")


# --- Spearman correlation: % AM Pathogenic vs % ClinVar P/LP per structural class ---
# Uses full datasets (not just merged) so the two estimates are independent.

from scipy.stats import spearmanr

# % AM pathogenic per structural class across ALL DBP predictions
am_full = pd.read_csv("Csvs/alphamissense_dbp.csv", usecols=["uniprot_id", "am_class"])
am_full = am_full.merge(dbp_df[["uniprot_id", "structural_class"]], on="uniprot_id", how="left")
pct_am_path = (
    am_full.groupby("structural_class")["am_class"]
    .apply(lambda x: (x == "pathogenic").mean() * 100)
    .rename("pct_am_path")
)

# %P/LP per structural class from full ClinVar data
cv_full = pd.read_csv("Csvs/clinvar_domain_hits.csv", usecols=["#AlleleID", "uniprot", "sig_group"])
cv_full = cv_full.drop_duplicates(subset=["#AlleleID"])  # one row per variant
cv_full = cv_full.merge(dbp_df.rename(columns={"uniprot_id": "uniprot"}), on="uniprot", how="left")
pct_plp = (
    cv_full.groupby("structural_class")["sig_group"]
    .apply(lambda x: (x == "Pathogenic/LP").mean() * 100)
    .rename("pct_plp")
)

spear = pct_am_path.reset_index().merge(pct_plp.reset_index(), on="structural_class").dropna()
rho, p_spear = spearmanr(spear["pct_am_path"], spear["pct_plp"])
print(f"\n=== Spearman correlation: % AM pathogenic vs % ClinVar P/LP (n={len(spear)} classes) ===")
print(f"rho={rho:.3f}, p={p_spear:.2e}")


