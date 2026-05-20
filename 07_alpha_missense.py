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
    dbp_df = pd.read_csv("Csvs/protein_classifications_hierarchy.csv", usecols=["uniprot_id", "is_te", "is_noncanonical"])
    dbp_df = dbp_df[~dbp_df["is_te"] & ~dbp_df["is_noncanonical"]]
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

dbp_df = pd.read_csv("Csvs/protein_classifications_hierarchy.csv", usecols=["uniprot_id", "structural_class", "is_te", "is_noncanonical"]).drop_duplicates(subset=["uniprot_id"])
dbp_df = dbp_df[~dbp_df["is_te"] & ~dbp_df["is_noncanonical"]]
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
print(f"ClinVar variants loaded: {len(cv):,}")
print(cv["sig_group"].value_counts())

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

# --- B/LB Discordance: variants ClinVar calls benign but AM calls pathogenic ---
blb_am = concordance_df[concordance_df["sig_group"] == "Benign/LB"].copy()
blb_discordant = blb_am[blb_am["am_class"] == "pathogenic"].copy()

n_blb_total = len(blb_am)
n_blb_disc  = len(blb_discordant)
print(f"\n=== B/LB Discordance: ClinVar Benign/LB called pathogenic by AM ===")
print(f"B/LB variants with AM data: {n_blb_total}")
print(f"AM calls pathogenic:        {n_blb_disc} ({100*n_blb_disc/n_blb_total:.1f}%)")
print(f"AM calls ambiguous:         {(blb_am['am_class']=='ambiguous').sum()} ({100*(blb_am['am_class']=='ambiguous').sum()/n_blb_total:.1f}%)")
print(f"AM calls benign:            {(blb_am['am_class']=='benign').sum()} ({100*(blb_am['am_class']=='benign').sum()/n_blb_total:.1f}%)")
print(f"\nAM score distribution for discordant B/LB variants (median, IQR):")
print(f"  median={blb_discordant['am_pathogenicity'].median():.3f}, "
      f"IQR=[{blb_discordant['am_pathogenicity'].quantile(0.25):.3f}, "
      f"{blb_discordant['am_pathogenicity'].quantile(0.75):.3f}], "
      f"min={blb_discordant['am_pathogenicity'].min():.3f}, max={blb_discordant['am_pathogenicity'].max():.3f}")

print("\n--- Structural class breakdown (discordant B/LB) ---")
sc_blb_total = blb_am.groupby("structural_class").size().rename("n_blb")
sc_blb_disc  = blb_discordant.groupby("structural_class").size().rename("n_discordant")
sc_blb_summary = sc_blb_total.to_frame().join(sc_blb_disc, how="left").fillna(0)
sc_blb_summary["n_discordant"] = sc_blb_summary["n_discordant"].astype(int)
sc_blb_summary["pct_discordant"] = 100 * sc_blb_summary["n_discordant"] / sc_blb_summary["n_blb"]
sc_blb_summary = sc_blb_summary[sc_blb_summary["n_discordant"] > 0].sort_values("n_discordant", ascending=False)
print(sc_blb_summary.round(1).to_string())

print("\n--- Discordant B/LB variants: protein and position detail ---")
print(blb_discordant[["uniprot", "protein_variant", "am_pathogenicity", "structural_class"]]
      .sort_values(["structural_class", "am_pathogenicity"], ascending=[True, False])
      .to_string(index=False))

# --- P/LP Discordance: variants ClinVar calls pathogenic but AM calls benign ---
plp_am = concordance_df[concordance_df["sig_group"] == "Pathogenic/LP"].copy()
plp_discordant = plp_am[plp_am["am_class"] == "benign"].copy()

n_plp_total = len(plp_am)
n_plp_disc  = len(plp_discordant)
print(f"\n=== P/LP Discordance: ClinVar Pathogenic/LP called benign by AM ===")
print(f"P/LP variants with AM data: {n_plp_total}")
print(f"AM calls benign:            {n_plp_disc} ({100*n_plp_disc/n_plp_total:.1f}%)")
print(f"AM calls ambiguous:         {(plp_am['am_class']=='ambiguous').sum()} ({100*(plp_am['am_class']=='ambiguous').sum()/n_plp_total:.1f}%)")
print(f"AM calls pathogenic:        {(plp_am['am_class']=='pathogenic').sum()} ({100*(plp_am['am_class']=='pathogenic').sum()/n_plp_total:.1f}%)")

if n_plp_disc > 0:
    print(f"\nAM score distribution for discordant P/LP variants (median, IQR):")
    print(f"  median={plp_discordant['am_pathogenicity'].median():.3f}, "
          f"IQR=[{plp_discordant['am_pathogenicity'].quantile(0.25):.3f}, "
          f"{plp_discordant['am_pathogenicity'].quantile(0.75):.3f}], "
          f"min={plp_discordant['am_pathogenicity'].min():.3f}, max={plp_discordant['am_pathogenicity'].max():.3f}")

    print("\n--- Structural class breakdown (discordant P/LP) ---")
    sc_plp_total = plp_am.groupby("structural_class").size().rename("n_plp")
    sc_plp_disc  = plp_discordant.groupby("structural_class").size().rename("n_discordant")
    sc_plp_summary = sc_plp_total.to_frame().join(sc_plp_disc, how="left").fillna(0)
    sc_plp_summary["n_discordant"] = sc_plp_summary["n_discordant"].astype(int)
    sc_plp_summary["pct_discordant"] = 100 * sc_plp_summary["n_discordant"] / sc_plp_summary["n_plp"]
    sc_plp_summary = sc_plp_summary[sc_plp_summary["n_discordant"] > 0].sort_values("n_discordant", ascending=False)
    print(sc_plp_summary.round(1).to_string())

    print("\n--- Discordant P/LP variants: protein and position detail ---")
    print(plp_discordant[["uniprot", "protein_variant", "am_pathogenicity", "structural_class"]]
          .sort_values(["structural_class", "am_pathogenicity"])
          .to_string(index=False))

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
dot_colors = ["#d62728" if v >= 0.564 else "#1f77b4" for v in sc_plot["median"]]
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
dot_colors_am = ["#d62728" if v > overall_pct_pathogenic else "#1f77b4" for v in sc_am_plot["pct_pathogenic"]]
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
palette_colors = ["#d62728", "grey", "#1f77b4"] 

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
stripplot_palette = {"Pathogenic/LP": "#d62728", "VUS": "grey", "Benign/LB": "#1f77b4"}
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
vus_location_path_n = vus_df[vus_df["am_class"] == "pathogenic"].groupby("location").size().rename("n_pathogenic")
vus_location_summary = vus_location_summary.join(vus_location_counts).join(vus_location_path_n).fillna({"n_pathogenic": 0})
vus_location_summary["n_pathogenic"] = vus_location_summary["n_pathogenic"].astype(int)

print("\n--- Proportion of VUS called 'pathogenic' by AlphaMissense ---")
print(vus_location_summary[["n_vus", "n_pathogenic", "pct_pathogenic"]].to_string())

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
vus_sc_path_n = in_domain_vus[in_domain_vus["am_class"] == "pathogenic"].groupby("structural_class").size().rename("n_pathogenic")
vus_sc_summary = vus_sc_summary.join(vus_sc_counts).join(vus_sc_path_n).fillna({"n_pathogenic": 0})
vus_sc_summary["n_pathogenic"] = vus_sc_summary["n_pathogenic"].astype(int)
vus_sc_summary = vus_sc_summary[vus_sc_summary["n_vus"] >= 10]
vus_sc_summary = vus_sc_summary.sort_values("pct_pathogenic", ascending=False)

print("\n--- Proportion of IN-DOMAIN VUS called 'pathogenic' by Structural Class (n>=10 VUS) ---")
print(vus_sc_summary[["n_vus", "n_pathogenic", "pct_pathogenic"]].round(1).to_string())


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


# --- AM in-domain vs out-of-domain pathogenicity enrichment per structural class ---
import numpy as np
from scipy.stats import fisher_exact

proteins_per_class_am = dbp_df.groupby("structural_class")["uniprot_id"].nunique().rename("n_proteins")

# In-domain: merged_dedup already has am_class + structural_class
in_dom = merged_dedup[merged_dedup["am_class"].notna()].copy()
in_dom["location"] = "in"

# Out-of-domain: merged_out already exists from the VUS section; add structural class
merged_out_dedup = merged_out.drop_duplicates(subset=["#AlleleID"]).copy()
merged_out_dedup = merged_out_dedup.merge(
    dbp_df.rename(columns={"uniprot_id": "uniprot"}), on="uniprot", how="left"
)
merged_out_dedup = merged_out_dedup[merged_out_dedup["am_class"].notna()].copy()
merged_out_dedup["location"] = "out"

combined = pd.concat([
    in_dom[["#AlleleID", "structural_class", "am_class", "location"]],
    merged_out_dedup[["#AlleleID", "structural_class", "am_class", "location"]],
], ignore_index=True)
combined["am_path"] = (combined["am_class"] == "pathogenic").astype(int)

am_enrich_results = []
for sc in combined["structural_class"].dropna().unique():
    in_sc  = combined[(combined["structural_class"] == sc) & (combined["location"] == "in")]
    out_sc = combined[(combined["structural_class"] == sc) & (combined["location"] == "out")]
    a, b = (in_sc["am_path"] == 1).sum(),  (in_sc["am_path"] == 0).sum()
    c, d = (out_sc["am_path"] == 1).sum(), (out_sc["am_path"] == 0).sum()
    if (a + b) < 5 or (c + d) < 5:
        continue
    or_val, p_raw = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    # Continuity correction when any cell is zero
    cc = min(a, b, c, d) == 0
    ac, bc, cc_n, dc = (a+0.5, b+0.5, c+0.5, d+0.5) if cc else (a, b, c, d)
    or_cc = (ac * dc) / (bc * cc_n)
    se = np.sqrt(1/ac + 1/bc + 1/cc_n + 1/dc)
    ci_lo = np.exp(np.log(or_cc) - 1.96 * se)
    ci_hi = np.exp(np.log(or_cc) + 1.96 * se)
    am_enrich_results.append({
        "structural_class": sc,
        "OR": or_cc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "continuity_corrected": cc,
        "in%AM_path":  100 * a / (a + b),
        "out%AM_path": 100 * c / (c + d),
        "p_raw": p_raw,
        "n_in": a + b,
        "n_out": c + d,
    })

am_enrich_df = pd.DataFrame(am_enrich_results)
am_enrich_df = am_enrich_df.join(proteins_per_class_am, on="structural_class")
am_enrich_df = am_enrich_df[am_enrich_df["n_proteins"] >= 2]
am_enrich_df["p_bonf"] = (am_enrich_df["p_raw"] * len(am_enrich_df)).clip(upper=1.0)
am_enrich_df = am_enrich_df.sort_values("OR", ascending=False)

n_tests = len(am_enrich_df)
print(f"\n=== AM in-domain vs out-of-domain pathogenicity enrichment per class (Bonferroni n={n_tests}) ===")
print(f"{'Class':<55} {'OR':>8}  {'95% CI':>20}  {'in%AM':>7}  {'out%AM':>7}  {'p_bonf':>12}")
print("-" * 115)
for _, row in am_enrich_df.iterrows():
    sig    = "*" if row["p_bonf"] < 0.05 else ""
    ci_str = f"[{row['ci_lo']:6.2f}, {row['ci_hi']:6.2f}]"
    print(f"{row['structural_class']:<55} {row['OR']:>8.2f}  {ci_str:>20}  {row['in%AM_path']:>6.1f}%  {row['out%AM_path']:>6.1f}%  {row['p_bonf']:>12.3e} {sig}")

am_enrich_df.to_csv("Csvs/am_inout_domain_enrichment.csv", index=False)
print("Saved: Csvs/am_inout_domain_enrichment.csv")


# --- Spearman: AM OR vs ClinVar OR per structural class ---
from scipy.stats import spearmanr

cv_enrich = pd.read_csv("Csvs/clinvar_inout_domain_enrichment.csv")[["structural_class", "or_in_vs_out"]].rename(columns={"or_in_vs_out": "OR_cv"})
am_enrich_clean = am_enrich_df[["structural_class", "OR"]].rename(columns={"OR": "OR_am"})
or_merged = am_enrich_clean.merge(cv_enrich, on="structural_class").replace([np.inf, -np.inf], np.nan).dropna()

rho_or, p_or = spearmanr(or_merged["OR_am"], or_merged["OR_cv"])
print(f"\n=== Spearman: AM OR vs ClinVar OR per structural class (n={len(or_merged)}) ===")
print(f"rho={rho_or:.3f}, p={p_or:.2e}")


# --- Plot 1: Scatter of log OR_am vs log OR_cv ---
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(np.log2(or_merged["OR_cv"]), np.log2(or_merged["OR_am"]),
           s=55, color="#1f77b4", alpha=0.85, zorder=3)

for _, row in or_merged.iterrows():
    ax.annotate(row["structural_class"], (np.log2(row["OR_cv"]), np.log2(row["OR_am"])),
                fontsize=6.5, xytext=(4, 3), textcoords="offset points", color="#333")

ax.axline((0, 0), slope=1, color="#aaa", linestyle="--", linewidth=1, label="x = y")
ax.set_xlabel("log₂ OR (ClinVar P/LP: in vs out of domain)")
ax.set_ylabel("log₂ OR (AM pathogenic: in vs out of domain)")
ax.set_title(f"Spearman ρ={rho_or:.3f}, p={p_or:.2e} (n={len(or_merged)} classes)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color="#eee", zorder=0)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#eee", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/am_vs_clinvar_or_scatter.png", dpi=150, bbox_inches="tight")

print("Saved Figures/am_vs_clinvar_or_scatter.png")


# --- Plot 2: Slope / bump chart comparing class rank ---so what 
or_merged["rank_cv"] = or_merged["OR_cv"].rank(ascending=False).astype(int)
or_merged["rank_am"] = or_merged["OR_am"].rank(ascending=False).astype(int)
or_merged_slope = or_merged.sort_values("rank_cv")

# Shorten long class names for readability
def shorten(name):
    replacements = {
        " superfamily": "", " domain": "", "transcription factor": "TF",
        "DNA-binding": "DBD", "Transcription factor, ": "",
        "containing": "cont.", "Nuclear hormone receptor-like": "NHR-like",
        "Hepatocyte nuclear factor 1, N-terminal": "HNF1 N-term",
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    return name.strip(", ")

or_merged_slope["label"] = or_merged_slope["structural_class"].apply(shorten)

fig, ax = plt.subplots(figsize=(10, 7))
n = len(or_merged_slope)

for _, row in or_merged_slope.iterrows():
    rank_change = row["rank_am"] - row["rank_cv"]
    color = "#d62728" if rank_change > 2 else "#1f77b4" if rank_change < -2 else "#888"
    ax.plot([0, 1], [row["rank_cv"], row["rank_am"]], color=color, linewidth=1.4, alpha=0.75)
    ax.text(-0.02, row["rank_cv"], row["label"], ha="right", va="center", fontsize=7, color=color)
    ax.text(1.02,  row["rank_am"], row["label"], ha="left",  va="center", fontsize=7, color=color)

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(n + 0.5, 0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["ClinVar OR rank\n(in vs out)", "AM OR rank\n(in vs out)"], fontsize=10)
ax.set_ylabel("Rank (1 = highest OR)")
ax.set_title("Class ranking: ClinVar P/LP enrichment vs AM pathogenic enrichment\n(in-domain vs out-of-domain)", fontsize=10)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#d62728", lw=1.5, label="Ranks higher in AM"),
    Line2D([0], [0], color="#1f77b4", lw=1.5, label="Ranks higher in ClinVar"),
    Line2D([0], [0], color="#888",    lw=1.5, label="Similar rank"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="upper center")
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.yaxis.set_visible(False)

plt.tight_layout()
plt.savefig("Figures/am_vs_clinvar_rank_slope.png", dpi=150, bbox_inches="tight")

print("Saved Figures/am_vs_clinvar_rank_slope.png")


# --- Forest plot: AM in-domain vs out-of-domain pathogenicity enrichment ---
NAME_SHORT = {
    "Hepatocyte nuclear factor 1, N-terminal domain superfamily"         : "HNF1 N-terminal",
    "Nuclear hormone receptor-like domain superfamily"                   : "Nuclear hormone receptor",
    "CTF transcription factor/nuclear factor 1, DNA-binding domain"     : "CTF/NF-1",
    "Transcription factor, MADS-box superfamily"                        : "MADS-box",
    "p53-like transcription factor, DNA-binding domain superfamily"     : "p53-like TF",
    "High mobility group box domain superfamily"                        : "HMG box",
    "Winged helix-like DNA-binding domain superfamily"                  : "Winged helix",
    "Homeodomain-like superfamily"                                      : "Homeodomain",
    "SMAD-like domain superfamily"                                      : "SMAD-like",
    "E2F-DP heterodimerization region"                                  : "E2F/DP",
    "Nucleic acid-binding, OB-fold"                                     : "OB-fold",
    "Zinc finger superfamily"                                           : "Zinc finger",
    "Bromodomain-like superfamily"                                      : "Bromodomain",
    "Sterile alpha motif/pointed domain superfamily"                    : "SAM/Pointed",
    "Basic-leucine zipper domain superfamily"                           : "bZIP",
    "Histone-fold"                                                      : "Histone-fold",
    "DM DNA-binding domain superfamily"                                 : "DM domain",
    "Immunoglobulin-like fold"                                          : "Ig-like",
    "Ankyrin repeat-containing domain superfamily"                      : "Ankyrin repeat",
    "GCM domain superfamily"                                            : "GCM domain",
    "Bromo adjacent homology (BAH) domain superfamily"                  : "BAH domain",
    "Nucleotide-binding alpha-beta plait domain superfamily"            : "RRM/alpha-beta plait",
    "SKP1/BTB/POZ domain superfamily"                                   : "BTB/POZ",
    "AT-hook motif"                                                     : "AT-hook",
    "NDT80, DNA-binding domain superfamily"                             : "NDT80",
    "TEA/ATTS domain superfamily"                                       : "TEA/ATTS",
    "Catenin binding domain superfamily"                                : "Catenin-binding",
    "EF-hand domain pair"                                               : "EF-hand",
    "P-loop containing nucleoside triphosphate hydrolase"               : "P-loop NTPase",
    "Fibronectin type III superfamily"                                  : "Fibronectin III",
}

# am_enrich_df already filtered to n_proteins >= 2; sort for plot
am_forest = am_enrich_df.copy()
am_forest = am_forest.sort_values("OR", ascending=True).reset_index(drop=True)
am_forest["label"] = am_forest["structural_class"].map(NAME_SHORT).fillna(am_forest["structural_class"])
am_forest["sig"] = am_forest["p_bonf"] < 0.05

CI_CAP_AM = 200
am_forest["ci_hi_plot"] = am_forest["ci_hi"].clip(upper=CI_CAP_AM)
am_forest["ci_capped"]  = am_forest["ci_hi"] > CI_CAP_AM

n_rows = len(am_forest)
fig, ax = plt.subplots(figsize=(9, max(5, n_rows * 0.42)))

for i, row in am_forest.iterrows():
    color = "#d62728" if row["sig"] else "#1f77b4"
    ax.hlines(i, row["ci_lo"], row["ci_hi_plot"], color=color, linewidth=1.8, alpha=0.6)
    if row["ci_capped"]:
        ax.annotate("", xy=(CI_CAP_AM, i), xytext=(CI_CAP_AM * 0.85, i),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    ax.scatter(row["OR"], i, color=color, s=45, zorder=3)

ax.axvline(1, color="#555", linestyle="--", linewidth=1)
ax.set_xscale("log")
x_lo = am_forest["ci_lo"].min() * 0.7
ax.set_xlim(x_lo, CI_CAP_AM * 1.05)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(
    [f"{r['label']}  (n={int(r['n_in'])}/{int(r['n_out'])})" for _, r in am_forest.iterrows()],
    fontsize=8,
)

for i, row in am_forest.iterrows():
    ax.text(1.02, i,
            f"{row['in%AM_path']:.0f}% / {row['out%AM_path']:.0f}%",
            transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7, color="#444")
ax.text(1.02, 1.01, "in% / out%", transform=ax.transAxes,
        va="bottom", ha="left", fontsize=7, color="#444", fontweight="bold")

ax.set_xlabel("Odds ratio (in-domain vs out-of-domain AM pathogenic), log scale")
ax.set_title(
    "AM pathogenic enrichment: in-domain vs out-of-domain by structural class\n"
    "Red = Bonferroni-significant  ·  → CI extends beyond axis",
    fontsize=9,
)
from matplotlib.lines import Line2D as _L2D_am
ax.legend(handles=[
    _L2D_am([0], [0], color="#d62728", lw=2, label="Bonferroni-significant"),
    _L2D_am([0], [0], color="#1f77b4", lw=2, label="Not significant"),
], fontsize=8, loc="lower right")

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/am_inout_domain_forest.png", dpi=150, bbox_inches="tight")
print("Saved Figures/am_inout_domain_forest.png")


