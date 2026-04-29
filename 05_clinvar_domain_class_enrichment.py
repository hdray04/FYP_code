"""
ClinVar Variant Enrichment by Structural Class — IN DNA-BINDING DOMAIN ONLY

Only counts variants whose residue position falls within a UniProt-annotated
"DNA binding" feature of that protein (sourced from clinvar_domain_hits.csv,
filtered to feature_type == "DNA binding").

Variants falling outside a DNA-binding domain interval are excluded entirely.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.lines import Line2D

# ── Paths ───────────────────────────────────────────────────────────────────────

DOMAIN_HITS_CSV = "Csvs/clinvar_domain_hits.csv"
CLASSIF_CSV     = "Csvs/protein_classifications_hierarchy.csv"

# ── Load data ───────────────────────────────────────────────────────────────────

print("Loading data...")
hits    = pd.read_csv(DOMAIN_HITS_CSV, low_memory=False)
classif = pd.read_csv(CLASSIF_CSV)
classif = classif[~classif["is_te"]]

# A variant can map to multiple DNA-binding domain intervals (e.g. multi-domain
# proteins).  Keep one row per unique variant — sig_group is the same for all
# copies, so drop_duplicates on AlleleID is safe.
hits = hits.drop_duplicates("#AlleleID")
print(f"  {len(hits):,} unique variants after deduplication")
print(f"  {hits['uniprot'].nunique()} proteins represented\n")

# ── Merge with structural class ─────────────────────────────────────────────────

merged = hits.merge(
    classif[["uniprot_id", "structural_class", "gene"]],
    left_on="uniprot",
    right_on="uniprot_id",
    how="left",
)
merged = merged[~merged["structural_class"].isin([
    "No structural annotation",
    "⚠ Transposable element / ERV",
    "Fibronectin type III superfamily",
])]

# ── Helpers ─────────────────────────────────────────────────────────────────────

SIG_GROUPS = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]
SIG_COLORS = {
    "Pathogenic/LP"    : "#d62728",
    "VUS"              : "#aec7e8",
    "Benign/LB"        : "#2ca02c",
    "Other/Conflicting": "#c7c7c7",
}


def odds_ratio_ci_95(a, b, c, d, continuity=0.5):
    """OR with 95 % CI using log(OR) ± 1.96·SE; Haldane-Anscombe if any cell zero."""
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + continuity, b + continuity, c + continuity, d + continuity
    or_hat = (a * d) / (b * c)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z      = scipy.stats.norm.ppf(0.975)
    lo     = np.exp(np.log(or_hat) - z * se)
    hi     = np.exp(np.log(or_hat) + z * se)
    return or_hat, lo, hi, corrected

# ── Per-class pivot ─────────────────────────────────────────────────────────────

proteins_per_class = (
    classif[~classif["structural_class"].isin(
        ["No structural annotation", "⚠ Transposable element / ERV"])]
    .groupby("structural_class")["uniprot_id"]
    .nunique()
    .rename("n_proteins")
)

pivot = (
    merged.groupby(["structural_class", "sig_group"])
    .size()
    .unstack(fill_value=0)
)
for col in SIG_GROUPS:
    if col not in pivot.columns:
        pivot[col] = 0

pivot["total_variants"] = pivot[SIG_GROUPS].sum(axis=1)
pivot = pivot.join(proteins_per_class)
pivot["variants_per_protein"] = (pivot["total_variants"] / pivot["n_proteins"]).round(1)
pivot = pivot.sort_values("total_variants", ascending=False)

print(f"Classes with variants: {int((pivot['total_variants'] > 0).sum())}")

print("=" * 85)
print("IN-DOMAIN VARIANT BURDEN BY STRUCTURAL CLASS  (DNA-binding domain residues only)")
print("=" * 85)
print(f"\n{'Class':<50} {'Proteins':>8} {'Total':>7} {'P/LP':>6} {'B/LB':>6} {'VUS':>7} {'Var/prot':>9}")
print("-" * 85)
for cls, row in pivot.iterrows():
    print(f"{cls:<50} {int(row.get('n_proteins',0)):>8} {int(row['total_variants']):>7,} "
          f"{int(row.get('Pathogenic/LP',0)):>6,} {int(row.get('Benign/LB',0)):>6,} "
          f"{int(row.get('VUS',0)):>7,} {row['variants_per_protein']:>9.1f}")

pivot.to_csv("Csvs/clinvar_domain_class_enrichment.csv")
print("\nSaved: Csvs/clinvar_domain_class_enrichment.csv\n")

# ── Fisher's exact: P/LP vs B/LB, each class vs all others ────────────────────

total_path   = pivot["Pathogenic/LP"].sum()
total_benign = pivot["Benign/LB"].sum()

test_classes = pivot[(pivot["n_proteins"] >= 2) & (pivot["total_variants"] > 0)].copy()
n_tests      = len(test_classes)

results = []
for cls, row in test_classes.iterrows():
    cls_path   = int(row.get("Pathogenic/LP", 0))
    cls_benign = int(row.get("Benign/LB", 0))
    other_path   = int(total_path)   - cls_path
    other_benign = int(total_benign) - cls_benign

    _, p_raw = scipy.stats.fisher_exact(
        [[cls_path, cls_benign], [other_path, other_benign]],
        alternative="greater",
    )
    or_hat, ci_lo, ci_hi, corrected = odds_ratio_ci_95(
        cls_path, other_path, cls_benign, other_benign
    )
    results.append({
        "structural_class"        : cls,
        "or_path_vs_benign"       : or_hat,
        "ci95_lo"                 : ci_lo,
        "ci95_hi"                 : ci_hi,
        "p_raw"                   : p_raw,
        "ci_continuity_corrected" : corrected,
        "cls_path"                : cls_path,
        "cls_benign"              : cls_benign,
        "other_path"              : other_path,
        "other_benign"            : other_benign,
    })

results_df = pd.DataFrame(results)
results_df["p_bonf"] = (results_df["p_raw"] * n_tests).clip(upper=1.0)
results_df = results_df.sort_values("or_path_vs_benign", ascending=False)

sig_classes = set(results_df[results_df["p_bonf"] < 0.05]["structural_class"])

print(f"Fisher's exact: P/LP vs B/LB enrichment per class (Bonferroni n_tests={n_tests})\n")
print(f"  {'Class':<50} {'OR':>6}  {'95% CI':>18}  {'p_raw':>10}  {'p_bonf':>10}")
print(f"  {'-'*100}")
for _, row in results_df.iterrows():
    cc  = "†" if row["ci_continuity_corrected"] else ""
    sig = " *" if row["p_bonf"] < 0.05 else ""
    print(f"  {row['structural_class']:<50} {row['or_path_vs_benign']:>6.2f}{cc}  "
          f"[{row['ci95_lo']:>6.2f}, {row['ci95_hi']:>6.2f}]  "
          f"{row['p_raw']:>10.3e}  {row['p_bonf']:>10.3e}{sig}")
print("  † Haldane-Anscombe continuity correction applied\n")

results_df.to_csv("Csvs/clinvar_domain_enrichment_stats.csv", index=False)
print("Saved: Csvs/clinvar_domain_enrichment_stats.csv\n")


# ── Option 2: P/LP vs all other variants (higher power) ──────────────────────
# 2x2 table: [P/LP, not-P/LP] x [this class, all others]
# Uses the full variant pool so power is much better than P/LP vs B/LB alone.

total_path2    = pivot["Pathogenic/LP"].sum()
total_notpath2 = pivot["total_variants"].sum() - total_path2

results2 = []
for cls, row in test_classes.iterrows():
    cls_path    = int(row.get("Pathogenic/LP", 0))
    cls_total   = int(row["total_variants"])
    cls_notpath = cls_total - cls_path
    other_path    = int(total_path2)    - cls_path
    other_notpath = int(total_notpath2) - cls_notpath

    _, p_raw2 = scipy.stats.fisher_exact(
        [[cls_path, cls_notpath], [other_path, other_notpath]],
        alternative="two-sided",
    )
    or_hat2, ci_lo2, ci_hi2, corrected2 = odds_ratio_ci_95(
        cls_path, other_path, cls_notpath, other_notpath
    )
    results2.append({
        "structural_class"        : cls,
        "or_plp_vs_all"           : or_hat2,
        "ci95_lo"                 : ci_lo2,
        "ci95_hi"                 : ci_hi2,
        "p_raw"                   : p_raw2,
        "ci_continuity_corrected" : corrected2,
        "cls_path"                : cls_path,
        "cls_total"               : cls_total,
        "cls_pct_path"            : round(100 * cls_path / cls_total, 1) if cls_total else 0,
    })

results2_df = pd.DataFrame(results2)
results2_df["p_bonf"] = (results2_df["p_raw"] * n_tests).clip(upper=1.0)
results2_df = results2_df.sort_values("or_plp_vs_all", ascending=True)

sig_classes2 = set(results2_df[results2_df["p_bonf"] < 0.05]["structural_class"])

print(f"Fisher's exact: P/LP vs all other variants per class (Bonferroni n_tests={n_tests})\n")
print(f"  {'Class':<50} {'OR':>6}  {'95% CI':>18}  {'%P/LP':>6}  {'p_bonf':>10}")
print(f"  {'-'*100}")
for _, row in results2_df.sort_values("or_plp_vs_all", ascending=False).iterrows():
    sig = " *" if row["p_bonf"] < 0.05 else ""
    print(f"  {row['structural_class']:<50} {row['or_plp_vs_all']:>6.2f}  "
          f"[{row['ci95_lo']:>6.2f}, {row['ci95_hi']:>6.2f}]  "
          f"{row['cls_pct_path']:>5.1f}%  {row['p_bonf']:>10.3e}{sig}")

results2_df.to_csv("Csvs/clinvar_domain_enrichment_stats_plp_vs_all.csv", index=False)
print("\nSaved: Csvs/clinvar_domain_enrichment_stats_plp_vs_all.csv\n")

NAME_SHORT = {
    "Hepatocyte nuclear factor 1, N-terminal domain superfamily"         : "HNF1",
    "Nuclear hormone receptor-like domain superfamily"                   : "NHR",
    "CTF transcription factor/nuclear factor 1, DNA-binding domain"     : "CTF",
    "Transcription factor, MADS-box superfamily"                        : "MADS",
    "p53-like transcription factor, DNA-binding domain superfamily"     : "p53",
    "High mobility group box domain superfamily"                        : "HMG",
    "Winged helix-like DNA-binding domain superfamily"                  : "WH",
    "Homeodomain-like superfamily"                                      : "HD",
    "SMAD-like domain superfamily"                                      : "SMAD",
    "E2F-DP heterodimerization region"                                  : "E2F",
    "Nucleic acid-binding, OB-fold"                                     : "OB",
    "Zinc finger superfamily"                                           : "ZF",
    "Bromodomain-like superfamily"                                      : "Bromo",
    "Sterile alpha motif/pointed domain superfamily"                    : "SAM/ETS",
    "Histone-fold"                                                      : "Histone",
    "DM DNA-binding domain superfamily"                                 : "DM",
    "Immunoglobulin-like fold"                                          : "Ig-like",
    "Ankyrin repeat-containing domain superfamily"                      : "Ankyrin",
    "GCM domain superfamily"                                            : "GCM",
    "Bromo adjacent homology (BAH) domain superfamily"                  : "BAH",
    "Nucleotide-binding alpha-beta plait domain superfamily"            : "RRM",
    "SKP1/BTB/POZ domain superfamily"                                   : "BTB",
    "AT-hook motif"                                                     : "AT-hook",
    "NDT80, DNA-binding domain superfamily"                             : "NDT80",
    "TEA/ATTS domain superfamily"                                       : "TEAD",
    "Catenin binding domain superfamily"                                : "Catenin",
    "EF-hand domain pair"                                               : "EF-hand",
    "P-loop containing nucleoside triphosphate hydrolase"               : "P-loop",
    "Fibronectin type III superfamily"                                  : "FN3",
}

# ── Stacked bar chart: variant classification proportions by structural class ──

bar_data = pivot[pivot["total_variants"] >= 30].copy()
bar_data["pct_path"]   = 100 * bar_data["Pathogenic/LP"]      / bar_data["total_variants"]
bar_data["pct_vus"]    = 100 * bar_data["VUS"]                / bar_data["total_variants"]
bar_data["pct_benign"] = 100 * bar_data["Benign/LB"]          / bar_data["total_variants"]
bar_data["pct_other"]  = 100 * bar_data["Other/Conflicting"]  / bar_data["total_variants"]
bar_data = bar_data.sort_values("pct_path", ascending=True)

overall_pct_path = 100 * pivot["Pathogenic/LP"].sum() / pivot["total_variants"].sum()

y_labels = [
    f"{NAME_SHORT.get(c, c[:18])}{'*' if c in sig_classes2 else ''}  "
    f"(n={int(row['total_variants']):,})"
    for c, row in bar_data.iterrows()
]

fig, ax = plt.subplots(figsize=(10, max(5, len(bar_data) * 0.5)))

lefts = np.zeros(len(bar_data))
for col, color, lbl in [
    ("pct_path",   "#d62728", "Pathogenic/LP"),
    ("pct_other",  "#c7c7c7", "Other/Conflicting"),
    ("pct_vus",    "#aec7e8", "VUS"),
    ("pct_benign", "#2ca02c", "Benign/LB"),
]:
    vals = bar_data[col].values
    ax.barh(range(len(bar_data)), vals, left=lefts, color=color, label=lbl, height=0.7)
    lefts += vals

ax.axvline(overall_pct_path, color="black", linestyle="--", linewidth=1.2,
           label=f"Overall % P/LP ({overall_pct_path:.1f}%)")
ax.set_yticks(range(len(bar_data)))
ax.set_yticklabels(y_labels, fontsize=9)
for lbl_obj, cls in zip(ax.get_yticklabels(), bar_data.index):
    if cls in sig_classes2:
        lbl_obj.set_fontweight("bold")
ax.set_xlabel("% of in-domain ClinVar variants", fontsize=11)
ax.set_xlim(0, 100)
ax.set_title(
    "ClinVar variant classification by structural class (in-domain variants only)\n"
    "* Bonferroni-significant enrichment/depletion of P/LP  ·  ordered by % P/LP",
    fontsize=10, fontweight="bold",
)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/clinvar_domain_stacked_bar.png", dpi=150, bbox_inches="tight")
print("Saved: Figures/clinvar_domain_stacked_bar.png")
plt.show()

# ── Per-protein P/LP fraction dot plot ───────────────────────────────────────

per_prot = (
    merged.groupby("uniprot")["sig_group"]
    .agg(total="size", plp=lambda s: (s == "Pathogenic/LP").sum())
    .reset_index()
)
per_prot = per_prot[per_prot["total"] >= 1].copy()
per_prot["plp_frac"] = per_prot["plp"] / per_prot["total"] * 100

per_prot = per_prot.merge(
    classif[["uniprot_id", "structural_class", "gene"]],
    left_on="uniprot", right_on="uniprot_id", how="left",
)
per_prot = per_prot[~per_prot["structural_class"].isin(
    ["No structural annotation", "⚠ Transposable element / ERV"]
)]

class_sizes   = per_prot.groupby("structural_class")["uniprot"].nunique()
class_medians = (
    per_prot.groupby("structural_class")["plp_frac"]
    .median()
    .loc[class_sizes[class_sizes >= 2].index]
    .sort_values(ascending=False)
)
classes_ordered = list(class_medians.index)
overall_mean    = per_prot["plp_frac"].mean()

LABEL_THRESHOLD = 15.0

fig, ax = plt.subplots(figsize=(15, 7))
np.random.seed(42)

for i, cls in enumerate(classes_ordered):
    subset_df = per_prot[per_prot["structural_class"] == cls].copy()
    subset    = subset_df["plp_frac"]
    is_sig    = cls in sig_classes
    color     = "#d62728" if is_sig else "#6baed6"
    jitter    = np.random.uniform(-0.25, 0.25, len(subset))
    jitter_map = dict(zip(subset_df.index, jitter))

    ax.scatter(i + jitter, subset, color=color, alpha=0.65, s=28, zorder=3)
    ax.hlines(subset.median(), i - 0.32, i + 0.32,
              colors="black" if is_sig else "#444", linewidth=2, zorder=4)

    for idx, row in subset_df[subset_df["plp_frac"] >= LABEL_THRESHOLD].iterrows():
        gene_label = row.get("gene", row["uniprot"]) or row["uniprot"]
        ax.annotate(
            gene_label,
            xy=(i + jitter_map[idx], row["plp_frac"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=6.5, color="black", zorder=5,
        )

ax.axhline(overall_mean, color="gray", linestyle="--", linewidth=1.2, zorder=2)
ax.set_ylim(bottom=-2, top=max(per_prot["plp_frac"].max() + 5, 70))

x_tick_labels = [
    f"{NAME_SHORT.get(c, c[:12])}{'★' if c in sig_classes else ''}"
    for c in classes_ordered
]
ax.set_xticks(range(len(classes_ordered)))
ax.set_xticklabels(x_tick_labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("% Pathogenic/LP variants per protein (in-domain only)", fontsize=11)
ax.set_title(
    "Pathogenic/LP fraction per protein · structural class · in-domain ClinVar variants\n"
    "dashed line = mean across all DBPs · ★ Bonferroni-significant class (Fisher's exact P/LP vs B/LB)",
    fontsize=10, fontweight="bold",
)
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=8, label="Individual protein (Bonferroni-sig. class)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#6baed6",
           markersize=8, label="Individual protein (other class)"),
    Line2D([0], [0], color="black", linewidth=2, label="Class median"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
           label=f"Overall mean ({overall_mean:.1f}%)"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/clinvar_domain_plp_dotplot.png", dpi=150, bbox_inches="tight")
print("Saved: Figures/clinvar_domain_plp_dotplot.png")
plt.show()

# --- Lollipop plot: % P/LP per structural class (n > 50 variants) ---
plot_df = results2_df.merge(pivot[["total_variants"]], on="structural_class")
plot_df = plot_df[plot_df["total_variants"] > 50].sort_values("cls_pct_path")

fig, ax = plt.subplots(figsize=(9, 7))
y = range(len(plot_df))
labels = [f"{c}  (n={int(v):,})" for c, v in zip(plot_df["structural_class"], plot_df["total_variants"])]

# Calculate overall %P/LP to use as a reference line
overall_pct_plp = 100 * pivot["Pathogenic/LP"].sum() / pivot[SIG_GROUPS].sum().sum()

ax.hlines(y, 0, plot_df["cls_pct_path"], color="#ccc", linewidth=1.5, zorder=1)
dot_colors = ["#d62728" if v > overall_pct_plp else "#2171b5" for v in plot_df["cls_pct_path"]]
ax.scatter(plot_df["cls_pct_path"], y, color=dot_colors, s=60, zorder=2)

ax.axvline(overall_pct_plp, color="#555", linestyle="--", linewidth=1)
ax.text(overall_pct_plp + 0.5, len(plot_df) - 0.5, f"Overall mean\n({overall_pct_plp:.1f}%)", fontsize=7.5, color="#555", va="top")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("% Pathogenic/Likely Pathogenic variants")
ax.set_xlim(0, max(plot_df["cls_pct_path"]) * 1.1)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/clinvar_pct_plp_by_structural_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved Figures/clinvar_pct_plp_by_structural_class.png")
