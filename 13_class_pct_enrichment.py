"""
Per-structural-class % pathogenic / benign
  ClinVar  : % of in-domain variants that are P/LP, B/LB, VUS
  AlphaMissense : % of all DBP predictions that are pathogenic, benign, ambiguous
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────
DOMAIN_HITS_CSV    = "Csvs/clinvar_domain_hits.csv"
CLASSIF_CSV        = "Csvs/protein_classifications_hierarchy.csv"
AM_DBP_CSV         = "Csvs/alphamissense_dbp.csv"
DOMAIN_FEATURES_CSV = "Csvs/domain_features.csv"

EXCLUDE = {
    "No structural annotation",
    "⚠ Transposable element / ERV",
    "Fibronectin type III superfamily",
}

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

# ── Load classifications ───────────────────────────────────────────────────────
classif = pd.read_csv(CLASSIF_CSV)
classif = classif[~classif["is_te"]]
classif = classif[classif["n_uniprot_domain_types"] <= 1]
proteins_per_class = (
    classif[~classif["structural_class"].isin(EXCLUDE)]
    .groupby("structural_class")["uniprot_id"].nunique()
    .rename("n_proteins")
)

# ── ClinVar in-domain variants ─────────────────────────────────────────────────
print("Loading ClinVar in-domain variants...")
hits = pd.read_csv(DOMAIN_HITS_CSV, low_memory=False).drop_duplicates("#AlleleID")
cv = hits.merge(
    classif[["uniprot_id", "structural_class"]],
    left_on="uniprot", right_on="uniprot_id", how="left",
)
cv = cv[~cv["structural_class"].isin(EXCLUDE)]
print(f"  {len(cv):,} unique in-domain variants across {cv['uniprot'].nunique()} proteins")

SIG_GROUPS = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]
cv_pivot = cv.groupby(["structural_class", "sig_group"]).size().unstack(fill_value=0)
for col in SIG_GROUPS:
    if col not in cv_pivot.columns:
        cv_pivot[col] = 0
cv_pivot["n_cv"]          = cv_pivot[SIG_GROUPS].sum(axis=1)
cv_pivot["n_cv_classified"] = cv_pivot["Pathogenic/LP"] + cv_pivot["Benign/LB"]
cv_pivot["pct_path"]    = 100 * cv_pivot["Pathogenic/LP"] / cv_pivot["n_cv_classified"]
cv_pivot["pct_benign"]  = 100 * cv_pivot["Benign/LB"]     / cv_pivot["n_cv_classified"]
cv_pivot["pct_vus"]     = 100 * cv_pivot["VUS"]           / cv_pivot["n_cv"]
cv_pivot["pct_other"]   = 100 * cv_pivot["Other/Conflicting"] / cv_pivot["n_cv"]

# ── AlphaMissense — restrict to in-domain residues ────────────────────────────
print("Loading AlphaMissense predictions and filtering to in-domain residues...")
am_raw = pd.read_csv(AM_DBP_CSV, usecols=["uniprot_id", "protein_variant", "am_class"])
am_pos = am_raw["protein_variant"].str.extract(r"^[A-Z](\d+)[A-Z]$")
am_raw["pos"] = pd.to_numeric(am_pos[0], errors="coerce")

domains = pd.read_csv(DOMAIN_FEATURES_CSV)
domains = domains[domains["feature_type"] == "DNA binding"][["uniprot", "start", "end"]]

# Merge each AM variant with every domain interval for that protein, then keep
# rows where the position falls inside the interval.
am_joined = am_raw.merge(domains, left_on="uniprot_id", right_on="uniprot", how="inner")
am_joined = am_joined[(am_joined["pos"] >= am_joined["start"]) & (am_joined["pos"] <= am_joined["end"])]
am = am_joined.drop_duplicates(subset=["uniprot_id", "protein_variant"])[["uniprot_id", "am_class"]]
am = am.merge(classif[["uniprot_id", "structural_class"]], on="uniprot_id", how="left")
am = am[~am["structural_class"].isin(EXCLUDE)]
print(f"  {len(am):,} in-domain AM predictions across {am['uniprot_id'].nunique()} proteins")

AM_GROUPS = ["pathogenic", "ambiguous", "benign"]
am_pivot = am.groupby(["structural_class", "am_class"]).size().unstack(fill_value=0)
for col in AM_GROUPS:
    if col not in am_pivot.columns:
        am_pivot[col] = 0
am_pivot["n_am"]              = am_pivot[AM_GROUPS].sum(axis=1)
am_pivot["n_am_classified"]   = am_pivot["pathogenic"] + am_pivot["benign"]
am_pivot["pct_am_path"]   = 100 * am_pivot["pathogenic"] / am_pivot["n_am_classified"]
am_pivot["pct_am_benign"] = 100 * am_pivot["benign"]     / am_pivot["n_am_classified"]
am_pivot["pct_am_amb"]    = 100 * am_pivot["ambiguous"]  / am_pivot["n_am"]

# ── Combined summary ───────────────────────────────────────────────────────────
summary = cv_pivot[[
    "n_cv", "n_cv_classified", "Pathogenic/LP", "Benign/LB", "VUS",
    "pct_path", "pct_benign", "pct_vus", "pct_other",
]].join(
    am_pivot[["n_am", "n_am_classified", "pathogenic", "benign", "ambiguous",
              "pct_am_path", "pct_am_benign", "pct_am_amb"]],
    how="outer",
).join(proteins_per_class)

summary = summary[summary["n_cv_classified"].fillna(0) >= 10].sort_values("pct_path", ascending=False)

print("\n" + "=" * 120)
print("STRUCTURAL CLASS  %  (ClinVar in-domain  &  AlphaMissense full predictions)")
print("=" * 120)
print(
    f"\n{'Class':<50} {'n_CV':>6}  {'%P/LP':>6}  {'%B/LB':>6}  {'%VUS':>6}"
    f"  {'n_AM':>8}  {'%AM_path':>9}  {'%AM_benign':>11}  {'n_prot':>7}"
)
print("-" * 120)
for cls, row in summary.iterrows():
    def _fmt(val, width=7, dec=1):
        return f"{val:{width}.{dec}f}%" if pd.notna(val) else " " * (width + 1) + "N/A"
    print(
        f"{cls:<50} {int(row['n_cv']):>6,}  "
        f"{_fmt(row['pct_path'], 5)}  {_fmt(row['pct_benign'], 5)}  {_fmt(row['pct_vus'], 5)}  "
        f"{int(row['n_am']) if pd.notna(row.get('n_am')) else 'N/A':>8}  "
        f"{_fmt(row.get('pct_am_path'), 7)}  {_fmt(row.get('pct_am_benign'), 9)}  "
        f"{int(row['n_proteins']) if pd.notna(row.get('n_proteins')) else 'N/A':>7}"
    )

summary.to_csv("Csvs/class_pct_enrichment.csv")
print("\nSaved: Csvs/class_pct_enrichment.csv\n")

# ── Overall reference values ───────────────────────────────────────────────────
overall_pct_path    = 100 * cv_pivot["Pathogenic/LP"].sum() / cv_pivot["n_cv_classified"].sum()
overall_pct_benign  = 100 * cv_pivot["Benign/LB"].sum()     / cv_pivot["n_cv_classified"].sum()
overall_am_path     = 100 * am_pivot["pathogenic"].sum()    / am_pivot["n_am_classified"].sum()
overall_am_benign   = 100 * am_pivot["benign"].sum()        / am_pivot["n_am_classified"].sum()

print(f"Overall ClinVar:        {overall_pct_path:.1f}% P/LP  |  {overall_pct_benign:.1f}% B/LB")
print(f"Overall AlphaMissense:  {overall_am_path:.1f}% pathogenic  |  {overall_am_benign:.1f}% benign\n")


# ── Plot 1: Lollipop — % P/LP per class (ClinVar, n_cv >= 30) ──────────────────
plot_cv = summary[summary["n_cv"] >= 30].sort_values("pct_path").copy()
plot_cv["label"] = [
    f"{NAME_SHORT.get(c, c[:35])}  (n={int(row['n_cv']):,})"
    for c, row in plot_cv.iterrows()
]

fig, ax = plt.subplots(figsize=(9, max(5, len(plot_cv) * 0.42)))
y = range(len(plot_cv))
ax.hlines(y, 0, plot_cv["pct_path"], color="#ccc", linewidth=1.5, zorder=1)
dot_colors = ["#d62728" if v > overall_pct_path else "#2171b5" for v in plot_cv["pct_path"]]
ax.scatter(plot_cv["pct_path"], y, color=dot_colors, s=60, zorder=2)
ax.axvline(overall_pct_path, color="#555", linestyle="--", linewidth=1)
ax.text(overall_pct_path + 0.5, len(plot_cv) - 0.5,
        f"Overall\n({overall_pct_path:.1f}%)", fontsize=7.5, color="#555", va="top")
ax.set_yticks(y)
ax.set_yticklabels(plot_cv["label"], fontsize=8)
ax.set_xlabel("% Pathogenic/Likely Pathogenic (ClinVar, in-domain variants)", fontsize=11)
ax.set_xlim(0, max(plot_cv["pct_path"]) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("Figures/pct_plp_by_class_clinvar.png", dpi=150, bbox_inches="tight")
print("Saved Figures/pct_plp_by_class_clinvar.png")


# ── Plot 2: Lollipop — % AM pathogenic per class (n_am >= 1000) ───────────────
plot_am = summary[summary["n_am"].fillna(0) >= 1000].sort_values("pct_am_path").copy()
plot_am["label"] = [
    f"{NAME_SHORT.get(c, c[:35])}  (n={int(row['n_am']):,})"
    for c, row in plot_am.iterrows()
]

fig, ax = plt.subplots(figsize=(9, max(5, len(plot_am) * 0.42)))
y = range(len(plot_am))
ax.hlines(y, 0, plot_am["pct_am_path"], color="#ccc", linewidth=1.5, zorder=1)
am_cv_colors = ["#d62728" if plot_am.loc[c, "pct_path"] > overall_pct_path else "#2171b5"
                for c in plot_am.index]
ax.scatter(plot_am["pct_am_path"], y, color=am_cv_colors, s=60, zorder=2)
ax.axvline(overall_am_path, color="#555", linestyle="--", linewidth=1)
ax.text(overall_am_path + 0.3, len(plot_am) - 0.5,
        f"Overall\n({overall_am_path:.1f}%)", fontsize=7.5, color="#555", va="top")
ax.set_yticks(y)
ax.set_yticklabels(plot_am["label"], fontsize=8)
ax.set_xlabel("% classified as 'pathogenic' by AlphaMissense (in-domain)", fontsize=11)
ax.set_xlim(0, max(plot_am["pct_am_path"]) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("Figures/pct_pathogenic_by_class_am.png", dpi=150, bbox_inches="tight")
print("Saved Figures/pct_pathogenic_by_class_am.png")


# ── Plot 3: Side-by-side lollipop — %P/LP (ClinVar) vs %AM pathogenic ─────────
# Classes with both enough ClinVar (n>=30) and AM (n>=1000) data
both = summary[(summary["n_cv"] >= 30) & (summary["n_am"].fillna(0) >= 1000)].copy()
both["short"] = [NAME_SHORT.get(c, c[:30]) for c in both.index]
both = both.sort_values("pct_path", ascending=True).reset_index()

n = len(both)
fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.45)), sharey=True)

# Colour determined solely by ClinVar P/LP enrichment so it's consistent across panels.
cv_enriched_colors = ["#d62728" if v > overall_pct_path else "#2171b5"
                      for v in both["pct_path"]]

for ax, col, overall, xlabel in [
    (axes[0], "pct_path",    overall_pct_path, "% P/LP (ClinVar in-domain)"),
    (axes[1], "pct_am_path", overall_am_path,  "% pathogenic (AlphaMissense, in-domain)"),
]:
    vals = both[col].values
    y = range(n)
    ax.hlines(y, 0, vals, color="#ccc", linewidth=1.5, zorder=1)
    ax.scatter(vals, y, color=cv_enriched_colors, s=55, zorder=2)
    ax.axvline(overall, color="#555", linestyle="--", linewidth=1)
    ax.text(overall + 0.3, n - 0.5,
            f"Overall ({overall:.1f}%)", fontsize=7, color="#555", va="top")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
    ax.set_axisbelow(True)

axes[0].set_yticks(range(n))
axes[0].set_yticklabels(
    [f"{row['short']}  (n={int(row['n_cv']):,})" for _, row in both.iterrows()],
    fontsize=8,
)

from scipy.stats import spearmanr as _spearmanr
rho_side, p_side = _spearmanr(both["pct_path"], both["pct_am_path"])
p_str = f"{p_side:.2e}" if p_side < 0.001 else f"{p_side:.3f}"
axes[1].text(
    0.97, 0.03,
    f"Spearman ρ = {rho_side:.2f}\np = {p_str}  (n = {n})",
    transform=axes[1].transAxes,
    ha="right", va="bottom", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.8),
)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=8, label=f"Above overall ClinVar % P/LP ({overall_pct_path:.1f}%)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2171b5",
           markersize=8, label=f"Below overall ClinVar % P/LP ({overall_pct_path:.1f}%)"),
]
axes[0].legend(handles=legend_elements, fontsize=8, loc="lower right")

fig.suptitle(
    "% Pathogenic per structural class: ClinVar in-domain variants vs AlphaMissense in-domain predictions",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("Figures/pct_path_cv_vs_am_side_by_side.png", dpi=150, bbox_inches="tight")
print("Saved Figures/pct_path_cv_vs_am_side_by_side.png")


# ── Plot 4: Stacked bar — ClinVar classification % by structural class ─────────
bar_cv = summary[summary["n_cv"] >= 30].sort_values("pct_path", ascending=True).copy()
bar_cv["short"] = [NAME_SHORT.get(c, c[:30]) for c in bar_cv.index]

fig, ax = plt.subplots(figsize=(10, max(5, len(bar_cv) * 0.5)))
lefts = np.zeros(len(bar_cv))
for col, color, lbl in [
    ("pct_path",   "#d62728", "Pathogenic/LP"),
    ("pct_other",  "#c7c7c7", "Other/Conflicting"),
    ("pct_vus",    "#aec7e8", "VUS"),
    ("pct_benign", "#2ca02c", "Benign/LB"),
]:
    vals = bar_cv[col].fillna(0).values
    ax.barh(range(len(bar_cv)), vals, left=lefts, color=color, label=lbl, height=0.7)
    lefts += vals

ax.axvline(overall_pct_path, color="black", linestyle="--", linewidth=1.2,
           label=f"Overall % P/LP ({overall_pct_path:.1f}%)")
y_labels = [
    f"{row['short']}  (n={int(row['n_cv']):,})"
    for _, row in bar_cv.iterrows()
]
ax.set_yticks(range(len(bar_cv)))
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("% of in-domain ClinVar variants", fontsize=11)
ax.set_xlim(0, 100)
ax.set_title(
    "ClinVar variant classification by structural class (in-domain, ≥30 variants)\nordered by % P/LP",
    fontsize=10, fontweight="bold",
)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/pct_stacked_bar_clinvar.png", dpi=150, bbox_inches="tight")
print("Saved Figures/pct_stacked_bar_clinvar.png")


# ── Plot 5: Stacked bar — AlphaMissense classification % by structural class ───
bar_am = summary[summary["n_am"].fillna(0) >= 1000].sort_values("pct_am_path", ascending=True).copy()
bar_am["short"] = [NAME_SHORT.get(c, c[:30]) for c in bar_am.index]

fig, ax = plt.subplots(figsize=(10, max(5, len(bar_am) * 0.5)))
lefts = np.zeros(len(bar_am))
for col, color, lbl in [
    ("pct_am_path",   "#d62728", "Pathogenic"),
    ("pct_am_amb",    "#aec7e8", "Ambiguous"),
    ("pct_am_benign", "#2ca02c", "Benign"),
]:
    vals = bar_am[col].fillna(0).values
    ax.barh(range(len(bar_am)), vals, left=lefts, color=color, label=lbl, height=0.7)
    lefts += vals

ax.axvline(overall_am_path, color="black", linestyle="--", linewidth=1.2,
           label=f"Overall % pathogenic ({overall_am_path:.1f}%)")
y_labels_am = [
    f"{row['short']}  (n={int(row['n_am']):,})"
    for _, row in bar_am.iterrows()
]
ax.set_yticks(range(len(bar_am)))
ax.set_yticklabels(y_labels_am, fontsize=9)
ax.set_xlabel("% of AlphaMissense predictions", fontsize=11)
ax.set_xlim(0, 100)
ax.set_title(
    "AlphaMissense classification by structural class (≥1,000 predictions)\nordered by % pathogenic",
    fontsize=10, fontweight="bold",
)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/pct_stacked_bar_am.png", dpi=150, bbox_inches="tight")
print("Saved Figures/pct_stacked_bar_am.png")


# ── Scatter: % ClinVar P/LP vs % AM pathogenic per class ──────────────────────
from scipy.stats import spearmanr

scatter = both.copy()  # already filtered to n_cv>=30 and n_am>=1000
rho, p_rho = spearmanr(scatter["pct_path"], scatter["pct_am_path"])
print(f"\nSpearman ρ = {rho:.3f}, p = {p_rho:.2e} (n={len(scatter)} classes)")
print("Correlation: % ClinVar P/LP vs % AM pathogenic per structural class\n")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(scatter["pct_path"], scatter["pct_am_path"], s=55, color="#2171b5", alpha=0.85, zorder=3)
for _, row in scatter.iterrows():
    ax.annotate(
        row["short"],
        (row["pct_path"], row["pct_am_path"]),
        fontsize=6.5, xytext=(4, 3), textcoords="offset points", color="#333",
    )
ax.axvline(overall_pct_path,  color="#999", linestyle=":", linewidth=1)
ax.axhline(overall_am_path, color="#999", linestyle=":", linewidth=1)
ax.set_xlabel("% P/LP (ClinVar in-domain)", fontsize=11)
ax.set_ylabel("% pathogenic (AlphaMissense)", fontsize=11)
ax.set_title(
    f"Spearman ρ = {rho:.3f}, p = {p_rho:.2e}  (n = {len(scatter)} classes)\n"
    "% pathogenic per structural class: ClinVar vs AlphaMissense",
    fontsize=10,
)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color="#eee", zorder=0)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#eee", zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("Figures/pct_path_scatter_cv_vs_am.png", dpi=150, bbox_inches="tight")
print("Saved Figures/pct_path_scatter_cv_vs_am.png")
