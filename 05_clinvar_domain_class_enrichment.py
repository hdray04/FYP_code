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
    """OR with 95 % CI using log(OR) ± 1.96·SE;"""
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

results_df.to_csv("Csvs/clinvar_domain_enrichment_stats.csv", index=False)
print("Saved: Csvs/clinvar_domain_enrichment_stats.csv\n")




# ── In-domain vs out-of-domain P/LP enrichment per class ─────────────────────
# 2x2: [in-domain P/LP, out-of-domain P/LP] x [in-domain not-P/LP, out-of-domain not-P/LP]
# OR > 1 means domain variants are more pathogenic than non-domain variants
# in the same proteins — a within-protein control.

out = pd.read_csv("Csvs/clinvar_out_of_domain_hits.csv", low_memory=False)
out = out.drop_duplicates("#AlleleID")

out_merged = out.merge(
    classif[["uniprot_id", "structural_class"]],
    left_on="uniprot", right_on="uniprot_id", how="left",
)
out_merged = out_merged[~out_merged["structural_class"].isin([
    "No structural annotation",
    "⚠ Transposable element / ERV",
    "Fibronectin type III superfamily",
])]

out_pivot = (
    out_merged.groupby(["structural_class", "sig_group"])
    .size()
    .unstack(fill_value=0)
)
for col in SIG_GROUPS:
    if col not in out_pivot.columns:
        out_pivot[col] = 0
out_pivot["total_out"] = out_pivot[SIG_GROUPS].sum(axis=1)

all_classes = pivot.index.union(out_pivot.index)
inout_results = []
for cls in all_classes:
    in_plp    = int(pivot.loc[cls, "Pathogenic/LP"])    if cls in pivot.index    else 0
    in_total  = int(pivot.loc[cls, "total_variants"])   if cls in pivot.index    else 0
    out_plp   = int(out_pivot.loc[cls, "Pathogenic/LP"]) if cls in out_pivot.index else 0
    out_total = int(out_pivot.loc[cls, "total_out"])     if cls in out_pivot.index else 0

    in_notplp  = in_total  - in_plp
    out_notplp = out_total - out_plp

    if in_total == 0 or out_total == 0 or in_plp == 0 or out_plp == 0:
        continue

    _, p_raw = scipy.stats.fisher_exact(
        [[in_plp, out_plp], [in_notplp, out_notplp]],
        alternative="two-sided",
    )
    or_hat, ci_lo, ci_hi, corrected = odds_ratio_ci_95(
        in_plp, out_plp, in_notplp, out_notplp
    )
    inout_results.append({
        "structural_class"       : cls,
        "or_in_vs_out"           : or_hat,
        "ci95_lo"                : ci_lo,
        "ci95_hi"                : ci_hi,
        "p_raw"                  : p_raw,
        "ci_continuity_corrected": corrected,
        "in_plp"                 : in_plp,
        "in_total"               : in_total,
        "in_pct_plp"             : round(100 * in_plp / in_total, 1),
        "out_plp"                : out_plp,
        "out_total"              : out_total,
        "out_pct_plp"            : round(100 * out_plp / out_total, 1),
    })

inout_df = pd.DataFrame(inout_results)
inout_df = inout_df.join(proteins_per_class, on="structural_class")
inout_df = inout_df[inout_df["n_proteins"] >= 2]
inout_df["p_bonf"] = (inout_df["p_raw"] * len(inout_df)).clip(upper=1.0)
inout_df = inout_df.sort_values("or_in_vs_out", ascending=False)

print(f"\nIn-domain vs out-of-domain P/LP enrichment per class (Bonferroni n={len(inout_df)}, ≥2 proteins)\n")
print(f"  {'Class':<50} {'OR':>6}  {'95% CI':>18}  {'in%PLP':>7}  {'out%PLP':>8}  {'p_bonf':>10}")
print(f"  {'-'*107}")
for _, row in inout_df.iterrows():
    cc  = "†" if row["ci_continuity_corrected"] else ""
    sig = " *" if row["p_bonf"] < 0.05 else ""
    print(
        f"  {row['structural_class']:<50} {row['or_in_vs_out']:>6.2f}{cc}  "
        f"[{row['ci95_lo']:>6.2f}, {row['ci95_hi']:>6.2f}]  "
        f"{row['in_pct_plp']:>6.1f}%  {row['out_pct_plp']:>7.1f}%  "
        f"{row['p_bonf']:>10.3e}{sig}"
    )


inout_df.to_csv("Csvs/clinvar_inout_domain_enrichment.csv", index=False)
print("Saved: Csvs/clinvar_inout_domain_enrichment.csv\n")


# ── Per-protein in-domain vs out-of-domain P/LP rate ─────────────────────────

in_prot = (
    merged.groupby("uniprot")["sig_group"]
    .agg(in_total="size", in_plp=lambda s: (s == "Pathogenic/LP").sum())
    .reset_index()
)
out_prot = (
    out_merged.groupby("uniprot")["sig_group"]
    .agg(out_total="size", out_plp=lambda s: (s == "Pathogenic/LP").sum())
    .reset_index()
)

prot_inout = (
    in_prot.merge(out_prot, on="uniprot", how="inner")
    .merge(classif[["uniprot_id", "structural_class", "gene"]],
           left_on="uniprot", right_on="uniprot_id", how="left")
    .drop(columns="uniprot_id")
)
prot_inout = prot_inout[~prot_inout["structural_class"].isin(
    ["No structural annotation", "⚠ Transposable element / ERV", "Fibronectin type III superfamily"]
)]

prot_inout["in_pct_plp"]  = 100 * prot_inout["in_plp"]  / prot_inout["in_total"]
prot_inout["out_pct_plp"] = 100 * prot_inout["out_plp"] / prot_inout["out_total"]
prot_inout["delta_pct"]   = prot_inout["in_pct_plp"] - prot_inout["out_pct_plp"]

prot_inout_results = []
for _, row in prot_inout.iterrows():
    a, b = int(row["in_plp"]),  int(row["in_total"]  - row["in_plp"])
    c, d = int(row["out_plp"]), int(row["out_total"] - row["out_plp"])
    if (a + b) < 3 or (c + d) < 3:
        continue
    _, p_raw = scipy.stats.fisher_exact([[a, c], [b, d]], alternative="two-sided")
    or_hat, ci_lo, ci_hi, corrected = odds_ratio_ci_95(a, c, b, d)
    prot_inout_results.append({
        **row.to_dict(),
        "or_in_vs_out": or_hat,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_raw": p_raw,
        "ci_continuity_corrected": corrected,
    })

prot_inout_df = pd.DataFrame(prot_inout_results)
n_prot_tests = len(prot_inout_df)
prot_inout_df["p_bonf"] = (prot_inout_df["p_raw"] * n_prot_tests).clip(upper=1.0)
prot_inout_df = prot_inout_df.sort_values("delta_pct", ascending=False)

print(f"Per-protein in-domain vs out-of-domain P/LP rate (n={n_prot_tests} proteins with ≥3 variants in both):")
print(f"{'Gene':<12} {'UniProt':<10} {'in%PLP':>7}  {'out%PLP':>8}  {'delta':>7}  {'OR':>6}  {'p_bonf':>10}  Structural class")
print("-" * 110)
for _, row in prot_inout_df.head(30).iterrows():
    gene = row["gene"] if pd.notna(row.get("gene")) and row["gene"] else "—"
    sig  = " *" if row["p_bonf"] < 0.05 else ""
    print(f"{gene:<12} {row['uniprot']:<10} {row['in_pct_plp']:>6.1f}%  {row['out_pct_plp']:>7.1f}%  "
          f"{row['delta_pct']:>+6.1f}%  {row['or_in_vs_out']:>6.2f}  {row['p_bonf']:>10.3e}{sig}  "
          f"{row['structural_class']}")

n_sig = (prot_inout_df["p_bonf"] < 0.05).sum()
n_higher_in = (prot_inout_df["in_pct_plp"] > prot_inout_df["out_pct_plp"]).sum()
print(f"\n  {n_higher_in}/{n_prot_tests} proteins have higher P/LP rate in-domain vs out-of-domain")
print(f"  {n_sig} Bonferroni-significant (p_bonf < 0.05)")

prot_inout_df.to_csv("Csvs/per_protein_inout_plp_rate.csv", index=False)
print("Saved: Csvs/per_protein_inout_plp_rate.csv\n")


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

# ── Stacked bar chart: variant classification proportions by structural class ──

bar_data = pivot[pivot["total_variants"] >= 30].copy()
bar_data["pct_path"]   = 100 * bar_data["Pathogenic/LP"]      / bar_data["total_variants"]
bar_data["pct_vus"]    = 100 * bar_data["VUS"]                / bar_data["total_variants"]
bar_data["pct_benign"] = 100 * bar_data["Benign/LB"]          / bar_data["total_variants"]
bar_data["pct_other"]  = 100 * bar_data["Other/Conflicting"]  / bar_data["total_variants"]
bar_data = bar_data.sort_values("pct_path", ascending=True)

overall_pct_path = 100 * pivot["Pathogenic/LP"].sum() / pivot["total_variants"].sum()

y_labels = [
    f"{NAME_SHORT.get(c, c[:18])}{'*' if c in sig_classes else ''}  "
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
    if cls in sig_classes:
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


# --- Lollipop plot: % P/LP per structural class (n > 50 variants) ---
plot_df = pivot[pivot["total_variants"] > 50].copy().reset_index()
plot_df["cls_pct_path"] = 100 * plot_df["Pathogenic/LP"] / plot_df["total_variants"]
plot_df = plot_df.sort_values("cls_pct_path")

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
print("Saved Figures/clinvar_pct_plp_by_structural_class.png")


# ── Forest plot: in-domain vs out-of-domain OR per structural class ───────────

# inout_df already filtered to n_proteins >= 2
forest_df = inout_df.copy()
forest_df = forest_df.sort_values("or_in_vs_out", ascending=True).reset_index(drop=True)
forest_df["label"] = forest_df["structural_class"].map(NAME_SHORT).fillna(forest_df["structural_class"])
forest_df["sig"] = forest_df["p_bonf"] < 0.05

# Cap CIs at a display maximum and flag with arrow
CI_CAP = 300
forest_df["ci_hi_plot"] = forest_df["ci95_hi"].clip(upper=CI_CAP)
forest_df["ci_capped"]  = forest_df["ci95_hi"] > CI_CAP

n_rows = len(forest_df)
fig, ax = plt.subplots(figsize=(9, max(5, n_rows * 0.42)))

for i, row in forest_df.iterrows():
    color = "#d62728" if row["sig"] else "#2171b5"
    ax.hlines(i, row["ci95_lo"], row["ci_hi_plot"], color=color, linewidth=1.8, alpha=0.6)
    if row["ci_capped"]:
        ax.annotate("", xy=(CI_CAP, i), xytext=(CI_CAP * 0.85, i),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    ax.scatter(row["or_in_vs_out"], i, color=color, s=45, zorder=3)

ax.axvline(1, color="#555", linestyle="--", linewidth=1)
ax.set_xscale("log")

# Explicit xlim to cover all plotted CI bars
x_lo = forest_df["ci95_lo"].min() * 0.7
ax.set_xlim(x_lo, CI_CAP * 1.05)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(
    [f"{r['label']}  (n={int(r['in_total'])}/{int(r['out_total'])})"
     for _, r in forest_df.iterrows()],
    fontsize=8,
)

# in% / out% as a text column in axes-fraction space so it never overlaps the plot
for i, row in forest_df.iterrows():
    ax.text(1.02, (i + 0.5) / n_rows,
            f"{row['in_pct_plp']:.0f}% / {row['out_pct_plp']:.0f}%",
            transform=ax.transAxes, va="center", ha="left", fontsize=7, color="#444")
ax.text(1.02, 1.01, "in% / out%", transform=ax.transAxes,
        va="bottom", ha="left", fontsize=7, color="#444", fontweight="bold")

ax.set_xlabel("Odds ratio (in-domain vs out-of-domain P/LP), log scale")
ax.set_title(
    "In-domain vs out-of-domain P/LP enrichment by structural class\n"
    "Red = Bonferroni-significant  ·  → CI extends beyond axis",
    fontsize=9,
)

from matplotlib.lines import Line2D as _L2D
ax.legend(handles=[
    _L2D([0], [0], color="#d62728", lw=2, label="Bonferroni-significant"),
    _L2D([0], [0], color="#2171b5", lw=2, label="Not significant"),
], fontsize=8, loc="lower right")

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/clinvar_inout_domain_forest.png", dpi=150, bbox_inches="tight")
print("Saved Figures/clinvar_inout_domain_forest.png")


# ── Forest plot: Fisher's exact P/LP vs B/LB enrichment per class ────────────

forest_plpb = results_df[
    (results_df["cls_path"] > 0) & (results_df["cls_benign"] > 0)
].copy()
forest_plpb = forest_plpb.sort_values("or_path_vs_benign", ascending=True).reset_index(drop=True)
forest_plpb["label"] = forest_plpb["structural_class"].map(NAME_SHORT).fillna(forest_plpb["structural_class"])
forest_plpb["sig"] = forest_plpb["p_bonf"] < 0.05

CI_CAP_PB = 300
forest_plpb["ci_hi_plot"] = forest_plpb["ci95_hi"].clip(upper=CI_CAP_PB)
forest_plpb["ci_capped"]  = forest_plpb["ci95_hi"] > CI_CAP_PB

n_rows_pb = len(forest_plpb)
fig, ax = plt.subplots(figsize=(9, max(5, n_rows_pb * 0.42)))

for i, row in forest_plpb.iterrows():
    color = "#d62728" if row["sig"] else "#2171b5"
    ax.hlines(i, row["ci95_lo"], row["ci_hi_plot"], color=color, linewidth=1.8, alpha=0.6)
    if row["ci_capped"]:
        ax.annotate("", xy=(CI_CAP_PB, i), xytext=(CI_CAP_PB * 0.85, i),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    ax.scatter(row["or_path_vs_benign"], i, color=color, s=45, zorder=3)

ax.axvline(1, color="#555", linestyle="--", linewidth=1)
ax.set_xscale("log")
ax.set_xlim(forest_plpb["ci95_lo"].min() * 0.7, CI_CAP_PB * 1.05)

ax.set_yticks(range(n_rows_pb))
ax.set_yticklabels(
    [f"{r['label']}  (P/LP={int(r['cls_path'])}, B/LB={int(r['cls_benign'])})"
     for _, r in forest_plpb.iterrows()],
    fontsize=8,
)

ax.set_xlabel("Odds ratio (P/LP vs B/LB, this class vs all others), log scale")
ax.set_title(
    f"Fisher's exact: P/LP vs B/LB enrichment by structural class\n"
    f"Bonferroni n_tests={n_tests}  ·  Red = significant  ·  → CI extends beyond axis",
    fontsize=9,
)
ax.legend(handles=[
    Line2D([0], [0], color="#d62728", lw=2, label="Bonferroni-significant"),
    Line2D([0], [0], color="#2171b5", lw=2, label="Not significant"),
], fontsize=8, loc="lower right")

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/clinvar_plpb_enrichment_forest.png", dpi=150, bbox_inches="tight")
print("Saved Figures/clinvar_plpb_enrichment_forest.png")
