"""
ClinVar Variant Enrichment by Structural Class

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.stats

# ── Paths ──────────────────────────────────────────────────────────────────────

CLINVAR_CSV       = "Data/clinvar_variants_28_08_2025.csv"
CLASSIF_CSV       = "Csvs/protein_classifications_hierarchy.csv"

# ── Load classifications ───────────────────────────────────────────────────────

print("Loading protein classifications...")
classif = pd.read_csv(CLASSIF_CSV)
# Drop TE-flagged proteins from enrichment analysis
classif = classif[~classif["is_te"]]
dbp_ids = set(classif["uniprot_id"])
print(f"  {len(classif)} proteins ({len(dbp_ids)} unique IDs, TEs excluded)\n")

# ── Load ClinVar and filter ────────────────────────────────────────────────────

print("Loading ClinVar...")
cv = pd.read_csv(CLINVAR_CSV, low_memory=False)
cv = cv[cv["OriginSimple"] == "germline"].copy()
cv = cv[~((cv["canonical_residue"] == "Met") & (cv["residue_seq"] == 1))].copy()  # remove start-lost
cv = cv.dropna(subset=["uniprot", "residue_seq"]).copy()
print(f"  {len(cv):,} total variants (germline, missense only)")

# Filter to DBP proteins
cv_dbp = cv[cv["uniprot"].isin(dbp_ids)].copy()
print(f"  {len(cv_dbp):,} variants mapping to {cv_dbp['uniprot'].nunique()} DBPs\n")

# ── Recode clinical significance ───────────────────────────────────────────────

PATHOGENIC_TERMS = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
    "Pathogenic; other",
    "Pathogenic; drug response",
    "Likely pathogenic; drug response",
    "Likely pathogenic/Likely risk allele",
}
BENIGN_TERMS = {
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
    "Benign; risk factor",
    "Benign; other",
}
VUS_TERMS = {
    "Uncertain significance",
    "Uncertain significance/Uncertain risk allele",
    "Uncertain significance; drug response",
}
CONFLICTING_TERMS = {
    "Conflicting classifications of pathogenicity",
    "Conflicting classifications of pathogenicity; other",
    "Conflicting classifications of pathogenicity; risk factor",
}

def recode(sig):
    if sig in PATHOGENIC_TERMS:
        return "Pathogenic/LP"
    if sig in BENIGN_TERMS:
        return "Benign/LB"
    if sig in VUS_TERMS:
        return "VUS"
    if sig in CONFLICTING_TERMS:
        return "Other/Conflicting"
    return "Other/Conflicting"

def odds_ratio_ci_95(a, b, c, d, continuity=0.5):
    """
    95% CI for odds ratio using log(OR) ± 1.96*SE.
    Applies Haldane-Anscombe continuity correction if any cell is zero.
    """
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + continuity, b + continuity, c + continuity, d + continuity

    or_hat = (a * d) / (b * c)
    se = np.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
    z = scipy.stats.norm.ppf(0.975)
    lo = np.exp(np.log(or_hat) - z * se)
    hi = np.exp(np.log(or_hat) + z * se)
    return or_hat, lo, hi, corrected

cv_dbp["sig_group"] = cv_dbp["ClinicalSignificance"].apply(recode)
cv["sig_group_all"] = cv["ClinicalSignificance"].apply(recode)

# Deduplicate by AlleleID so each variant counts once regardless of how many
# proteins it maps to (82 variants appear on two DBP proteins in the raw data).
cv_uniq     = cv.drop_duplicates("#AlleleID")
cv_dbp_uniq = cv_dbp.drop_duplicates("#AlleleID")

print("Clinical significance breakdown (ALL ClinVar variants):")
total_cv = len(cv_uniq)
for grp, n in cv_uniq["sig_group_all"].value_counts().items():
    pct = 100 * n / total_cv
    print(f"  {grp:<20} {n:>8,}  ({pct:.1f}%)")
print()

print("Clinical significance breakdown (DBP variants only):")
for grp, n in cv_dbp_uniq["sig_group"].value_counts().items():
    pct = 100 * n / len(cv_dbp_uniq)
    print(f"  {grp:<20} {n:>8,}  ({pct:.1f}%)")
print()

# ── DBP enrichment vs rest of ClinVar (odds ratio per sig group) ───────────────

SIG_GROUPS = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]

n_dbp     = len(cv_dbp_uniq)
n_non_dbp = len(cv_uniq) - n_dbp

dbp_counts    = cv_dbp_uniq["sig_group"].value_counts()
all_counts    = cv_uniq["sig_group_all"].value_counts()
non_dbp_counts = (all_counts - dbp_counts).fillna(all_counts)

n_dbp_tests = len(SIG_GROUPS)
print("Enrichment of clinical significance in DBPs vs rest of ClinVar (Fisher's exact):")
print(f"  Note: ascertainment bias likely inflates enrichment for well-studied genes.")
print(f"  Bonferroni correction: n_tests={n_dbp_tests}")
print(f"  {'Sig group':<20} {'DBP n':>8} {'DBP%':>6}  {'nonDBP n':>9} {'nonDBP%':>8}  {'OR':>6}  {'95% CI':>18}  {'p_raw':>10}  {'p_bonf':>10}")
print(f"  {'-'*115}")

dbp_enrichment_rows = []
for sig in SIG_GROUPS:
    a = int(dbp_counts.get(sig, 0))          # DBP, this sig
    b = int(non_dbp_counts.get(sig, 0))      # non-DBP, this sig
    c = n_dbp - a                             # DBP, other sigs
    d = n_non_dbp - b                         # non-DBP, other sigs

    _, p_raw = scipy.stats.fisher_exact([[a, b], [c, d]])
    or_hat, ci_lo, ci_hi, corrected = odds_ratio_ci_95(a, c, b, d)
    p_bonf = min(p_raw * n_dbp_tests, 1.0)

    pct_dbp    = 100 * a / n_dbp     if n_dbp     else 0
    pct_nondbp = 100 * b / n_non_dbp if n_non_dbp else 0
    cc = "†" if corrected else ""
    stars = " ***" if p_bonf < 0.001 else (" **" if p_bonf < 0.01 else (" *" if p_bonf < 0.05 else ""))
    print(f"  {sig:<20} {a:>8,} {pct_dbp:>6.1f}%  {b:>9,} {pct_nondbp:>7.1f}%  "
          f"{or_hat:>6.2f}{cc}  [{ci_lo:>6.2f}, {ci_hi:>6.2f}]  {p_raw:>10.3e}  {p_bonf:>10.3e}{stars}")
    dbp_enrichment_rows.append({
        "sig_group": sig, "n_dbp": a, "n_non_dbp": b,
        "pct_dbp": round(pct_dbp, 2), "pct_non_dbp": round(pct_nondbp, 2),
        "odds_ratio": round(or_hat, 3), "ci95_lo": round(ci_lo, 3), "ci95_hi": round(ci_hi, 3),
        "p_raw": p_raw, "p_bonf": p_bonf, "continuity_corrected": corrected,
    })

print("  † Haldane-Anscombe continuity correction applied\n")

pd.DataFrame(dbp_enrichment_rows).to_csv("Csvs/clinvar_dbp_sig_enrichment.csv", index=False)
print("Saved: clinvar_dbp_sig_enrichment.csv\n")

# ── Merge with structural classes ──────────────────────────────────────────────

merged = cv_dbp.merge(
    classif[["uniprot_id", "structural_class"]],
    left_on="uniprot",
    right_on="uniprot_id",
    how="left",
)

# Exclude unannotated proteins from enrichment plots
merged = merged[merged["structural_class"] != "No structural annotation"]


"""
# ── Per-class summary ──────────────────────────────────────────────────────────

# Proteins per class (denominator for variants-per-protein)
proteins_per_class = (
    classif[classif["structural_class"] != "No structural annotation"]
    .groupby("structural_class")["uniprot_id"]
    .nunique()
    .rename("n_proteins")
)

# Pivot: rows = structural class, columns = sig_group
pivot = (
    merged.groupby(["structural_class", "sig_group"])
    .size()
    .unstack(fill_value=0)
)
# Add after pivot is created
print("\nP-loop proteins detail:")
ploop_proteins = classif[classif["structural_class"] == 
                 "P-loop containing nucleoside triphosphate hydrolase"]
for _, row in ploop_proteins.iterrows():
    n_vars = len(cv_dbp[cv_dbp["uniprot"] == row["uniprot_id"]])
    print(f"  {row['uniprot_id']}  {n_vars} variants")

# Ensure all sig groups present
for col in ["Pathogenic/LP", "Benign/LB", "VUS", "Other/Conflicting"]:
    if col not in pivot.columns:
        pivot[col] = 0

pivot["total_variants"] = pivot.sum(axis=1)
pivot = pivot.join(proteins_per_class)
pivot["variants_per_protein"] = (pivot["total_variants"] / pivot["n_proteins"]).round(1)

# Sort by total variants descending for display
pivot = pivot.sort_values("total_variants", ascending=False)

print("=" * 85)
print("VARIANT ENRICHMENT BY STRUCTURAL CLASS  (annotated classes)")
print("=" * 85)
print(f"\n{'Class':<45} {'Proteins':>8} {'Total':>7} {'P/LP':>6} {'B/LB':>6} {'VUS':>7} {'Var/prot':>9}")
print("-" * 85)
for cls, row in pivot.iterrows():
    print(f"{cls:<45} {row['n_proteins']:>8} {row['total_variants']:>7,} "
          f"{row.get('Pathogenic/LP',0):>6,} {row.get('Benign/LB',0):>6,} "
          f"{row.get('VUS',0):>7,} {row['variants_per_protein']:>9.1f}")

pivot.to_csv("Csvs/clinvar_class_enrichment.csv")
print("\nSaved: Csvs/clinvar_class_enrichment.csv\n")

SIG_ORDER  = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]
SIG_COLORS = {
    "Pathogenic/LP"    : "#d62728",
    "VUS"              : "#aec7e8",
    "Benign/LB"        : "#2ca02c",
    "Other/Conflicting": "#c7c7c7",
}

# ── Build normalised table: variants per protein per sig group ─────────────────

# Only classes with ≥2 proteins (removes singleton classes)
norm = pivot[pivot["n_proteins"] >= 2].copy()
for sig in SIG_ORDER:
    if sig in norm.columns:
        norm[f"{sig}_pp"] = norm[sig] / norm["n_proteins"]
    else:
        norm[f"{sig}_pp"] = 0.0

norm["total_pp"] = norm["variants_per_protein"]
norm = norm.sort_values("total_pp", ascending=True)

# Save normalised table
norm_out = norm[["n_proteins", "total_variants", "total_pp"] +
                [f"{s}_pp" for s in SIG_ORDER]].copy()
norm_out.columns = ["n_proteins", "total_variants", "total_per_protein"] + \
                   [f"{s}_per_protein" for s in SIG_ORDER]
norm_out.to_csv("Csvs/clinvar_class_enrichment_normalised.csv")
print("Saved: Csvs/clinvar_class_enrichment_normalised.csv\n")

# ── Plot 1: Normalised stacked bar — variants per protein ─────────────────────

fig, ax = plt.subplots(figsize=(12, 8))
left = np.zeros(len(norm))

for sig in SIG_ORDER:
    col = f"{sig}_pp"
    if col in norm.columns:
        vals = norm[col].values
        ax.barh(norm.index, vals, left=left,
                color=SIG_COLORS[sig], label=sig)
        left += vals

# Annotate total at end of bar
for i, (cls, row) in enumerate(norm.iterrows()):
    ax.text(row["total_pp"] + 0.3, i,
            f"{row['total_pp']:.1f}  (n={int(row['n_proteins'])} proteins)",
            va="center", fontsize=7.5)

ax.set_xlabel("ClinVar variants per protein", fontsize=11)
ax.set_title(
    "ClinVar variant burden per protein by structural class\n"
    "(normalised by number of proteins in each class)",
    fontsize=11, fontweight="bold"
)
ax.legend(title="Clinical significance", bbox_to_anchor=(1, 1), loc="upper left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("clinvar_class_enrichment_normalised.png", dpi=150, bbox_inches="tight")
print("Saved: clinvar_class_enrichment_normalised.png")
plt.show()

# Pathogenic fraction per class
pivot["path_fraction"] = (
    pivot.get("Pathogenic/LP", 0) / 
    pivot["total_variants"].replace(0, np.nan)
).round(3)

pivot["vus_fraction"] = (
    pivot.get("VUS", 0) / 
    pivot["total_variants"].replace(0, np.nan)
).round(3)

print("\nPathogenic fraction by class (n≥5 proteins, sorted):")
path_view = pivot[pivot["n_proteins"] >= 5].sort_values(
    "path_fraction", ascending=False
)
for cls, row in path_view.iterrows():
    print(f"  {cls:<45} "
          f"P/LP: {row['path_fraction']:.1%}  "
          f"VUS: {row['vus_fraction']:.1%}  "
          f"n={int(row['n_proteins'])}")

# Plot 2: Pathogenic fraction per class
fig, ax = plt.subplots(figsize=(10, 8))

path_data = norm.copy()
path_data["path_frac"] = (
    path_data.get("Pathogenic/LP", 0) / 
    path_data["total_variants"].replace(0, np.nan)
)
path_data = path_data.sort_values("path_frac", ascending=True)

colors = ["#d62728" if v >= 0.1 else "#aec7e8" 
          for v in path_data["path_frac"]]

ax.barh(path_data.index, path_data["path_frac"] * 100, color=colors)
ax.axvline(x=path_data["path_frac"].mean() * 100, 
           color="black", linestyle="--", 
           linewidth=1, label="Mean")
ax.set_xlabel("Pathogenic/LP variants (%)", fontsize=11)
ax.set_title(
    "Pathogenic variant fraction by structural class\n"
    "(classes with ≥2 proteins)",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("clinvar_pathogenic_fraction.png", dpi=150, bbox_inches="tight")


# Livesey-Marsh style relative enrichment
# For each class vs all others: Pathogenic/LP vs Benign/LB
print("\nFisher's exact test — relative enrichment (Pathogenic/LP vs Benign/LB):")
total_path = pivot["Pathogenic/LP"].sum()
total_benign = pivot["Benign/LB"].sum()

test_classes = pivot[pivot["n_proteins"] >= 2].copy()
test_classes = test_classes[test_classes["total_variants"] > 0]
n_tests = len(test_classes)

results_path_vs_benign = []
for cls, row in test_classes.iterrows():
    cls_path = row.get("Pathogenic/LP", 0)
    cls_benign = row.get("Benign/LB", 0)
    other_path = total_path - cls_path
    other_benign = total_benign - cls_benign

    # Fisher on 2x2 table: [class vs other] x [pathogenic vs benign]
    oddsratio_pb, pval_pb = scipy.stats.fisher_exact(
        [[cls_path, cls_benign],
         [other_path, other_benign]],
        alternative="greater"
    )
    or_ci, ci_lo, ci_hi, corrected = odds_ratio_ci_95(
        cls_path, other_path, cls_benign, other_benign
    )
    results_path_vs_benign.append(
        (cls, oddsratio_pb, pval_pb, or_ci, ci_lo, ci_hi, corrected,
         cls_path, cls_benign, other_path, other_benign)
    )

print(f"  (Bonferroni correction: n_tests={n_tests})\n")
print(f"  {'Class':<45} {'OR':>6}  {'95% CI':>23}  {'p_raw':>10}  {'p_bonf':>10}")
print(f"  {'-'*110}")
for cls, _, p_raw, or_ci, ci_lo, ci_hi, corrected, *_ in sorted(
    results_path_vs_benign, key=lambda x: x[3], reverse=True
):
    p_bonf = min(p_raw * n_tests, 1.0)
    sig = " *" if p_bonf < 0.05 else ""
    cc = "†" if corrected else ""
    print(
        f"  {cls:<45} {or_ci:>6.2f}{cc}  [{ci_lo:>6.2f}, {ci_hi:>6.2f}]"
        f"  {p_raw:>10.3e}  {p_bonf:>10.3e}{sig}"
    )
print("  † continuity correction (0.5) applied for CI due to zero cell(s)")

# Save Livesey-Marsh enrichment stats for downstream reporting
path_ben_df = pd.DataFrame(results_path_vs_benign, columns=[
    "structural_class",
    "or_fisher_path_vs_benign", "p_raw_path_vs_benign",
    "or_path_vs_benign", "or_ci95_lo", "or_ci95_hi", "ci_continuity_corrected",
    "cls_path", "cls_benign", "other_path", "other_benign",
])
path_ben_df["p_bonf_path_vs_benign"] = (
    path_ben_df["p_raw_path_vs_benign"] * n_tests
).clip(upper=1.0)

enrich_stats = path_ben_df.sort_values("or_path_vs_benign", ascending=False)
enrich_stats.to_csv("Csvs/clinvar_enrichment_stats.csv", index=False)
print("\nSaved: Csvs/clinvar_enrichment_stats.csv")

sig_classes = {
    cls for cls, _, p_raw, *_ in results_path_vs_benign if min(p_raw * n_tests, 1.0) < 0.05
}

# ── Plot 3: Proportional stacked bar (class composition) ─────────────────────

sig_cols = [c for c in ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]
            if c in pivot.columns]

# Classes with ≥2 proteins, sorted by P/LP fraction ascending (for barh)
plot_pivot = pivot[pivot["n_proteins"] >= 2].copy()
pivot_pct = plot_pivot[sig_cols].div(plot_pivot["total_variants"], axis=0) * 100
pivot_pct = pivot_pct.sort_values("Pathogenic/LP", ascending=True)

class_n = plot_pivot["n_proteins"].to_dict()

fig, ax = plt.subplots(figsize=(10, 12))

prop_colors = {
    "Pathogenic/LP"    : "#A32D2D",
    "VUS"              : "#6B9FD4",
    "Benign/LB"        : "#4A7C3F",
    "Other/Conflicting": "#888780",
}

left = np.zeros(len(pivot_pct))
for col in pivot_pct.columns:
    ax.barh(range(len(pivot_pct)), pivot_pct[col], left=left,
            color=prop_colors[col], label=col, height=0.7)
    left += pivot_pct[col].values

# Label P/LP % at right of each bar
for i, (idx, row) in enumerate(pivot_pct.iterrows()):
    plp = row.get("Pathogenic/LP", 0)
    if plp > 0.5:
        ax.text(101, i, f"{plp:.1f}%", va="center", fontsize=8,
                color="#A32D2D", fontweight="bold")

# Y-axis labels with protein count
yticklabels = [f"{idx}  (n={int(class_n.get(idx, 0))})" for idx in pivot_pct.index]
ax.set_yticks(range(len(pivot_pct)))
ax.set_yticklabels(yticklabels, fontsize=8)

# Highlight Bonferroni-significant classes in red
for label, cls in zip(ax.get_yticklabels(), pivot_pct.index):
    if cls in sig_classes:
        label.set_color("#A32D2D")
        label.set_fontweight("bold")

ax.set_xlim(0, 115)
ax.set_xlabel("% of ClinVar variants", fontsize=10)
ax.set_title(
    "Clinical significance breakdown by structural class\n"
    "(classes with ≥2 proteins; red bold = Bonferroni-significant)",
    fontsize=10, fontweight="bold"
)
ax.axvline(x=100, color="black", linewidth=0.5)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("clinvar_class_sig_composition_pct.png", dpi=150, bbox_inches="tight")
print("Saved: clinvar_class_sig_composition_pct.png")
plt.show()

# ── Plot 4: Per-protein P/LP fraction dot plot ───────────────────────────────

# Per-protein counts
per_prot = (
    cv_dbp.groupby("uniprot")["sig_group"]
    .agg(total="size", plp=lambda s: (s == "Pathogenic/LP").sum())
    .reset_index()
)
per_prot = per_prot[per_prot["total"] >= 1].copy()
per_prot["plp_frac"] = per_prot["plp"] / per_prot["total"] * 100

per_prot = per_prot.merge(
    classif[["uniprot_id", "structural_class", "gene"]],
    left_on="uniprot", right_on="uniprot_id", how="left"
)
per_prot = per_prot[~per_prot["structural_class"].isin(
    ["No structural annotation", "⚠ Transposable element / ERV"]
)]

# Sort classes by median P/LP fraction descending; keep only ≥2 proteins
class_sizes   = per_prot.groupby("structural_class")["uniprot"].nunique()
class_medians = (
    per_prot.groupby("structural_class")["plp_frac"]
    .median()
    .loc[class_sizes[class_sizes >= 2].index]
    .sort_values(ascending=False)
)
classes_ordered = list(class_medians.index)

overall_mean = per_prot["plp_frac"].mean()

# Short display names for x-axis
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
    "SMAD-like domain superfamily"                                      : "SMAD",
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
    "Winged helix-like DNA-binding domain superfamily"                  : "WH",
}

from matplotlib.lines import Line2D

# Label outlier dots above this threshold with gene name
LABEL_THRESHOLD = 15.0   # % P/LP fraction

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

    # Label outliers
    for idx, row in subset_df[subset_df["plp_frac"] >= LABEL_THRESHOLD].iterrows():
        gene_label = row.get("gene", row["uniprot"]) or row["uniprot"]
        ax.annotate(
            gene_label,
            xy=(i + jitter_map[idx], row["plp_frac"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=6.5, color="black", zorder=5,
        )

ax.axhline(overall_mean, color="gray", linestyle="--", linewidth=1.2, zorder=2)

# Extend y-axis to accommodate top labels
ax.set_ylim(bottom=-2, top=max(per_prot["plp_frac"].max() + 5, 70))

x_tick_labels = [
    f"{NAME_SHORT.get(c, c[:12])}{'★' if c in sig_classes else ''}"
    for c in classes_ordered
]
ax.set_xticks(range(len(classes_ordered)))
ax.set_xticklabels(x_tick_labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("% Pathogenic/LP variants per protein", fontsize=11)
ax.set_title(
    "Pathogenic/LP fraction per protein · structural class · ClinVar variants\n"
    "dashed line = proteome-wide DBP mean · ★ Bonferroni-significant class (Fisher's exact)",
    fontsize=10, fontweight="bold"
)
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=8, label="Individual protein (Bonferroni-sig. class)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#6baed6",
           markersize=8, label="Individual protein (other class)"),
    Line2D([0], [0], color="black", linewidth=2, label="Class median"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
           label=f"Overall DBP mean ({overall_mean:.1f}%)"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("clinvar_plp_dotplot.png", dpi=150, bbox_inches="tight")
print("Saved: clinvar_plp_dotplot.png")
plt.show()

# ── Plot 5: Variant skew across DBPs ──────────────────────────────────────────

# Per-protein total variant counts (all 529 DBPs with ≥1 variant)
prot_counts = (
    cv_dbp.groupby("uniprot")
    .size()
    .reset_index(name="n_variants")
    .merge(classif[["uniprot_id", "gene"]], left_on="uniprot", right_on="uniprot_id", how="left")
    .sort_values("n_variants", ascending=False)
    .reset_index(drop=True)
)
prot_counts["rank"] = prot_counts.index + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: ranked variant count (log scale) ────────────────────────────────────
ax = axes[0]
ax.plot(prot_counts["rank"], prot_counts["n_variants"],
        color="#4393c3", linewidth=1.2, zorder=2)
ax.fill_between(prot_counts["rank"], prot_counts["n_variants"],
                alpha=0.15, color="#4393c3")

# Label top 15 proteins
TOP_N = 15
for _, row in prot_counts.head(TOP_N).iterrows():
    label = row["gene"] if pd.notna(row["gene"]) and row["gene"] else row["uniprot"]
    ax.annotate(
        label,
        xy=(row["rank"], row["n_variants"]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=7, va="center", color="#d62728",
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.5),
    )

ax.set_yscale("log")
ax.set_xlabel("DBP rank (by variant count)", fontsize=10)
ax.set_ylabel("ClinVar variants (log scale)", fontsize=10)
ax.set_title(
    "Variant count distribution across DBPs\n(ranked, log scale)",
    fontsize=10, fontweight="bold"
)
ax.spines[["top", "right"]].set_visible(False)

# ── Right: cumulative proportion plot ─────────────────────────────────────────
ax = axes[1]
total_variants = prot_counts["n_variants"].sum()
prot_counts["cumulative_pct"] = prot_counts["n_variants"].cumsum() / total_variants * 100
prot_counts["rank_pct"] = prot_counts["rank"] / len(prot_counts) * 100

ax.plot(prot_counts["rank_pct"], prot_counts["cumulative_pct"],
        color="#4393c3", linewidth=1.5, zorder=3)
ax.plot([0, 100], [0, 100], color="gray", linestyle="--",
        linewidth=1, label="Perfect equality", zorder=2)

# Annotate key quantiles
for pct in [10, 25, 50]:
    idx = (prot_counts["rank_pct"] - pct).abs().idxmin()
    cum = prot_counts.loc[idx, "cumulative_pct"]
    ax.annotate(
        f"Top {pct}% proteins\n= {cum:.0f}% of variants",
        xy=(prot_counts.loc[idx, "rank_pct"], cum),
        xytext=(pct + 5, cum - 10),
        fontsize=7.5, color="#333",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
    )

ax.set_xlabel("% of DBPs (ranked by variant count)", fontsize=10)
ax.set_ylabel("Cumulative % of all DBP variants", fontsize=10)
ax.set_title(
    "Cumulative variant distribution across DBPs\n(Lorenz-style)",
    fontsize=10, fontweight="bold"
)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    f"Variant burden skew across {len(prot_counts)} DBPs with ClinVar entries  "
    f"(total = {total_variants:,} variants)",
    fontsize=11, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("clinvar_variant_skew.png", dpi=150, bbox_inches="tight")
print("Saved: clinvar_variant_skew.png")
plt.show()

# Print summary stats
print(f"\nVariant skew summary:")
print(f"  Proteins with ≥1 variant: {len(prot_counts)}")
print(f"  Mean variants/protein:    {prot_counts['n_variants'].mean():.1f}")
print(f"  Median variants/protein:  {prot_counts['n_variants'].median():.0f}")
print(f"  Max (single protein):     {prot_counts['n_variants'].max():,}  "
      f"({prot_counts.iloc[0]['gene'] or prot_counts.iloc[0]['uniprot']})")
for pct in [10, 25]:
    idx = (prot_counts["rank_pct"] - pct).abs().idxmin()
    cum = prot_counts.loc[idx, "cumulative_pct"]
    print(f"  Top {pct}% of proteins account for {cum:.1f}% of all variants")

# Top 20 proteins by variant count
top20 = prot_counts.head(20).merge(
    classif[["uniprot_id", "structural_class"]],
    left_on="uniprot", right_on="uniprot_id", how="left"
)
print(f"\n{'Top 20 DBPs by ClinVar variant count':}")
print(f"{'Rank':>4}  {'Gene':<12} {'UniProt':<10} {'Variants':>8}  {'Structural class'}")
print("-" * 90)
for _, row in top20.iterrows():
    gene = row["gene"] if pd.notna(row["gene"]) and row["gene"] else "—"
    cls  = row["structural_class"] if pd.notna(row.get("structural_class")) else "—"
    print(f"{int(row['rank']):>4}  {gene:<12} {row['uniprot']:<10} {int(row['n_variants']):>8,}  {cls}")

print("\nDone.")

"""