"""
DBP Variant Landscape
=====================
1. Clinical significance breakdown: DBPs vs rest of ClinVar (Fisher's exact)
2. Variant burden: top-20 DBPs by raw count; variant skew plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

CLINVAR_CSV = "Data/clinvar_variants_28_08_2025.csv"
CLASSIF_CSV = "Csvs/protein_classifications_hierarchy.csv"

# ── Load classifications ───────────────────────────────────────────────────────

print("Loading protein classifications...")
classif = pd.read_csv(CLASSIF_CSV)
classif = classif[~classif["is_te"] & ~classif["is_noncanonical"]]
dbp_ids = set(classif["uniprot_id"])
print(f"  {len(classif)} proteins (TEs and non-canonical excluded)\n")

# ── Load ClinVar and filter ────────────────────────────────────────────────────

print("Loading ClinVar...")
cv = pd.read_csv(CLINVAR_CSV, low_memory=False)
cv = cv[cv["OriginSimple"] == "germline"].copy()
cv = cv[~((cv["canonical_residue"] == "Met") & (cv["residue_seq"] == 1))].copy()  # remove start-lost
cv = cv.dropna(subset=["uniprot", "residue_seq"]).copy()
print(f"  {len(cv):,} total variants (germline, start-lost removed)")

cv_dbp = cv[cv["uniprot"].isin(dbp_ids)].copy()
print(f"  {len(cv_dbp):,} variants mapping to {cv_dbp['uniprot'].nunique()} DBPs\n")

# ── Recode clinical significance ───────────────────────────────────────────────

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
CONFLICTING_TERMS = {
    "Conflicting classifications of pathogenicity",
    "Conflicting classifications of pathogenicity; other",
    "Conflicting classifications of pathogenicity; risk factor",
}

def recode(sig):
    if sig in PATHOGENIC_TERMS: return "Pathogenic/LP"
    if sig in BENIGN_TERMS:     return "Benign/LB"
    if sig in VUS_TERMS:        return "VUS"
    if sig in CONFLICTING_TERMS: return "Other/Conflicting"
    return "Other/Conflicting"

def odds_ratio_ci_95(a, b, c, d, continuity=0.5):
    """95% CI via log(OR) ± 1.96·SE; Haldane-Anscombe correction if any cell is zero."""
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + continuity, b + continuity, c + continuity, d + continuity
    or_hat = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z  = scipy.stats.norm.ppf(0.975)
    lo = np.exp(np.log(or_hat) - z * se)
    hi = np.exp(np.log(or_hat) + z * se)
    return or_hat, lo, hi, corrected

cv_dbp["sig_group"]  = cv_dbp["ClinicalSignificance"].apply(recode)
cv["sig_group_all"]  = cv["ClinicalSignificance"].apply(recode)

# Deduplicate by AlleleID (82 variants appear on two DBP proteins in the raw data)
cv_uniq     = cv.drop_duplicates("#AlleleID")
cv_dbp_uniq = cv_dbp.drop_duplicates("#AlleleID")

# ── Clinical significance breakdown ───────────────────────────────────────────

print("Clinical significance breakdown (ALL ClinVar variants):")
total_cv = len(cv_uniq)
for grp, n in cv_uniq["sig_group_all"].value_counts().items():
    print(f"  {grp:<20} {n:>8,}  ({100 * n / total_cv:.1f}%)")
print()

print("Clinical significance breakdown (DBP variants only):")
for grp, n in cv_dbp_uniq["sig_group"].value_counts().items():
    print(f"  {grp:<20} {n:>8,}  ({100 * n / len(cv_dbp_uniq):.1f}%)")
print()

# ── DBP vs non-DBP enrichment (Fisher's exact) ────────────────────────────────

SIG_GROUPS = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]

n_dbp      = len(cv_dbp_uniq)
n_non_dbp  = len(cv_uniq) - n_dbp
dbp_counts = cv_dbp_uniq["sig_group"].value_counts()
all_counts = cv_uniq["sig_group_all"].value_counts()
non_dbp_counts = (all_counts - dbp_counts).fillna(all_counts)

n_tests = len(SIG_GROUPS)
print("DBP vs non-DBP enrichment (Fisher's exact, Bonferroni corrected):")
print(f"  Note: ascertainment bias likely inflates enrichment for well-studied genes.")
print(f"  Bonferroni correction: n_tests={n_tests}")
print(f"  {'Sig group':<20} {'DBP n':>8} {'DBP%':>6}  {'nonDBP n':>9} {'nonDBP%':>8}  {'OR':>6}  {'95% CI':>18}  {'p_raw':>10}  {'p_bonf':>10}")
print(f"  {'-'*115}")

rows = []
for sig in SIG_GROUPS:
    a = int(dbp_counts.get(sig, 0))
    b = int(non_dbp_counts.get(sig, 0))
    c = n_dbp - a
    d = n_non_dbp - b

    _, p_raw = scipy.stats.fisher_exact([[a, b], [c, d]])
    or_hat, ci_lo, ci_hi, corrected = odds_ratio_ci_95(a, c, b, d)
    p_bonf = min(p_raw * n_tests, 1.0)

    pct_dbp    = 100 * a / n_dbp     if n_dbp     else 0
    pct_nondbp = 100 * b / n_non_dbp if n_non_dbp else 0
    cc    = "†" if corrected else ""
    stars = " ***" if p_bonf < 0.001 else (" **" if p_bonf < 0.01 else (" *" if p_bonf < 0.05 else ""))
    print(f"  {sig:<20} {a:>8,} {pct_dbp:>6.1f}%  {b:>9,} {pct_nondbp:>7.1f}%  "
          f"{or_hat:>6.2f}{cc}  [{ci_lo:>6.2f}, {ci_hi:>6.2f}]  {p_raw:>10.3e}  {p_bonf:>10.3e}{stars}")
    rows.append({
        "sig_group": sig, "n_dbp": a, "n_non_dbp": b,
        "pct_dbp": round(pct_dbp, 2), "pct_non_dbp": round(pct_nondbp, 2),
        "odds_ratio": round(or_hat, 3), "ci95_lo": round(ci_lo, 3), "ci95_hi": round(ci_hi, 3),
        "p_raw": p_raw, "p_bonf": p_bonf, "continuity_corrected": corrected,
    })

print("  † Haldane-Anscombe continuity correction applied\n")
pd.DataFrame(rows).to_csv("Csvs/clinvar_dbp_sig_enrichment.csv", index=False)
print("Saved: Csvs/clinvar_dbp_sig_enrichment.csv\n")

# ── Per-protein variant counts ─────────────────────────────────────────────────

prot_counts = (
    cv_dbp.groupby("uniprot")
    .size()
    .reset_index(name="n_variants")
    .merge(classif[["uniprot_id", "gene", "structural_class"]],
           left_on="uniprot", right_on="uniprot_id", how="left")
    .sort_values("n_variants", ascending=False)
    .reset_index(drop=True)
)
prot_counts["rank"] = prot_counts.index + 1

# Top-20 by raw count
print("Top 20 DBPs by ClinVar variant count:")
print(f"{'Rank':>4}  {'Gene':<12} {'UniProt':<10} {'Variants':>8}  Structural class")
print("-" * 90)
for _, row in prot_counts.head(20).iterrows():
    gene = row["gene"] if pd.notna(row["gene"]) and row["gene"] else "—"
    cls  = row["structural_class"] if pd.notna(row.get("structural_class")) else "—"
    print(f"{int(row['rank']):>4}  {gene:<12} {row['uniprot']:<10} {int(row['n_variants']):>8,}  {cls}")

prot_counts.head(20)[["uniprot", "gene", "structural_class", "n_variants"]].to_csv(
    "Csvs/top20_variants_raw.csv", index=False
)
print("\nSaved: Csvs/top20_variants_raw.csv\n")

# ── Per-protein sig_group breakdown ───────────────────────────────────────────

sig_pivot = (
    cv_dbp_uniq.groupby(["uniprot", "sig_group"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=SIG_GROUPS, fill_value=0)
)
sig_pivot["total"] = sig_pivot.sum(axis=1)
for col in SIG_GROUPS:
    sig_pivot[f"pct_{col.lower().replace('/', '_').replace(' ', '_')}"] = (
        100 * sig_pivot[col] / sig_pivot["total"]
    )

prot_sig = (
    sig_pivot.reset_index()
    .merge(classif[["uniprot_id", "gene", "structural_class"]],
           left_on="uniprot", right_on="uniprot_id", how="left")
    .drop(columns="uniprot_id")
    .sort_values("VUS", ascending=False)
    .reset_index(drop=True)
)

print("Top 20 DBPs by VUS count:")
print(f"{'Gene':<12} {'UniProt':<10} {'VUS':>6}  {'%VUS':>6}  {'P/LP':>6}  {'B/LB':>6}  {'Total':>7}  Structural class")
print("-" * 100)
for _, row in prot_sig.head(20).iterrows():
    gene = row["gene"] if pd.notna(row.get("gene")) and row["gene"] else "—"
    cls  = row["structural_class"] if pd.notna(row.get("structural_class")) else "—"
    print(f"{gene:<12} {row['uniprot']:<10} {int(row['VUS']):>6,}  "
          f"{row['pct_vus']:>5.1f}%  {int(row['Pathogenic/LP']):>6,}  "
          f"{int(row['Benign/LB']):>6,}  {int(row['total']):>7,}  {cls}")

vus_counts = sig_pivot["VUS"]
print(f"\nVUS burden summary across {len(sig_pivot)} DBPs with variants:")
print(f"  Proteins with ≥1 VUS:      {(vus_counts > 0).sum()}")
print(f"  Median VUS/protein:        {vus_counts.median():.0f}")
print(f"  Top 10% of proteins account for "
      f"{100 * vus_counts.nlargest(max(1, len(vus_counts)//10)).sum() / vus_counts.sum():.1f}% of all VUS")

prot_sig.to_csv("Csvs/per_protein_sig_breakdown.csv", index=False)
print("Saved: Csvs/per_protein_sig_breakdown.csv\n")

# ── Variant skew plots ─────────────────────────────────────────────────────────

total_variants = prot_counts["n_variants"].sum()
prot_counts["cumulative_pct"] = prot_counts["n_variants"].cumsum() / total_variants * 100
prot_counts["rank_pct"]       = prot_counts["rank"] / len(prot_counts) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: ranked variant count (log scale)
ax = axes[0]
ax.plot(prot_counts["rank"], prot_counts["n_variants"],
        color="#4393c3", linewidth=1.2, zorder=2)
ax.fill_between(prot_counts["rank"], prot_counts["n_variants"],
                alpha=0.15, color="#4393c3")
for _, row in prot_counts.head(15).iterrows():
    label = row["gene"] if pd.notna(row["gene"]) and row["gene"] else row["uniprot"]
    ax.annotate(label, xy=(row["rank"], row["n_variants"]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7, va="center", color="#d62728",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.5))
ax.set_yscale("log")
ax.set_xlabel("DBP rank (by variant count)", fontsize=10)
ax.set_ylabel("ClinVar variants (log scale)", fontsize=10)
ax.set_title("Variant count distribution across DBPs\n(ranked, log scale)",
             fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# Right: cumulative proportion (Lorenz-style)
ax = axes[1]
ax.plot(prot_counts["rank_pct"], prot_counts["cumulative_pct"],
        color="#4393c3", linewidth=1.5, zorder=3)
ax.plot([0, 100], [0, 100], color="gray", linestyle="--",
        linewidth=1, label="Perfect equality", zorder=2)
for pct in [10, 25, 50]:
    idx = (prot_counts["rank_pct"] - pct).abs().idxmin()
    cum = prot_counts.loc[idx, "cumulative_pct"]
    ax.annotate(f"Top {pct}% proteins\n= {cum:.0f}% of variants",
                xy=(prot_counts.loc[idx, "rank_pct"], cum),
                xytext=(pct + 5, cum - 10), fontsize=7.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
ax.set_xlabel("% of DBPs (ranked by variant count)", fontsize=10)
ax.set_ylabel("Cumulative % of all DBP variants", fontsize=10)
ax.set_title("Cumulative variant distribution across DBPs\n(Lorenz-style)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    f"Variant burden skew across {len(prot_counts)} DBPs with ClinVar entries  "
    f"(total = {total_variants:,} variants)",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("Figures/clinvar_variant_skew.png", dpi=150, bbox_inches="tight")
print("Saved: Figures/clinvar_variant_skew.png")
plt.show()

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

print("\nDone.")
