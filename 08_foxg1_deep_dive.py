"""
FOXG1 Deep Dive
===============
1. All ClinVar variants (in-domain and out-of-domain) with residue positions and sig_group
2. AlphaMissense scores for all possible FOXG1 missense substitutions
3. Pathogenic variant clustering within the forkhead domain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, fisher_exact

FOXG1_UNIPROT   = "P55316"
DOMAIN_START    = 181
DOMAIN_END      = 275

SIG_ORDER  = ["Pathogenic/LP", "VUS", "Benign/LB", "Other/Conflicting"]
SIG_COLORS = {
    "Pathogenic/LP"    : "#d62728",
    "VUS"              : "#888888",
    "Benign/LB"        : "#2171b5",
    "Other/Conflicting": "#9ecae1",
}

# ── 1. Load ClinVar variants ──────────────────────────────────────────────────

cv_in = pd.read_csv("Csvs/clinvar_domain_hits.csv", low_memory=False)
cv_in = cv_in[cv_in["uniprot"] == FOXG1_UNIPROT].drop_duplicates("#AlleleID").copy()
cv_in["location"] = "in-domain"

cv_out = pd.read_csv("Csvs/clinvar_out_of_domain_hits.csv", low_memory=False)
cv_out = cv_out[cv_out["uniprot"] == FOXG1_UNIPROT].drop_duplicates("#AlleleID").copy()
cv_out["location"] = "out-of-domain"

# Combine, keep only columns needed
KEEP_COLS = ["#AlleleID", "uniprot", "canonical_residue", "residue_seq",
             "modified_residue", "sig_group", "location", "ReviewStatus"]
cv_all = pd.concat([
    cv_in[[c for c in KEEP_COLS if c in cv_in.columns]],
    cv_out[[c for c in KEEP_COLS if c in cv_out.columns]],
], ignore_index=True)
cv_all["residue_seq"] = pd.to_numeric(cv_all["residue_seq"], errors="coerce")


print(f"In-domain  (forkhead, {DOMAIN_START}–{DOMAIN_END}): {len(cv_in):,}")
print(f"Out-of-domain:                              {len(cv_out):,}")
print(f"Total:                                      {len(cv_all):,}\n")

print("In-domain sig_group breakdown:")
print(cv_in["sig_group"].value_counts().to_string())
print("\nOut-of-domain sig_group breakdown:")
print(cv_out["sig_group"].value_counts().to_string())


# ── 2. Load AlphaMissense scores ──────────────────────────────────────────────

am_raw = pd.read_csv("Csvs/alphamissense_dbp.csv", low_memory=False)
am = am_raw[am_raw["uniprot_id"] == FOXG1_UNIPROT].copy()

parts = am["protein_variant"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
am["ref_aa"]  = parts[0]
am["pos"]     = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
am["alt_aa"]  = parts[2]
am["in_domain"] = (am["pos"] >= DOMAIN_START) & (am["pos"] <= DOMAIN_END)

print(f"\n=== FOXG1 AlphaMissense data ===")
print(f"Total possible substitutions: {len(am):,}")
print(f"In forkhead domain ({DOMAIN_START}–{DOMAIN_END}): {am['in_domain'].sum():,}")
print(f"\nAM class distribution (all):")
print(am["am_class"].value_counts().to_string())
print(f"\nAM class distribution (in-domain only):")
print(am[am["in_domain"]]["am_class"].value_counts().to_string())


# ── 3. Per-position AM score summary (domain only) ───────────────────────────

am_dom = am[am["in_domain"]].copy()
pos_summary = (
    am_dom.groupby("pos")["am_pathogenicity"]
    .agg(median="median", max="max", pct_pathogenic=lambda x: (x >= 0.564).mean() * 100)
    .reset_index()
)

# ClinVar P/LP positions in domain
plp_positions  = cv_in[cv_in["sig_group"] == "Pathogenic/LP"]["residue_seq"].dropna().astype(int)
vus_positions  = cv_in[cv_in["sig_group"] == "VUS"]["residue_seq"].dropna().astype(int)
blb_positions  = cv_in[cv_in["sig_group"] == "Benign/LB"]["residue_seq"].dropna().astype(int)

# Hotspot positions: ≥2 independent P/LP variants at same residue
plp_pos_counts = plp_positions.value_counts()
hotspots = plp_pos_counts[plp_pos_counts >= 2].index.tolist()
print(f"\nHotspot residues (≥2 P/LP ClinVar variants): {sorted(hotspots)}")

print("\nP/LP variant positions (sorted):")
print(sorted(plp_positions.tolist()))
print("\nVUS positions (sorted):")
print(sorted(vus_positions.tolist()))
print("\nB/LB positions (sorted):")
print(sorted(blb_positions.tolist()))

# Residues with NO ClinVar variant at all (any sig_group)
all_observed = set(cv_in["residue_seq"].dropna().astype(int))
all_domain_positions = set(range(DOMAIN_START, DOMAIN_END + 1))
unobserved = sorted(all_domain_positions - all_observed)
print(f"\nResidues with NO ClinVar variant (any sig_group) — {len(unobserved)} of {len(all_domain_positions)}:")
print(unobserved)

# Also print with the ref amino acid from AM data so you can cross-check hydropathy
am_ref = (
    am_dom[["pos", "ref_aa"]]
    .drop_duplicates("pos")
    .set_index("pos")["ref_aa"]
)
print("\nUnobserved residues with identity (cross-check hydrophobic core on PDB):")
for pos in unobserved:
    aa = am_ref.get(pos, "?")
    print(f"  {pos}  {aa}")


# ── 4. AM score at ClinVar variant positions ──────────────────────────────────

aa3_to_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
}
cv_in["ref_aa_1"] = cv_in["canonical_residue"].map(aa3_to_1)
cv_in["alt_aa_1"] = cv_in["modified_residue"].map(aa3_to_1)

cv_in_am = cv_in.merge(
    am[["ref_aa", "pos", "alt_aa", "am_pathogenicity", "am_class"]],
    left_on=["ref_aa_1", "residue_seq", "alt_aa_1"],
    right_on=["ref_aa", "pos", "alt_aa"],
    how="left",
)

matched = cv_in_am["am_pathogenicity"].notna().sum()
print(f"\nIn-domain ClinVar variants matched to AM score: {matched}/{len(cv_in_am)}")

print("\nMedian AM score by sig_group (in-domain ClinVar variants):")
for grp in SIG_ORDER:
    subset = cv_in_am[cv_in_am["sig_group"] == grp]["am_pathogenicity"].dropna()
    if len(subset):
        print(f"  {grp:<22} n={len(subset):>3}  median={subset.median():.3f}")

# Mann-Whitney: P/LP vs B/LB AM scores
plp_am = cv_in_am[cv_in_am["sig_group"] == "Pathogenic/LP"]["am_pathogenicity"].dropna()
blb_am = cv_in_am[cv_in_am["sig_group"] == "Benign/LB"]["am_pathogenicity"].dropna()
if len(plp_am) and len(blb_am):
    stat, p_mw = mannwhitneyu(plp_am, blb_am, alternative="two-sided")
    print(f"\nMann-Whitney P/LP vs B/LB AM scores: U={stat:.0f}, p={p_mw:.3e}")


# ── 5. Are P/LP variants enriched at high-AM-score positions? ─────────────────
# Compare median positional AM score (max across all substitutions) for P/LP vs VUS/B/LB

def max_am_at_pos(positions, pos_summary_df):
    return pos_summary_df[pos_summary_df["pos"].isin(positions)]["max"].values

plp_max_am = max_am_at_pos(plp_positions, pos_summary)
vus_max_am = max_am_at_pos(vus_positions, pos_summary)

if len(plp_max_am) and len(vus_max_am):
    stat_pv, p_pv = mannwhitneyu(plp_max_am, vus_max_am, alternative="greater")
    print(f"\nMann-Whitney (P/LP pos max AM > VUS pos max AM): U={stat_pv:.0f}, p={p_pv:.3e}")
    print(f"  P/LP positions median max-AM: {np.median(plp_max_am):.3f}")
    print(f"  VUS  positions median max-AM: {np.median(vus_max_am):.3f}")


# ── Plots ─────────────────────────────────────────────────────────────────────

# Plot 1: Full protein — variant positions coloured by sig_group
protein_len = cv_all["residue_seq"].max()
fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=False,
                          gridspec_kw={"height_ratios": [1, 1]})

for ax_idx, (loc, df_loc) in enumerate(
    [("in-domain", cv_in), ("out-of-domain", cv_out)]
):
    ax = axes[ax_idx]
    for grp in SIG_ORDER:
        subset = df_loc[df_loc["sig_group"] == grp]
        ax.scatter(
            subset["residue_seq"], [0.5] * len(subset),
            color=SIG_COLORS[grp], s=40, alpha=0.7,
            label=f"{grp} (n={len(subset)})", zorder=3,
        )
    if ax_idx == 0:
        ax.axvspan(DOMAIN_START, DOMAIN_END, alpha=0.08, color="#d62728",
                   label=f"Forkhead domain ({DOMAIN_START}–{DOMAIN_END})")
    ax.set_yticks([])
    ax.set_xlim(1, max(protein_len + 5, DOMAIN_END + 20))
    ax.set_title(f"FOXG1 {loc} variants (n={len(df_loc)})", fontsize=10)
    ax.set_xlabel("Residue position", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.spines[["top", "right", "left"]].set_visible(False)

plt.suptitle("FOXG1 ClinVar variant positions by clinical significance", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("Figures/foxg1_variant_positions.png", dpi=150, bbox_inches="tight")
print("\nSaved Figures/foxg1_variant_positions.png")


# Plot 2: Forkhead domain — AM score landscape + ClinVar variant rug
# Use min AM score per position (most tolerated substitution) to show variation;
# median is ~1.0 across the whole domain and gives a flat, uninformative landscape.
pos_summary["min"] = am_dom.groupby("pos")["am_pathogenicity"].min().values

fig, ax = plt.subplots(figsize=(12, 5))

# Show the range between min and median to give a sense of spread
ax.fill_between(pos_summary["pos"], pos_summary["min"], pos_summary["median"],
                alpha=0.25, color="#888", label="Min–median AM score range")
ax.plot(pos_summary["pos"], pos_summary["median"],
        color="#555", linewidth=1.2, zorder=2, label="Median AM score")
ax.plot(pos_summary["pos"], pos_summary["min"],
        color="#aaa", linewidth=0.8, zorder=2, linestyle="--", label="Min AM score (most tolerated sub.)")

# Rug marks for ClinVar variants
rug_y = {"Pathogenic/LP": -0.06, "VUS": -0.11, "Benign/LB": -0.16}
for grp, y in rug_y.items():
    positions = cv_in[cv_in["sig_group"] == grp]["residue_seq"].dropna()
    ax.scatter(positions, [y] * len(positions),
               marker="|", s=90, color=SIG_COLORS[grp],
               linewidths=1.8, zorder=4, label=f"{grp} (n={len(positions)})", clip_on=False)

# Only annotate the top 5 most recurrent P/LP positions
top_hotspots = plp_pos_counts.nlargest(5)
for pos, n in top_hotspots.items():
    min_val = pos_summary.loc[pos_summary["pos"] == pos, "min"]
    y_arrow = float(min_val.values[0]) if len(min_val) else 0.5
    ax.annotate(f"p.{pos} (n={n})",
                xy=(pos, y_arrow),
                xytext=(0, -28), textcoords="offset points",
                fontsize=7.5, color="#d62728", ha="center",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8),
                clip_on=False)

ax.axhline(0.564, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.6)
ax.axhline(0.340, color="#2171b5", linestyle="--", linewidth=0.9, alpha=0.6)
ax.text(DOMAIN_END + 0.5, 0.572, "Pathogenic\nthreshold", fontsize=7, color="#d62728", va="bottom")
ax.text(DOMAIN_END + 0.5, 0.348, "Benign\nthreshold",     fontsize=7, color="#2171b5", va="bottom")

ax.set_xlim(DOMAIN_START - 1, DOMAIN_END + 8)
ax.set_ylim(-0.22, 1.05)
ax.set_xlabel("Residue position within forkhead domain", fontsize=10)
ax.set_ylabel("AlphaMissense pathogenicity score", fontsize=9)
ax.set_title(
    f"FOXG1 forkhead domain ({DOMAIN_START}–{DOMAIN_END}): AM score landscape and ClinVar variant positions",
    fontsize=10, fontweight="bold",
)
ax.legend(fontsize=8, loc="lower left", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/foxg1_domain_am_landscape.png", dpi=150, bbox_inches="tight")
print("Saved Figures/foxg1_domain_am_landscape.png")


# Plot 3: AM score distribution for ClinVar variants by sig_group (boxplot)
import seaborn as sns

matched_with_am = cv_in_am[
    cv_in_am["sig_group"].isin(["Pathogenic/LP", "VUS", "Benign/LB"]) &
    cv_in_am["am_pathogenicity"].notna()
].copy()

if len(matched_with_am):
    fig, ax = plt.subplots(figsize=(6, 5))
    order = ["Pathogenic/LP", "VUS", "Benign/LB"]
    palette = {g: SIG_COLORS[g] for g in order}
    sns.boxplot(data=matched_with_am, x="sig_group", y="am_pathogenicity",
                order=order, palette=palette, hue="sig_group", legend=False,
                showfliers=False, width=0.5, ax=ax,
                medianprops=dict(color="black", linewidth=2),
                boxprops=dict(alpha=0.75))
    sns.stripplot(data=matched_with_am, x="sig_group", y="am_pathogenicity",
                  order=order, palette=palette, size=3, jitter=0.2, alpha=0.4, ax=ax)
    ax.axhline(0.564, color="#555", linestyle="--", linewidth=1)
    ax.axhline(0.340, color="#555", linestyle=":",  linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("AlphaMissense pathogenicity score")
    ax.set_ylim(0, 1.05)
    ax.set_title("FOXG1 in-domain variants: AM score by ClinVar sig group", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("Figures/foxg1_am_by_siggroup.png", dpi=150, bbox_inches="tight")
    print("Saved Figures/foxg1_am_by_siggroup.png")


# ── Save combined table ───────────────────────────────────────────────────────

cv_in_am_out = cv_in_am[[
    "#AlleleID", "residue_seq", "ref_aa_1", "alt_aa_1", "sig_group",
    "am_pathogenicity", "am_class", "ReviewStatus",
]].rename(columns={"residue_seq": "pos", "ref_aa_1": "ref_aa", "alt_aa_1": "alt_aa"})
cv_in_am_out["location"] = "in-domain"

cv_out_export = cv_out[["#AlleleID", "residue_seq", "sig_group"]].copy()
cv_out_export["location"] = "out-of-domain"

cv_in_am_out.to_csv("Csvs/foxg1_indomain_variants_with_am.csv", index=False)
cv_out_export.to_csv("Csvs/foxg1_outdomain_variants.csv", index=False)
print("Saved Csvs/foxg1_indomain_variants_with_am.csv")
print("Saved Csvs/foxg1_outdomain_variants.csv")
