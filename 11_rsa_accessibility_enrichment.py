"""
RSA distribution: ClinVar P/LP vs B/LB.
Restricted to in-domain residues with pLDDT >= 70.
Reports median RSA per group and Mann-Whitney U test.
Produces violin + KDE overlay figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, gaussian_kde, fisher_exact

MERGED_CSV = "Csvs/clinvar_am_m3d_dssp_merged.csv"
PLDDT_THRESHOLD = 70
FIGURE_PATH = "Figures/rsa_distribution_plp_vs_blb.png"

COLOURS = {"Pathogenic/LP": "#d62728", "Benign/LB": "#1f77b4"}
LABELS  = {"Pathogenic/LP": "P/LP",    "Benign/LB":  "B/LB"}

MECH_COLS = [
    "clash",
    "buried_H_bond_breakage",
    "buried_hydrophilic_introduced",
    "buried_charge_introduced",
    "buried_charge_switch",
    "buried_salt_bridge_breakage",
    "buried_exposed_switch",
    "secondary_structure_altered",
    "cavity_altered",
    "gly_in_a_bend",
    "buried_Pro_introduced",
    "disulphide_breakage",
    "disallowed_phi_psi",
    "buried_charge_replaced",
    "buried_Gly_replaced",
    "cis_pro_replaced",
]


def load_analysis_set(path):
    df = pd.read_csv(path)
    return df[
        df["in_domain"]
        & (df["plddt"] >= PLDDT_THRESHOLD)
        & df["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])
    ].copy()


def zone_counts(rsa_arr):
    buried = (rsa_arr < 0.25).sum()
    intermediate = ((rsa_arr >= 0.25) & (rsa_arr < 0.75)).sum()
    exposed = (rsa_arr >= 0.75).sum()
    return buried, intermediate, exposed


def report_stats(plp_rsa, blb_rsa):
    stat, p = mannwhitneyu(plp_rsa, blb_rsa, alternative="two-sided")
    print("=" * 55)
    print(f"RSA distribution  (in-domain, pLDDT >= {PLDDT_THRESHOLD})")
    print("=" * 55)
    print(f"  P/LP  n={len(plp_rsa):4d}  median RSA = {np.median(plp_rsa):.3f}")
    print(f"  B/LB  n={len(blb_rsa):4d}  median RSA = {np.median(blb_rsa):.3f}")
    print(f"  Mann-Whitney U = {stat:.0f},  p = {p:.4g}")
    print("=" * 55)

    print()
    print(f"{'':6s} {'Buried (<0.25)':>16} {'Intermediate (0.25–0.75)':>24} {'Exposed (>0.75)':>16}")
    print("-" * 65)
    for label, arr in [("P/LP", plp_rsa), ("B/LB", blb_rsa)]:
        b, i, e = zone_counts(arr)
        n = len(arr)
        print(f"  {label:<4s} {b:>6d} ({100*b/n:4.1f}%)  {i:>6d} ({100*i/n:4.1f}%)          {e:>6d} ({100*e/n:4.1f}%)")
    print("-" * 65)
    print()

    return stat, p


def plot_rsa_distribution(plp_rsa, blb_rsa, stat, p, path):
    fig, ax2 = plt.subplots(figsize=(7, 5))

    groups = [plp_rsa, blb_rsa]
    group_keys = ["Pathogenic/LP", "Benign/LB"]

    rsa_grid = np.linspace(0, 1, 300)

    # Zone background shading (drawn first so curves sit on top)
    zone_shade = "#b0b0b0"
    ax2.axvspan(0,    0.25, alpha=0.10, color=zone_shade, zorder=0)
    ax2.axvspan(0.25, 0.75, alpha=0.04, color=zone_shade, zorder=0)
    ax2.axvspan(0.75, 1.0,  alpha=0.10, color=zone_shade, zorder=0)

    # Solid grey lines mark zone boundaries (solid = threshold; dashed = median)
    for x_bound in (0.25, 0.75):
        ax2.axvline(x_bound, color="#888888", linestyle="-", linewidth=0.9,
                    alpha=0.7, zorder=1)

    for key, arr in zip(group_keys, groups):
        kde = gaussian_kde(arr, bw_method="scott")
        density = kde(rsa_grid)
        ax2.fill_between(rsa_grid, density, alpha=0.20, color=COLOURS[key], zorder=1)
        ax2.plot(rsa_grid, density, color=COLOURS[key], linewidth=2.2, zorder=2,
                 label=f"{LABELS[key]}  (n={len(arr)})")

    # Zone labels: dark, bold, just above the x-axis tick line
    for x_mid, label in [(0.125, "Buried"), (0.50, "Intermediate"), (0.875, "Exposed")]:
        ax2.text(x_mid, 0.04, label, ha="center", va="bottom",
                 fontsize=8.5, color="#333333", fontweight="bold",
                 transform=ax2.transAxes)

    ax2.set_xlabel("RSA", fontsize=10, labelpad=14)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=9, frameon=False, loc="upper right")

    # Shared title with test result
    p_str = f"p = {p:.3g}" if p >= 0.001 else f"p = {p:.2e}"
    fig.suptitle(
        f"RSA distribution: ClinVar P/LP vs B/LB  (in-domain, pLDDT ≥ {PLDDT_THRESHOLD})\n"
        f"Mann-Whitney U = {stat:.0f},  {p_str}",
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")



def load_m3d_set(path):
    """In-domain, pLDDT>=70, M3D-matched variants (all clinical groups)."""
    df = pd.read_csv(path)
    return df[
        df["in_domain"]
        & (df["plddt"] >= PLDDT_THRESHOLD)
        & df["damaging"].notna()
        & df["rsa"].notna()
    ].copy()


def m3d_rsa_fisher(df):
    """2×2 Fisher's: M3D damaging/neutral × buried (RSA<0.25) / exposed (RSA>=0.75).
    Intermediate residues (0.25–0.75) are excluded for a clean contrast."""
    df = df.copy()
    df = df[df["rsa"].notna()].copy()
    df = df[(df["rsa"] < 0.25) | (df["rsa"] >= 0.75)].copy()
    df["is_damaging"] = df["damaging"] == "Y"
    df["is_buried"]   = df["rsa"] < 0.25

    a = int(( df["is_damaging"] &  df["is_buried"]).sum())   # damaging + buried
    b = int(( df["is_damaging"] & ~df["is_buried"]).sum())   # damaging + exposed
    c = int((~df["is_damaging"] &  df["is_buried"]).sum())   # neutral  + buried
    d = int((~df["is_damaging"] & ~df["is_buried"]).sum())   # neutral  + exposed

    or_fe, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    # 95% CI via Woolf log method (Haldane correction if any cell = 0)
    a_, b_, c_, d_ = (a+0.5, b+0.5, c+0.5, d+0.5) if min(a,b,c,d)==0 else (a,b,c,d)
    or_cc  = (a_ * d_) / (b_ * c_)
    se     = np.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
    ci_lo  = np.exp(np.log(or_cc) - 1.96 * se)
    ci_hi  = np.exp(np.log(or_cc) + 1.96 * se)

    p_str = f"{p:.3g}" if p >= 0.001 else f"{p:.2e}"
    print()
    print("=" * 60)
    print(f"M3D call × RSA zone  (in-domain, pLDDT ≥ {PLDDT_THRESHOLD})")
    print(f"Intermediate residues (0.25–0.75) excluded; buried vs exposed only")
    print("=" * 60)
    print(f"  {'':20s} {'Buried':>10} {'Exposed':>10} {'Total':>8}")
    print(f"  {'Damaging':20s} {a:>10} {b:>10} {a+b:>8}")
    print(f"  {'Neutral':20s} {c:>10} {d:>10} {c+d:>8}")
    print(f"  {'Total':20s} {a+c:>10} {b+d:>10} {a+b+c+d:>8}")
    print(f"\n  OR = {or_fe:.2f}  (95% CI: {ci_lo:.2f}–{ci_hi:.2f})")
    print(f"  Fisher's p = {p_str}")
    print(f"  OR > 1 → damaging calls enriched at buried positions")
    print("=" * 60)


def category_buried_fractions(df):
    """For each M3D mechanism, report buried% among variants where it fires."""
    df = df.copy()
    df["is_buried"] = df["rsa"] < 0.25

    available = [c for c in MECH_COLS if c in df.columns]
    missing   = [c for c in MECH_COLS if c not in df.columns]
    rows = []
    for col in available:
        fired = df[df[col] == "Y"]
        n = len(fired)
        n_buried = int(fired["is_buried"].sum())
        pct = 100 * n_buried / n if n > 0 else float("nan")
        rows.append((col, n, n_buried, pct))
    rows.sort(key=lambda x: (x[1] == 0, -x[3] if x[1] > 0 else 0))

    print()
    print("=" * 65)
    print("M3D category: buried% among variants where category fires")
    print(f"  (in-domain, pLDDT ≥ {PLDDT_THRESHOLD}, buried = RSA < 0.25)")
    print("=" * 65)
    print(f"  {'Category':<40} {'n':>6}  {'n_buried':>9}  {'%buried':>8}")
    print("-" * 65)
    for col, n, nb, pct in rows:
        pct_str = f"{pct:>7.1f}%" if n > 0 else "     N/A"
        print(f"  {col:<40} {n:>6}  {nb:>9}  {pct_str}")
    if missing:
        print()
        print(f"  Not in data: {', '.join(missing)}")
    print("-" * 65)
    overall_buried = 100 * df["is_buried"].mean()
    print(f"  {'Background (all M3D-matched)':<40} {len(df):>6}  "
          f"{int(df['is_buried'].sum()):>9}  {overall_buried:>7.1f}%")
    print("=" * 65)


def m3d_damaging_exposed_clinvar(df):
    """Report ClinVar classifications of M3D damaging variants at exposed residues (RSA >= 0.75)."""
    exposed_damaging = df[(df["damaging"] == "Y") & (df["rsa"] >= 0.75)].copy()

    n = len(exposed_damaging)
    print()
    print("=" * 65)
    print(f"M3D damaging + exposed residues (RSA >= 0.75)  —  ClinVar classifications")
    print(f"  (in-domain, pLDDT ≥ {PLDDT_THRESHOLD})")
    print("=" * 65)
    print(f"  Total: {n} variants\n")

    # Summary by sig_group
    group_counts = exposed_damaging["sig_group"].value_counts()
    print("  ClinVar group summary:")
    for grp, cnt in group_counts.items():
        print(f"    {grp:<30} {cnt:>4}  ({100*cnt/n:.1f}%)")
    print()

    # Full ClinicalSignificance breakdown
    sig_counts = exposed_damaging["ClinicalSignificance"].value_counts()
    print("  ClinicalSignificance breakdown:")
    for sig, cnt in sig_counts.items():
        print(f"    {sig:<45} {cnt:>4}  ({100*cnt/n:.1f}%)")
    print()

    # Per-variant listing
    active_mechs = [c for c in MECH_COLS if c in exposed_damaging.columns]
    print("  Per-variant detail:")
    print(f"  {'Variant':<25} {'RSA':>6}  {'sig_group':<30} {'ClinicalSignificance':<48} M3D mechanisms")
    print("  " + "-" * 150)
    for _, row in exposed_damaging.sort_values("rsa", ascending=False).iterrows():
        variant  = row.get("protein_variant", "?")
        rsa_val  = row["rsa"]
        sig      = row.get("ClinicalSignificance", "?")
        grp      = row.get("sig_group", "?")
        mechs    = [c for c in active_mechs if row.get(c) == "Y"]
        mech_str = ", ".join(mechs) if mechs else "—"
        print(f"  {str(variant):<25} {rsa_val:>6.3f}  {str(grp):<30} {str(sig):<48} {mech_str}")
    print("=" * 65)


def secondary_structure_chisq(df):
    """Chi-squared test: secondary structure category × P/LP vs B/LB."""
    from scipy.stats import chi2_contingency

    SS_MAP = {
        "H": "Helix", "G": "Helix", "I": "Helix",
        "E": "Strand", "B": "Strand",
        "S": "Loop/coil", "T": "Loop/coil", "C": "Loop/coil", "-": "Loop/coil",
    }

    df = df.copy()
    df["ss_category"] = df["ss"].map(SS_MAP)
    df = df[df["ss_category"].notna()]

    table = (
        df.groupby(["ss_category", "sig_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(["Helix", "Strand", "Loop/coil"])
    )
    for col in ["Pathogenic/LP", "Benign/LB"]:
        if col not in table.columns:
            table[col] = 0
    table = table[["Pathogenic/LP", "Benign/LB"]]

    chi2, p, dof, expected = chi2_contingency(table.values)

    print()
    print("=" * 65)
    print("Secondary structure × ClinVar classification  (in-domain, pLDDT ≥ 70)")
    print("=" * 65)
    print(f"  {'Category':<18} {'P/LP':>8} {'B/LB':>8}")
    print("-" * 37)
    for cat in ["Helix", "Strand", "Loop/coil"]:
        plp = int(table.loc[cat, "Pathogenic/LP"])
        blb = int(table.loc[cat, "Benign/LB"])
        print(f"  {cat:<18} {plp:>8} {blb:>8}")
    print("-" * 37)
    p_str = f"{p:.3g}" if p >= 0.001 else f"{p:.2e}"
    print(f"\n  χ² = {chi2:.3f},  df = {dof},  p = {p_str}")
    print("=" * 65)


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

EXCLUDE = {
    "No structural annotation",
    "⚠ Transposable element / ERV",
    "Fibronectin type III superfamily",
}


def plot_rsa_by_class(df, min_n=75, n_bootstrap=2000,
                      out_path="Figures/rsa_by_structural_class.png"):
    """Median RSA of P/LP variants for classes with n >= min_n, with bootstrap 95% CI
    and Kruskal-Wallis test across classes."""
    from scipy.stats import kruskal

    ZONE_COLOURS = {"Buried": "#9e7fbd", "Intermediate": "#555555", "Exposed": "#e8a87c"}

    def zone(rsa):
        if rsa < 0.25:
            return "Buried"
        if rsa < 0.75:
            return "Intermediate"
        return "Exposed"

    plp = df[
        (df["sig_group"] == "Pathogenic/LP")
        & df["rsa"].notna()
        & df["structural_class"].notna()
        & ~df["structural_class"].isin(EXCLUDE)
    ].copy()

    counts = plp.groupby("structural_class")["rsa"].count().sort_values(ascending=False)
    top_classes = counts[counts >= min_n].index.tolist()

    rng = np.random.default_rng(42)
    records = []
    class_arrays = []
    for cls in top_classes:
        vals = plp.loc[plp["structural_class"] == cls, "rsa"].values
        class_arrays.append(vals)
        med = np.median(vals)
        boots = np.array([np.median(rng.choice(vals, len(vals), replace=True))
                          for _ in range(n_bootstrap)])
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        records.append({
            "class": cls,
            "short": NAME_SHORT.get(cls, cls[:30]),
            "n": len(vals),
            "median_rsa": med,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "zone": zone(med),
        })

    records.sort(key=lambda r: r["median_rsa"])
    class_arrays_sorted = [class_arrays[top_classes.index(r["class"])] for r in records]

    kw_stat, kw_p = kruskal(*class_arrays_sorted)
    kw_p_str = f"{kw_p:.2e}" if kw_p < 0.001 else f"{kw_p:.3f}"
    print(f"\nKruskal-Wallis across classes (n >= {min_n}): H = {kw_stat:.2f}, p = {kw_p_str}")

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(7.5, max(4, len(records) * 0.75)))
    y = range(len(records))

    # Print median RSA table to console
    print()
    print("=" * 62)
    print("Median RSA of P/LP variants per structural class")
    print("=" * 62)
    print(f"  {'Class':<30} {'n':>5}  {'Median RSA':>10}  {'95% CI':>18}  Zone")
    print("-" * 62)
    for rec in records:
        print(f"  {rec['short']:<30} {rec['n']:>5}  {rec['median_rsa']:>10.3f}  "
              f"[{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]  {rec['zone']}")
    print("=" * 62)

    from matplotlib.transforms import blended_transform_factory
    xdata_yaxes = blended_transform_factory(ax.transData, ax.transAxes)

    # Zone shading
    ax.axvspan(0,    0.25, color=ZONE_COLOURS["Buried"],       alpha=0.10, zorder=0)
    ax.axvspan(0.25, 0.75, color=ZONE_COLOURS["Intermediate"], alpha=0.07, zorder=0)
    ax.axvspan(0.75, 1.0,  color=ZONE_COLOURS["Exposed"],      alpha=0.10, zorder=0)

    # Threshold lines only — no RSA= labels
    for x_thresh in (0.25, 0.75):
        ax.axvline(x_thresh, color="black", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)

    # Zone labels below x-axis
    for x_mid, label in [(0.125, "Buried"), (0.50, "Intermediate"), (0.875, "Exposed")]:
        ax.text(x_mid, -0.10, label, ha="center", va="top", fontsize=8.5,
                color=ZONE_COLOURS[label], fontweight="bold",
                transform=xdata_yaxes)

    # CI bars and points coloured by zone; median value annotated to the right of CI bar
    for i, rec in enumerate(records):
        col = ZONE_COLOURS[rec["zone"]]
        ax.plot(
            [rec["ci_lo"], rec["ci_hi"]], [i, i],
            color=col, linewidth=2, solid_capstyle="round", zorder=2, alpha=0.7,
        )
        ax.scatter(rec["median_rsa"], i, color=col, s=60, zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{r['short']}  (n={r['n']:,})" for r in records], fontsize=9
    )
    ax.set_ylabel("Structural class", fontsize=10)
    ax.set_xlabel("Median RSA of P/LP variants (in-domain, pLDDT ≥ 70)", fontsize=10, labelpad=22)
    ax.set_xlim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    ax.text(0.99, 0.02,
            f"Kruskal-Wallis H = {kw_stat:.2f}\np = {kw_p_str}  (n = {len(records)} classes)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot_rsa_zone_bars(plp_rsa, blb_rsa, mw_p=None,
                       out_path="Figures/rsa_zone_stacked_bar.png"):
    """Stacked bar: % Buried / Intermediate / Exposed for P/LP vs B/LB."""
    plt.rcParams["font.family"] = "Arial"
    zone_colours = {"Buried": "#9e7fbd", "Intermediate": "#aaaaaa", "Exposed": "#e8a87c"}

    def fracs(arr):
        n = len(arr)
        return (
            100 * (arr < 0.25).sum() / n,
            100 * ((arr >= 0.25) & (arr < 0.75)).sum() / n,
            100 * (arr >= 0.75).sum() / n,
        )

    groups   = ["P/LP", "B/LB"]
    arrays   = [plp_rsa, blb_rsa]
    zones    = ["Buried", "Intermediate", "Exposed"]
    data     = [fracs(a) for a in arrays]

    fig, ax = plt.subplots(figsize=(4, 4.5))
    bottoms = [0.0, 0.0]

    for zi, zone in enumerate(zones):
        vals = [data[gi][zi] for gi in range(len(groups))]
        bars = ax.bar(groups, vals, bottom=bottoms, color=zone_colours[zone],
                      label=zone, width=0.5, edgecolor="white", linewidth=0.8)
        for bar, val, bot in zip(bars, vals, bottoms):
            if val >= 1:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + val / 2, f"{val:.1f}%",
                        ha="center", va="center", fontsize=7, color="black", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylabel("Percentage of variants (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="RSA zone", fontsize=9, title_fontsize=9,
              frameon=False, loc="upper left", bbox_to_anchor=(1.05, 1))

    ns = [len(a) for a in arrays]
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={n:,})" for g, n in zip(groups, ns)], fontsize=10)

    if mw_p is not None:
        p_str = f"p = {mw_p:.2e}" if mw_p < 0.001 else f"p = {mw_p:.3f}"
        ax.text(0.5, -0.18, f"Mann-Whitney U  {p_str}",
                ha="center", va="top", fontsize=8, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    df = load_analysis_set(MERGED_CSV)

    plp_rsa = df.loc[df["sig_group"] == "Pathogenic/LP", "rsa"].dropna().values
    blb_rsa = df.loc[df["sig_group"] == "Benign/LB",     "rsa"].dropna().values

    stat, p = report_stats(plp_rsa, blb_rsa)
    plot_rsa_distribution(plp_rsa, blb_rsa, stat, p, FIGURE_PATH)
    plot_rsa_zone_bars(plp_rsa, blb_rsa, mw_p=p)

    # --- Median RSA per structural class ---
    plot_rsa_by_class(df)

    # --- Secondary structure × classification ---
    secondary_structure_chisq(df)

    # --- M3D × RSA analyses ---
    m3d_df = load_m3d_set(MERGED_CSV)
    m3d_rsa_fisher(m3d_df)
    category_buried_fractions(m3d_df)
    m3d_damaging_exposed_clinvar(m3d_df)


if __name__ == "__main__":
    main()
