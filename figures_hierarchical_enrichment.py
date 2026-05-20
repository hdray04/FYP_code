"""
Hierarchical enrichment — bars + internal counts only.
No text, labels, OR boxes or legend. Add those externally.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams["font.family"] = "Arial"

# ── Data ──────────────────────────────────────────────────────────────────────
enrich   = pd.read_csv("Csvs/clinvar_dbp_sig_enrichment.csv").set_index("sig_group")
dom_hits = pd.read_csv("Csvs/clinvar_domain_hits.csv", low_memory=False).drop_duplicates("#AlleleID")
out_hits = pd.read_csv("Csvs/clinvar_out_of_domain_hits.csv", low_memory=False).drop_duplicates("#AlleleID")

dbp_plp    = int(enrich.loc["Pathogenic/LP", "n_dbp"])
dbp_blb    = int(enrich.loc["Benign/LB",     "n_dbp"])
nondbp_plp = int(enrich.loc["Pathogenic/LP", "n_non_dbp"])
nondbp_blb = int(enrich.loc["Benign/LB",     "n_non_dbp"])
gw_plp     = dbp_plp + nondbp_plp
gw_blb     = dbp_blb + nondbp_blb
ind_plp    = int((dom_hits["sig_group"] == "Pathogenic/LP").sum())
ind_blb    = int((dom_hits["sig_group"] == "Benign/LB").sum())
out_plp    = int((out_hits["sig_group"] == "Pathogenic/LP").sum())
out_blb    = int((out_hits["sig_group"] == "Benign/LB").sum())

C_PATH = "#d62728"
C_BEN  = "#1f77b4"
BH     = 0.10   # bar height in figure coords — thicker

# Bar rects [left, bottom, width, height]
# In-domain and outside domain are symmetric around DBPs centre (0.30)
R = {
    "gw":     [0.175, 0.80, 0.65,  BH],
    "dbp":    [0.11,  0.54, 0.38,  BH],
    "nondbp": [0.55,  0.54, 0.38,  BH],
    "ind":    [0.025, 0.22, 0.275, BH],
    "out":    [0.325, 0.22, 0.275, BH],
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9))
fig.patch.set_facecolor("white")

def bar_node(rect, plp, blb):
    ax = fig.add_axes(rect)
    total = plp + blb
    p_plp = 100 * plp / total
    p_blb = 100 - p_plp

    ax.barh([0], [p_plp],               color=C_PATH, height=0.85)
    ax.barh([0], [p_blb], left=[p_plp], color=C_BEN,  height=0.85)

    if p_plp > 6:
        ax.text(p_plp / 2, 0, f"{plp:,}",
                ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if p_blb > 6:
        ax.text(p_plp + p_blb / 2, 0, f"{blb:,}",
                ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.axis("off")

bar_node(R["gw"],     gw_plp,     gw_blb)
bar_node(R["dbp"],    dbp_plp,    dbp_blb)
bar_node(R["nondbp"], nondbp_plp, nondbp_blb)
bar_node(R["ind"],    ind_plp,    ind_blb)
bar_node(R["out"],    out_plp,    out_blb)

import matplotlib.patches as mpatches
fig.legend(handles=[
    mpatches.Patch(facecolor=C_PATH, label="Pathogenic / Likely pathogenic (P/LP)"),
    mpatches.Patch(facecolor=C_BEN,  label="Benign / Likely benign (B/LB)"),
], loc="lower center", ncol=2, fontsize=9,
   framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, 0.01))

plt.savefig("Figures/hierarchical_enrichment.png", dpi=150, bbox_inches="tight")
print("Saved Figures/hierarchical_enrichment.png")
