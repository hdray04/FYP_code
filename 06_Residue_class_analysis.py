import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# --- Load data ---
cv = pd.read_csv("Csvs/clinvar_domain_hits.csv")
clf = pd.read_csv("Csvs/protein_classifications_hierarchy.csv",
                  usecols=["uniprot_id", "structural_class"]).drop_duplicates()

cv = cv.merge(clf, left_on="uniprot", right_on="uniprot_id", how="left")

# --- Filter p53-like domain ---
p53 = cv[
    (cv["structural_class"] == "p53-like transcription factor, DNA-binding domain superfamily")
].drop_duplicates(subset=["#AlleleID"]).copy()

p53["domain_length"] = pd.to_numeric(p53["domain_end"], errors="coerce") - \
                       pd.to_numeric(p53["domain_start"], errors="coerce")
p53["residue_seq"] = pd.to_numeric(p53["residue_seq"], errors="coerce")
p53["domain_start"] = pd.to_numeric(p53["domain_start"], errors="coerce")

# --- Domain length check ---
print(f"Total p53 variants before length filter: {len(p53)}")
print(f"\nDomain length distribution:")
print(p53["domain_length"].value_counts().sort_index().to_string())
modal_len = p53["domain_length"].mode()[0]
n_modal = (p53["domain_length"] == modal_len).sum()
print(f"\nMost common length: {int(modal_len)}aa  ({n_modal} variants, {100*n_modal/len(p53):.1f}%)")

print("\n=== 178-187aa cluster ===")
p53_low = p53[p53["domain_length"].between(178, 187)]
print(p53_low.groupby(["GeneSymbol", "uniprot"]).size().sort_values(ascending=False).head(10))

print("\n=== 190-192aa cluster ===")
p53_high = p53[p53["domain_length"].between(190, 192)]
print(p53_high.groupby(["GeneSymbol", "uniprot"]).size().sort_values(ascending=False).head(10))

print("\n=== outliers outside 178-192aa ===")
p53_other = p53[~p53["domain_length"].between(178, 192)]
print(p53_other.groupby(["GeneSymbol", "uniprot", "domain_length"]).size().sort_values(ascending=False).head(10))

hmg = p53[p53["domain_length"].between(190, 192)].copy()
print(f"After outlier removal: {len(hmg)} variants")

# Relative position
hmg["rel_pos"] = (hmg["residue_seq"] - hmg["domain_start"]) / hmg["domain_length"]
hmg = hmg[hmg["rel_pos"].between(0, 1)].copy()

# --- Canonical HMG box contact positions (1-indexed within 68aa box) ---
# From SRY crystal structure (PDB 1J46) and SOX2/SOX4 structures
contact_positions = {
    9:  "N-arm contact",
    10: "N-arm contact",
    12: "Phe intercalation",
    17: "Arg phosphate",
    19: "Arg phosphate",
    43: "Trp intercalation",
    47: "Contact",
}
domain_length_canonical = 68
contact_rel = {pos/domain_length_canonical: label
               for pos, label in contact_positions.items()}

# --- Position distribution plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

sig_groups = ["Pathogenic/LP", "VUS", "Benign/LB"]
colors = ["#d62728", "grey", "#2171b5"]

for ax, sg, col in zip(axes, sig_groups, colors):
    subset = hmg[hmg["sig_group"] == sg]["rel_pos"].dropna()
    if len(subset) == 0:
        ax.text(0.5, 0.5, f"No {sg} variants", ha="center", transform=ax.transAxes)
        continue

    ax.hist(subset, bins=34, color=col, alpha=0.7, density=True)

    for rel, label in contact_rel.items():
        ax.axvline(rel, color="black", linestyle="--", linewidth=1, alpha=0.8)

    ax.set_ylabel(f"{sg}\n(n={len(subset):,})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

# Label contact positions on top panel only
for rel, label in contact_rel.items():
    axes[0].text(rel, axes[0].get_ylim()[1],
                 f"p{int(rel*68)}", fontsize=7, ha="center",
                 rotation=90, va="bottom")

axes[-1].set_xlabel("Relative position within HMG box (0 = start, 1 = end)")
axes[0].set_title("HMG box domain superfamily (SOX family): variant position distribution")
plt.tight_layout()
plt.savefig("Figures/hmg_position_distribution.png", dpi=150, bbox_inches="tight")


# --- Hotspot analysis ---
plp_hmg = hmg[hmg["sig_group"] == "Pathogenic/LP"].copy()
hotspots = (
    plp_hmg.groupby(["GeneSymbol", "residue_seq"])
    .size()
    .reset_index(name="n_variants")
    .sort_values("n_variants", ascending=False)
)
print("\n=== HMG box P/LP hotspots (n>=2) ===")
print(hotspots[hotspots["n_variants"] >= 2].to_string(index=False))

# --- Are hotspot positions near known contact residues? ---
plp_hmg["pos_in_domain"] = (plp_hmg["residue_seq"] - plp_hmg["domain_start"]).round()
print("\n=== P/LP position within domain (absolute) ===")
print(plp_hmg["pos_in_domain"].value_counts().head(20).to_string())

# --- Contact vs non-contact enrichment ---
# Define contact window: positions 8-20 and 40-48 (1-indexed)
contact_windows = [(8, 20), (40, 48)]

def is_contact(pos):
    for start, end in contact_windows:
        if start <= pos <= end:
            return True
    return False

plp_hmg["is_contact"] = plp_hmg["pos_in_domain"].apply(is_contact)
blb_hmg = hmg[hmg["sig_group"] == "Benign/LB"].copy()
blb_hmg["pos_in_domain"] = (blb_hmg["residue_seq"] - blb_hmg["domain_start"]).round()
blb_hmg["is_contact"] = blb_hmg["pos_in_domain"].apply(is_contact)

plp_contact    = plp_hmg["is_contact"].sum()
plp_noncontact = (~plp_hmg["is_contact"]).sum()
blb_contact    = blb_hmg["is_contact"].sum()
blb_noncontact = (~blb_hmg["is_contact"]).sum()

print(f"\n=== Contact region enrichment ===")
print(f"P/LP in contact regions: {plp_contact}/{len(plp_hmg)} ({100*plp_contact/len(plp_hmg):.1f}%)")
print(f"B/LB in contact regions: {blb_contact}/{len(blb_hmg)} ({100*blb_contact/len(blb_hmg):.1f}%)")

if blb_contact + blb_noncontact > 0:
    ct = [[plp_contact, plp_noncontact], [blb_contact, blb_noncontact]]
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi-square: p={p:.3e}")

# --- Substitution analysis ---
aa3_to_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V"
}
charge = {"R":1,"K":1,"H":0.5,"D":-1,"E":-1,
          "A":0,"N":0,"C":0,"F":0,"G":0,"I":0,"L":0,
          "M":0,"P":0,"Q":0,"S":0,"T":0,"V":0,"W":0,"Y":0}

hmg["ref_aa"] = hmg["canonical_residue"].map(aa3_to_1)
hmg["alt_aa"] = hmg["modified_residue"].map(aa3_to_1)
hmg["charge_change"] = hmg["alt_aa"].map(charge) - hmg["ref_aa"].map(charge)

plp_charge = hmg[hmg["sig_group"] == "Pathogenic/LP"]["charge_change"].dropna()
blb_charge = hmg[hmg["sig_group"] == "Benign/LB"]["charge_change"].dropna()

print(f"\n=== Substitution analysis ===")
print(f"P/LP with charge change: {(plp_charge != 0).sum()} ({100*(plp_charge != 0).mean():.1f}%)")
print(f"B/LB with charge change: {(blb_charge != 0).sum()} ({100*(blb_charge != 0).mean():.1f}%)")

print(f"\nTop P/LP substitutions:")
plp_subs = hmg[hmg["sig_group"] == "Pathogenic/LP"].dropna(subset=["ref_aa", "alt_aa"]).copy()
plp_subs["sub"] = plp_subs["ref_aa"] + ">" + plp_subs["alt_aa"]
print(plp_subs["sub"].value_counts().head(10).to_string())