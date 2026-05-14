import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2_contingency

TWO_STAR_STATUSES = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}

# Used only in section 7 (VUS) to identify 2-star variants for labelling;
# all other analyses in this script use all ClinVar variants unfiltered.

aa3_to_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V"
}

# --- Load M3D (DBP proteins only) ---
dbp_df = pd.read_csv(
    "Csvs/protein_classifications_hierarchy.csv",
    usecols=["uniprot_id", "structural_class", "is_te", "is_noncanonical"]
).drop_duplicates(subset=["uniprot_id"])
dbp_df = dbp_df[~dbp_df["is_te"] & ~dbp_df["is_noncanonical"]]
dbp_ids = set(dbp_df["uniprot_id"])

print("Loading M3D results for DBPs...")
m3d = pd.read_csv(
    "Data/M3D_results.csv",
    usecols=["uniprot", "position", "wildtype", "mutant", "damaging",
             "clash", "buried_H_bond_breakage", "cavity_altered",
             "buried_hydrophilic_introduced", "buried_charge_introduced",
             "disallowed_phi_psi", "buried_Gly_replaced",
             "buried_charge_replaced", "buried_salt_bridge_breakage",
             "buried_exposed_switch", "buried_Pro_introduced",
             "disulphide_breakage", "secondary_structure_altered",
             "buried_charge_switch", "cis_pro_replaced", "gly_in_a_bend"],
)
m3d = m3d[m3d["uniprot"].isin(dbp_ids)].copy()
m3d["damaging_bool"] = m3d["damaging"] == "Y"
print(f"M3D DBP rows: {len(m3d):,}  |  {m3d['uniprot'].nunique()} proteins")
print(f"Overall % damaging: {m3d['damaging_bool'].mean()*100:.1f}%")


def prep_cv(path):
    df = pd.read_csv(
        path,
        usecols=["#AlleleID", "uniprot", "canonical_residue", "residue_seq",
                 "modified_residue", "sig_group"],
    )
    df["ref_aa"] = df["canonical_residue"].map(aa3_to_1)
    df["alt_aa"] = df["modified_residue"].map(aa3_to_1)
    df["pos"]    = pd.to_numeric(df["residue_seq"], errors="coerce")
    return df


cv_in  = prep_cv("Csvs/clinvar_domain_hits.csv")
cv_out = prep_cv("Csvs/clinvar_out_of_domain_hits.csv")
cv_in["location"]  = "In DNA-binding domain"
cv_out["location"] = "Outside DNA-binding domain"

cv_in  = cv_in.merge(dbp_df, left_on="uniprot", right_on="uniprot_id", how="left")


def merge_m3d(cv_df):
    return cv_df.merge(
        m3d,
        left_on=["uniprot", "pos", "ref_aa", "alt_aa"],
        right_on=["uniprot", "position", "wildtype", "mutant"],
        how="left",
    ).drop_duplicates(subset=["#AlleleID"])


in_m  = merge_m3d(cv_in)
out_m = merge_m3d(cv_out)

print(f"\nIn-domain  ClinVar rows: {len(in_m):,}  |  M3D matched: {in_m['damaging'].notna().sum():,} ({in_m['damaging'].notna().mean()*100:.1f}%)")
print(f"Out-domain ClinVar rows: {len(out_m):,}  |  M3D matched: {out_m['damaging'].notna().sum():,} ({out_m['damaging'].notna().mean()*100:.1f}%)")


# ============================================================
# 1. In-domain vs out-of-domain P/LP: % M3D damaging
# ============================================================
MECH_COLS = [
    "clash", "buried_H_bond_breakage", "cavity_altered",
    "buried_hydrophilic_introduced", "buried_charge_introduced",
    "disallowed_phi_psi", "buried_Gly_replaced",
    "buried_charge_replaced", "buried_salt_bridge_breakage",
    "buried_exposed_switch", "buried_Pro_introduced",
    "disulphide_breakage", "secondary_structure_altered",
    "buried_charge_switch", "cis_pro_replaced", "gly_in_a_bend",
]

plp_in  = in_m[(in_m["sig_group"] == "Pathogenic/LP") & in_m["damaging"].notna()]
plp_out = out_m[(out_m["sig_group"] == "Pathogenic/LP") & out_m["damaging"].notna()]

n_in_dam  = plp_in["damaging_bool"].sum()
n_out_dam = plp_out["damaging_bool"].sum()
n_in      = len(plp_in)
n_out     = len(plp_out)

pct_in  = 100 * n_in_dam  / n_in
pct_out = 100 * n_out_dam / n_out

print("\n" + "="*60)
print("1. M3D structural disruption: in-domain vs out-of-domain P/LP")
print("="*60)
print(f"  In-domain  P/LP:  {n_in_dam}/{n_in} damaging ({pct_in:.1f}%)")
print(f"  Out-domain P/LP:  {n_out_dam}/{n_out} damaging ({pct_out:.1f}%)")

ct_loc = [[n_in_dam, n_in - n_in_dam], [n_out_dam, n_out - n_out_dam]]
or_loc, p_loc = fisher_exact(ct_loc, alternative="two-sided")
print(f"\n  Fisher's exact: OR={or_loc:.2f}, p={p_loc:.2e}")
print("  (OR > 1 = in-domain variants more often structurally disrupting)")

# Structural class breakdown for in-domain P/LP
print("\n--- In-domain P/LP: % M3D damaging by structural class ---")
sc_stats = []
for sc, grp in plp_in.groupby("structural_class"):
    n  = len(grp)
    nd = grp["damaging_bool"].sum()
    if n >= 5:
        sc_stats.append({"structural_class": sc, "n": n, "n_damaging": nd,
                         "pct_damaging": 100 * nd / n})
sc_df = pd.DataFrame(sc_stats).sort_values("pct_damaging", ascending=False)
print(sc_df.to_string(index=False))


# ============================================================
# 2. M3D mechanism breakdown for in-domain P/LP
# ============================================================
plp_in_dam = plp_in[plp_in["damaging_bool"]]

print("\n" + "="*60)
print("2. Structural mechanisms for in-domain P/LP (M3D damaging)")
print("="*60)
mech_counts = {c: (plp_in_dam[c] == "Y").sum() for c in MECH_COLS if c in plp_in_dam.columns}
mech_pct    = {c: 100 * v / len(plp_in_dam) for c, v in mech_counts.items()}
mech_series = pd.Series(mech_pct).sort_values(ascending=False)
for mech, pct in mech_series.items():
    if mech_counts[mech] > 0:
        print(f"  {mech:<40} {mech_counts[mech]:>4}  ({pct:.1f}%)")


# ============================================================
# 3. Concordance: M3D damaging vs ClinVar sig group (2-star)
# ============================================================
all_m = pd.concat([
    in_m[["#AlleleID", "sig_group", "damaging", "damaging_bool", "location"]],
    out_m[["#AlleleID", "sig_group", "damaging", "damaging_bool", "location"]],
], ignore_index=True)
all_m = all_m[all_m["damaging"].notna()].copy()

sig_order = ["Pathogenic/LP", "VUS", "Benign/LB"]
conc = all_m[all_m["sig_group"].isin(sig_order)].copy()

ct_conc = pd.crosstab(conc["sig_group"], conc["damaging"])[["Y", "N"]].reindex(sig_order)
ct_pct  = ct_conc.div(ct_conc.sum(axis=1), axis=0) * 100

print("\n" + "="*60)
print("3. M3D concordance with ClinVar (all variants, 2-star+)")
print("="*60)
print("\n--- Counts ---")
print(ct_conc.to_string())
print("\n--- Row % ---")
print(ct_pct.round(1).to_string())

chi2, p_chi, dof, _ = chi2_contingency(ct_conc)
print(f"\nChi-squared: {chi2:.1f}, df={dof}, p={p_chi:.2e}")

n_plp_dam = ct_conc.loc["Pathogenic/LP", "Y"]; tot_plp = ct_conc.loc["Pathogenic/LP"].sum()
n_blb_ben = ct_conc.loc["Benign/LB",     "N"]; tot_blb = ct_conc.loc["Benign/LB"].sum()
print(f"\nP/LP called damaging by M3D:   {n_plp_dam}/{tot_plp} ({100*n_plp_dam/tot_plp:.1f}%)")
print(f"B/LB called not-damaging by M3D: {n_blb_ben}/{tot_blb} ({100*n_blb_ben/tot_blb:.1f}%)")

# In- vs out-of-domain concordance separately
print("\n--- Concordance split by location ---")
for loc in ["In DNA-binding domain", "Outside DNA-binding domain"]:
    sub = conc[conc["location"] == loc]
    ct  = pd.crosstab(sub["sig_group"], sub["damaging"])[["Y", "N"]].reindex(
        [s for s in sig_order if s in sub["sig_group"].values]
    )
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print(f"\n  {loc}:")
    print(ct.join(pct.round(1).add_suffix("_%")).to_string())


# ============================================================
# 4. B/LB discordants: do AM-pathogenic B/LB variants also
#    show M3D structural disruption?
# ============================================================
am_merged = pd.read_csv("Csvs/clinvar_alphamissense_merged_dedup.csv",
                        usecols=["#AlleleID", "sig_group", "am_class",
                                 "am_pathogenicity", "uniprot",
                                 "canonical_residue", "residue_seq",
                                 "modified_residue", "structural_class"])
am_merged["ref_aa"] = am_merged["canonical_residue"].map(aa3_to_1)
am_merged["alt_aa"] = am_merged["modified_residue"].map(aa3_to_1)
am_merged["pos"]    = pd.to_numeric(am_merged["residue_seq"], errors="coerce")

blb_am_path = am_merged[
    (am_merged["sig_group"] == "Benign/LB") & (am_merged["am_class"] == "pathogenic")
].copy()

blb_am_path_m3d = blb_am_path.merge(
    m3d[["uniprot", "position", "wildtype", "mutant", "damaging"] + MECH_COLS],
    left_on=["uniprot", "pos", "ref_aa", "alt_aa"],
    right_on=["uniprot", "position", "wildtype", "mutant"],
    how="left",
)

print("\n" + "="*60)
print("4. B/LB variants AM calls pathogenic: M3D assessment")
print("="*60)
print(f"  Total B/LB AM-pathogenic (2-star): {len(blb_am_path_m3d)}")
print(f"  M3D match found: {blb_am_path_m3d['damaging'].notna().sum()}")
matched = blb_am_path_m3d[blb_am_path_m3d["damaging"].notna()]
print(f"  M3D also damaging: {(matched['damaging']=='Y').sum()} / {len(matched)}")

print("\n  Per-variant detail:")
cols_show = ["uniprot", "am_pathogenicity", "structural_class", "damaging"]
avail_mech = [c for c in MECH_COLS if c in matched.columns]
mech_flags = matched[avail_mech].apply(lambda r: ", ".join(c for c in avail_mech if r[c]=="Y"), axis=1)
matched = matched.copy()
matched["m3d_mechanisms"] = mech_flags
print(matched[["uniprot", "am_pathogenicity", "structural_class", "damaging", "m3d_mechanisms"]]
      .sort_values("am_pathogenicity", ascending=False)
      .to_string(index=False))


# ============================================================
# 5. Bar chart: % M3D damaging by sig group and location
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

bar_data = (
    conc.groupby(["sig_group", "location"])["damaging_bool"]
    .agg(pct=lambda x: 100 * x.mean(), n="count")
    .reset_index()
)

locs   = ["In DNA-binding domain", "Outside DNA-binding domain"]
colors = ["#d62728", "#2171b5"]
x      = np.arange(len(sig_order))
width  = 0.35

for i, (loc, col) in enumerate(zip(locs, colors)):
    sub  = bar_data[bar_data["location"] == loc].set_index("sig_group")
    vals = [sub.loc[s, "pct"] if s in sub.index else 0 for s in sig_order]
    ns   = [int(sub.loc[s, "n"]) if s in sub.index else 0 for s in sig_order]
    bars = ax.bar(x + (i - 0.5) * width, vals, width, label=loc, color=col, alpha=0.8)
    for bar, v, n in zip(bars, vals, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(sig_order)
ax.set_ylabel("% variants predicted structurally disrupting (Missense3D)")
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/m3d_damaging_by_sig_group_location.png", dpi=150, bbox_inches="tight")
print("\nSaved Figures/m3d_damaging_by_sig_group_location.png")


# ============================================================
# 6. OR per structural class: M3D in-domain vs out-of-domain
#    damaging, then Spearman against ClinVar OR
# ============================================================
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

# Build combined in/out dataframe with structural_class
in_m_sc  = in_m[in_m["damaging"].notna()].copy()
out_m_sc = out_m[out_m["damaging"].notna()].copy()

# out_m doesn't have structural_class yet — add it
out_m_sc = out_m_sc.merge(
    dbp_df[["uniprot_id", "structural_class"]].rename(columns={"uniprot_id": "uniprot"}),
    on="uniprot", how="left"
)

in_m_sc["loc"]  = "in"
out_m_sc["loc"] = "out"
# Re-cast after left merge (dtype can downgrade to object)
in_m_sc["damaging_bool"]  = in_m_sc["damaging_bool"].astype(bool)
out_m_sc["damaging_bool"] = out_m_sc["damaging_bool"].astype(bool)
combined_sc = pd.concat(
    [in_m_sc[["structural_class", "damaging_bool", "loc"]],
     out_m_sc[["structural_class", "damaging_bool", "loc"]]],
    ignore_index=True,
).dropna(subset=["structural_class"])

proteins_per_class = dbp_df.groupby("structural_class")["uniprot_id"].nunique().rename("n_proteins")

m3d_or_results = []
for sc, grp in combined_sc.groupby("structural_class"):
    in_sc  = grp[grp["loc"] == "in"]
    out_sc = grp[grp["loc"] == "out"]
    a = int((in_sc["damaging_bool"]).sum());   b = int((~in_sc["damaging_bool"]).sum())
    c = int((out_sc["damaging_bool"]).sum());  d = int((~out_sc["damaging_bool"]).sum())
    if (a + b) < 5 or (c + d) < 5:
        continue
    _, p_raw = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    cc = min(a, b, c, d) == 0
    ac, bc, cc_n, dc = (a+0.5, b+0.5, c+0.5, d+0.5) if cc else (a, b, c, d)
    or_cc = (ac * dc) / (bc * cc_n)
    se    = np.sqrt(1/ac + 1/bc + 1/cc_n + 1/dc)
    ci_lo = np.exp(np.log(or_cc) - 1.96 * se)
    ci_hi = np.exp(np.log(or_cc) + 1.96 * se)
    m3d_or_results.append({
        "structural_class": sc,
        "OR_m3d": or_cc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_raw": p_raw,
        "in_pct_dam":  100 * a / (a + b) if (a + b) > 0 else np.nan,
        "out_pct_dam": 100 * c / (c + d) if (c + d) > 0 else np.nan,
        "n_in": a + b,
        "n_out": c + d,
    })

m3d_or_df = pd.DataFrame(m3d_or_results) if m3d_or_results else pd.DataFrame(columns=["structural_class","OR_m3d","ci_lo","ci_hi","p_raw","in_pct_dam","out_pct_dam","n_in","n_out"])
m3d_or_df = m3d_or_df.merge(proteins_per_class.reset_index(), on="structural_class", how="left")
m3d_or_df = m3d_or_df[m3d_or_df["n_proteins"] >= 2]
n_tests = len(m3d_or_df)
m3d_or_df["p_bonf"] = (m3d_or_df["p_raw"] * n_tests).clip(upper=1.0)
m3d_or_df = m3d_or_df.sort_values("OR_m3d", ascending=False)

print("\n" + "="*60)
print("6. M3D in-domain vs out-of-domain damaging enrichment per class")
print("="*60)
print(f"{'Class':<55} {'OR':>7}  {'in%':>6}  {'out%':>6}  {'p_bonf':>10}")
print("-"*90)
for _, row in m3d_or_df.iterrows():
    sig = "*" if row["p_bonf"] < 0.05 else ""
    print(f"{row['structural_class']:<55} {row['OR_m3d']:>7.2f}  "
          f"{row['in_pct_dam']:>5.1f}%  {row['out_pct_dam']:>5.1f}%  "
          f"{row['p_bonf']:>10.3e} {sig}")

# --- Spearman: M3D OR vs ClinVar OR ---
cv_enrich = pd.read_csv("Csvs/clinvar_inout_domain_enrichment.csv")[
    ["structural_class", "or_in_vs_out"]
].rename(columns={"or_in_vs_out": "OR_cv"})

or_merged = m3d_or_df[["structural_class", "OR_m3d"]].merge(cv_enrich, on="structural_class")
or_merged = or_merged.replace([np.inf, -np.inf], np.nan).dropna()

rho, p_rho = spearmanr(or_merged["OR_m3d"], or_merged["OR_cv"])
print(f"\nSpearman: M3D OR vs ClinVar OR (n={len(or_merged)} classes)")
print(f"rho={rho:.3f}, p={p_rho:.2e}")

# --- Scatter: log2 OR_m3d vs log2 OR_cv ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(np.log2(or_merged["OR_cv"]), np.log2(or_merged["OR_m3d"]),
           s=55, color="#2171b5", alpha=0.85, zorder=3)
for _, row in or_merged.iterrows():
    ax.annotate(row["structural_class"].replace(" superfamily","").replace(" domain",""),
                (np.log2(row["OR_cv"]), np.log2(row["OR_m3d"])),
                fontsize=6.5, xytext=(4, 3), textcoords="offset points", color="#333")
ax.axline((0, 0), slope=1, color="#aaa", linestyle="--", linewidth=1, label="x = y")
ax.axhline(0, color="#ccc", linewidth=0.8)
ax.axvline(0, color="#ccc", linewidth=0.8)
ax.set_xlabel("log₂ OR (ClinVar P/LP: in vs out of domain)")
ax.set_ylabel("log₂ OR (M3D damaging: in vs out of domain)")
ax.set_title(f"Spearman ρ={rho:.3f}, p={p_rho:.2e} (n={len(or_merged)} classes)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color="#eee", zorder=0)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#eee", zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("Figures/m3d_vs_clinvar_or_scatter.png", dpi=150, bbox_inches="tight")
print("Saved Figures/m3d_vs_clinvar_or_scatter.png")

# --- Slope / bump chart: class rank comparison ---
or_merged["rank_cv"]  = or_merged["OR_cv"].rank(ascending=False).astype(int)
or_merged["rank_m3d"] = or_merged["OR_m3d"].rank(ascending=False).astype(int)
or_merged = or_merged.sort_values("rank_cv")

def shorten(name):
    for k, v in {" superfamily":"", " domain":"", "transcription factor":"TF",
                 "DNA-binding":"DBD", "Transcription factor, ":"",
                 "containing":"cont.", "Nuclear hormone receptor-like":"NHR-like",
                 "Hepatocyte nuclear factor 1, N-terminal":"HNF1 N-term"}.items():
        name = name.replace(k, v)
    return name.strip(", ")

or_merged["label"] = or_merged["structural_class"].apply(shorten)

fig, ax = plt.subplots(figsize=(10, 7))
n = len(or_merged)
for _, row in or_merged.iterrows():
    delta = row["rank_m3d"] - row["rank_cv"]
    color = "#d62728" if delta > 2 else "#2171b5" if delta < -2 else "#888"
    ax.plot([0, 1], [row["rank_cv"], row["rank_m3d"]], color=color, linewidth=1.4, alpha=0.75)
    ax.text(-0.02, row["rank_cv"], row["label"], ha="right", va="center", fontsize=7, color=color)
    ax.text(1.02,  row["rank_m3d"], row["label"], ha="left",  va="center", fontsize=7, color=color)

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(n + 0.5, 0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["ClinVar OR rank\n(in vs out)", "M3D OR rank\n(in vs out)"], fontsize=10)
ax.set_ylabel("Rank (1 = highest OR)")
ax.set_title("Class ranking: ClinVar P/LP enrichment vs M3D damaging enrichment\n(in-domain vs out-of-domain)", fontsize=10)
ax.legend(handles=[
    Line2D([0],[0], color="#d62728", lw=1.5, label="Ranks higher in M3D"),
    Line2D([0],[0], color="#2171b5", lw=1.5, label="Ranks higher in ClinVar"),
    Line2D([0],[0], color="#888",    lw=1.5, label="Similar rank"),
], fontsize=8, loc="upper center")
ax.spines[["top","right","left","bottom"]].set_visible(False)
ax.yaxis.set_visible(False)
plt.tight_layout()
plt.savefig("Figures/m3d_vs_clinvar_rank_slope.png", dpi=150, bbox_inches="tight")
print("Saved Figures/m3d_vs_clinvar_rank_slope.png")


# ============================================================
# 7. VUS prioritisation: AM pathogenic AND M3D damaging
#    No review-status filter — VUS are uncertain by definition
#    and single-submitter VUS are exactly the ones that need
#    computational prioritisation most.
# ============================================================
cv_vus_cols = {"#AlleleID", "uniprot", "canonical_residue", "residue_seq",
               "modified_residue", "sig_group", "ReviewStatus"}
cv_vus_raw = pd.read_csv("Csvs/clinvar_domain_hits.csv",
                         usecols=lambda c: c in cv_vus_cols)
cv_vus_raw = cv_vus_raw[cv_vus_raw["sig_group"] == "VUS"].copy()
cv_vus_raw["ref_aa"] = cv_vus_raw["canonical_residue"].map(aa3_to_1)
cv_vus_raw["alt_aa"] = cv_vus_raw["modified_residue"].map(aa3_to_1)
cv_vus_raw["pos"]    = pd.to_numeric(cv_vus_raw["residue_seq"], errors="coerce")

# Add structural class
cv_vus_raw = cv_vus_raw.merge(dbp_df[["uniprot_id", "structural_class"]],
                               left_on="uniprot", right_on="uniprot_id", how="left")

# Add AM predictions
am_vus = pd.read_csv("Csvs/alphamissense_dbp.csv",
                     usecols=["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"])
am_parts_vus = am_vus["protein_variant"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
am_vus["ref_aa"]  = am_parts_vus[0]
am_vus["pos"]     = pd.to_numeric(am_parts_vus[1], errors="coerce")
am_vus["alt_aa"]  = am_parts_vus[2]

vus_in = cv_vus_raw.merge(
    am_vus[["uniprot_id", "ref_aa", "pos", "alt_aa", "am_pathogenicity", "am_class"]],
    left_on=["uniprot", "ref_aa", "pos", "alt_aa"],
    right_on=["uniprot_id", "ref_aa", "pos", "alt_aa"],
    how="left",
).drop_duplicates(subset=["#AlleleID"])

# Add M3D prediction
vus_m3d = vus_in.merge(
    m3d[["uniprot", "position", "wildtype", "mutant", "damaging"] + MECH_COLS],
    left_on=["uniprot", "pos", "ref_aa", "alt_aa"],
    right_on=["uniprot", "position", "wildtype", "mutant"],
    how="left",
).drop_duplicates(subset=["#AlleleID"])

n_vus_total  = len(vus_m3d)
n_m3d_match  = vus_m3d["damaging"].notna().sum()
vus_matched  = vus_m3d[vus_m3d["damaging"].notna()].copy()

# Tiers
n_am_path    = (vus_matched["am_class"] == "pathogenic").sum()
n_m3d_dam    = (vus_matched["damaging"] == "Y").sum()
n_both       = ((vus_matched["am_class"] == "pathogenic") & (vus_matched["damaging"] == "Y")).sum()
n_am_only    = ((vus_matched["am_class"] == "pathogenic") & (vus_matched["damaging"] == "N")).sum()
n_m3d_only   = ((vus_matched["am_class"] != "pathogenic") & (vus_matched["damaging"] == "Y")).sum()
n_neither    = ((vus_matched["am_class"] != "pathogenic") & (vus_matched["damaging"] == "N")).sum()

print("\n" + "="*60)
print("7. VUS prioritisation: AM pathogenic AND M3D damaging")
print("="*60)
print(f"In-domain VUS (all): {n_vus_total}")
print(f"  With M3D prediction:  {n_m3d_match} ({100*n_m3d_match/n_vus_total:.1f}%)")
print(f"\nOf {n_m3d_match} matched VUS:")
print(f"  AM pathogenic only (M3D not damaging): {n_am_only:>4}  ({100*n_am_only/n_m3d_match:.1f}%)")
print(f"  M3D damaging only  (AM not pathogenic):{n_m3d_only:>4}  ({100*n_m3d_only/n_m3d_match:.1f}%)")
print(f"  BOTH AM path + M3D damaging:           {n_both:>4}  ({100*n_both/n_m3d_match:.1f}%)  ← highest priority")
print(f"  Neither:                               {n_neither:>4}  ({100*n_neither/n_m3d_match:.1f}%)")

# --- Structural class breakdown of dual-flagged VUS ---
both_vus = vus_matched[(vus_matched["am_class"] == "pathogenic") & (vus_matched["damaging"] == "Y")].copy()

print(f"\n--- Structural class breakdown: dual-flagged VUS ---")
sc_vus_total  = vus_matched.groupby("structural_class").size().rename("n_vus")
sc_am_path    = vus_matched[vus_matched["am_class"] == "pathogenic"].groupby("structural_class").size().rename("n_am_path")
sc_both       = both_vus.groupby("structural_class").size().rename("n_both")
sc_vus_sum    = sc_vus_total.to_frame().join(sc_am_path, how="left").join(sc_both, how="left").fillna(0)
sc_vus_sum    = sc_vus_sum.astype({"n_am_path": int, "n_both": int})
sc_vus_sum["pct_am_path"] = 100 * sc_vus_sum["n_am_path"] / sc_vus_sum["n_vus"]
sc_vus_sum["pct_both"]    = 100 * sc_vus_sum["n_both"]    / sc_vus_sum["n_vus"]
sc_vus_sum    = sc_vus_sum[sc_vus_sum["n_vus"] >= 5].sort_values("n_both", ascending=False)
print(f"\n{'Structural class':<55} {'n_VUS':>6}  {'AM_path%':>9}  {'Both%':>7}  {'n_both':>7}")
print("-"*95)
for sc, row in sc_vus_sum.iterrows():
    print(f"{sc:<55} {int(row['n_vus']):>6}  {row['pct_am_path']:>8.1f}%  "
          f"{row['pct_both']:>6.1f}%  {int(row['n_both']):>7}")

# --- Candidate list: the dual-flagged variants ---
mech_flags = both_vus[[c for c in MECH_COLS if c in both_vus.columns]].apply(
    lambda r: ", ".join(c for c in MECH_COLS if c in r.index and r[c] == "Y"), axis=1
)
both_vus = both_vus.copy()
both_vus["m3d_mechanisms"] = mech_flags

print(f"\n--- Candidate variants (AM pathogenic + M3D damaging, n={len(both_vus)}) ---")
print(both_vus[["uniprot", "am_pathogenicity", "structural_class", "m3d_mechanisms"]]
      .sort_values(["structural_class", "am_pathogenicity"], ascending=[True, False])
      .to_string(index=False))

# --- Lollipop: % dual-flagged by structural class ---
sc_plot = sc_vus_sum[sc_vus_sum["n_vus"] >= 10].sort_values("pct_both")
if len(sc_plot) > 0:
    NAME_SHORT = {
        "Hepatocyte nuclear factor 1, N-terminal domain superfamily": "HNF1 N-terminal",
        "Nuclear hormone receptor-like domain superfamily":            "Nuclear hormone receptor",
        "CTF transcription factor/nuclear factor 1, DNA-binding domain": "CTF/NF-1",
        "Transcription factor, MADS-box superfamily":                  "MADS-box",
        "p53-like transcription factor, DNA-binding domain superfamily": "p53-like TF",
        "High mobility group box domain superfamily":                  "HMG box",
        "Winged helix-like DNA-binding domain superfamily":            "Winged helix",
        "Homeodomain-like superfamily":                                "Homeodomain",
        "Zinc finger superfamily":                                     "Zinc finger",
    }
    labels = [NAME_SHORT.get(sc, sc) + f"  (n={int(sc_plot.loc[sc,'n_vus'])})" for sc in sc_plot.index]
    fig, ax = plt.subplots(figsize=(9, max(4, len(sc_plot) * 0.45)))
    y = range(len(sc_plot))
    overall_pct_both = 100 * n_both / n_m3d_match

    ax.hlines(y, 0, sc_plot["pct_am_path"], color="#aaddff", linewidth=6, alpha=0.6, zorder=1, label="AM pathogenic only")
    ax.hlines(y, 0, sc_plot["pct_both"],    color="#d62728", linewidth=3, zorder=2, label="Both AM+M3D")
    ax.scatter(sc_plot["pct_am_path"], y, color="#2171b5", s=50, zorder=3)
    ax.scatter(sc_plot["pct_both"],    y, color="#d62728", s=50, zorder=4)

    ax.axvline(overall_pct_both, color="#555", linestyle="--", linewidth=1)
    ax.text(overall_pct_both + 0.3, len(sc_plot) - 0.5,
            f"Overall\n({overall_pct_both:.1f}%)", fontsize=7.5, color="#555", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("% of in-domain VUS flagged")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("Figures/vus_dual_flagged_by_class.png", dpi=150, bbox_inches="tight")
    print("\nSaved Figures/vus_dual_flagged_by_class.png")
