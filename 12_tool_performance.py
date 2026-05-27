import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv("Csvs/clinvar_am_m3d_dssp_merged.csv")

# Ground truth: binary label
gt_mask = df["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])
gt = df[gt_mask].copy()
gt["true_pos"] = (gt["sig_group"] == "Pathogenic/LP").astype(int)

print(f"Ground-truth variants: {len(gt):,}  "
      f"({gt['true_pos'].sum()} P/LP, {(1 - gt['true_pos']).sum()} B/LB)")


# ── helpers ─────────────────────────────────────────────────

def confusion(y_true, y_pred):
    """Return TP, FP, TN, FN."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    sens   = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec   = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    acc    = (tp + tn) / total if total > 0 else np.nan
    denom  = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc    = (tp*tn - fp*fn) / denom if denom > 0 else np.nan
    return dict(sensitivity=sens, specificity=spec, accuracy=acc, mcc=mcc)


def print_metrics(name, tp, fp, tn, fn):
    m = metrics(tp, fp, tn, fn)
    n = tp + fp + tn + fn
    print(f"\n  {name}  (n={n})")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    for k, v in m.items():
        print(f"    {k:<14} {v:.3f}")
    return m


# ============================================================
# 1. AlphaMissense  (binary: pathogenic vs benign)
#    Ambiguous calls are excluded from the binary analysis but
#    counted separately.
# ============================================================
am_sub = gt[gt["am_class"].isin(["pathogenic", "benign"])].copy()
am_ambig = gt[gt["am_class"] == "ambiguous"]
am_sub["pred"] = (am_sub["am_class"] == "pathogenic").astype(int)

print("\n" + "="*60)
print("1. AlphaMissense vs ClinVar ground truth")
print("="*60)
print(f"  Ambiguous calls excluded: {len(am_ambig)}")

tp_am, fp_am, tn_am, fn_am = confusion(am_sub["true_pos"], am_sub["pred"])
m_am = print_metrics("AlphaMissense", tp_am, fp_am, tn_am, fn_am)

# ROC / AUC using continuous score
am_roc = gt[gt["am_pathogenicity"].notna()].copy()
auc_am = roc_auc_score(am_roc["true_pos"], am_roc["am_pathogenicity"])
fpr_am, tpr_am, _ = roc_curve(am_roc["true_pos"], am_roc["am_pathogenicity"])
print(f"\n  AUC-ROC (continuous score, n={len(am_roc)}): {auc_am:.3f}")


# ============================================================
# 2. Missense3D  (binary: Y vs N)
# ============================================================
m3d_sub = gt[gt["damaging"].isin(["Y", "N"])].copy()
m3d_sub["pred"] = (m3d_sub["damaging"] == "Y").astype(int)

print("\n" + "="*60)
print("2. Missense3D vs ClinVar ground truth")
print("="*60)

tp_m3, fp_m3, tn_m3, fn_m3 = confusion(m3d_sub["true_pos"], m3d_sub["pred"])
m_m3 = print_metrics("Missense3D", tp_m3, fp_m3, tn_m3, fn_m3)


# ============================================================
# 3. Summary table comparing both tools on their own sets
# ============================================================
print("\n" + "="*60)
print("3. Summary comparison (each tool on its own coverage)")
print("="*60)
metric_keys = ["sensitivity", "specificity", "accuracy", "mcc"]
header = f"{'Metric':<16}  {'AM (n='+str(len(am_sub))+')':>14}  {'M3D (n='+str(len(m3d_sub))+')':>14}  {'Δ (AM−M3D)':>12}"
print(header)
print("-" * len(header))
for k in metric_keys:
    v_am  = m_am[k]
    v_m3  = m_m3[k]
    delta = v_am - v_m3
    print(f"{k:<16}  {v_am:>14.3f}  {v_m3:>14.3f}  {delta:>+12.3f}")


# ============================================================
# 4. Figures
# ============================================================
METRIC_LABELS = ["Sensitivity", "Specificity", "Accuracy", "MCC"]
METRIC_KEYS   = ["sensitivity", "specificity", "accuracy", "mcc"]
AM_COLOR  = "#2171b5"
M3D_COLOR = "#d62728"

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── 4a. Grouped bar: metric comparison ──────────────────────
ax1 = fig.add_subplot(gs[0, :])
x     = np.arange(len(METRIC_LABELS))
width = 0.35
vals_am  = [m_am[k] for k in METRIC_KEYS]
vals_m3d = [m_m3[k] for k in METRIC_KEYS]

bars_am  = ax1.bar(x - width/2, vals_am,  width, label="AlphaMissense", color=AM_COLOR,  alpha=0.85)
bars_m3d = ax1.bar(x + width/2, vals_m3d, width, label="Missense3D",    color=M3D_COLOR, alpha=0.85)

for bar, v in zip(bars_am, vals_am):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.2f}", ha="center", va="bottom", fontsize=8)
for bar, v in zip(bars_m3d, vals_m3d):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.2f}", ha="center", va="bottom", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(METRIC_LABELS)
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.12)
ax1.set_title(f"AM vs M3D performance vs ClinVar ground truth\n(AM n={len(am_sub)}, M3D n={len(m3d_sub)}, each on own coverage)")
ax1.legend(fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax1.set_axisbelow(True)

# ── 4b. Confusion matrix: AM ─────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
cm_am = np.array([[tp_am, fn_am], [fp_am, tn_am]])
im = ax2.imshow(cm_am, cmap="Blues", vmin=0)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred +\n(pathogenic)", "Pred −\n(benign)"])
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["True +\n(P/LP)", "True −\n(B/LB)"])
ax2.set_title("AlphaMissense — confusion matrix")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm_am[i, j]), ha="center", va="center",
                 fontsize=13, color="white" if cm_am[i, j] > cm_am.max()/2 else "black")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# ── 4c. Confusion matrix: M3D ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
cm_m3 = np.array([[tp_m3, fn_m3], [fp_m3, tn_m3]])
im3 = ax3.imshow(cm_m3, cmap="Blues", vmin=0)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["Pred +\n(damaging)", "Pred −\n(not dam.)"])
ax3.set_yticks([0, 1]); ax3.set_yticklabels(["True +\n(P/LP)", "True −\n(B/LB)"])
ax3.set_title("Missense3D — confusion matrix")
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm_m3[i, j]), ha="center", va="center",
                 fontsize=13, color="white" if cm_m3[i, j] > cm_m3.max()/2 else "black")
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.savefig("Figures/tool_performance_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved Figures/tool_performance_comparison.png")



# ============================================================
# 5. In-domain vs out-of-domain: AM and M3D performance
#    Uses the separate in/out ClinVar CSVs so the out-of-domain
#    set is properly represented (the merged CSV is in-domain only).
#    DBP protein filter applied (same criteria as 09_missense3D.py).
# ============================================================
from scipy.stats import fisher_exact

aa3_to_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
}

# DBP protein filter
dbp_df = pd.read_csv(
    "Csvs/protein_classifications_hierarchy.csv",
    usecols=["uniprot_id", "is_te", "is_noncanonical"],
).drop_duplicates(subset=["uniprot_id"])
dbp_df = dbp_df[~dbp_df["is_te"] & ~dbp_df["is_noncanonical"]]
dbp_ids = set(dbp_df["uniprot_id"])

# AM lookup table
am_lu = pd.read_csv(
    "Csvs/alphamissense_dbp.csv",
    usecols=["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"],
)
_p = am_lu["protein_variant"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
am_lu["ref_aa"] = _p[0]
am_lu["pos"]    = pd.to_numeric(_p[1], errors="coerce")
am_lu["alt_aa"] = _p[2]
am_lu = am_lu.rename(columns={"uniprot_id": "uniprot"})

# M3D lookup table
m3d_lu = pd.read_csv(
    "Data/M3D_results.csv",
    usecols=["uniprot", "position", "wildtype", "mutant", "damaging"],
)
m3d_lu = m3d_lu[m3d_lu["uniprot"].isin(dbp_ids)].copy()


def load_cv_location(path, label):
    df = pd.read_csv(
        path,
        usecols=["#AlleleID", "uniprot", "canonical_residue",
                 "residue_seq", "modified_residue", "sig_group"],
    )
    df = df[df["uniprot"].isin(dbp_ids)].copy()
    df["ref_aa"]   = df["canonical_residue"].map(aa3_to_1)
    df["alt_aa"]   = df["modified_residue"].map(aa3_to_1)
    df["pos"]      = pd.to_numeric(df["residue_seq"], errors="coerce")
    df["location"] = label
    return df


cv_in_loc  = load_cv_location("Csvs/clinvar_domain_hits.csv",         "In-domain")
cv_out_loc = load_cv_location("Csvs/clinvar_out_of_domain_hits.csv",  "Out-of-domain")


def add_tool_preds(cv_df):
    """Merge AM and M3D predictions onto a ClinVar dataframe."""
    merged = cv_df.merge(
        am_lu[["uniprot", "ref_aa", "pos", "alt_aa", "am_pathogenicity", "am_class"]],
        on=["uniprot", "ref_aa", "pos", "alt_aa"],
        how="left",
    ).drop_duplicates(subset=["#AlleleID"])
    merged = merged.merge(
        m3d_lu,
        left_on=["uniprot", "pos", "ref_aa", "alt_aa"],
        right_on=["uniprot", "position", "wildtype", "mutant"],
        how="left",
    ).drop_duplicates(subset=["#AlleleID"])
    return merged


in_loc  = add_tool_preds(cv_in_loc)
out_loc = add_tool_preds(cv_out_loc)


def perf_location(df, tool, pred_col, pos_val):
    sub = df[df["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])].copy()
    n_total  = len(sub)
    has_pred = (sub[pred_col].isin(["pathogenic", "benign"])
                if tool == "AM" else sub[pred_col].notna())
    coverage = has_pred.mean()
    sub = sub[has_pred].copy()
    tp, fp, tn, fn = confusion(
        (sub["sig_group"] == "Pathogenic/LP").astype(int),
        (sub[pred_col] == pos_val).astype(int),
    )
    m = metrics(tp, fp, tn, fn)
    return dict(tool=tool, TP=tp, FP=fp, TN=tn, FN=fn,
                n=tp+fp+tn+fn, n_total=n_total, coverage=coverage, **m)


rows_loc = []
for tool, pred_col, pos_val in [("AM", "am_class", "pathogenic"), ("M3D", "damaging", "Y")]:
    for label, df in [("In-domain", in_loc), ("Out-of-domain", out_loc)]:
        row = perf_location(df, tool, pred_col, pos_val)
        row["location"] = label
        rows_loc.append(row)

loc_df = pd.DataFrame(rows_loc)

print("\n" + "="*70)
print("5. Tool performance: in-domain vs out-of-domain")
print("   (ClinVar P/LP = positive, B/LB = negative; AM ambiguous excluded)")
print("="*70)
for _, row in loc_df.iterrows():
    print(f"\n  {row['tool']} — {row['location']}")
    print(f"    n evaluated : {row['n']:,}  (coverage {row['coverage']*100:.1f}% of P/LP+B/LB)")
    print(f"    TP={row['TP']}  FP={row['FP']}  TN={row['TN']}  FN={row['FN']}")
    for k in ["sensitivity", "specificity", "accuracy", "mcc"]:
        print(f"    {k:<14} {row[k]:.3f}")

print("\n--- Fisher's exact: sensitivity and specificity in vs out ---")
for tool in ["AM", "M3D"]:
    r_in  = loc_df[(loc_df["tool"] == tool) & (loc_df["location"] == "In-domain")].iloc[0]
    r_out = loc_df[(loc_df["tool"] == tool) & (loc_df["location"] == "Out-of-domain")].iloc[0]
    _, p_sens = fisher_exact([[r_in["TP"], r_in["FN"]], [r_out["TP"], r_out["FN"]]])
    _, p_spec = fisher_exact([[r_in["TN"], r_in["FP"]], [r_out["TN"], r_out["FP"]]])
    print(f"\n  {tool}")
    print(f"    Sensitivity: in={r_in['sensitivity']*100:.1f}%  "
          f"out={r_out['sensitivity']*100:.1f}%  p={p_sens:.2e}")
    print(f"    Specificity: in={r_in['specificity']*100:.1f}%  "
          f"out={r_out['specificity']*100:.1f}%  p={p_spec:.2e}")

# ── Figure: 2-panel grouped bar chart ───────────────────────
METRICS_PLOT   = ["sensitivity", "specificity", "accuracy", "mcc"]
METRIC_LABELS_ = ["Sensitivity", "Specificity", "Accuracy", "MCC"]
LOC_COLOURS    = {"In-domain": "#d62728", "Out-of-domain": "#2171b5"}

fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
x     = np.arange(len(METRICS_PLOT))
width = 0.35

for ax, tool in zip(axes5, ["AM", "M3D"]):
    for i, (loc, col) in enumerate(LOC_COLOURS.items()):
        row  = loc_df[(loc_df["tool"] == tool) & (loc_df["location"] == loc)].iloc[0]
        vals = [row[m] * 100 for m in METRICS_PLOT]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=loc, color=col, alpha=0.82)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
    tool_name = "AlphaMissense" if tool == "AM" else "Missense3D"
    n_in  = loc_df[(loc_df["tool"] == tool) & (loc_df["location"] == "In-domain")].iloc[0]["n"]
    n_out = loc_df[(loc_df["tool"] == tool) & (loc_df["location"] == "Out-of-domain")].iloc[0]["n"]
    ax.set_title(f"{tool_name}\n(in n={n_in:,}, out n={n_out:,})", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS_, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Performance (%)" if ax is axes5[0] else "")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

fig5.suptitle(
    "Tool performance: in-domain vs out-of-domain\n"
    "(ClinVar P/LP = positive, B/LB = negative; AM ambiguous excluded)",
    fontsize=11, y=1.02,
)
plt.tight_layout()
plt.savefig("Figures/tool_performance_inout_domain.png", dpi=150, bbox_inches="tight")
print("\nSaved Figures/tool_performance_inout_domain.png")


# ── ROC curve: in-domain vs out-of-domain ───────────────────
def roc_for_subset(df):
    sub = df[df["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])].copy()
    sub = sub[sub["am_pathogenicity"].notna()].copy()
    y_true = (sub["sig_group"] == "Pathogenic/LP").astype(int)
    auc = roc_auc_score(y_true, sub["am_pathogenicity"])
    fpr, tpr, _ = roc_curve(y_true, sub["am_pathogenicity"])
    return fpr, tpr, auc, len(sub)

fpr_in,  tpr_in,  auc_in,  n_in  = roc_for_subset(in_loc)
fpr_out, tpr_out, auc_out, n_out = roc_for_subset(out_loc)

print(f"\n  AUC-ROC in-domain     (n={n_in:,}): {auc_in:.3f}")
print(f"  AUC-ROC out-of-domain (n={n_out:,}): {auc_out:.3f}")

fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
ax_roc.plot(fpr_in,  tpr_in,  color=LOC_COLOURS["In-domain"],
            lw=2, label=f"In-domain  AUC = {auc_in:.3f}  (n={n_in:,})")
ax_roc.plot(fpr_out, tpr_out, color=LOC_COLOURS["Out-of-domain"],
            lw=2, label=f"Out-of-domain  AUC = {auc_out:.3f}  (n={n_out:,})")
ax_roc.plot([0, 1], [0, 1], color="#aaa", linestyle="--", linewidth=1)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(fontsize=9)
ax_roc.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/am_roc_curve_inout.png", dpi=150, bbox_inches="tight")
print("Saved Figures/am_roc_curve_inout.png")


# ============================================================
# 6. M3D performance stratified by predicted mechanism
#    Mechanism flags are only Y when damaging=Y, so per-mechanism
#    performance is expressed as precision (PPV): of all variants
#    M3D flags via mechanism X, what fraction are truly P/LP?
# ============================================================
MECHANISMS = [
    "disulphide_breakage", "buried_Pro_introduced", "clash",
    "buried_hydrophilic_introduced", "buried_charge_introduced",
    "secondary_structure_altered", "buried_charge_switch",
    "disallowed_phi_psi", "buried_charge_replaced", "buried_Gly_replaced",
    "buried_H_bond_breakage", "buried_salt_bridge_breakage",
    "cavity_altered", "buried_exposed_switch", "cis_pro_replaced",
    "gly_in_a_bend",
]

MECH_LABELS = {
    "disulphide_breakage":          "Disulphide breakage",
    "buried_Pro_introduced":        "Buried Pro introduced",
    "clash":                        "Clash",
    "buried_hydrophilic_introduced":"Buried hydrophilic introduced",
    "buried_charge_introduced":     "Buried charge introduced",
    "secondary_structure_altered":  "Secondary structure altered",
    "buried_charge_switch":         "Buried charge switch",
    "disallowed_phi_psi":           "Disallowed φ/ψ",
    "buried_charge_replaced":       "Buried charge replaced",
    "buried_Gly_replaced":          "Buried Gly replaced",
    "buried_H_bond_breakage":       "Buried H-bond breakage",
    "buried_salt_bridge_breakage":  "Buried salt-bridge breakage",
    "cavity_altered":               "Cavity altered",
    "buried_exposed_switch":        "Buried/exposed switch",
    "cis_pro_replaced":             "Cis-Pro replaced",
    "gly_in_a_bend":                "Gly in a bend",
}

MIN_N = 5  # exclude mechanisms with too few variants for reliable precision

gt_m3d = gt[gt["damaging"].isin(["Y", "N"])].copy()
gt_m3d["is_plp"] = (gt_m3d["sig_group"] == "Pathogenic/LP").astype(int)

overall_prec = gt_m3d["is_plp"][gt_m3d["damaging"] == "Y"].mean()

mech_rows = []
for mech in MECHANISMS:
    flagged = gt_m3d[gt_m3d[mech] == "Y"]
    n = len(flagged)
    if n < MIN_N:
        continue
    plp = flagged["is_plp"].sum()
    precision = plp / n
    mech_rows.append(dict(
        mechanism=mech,
        label=MECH_LABELS[mech],
        n=n,
        plp=plp,
        blb=n - plp,
        precision=precision,
    ))

mech_df = pd.DataFrame(mech_rows).sort_values("precision", ascending=True)

print("\n" + "="*70)
print("6. M3D precision by predicted mechanism (mechanism flag = Y)")
print(f"   Overall M3D precision (all damaging=Y): {overall_prec:.3f}")
print(f"   Excluded mechanisms with n < {MIN_N}")
print("="*70)
print(f"\n  {'Mechanism':<35} {'n':>5}  {'P/LP':>5}  {'B/LB':>5}  {'Precision':>10}")
print("  " + "-" * 65)
for _, r in mech_df.sort_values("precision", ascending=False).iterrows():
    _, p_mech = fisher_exact([[r["plp"], r["blb"]],
                               [int(gt_m3d["is_plp"].sum()) - r["plp"],
                                int((gt_m3d["is_plp"] == 0).sum()) - r["blb"]]])
    print(f"  {r['label']:<35} {r['n']:>5}  {r['plp']:>5}  {r['blb']:>5}  {r['precision']:>10.3f}")

fig6, ax6 = plt.subplots(figsize=(8, max(4, len(mech_df) * 0.5 + 1.5)))

colors = ["#d62728" if p < overall_prec else "#2171b5" for p in mech_df["precision"]]
bars6 = ax6.barh(mech_df["label"], mech_df["precision"] * 100,
                 color=colors, alpha=0.82, height=0.65)

ax6.axvline(overall_prec * 100, color="#555", linestyle="--", linewidth=1.2,
            label=f"Overall M3D precision ({overall_prec*100:.1f}%)")

for bar, (_, r) in zip(bars6, mech_df.iterrows()):
    ax6.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
             f"n={r['n']}", va="center", fontsize=8, color="#444")

ax6.set_xlabel("Precision — % of flagged variants truly P/LP")
ax6.set_xlim(0, 115)
ax6.set_title(
    "M3D precision by predicted mechanism\n"
    "(blue = above overall precision, red = below; mechanism flags only set for damaging=Y)",
    fontsize=10,
)
ax6.legend(fontsize=9)
ax6.spines[["top", "right"]].set_visible(False)
ax6.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
ax6.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/m3d_precision_by_mechanism.png", dpi=150, bbox_inches="tight")
print("\nSaved Figures/m3d_precision_by_mechanism.png")


# ============================================================
# 7. M3D and AM performance stratified by RSA zone
#    (buried < 0.25 | intermediate 0.25–0.75 | exposed >= 0.75)
#    Uses the merged CSV which carries rsa, plddt, in_domain.
#    Restricted to in-domain, pLDDT >= 70, P/LP or B/LB.
# ============================================================
PLDDT_THRESH = 70

rsa_base = pd.read_csv("Csvs/clinvar_am_m3d_dssp_merged.csv")
rsa_base = rsa_base[
    rsa_base["in_domain"]
    & (rsa_base["plddt"] >= PLDDT_THRESH)
    & rsa_base["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])
    & rsa_base["rsa"].notna()
].copy()
rsa_base["true_pos"] = (rsa_base["sig_group"] == "Pathogenic/LP").astype(int)

def rsa_zone(rsa):
    if rsa < 0.25:
        return "Buried"
    if rsa < 0.75:
        return "Intermediate"
    return "Exposed"

rsa_base["zone"] = rsa_base["rsa"].map(rsa_zone)

ZONES      = ["Buried", "Intermediate", "Exposed"]
ZONE_COLS  = {"Buried": "#9e7fbd", "Intermediate": "#555555", "Exposed": "#e8a87c"}
METRIC_KEYS_7 = ["sensitivity", "specificity", "mcc"]
METRIC_LABELS_7 = ["Sensitivity", "Specificity", "MCC"]

def perf_zone(df, tool, pred_col, pos_val):
    """Metrics for one tool × one RSA zone."""
    if tool == "AM":
        sub = df[df[pred_col].isin(["pathogenic", "benign"])].copy()
    else:
        sub = df[df[pred_col].isin(["Y", "N"])].copy()
    if len(sub) == 0:
        return None
    y_true = sub["true_pos"]
    y_pred = (sub[pred_col] == pos_val).astype(int)
    tp, fp, tn, fn = confusion(y_true, y_pred)
    m = metrics(tp, fp, tn, fn)
    return dict(tool=tool, TP=tp, FP=fp, TN=tn, FN=fn, n=tp+fp+tn+fn, **m)

zone_rows = []
for zone in ZONES:
    z_df = rsa_base[rsa_base["zone"] == zone]
    for tool, pred_col, pos_val in [("AM", "am_class", "pathogenic"), ("M3D", "damaging", "Y")]:
        r = perf_zone(z_df, tool, pred_col, pos_val)
        if r:
            r["zone"] = zone
            zone_rows.append(r)

zone_df = pd.DataFrame(zone_rows)

print("\n" + "=" * 75)
print("7. M3D and AM performance stratified by RSA zone")
print(f"   (in-domain, pLDDT >= {PLDDT_THRESH}, ClinVar P/LP vs B/LB)")
print("=" * 75)
for tool in ["AM", "M3D"]:
    print(f"\n  {tool}")
    t_df = zone_df[zone_df["tool"] == tool]
    print(f"  {'Zone':<14} {'n':>6}  {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  "
          f"{'Sens':>7} {'Spec':>7} {'Acc':>7} {'MCC':>7}")
    print("  " + "-" * 72)
    for zone in ZONES:
        row = t_df[t_df["zone"] == zone]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {zone:<14} {r['n']:>6}  {r['TP']:>5} {r['FP']:>5} {r['TN']:>5} {r['FN']:>5}  "
              f"{r['sensitivity']:>7.3f} {r['specificity']:>7.3f} {r['accuracy']:>7.3f} {r['mcc']:>7.3f}")
    print()

# Pairwise Fisher's exact: all 3 zone pairs × sensitivity + specificity = 6 tests
# Bonferroni correction: alpha_corr = 0.05 / 6
N_ZONE_TESTS  = 6
ALPHA_CORR    = 0.05 / N_ZONE_TESTS   # 0.00833...
ZONE_PAIRS    = [("Buried", "Intermediate"), ("Buried", "Exposed"), ("Intermediate", "Exposed")]

print(f"  Fisher's exact (pairwise zone comparisons, M3D, Bonferroni n={N_ZONE_TESTS}, α_corr={ALPHA_CORR:.4f})")
print(f"  {'Comparison':<30} {'Metric':>12}  {'Zone A %':>9}  {'Zone B %':>9}  {'p_raw':>10}  {'p_bonf':>10}  sig")
print("  " + "-" * 90)

m3d_z = zone_df[zone_df["tool"] == "M3D"]
for zone_a, zone_b in ZONE_PAIRS:
    ra = m3d_z[m3d_z["zone"] == zone_a]
    rb = m3d_z[m3d_z["zone"] == zone_b]
    if ra.empty or rb.empty:
        continue
    ra, rb = ra.iloc[0], rb.iloc[0]
    _, p_sens = fisher_exact([[ra["TP"], ra["FN"]], [rb["TP"], rb["FN"]]])
    _, p_spec = fisher_exact([[ra["TN"], ra["FP"]], [rb["TN"], rb["FP"]]])
    p_sens_bonf = min(p_sens * N_ZONE_TESTS, 1.0)
    p_spec_bonf = min(p_spec * N_ZONE_TESTS, 1.0)
    label = f"{zone_a} vs {zone_b}"
    for metric, pct_a, pct_b, p_raw, p_bonf in [
        ("Sensitivity", ra["sensitivity"]*100, rb["sensitivity"]*100, p_sens, p_sens_bonf),
        ("Specificity", ra["specificity"]*100, rb["specificity"]*100, p_spec, p_spec_bonf),
    ]:
        sig = "*" if p_bonf < 0.05 else ""
        print(f"  {label:<30} {metric:>12}  {pct_a:>8.1f}%  {pct_b:>8.1f}%  "
              f"{p_raw:>10.2e}  {p_bonf:>10.2e}  {sig}")

print()

# Bias check: among ALL M3D-assessed variants (any ClinVar sig),
# what fraction of buried vs exposed are called damaging?
rsa_all_m3d = pd.read_csv("Csvs/clinvar_am_m3d_dssp_merged.csv")
rsa_all_m3d = rsa_all_m3d[
    rsa_all_m3d["in_domain"]
    & (rsa_all_m3d["plddt"] >= PLDDT_THRESH)
    & rsa_all_m3d["damaging"].isin(["Y", "N"])
    & rsa_all_m3d["rsa"].notna()
].copy()
rsa_all_m3d["zone"] = rsa_all_m3d["rsa"].map(rsa_zone)

print("  M3D damaging-call rate by RSA zone (all ClinVar sig groups)")
print(f"  {'Zone':<14} {'n total':>9}  {'n damaging':>11}  {'% damaging':>11}")
print("  " + "-" * 50)
for zone in ZONES:
    z_all = rsa_all_m3d[rsa_all_m3d["zone"] == zone]
    n_tot = len(z_all)
    n_dam = int((z_all["damaging"] == "Y").sum())
    pct   = 100 * n_dam / n_tot if n_tot > 0 else float("nan")
    print(f"  {zone:<14} {n_tot:>9}  {n_dam:>11}  {pct:>10.1f}%")
print()

# ── Figure: grouped bar per zone, one panel per metric ──────
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

FONT_PROP = {"fontname": "Arial"}

# n per zone (AM and M3D share the same denominator within zone_df)
zone_n = {zone: int(zone_df[(zone_df["tool"] == "AM") & (zone_df["zone"] == zone)]["n"].values[0])
          for zone in ZONES}
x_labels = [f"{z}\n(n={zone_n[z]})" for z in ZONES]

AM_COLOR_7  = "#b0b0b0"       # light grey, solid (clear)
M3D_COLOR_7 = "#777777"       # darker grey, hatched

fig7, axes7 = plt.subplots(1, len(METRIC_KEYS_7), figsize=(11, 5), sharey=False)

x     = np.arange(len(ZONES))
width = 0.35

for ax, metric, label in zip(axes7, METRIC_KEYS_7, METRIC_LABELS_7):
    for i, (tool, col, hatch) in enumerate([("AM", AM_COLOR_7, ""), ("M3D", M3D_COLOR_7, "///")]):
        vals = []
        for zone in ZONES:
            row = zone_df[(zone_df["tool"] == tool) & (zone_df["zone"] == zone)]
            vals.append(row.iloc[0][metric] * 100 if not row.empty else 0)
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=tool, color=col, hatch=hatch,
               edgecolor="#888888" if hatch == "" else "#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, **FONT_PROP)
    ax.set_title(label, fontsize=10, **FONT_PROP)
    ax.set_ylim(-10 if metric == "mcc" else 0, 105)
    if metric == "mcc":
        ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.set_ylabel("%", fontsize=9, **FONT_PROP) if ax is axes7[0] else ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontsize(9)

legend_handles = [
    mpatches.Patch(facecolor=AM_COLOR_7,  edgecolor="#888888", linewidth=0.8, label="AM"),
    mpatches.Patch(facecolor=M3D_COLOR_7, edgecolor="#444444", linewidth=0.8, hatch="///", label="M3D"),
]
fig7.legend(handles=legend_handles, loc="upper center", ncol=2,
            fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.04),
            prop={"family": "Arial", "size": 9})

plt.tight_layout()
plt.savefig("Figures/tool_performance_rsa_zone.png", dpi=150, bbox_inches="tight")
print("Saved Figures/tool_performance_rsa_zone.png")


# ============================================================
# 8. M3D and AM performance stratified by structural class
#    Uses the merged CSV (P/LP vs B/LB, structural_class not null).
#    Only classes where M3D has ≥ MIN_N_CLASS ground-truth variants
#    are shown; AM is plotted for the same class set.
# ============================================================
MIN_N_CLASS = 30

CLASS_SHORT = {
    "Homeodomain-like superfamily":
        "Homeodomain-like",
    "p53-like transcription factor, DNA-binding domain superfamily":
        "p53-like TF",
    "Winged helix-like DNA-binding domain superfamily":
        "Winged helix-like",
    "High mobility group box domain superfamily":
        "HMG box",
    "Nuclear hormone receptor-like domain superfamily":
        "NHR-like",
    "Hepatocyte nuclear factor 1, N-terminal domain superfamily":
        "HNF1 N-terminal",
    "Paired domain superfamily":
        "Paired domain",
    "CTF transcription factor/nuclear factor 1, DNA-binding domain":
        "CTF/NF1",
    "IRF domain superfamily":
        "IRF domain",
}

def shorten_class(name):
    return CLASS_SHORT.get(name, name.split(",")[0].replace(" superfamily", ""))

sc_base = pd.read_csv("Csvs/clinvar_am_m3d_dssp_merged.csv")
sc_base = sc_base[
    sc_base["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])
    & sc_base["structural_class"].notna()
].copy()
sc_base["true_pos"] = (sc_base["sig_group"] == "Pathogenic/LP").astype(int)

# Identify classes with sufficient M3D ground-truth coverage
m3d_sc_all = sc_base[sc_base["damaging"].isin(["Y", "N"])].copy()
m3d_sc_all["pred"] = (m3d_sc_all["damaging"] == "Y").astype(int)
valid_classes_8 = (
    m3d_sc_all["structural_class"]
    .value_counts()
    .pipe(lambda s: s[s >= MIN_N_CLASS])
    .index.tolist()
)

sc_rows_8 = []
for tool, pred_col, pos_val in [("M3D", "damaging", "Y"), ("AM", "am_class", "pathogenic")]:
    if tool == "AM":
        sub_df = sc_base[sc_base["am_class"].isin(["pathogenic", "benign"])].copy()
    else:
        sub_df = m3d_sc_all.copy()
    for cls in valid_classes_8:
        sub = sub_df[sub_df["structural_class"] == cls]
        if len(sub) < 5:
            continue
        y_true = sub["true_pos"]
        y_pred = (sub[pred_col] == pos_val).astype(int)
        tp, fp, tn, fn = confusion(y_true, y_pred)
        m = metrics(tp, fp, tn, fn)
        sc_rows_8.append(dict(
            tool=tool, structural_class=cls, short_label=shorten_class(cls),
            n=len(sub), TP=tp, FP=fp, TN=tn, FN=fn, **m,
        ))

sc_df_8 = pd.DataFrame(sc_rows_8)

print("\n" + "=" * 75)
print("8. M3D and AM performance stratified by structural class")
print(f"   (ClinVar P/LP vs B/LB; classes with M3D n ≥ {MIN_N_CLASS})")
print("=" * 75)
for tool in ["M3D", "AM"]:
    t_df = sc_df_8[sc_df_8["tool"] == tool].sort_values("sensitivity", ascending=False)
    print(f"\n  {tool}")
    print(f"  {'Class':<22} {'n':>5}  {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  "
          f"{'Sens':>7} {'Spec':>7} {'MCC':>7}")
    print("  " + "-" * 72)
    for _, r in t_df.iterrows():
        print(f"  {r['short_label']:<22} {r['n']:>5}  {r['TP']:>5} {r['FP']:>5} "
              f"{r['TN']:>5} {r['FN']:>5}  "
              f"{r['sensitivity']:>7.3f} {r['specificity']:>7.3f} {r['mcc']:>7.3f}")

# Fisher's exact: each class vs rest of valid-class M3D variants
m3d_valid_all = m3d_sc_all[m3d_sc_all["structural_class"].isin(valid_classes_8)]
all_tp8 = int(((m3d_valid_all["true_pos"] == 1) & (m3d_valid_all["pred"] == 1)).sum())
all_fn8 = int(((m3d_valid_all["true_pos"] == 1) & (m3d_valid_all["pred"] == 0)).sum())
all_tn8 = int(((m3d_valid_all["true_pos"] == 0) & (m3d_valid_all["pred"] == 0)).sum())
all_fp8 = int(((m3d_valid_all["true_pos"] == 0) & (m3d_valid_all["pred"] == 1)).sum())

m3d_8 = sc_df_8[sc_df_8["tool"] == "M3D"].sort_values("sensitivity", ascending=False)
n_tests_8 = len(m3d_8) * 2
alpha_8 = 0.05 / n_tests_8

print(f"\n  M3D — Fisher's exact: each class vs rest (Bonferroni n={n_tests_8}, α_corr={alpha_8:.4f})")
print(f"  {'Class':<22} {'Metric':>12}  {'Class %':>9}  {'Rest %':>9}  "
      f"{'p_raw':>10}  {'p_bonf':>10}  sig")
print("  " + "-" * 90)

for _, r in m3d_8.iterrows():
    rest_tp = all_tp8 - r["TP"];  rest_fn = all_fn8 - r["FN"]
    rest_tn = all_tn8 - r["TN"];  rest_fp = all_fp8 - r["FP"]
    _, p_sens = fisher_exact([[r["TP"], r["FN"]], [rest_tp, rest_fn]])
    _, p_spec = fisher_exact([[r["TN"], r["FP"]], [rest_tn, rest_fp]])
    p_sens_b = min(p_sens * n_tests_8, 1.0)
    p_spec_b = min(p_spec * n_tests_8, 1.0)
    rest_sens = rest_tp / (rest_tp + rest_fn) if (rest_tp + rest_fn) > 0 else float("nan")
    rest_spec = rest_tn / (rest_tn + rest_fp) if (rest_tn + rest_fp) > 0 else float("nan")
    for metric, cls_pct, rest_pct, p_raw, p_bonf in [
        ("Sensitivity", r["sensitivity"] * 100, rest_sens * 100, p_sens, p_sens_b),
        ("Specificity", r["specificity"] * 100, rest_spec * 100, p_spec, p_spec_b),
    ]:
        sig = "*" if p_bonf < 0.05 else ""
        print(f"  {r['short_label']:<22} {metric:>12}  {cls_pct:>8.1f}%  {rest_pct:>8.1f}%  "
              f"{p_raw:>10.2e}  {p_bonf:>10.2e}  {sig}")

# ── Figure: horizontal grouped bars, one panel per metric ───
METRICS_8       = ["sensitivity", "specificity", "mcc"]
METRIC_LABELS_8 = ["Sensitivity", "Specificity", "MCC"]
AM_COLOR_8      = "#b0b0b0"
M3D_COLOR_8     = "#777777"

# Sort classes by M3D sensitivity (lowest at top of chart)
order_8 = (
    m3d_8.sort_values("sensitivity", ascending=True)["structural_class"].tolist()
)
short_labels_8 = [shorten_class(c) for c in order_8]
y_pos_8 = np.arange(len(order_8))
height_8 = 0.35
n_fig_rows = len(order_8)

fig8, axes8 = plt.subplots(
    1, len(METRICS_8),
    figsize=(13, max(4, n_fig_rows * 0.6 + 2)),
    sharey=True,
)

for ax, metric, mlabel in zip(axes8, METRICS_8, METRIC_LABELS_8):
    for i, (tool, col, hatch) in enumerate([("AM", AM_COLOR_8, ""), ("M3D", M3D_COLOR_8, "///")]):
        vals = []
        for cls in order_8:
            row = sc_df_8[(sc_df_8["tool"] == tool) & (sc_df_8["structural_class"] == cls)]
            vals.append(row.iloc[0][metric] * 100 if not row.empty else np.nan)
        ax.barh(y_pos_8 + (i - 0.5) * height_8, vals, height_8,
                label=tool, color=col, hatch=hatch,
                edgecolor="#888888" if hatch == "" else "#444444", linewidth=0.8)
    ax.set_yticks(y_pos_8)
    ax.set_yticklabels(short_labels_8 if ax is axes8[0] else [], fontsize=9, **FONT_PROP)
    ax.set_title(mlabel, fontsize=10, **FONT_PROP)
    if metric == "mcc":
        ax.axvline(0, color="black", linewidth=0.8, zorder=0)
        ax.set_xlim(-30, 110)
    else:
        ax.set_xlim(0, 115)
    ax.set_xlabel("%", fontsize=9, **FONT_PROP)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontsize(9)

legend_handles_8 = [
    mpatches.Patch(facecolor=AM_COLOR_8,  edgecolor="#888888", linewidth=0.8, label="AM"),
    mpatches.Patch(facecolor=M3D_COLOR_8, edgecolor="#444444", linewidth=0.8,
                   hatch="///", label="M3D"),
]
fig8.legend(handles=legend_handles_8, loc="upper center", ncol=2,
            fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.04),
            prop={"family": "Arial", "size": 9})

fig8.suptitle(
    f"M3D and AM performance by structural class\n"
    f"(ClinVar P/LP vs B/LB; M3D n ≥ {MIN_N_CLASS} per class; sorted by M3D sensitivity)",
    fontsize=11, y=1.08,
)
plt.tight_layout()
plt.savefig("Figures/tool_performance_structural_class.png", dpi=150, bbox_inches="tight")
print("\nSaved Figures/tool_performance_structural_class.png")
