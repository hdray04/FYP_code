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
