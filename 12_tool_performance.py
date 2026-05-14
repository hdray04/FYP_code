import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================
# Tool performance: AM and M3D vs ClinVar ground truth
#
# Ground truth: ClinVar P/LP (positive) vs B/LB (negative).
# VUS and Other/Conflicting are excluded — their label is
# genuinely uncertain, so they cannot ground a confusion matrix.
#
# This gives per-tool sensitivity, specificity, accuracy, PPV,
# NPV, and MCC, plus an ROC curve for AM (continuous score).
# Run before discordance analyses so every discordance finding
# can be contextualised against each tool's baseline performance.
# ============================================================

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
    ppv    = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv    = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc    = (tp + tn) / total if total > 0 else np.nan
    denom  = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc    = (tp*tn - fp*fn) / denom if denom > 0 else np.nan
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                accuracy=acc, mcc=mcc)


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
metric_keys = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "mcc"]
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
METRIC_LABELS = ["Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "MCC"]
METRIC_KEYS   = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "mcc"]
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
im3 = ax3.imshow(cm_m3, cmap="Reds", vmin=0)
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

# ── 4d. ROC curve (AM continuous score) ──────────────────────
fig2, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_am, tpr_am, color=AM_COLOR, lw=2,
        label=f"AlphaMissense AUC = {auc_am:.3f}")
ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve: AlphaMissense vs ClinVar P/LP ground truth")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Figures/am_roc_curve.png", dpi=150, bbox_inches="tight")
print("Saved Figures/am_roc_curve.png")
