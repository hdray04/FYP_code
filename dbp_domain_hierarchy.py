"""
DBP Domain Classification via InterPro Parent-Child Hierarchy
=============================================================
Uses the official InterPro files to classify 611 DNA-binding proteins by
structural domain family:

  1. entry.list.txt  — maps every IPR id to its entry type
                       (Homologous_superfamily, Domain, Family, etc.)
  2. ParentChildTreeFile.txt — parent-child relationships between IPR entries

Classification strategy per protein:
  - Preferentially use Homologous_superfamily entries (structural fold)
  - Fall back to Domain entries, then Family entries
  - For each selected entry, walk up the hierarchy to its root ancestor
  - Assign the protein to its most frequent root

This is reproducible and citable: classification is driven entirely by
InterPro's own ontology, not hand-crafted string rules.

Usage:
    python dbp_domain_hierarchy.py
"""

import csv
import json
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────────

ENTRY_LIST     = "Data/entry.list.txt"
HIERARCHY_FILE = "Data/ParentChildTreeFile.txt"
JSON_FILE      = "data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json"

# Entry-type preference order for structural classification
TYPE_PRIORITY = ["Homologous_superfamily", "Domain", "Family"]

# Manual collapse map: any class name on the left is merged into the label on the right.
# Extend this after reviewing the full class list.
COLLAPSE_MAP = {
    # Zinc finger variants → single "Zinc finger superfamily" class
    "Zinc finger C2H2 superfamily"       : "Zinc finger superfamily",
    "Zinc finger, FYVE/PHD-type"         : "Zinc finger superfamily",
    "Zinc finger, RING/FYVE/PHD-type"    : "Zinc finger superfamily",

    # InterPro typo: their superfamily is named "Homedomain-like" (missing 'e')
    # Correct it here, and also collapse the separate "Homeodomain" Domain entry
    "Homedomain-like superfamily"        : "Homeodomain-like superfamily",
    "Homeodomain"                        : "Homeodomain-like superfamily",

    # SMAD — two separate roots in the hierarchy; both are SMAD proteins
    "SMAD/FHA domain superfamily"        : "SMAD-like domain superfamily",

    # Lambda repressor HTH is structurally homeodomain-like (SCOP/InterPro)
    "Lambda repressor-like, DNA-binding domain superfamily" : "Homeodomain-like superfamily",

    # HMGA proteins are often grouped with HMG proteins by name, but their
    # DNA-binding mechanism is via AT-hook motifs rather than HMG-box folds.
    "High mobility group protein HMGA"   : "AT-hook motif",

    # ELYS (AHCTF1) and GLYR1 (N-PAC) — both have AT-hook DNA-binding features;
    # hierarchy walks to uninformative metabolic enzyme roots due to shared folds
    "Quinoprotein alcohol dehydrogenase-like superfamily"                  : "AT-hook motif",
    "6-phosphogluconate dehydrogenase-like, C-terminal domain superfamily" : "AT-hook motif",
}

# ── Step 1: Load entry.list ──────────────────────────────────────────────────
# Gives us ENTRY_TYPE and ENTRY_NAME for every IPR id.

print("Loading entry.list.txt...")
entry_info = {}   # ipr_id -> {"type": ..., "name": ...}
with open(ENTRY_LIST) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        entry_info[row["ENTRY_AC"]] = {
            "type": row["ENTRY_TYPE"],
            "name": row["ENTRY_NAME"],
        }
print(f"  {len(entry_info):,} InterPro entries in entry.list\n")

# ── Step 2: Parse the ParentChildTreeFile ────────────────────────────────────
# Format: {--*}IPRxxxxxx::Name::
# Depth  = number of leading '--' pairs

def parse_hierarchy(filepath):
    child_to_parent = {}   # ipr_id -> parent ipr_id (None if root)
    stack = []             # (depth, ipr_id)

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            stripped = line.lstrip("-")
            depth    = (len(line) - len(stripped)) // 2
            m = re.match(r"(IPR\d+)::", stripped)
            if not m:
                continue
            ipr_id = m.group(1)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            child_to_parent[ipr_id] = stack[-1][1] if stack else None
            stack.append((depth, ipr_id))

    return child_to_parent


print("Parsing ParentChildTreeFile.txt...")
child_to_parent = parse_hierarchy(HIERARCHY_FILE)
print(f"  {len(child_to_parent):,} entries with hierarchy relationships\n")


def get_root(ipr_id, max_depth=30):
    """Walk up child_to_parent to the root. Returns root IPR id."""
    visited = set()
    current = ipr_id
    for _ in range(max_depth):
        if current not in child_to_parent:
            return current          # not in hierarchy — treat as own root
        parent = child_to_parent[current]
        if parent is None:
            return current          # this IS the root
        if parent in visited:
            return current          # cycle guard
        visited.add(current)
        current = parent
    return current


# ── Step 3: Load proteins ────────────────────────────────────────────────────

print("Loading proteins...")
with open(JSON_FILE) as f:
    data = json.load(f)
proteins = data["results"]
print(f"  {len(proteins)} proteins loaded\n")


def get_ipr_ids(protein):
    """Return list of IPR ids for a protein."""
    ids = []
    for x in protein.get("uniProtKBCrossReferences", []):
        if x.get("database") == "InterPro":
            ids.append(x["id"])
    return ids


# ── Step 4: Classify each protein ────────────────────────────────────────────

# Proteins excluded entirely — confirmed non-transcriptional DBPs
EXCLUDE_IDS = {
    "P01011",   # SERPINA3 — serine protease inhibitor; spurious DNA-binding annotation
}

# Proteins flagged as non-canonical — retained but marked in CSV
NONCANONICAL_IDS = {
    "P02751",   # FN1 (Fibronectin) — extracellular matrix protein; DNA-binding via
                # heparin-binding domain, not transcriptional regulation
}

def is_transposable_element(protein):
    """Return True if the protein carries the 'Transposable element' UniProt keyword."""
    return any(k.get("name") == "Transposable element" for k in protein.get("keywords", []))


rows = []

for p in proteins:
    pid       = p["primaryAccession"]
    gene      = (p.get("genes") or [{}])[0].get("geneName", {}).get("value", "")
    prot_name = (p.get("proteinDescription", {})
                  .get("recommendedName", {})
                  .get("fullName", {})
                  .get("value", ""))

    # Skip excluded proteins entirely
    if pid in EXCLUDE_IDS:
        continue

    # Flag transposable element proteins — kept in dataset for transparency
    # but marked so they can be excluded from downstream analyses
    if is_transposable_element(p):
        rows.append({
            "uniprot_id"         : pid,
            "gene"               : gene,
            "protein_name"       : prot_name,
            "structural_class"   : "⚠ Transposable element / ERV",
            "root_ipr"           : "TE",
            "entry_type_used"    : "excluded",
            "n_ipr_entries"      : len(get_ipr_ids(p)),
            "n_unique_roots"     : 0,
            "is_te"              : True,
            "is_noncanonical"    : False,
        })
        continue

    ipr_ids = get_ipr_ids(p)

    # Group by entry type
    by_type = {t: [] for t in TYPE_PRIORITY}
    for ipr_id in ipr_ids:
        t = entry_info.get(ipr_id, {}).get("type")
        if t in by_type:
            by_type[t].append(ipr_id)

    # Select the highest-priority type that has entries
    selected_type = None
    selected_ids  = []
    for t in TYPE_PRIORITY:
        if by_type[t]:
            selected_type = t
            selected_ids  = by_type[t]
            break

    # Walk each selected entry to its root
    if selected_ids:
        roots = [get_root(ipr_id) for ipr_id in selected_ids]
        root_counter   = Counter(roots)
        primary_root   = root_counter.most_common(1)[0][0]
        primary_name   = entry_info.get(primary_root, {}).get("name", primary_root)
        n_unique_roots = len(root_counter)
    else:
        primary_root   = "NONE"
        primary_name   = "No structural annotation"
        selected_type  = "none"
        n_unique_roots = 0

    # Apply manual collapse
    primary_name = COLLAPSE_MAP.get(primary_name, primary_name)

    rows.append({
        "uniprot_id"         : pid,
        "gene"               : gene,
        "protein_name"       : prot_name,
        "structural_class"   : primary_name,
        "root_ipr"           : primary_root,
        "entry_type_used"    : selected_type,
        "n_ipr_entries"      : len(ipr_ids),
        "n_unique_roots"     : n_unique_roots,
        "is_te"              : False,
        "is_noncanonical"    : pid in NONCANONICAL_IDS,
    })

df = pd.DataFrame(rows)

# ── Step 5: Summary ───────────────────────────────────────────────────────────

print("=" * 65)
print("STRUCTURAL CLASS SUMMARY  (top 30 by protein count)")
print("=" * 65)

class_counts = df["structural_class"].value_counts()
total        = len(df)

print(f"\n{'Class':<50} {'N':>5}  {'%':>6}")
print("-" * 65)
for cls, n in class_counts.head(30).items():
    print(f"{cls:<50} {n:>5}  {100*n/total:>5.1f}%")

te_count           = df["is_te"].sum()
noncanonical_count = df["is_noncanonical"].sum()
unannotated        = (df["root_ipr"] == "NONE").sum()
print(f"\nTotal proteins        : {total}  (SERPINA3 excluded entirely)")
print(f"  of which TE/ERV     : {te_count}  (retained in CSV, excluded from plot)")
print(f"  of which non-canonical: {noncanonical_count}  (FN1; flagged in CSV)")
print(f"Structurally annotated: {total - unannotated - te_count}")
print(f"No structural entry   : {unannotated}")

# ── Multi-domain subgroup ─────────────────────────────────────────────────────

multi = df[df["n_unique_roots"] >= 2].sort_values("n_unique_roots", ascending=False)
print(f"\n{'=' * 65}")
print(f"MULTI-DOMAIN DBPs  (≥2 distinct structural root families)")
print(f"{'=' * 65}")
print(f"{len(multi)} proteins ({100*len(multi)/total:.1f}% of {total})\n")
print("Breakdown by number of distinct structural root families:")
for n, count in df["n_unique_roots"].value_counts().sort_index().items():
    label = "no annotation" if n == 0 else f"{n} structural familie{'s' if n > 1 else ''}"
    print(f"  {n} roots  →  {count:>4} proteins  ({100*count/total:.1f}%)  [{label}]")
print()
print(f"{'Gene':<12} {'UniProt':<12} {'Roots':>6}  {'Primary class':<40}  Protein name")
print("-" * 110)
for _, row in multi.iterrows():
    print(f"{row['gene']:<12} {row['uniprot_id']:<12} {row['n_unique_roots']:>6}  "
          f"{row['structural_class']:<40}  {row['protein_name'][:50]}")

type_breakdown = df["entry_type_used"].value_counts()
print("\nEntry type used for classification:")
for t, n in type_breakdown.items():
    print(f"  {t:<25} {n}")

# ── Step 6: Save ──────────────────────────────────────────────────────────────

df.to_csv("protein_classifications_hierarchy.csv", index=False)

summary = class_counts.reset_index()
summary.columns = ["structural_class", "n_proteins"]
summary["percent"] = (summary["n_proteins"] / total * 100).round(1)
summary.to_csv("class_summary_hierarchy.csv", index=False)

print("\nFiles saved:")
print("  protein_classifications_hierarchy.csv")
print("  class_summary_hierarchy.csv")

# ── Step 7: Plot ──────────────────────────────────────────────────────────────

plot_df = (summary[~summary["structural_class"].isin(
               ["No structural annotation", "⚠ Transposable element / ERV"])]
           .head(20)
           .sort_values("n_proteins", ascending=True))

fig, ax = plt.subplots(figsize=(11, 8))
colors = plt.cm.tab20.colors
bars   = ax.barh(plot_df["structural_class"], plot_df["n_proteins"],
                 color=colors[:len(plot_df)], edgecolor="white")

for bar, (_, row) in zip(bars, plot_df.iterrows()):
    ax.text(row["n_proteins"] + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{row["n_proteins"]}  ({row["percent"]}%)',
            va="center", fontsize=8)

ax.set_xlabel("Number of proteins", fontsize=11)
ax.set_title(
    "DNA-binding proteins — structural classification\n"
    "via InterPro hierarchy (Homologous_superfamily > Domain > Family)",
    fontsize=11, fontweight="bold"
)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("domain_class_hierarchy_bar.png", dpi=150, bbox_inches="tight")
print("  domain_class_hierarchy_bar.png")
plt.show()

# ── Pie chart ─────────────────────────────────────────────────────────────────
# Top 8 classes shown individually; everything else collapsed into "Other"

EXCLUDE_FROM_PIE = {"No structural annotation", "⚠ Transposable element / ERV"}
pie_data = summary[~summary["structural_class"].isin(EXCLUDE_FROM_PIE)].copy()

top_n    = 8
top      = pie_data.head(top_n)
other_n  = pie_data.iloc[top_n:]["n_proteins"].sum()

pie_labels = list(top["structural_class"]) + [f"Other ({len(pie_data) - top_n} classes)"]
pie_values = list(top["n_proteins"])       + [other_n]

fig2, ax2 = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax2.pie(
    pie_values,
    labels=None,          # use legend instead to avoid label overlap
    autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
    colors=colors[:len(pie_labels)],
    startangle=140,
    pctdistance=0.78,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(8)

ax2.legend(
    wedges, pie_labels,
    title="Structural class",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=8,
    title_fontsize=9,
)
ax2.set_title(
    "Structural class distribution\namong 610 DNA-binding proteins",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("domain_class_hierarchy_pie.png", dpi=150, bbox_inches="tight")
print("  domain_class_hierarchy_pie.png")
plt.show()
print("\nDone.")
