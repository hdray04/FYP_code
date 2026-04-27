"""
DBP Domain Analysis
====================
Loads your UniProt JSON, classifies each protein into a structural domain
family using the InterPro short names already embedded in the file,
and produces summary tables + plots.

No API calls. No external dependencies beyond pandas + matplotlib.

Usage:
    python dbp_domain_analysis.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter, defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

JSON_FILE = "data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json"

# Mapping from InterPro short-name substrings → clean structural class label.
# Order matters: first match wins. All comparisons are lower-case.
DOMAIN_MAP = [
    # Homeodomain family (largest group — keep specific subtypes first)
    ("tale_homeobox",       "Homeodomain – TALE"),
    ("pou",                 "Homeodomain – POU"),
    ("nk-like",             "Homeodomain – NK"),
    ("antennapedia",        "Homeodomain – Antennapedia"),
    ("hox",                 "Homeodomain – Hox"),
    ("homeodomain",         "Homeodomain"),
    ("homeobox",            "Homeodomain"),
    ("hd",                  "Homeodomain"),          # short name HD / HD_metazoa

    # Winged-helix / Forkhead
    ("fork_head",           "Winged-helix / Forkhead"),
    ("forkhead",            "Winged-helix / Forkhead"),
    ("fox",                 "Winged-helix / Forkhead"),
    ("wh-like",             "Winged-helix / Forkhead"),
    ("wh_dna",              "Winged-helix / Forkhead"),

    # HMG box (includes SOX, SRY)
    ("hmg",                 "HMG box"),
    ("sry",                 "HMG box"),

    # Nuclear receptor
    ("nuclear_hrmn",        "Nuclear receptor"),
    ("nucl_hrmn",           "Nuclear receptor"),
    ("nhr",                 "Nuclear receptor"),
    ("nuclear_hormone",     "Nuclear receptor"),

    # Zinc finger subtypes
    ("znf_c2h2",            "Zinc finger – C2H2"),
    ("znf_nhr",             "Zinc finger – NHR/GATA"),
    ("znf_gata",            "Zinc finger – NHR/GATA"),
    ("gata",                "Zinc finger – NHR/GATA"),
    ("znf",                 "Zinc finger – other"),
    ("zinc_finger",         "Zinc finger – other"),

    # ETS domain
    ("ets",                 "ETS domain"),

    # Helix-turn-helix
    ("hth",                 "Helix-turn-helix"),
    ("helix-turn-helix",    "Helix-turn-helix"),
    ("lambda_dna",          "Helix-turn-helix"),

    # p53-like
    ("p53",                 "p53-like"),

    # T-box
    ("t-box",               "T-box"),
    ("t_box",               "T-box"),

    # bHLH
    ("bhlh",                "bHLH"),
    ("helix-loop-helix",    "bHLH"),

    # bZIP
    ("bzip",                "bZIP"),
    ("leucine_zipper",      "bZIP"),

    # RB / E2F
    ("e2f",                 "E2F / RB-associated"),
    ("rb",                  "E2F / RB-associated"),

    # SMAD
    ("smad",                "SMAD"),
    ("mad_homology",        "SMAD"),

    # Runt
    ("runt",                "Runt domain"),

    # Paired / PAX
    ("paired",              "Paired domain"),
    ("pax",                 "Paired domain"),

    # Transposable element — flag for exclusion
    ("integrase",           "⚠ Transposable element"),
    ("transposase",         "⚠ Transposable element"),
    ("reverse_transcriptase","⚠ Transposable element"),
    ("gag_",                "⚠ Transposable element"),
]

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading data...")
with open(JSON_FILE) as f:
    data = json.load(f)
proteins = data["results"]
print(f"  {len(proteins)} proteins loaded\n")

# ── Extract per-protein information ───────────────────────────────────────────

def get_interpro_names(protein):
    """Return list of InterPro short names for a protein."""
    xrefs = protein.get("uniProtKBCrossReferences", [])
    names = []
    for x in xrefs:
        if x.get("database") == "InterPro":
            for prop in x.get("properties", []):
                if prop["key"] == "EntryName":
                    names.append(prop["value"])
    return names


def classify_protein(interpro_names):
    """
    Given a list of InterPro short names for one protein, return its
    primary structural class label.
    Strategy: collect all matching labels across all domains, return
    the most frequent non-TE, non-Unclassified one.
    """
    matched = []
    for ipr_name in interpro_names:
        lower = ipr_name.lower()
        for keyword, label in DOMAIN_MAP:
            if keyword in lower:
                matched.append(label)
                break   # one label per domain entry

    # Separate TE hits
    te_hits    = [l for l in matched if l.startswith("⚠")]
    clean_hits = [l for l in matched if not l.startswith("⚠")]

    if clean_hits:
        return Counter(clean_hits).most_common(1)[0][0]
    elif te_hits:
        return "⚠ Transposable element"
    else:
        return "Unclassified"


# Build the main protein table
rows = []
for p in proteins:
    pid          = p["primaryAccession"]
    gene         = (p.get("genes") or [{}])[0].get("geneName", {}).get("value", "")
    prot_name    = (p.get("proteinDescription", {})
                     .get("recommendedName", {})
                     .get("fullName", {})
                     .get("value", ""))
    ipr_names    = get_interpro_names(p)
    struct_class = classify_protein(ipr_names)
    n_domains    = len(ipr_names)

    rows.append({
        "uniprot_id"     : pid,
        "gene"           : gene,
        "protein_name"   : prot_name,
        "structural_class": struct_class,
        "n_interpro_domains": n_domains,
        "interpro_names" : "; ".join(ipr_names),
    })

df = pd.DataFrame(rows)

# ── Summary ────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STRUCTURAL CLASS SUMMARY")
print("=" * 60)

class_counts = df["structural_class"].value_counts()
total = len(df)
print(f"\n{'Class':<35} {'N':>5}  {'%':>6}")
print("-" * 50)
for cls, n in class_counts.items():
    print(f"{cls:<35} {n:>5}  {100*n/total:>5.1f}%")

print(f"\nTotal proteins : {total}")
print(f"Classified     : {total - class_counts.get('Unclassified', 0) - class_counts.get('⚠ Transposable element', 0)}")
print(f"TE-flagged     : {class_counts.get('⚠ Transposable element', 0)}")
print(f"Unclassified   : {class_counts.get('Unclassified', 0)}")

# ── Save tables ────────────────────────────────────────────────────────────────

df.to_csv("Csvs/protein_classifications.csv", index=False)

summary = class_counts.reset_index()
summary.columns = ["structural_class", "n_proteins"]
summary["percent"] = (summary["n_proteins"] / total * 100).round(1)
summary.to_csv("Csvs/class_summary.csv", index=False)

print("\nFiles saved:")
print("  protein_classifications.csv  — one row per protein")
print("  class_summary.csv            — counts per structural class")

# ── Plot ───────────────────────────────────────────────────────────────────────

# Filter out TE and unclassified for the main plots
plot_df = summary[
    ~summary["structural_class"].str.startswith("⚠") &
    (summary["structural_class"] != "Unclassified")
].sort_values("n_proteins", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# --- Bar chart ---
colors = plt.cm.tab20.colors
ax1.barh(plot_df["structural_class"], plot_df["n_proteins"],
         color=colors[:len(plot_df)], edgecolor="white")
for i, (_, row) in enumerate(plot_df.iterrows()):
    ax1.text(row["n_proteins"] + 1, i, f'{row["n_proteins"]}',
             va="center", fontsize=8)
ax1.set_xlabel("Number of proteins", fontsize=11)
ax1.set_title("DNA-binding proteins\nby structural domain class", fontsize=12, fontweight="bold")
ax1.spines[["top", "right"]].set_visible(False)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# --- Pie chart (top classes + Other) ---
top_n    = 8
top      = plot_df.sort_values("n_proteins", ascending=False).head(top_n)
other_n  = plot_df.sort_values("n_proteins", ascending=False).iloc[top_n:]["n_proteins"].sum()
# Also add unclassified/TE to "Other" for completeness
other_n += class_counts.get("Unclassified", 0) + class_counts.get("⚠ Transposable element", 0)

pie_labels = list(top["structural_class"]) + ["Other / Unclassified"]
pie_values = list(top["n_proteins"])       + [other_n]

wedges, texts, autotexts = ax2.pie(
    pie_values, labels=pie_labels, autopct="%1.1f%%",
    colors=colors[:len(pie_labels)], startangle=140,
    pctdistance=0.75
)
for t in texts:
    t.set_fontsize(8)
for at in autotexts:
    at.set_fontsize(8)
ax2.set_title("Proportion of each structural class\n(611 proteins)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("domain_class_distribution.png", dpi=150, bbox_inches="tight")
print("  domain_class_distribution.png — figure")
plt.show()

print("\nDone.")
