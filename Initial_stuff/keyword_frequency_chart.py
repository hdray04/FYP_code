import json
from collections import Counter
import matplotlib.pyplot as plt

import numpy as np

# Load UniProt data
with open('./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json') as f:
    data = json.load(f)

proteins = data['results']
n_total  = len(proteins)

# ── Step 1: count disease keywords ───────────────────────────────────────────
# For each protein we collect the SET of disease keyword names it carries, so
# a keyword scores 1 per protein regardless of how many variants are annotated.
# We also count how many proteins have ANY disease keyword at all.

disease_keyword_count = Counter()
n_disease_annotated   = 0

for protein in proteins:
    disease_kws = {
        kw['name']
        for kw in protein.get('keywords', [])
        if kw['category'] == 'Disease'
    }
    if disease_kws:
        n_disease_annotated += 1
    for name in disease_kws:
        disease_keyword_count[name] += 1

# Keep only functional disease flags (not named diseases)
FUNCTIONAL_FLAGS = {'Disease variant', 'Proto-oncogene', 'Tumor suppressor'}
filtered = [(kw, cnt) for kw, cnt in disease_keyword_count.most_common()
            if kw in FUNCTIONAL_FLAGS]
labels, counts = zip(*filtered)

# ── Step 2: colour coding ─────────────────────────────────────────────────────
bar_colours = ['#E53935', '#FF7043', '#78909C'][:len(labels)]

pct_of_total    = [c / n_total              * 100 for c in counts]
pct_of_diseased = [c / n_disease_annotated  * 100 for c in counts]

# ── Step 3: plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

y_pos = np.arange(len(labels))
bars  = ax.barh(y_pos, counts, color=bar_colours, edgecolor='white', linewidth=0.5)

for bar, cnt, p_all, p_dis in zip(bars, counts, pct_of_total, pct_of_diseased):
    ax.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f'{cnt}   {p_all:.0f}% of all  |  {p_dis:.0f}% of disease-annotated',
        va='center', ha='left', fontsize=8, color='#333333'
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel('Number of proteins', fontsize=11)
ax.set_title(
    f'Disease annotations across {n_total} human DNA-binding proteins\n'
    f'{n_disease_annotated} proteins ({n_disease_annotated/n_total*100:.0f}%) '
    f'carry at least one disease keyword',
    fontsize=12, fontweight='bold', pad=12
)
ax.set_xlim(0, max(counts) * 1.85)
ax.spines[['top', 'right']].set_visible(False)

ax.legend(handles=[], frameon=False)  # no legend needed with only 3 bars

plt.tight_layout()
plt.savefig('disease_keyword_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: disease_keyword_breakdown.png")

# ── Step 4: print table ───────────────────────────────────────────────────────
print(f"\n{'Disease keyword':<40} {'N':>5}  {'% all':>6}  {'% diseased':>10}")
print('-' * 68)
for kw, cnt in filtered:
    flag = ' *' if kw in FUNCTIONAL_FLAGS else ''
    print(f"{kw+flag:<40} {cnt:>5}  {cnt/n_total*100:>5.1f}%  "
          f"{cnt/n_disease_annotated*100:>9.1f}%")
print(f"\n* functional flag, not a named disease")
print(f"\nTotal disease-annotated: {n_disease_annotated}/{n_total} "
      f"({n_disease_annotated/n_total*100:.1f}%)")
