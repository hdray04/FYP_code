# FYP: Pathogenic Variant Enrichment in Human DNA-Binding Proteins

Analysis pipeline for classifying human DNA-binding proteins (DBPs) by structural domain family and testing for enrichment of pathogenic ClinVar variants within and across those classes.

---

## Directory structure

```
FYP_code/
├── Data/                    Raw input data (not modified)
├── Csvs/                    Pipeline outputs (CSV tables)
├── Figures/                 Generated figures (PNG)
├── Initial_stuff/           Exploratory/developmental scripts (not part of final pipeline)
├── venv/                    Python virtual environment
└── 01_*.py – 08_*.py       Main analysis pipeline (run in order from repo root)
```

---

## Data sources

| File | Source | Version / download date |
|---|---|---|
| `Data/clinvar_variants_28_08_2025.csv` | ClinVar variant summary | 28 August 2025 |
| `Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json` | UniProt Swiss-Prot, human (*Homo sapiens*, taxon 9606), reviewed only | 06 March 2026 |
| `Data/entry.list.txt` | InterPro — maps every IPR accession to its entry type and name | Downloaded alongside ParentChildTreeFile |
| `Data/ParentChildTreeFile.txt` | InterPro — parent-child hierarchy of all IPR entries | Same release as entry.list.txt |

All data files are used read-only. No data file was manually edited.

---

## Pipeline overview

Run all scripts from the repository root (`FYP_code/`) so that relative paths (`Data/`, `Csvs/`) resolve correctly.

```
Step 1  01_data_checking.py                    Quality-check annotation coverage
Step 2  02_dbp_domain_hierarchy.py             Classify 611 DBPs by structural domain → protein_classifications_hierarchy.csv
Step 3  03_clinvar_domain_mapping.py           Map ClinVar variants to DNA-binding domain intervals → clinvar_domain_hits.csv
Step 4  04_clinvar_variant_landscape.py        DBP vs non-DBP enrichment; variant burden summary and skew plots
Step 5  06_clinvar_domain_class_enrichment.py  Enrichment restricted to in-domain variants by structural class
Step 6  07_enrichment_by_domain.py             [Draft — to be reviewed]
Step 7  08_sensitivity_sim_only_check.py       Sensitivity: remove similarity-only proteins, recheck ORs
```

---

## Step-by-step detail

### Step 1 — `01_data_checking.py`

**Purpose:** Quality assessment of the UniProt and ClinVar datasets before the main analysis.

**Checks performed:**
- Percentage of 611 UniProt proteins with InterPro annotations
- Percentage with Pfam annotations
- ClinVar variant mapping rate to UniProt accessions
- Review status and evidence distribution

**Key findings noted:**
- 1 protein (B2CW77 / KLLN) has neither InterPro nor Pfam annotations — classified as "No structural annotation" in Step 2
- 52 proteins have DNA-binding annotations inferred by homology only (no direct experimental evidence)

**Output:** Console report only (no CSV written).

---

### Step 2 — `02_dbp_domain_hierarchy.py`

**Purpose:** Classify all 611 human DNA-binding proteins into structural domain families using the InterPro ontology hierarchy.

**Classification strategy (per protein):**
1. Extract all InterPro cross-references from the UniProt JSON entry
2. Prioritise **Homologous_superfamily** entries (structural fold), fall back to Domain, then Family, then Conserved_site as a last resort (`TYPE_PRIORITY`)
3. For each selected InterPro entry, walk up the `ParentChildTreeFile` hierarchy to its root ancestor
4. Assign the protein the most frequently occurring root across all its entries
5. Apply `COLLAPSE_MAP` (see manual edits below)

**Output:** `Csvs/protein_classifications_hierarchy.csv` — one row per protein; columns: `uniprot_id`, `gene`, `protein_name`, `structural_class`, `root_ipr`, `entry_type_used`, `n_ipr_entries`, `n_unique_roots`, `is_te`, `is_noncanonical`

**Figures saved to `Figures/`:** `domain_class_hierarchy_bar.png`, `domain_class_hierarchy_bar_merged.png`, `domain_class_hierarchy_pie.png`

#### Manual edits — `COLLAPSE_MAP`

The raw hierarchy produces >40 distinct root classes, several of which contain InterPro naming inconsistencies or reflect shared fold topology rather than DNA-binding mechanism. The following manual merges were applied after inspecting the initial classification output:

| Raw class name(s) | Collapsed into | Rationale |
|---|---|---|
| `"Zinc finger C2H2 superfamily"`, `"Zinc finger, FYVE/PHD-type"`, `"Zinc finger, RING/FYVE/PHD-type"` | `"Zinc finger superfamily"` | Closely related structural variants of the zinc-coordinating finger fold; merging increases statistical power without losing biological meaning |
| `"Homedomain-like superfamily"` | `"Homeodomain-like superfamily"` | Corrects an InterPro spelling error (missing 'e') |
| `"Homeodomain"` | `"Homeodomain-like superfamily"` | The Domain-type entry and the Homologous_superfamily entry describe the same fold; collapse to single class |
| `"SMAD/FHA domain superfamily"` | `"SMAD-like domain superfamily"` | Two separate root entries in the hierarchy; both represent SMAD proteins |
| `"High mobility group protein HMGA"` | `"AT-hook motif"` | HMGA proteins bind DNA via AT-hook motifs rather than an HMG-box fold; functional class takes precedence |
| `"Quinoprotein alcohol dehydrogenase-like superfamily"`, `"6-phosphogluconate dehydrogenase-like, C-terminal domain superfamily"` | `"AT-hook motif"` | ELYS (AHCTF1) and GLYR1 (N-PAC) carry AT-hook DNA-binding features, but walking the InterPro hierarchy leads to metabolic enzyme roots due to shared beta-barrel topology; overridden to reflect DNA-binding function |
| `"AT hook, DNA-binding motif"` | `"AT-hook motif"` | SETBP1's sole InterPro entry (IPR017956) is a `Conserved_site` type, which falls outside the normal `Homologous_superfamily / Domain / Family` priority order and would otherwise yield "No structural annotation". The entry name directly identifies AT-hook DNA binding; manually assigned to the AT-hook class |

**Note:** `"Lambda repressor-like, DNA-binding domain superfamily"` (IPR010982; 1 protein: EDF1) is retained as its own structural class. Although it shares HTH topology with homeodomains, SCOP classifies it as a distinct superfamily; merging would be structurally imprecise.

#### Excluded and flagged proteins

| UniProt | Gene | Action | Reason |
|---|---|---|---|
| P01011 | SERPINA3 | Excluded entirely | Serine protease inhibitor; DNA-binding annotation is spurious |
| P02751 | FN1 | Retained, flagged `is_noncanonical = True` | Extracellular matrix protein; DNA binding is via heparin-binding domain, not transcriptional regulation |

---

### Step 3 — `03_clinvar_domain_mapping.py`

**Purpose:** Map each ClinVar variant to the UniProt-annotated DNA-binding domain interval(s) of its target protein, and test whether pathogenic variants are enriched inside those intervals.

**Processing steps:**
1. Extract all `"DNA binding"` feature ranges (`start`, `end`) from the UniProt JSON for every DBP; remove one duplicate domain record for O15266
2. Load ClinVar and apply the standard variant filters (see **Manual filtering decisions** below)
3. For each DBP protein, check whether each variant's `residue_seq` falls within any domain interval (`start ≤ residue_seq ≤ end`)
4. Recode clinical significance and save all variant–domain hits
5. Test whether in-domain and out-of-domain variants differ in clinical significance using Fisher's exact test with Bonferroni correction

**Statistical approach:**
- 2×2 Fisher's exact test per significance group (in-domain vs out-of-domain)
- OR with 95% CI via log-normal method; Haldane-Anscombe continuity correction if any cell = 0
- Bonferroni correction for 4 simultaneous tests

**Boundary sensitivity check:** ORs recalculated with domain boundaries shrunk and expanded by ±3 and ±5 residues to confirm results are not sensitive to UniProt coordinate precision.

**Output:** `Csvs/clinvar_domain_hits.csv` — one row per variant–domain hit; includes `sig_group` (recoded significance) and `feature_type = "DNA binding"` columns required by downstream scripts.

---

### Step 4 — `04_clinvar_variant_landscape.py`

**Purpose:** Characterise the DBP variant landscape — clinical significance profile, enrichment relative to the rest of ClinVar, and variant burden distribution.

**Sections:**
1. **Clinical significance breakdown** — counts and percentages for all ClinVar and DBP-only variants
2. **DBP vs non-DBP enrichment** — Fisher's exact test comparing the significance profile of DBP variants against all other ClinVar variants; Bonferroni corrected
3. **Top-20 DBPs by variant count** — ranked table saved to CSV
4. **Variant skew plots** — ranked log-scale plot and Lorenz-style cumulative distribution showing that variant burden is heavily concentrated in a small fraction of DBPs

**Note on ascertainment bias:** The DBP vs non-DBP enrichment test is noted in output as potentially inflated by ascertainment bias — well-studied transcription factors are more likely to be sequenced and submitted to ClinVar.

**Outputs:**
- `Csvs/clinvar_dbp_sig_enrichment.csv` — OR table for DBP vs non-DBP comparison
- `Csvs/top20_variants_raw.csv` — top 20 DBPs by raw variant count
- `Figures/clinvar_variant_skew.png`

---

### Step 5 — `06_clinvar_domain_class_enrichment.py`

**Purpose:** Test whether pathogenic variants are enriched in specific structural classes, using only variants that fall within annotated DNA-binding domain intervals.

**Key distinction from Step 4:** Only in-domain variants (from `clinvar_domain_hits.csv`) are used. Variants outside domain boundaries are excluded from both numerator and denominator, making this a test of domain-level pathogenic constraint rather than whole-protein variant burden.

**Deduplication:** Variants mapping to multiple domain intervals in multi-domain proteins are counted once via `drop_duplicates("#AlleleID")`.

**Statistical approach:**
- Two Fisher's exact tests per class: (1) P/LP vs B/LB, and (2) P/LP vs all other variants
- One-sided test (alternative="greater") for directional enrichment
- Bonferroni correction; classes with <2 proteins excluded

**Outputs:**
- `Csvs/clinvar_domain_class_enrichment.csv`
- `Csvs/clinvar_domain_enrichment_stats.csv` — P/LP vs B/LB results
- `Csvs/clinvar_domain_enrichment_stats_plp_vs_all.csv` — P/LP vs all results
- `Figures/clinvar_domain_OR_forest_plp_vs_all.png` — forest plot
- `Figures/clinvar_domain_plp_dotplot.png` — per-protein P/LP fraction dot plot

---

### Step 6 — `07_enrichment_by_domain.py`

**Status:** Earlier draft of Step 5 — to be reviewed and either updated or retired.

---

### Step 7 — `08_sensitivity_sim_only_check.py`

**Purpose:** Test whether enrichment results are robust to annotation quality concerns.

**Background:** 52 of the 611 DBPs have DNA-binding annotations derived solely from sequence similarity (no direct experimental evidence; ECO:0000250, ECO:0000255, ECO:0000256).

**Method:** Recalculate ClinVar enrichment odds ratios after removing these 52 proteins and their associated variants. Report absolute and percentage change in OR per significance group.

**Output:** Console report only.

---

## Manual filtering decisions

The following filters are applied consistently across all scripts that load ClinVar data (Steps 3, 4, 5):

| Filter | Code | Rationale |
|---|---|---|
| Germline variants only | `cv[cv["OriginSimple"] == "germline"]` | Somatic variants have fundamentally different clinical significance patterns and are not the subject of this analysis |
| Remove start-lost variants | `~((cv["canonical_residue"] == "Met") & (cv["residue_seq"] == 1))` | Variants at the initiator methionine represent translation initiation loss, not missense substitution; their pathogenicity mechanism is distinct and would confound the analysis |
| Remove variants missing UniProt ID or residue position | `dropna(subset=["uniprot", "residue_seq"])` | Variants without a protein accession or amino acid position cannot be mapped to a domain or assigned to a structural class |
| Exclude transposable element proteins | `classif[~classif["is_te"]]` | TE-derived proteins (flagged `is_te == True` in Step 2 output) are not canonical DNA-binding proteins |
| Deduplicate on AlleleID | `drop_duplicates("#AlleleID")` | 82 variants in ClinVar map to more than one DBP protein; each variant is counted once to avoid inflating class counts |

### Clinical significance recoding

ClinVar's detailed classification strings are collapsed to four groups:

| Group | ClinVar terms included |
|---|---|
| `Pathogenic/LP` | Pathogenic; Likely pathogenic; Pathogenic/Likely pathogenic; Pathogenic; other; Pathogenic; drug response; Likely pathogenic; drug response; Likely pathogenic/Likely risk allele |
| `Benign/LB` | Benign; Likely benign; Benign/Likely benign; Benign; risk factor; Benign; other |
| `VUS` | Uncertain significance; Uncertain significance/Uncertain risk allele; Uncertain significance; drug response |
| `Other/Conflicting` | Conflicting classifications of pathogenicity (all subtypes); all remaining terms |

---

## Exploratory scripts (`Initial_stuff/`)

These scripts represent earlier approaches that were superseded by the final pipeline. They are retained for reference but are not part of the reproducible analysis.

| Script | What it tried | Why superseded |
|---|---|---|
| `filter_go_terms.py` | Counted proteins with specific GO terms (DNA binding, transcription, repair, etc.) | GO-term filtering replaced by UniProt Swiss-Prot reviewed protein set |
| `dna_binding_analysis.py` | Manual InterPro ID → domain name mapping | Replaced by automated hierarchy traversal in Step 2 |
| `domain_analysis_api.py` | InterPro REST API calls to fetch hierarchy | Replaced by local `ParentChildTreeFile` to avoid rate limits and ensure reproducibility |
| `domain_analysis_viz.py` | Visualisations from manual mapping | Superseded by Step 2 plots |
| `unified_domain_analysis.py` | API-based classification with local cache | Replaced by local file parsing in Step 2 |
| `interpro_fix.py` | Corrected malformed InterPro data | No longer needed after switching to local files |
| `keyword_frequency_chart.py` | Keyword-based domain identification as fallback | Not used in final pipeline |
| `dbp_domain_analysis_intial.py` | Substring matching on InterPro short names | Replaced by root-ancestry approach in Step 2 |
| `simple_domain_viz.py` | Raw InterPro ID frequency plots | Works at raw IPR accession level; predates hierarchy-based classification |
