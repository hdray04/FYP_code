import json

# Input file
JSON_FILE = "./Data/uniprotkb_organism_id_9606_AND_reviewed_2026_03_06.json"

# GO terms to filter by
GO_TERMS = {
    "GO:0003677": "DNA binding",
    "GO:0006355": "Transcription regulation",
    "GO:0006281": "DNA repair",
    "GO:0006260": "DNA replication",
    "GO:0006325": "Chromatin organisation",
}

# Load JSON
with open(JSON_FILE) as f:
    data = json.load(f)

proteins = data["results"]


def get_go_ids(protein):
    """Get all GO term IDs for a protein."""
    xrefs = protein.get("uniProtKBCrossReferences", [])
    return {ref["id"] for ref in xrefs if ref.get("database") == "GO"}


# Count proteins for each GO term
go_counts = {go_id: 0 for go_id in GO_TERMS}

for protein in proteins:
    go_ids = get_go_ids(protein)
    for go_id in GO_TERMS:
        if go_id in go_ids:
            go_counts[go_id] += 1

# Print results
print("Protein counts per GO term:")
for go_id, label in GO_TERMS.items():
    print(f"  {go_id} ({label}): {go_counts[go_id]} proteins")
