"""
Parse DSSP outputs + AlphaFold pLDDT JSONs into a single residue-level CSV.
Just extracts raw values - RSA computation and thresholding happens downstream.
"""

import json
import re
from pathlib import Path
import pandas as pd
from Bio.PDB.DSSP import make_dssp_dict

UNIPROT_RE = re.compile(r"AF-([A-Z0-9]+(?:-\d+)?)-F\d+-model_v\d+")

# Max ASA per amino acid (Tien et al. 2013, empirical extended-chain values)
MAX_ASA = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
    'E': 223.0, 'Q': 225.0, 'G': 104.0, 'H': 224.0, 'I': 197.0,
    'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
    'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0,
}


def parse_plddt_json(plddt_path):
    """Return dict: residue position (1-indexed) -> pLDDT score."""
    with open(plddt_path) as f:
        data = json.load(f)
    scores = data.get("confidenceScore") or data.get("plddt") or []
    positions = data.get("residueNumber") or list(range(1, len(scores) + 1))
    return dict(zip(positions, scores))


def parse_one_protein(dssp_path, plddt_path):
    """Parse one protein's DSSP + pLDDT into a list of residue dicts."""
    uniprot_match = UNIPROT_RE.search(dssp_path.name)
    if not uniprot_match:
        print(f"  WARNING: could not extract UniProt ID from {dssp_path.name}")
        return []
    uniprot_id = uniprot_match.group(1)

    try:
        dssp_dict, keys = make_dssp_dict(str(dssp_path))
    except Exception as e:
        print(f"  WARNING: failed to parse {dssp_path.name}: {e}")
        return []

    plddt_map = {}
    if plddt_path.exists():
        try:
            plddt_map = parse_plddt_json(plddt_path)
        except Exception as e:
            print(f"  WARNING: failed to parse pLDDT for {uniprot_id}: {e}")

    rows = []
    for key in keys:
        chain_id, res_id = key
        position = res_id[1]
        record = dssp_dict[key]
        aa, ss, abs_asa, phi, psi = record[0], record[1], record[2], record[3], record[4]
        dssp_idx = record[5] if len(record) > 5 else position

        max_asa = MAX_ASA.get(aa)
        rsa = round(abs_asa / max_asa, 4) if max_asa else None

        rows.append({
            "uniprot_id": uniprot_id,
            "chain": chain_id,
            "position": position,
            "dssp_index": dssp_idx,
            "aa": aa,
            "abs_asa": abs_asa,
            "rsa": rsa,
            "ss": ss,
            "phi": phi,
            "psi": psi,
            "plddt": plddt_map.get(position),
        })

    return rows


def main(dssp_dir, plddt_dir, output_csv):
    dssp_dir = Path(dssp_dir)
    plddt_dir = Path(plddt_dir)

    dssp_files = sorted(dssp_dir.glob("*.dssp"))
    print(f"Found {len(dssp_files)} DSSP files")

    all_rows = []
    parsed_ok = 0
    parsed_fail = 0

    for i, dssp_path in enumerate(dssp_files, 1):
        plddt_name = dssp_path.stem.replace("-model_v", "-confidence_v") + ".json"
        plddt_path = plddt_dir / plddt_name

        rows = parse_one_protein(dssp_path, plddt_path)
        if rows:
            all_rows.extend(rows)
            parsed_ok += 1
        else:
            parsed_fail += 1

        if i % 50 == 0:
            print(f"  [{i}/{len(dssp_files)}] ok={parsed_ok}, failed={parsed_fail}")

    print(f"\nTotal: ok={parsed_ok}, failed={parsed_fail}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({df.shape[0]:,} rows, {df.shape[1]} cols)")
    print(f"Unique proteins: {df['uniprot_id'].nunique()}")


def merge_variants_with_dssp(
    dssp_csv,
    clinvar_csv,
    m3d_csv,
    domains_csv,
    output_csv,
):
    """
    Join ClinVar+AM variants with M3D predictions and DSSP residue data,
    then flag each variant as in_domain or not.
    """
    dssp = pd.read_csv(dssp_csv)
    clinvar = pd.read_csv(clinvar_csv)
    m3d = pd.read_csv(m3d_csv)
    domains = pd.read_csv(domains_csv)

    m3d_cols = [
        "uniprot", "position", "wildtype", "mutant", "damaging",
        "buried_hydrophilic_introduced", "buried_charge_introduced",
        "secondary_structure_altered", "buried_charge_switch",
        "buried_H_bond_breakage", "buried_salt_bridge_breakage",
        "buried_exposed_switch", "clash",
        "cavity_altered", "gly_in_a_bend", "buried_Pro_introduced",
        "disulphide_breakage", "disallowed_phi_psi", "buried_charge_replaced",
        "buried_Gly_replaced", "cis_pro_replaced",
    ]
    merged = clinvar.merge(
        m3d[m3d_cols],
        left_on=["uniprot_id", "pos", "ref_aa_1", "alt_aa_1"],
        right_on=["uniprot", "position", "wildtype", "mutant"],
        how="left",
        suffixes=("", "_m3d"),
    ).drop(columns=["uniprot_m3d", "position", "wildtype", "mutant"])

    dssp_sub = dssp[["uniprot_id", "position", "plddt", "rsa", "ss", "abs_asa"]]
    merged = merged.merge(
        dssp_sub,
        left_on=["uniprot_id", "pos"],
        right_on=["uniprot_id", "position"],
        how="left",
    ).drop(columns=["position"])

    merged["in_domain"] = False
    for _, dom in domains[["uniprot", "start", "end"]].iterrows():
        mask = (
            (merged["uniprot_id"] == dom["uniprot"])
            & (merged["pos"] >= dom["start"])
            & (merged["pos"] <= dom["end"])
        )
        merged.loc[mask, "in_domain"] = True

    merged.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({len(merged):,} variants, {merged.shape[1]} cols)")
    print(f"  In-domain: {merged['in_domain'].sum():,}")
    print(f"  Has DSSP:  {merged['plddt'].notna().sum():,}")
    print(f"  Has M3D:   {merged['damaging'].notna().sum():,}")
    print(f"  In-domain + pLDDT>=70: {((merged['in_domain']) & (merged['plddt'] >= 70)).sum():,}")


if __name__ == "__main__":
    main(
        dssp_dir="/Users/harrietdray/FYP_code/Data/dssp_output",
        plddt_dir="/Users/harrietdray/FYP_code/Data/plddt_json",
        output_csv="/Users/harrietdray/FYP_code/Data/dssp_residue_table.csv",
    )

    merge_variants_with_dssp(
        dssp_csv="/Users/harrietdray/FYP_code/Data/dssp_residue_table.csv",
        clinvar_csv="/Users/harrietdray/FYP_code/Csvs/clinvar_alphamissense_merged_dedup.csv",
        m3d_csv="/Users/harrietdray/FYP_code/Data/M3D_results.csv",
        domains_csv="/Users/harrietdray/FYP_code/Csvs/domain_features.csv",
        output_csv="/Users/harrietdray/FYP_code/Csvs/clinvar_am_m3d_dssp_merged.csv",
    )