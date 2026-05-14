"""
RSA accessibility-class enrichment: P/LP vs B/LB.
Restricted to in-domain residues with pLDDT >= 70.
Three-way classification: Buried / Intermediate / Surface.
Three Fisher's exact tests with Bonferroni correction (n=3).
"""

import pandas as pd
from scipy.stats import fisher_exact

MERGED_CSV = "Csvs/clinvar_am_m3d_dssp_merged.csv"
OUTPUT_CSV = "Csvs/rsa_accessibility_enrichment.csv"

PLDDT_THRESHOLD = 70
RSA_BINS = [-0.001, 0.20, 0.50, 999]
RSA_LABELS = ["Buried", "Intermediate", "Surface"]


def load_analysis_set(path):
    df = pd.read_csv(path)
    return df[
        df["in_domain"]
        & (df["plddt"] >= PLDDT_THRESHOLD)
        & df["sig_group"].isin(["Pathogenic/LP", "Benign/LB"])
    ].copy()


def assign_rsa_class(df):
    df["rsa_class"] = pd.cut(df["rsa"], bins=RSA_BINS, labels=RSA_LABELS)
    return df


def run_enrichment(df):
    plp = df[df["sig_group"] == "Pathogenic/LP"]
    blb = df[df["sig_group"] == "Benign/LB"]
    n_tests = len(RSA_LABELS)

    rows = []
    for cls in RSA_LABELS:
        a = (plp["rsa_class"] == cls).sum()
        b = (blb["rsa_class"] == cls).sum()
        c = (plp["rsa_class"] != cls).sum()
        d = (blb["rsa_class"] != cls).sum()
        or_, p = fisher_exact([[a, b], [c, d]])
        rows.append({
            "rsa_class": cls,
            "n_plp": a,
            "n_blb": b,
            "odds_ratio": round(or_, 3),
            "p_value": round(p, 6),
            "p_bonferroni": round(min(p * n_tests, 1.0), 6),
            "significant": min(p * n_tests, 1.0) < 0.05,
        })

    return pd.DataFrame(rows)


def main():
    df = load_analysis_set(MERGED_CSV)
    df = assign_rsa_class(df)

    plp = df[df["sig_group"] == "Pathogenic/LP"]
    blb = df[df["sig_group"] == "Benign/LB"]
    print(f"Analysis set: P/LP n={len(plp)}, B/LB n={len(blb)}")
    print(f"pLDDT threshold: >={PLDDT_THRESHOLD}, in-domain only\n")

    print("RSA class counts:")
    print(df.groupby(["sig_group", "rsa_class"]).size().unstack(fill_value=0))
    print()

    results = run_enrichment(df)
    print("Enrichment results (Bonferroni n=3):")
    print(results.to_string(index=False))

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
