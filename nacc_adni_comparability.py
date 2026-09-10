#!/usr/bin/env python
# ============================================================================
# NACC <-> ADNI POPULATION-COMPARABILITY CHECK (NOT a model validation)
# ============================================================================
# Purpose: Compare the demographic makeup of the NACC UDS cohort and the ADNI
# cohort on the subset of features with a CONFIRMED, verifiable NACC->ADNI map.
# This is a POPULATION-COMPARABILITY check only. It does NOT train or evaluate
# any model, and its numbers must NOT be reported alongside Config 1's
# 88%/85% headline metrics (those are model-performance numbers, not
# population-comparison numbers).
#
# Verified mappings used:
#   APOE4 e4-count : NACC NACCNE4S (0/1/2, 9=unknown)  <->  ADNI APOE4 (0/1/2)
#                    (direct match -- no genotype-pair conversion needed)
#   Sex            : NACC SEX (1=M,2=F)                <->  ADNI PTGENDER
#   Race           : NACC RACE                          <->  ADNI PTRACCAT
#   Ethnicity      : NACC HISPANIC (0=No,1=Yes,9=Unk)   <->  ADNI PTETHCAT
#   Age at baseline: NACC NACCAGE at NACCVNUM==1        <->  ADNI AGE at VISCODE=='bl'
#
# NOTE: The `target` (4-class CDR-based next-visit diagnosis) is NOT reconstructed
# here. ADNI's DX_bl is a 3-class clinical diagnosis (CN/MCI/AD + SMC) on a
# different instrument, and ADNIMERGE has no raw Global CDR column. Converting
# CDRSB->CDR-Global would require a specifically cited published mapping and
# would only be an approximation of a different measure -- out of scope here.
# ============================================================================

import pandas as pd
import numpy as np

NACC_FILE = "output/features_engineered.csv"
ADNI_FILE = "ADNIMERGE.csv"

# ---- normalized category labels (shared across both cohorts) ----
SEX_LABELS = {1: "Male", 2: "Female"}
RACE_LABELS = {
    1: "White", 2: "Black/African American", 3: "American Indian/Alaska Native",
    4: "Native Hawaiian/Pacific Islander", 5: "Asian", 50: "Other/Multiracial", 99: "Unknown",
}
ETH_LABELS = {0: "Not Hispanic/Latino", 1: "Hispanic/Latino", 9: "Unknown"}
APOE_LABELS = {0: "0 copies", 1: "1 copy", 2: "2 copies", 9: "Unknown"}


def load_nacc():
    d = pd.read_csv(NACC_FILE, low_memory=False)
    # Baseline visit = NACCVNUM == 1 (one row per patient)
    d = d[d["NACCVNUM"] == 1].copy()
    d = d.dropna(subset=["NACCAGE", "SEX", "RACE", "HISPANIC", "NACCNE4S"])
    out = pd.DataFrame({
        "Age": d["NACCAGE"].astype(float),
        "Sex": d["SEX"].map(SEX_LABELS).astype(str),
        "Race": d["RACE"].map(RACE_LABELS).astype(str),
        "Ethnicity": d["HISPANIC"].map(ETH_LABELS).astype(str),
        "APOE4": d["NACCNE4S"].map(APOE_LABELS).astype(str),
    })
    return out


def load_adni():
    d = pd.read_csv(ADNI_FILE, low_memory=False)
    # Baseline visit = VISCODE == 'bl'
    d = d[d["VISCODE"] == "bl"].copy()
    # ADNI age at baseline = AGE + Years_bl (Years_bl >= 0); at bl Years_bl ~ 0.
    d["age_base"] = d["AGE"] + d["Years_bl"]
    d = d.dropna(subset=["age_base", "PTGENDER", "PTRACCAT", "PTETHCAT", "APOE4"])
    out = pd.DataFrame({
        "Age": d["age_base"].astype(float),
        "Sex": d["PTGENDER"].astype(str),
        "Race": d["PTRACCAT"].astype(str),
        "Ethnicity": d["PTETHCAT"].astype(str),
        "APOE4": d["APOE4"].map({0: "0 copies", 1: "1 copy", 2: "2 copies"}).fillna("Unknown").astype(str),
    })
    return out


def fmt_pct(ser, order=None):
    counts = ser.value_counts(dropna=False)
    total = counts.sum()
    rows = []
    for label in (order if order else counts.index):
        c = counts.get(label, 0)
        rows.append(f"{label}: {c} ({100*c/total:.1f}%)")
    return "\n      ".join(rows)


def main():
    print("=" * 78)
    print("NACC UDS  vs  ADNI  --  POPULATION-COMPARABILITY CHECK (baseline visits)")
    print("=" * 78)

    nacc = load_nacc()
    adni = load_adni()
    print(f"\nNACC baseline patients: {len(nacc):,}")
    print(f"ADNI baseline patients: {len(adni):,}")

    print("\n--- AGE (baseline, years) ---")
    print(f"  NACC  n={len(nacc):,}  mean={nacc['Age'].mean():.1f}  SD={nacc['Age'].std():.1f}  "
          f"median={nacc['Age'].median():.1f}  range={nacc['Age'].min():.0f}-{nacc['Age'].max():.0f}")
    print(f"  ADNI  n={len(adni):,}  mean={adni['Age'].mean():.1f}  SD={adni['Age'].std():.1f}  "
          f"median={adni['Age'].median():.1f}  range={adni['Age'].min():.0f}-{adni['Age'].max():.0f}")

    print("\n--- SEX ---")
    print("  NACC:\n      " + fmt_pct(nacc["Sex"], order=["Male", "Female"]))
    print("  ADNI:\n      " + fmt_pct(adni["Sex"], order=["Male", "Female"]))

    print("\n--- RACE ---")
    nac_race_order = list(RACE_LABELS.values())
    print("  NACC:\n      " + fmt_pct(nacc["Race"], order=nac_race_order))
    adni_race_order = ["White", "Black", "Asian", "Am Indian/Alaskan",
                       "Hawaiian/Other PI", "More than one", "Unknown"]
    print("  ADNI:\n      " + fmt_pct(adni["Race"], order=adni_race_order))

    print("\n--- ETHNICITY (Hispanic/Latino) ---")
    print("  NACC:\n      " + fmt_pct(nacc["Ethnicity"], order=["Not Hispanic/Latino", "Hispanic/Latino", "Unknown"]))
    print("  ADNI:\n      " + fmt_pct(adni["Ethnicity"], order=["Not Hisp/Latino", "Hisp/Latino", "Unknown"]))

    print("\n--- APOE4 (e4 allele count) ---")
    print("  NACC:\n      " + fmt_pct(nacc["APOE4"], order=["0 copies", "1 copy", "2 copies", "Unknown"]))
    print("  ADNI:\n      " + fmt_pct(adni["APOE4"], order=["0 copies", "1 copy", "2 copies", "Unknown"]))

    print("\n" + "=" * 78)
    print("Interpretation: This table compares the demographic DISTRIBUTION of the")
    print("two cohorts. It is a population-comparability check, NOT a validation of")
    print("any model, and must not be compared to Config 1's 88%/85% performance.")
    print("=" * 78)

    # ---- Save the population-comparison table to CSV ----
    rows = []
    for label, nacc_val, adni_val in [
        ("N_baseline_patients", len(nacc), len(adni)),
        ("Age_mean", round(nacc["Age"].mean(), 1), round(adni["Age"].mean(), 1)),
        ("Age_SD", round(nacc["Age"].std(), 1), round(adni["Age"].std(), 1)),
        ("Age_median", round(nacc["Age"].median(), 1), round(adni["Age"].median(), 1)),
        ("Age_min", int(nacc["Age"].min()), int(adni["Age"].min())),
        ("Age_max", int(nacc["Age"].max()), int(adni["Age"].max())),
    ]:
        rows.append({"Feature": label, "NACC": nacc_val, "ADNI": adni_val})

    # Per-category rows (percent), using each cohort's own label set
    def pct_cell(df, col, label):
        counts = df[col].value_counts(dropna=False)
        total = counts.sum()
        return round(100*counts.get(label, 0)/total, 1) if total else np.nan

    # Consolidated categories mapped to a SHARED label per cohort for side-by-side alignment.
    # NACC RACE labels vs ADNI PTRACCAT labels are different strings; map them to one label each.
    race_map = {
        "White": ("White", "White"),
        "Black/African American": ("Black/African American", "Black"),
        "American Indian/Alaska Native": ("American Indian/Alaska Native", "Am Indian/Alaskan"),
        "Native Hawaiian/Pacific Islander": ("Native Hawaiian/Pacific Islander", "Hawaiian/Other PI"),
        "Asian": ("Asian", "Asian"),
        "Other/Multiracial": ("Other/Multiracial", "More than one"),
        "Unknown": ("Unknown", "Unknown"),
    }
    eth_map = {
        "Not Hispanic/Latino": ("Not Hispanic/Latino", "Not Hisp/Latino"),
        "Hispanic/Latino": ("Hispanic/Latino", "Hisp/Latino"),
        "Unknown": ("Unknown", "Unknown"),
    }

    cat = [
        ("Sex", ["Male", "Female"], None),
        ("Race", list(race_map.keys()), race_map),
        ("Ethnicity", list(eth_map.keys()), eth_map),
        ("APOE4", ["0 copies", "1 copy", "2 copies", "Unknown"], None),
    ]
    for col, order, mapping in cat:
        for label in order:
            if mapping:
                nac_lbl, adni_lbl = mapping[label]
                nacc_cell = pct_cell(nacc, col, nac_lbl)
                adni_cell = pct_cell(adni, col, adni_lbl)
            else:
                nacc_cell = pct_cell(nacc, col, label)
                adni_cell = pct_cell(adni, col, label)
            rows.append({"Feature": f"{col}: {label}",
                         "NACC": nacc_cell, "ADNI": adni_cell})

    out = pd.DataFrame(rows)
    out_path = "output/nacc_adni_population_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved population-comparison table to: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
