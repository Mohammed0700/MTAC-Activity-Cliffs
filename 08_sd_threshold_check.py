#!/usr/bin/env python
"""
08_sd_threshold_check.py
========================
Editor pre-check (round 2), point 2: the replicate standard-deviation filter.

WHAT IT ANSWERS
---------------
1. Does our filter reproduce the MoleculeACE implementation exactly?
   MoleculeACE v3.0.0, MoleculeACE/benchmark/data_fetching/exp_property_curation.py,
   curate_dataframe():
       df_curated.drop(df_curated.loc[np.log10(df_curated['exp_std']) >= 1].index)
   with convert_to_log=False by default (main_curator.py), i.e. the SD is computed on
   untransformed nM values (np.std, ddof=0) after Dixon's Q-test. log10(SD) >= 1 is the
   same test as SD >= 10 nM. Our notebooks 02 and 03 apply exp_std >= 10 on the same
   quantity, so the two criteria are identical. This script re-derives every curated
   file from the replicate-level *_filtered.csv / *_filter.csv files and checks that.

2. Would a potency-independent criterion change compound inclusion?
   It re-applies the filter on the log scale (SD of pKi across replicates >= 1.0 and
   >= 0.5 log units) and reports, per target, how many curated compounds would be
   removed, how many would be added, and how many of those are in the final prepared
   (train/test) datasets. If those numbers are zero or negligible the 10 nM criterion
   can be kept and justified in the text; if not, the datasets must be regenerated.

No RDKit is needed. Run from the repository root:

    python 08_sd_threshold_check.py --repo . --out Results/sd_threshold_check

Outputs:
    <out>/sd_threshold_per_target.csv   one row per target
    <out>/sd_threshold_summary.txt      panel totals and the reproduction check
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- identical to notebooks 02/03 and to MoleculeACE ----------------------------
Q_TAB_005 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466, 0.444, 0.426, 0.41, 0.396,
             0.384, 0.374, 0.365, 0.356, 0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317,
             0.312, 0.308, 0.305, 0.301, 0.29]


def q_dict(max_offset):
    # notebook 02 and MoleculeACE: range(3, len+1) -> n = 3..28
    # notebook 03 (BindingDB):     range(3, len+3) -> n = 3..30
    return {n: q for n, q in zip(range(3, len(Q_TAB_005) + max_offset), Q_TAB_005)}


def dixon_outliers(values, qd):
    if not (3 <= len(values) <= max(qd)):
        return []
    s = sorted(values)
    qmin, qmax = (0, 0), (0, 0)
    rng = s[-1] - s[0]
    if rng != 0:
        qmin = ((s[1] - s[0]) / rng - qd[len(values)], s[0])
        qmax = (abs(s[-2] - s[-1]) / abs(s[0] - s[-1]) - qd[len(values)], s[-1])
    if not qmin[0] > 0 and not qmax[0] > 0:
        return []
    if qmin[0] == qmax[0]:
        return [qmin[1], qmax[1]]
    return [qmin[1]] if qmin[0] > qmax[0] else [qmax[1]]


def aggregate(df, qd):
    """Replicates -> one row per CuratedSmiles: mean/SD in nM (ddof=0) and SD of pKi."""
    rows = []
    for smi, g in df.groupby("CuratedSmiles", sort=False):
        v = g["value"].to_numpy(dtype=float)
        out = dixon_outliers(list(v), qd)
        keep = v[~np.isin(v, np.array(out))] if out else v
        if len(keep) == 0:
            continue
        pki = 9.0 - np.log10(keep)
        rows.append({"smiles": smi, "exp_mean": keep.mean(), "exp_std": keep.std(ddof=0),
                     "pki_std": pki.std(ddof=0), "n_rep": len(keep)})
    return pd.DataFrame(rows)


def find(repo, stem, suffixes):
    for s in suffixes:
        p = repo / f"{stem}{s}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--out", default="Results/sd_threshold_check")
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prepared = sorted((repo / "Data").glob("*_prepared.csv"))
    if not prepared:
        sys.exit("No Data/*_prepared.csv found; run from the repository root or pass --repo")

    rows = []
    for p in prepared:
        t = p.stem.replace("_prepared", "")
        is_bdb = not t.startswith("CHEMBL")
        filt = find(repo, t, ["_filtered.csv", "_filter.csv"])
        cur = find(repo, t, ["_curated.csv"])
        if filt is None or cur is None:
            print(f"  {t}: replicate or curated file missing, skipped")
            continue
        f = pd.read_csv(filt)
        if "units" in f.columns:                       # unit filter, as in the corrected extraction
            f = f[f["units"].astype(str).str.strip() == "nM"]
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        f = f.dropna(subset=["CuratedSmiles", "value"])
        f = f[f["value"] > 0]

        agg = aggregate(f, q_dict(3 if is_bdb else 1))
        keep_10nM = set(agg.loc[~(agg.exp_std >= 10), "smiles"])
        keep_log1 = set(agg.loc[~(agg.pki_std >= 1.0), "smiles"])
        keep_log05 = set(agg.loc[~(agg.pki_std >= 0.5), "smiles"])

        curated = set(pd.read_csv(cur)["smiles"])
        prep = set(pd.read_csv(p)["smiles"])
        rows.append({
            "target": t, "source": "BindingDB" if is_bdb else "ChEMBL",
            "n_compounds_before_sd_filter": len(agg),
            "n_curated_file": len(curated),
            "n_kept_sd_ge_10nM_rule": len(keep_10nM),
            "reproduces_curated_file": keep_10nM == curated,
            "n_with_replicates": int((agg.n_rep >= 2).sum()),
            "removed_by_10nM": len(agg) - len(keep_10nM),
            "log1_would_remove": len(keep_10nM - keep_log1),
            "log1_would_add": len(keep_log1 - keep_10nM),
            "log05_would_remove": len(keep_10nM - keep_log05),
            "log05_would_add": len(keep_log05 - keep_10nM),
            "n_prepared": len(prep),
            "prepared_affected_by_log1": len(prep & (keep_10nM - keep_log1)),
        })
        r = rows[-1]
        print(f"  {t:22s} curated {r['n_curated_file']:5d}  reproduced={r['reproduces_curated_file']!s:5s}  "
              f"10nM removed {r['removed_by_10nM']:4d} | 1-log: -{r['log1_would_remove']} "
              f"+{r['log1_would_add']} | 0.5-log: -{r['log05_would_remove']} +{r['log05_would_add']}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "sd_threshold_per_target.csv", index=False)
    tot = res.select_dtypes("number").sum()
    lines = [
        "Replicate-SD filter check",
        "=" * 60,
        f"targets checked                          : {len(res)}",
        f"curated files reproduced exactly (10 nM) : {int(res.reproduces_curated_file.sum())} / {len(res)}",
        f"compounds before SD filter               : {int(tot.n_compounds_before_sd_filter):,}",
        f"compounds with >= 2 replicates           : {int(tot.n_with_replicates):,}",
        f"removed by SD >= 10 nM (current)         : {int(tot.removed_by_10nM):,}",
        "",
        "Switching to a log-scale criterion would change inclusion by:",
        f"  SD(pKi) >= 1.0 : remove {int(tot.log1_would_remove):,}, add {int(tot.log1_would_add):,} "
        f"(targets affected: {int(((res.log1_would_remove + res.log1_would_add) > 0).sum())})",
        f"  SD(pKi) >= 0.5 : remove {int(tot.log05_would_remove):,}, add {int(tot.log05_would_add):,} "
        f"(targets affected: {int(((res.log05_would_remove + res.log05_would_add) > 0).sum())})",
    ]
    text = "\n".join(lines)
    (out / "sd_threshold_summary.txt").write_text(text)
    print("\n" + text + f"\n\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()
