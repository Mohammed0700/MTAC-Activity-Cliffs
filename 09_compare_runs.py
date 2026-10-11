#!/usr/bin/env python
"""
09_compare_runs.py
==================
Before/after comparison for revision R2 (editor pre-check, round 2, point 1).

Compares the previous run (training cliff labels from MoleculeACE split_data(),
assigned on the full dataset) with the R2 run (training cliff labels recomputed
from the outer training set alone). Needs only pandas and SciPy.

    python 09_compare_runs.py --before Results/rerun --after Results/rerun_r2

Writes to <after>/comparison/:
    per_target_before_after.csv   protocol-B choice and errors for every target, both runs
    panel_before_after.csv        the panel-level numbers quoted in the paper, both runs
    TableS8_train_cliff_labels.csv  proposed new supplementary table (label audit)
    comparison.txt                plain-text summary for the response letter
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NAMES = {  # file stem -> (protein name, target ID), as in Table 1
    "CHEMBL233": ("Mu-opioid receptor", "CHEMBL233"), "CHEMBL214": ("5-HT1A receptor", "CHEMBL214"),
    "CHEMBL3371": ("5-HT6 receptor", "CHEMBL3371"), "5-HT7_receptor": ("5-HT7 receptor", "BindingDB"),
    "CHEMBL1862": ("ABL1", "CHEMBL1862"), "Acetylcholinesterase": ("Acetylcholinesterase", "BindingDB"),
    "CHEMBL4282": ("AKT1", "CHEMBL4282"), "CHEMBL4816": ("AKT3", "CHEMBL4816"),
    "CHEMBL1978": ("Aromatase", "CHEMBL1978"), "Bcl-2": ("Bcl-2", "BindingDB"),
    "CHEMBL267": ("c-Src", "CHEMBL267"), "CHEMBL261": ("Carbonic anhydrase 1", "CHEMBL261"),
    "CHEMBL4072": ("Cathepsin B", "CHEMBL4072"), "CHEMBL301": ("CDK2", "CHEMBL301"),
    "CHEMBL331": ("CDK4", "CHEMBL331"), "CHEMBL1907601": ("CDK4/cyclin D1", "CHEMBL1907601"),
    "CHEMBL217": ("D2 dopamine receptor", "CHEMBL217"), "CHEMBL203": ("EGFR", "CHEMBL203"),
    "CHEMBL4040": ("ERK2", "CHEMBL4040"), "CHEMBL2695": ("FAK1", "CHEMBL2695"),
    "CHEMBL262": ("GSK-3β", "CHEMBL262"), "CHEMBL2276": ("JNK1", "CHEMBL2276"),
    "CHEMBL1936": ("KIT", "CHEMBL1936"), "CHEMBL5023": ("MDM2", "CHEMBL5023"),
    "CHEMBL3587": ("MEK1", "CHEMBL3587"), "CHEMBL2842": ("mTOR", "CHEMBL2842"),
    "CHEMBL4792": ("Orexin receptor 2", "CHEMBL4792"), "CHEMBL4482": ("PAK4", "CHEMBL4482"),
    "CHEMBL1827": ("PDE5", "CHEMBL1827"), "CHEMBL299": ("PKCα", "CHEMBL299"),
    "CHEMBL4501": ("S6K1", "CHEMBL4501"), "CHEMBL204": ("Thrombin", "CHEMBL204"),
    "CHEMBL279": ("VEGFR2", "CHEMBL279"),
}


def name(t):
    return NAMES.get(t, (t, t))[0]


def panel(res):
    """Panel-level statistics, computed exactly as in Sections 2.5-2.8."""
    g, c = res["B_test_rmse"], res["B_test_cliff_rmse"]
    d = 100 * (c - g) / g
    w = stats.wilcoxon(c, g, method="exact") if len(res) > 1 else None
    opt = res["optimism_cliff_rmse"]
    small = res["n_cliff_test"] < 20
    mw = (stats.mannwhitneyu(opt[small], opt[~small], alternative="two-sided")
          if small.any() and (~small).any() else None)
    r = (stats.pearsonr(res["n_cliff_test"], opt) if len(res) > 2 and opt.std() > 0
         else (np.nan, np.nan))
    out = {
        "targets": len(res),
        "cliff > global (n)": int((c > g).sum()),
        "mean Δ (%)": round(d.mean(), 1), "median Δ (%)": round(d.median(), 1),
        "Δ range (%)": f"{d.min():.1f} to {d.max():.1f}",
        "Wilcoxon W": None if w is None else float(w.statistic),
        "Wilcoxon p": None if w is None else float(f"{w.pvalue:.2g}"),
        "mean cliff RMSE, A (test-selected)": round(res["A_test_cliff_rmse"].mean(), 3),
        "mean cliff RMSE, B (CV-selected)": round(c.mean(), 3),
        "same combination A and B (n)": int(res["same_combination"].astype(str).str.lower()
                                            .eq("true").sum()),
        "mean selection optimism": round(opt.mean(), 3),
        "median selection optimism": round(opt.median(), 3),
        "max selection optimism": round(opt.max(), 3),
        "optimism vs n cliff test, Pearson r": round(r[0], 2),
        "optimism vs n cliff test, p": float(f"{r[1]:.2g}") if not np.isnan(r[1]) else None,
        "Mann-Whitney U (<20 vs >=20)": None if mw is None else float(mw.statistic),
        "Mann-Whitney p": None if mw is None else float(f"{mw.pvalue:.2g}"),
        "cliff < global under A (n)": int((res["A_test_cliff_rmse"] < res["A_test_rmse"]).sum()),
        "cliff < global under B (n)": int((c < g).sum()),
    }
    for k, v in res["B_descriptor"].value_counts().items():
        out[f"descriptor {k} (n)"] = int(v)
    for k, v in res["B_model"].value_counts().items():
        out[f"algorithm {k} (n)"] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="Results/rerun")
    ap.add_argument("--after", default="Results/rerun_r2")
    args = ap.parse_args()
    b_path = Path(args.before) / "per_target_results.csv"
    a_path = Path(args.after) / "per_target_results.csv"
    for p in (b_path, a_path):
        if not p.exists():
            sys.exit(f"missing {p}")
    out = Path(args.after) / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    b = pd.read_csv(b_path).drop_duplicates("target", keep="last").set_index("target")
    a = pd.read_csv(a_path).drop_duplicates("target", keep="last").set_index("target")
    common = [t for t in a.index if t in b.index]
    if len(common) < len(a):
        print(f"note: {len(a) - len(common)} target(s) only in the R2 run are ignored")
    b, a = b.loc[common], a.loc[common]

    # ---- per target -------------------------------------------------------------
    rows = []
    for t in common:
        pb, pa = b.loc[t], a.loc[t]
        cb = f"{pb.B_descriptor}-{pb.B_model}"
        ca = f"{pa.B_descriptor}-{pa.B_model}"
        rows.append({
            "Protein name": name(t), "target": t, "n cliff (test)": int(pa.n_cliff_test),
            "train cliffs, global labels": int(pa.get("n_cliff_train_global", pb.n_cliff_train)),
            "train cliffs, train-only labels": int(pa.n_cliff_train),
            "B before": cb, "B after": ca, "B changed": cb != ca,
            "B selection criterion": pa.get("B_selection_criterion", "cv_cliff_rmse"),
            "Cliff RMSE before": round(pb.B_test_cliff_rmse, 3),
            "Cliff RMSE after": round(pa.B_test_cliff_rmse, 3),
            "Δ (%) before": round(100 * (pb.B_test_cliff_rmse - pb.B_test_rmse) / pb.B_test_rmse, 1),
            "Δ (%) after": round(100 * (pa.B_test_cliff_rmse - pa.B_test_rmse) / pa.B_test_rmse, 1),
            "optimism before": round(pb.optimism_cliff_rmse, 3),
            "optimism after": round(pa.optimism_cliff_rmse, 3),
        })
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_target_before_after.csv", index=False)

    # ---- panel ------------------------------------------------------------------
    pb_, pa_ = panel(b.reset_index()), panel(a.reset_index())
    keys = list(dict.fromkeys(list(pb_) + list(pa_)))
    pan = pd.DataFrame({"statistic": keys,
                        "before (global labels)": [pb_.get(k) for k in keys],
                        "after (train-only labels)": [pa_.get(k) for k in keys]})
    pan.to_csv(out / "panel_before_after.csv", index=False)

    # ---- proposed Table S8 ------------------------------------------------------
    if "n_cliff_train_global" in a.columns:
        s8 = pd.DataFrame({
            "Protein name": [name(t) for t in common],
            "Target ID": [NAMES.get(t, (t, t))[1] for t in common],
            "n train": [int(a.loc[t, "n_train"]) for t in common],
            "Train cliff compounds (MoleculeACE labels)": [int(a.loc[t, "n_cliff_train_global"]) for t in common],
            "Train cliff compounds (train-only labels)": [int(a.loc[t, "n_cliff_train"]) for t in common],
            "Relabelled": [int(a.loc[t, "n_cliff_train_relabelled"]) for t in common],
            "Selected combination": [f"{a.loc[t, 'B_descriptor']}-{a.loc[t, 'B_model']}" for t in common],
            "Changed from previous selection": per["B changed"].tolist(),
        })
        s8.to_csv(out / "TableS8_train_cliff_labels.csv", index=False)

    # ---- text -------------------------------------------------------------------
    n_ch = int(per["B changed"].sum())
    relab = (a["n_cliff_train_relabelled"].sum() if "n_cliff_train_relabelled" in a.columns else None)
    glob_ = (a["n_cliff_train_global"].sum() if "n_cliff_train_global" in a.columns else None)
    lines = ["Before/after comparison: training cliff labels for CV model selection", "=" * 68,
             f"targets compared                         : {len(common)}"]
    if relab is not None:
        lines.append(f"training cliff labels relabelled (1 -> 0): {int(relab)} of {int(glob_)} "
                     f"({100 * relab / max(glob_, 1):.1f}%), in "
                     f"{int((a['n_cliff_train_relabelled'] > 0).sum())} targets")
    lines += [f"protocol-B selection changed             : {n_ch} of {len(common)} targets"]
    if n_ch:
        lines.append("  " + "; ".join(f"{r['Protein name']} {r['B before']} -> {r['B after']}"
                                     for _, r in per[per["B changed"]].iterrows()))
    fb = per["B selection criterion"].ne("cv_cliff_rmse")
    if fb.any():
        lines.append("targets selected on global CV RMSE (no train-only cliff compound): "
                     + ", ".join(per.loc[fb, "Protein name"]))
    lines += ["", pan.to_string(index=False)]
    text = "\n".join(lines)
    (out / "comparison.txt").write_text(text, encoding="utf-8")
    print(text + f"\n\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()
