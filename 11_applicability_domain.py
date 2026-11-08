"""Table S11 and Section 2.12: distance to the training set and the
concentration of cliff-specific error.

For every test compound the nearest-neighbour similarity is the highest
extended-connectivity fingerprint Tanimoto coefficient to any training compound
of the same target, computed with the fingerprint settings used throughout
(radius 2, 1024 bits). Absolute errors are those of the descriptor and algorithm
combination selected for that target by cross-validation inside the training set.

Targets with fewer than three activity-cliff compounds in the test set are
omitted. The cliff-only correlation is left blank where there are fewer than
eight cliff compounds.

Usage:
    python 11_applicability_domain.py DATA_DIR RESULTS_DIR NAMES_JSON OUT_DIR
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import stats

RDLogger.DisableLog("rdApp.*")

MIN_CLIFFS_TABLE = 3
MIN_CLIFFS_RHO = 8


def fingerprints(smiles, radius=2, n_bits=1024):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(None if m is None else
                   AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits))
    return out


def nearest_neighbour_similarity(fp_test, fp_train):
    train = [f for f in fp_train if f is not None]
    out = np.full(len(fp_test), np.nan)
    for i, f in enumerate(fp_test):
        if f is None or not train:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(f, train))
    return out


def main():
    data_dir, res_dir, names_json, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    names = json.load(open(names_json))
    per_target = pd.read_csv(os.path.join(res_dir, "per_target_results.csv"))
    preds_all = pd.read_csv(os.path.join(res_dir, "test_predictions.csv"))

    rows = []
    for t in sorted(names):
        df = pd.read_csv(os.path.join(data_dir, f"{t}_prepared.csv"))
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"].reset_index(drop=True)
        sel = per_target[per_target.target == t].iloc[0]
        p = preds_all[(preds_all.target == t) & (preds_all.descriptor == sel.B_descriptor)
                      & (preds_all.model == sel.B_model)]
        pred = dict(zip(p.smiles, p.y_pred))
        if not set(te.smiles).issubset(pred):
            print(f"  {t:22s} no matching predictions, skipped", flush=True)
            continue

        cliff = te.cliff_mol.to_numpy().astype(bool)
        if cliff.sum() < MIN_CLIFFS_TABLE:
            print(f"  {t:22s} only {int(cliff.sum())} cliff compounds, omitted", flush=True)
            continue

        y = te["y"].to_numpy(dtype=float)
        yp = np.array([pred[s] for s in te.smiles], dtype=float)
        ae = np.abs(yp - y)
        nn = nearest_neighbour_similarity(fingerprints(te.smiles.tolist()),
                                          fingerprints(tr.smiles.tolist()))
        ok = ~np.isnan(nn)

        r_all = stats.spearmanr(nn[ok], ae[ok])
        if int(cliff.sum()) >= MIN_CLIFFS_RHO:
            sub = ok & cliff
            r_cl = stats.spearmanr(nn[sub], ae[sub])
            rho_cliff, p_cliff = float(r_cl.statistic), float(r_cl.pvalue)
        else:
            rho_cliff, p_cliff = np.nan, np.nan

        # share of the cliff error carried by the worst tenth of cliff compounds
        ce = np.sort(ae[cliff])[::-1]
        k = max(1, int(math.ceil(0.10 * len(ce))))
        share = float(ce[:k].sum() / ce.sum()) if ce.sum() > 0 else np.nan

        rows.append(dict(
            target=t, protein=names[t], n_test=len(te), n_cliff=int(cliff.sum()),
            nn_sim_cliff=float(np.nanmean(nn[cliff])),
            nn_sim_noncliff=float(np.nanmean(nn[~cliff])),
            mean_abs_err_cliff=float(ae[cliff].mean()),
            mean_abs_err_noncliff=float(ae[~cliff].mean()),
            rho_all=float(r_all.statistic), p_all=float(r_all.pvalue),
            rho_cliff=rho_cliff, p_cliff=p_cliff,
            share_worst_decile=share))
        print(f"  {t:22s} n_test={len(te):5d} cliff={int(cliff.sum()):4d}", flush=True)

    tab = pd.DataFrame(rows).sort_values("protein")
    tab.to_csv(os.path.join(out_dir, "TableS11_applicability_domain.csv"), index=False)

    w = stats.wilcoxon(tab.nn_sim_cliff, tab.nn_sim_noncliff)
    sig_all = int((tab.p_all < 0.05).sum())
    neg_all = int((tab.rho_all < 0).sum())
    cl = tab.dropna(subset=["rho_cliff"])
    lines = [
        f"targets with at least {MIN_CLIFFS_TABLE} test cliff compounds   {len(tab)}",
        f"mean nearest-neighbour similarity, cliff      {tab.nn_sim_cliff.mean():.3f}",
        f"mean nearest-neighbour similarity, non-cliff  {tab.nn_sim_noncliff.mean():.3f}",
        f"targets where cliff exceeds non-cliff         {int((tab.nn_sim_cliff > tab.nn_sim_noncliff).sum())} of {len(tab)}",
        f"Wilcoxon signed-rank                          p = {w.pvalue:.3g}",
        f"rho(NN, |error|) negative                     {neg_all} of {len(tab)} targets",
        f"rho(NN, |error|) significant at 0.05          {sig_all} of {len(tab)} targets",
        f"mean rho(NN, |error|), all compounds          {tab.rho_all.mean():+.2f}",
        f"mean rho(NN, |error|), cliff compounds        {cl.rho_cliff.mean():+.2f}",
        f"cliff-only rho significant                    {int((cl.p_cliff < 0.05).sum())} of {len(cl)} targets",
        f"share of cliff error in the worst tenth       mean {100*tab.share_worst_decile.mean():.1f}%, "
        f"median {100*tab.share_worst_decile.median():.1f}%",
    ]
    text = "\n".join(lines)
    open(os.path.join(out_dir, "TableS11_panel_statistics.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
