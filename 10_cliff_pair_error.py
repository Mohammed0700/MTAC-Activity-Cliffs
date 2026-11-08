"""Table S10 and Section 2.11: direction and magnitude of the prediction error
within an activity-cliff pair.

An activity-cliff pair is counted when both of its members fall in the test set
of the same target, so that the two predictions come from the same model and are
directly comparable. Predictions are those of the descriptor and algorithm
combination selected for that target by cross-validation inside the training set
(column B of per_target_results.csv).

For each pair the member with the lower Ki is the more potent one. Signed error
is the predicted minus the measured pKi, so a negative value on the potent member
means the model placed it lower than it is.

Usage:
    python 10_cliff_pair_error.py DATA_DIR RESULTS_DIR NAMES_JSON OUT_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtac_cliff_api import (CLIFF_POTENCY_FOLD, CLIFF_SIMILARITY, get_fc,
                            get_levenshtein_matrix, get_scaffold_matrix,
                            get_tanimoto_matrix)


def similarity_matrices(smiles):
    """Return (count, best) for every pair.

    similar is MoleculeACE's own test, a pair being similar when any one of the
    three measures reaches the threshold. It is written here the way
    moleculeace_similarity writes it, as the elementwise union of the three
    boolean matrices, so that the criterion is identical to the one used to
    label the activity cliffs themselves.

    best is the highest of the three coefficients, used only as a continuous
    measure of how alike the two structures are.
    """
    t = get_tanimoto_matrix(smiles, hide=True)
    s = get_scaffold_matrix(smiles, hide=True)
    l = get_levenshtein_matrix(smiles, hide=True)
    similar = ((t >= CLIFF_SIMILARITY) + (s >= CLIFF_SIMILARITY)
               + (l >= CLIFF_SIMILARITY)).astype(int)
    return similar, np.maximum(np.maximum(t, s), l)


def main():
    data_dir, res_dir, names_json, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    names = json.load(open(names_json))
    per_target = pd.read_csv(os.path.join(res_dir, "per_target_results.csv"))
    preds_all = pd.read_csv(os.path.join(res_dir, "test_predictions.csv"))

    rows, pair_rows = [], []
    for t in sorted(names):
        df = pd.read_csv(os.path.join(data_dir, f"{t}_prepared.csv"))
        te = df[df["split"] == "test"].reset_index(drop=True)
        sel = per_target[per_target.target == t].iloc[0]
        p = preds_all[(preds_all.target == t) & (preds_all.descriptor == sel.B_descriptor)
                      & (preds_all.model == sel.B_model)]
        pred = dict(zip(p.smiles, p.y_pred))
        if len(p) == 0 or not set(te.smiles).issubset(pred):
            print(f"  {t:22s} no matching predictions, skipped", flush=True)
            continue

        smi = te.smiles.tolist()
        ki = te["exp_mean [nM]"].to_numpy(dtype=float)
        y = te["y"].to_numpy(dtype=float)
        yp = np.array([pred[s] for s in smi], dtype=float)
        err = yp - y

        similar, sim = similarity_matrices(smi)
        fc = get_fc(ki)
        iu = np.triu_indices(len(smi), k=1)
        keep = (similar[iu] == 1) & (fc[iu] > CLIFF_POTENCY_FOLD)
        I, J = iu[0][keep], iu[1][keep]
        if len(I) == 0:
            print(f"  {t:22s} no qualifying pair", flush=True)
            continue

        # the more potent member is the one with the lower Ki
        potent = np.where(ki[I] <= ki[J], I, J)
        weak = np.where(ki[I] <= ki[J], J, I)
        e_pot, e_weak = err[potent], err[weak]
        a_pot, a_weak = np.abs(e_pot), np.abs(e_weak)
        ratio = np.maximum(a_pot, a_weak) / np.maximum(np.minimum(a_pot, a_weak), 1e-12)

        rows.append(dict(
            target=t, protein=names[t], n_pairs=len(I),
            potent_under_pct=100 * float((e_pot < 0).mean()),
            weak_over_pct=100 * float((e_weak > 0).mean()),
            mean_signed_potent=float(e_pot.mean()),
            mean_signed_weak=float(e_weak.mean()),
            median_error_ratio=float(np.median(ratio))))
        pair_rows.append(pd.DataFrame(dict(
            target=t, protein=names[t],
            smiles_potent=[smi[k] for k in potent], smiles_weak=[smi[k] for k in weak],
            signed_error_potent=e_pot, signed_error_weak=e_weak,
            delta_pki=np.abs(y[potent] - y[weak]),
            pair_similarity=sim[I, J], mean_abs_error=(a_pot + a_weak) / 2)))
        print(f"  {t:22s} {len(I):5d} pairs", flush=True)

    tab = pd.DataFrame(rows).sort_values("protein")
    pairs = pd.concat(pair_rows, ignore_index=True)
    tab.to_csv(os.path.join(out_dir, "TableS10_cliff_pair_error.csv"), index=False)
    pairs.to_csv(os.path.join(out_dir, "TableS10_pairs_long.csv"), index=False)

    # ---------------------------------------------------------- panel statistics
    ep, ew = pairs.signed_error_potent.to_numpy(), pairs.signed_error_weak.to_numpy()
    ap, aw = np.abs(ep), np.abs(ew)
    w = stats.wilcoxon(ep, ew)
    both = float(((ep < 0) & (ew > 0)).mean())
    split = float((((ap < 0.5) & (aw > 1.0)) | ((aw < 0.5) & (ap > 1.0))).mean())
    ratio = np.maximum(ap, aw) / np.maximum(np.minimum(ap, aw), 1e-12)
    r_step = stats.spearmanr(pairs.mean_abs_error, pairs.delta_pki)
    r_sim = stats.spearmanr(pairs.mean_abs_error, 1 - pairs.pair_similarity)

    lines = [
        f"pairs                               {len(pairs)}",
        f"targets with at least one pair      {tab.shape[0]}",
        f"potent member under-predicted       {100*(ep<0).mean():.1f}%",
        f"weak member over-predicted          {100*(ew>0).mean():.1f}%",
        f"both in the same pair               {100*both:.1f}%",
        f"mean signed error, potent member    {ep.mean():+.3f}",
        f"mean signed error, weak member      {ew.mean():+.3f}",
        f"Wilcoxon signed-rank                p = {w.pvalue:.3g}",
        f"median ratio of the two |errors|    {np.median(ratio):.2f}",
        f"one member <0.5 and partner >1.0    {100*split:.1f}%",
        f"Spearman |error| vs potency step    rho = {r_step.statistic:+.2f}, p = {r_step.pvalue:.3g}",
        f"Spearman |error| vs dissimilarity   rho = {r_sim.statistic:+.2f}, p = {r_sim.pvalue:.3g}",
    ]
    text = "\n".join(lines)
    open(os.path.join(out_dir, "TableS10_panel_statistics.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
