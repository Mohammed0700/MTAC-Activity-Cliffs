"""How each learner responds to refitting the descriptor normalisation per fold.

Support vector machines and nearest neighbours work on distances between feature
vectors, so rescaling the columns changes what they compute.

Tree ensembles might be expected not to respond. They split on a threshold in
one feature at a time, and a per-column affine map with a positive scale
preserves the order of the values within every column, so in exact arithmetic
the candidate splits correspond one to one and the fitted trees are identical.
In floating point that is not what happens. The rescaling perturbs the computed
impurity sums in the last digits, so a split that is close to a tie can be
decided the other way, and once one split differs the trees diverge.

This script measures the response of each learner rather than arguing about it.
For each target it builds the physicochemical descriptors, forms the matrix the
deposited run cross-validated, and compares the pooled out-of-fold predictions
with and without the per-fold refit. For the random forest it repeats the
comparison across ensemble sizes, since averaging over more trees is the obvious
candidate for suppressing a tie-breaking artefact.

What the output shows is that all four learners move. The random forest moves
less as the ensemble grows, which is consistent with a tie-breaking origin, but
the movement does not disappear. No learner can therefore be excluded from the
measurement on an invariance argument, which is why 13_scaler_refit_effect.py
rescores all four.

Usage:
    python 14_scaling_invariance_check.py DATA_DIR CLIFFS_DIR OUT_CSV [TARGET ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaler_effect_lib import HYPERPARAMS, folds_for, physchem, rmse

ALGORITHMS = {"RF": RandomForestRegressor, "SVM": SVR,
              "GBM": GradientBoostingRegressor, "KNN": KNeighborsRegressor}
TREE_COUNTS = (10, 50, 100, 500)
DEFAULT_TARGETS = ("CHEMBL1827", "CHEMBL1907601", "CHEMBL267", "CHEMBL2695",
                   "CHEMBL331", "CHEMBL4282", "CHEMBL4501", "CHEMBL5023")


def oof(estimator, params, x, y, folds, rescale):
    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        x_tr, x_va = x[tr], x[va]
        if rescale:
            sc = StandardScaler().fit(x_tr)
            x_tr, x_va = sc.transform(x_tr), sc.transform(x_va)
        m = estimator(**params)
        m.fit(x_tr, y[tr])
        pred[va] = m.predict(x_va)
    return pred


def load(target, data_dir, cliffs_dir):
    df = pd.read_csv(os.path.join(data_dir, f"{target}_prepared.csv"))
    tr = df[df["split"] == "train"].reset_index(drop=True)
    lab = pd.read_csv(os.path.join(cliffs_dir, f"{target}_train_only_cliffs.csv"))
    if list(lab["smiles"]) != list(tr["smiles"]):
        raise RuntimeError(f"{target}: training compounds are not in the deposited order")
    cliff = lab["cliff_mol_train_only"].to_numpy().astype(int)
    y = tr["y"].to_numpy(dtype=float)
    x = StandardScaler().fit_transform(physchem(tr["smiles"]))
    folds, _ = folds_for(cliff)
    return x, y, cliff.astype(bool), folds


def main():
    data_dir, cliffs_dir, out_csv = sys.argv[1:4]
    targets = sys.argv[4:] or list(DEFAULT_TARGETS)

    rows = []
    print("per-learner effect of the per-fold refit, physicochemical descriptors")
    for t in targets:
        x, y, m, folds = load(t, data_dir, cliffs_dir)
        for algo, est in ALGORITHMS.items():
            p = dict(HYPERPARAMS[f"{algo}_PHYSCHEM"])
            a = oof(est, p, x, y, folds, rescale=False)
            b = oof(est, p, x, y, folds, rescale=True)
            rows.append(dict(target=t, check="per_learner", model=algo,
                             n_estimators=p.get("n_estimators"),
                             max_abs_prediction_difference=float(np.abs(a - b).max()),
                             mean_abs_prediction_difference=float(np.abs(a - b).mean()),
                             cliff_rmse_difference=rmse(y[m], b[m]) - rmse(y[m], a[m])))
            print(f"  {t:16s} {algo:4s} max |Δ| = {np.abs(a-b).max():.3e}   "
                  f"mean |Δ| = {np.abs(a-b).mean():.3e}", flush=True)

    print("\nrandom forest, same comparison over ensemble size")
    for t in targets:
        x, y, m, folds = load(t, data_dir, cliffs_dir)
        for n in TREE_COUNTS:
            p = dict(HYPERPARAMS["RF_PHYSCHEM"])
            p["n_estimators"] = n
            a = oof(RandomForestRegressor, p, x, y, folds, rescale=False)
            b = oof(RandomForestRegressor, p, x, y, folds, rescale=True)
            rows.append(dict(target=t, check="rf_ensemble_size", model="RF",
                             n_estimators=n,
                             max_abs_prediction_difference=float(np.abs(a - b).max()),
                             mean_abs_prediction_difference=float(np.abs(a - b).mean()),
                             cliff_rmse_difference=rmse(y[m], b[m]) - rmse(y[m], a[m])))
        print(f"  {t:16s} " + "  ".join(
            f"{n}:{rows[-len(TREE_COUNTS)+i]['mean_abs_prediction_difference']:.4f}"
            for i, n in enumerate(TREE_COUNTS)), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    per = out[out.check == "per_learner"]
    print("\nlargest difference in any out-of-fold prediction, by learner")
    for k, v in per.groupby("model").max_abs_prediction_difference.max().items():
        print(f"  {k:4s} {v:.3e}")
    ens = out[out.check == "rf_ensemble_size"].groupby("n_estimators")[
        ["max_abs_prediction_difference", "mean_abs_prediction_difference"]].mean()
    print("\nrandom forest against ensemble size, averaged over targets")
    for n, r in ens.iterrows():
        print(f"  {int(n):4d} trees   max {r.max_abs_prediction_difference:.4f}"
              f"   mean {r.mean_abs_prediction_difference:.4f}")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
