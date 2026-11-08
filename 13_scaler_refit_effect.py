"""Table S13: effect of refitting the descriptor normalisation inside each
cross-validation fold, and whether it changes the selected model.

MoleculeACE fits a StandardScaler on the whole training set at featurisation
time. The model-selection script as first deposited (tag results-as-published)
passed that matrix straight into the cross-validation loop, so the normalisation
carried information from the validation fold into the training fold. The current
script refits the scaler on each training fold.

Because the normalisation is a per-column affine map, refitting a StandardScaler
on the already-normalised matrix is exactly equivalent to refitting it on the raw
descriptors, so the corrected arm can be computed without re-featurising and
without regenerating conformers.

For every affected combination this script computes the pooled out-of-fold Cliff
RMSE under both treatments, using the deposited train-only cliff labels and the
same folds, seeds and hyperparameters as the model-selection run, and then
re-runs the per-target selection with the corrected scores substituted.

Only PHYSCHEM and WHIM are normalised, so eight of the sixteen combinations per
target are affected and the binary fingerprints are untouched. All four learners
are measured. Support vector machines and nearest neighbours work on distances
and respond to scaling directly. Tree ensembles split on one feature at a time
and are invariant to a per-column affine map in exact arithmetic, but the
rescaling perturbs the computed impurity sums at the level of floating-point
rounding, so a split that is close to a tie can be decided the other way and the
fitted trees then diverge. 14_scaling_invariance_check.py measures how each
learner responds and finds that all four move by amounts of the same order, so
all four are measured here rather than argued about.

PHYSCHEM is evaluated at every target. WHIM, which needs a conformer for every
molecule, is evaluated at every target where a WHIM combination is close enough
to the selected one to be able to displace it; elsewhere the gap is reported so
that the bound is explicit.

Usage:
    python 13_scaler_refit_effect.py DATA_DIR RESULTS_DIR NAMES_JSON OUT_DIR
                                     [WHIM_GAP_THRESHOLD]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaler_effect_lib import (HYPERPARAMS, SCALED_DESCRIPTORS, folds_for,
                               physchem, rmse, whim)

ALGORITHMS = {"RF": RandomForestRegressor, "SVM": SVR,
              "GBM": GradientBoostingRegressor, "KNN": KNeighborsRegressor}
DEFAULT_WHIM_GAP = 0.25


def oof(algo, params, x, y, folds, rescale):
    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        x_tr, x_va = x[tr], x[va]
        if rescale:
            sc = StandardScaler().fit(x_tr)
            x_tr, x_va = sc.transform(x_tr), sc.transform(x_va)
        model = ALGORITHMS[algo](**params)
        model.fit(x_tr, y[tr])
        pred[va] = model.predict(x_va)
    return pred


def measure(target, desc_name, data_dir, cliffs_dir, cache_path):
    df = pd.read_csv(os.path.join(data_dir, f"{target}_prepared.csv"))
    tr = df[df["split"] == "train"].reset_index(drop=True)
    lab = pd.read_csv(os.path.join(cliffs_dir, f"{target}_train_only_cliffs.csv"))
    if list(lab["smiles"]) != list(tr["smiles"]):
        raise RuntimeError(f"{target}: training compounds are not in the deposited order")
    cliff = lab["cliff_mol_train_only"].to_numpy().astype(int)
    y = tr["y"].to_numpy(dtype=float)
    x_raw = (physchem(tr["smiles"]) if desc_name == "PHYSCHEM"
             else whim(tr["smiles"], cache_path))

    # the matrix the deposited run cross-validated
    x = StandardScaler().fit_transform(x_raw)
    folds, strat = folds_for(cliff)
    m = cliff.astype(bool)

    rows = []
    for algo in ALGORITHMS:
        p = dict(HYPERPARAMS[f"{algo}_{desc_name}"])
        before = oof(algo, p, x, y, folds, rescale=False)
        after = oof(algo, p, x, y, folds, rescale=True)
        rows.append(dict(
            target=target, descriptor=desc_name, model=algo, n_train=len(y),
            n_cliff_train=int(cliff.sum()), cv_stratified=strat,
            cv_cliff_rmse_before=rmse(y[m], before[m]),
            cv_cliff_rmse_after=rmse(y[m], after[m]),
            cv_rmse_before=rmse(y, before), cv_rmse_after=rmse(y, after)))
    return rows


def main():
    data_dir, res_dir, names_json, out_dir = sys.argv[1:5]
    whim_gap = float(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_WHIM_GAP
    # "summarise" repeats only the reporting, reading the measurement already written
    summarise_only = len(sys.argv) > 6 and sys.argv[6] == "summarise"
    os.makedirs(out_dir, exist_ok=True)
    cliffs_dir = os.path.join(res_dir, "train_only_cliffs")
    names = json.load(open(names_json))
    grid = pd.read_csv(os.path.join(res_dir, "full_grid.csv"))
    grid["affected"] = grid.descriptor.isin(SCALED_DESCRIPTORS)
    eff_path = os.path.join(out_dir, "TableS13_scaler_refit_effect.csv")

    if summarise_only:
        eff = pd.read_csv(eff_path)
    else:
        # WHIM is measured where it could still displace the selected combination
        whim_targets = []
        for t, d in grid.groupby("target"):
            w = d[d.descriptor == "WHIM"]
            if len(w) and (w.cv_cliff_rmse.min() - d.cv_cliff_rmse.min()) < whim_gap:
                whim_targets.append(t)

        rows = []
        for t in sorted(names):
            for desc_name in SCALED_DESCRIPTORS:
                if desc_name == "WHIM" and t not in whim_targets:
                    continue
                if not len(grid[(grid.target == t) & (grid.descriptor == desc_name)]):
                    continue
                t0 = time.time()
                rows += measure(t, desc_name, data_dir, cliffs_dir,
                                os.path.join(out_dir, f"{desc_name}_conformers.pkl"))
                print(f"  {t:22s} {desc_name:9s} {time.time()-t0:6.1f}s", flush=True)

        eff = pd.DataFrame(rows)
        eff["delta_cliff_rmse"] = eff.cv_cliff_rmse_after - eff.cv_cliff_rmse_before
        eff.insert(1, "protein", eff.target.map(names))
        eff.to_csv(eff_path, index=False)

    # the uncorrected arm must reproduce the deposited grid
    chk = eff.merge(grid, on=["target", "descriptor", "model"])
    repro = float((chk.cv_cliff_rmse_before - chk.cv_cliff_rmse).abs().max())

    # ------------------------------------------------- selection under correction
    after = {(r.target, r.descriptor, r.model): r.cv_cliff_rmse_after
             for r in eff.itertuples()}
    grid["cv_cliff_rmse_corrected"] = [
        after.get((r.target, r.descriptor, r.model), r.cv_cliff_rmse)
        for r in grid.itertuples()]
    grid["measured"] = [(r.target, r.descriptor, r.model) in after
                        for r in grid.itertuples()]

    shift = eff.groupby("target").delta_cliff_rmse.apply(lambda s: s.abs().max())
    n_meas = eff.groupby("target").size()

    sel = []
    for t, d in grid.groupby("target"):
        b = d.loc[d.cv_cliff_rmse.idxmin()]
        a = d.loc[d.cv_cliff_rmse_corrected.idxmin()]
        un = d[d.affected & ~d.measured]
        sel.append(dict(
            target=t, protein=names[t],
            selected_before=f"{b.descriptor}-{b.model}",
            cv_cliff_rmse_before=b.cv_cliff_rmse,
            n_affected_measured=int(n_meas.get(t, 0)),
            largest_shift=float(shift.get(t, np.nan)),
            selected_after=f"{a.descriptor}-{a.model}",
            cv_cliff_rmse_after=a.cv_cliff_rmse_corrected,
            changed=(b.descriptor, b.model) != (a.descriptor, a.model),
            margin_to_best_affected=float(
                d[d.affected].cv_cliff_rmse.min() - d.cv_cliff_rmse.min()),
            gap_of_unmeasured=float((un.cv_cliff_rmse - d.cv_cliff_rmse_corrected.min()).min())
            if len(un) else np.nan))
    sel = pd.DataFrame(sel).sort_values("protein")
    sel.to_csv(os.path.join(out_dir, "TableS13_selection_under_correction.csv"), index=False)

    # the sentence quoted in the Limitations, written from the measurement itself
    ch = sel[sel.changed]
    if len(ch) == 0:
        sentence = ("Substituting the corrected scores leaves the combination selected "
                    "for every one of the 33 targets unchanged.")
    else:
        READABLE = {"RF": "a random forest", "SVM": "a support vector machine",
                    "GBM": "gradient boosting", "KNN": "nearest neighbours"}
        def phrase(combo):
            desc, mod = combo.split("-")
            return f"{desc} with {READABLE.get(mod, mod)}"
        bits = "; ".join(f"{r.protein}, where {phrase(r.selected_after)} "
                         f"replaces {phrase(r.selected_before)}"
                         for _, r in ch.iterrows())
        WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        n_word = WORDS.get(len(ch), str(len(ch)))
        sentence = (f"Substituting the corrected scores changes the combination selected "
                    f"for {n_word} of the 33 targets, {bits}.")

    # what that substitution would do to the reported panel figures. Test-set
    # scores are unaffected by the correction, so the corrected panel is read
    # straight out of the deposited grid at the newly selected combinations.
    per = pd.read_csv(os.path.join(res_dir, "per_target_results.csv"))
    look = {(r.target, r.descriptor, r.model): r for r in grid.itertuples()}
    cor = per.copy()
    for _, r in ch.iterrows():
        desc, mod = r.selected_after.split("-")
        g = look[(r.target, desc, mod)]
        i = cor.index[cor.target == r.target][0]
        cor.loc[i, ["B_descriptor", "B_model", "B_test_rmse", "B_test_cliff_rmse"]] = \
            [desc, mod, g.test_rmse, g.test_cliff_rmse]

    def panel(df):
        delta = 100 * (df.B_test_cliff_rmse - df.B_test_rmse) / df.B_test_rmse
        return (float(delta.mean()),
                float((df.B_test_cliff_rmse - df.A_test_cliff_rmse).mean()),
                int((df.B_test_cliff_rmse > df.B_test_rmse).sum()))
    d0, o0, c0 = panel(per)
    d1, o1, c1 = panel(cor)
    if len(ch) == 0:
        consequence = "No reported value would change."
    else:
        parts = []
        for _, r in ch.iterrows():
            g = look[(r.target, *r.selected_after.split("-"))]
            was = per[per.target == r.target].iloc[0]
            parts.append(f"the cliff-specific test error of {r.protein} would move from "
                         f"{was.B_test_cliff_rmse:.3f} to {g.test_cliff_rmse:.3f} pKi units")
        consequence = (f"With that substitution {', and '.join(parts)}, the panel mean "
                       f"increase in cliff-specific error would move from {d0:.1f}% to "
                       f"{d1:.1f}%, the mean selection optimism from {o0:.3f} to {o1:.3f} "
                       f"pKi units, and cliff error would still exceed global error for "
                       f"{c1} of the 33 targets.")
    cor.to_csv(os.path.join(out_dir, "per_target_results_under_correction.csv"), index=False)
    with open(os.path.join(out_dir, "changed.json"), "w") as fh:
        json.dump({"n_changed": int(len(ch)), "sentence": sentence,
                   "consequence": consequence,
                   "mean_delta_before": d0, "mean_delta_after": d1,
                   "mean_optimism_before": o0, "mean_optimism_after": o1,
                   "cliff_gt_global_before": c0, "cliff_gt_global_after": c1},
                  fh, indent=1)

    worse = int((eff.delta_cliff_rmse > 0).sum())
    lines = [
        f"affected combinations measured            {len(eff)}",
        f"  PHYSCHEM                                {int((eff.descriptor=='PHYSCHEM').sum())}",
        f"  WHIM                                    {int((eff.descriptor=='WHIM').sum())}",
        f"uncorrected arm vs deposited full_grid    max difference {repro:.2e}",
        "",
        f"shift in pooled out-of-fold Cliff RMSE",
        f"  median |shift|                          {eff.delta_cliff_rmse.abs().median():.4f}",
        f"  95th percentile |shift|                 {eff.delta_cliff_rmse.abs().quantile(.95):.4f}",
        f"  largest |shift|                         {eff.delta_cliff_rmse.abs().max():.4f}",
        f"  score worsens                           {worse} of {len(eff)} combinations",
        f"  score improves                          {len(eff)-worse} of {len(eff)} combinations",
        "",
        f"targets whose selected combination changes {int(sel.changed.sum())} of {len(sel)}",
    ]
    for _, r in sel[sel.changed].iterrows():
        lines.append(f"  {r.protein}: {r.selected_before} -> {r.selected_after}")
    lines += [
        "",
        f"smallest gap for an affected combination not measured "
        f"{sel.gap_of_unmeasured.min():.3f}",
        f"  that is {sel.gap_of_unmeasured.min()/eff.delta_cliff_rmse.abs().max():.1f} times the "
        f"largest shift observed anywhere, so those combinations cannot displace a selection",
    ]
    text = "\n".join(lines)
    open(os.path.join(out_dir, "TableS13_panel_statistics.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
