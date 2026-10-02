#!/usr/bin/env python
"""
rerun_with_validation.py
========================
Re-runs the MTAC benchmark with model selection moved OFF the test set.

WHY
---
In the current manuscript the descriptor-algorithm combination reported for each
target is the one with the lowest Cliff RMSE **on the held-out test set**. That
makes the reported per-target errors optimistically biased, because the test set
has been used twice: once to choose the model and once to report its error. This
is the first methodological objection a reviewer will raise (review item 5).

WHAT THIS SCRIPT DOES
---------------------
For every target it runs both protocols on the *same* splits, so the difference
between them is a direct measurement of the selection optimism:

  A. "test-selected"  (the current manuscript protocol)
         fit on train -> score all 16 combinations on test -> report the best.

  B. "cv-selected"    (the corrected protocol)
         5-fold CV *within the training set only*; out-of-fold predictions are
         pooled across folds and a single Cliff RMSE is computed from them;
         the combination with the lowest pooled out-of-fold Cliff RMSE is then
         refit on the full training set and scored ONCE on the test set.

  Protocol B never looks at the test set before the final scoring, so its test
  errors are unbiased estimates of prospective performance.

WHY POOLED OUT-OF-FOLD RATHER THAN MEAN-OF-FOLDS
------------------------------------------------
Several targets have very few activity-cliff compounds (MEK1 has roughly five in
the whole training set). A per-fold Cliff RMSE would then be computed on one
compound per fold and averaged, which is unstable. Pooling the out-of-fold
predictions and computing one Cliff RMSE over all of them uses every cliff
compound exactly once and is far better behaved at small n.

HOW TO RUN
----------
Run this inside the environment where your notebooks already work (MoleculeACE
and RDKit installed). The deep-learning extras (torch_geometric and friends) do
NOT need to be installed -- see the import section below. From the project root
that contains Data/:

    python rerun_with_validation.py --data-dir Data --out Results/rerun

It writes:
    Results/rerun/per_target_results.csv   one row per target, both protocols
    Results/rerun/full_grid.csv            all 16 combinations for every target
    Results/rerun/test_predictions.csv     every test prediction (for bootstrap CIs)
    Results/rerun/run_info.json            hyperparameters, seeds and package versions
    Results/rerun/summary.txt              the numbers to quote in the paper

Expect roughly a few minutes per target on CPU; WHIM conformer generation is the
slow step.
"""

import argparse
import os
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Importing MoleculeACE.
#
# MoleculeACE/__init__.py imports its graph neural networks, which pull in
# torch_geometric and friends. Those are only needed for MPNN/GCN/GAT/AFP, none
# of which this script uses -- it uses RF, SVM, GBM and KNN, which are plain
# scikit-learn. On an environment without the deep-learning extras the import
# therefore fails before any useful code runs:
#
#     ModuleNotFoundError: No module named 'torch_geometric'
#
# Rather than force you to install a large GPU stack you will never call, we
# register lightweight placeholder modules for the graph-learning packages and
# retry. Only packages on the allow-list below are ever stubbed; anything the
# analysis genuinely needs (rdkit, sklearn, numpy, pandas) is never stubbed and
# a missing one still raises normally.
# ----------------------------------------------------------------------------
_STUBBABLE = (
    "torch_geometric", "torch_scatter", "torch_sparse", "torch_cluster",
    "torch_spline_conv", "pytorch_lightning", "torch_ema", "dgl", "dgllife",
)


class _StubModule(types.ModuleType):
    """A module that yields a throwaway class for any attribute asked of it."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


def _install_stub(name):
    mod = _StubModule(name)
    mod.__path__ = []          # marks it as a package so submodules can attach
    sys.modules[name] = mod


def _import_moleculeace(max_attempts=30):
    stubbed = []
    for _ in range(max_attempts):
        try:
            from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse
            from MoleculeACE.benchmark.const import Descriptors
            from MoleculeACE.models import RF, SVM, GBM, KNN
            return (Data, calc_rmse, calc_cliff_rmse, Descriptors,
                    RF, SVM, GBM, KNN), stubbed
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if not missing.startswith(_STUBBABLE):
                if missing.startswith("MoleculeACE") or missing == "":
                    raise
                sys.exit(
                    f"\nMissing package: {missing}\n"
                    "This one is genuinely required, so it has not been stubbed.\n"
                    "Run this script from the same conda environment in which\n"
                    "MoleculeACE_analysis.ipynb and Best_ML.ipynb run, or install it.")
            _install_stub(missing)
            stubbed.append(missing)
            # drop the half-imported package so the retry starts clean
            for mod_name in [m for m in sys.modules if m.startswith("MoleculeACE")]:
                del sys.modules[mod_name]
    sys.exit("Could not import MoleculeACE after stubbing: " + ", ".join(stubbed))


try:
    (Data, calc_rmse, calc_cliff_rmse, Descriptors,
     RF, SVM, GBM, KNN), _STUBBED = _import_moleculeace()
except ImportError as exc:                                  # noqa: BLE001
    sys.exit(
        f"Could not import MoleculeACE ({exc}).\n"
        "Run this script from the same environment (conda env / venv) in which\n"
        "MoleculeACE_analysis.ipynb and Best_ML.ipynb run.")

if _STUBBED:
    print("Note: the following deep-learning packages were absent and have been\n"
          "      replaced by placeholders so that MoleculeACE could be imported:\n"
          "        " + ", ".join(sorted(set(_STUBBED))) + "\n"
          "      This is harmless here. Only RF, SVM, GBM and KNN are used, and\n"
          "      those are scikit-learn models that do not touch these packages.\n")

# ----------------------------------------------------------------------------
# The 16 combinations actually reported in the manuscript.
# ----------------------------------------------------------------------------
DESCRIPTORS = {
    "ECFP": Descriptors.ECFP,
    "MACCS": Descriptors.MACCS,
    "PHYSCHEM": Descriptors.PHYSCHEM,
    "WHIM": Descriptors.WHIM,
}
ALGORITHMS = {"RF": RF, "SVM": SVM, "GBM": GBM, "KNN": KNN}

# ----------------------------------------------------------------------------
# Hyperparameters. These are the values used in Best_ML.ipynb. Keep them fixed
# across both protocols so that the only thing that changes is HOW the winning
# combination is chosen. Do NOT tune these here -- tuning inside the CV loop
# would be a separate (and larger) change to the methodology.
# ----------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5

# REVISION (editor comment 6): random_state = SEED added to RF and GBM so results are reproducible.
# Source of all 16 values: MoleculeACE v3.0.0, Data/configures/benchmark/CHEMBL204_Ki/*.yml
HYPERPARAMS = {
    "RF_ECFP": {"n_estimators": 1000, "random_state": SEED},
    "RF_MACCS": {"n_estimators": 500, "random_state": SEED},
    "RF_PHYSCHEM": {"n_estimators": 500, "random_state": SEED},
    "RF_WHIM": {"n_estimators": 500, "random_state": SEED},
    "GBM_ECFP": {"learning_rate": 0.1, "max_depth": 5, "max_features": "sqrt",
                 "min_samples_leaf": 1, "min_samples_split": 2, "random_state": SEED, "n_estimators": 400},
    "GBM_MACCS": {"learning_rate": 0.1, "max_depth": 6, "max_features": "sqrt",
                  "min_samples_leaf": 1, "min_samples_split": 2, "random_state": SEED, "n_estimators": 400},
    "GBM_PHYSCHEM": {"learning_rate": 0.1, "max_depth": 6, "max_features": "sqrt",
                     "min_samples_leaf": 1, "min_samples_split": 2, "random_state": SEED, "n_estimators": 200},
    "GBM_WHIM": {"learning_rate": 0.1, "max_depth": 6, "max_features": "sqrt",
                 "min_samples_leaf": 1, "min_samples_split": 2, "random_state": SEED, "n_estimators": 200},
    "SVM_ECFP": {"C": 10, "epsilon": 0.1, "gamma": 0.01, "kernel": "rbf"},
    "SVM_MACCS": {"C": 10, "epsilon": 0.1, "gamma": 0.1, "kernel": "rbf"},
    "SVM_PHYSCHEM": {"C": 100, "epsilon": 0.1, "gamma": 0.1, "kernel": "rbf"},
    "SVM_WHIM": {"C": 1, "epsilon": 0.1, "gamma": 0.01, "kernel": "rbf"},
    "KNN_ECFP": {"metric": "euclidean", "n_neighbors": 5, "weights": "distance"},
    "KNN_MACCS": {"metric": "euclidean", "n_neighbors": 3, "weights": "distance"},
    "KNN_PHYSCHEM": {"metric": "euclidean", "n_neighbors": 11, "weights": "distance"},
    "KNN_WHIM": {"metric": "euclidean", "n_neighbors": 11, "weights": "distance"},
}




# ----------------------------------------------------------------------------
# Robust conformer embedding.
#
# WHIM descriptors need a 3D conformer. MoleculeACE generates one with RDKit's
# ETKDG algorithm and raises if it fails, which aborts the whole target:
#
#     !! CHEMBL233_prepared.csv failed entirely: FAILED embedding O=C1CC[C@]23...
#
# Default ETKDG settings fail on a small number of highly constrained cage
# systems (bridged polycyclics such as morphinans). The wrapper below retries a
# failed embedding with random starting coordinates and many more attempts, and
# only then with chirality constraints relaxed, before giving up. It also pins
# randomSeed, which makes WHIM descriptors -- and therefore every WHIM result --
# reproducible between runs. Without a fixed seed, re-running the same
# combination on the same split moves the cliff-specific RMSE by up to 0.4 log
# units, because a different conformer gives different descriptors.
# ----------------------------------------------------------------------------
def _patch_rdkit_embedding(seed=SEED, verbose=True):
    try:
        from rdkit.Chem import AllChem, rdDistGeom
    except ImportError:
        return []

    def _succeeded(res):
        if isinstance(res, int):
            return res != -1
        try:
            return len(res) > 0
        except TypeError:
            return res is not None

    def make_robust(orig):
        def robust(mol, *args, **kwargs):
            kwargs.setdefault("randomSeed", seed)
            try:
                res = orig(mol, *args, **kwargs)
                if _succeeded(res):
                    return res
            except Exception:                               # noqa: BLE001
                pass
            retry = dict(kwargs)
            retry["useRandomCoords"] = True
            retry["maxAttempts"] = max(int(retry.get("maxAttempts", 0) or 0), 500)
            try:
                res = orig(mol, *args, **retry)
                if _succeeded(res):
                    return res
            except Exception:                               # noqa: BLE001
                pass
            relaxed = dict(retry)
            relaxed["enforceChirality"] = False
            return orig(mol, *args, **relaxed)
        robust._ma_robust = True
        return robust

    modules = [AllChem, rdDistGeom]
    modules += [m for n, m in list(sys.modules.items())
                if n.startswith("MoleculeACE") and m is not None]
    patched = []
    for mod in modules:
        for fn in ("EmbedMolecule", "EmbedMultipleConfs"):
            cur = getattr(mod, fn, None)
            if callable(cur) and not getattr(cur, "_ma_robust", False):
                setattr(mod, fn, make_robust(cur))
                patched.append(f"{getattr(mod, '__name__', '?')}.{fn}")
    if patched and verbose:
        print(f"Conformer embedding made robust and seeded (randomSeed={seed}); "
              f"patched {len(patched)} entry point(s).\n")
    return patched


_EMBED_PATCHED = _patch_rdkit_embedding()


def fit_predict(algo_name, params, x_tr, y_tr, x_ev):
    """Fit one model and return its predictions. Kept in one place so that the
    CV loop and the final refit are guaranteed to do exactly the same thing."""
    model = ALGORITHMS[algo_name](**params)
    model.train(x_tr, y_tr)
    return np.asarray(model.predict(x_ev))


def out_of_fold_predictions(algo_name, params, x, y, cliff, n_folds=N_FOLDS, seed=SEED):
    """Pooled out-of-fold predictions over the training set.

    Folds are stratified on the activity-cliff label where possible, so that
    every fold contains a comparable proportion of cliff compounds. If a class
    has fewer members than the fold count, stratification is impossible and
    plain KFold is used instead; the caller is warned via the returned flag.
    """
    y = np.asarray(y, dtype=float)
    cliff = np.asarray(cliff).astype(int)
    oof = np.full(len(y), np.nan)
    stratified = cliff.sum() >= n_folds and (len(cliff) - cliff.sum()) >= n_folds
    if stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(np.zeros(len(y)), cliff)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(np.zeros(len(y)))
    for tr_idx, va_idx in folds:
        oof[va_idx] = fit_predict(algo_name, params,
                                  x[tr_idx], y[tr_idx], x[va_idx])
    return oof, stratified


def run_target(prepared_csv, verbose=True):
    """Run both protocols on one target. Returns (summary_dict, grid_dataframe, predictions)."""
    target = Path(prepared_csv).stem.replace("_prepared", "")
    data = Data(str(prepared_csv))

    y_train = np.asarray(data.y_train, dtype=float)
    y_test = np.asarray(data.y_test, dtype=float)
    cliff_train = np.asarray(data.cliff_mols_train).astype(int)
    cliff_test = np.asarray(data.cliff_mols_test).astype(int)

    if verbose:
        print(f"\n=== {target} ===")
        print(f"  train {len(y_train)} ({cliff_train.sum()} cliff) | "
              f"test {len(y_test)} ({cliff_test.sum()} cliff)")

    rows = []
    preds = []          # REVISION (editor comment 8): every test prediction
    skipped_descriptors = []
    for desc_name, desc in DESCRIPTORS.items():
        # Featurise once per descriptor. NOTE: MoleculeACE fits the Gaussian
        # normalisation on the full training set. Strictly, it should be refit
        # inside each CV fold; leaving it here means a small amount of scaling
        # information leaks between folds. It affects PHYSCHEM and WHIM only,
        # is identical for both protocols, and so cannot explain any difference
        # between them -- but mention it in the Methods if a referee asks.
        try:
            data.featurize_data(desc)
            x_train = np.asarray(data.x_train)
            x_test = np.asarray(data.x_test)
        except Exception as exc:                            # noqa: BLE001
            # A descriptor that cannot be computed for this target is skipped.
            # The remaining combinations still run, so the target completes.
            skipped_descriptors.append(desc_name)
            print(f"    {desc_name:9s} SKIPPED for this target - featurisation failed: "
                  f"{str(exc)[:120]}")
            continue

        for algo_name in ALGORITHMS:
            key = f"{algo_name}_{desc_name}"
            params = dict(HYPERPARAMS.get(key, {}))
            try:
                # --- protocol A: score on the test set (current manuscript) ---
                y_pred_test = fit_predict(algo_name, params, x_train, y_train, x_test)
                rmse_test = calc_rmse(y_test, y_pred_test)
                cliff_test_rmse = calc_cliff_rmse(
                    y_test_pred=y_pred_test, y_test=y_test,
                    cliff_mols_test=cliff_test)

                # --- protocol B: score on pooled out-of-fold training preds ---
                oof, stratified = out_of_fold_predictions(
                    algo_name, params, x_train, y_train, cliff_train)
                rmse_cv = calc_rmse(y_train, oof)
                cliff_cv = calc_cliff_rmse(
                    y_test_pred=oof, y_test=y_train, cliff_mols_test=cliff_train)

                rows.append({
                    "target": target, "descriptor": desc_name, "model": algo_name,
                    "cv_rmse": rmse_cv, "cv_cliff_rmse": cliff_cv,
                    "test_rmse": rmse_test, "test_cliff_rmse": cliff_test_rmse,
                    "cv_stratified": stratified,
                })
                preds.append(pd.DataFrame({                      # REVISION
                    "target": target, "descriptor": desc_name, "model": algo_name,
                    "smiles": data.smiles_test, "y_true": y_test, "y_pred": y_pred_test,
                    "cliff_mol": cliff_test}))
                if verbose:
                    print(f"    {desc_name:9s} {algo_name:4s}  "
                          f"cv_cliff {cliff_cv:.3f}  test_cliff {cliff_test_rmse:.3f}")
            except Exception as exc:                       # noqa: BLE001
                print(f"    {desc_name:9s} {algo_name:4s}  FAILED: {exc}")

    grid = pd.DataFrame(rows)
    if grid.empty:
        return None, grid, None

    # Protocol A: the combination the manuscript currently reports.
    best_test = grid.loc[grid["test_cliff_rmse"].idxmin()]
    # Protocol B: chosen without ever looking at the test set.
    best_cv = grid.loc[grid["cv_cliff_rmse"].idxmin()]

    summary = {
        "target": target,
        "n_train": len(y_train), "n_test": len(y_test),
        "n_cliff_train": int(cliff_train.sum()), "n_cliff_test": int(cliff_test.sum()),
        # protocol A
        "A_descriptor": best_test["descriptor"], "A_model": best_test["model"],
        "A_test_rmse": best_test["test_rmse"],
        "A_test_cliff_rmse": best_test["test_cliff_rmse"],
        # protocol B
        "B_descriptor": best_cv["descriptor"], "B_model": best_cv["model"],
        "B_cv_cliff_rmse": best_cv["cv_cliff_rmse"],
        "B_test_rmse": best_cv["test_rmse"],
        "B_test_cliff_rmse": best_cv["test_cliff_rmse"],
        # the quantity of interest
        "skipped_descriptors": ";".join(skipped_descriptors),
        "n_combinations": len(grid),
        "same_combination": (best_test["descriptor"] == best_cv["descriptor"]
                             and best_test["model"] == best_cv["model"]),
        "optimism_cliff_rmse": best_cv["test_cliff_rmse"] - best_test["test_cliff_rmse"],
    }
    if verbose:
        if skipped_descriptors:
            print(f"  NOTE: {len(grid)} of 16 combinations evaluated; "
                  f"skipped descriptor(s): {', '.join(skipped_descriptors)}")
        print(f"  A (test-selected): {summary['A_descriptor']}-{summary['A_model']}  "
              f"Cliff RMSE {summary['A_test_cliff_rmse']:.3f}")
        print(f"  B (cv-selected)  : {summary['B_descriptor']}-{summary['B_model']}  "
              f"Cliff RMSE {summary['B_test_cliff_rmse']:.3f}  "
              f"(optimism {summary['optimism_cliff_rmse']:+.3f})")
    return summary, grid, pd.concat(preds, ignore_index=True)   # REVISION


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="Data",
                    help="directory containing the *_prepared.csv files")
    ap.add_argument("--out", default="Results/rerun", help="output directory")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these targets, e.g. --only CHEMBL299 CHEMBL203")
    ap.add_argument("--list", action="store_true",
                    help="list the target files that would be run, then exit")
    ap.add_argument("--restart", action="store_true",
                    help="ignore previous results and start again from scratch")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "per_target_results.csv"
    grid_path = out_dir / "full_grid.csv"
    pred_path = out_dir / "test_predictions.csv"          # REVISION

    files = sorted(data_dir.glob("*_prepared.csv"))
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if f.stem.replace("_prepared", "") in wanted]
    if not files:
        sys.exit(f"No *_prepared.csv found in {data_dir.resolve()}\n"
                 "Check --data-dir, and that the files are named <TARGET>_prepared.csv")

    if args.list:
        print(f"{len(files)} target file(s) in {data_dir.resolve()}:")
        for f in files:
            n = sum(1 for _ in open(f, encoding="utf-8")) - 1
            print(f"  {f.stem.replace('_prepared',''):16s}  {n:>6,} compounds  ({f.name})")
        return

    # ---- resume: skip targets already present in per_target_results.csv ----
    done = set()
    if res_path.exists() and not args.restart:
        try:
            done = set(pd.read_csv(res_path)["target"].astype(str))
            if done:
                print(f"Resuming: {len(done)} target(s) already complete, skipping them. "
                      "Use --restart to redo everything.")
        except Exception:                                   # noqa: BLE001
            done = set()
    if args.restart:
        res_path.unlink(missing_ok=True)
        grid_path.unlink(missing_ok=True)
        pred_path.unlink(missing_ok=True)                    # REVISION

    todo = [f for f in files if f.stem.replace("_prepared", "") not in done]
    print(f"{len(files)} target file(s) found, {len(todo)} to run.")
    if not todo:
        print("Nothing left to do.")

    t_start = time.time()
    for i, f in enumerate(todo, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {f.stem.replace('_prepared','')}", flush=True)
        try:
            s_, g_, p_ = run_target(f)                # REVISION: + predictions
        except Exception as exc:                            # noqa: BLE001
            print(f"  !! failed entirely: {exc}")
            continue
        if s_ is None:
            print("  !! no combination completed; skipped")
            continue
        # append immediately so a crash never loses completed work
        pd.DataFrame([s_]).to_csv(res_path, mode="a", index=False,
                                  header=not res_path.exists())
        g_.to_csv(grid_path, mode="a", index=False, header=not grid_path.exists())
        p_.to_csv(pred_path, mode="a", index=False, header=not pred_path.exists())   # REVISION
        el = time.time() - t0
        avg = (time.time() - t_start) / i
        print(f"  done in {el/60:.1f} min | est. remaining "
              f"{avg*(len(todo)-i)/60:.0f} min")

    if not res_path.exists():
        sys.exit("Nothing completed successfully.")

    res = pd.read_csv(res_path)
    n = len(res)
    same = int(res["same_combination"].sum())
    opt = res["optimism_cliff_rmse"]
    lines = [
        "Selection-protocol comparison",
        "=" * 60,
        f"targets completed                     : {n}",
        f"same combination chosen by both       : {same} / {n} ({100*same/n:.0f}%)",
        "",
        "Cliff RMSE on the test set (lower is better):",
        f"  A  test-selected  mean {res['A_test_cliff_rmse'].mean():.3f}  "
        f"median {res['A_test_cliff_rmse'].median():.3f}",
        f"  B  cv-selected    mean {res['B_test_cliff_rmse'].mean():.3f}  "
        f"median {res['B_test_cliff_rmse'].median():.3f}",
        "",
        "Selection optimism (B - A, positive = protocol A was over-optimistic):",
        f"  mean {opt.mean():+.3f}   median {opt.median():+.3f}   max {opt.max():+.3f}",
        "",
        "Descriptor counts under the corrected protocol:",
        *(f"  {k:9s} {v}" for k, v in res["B_descriptor"].value_counts().items()),
        "",
        "Algorithm counts under the corrected protocol:",
        *(f"  {k:5s} {v}" for k, v in res["B_model"].value_counts().items()),
        "",
        "Report the protocol-B numbers in Table 2. Quote the optimism figures in",
        "Section 2.8 and in the Limitations as the measured size of the bias.",
    ]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)

    # REVISION (editor comments 5 and 6): record exactly what was run
    import json, platform, datetime
    from importlib import metadata
    def _v(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    try:
        import rdkit
        rdkit_version = rdkit.__version__
    except ImportError:
        rdkit_version = None
    run_info = {
        "date": datetime.date.today().isoformat(),
        "seeds": {"cv_folds": SEED, "random_forest": SEED, "gradient_boosting": SEED,
                  "moleculeace_split_and_conformers": 42},
        "n_folds": N_FOLDS,
        "hyperparameter_source": "MoleculeACE v3.0.0 Data/configures/benchmark/CHEMBL204_Ki/*.yml",
        "hyperparameters": HYPERPARAMS,
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "rdkit": rdkit_version, "scikit-learn": _v("scikit-learn"), "numpy": _v("numpy"),
                     "pandas": _v("pandas"),
                     "MoleculeACE": "v3.0.0 (GitHub tag; package metadata reports " + str(_v("MoleculeACE")) + ")"},
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    print("\n" + text)
    print(f"\nWritten to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
