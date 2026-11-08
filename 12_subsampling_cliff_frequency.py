"""Table S12 and Section 2.4: effect of random subsampling on activity-cliff
frequency.

Activity-cliff frequency counts compounds that have at least one qualifying
partner inside the same dataset, so it depends on how densely a dataset samples
its own chemical space. Thinning a dataset at random therefore lowers the
measured frequency even though the chemistry is unchanged. This script measures
that dependence on the six largest targets of the panel together with Bcl-2, the
most cliff-rich dataset in the panel.

Each target is subsampled at random to each of the sizes given, ten independent
draws at each size, and the cliff frequency is recomputed on the draw under the
criterion of Section 4.5, using MoleculeACE's own similarity and fold-change
routines.

Usage:
    python 12_subsampling_cliff_frequency.py DATA_DIR NAMES_JSON OUT_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtac_cliff_api import (CLIFF_POTENCY_FOLD, CLIFF_SIMILARITY, get_fc,
                            moleculeace_similarity)

SIZES = (100, 150, 200, 300, 500)
N_DRAWS = 10
SEED = 42
N_LARGEST = 6
ALWAYS_INCLUDE = "Bcl-2"


def cliff_fraction(smiles, ki_nM):
    """Percentage of compounds that are activity-cliff compounds within this set."""
    sim = moleculeace_similarity(list(smiles), similarity=CLIFF_SIMILARITY, hide=True)
    fc = get_fc(np.asarray(ki_nM, dtype=float))
    pair = (sim == 1) & (fc > CLIFF_POTENCY_FOLD)
    return 100.0 * float(pair.any(axis=1).mean())


def main():
    data_dir, names_json, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    names = json.load(open(names_json))

    sizes_by_target = {}
    for t in names:
        sizes_by_target[t] = len(pd.read_csv(os.path.join(data_dir, f"{t}_prepared.csv")))
    chosen = sorted(sizes_by_target, key=lambda t: -sizes_by_target[t])[:N_LARGEST]
    if ALWAYS_INCLUDE not in chosen:
        chosen.append(ALWAYS_INCLUDE)

    rows = []
    for t in sorted(chosen, key=lambda t: -sizes_by_target[t]):
        df = pd.read_csv(os.path.join(data_dir, f"{t}_prepared.csv"))
        full_n = len(df)
        full_pct = 100.0 * float(df.cliff_mol.mean())
        rng = np.random.default_rng(SEED)
        for n in SIZES:
            if n >= full_n:
                continue
            vals = []
            for _ in range(N_DRAWS):
                idx = rng.choice(full_n, size=n, replace=False)
                sub = df.iloc[idx]
                vals.append(cliff_fraction(sub.smiles.tolist(),
                                           sub["exp_mean [nM]"].to_numpy(dtype=float)))
            vals = np.array(vals)
            rows.append(dict(
                target=t, protein=names[t], full_n=full_n, full_pct_cliff=full_pct,
                subsample_n=n, mean_pct_cliff=float(vals.mean()),
                sd=float(vals.std(ddof=1)),
                fraction_retained=100.0 * float(vals.mean()) / full_pct))
            print(f"  {names[t]:24s} n={n:4d}  %Cliff {vals.mean():6.2f} "
                  f"(SD {vals.std(ddof=1):4.2f})  {100*vals.mean()/full_pct:5.1f}% retained",
                  flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "TableS12_subsampling.csv"), index=False)

    at150 = tab[tab.subsample_n == 150]
    lines = [
        f"targets subsampled                    {tab.target.nunique()}",
        f"draws per size                        {N_DRAWS}",
        f"mean fraction retained at n = 150     {at150.fraction_retained.mean():.1f}%",
        f"range at n = 150                      "
        f"{at150.fraction_retained.min():.1f}% to {at150.fraction_retained.max():.1f}%",
    ]
    for _, r in at150.iterrows():
        lines.append(f"  {r.protein:24s} {r.full_pct_cliff:6.2f}% -> {r.mean_pct_cliff:6.2f}% "
                     f"({r.fraction_retained:.2f}% retained)")
    text = "\n".join(lines)
    open(os.path.join(out_dir, "TableS12_panel_statistics.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
