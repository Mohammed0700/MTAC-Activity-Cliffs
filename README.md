# MTAC-Activity-Cliffs

Data, code and results for *Target-Specific Benchmarking of Potency Prediction on
Activity-Cliff Compounds Across 33 Human Protein Targets Using Inhibition
Constants*.

Every number, table and figure in the manuscript and in the Supplementary
Materials can be traced to a file in this repository. This README says which
file produces which result, which code version produced the results that are
reported, and how to reproduce them.

---

## 1. Layout

```
01_conform_pure_data_extract.ipynb    confirmation that the raw pull is Ki only
02_CHEMBL_Data_extract.ipynb          ChEMBL release 37 extraction and curation
03_BindingDB_analysis.ipynb           BindingDB extraction for the three non-ChEMBL targets
04_MoleculeACE_analysis.ipynb         activity-cliff annotation and partitioning
05_Best_ML.ipynb                      exploratory model fitting
06_model_selection_cv_vs_test.py      the benchmark: both selection protocols
07_Statistics_tables_figures.ipynb    manuscript tables and figures
08_sd_threshold_check.py              the replicate standard-deviation filter
09_compare_runs.py                    difference between the two result runs
10_cliff_pair_error.py                Table S10
11_applicability_domain.py            Table S11
12_subsampling_cliff_frequency.py     Table S12
13_scaler_refit_effect.py             Table S13
14_scaling_invariance_check.py        supporting check for Table S13
mtac_cliff_api.py                     access to the MoleculeACE cliff routines
scaler_effect_lib.py                  descriptors, folds and hyperparameters for 13 and 14

<target>_raw.csv                     the extraction, before any curation
<target>_curated.csv                 after standardisation and replicate aggregation
<target>_filtered.csv                after the quality filters of Section 4.3
<target>_final.csv                   the curated endpoint of the per-target pipeline
Process_Steps/                        per-target intermediates, and a copy of _final
Data/<target>_prepared.csv            the modelling input: SMILES, Ki in nM, pKi,
                                      MoleculeACE cliff label, train/test split
Results/rerun/                        first complete run, 2 October 2026
Results/rerun_r2/                     second complete run, 8 October 2026 (reported)
Results/analyses/                     outputs of scripts 10 to 14
Results/sd_threshold_check/           output of 08
environment.yml, requirements.txt     the environment
```

The four per-target CSVs at the repository root are the curation trail, one set
for each of the 33 targets, written in the order `_raw`, `_curated`, `_filtered`,
`_final` by notebooks 02 and 03. Thrombin is typical: 4,232 records are
extracted, 3,144 survive standardisation and replicate aggregation, and 2,929
enter the modelling set once activity-cliff annotation has removed pairs that
share a fingerprint but differ in canonical SMILES. `Process_Steps/` holds the
`_cleaned` intermediates and a second copy of each `_final` file, identical to
the one at the root.

Each `Data/<target>_prepared.csv` has one row per curated compound and the
columns `smiles`, `exp_mean [nM]`, `y`, `cliff_mol`, `split`. These 33 files are
the input to every result in the manuscript. `y` is `-log10(Ki in nM)`, which
differs from the pKi reported in the manuscript by the constant 9 and so gives
identical errors and identical rankings.

---

## 2. The two result runs

Both complete runs are kept, as stated in our earlier response to the editor.

| | `Results/rerun` | `Results/rerun_r2` |
|---|---|---|
| Date | 2 October 2026 | 8 October 2026 |
| Cliff labels used for cross-validation | MoleculeACE labels from the full dataset | recomputed from the outer training set alone |
| Reported in the manuscript | no | **yes** |

The two differ only in the activity-cliff labels used to stratify the
cross-validation folds and to score the selection criterion. In the first run
those labels came from the full dataset, so a training-set label could depend on
a test-set compound. The second run recomputes them from the outer training set
alone. The change is described in Section 4.9 and its effect is reported in
Table S9. `09_compare_runs.py` prints the difference between the two runs.

Test-set labels are the ones assigned by MoleculeACE `split_data()` in both
runs, and are used for reporting only.

---

## 3. Code version of the reported results

The cross-validation in `06_model_selection_cv_vs_test.py` originally passed
MoleculeACE's normalised descriptor matrix straight into the fold loop.
MoleculeACE fits that normalisation, a `StandardScaler`, on the whole training
set at featurisation time, so for the two scaled descriptor families the
normalisation carried information from the validation fold into the training
fold. The current version of the script refits the scaler inside each fold.

**Every result reported in the manuscript was produced by the version of the
script before that change.** That version is preserved in this repository at

```
commit a8c12cf   Add MTAC_Activity_Cliffs
tag    results-as-published
```

and can be recovered with

```bash
git show results-as-published:06_model_selection_cv_vs_test.py
```

The correction is commit `c7cbfd6`, which is the version at `HEAD`. It affects
only the cross-validated selection score. Test-set predictions are unaffected in
either version, because the test-set transform is the one fitted on the training
set, which is the correct treatment for held-out data.

`13_scaler_refit_effect.py` measures what the correction changes.
Substituting the corrected scores changes the combination selected for one of the 33 targets, S6K1, where WHIM with nearest neighbours replaces WHIM with a support vector machine. With that substitution the cliff-specific test error of S6K1 would move from 0.803 to 0.741 pKi units, the panel mean increase in cliff-specific error would move from 12.8% to 12.6%, the mean selection optimism from 0.119 to 0.117 pKi units, and cliff error would still exceed global error for 26 of the 33 targets.
The measured effect is reported in Table S13 and discussed in the Limitations.

---

## 4. Reproducing

```bash
conda env create -f environment.yml
conda activate mtac
python 06_model_selection_cv_vs_test.py          # writes Results/rerun_r2
```

The run is deterministic. Random seeds are fixed at 42 for the fold assignment,
the random forest, the gradient boosting machine, and MoleculeACE's splitting and
conformer generation, and are recorded in `Results/rerun_r2/run_info.json`
together with the complete hyperparameter set.

The analysis scripts read the deposited results and do not require the benchmark
to be rerun:

```bash
python 10_cliff_pair_error.py            Data Results/rerun_r2 target_names.json Results/analyses
python 11_applicability_domain.py        Data Results/rerun_r2 target_names.json Results/analyses
python 12_subsampling_cliff_frequency.py Data target_names.json Results/analyses
python 13_scaler_refit_effect.py         Data Results/rerun_r2 target_names.json Results/analyses 0.25
python 14_scaling_invariance_check.py    Data Results/rerun_r2/train_only_cliffs \
                                         Results/analyses/scaling_invariance_check.csv
```

Script 13 needs a conformer for every molecule of each target it evaluates with
WHIM descriptors. It caches them in `Results/analyses/WHIM_conformers.pkl`, so a
run can be interrupted and resumed.

---

## 5. Which file produces which result

| Result | Produced by | Output |
|---|---|---|
| Table 1, Figure 3 | `07_Statistics_tables_figures.ipynb` | `Results/rerun_r2/tables`, `figures` |
| Table 2, Figures 4 to 6 | `07_Statistics_tables_figures.ipynb` | `Results/rerun_r2/tables`, `figures` |
| Tables S1, S2, S4 to S8 | `07_Statistics_tables_figures.ipynb` | `Results/rerun_r2/tables` |
| Table S3 | `04_MoleculeACE_analysis.ipynb` | `Results/rerun_r2/tables` |
| Table S9 | `09_compare_runs.py` | `Results/rerun_r2/train_only_cliffs` |
| Table S10 | `10_cliff_pair_error.py` | `Results/analyses/TableS10_*` |
| Table S11 | `11_applicability_domain.py` | `Results/analyses/TableS11_*` |
| Table S12 | `12_subsampling_cliff_frequency.py` | `Results/analyses/TableS12_*` |
| Table S13 | `13_scaler_refit_effect.py` | `Results/analyses/TableS13_*` |
| Replicate SD filter | `08_sd_threshold_check.py` | `Results/sd_threshold_check` |

Each analysis script writes its table as a CSV and a short plain-text file of the
panel-level statistics quoted in the corresponding section of the manuscript, so
that each sentence can be checked against the file that produced it.

---

## 6. Notes on the analysis scripts

`10_cliff_pair_error.py` and `12_subsampling_cliff_frequency.py` call
MoleculeACE's own similarity and fold-change routines through `mtac_cliff_api.py`,
so the activity-cliff criterion is identical to the one used to label the
datasets: a pair is similar when any of the three measures reaches 0.90, and is a
cliff when the fold change in Ki exceeds 10. The fold-change test is strict, as
in `ActivityCliffs.find_cliffs`, so a pair at exactly ten-fold is not a cliff.

`13_scaler_refit_effect.py` recomputes the cross-validated score of all eight
affected combinations, the two normalised descriptor families crossed with the
four learners. Its uncorrected arm reproduces `Results/rerun_r2/full_grid.csv` to
machine precision, which is printed at the end of the run and is the check that
the measurement is anchored to the deposited results.

`14_scaling_invariance_check.py` records how each learner responds to the change.
Support vector machines and nearest neighbours work on distances and respond
directly. Tree ensembles are invariant to a per-column affine map in exact
arithmetic, so they might be expected not to respond, but rescaling perturbs the
computed impurity sums at the level of floating-point rounding, a split close to
a tie can then be decided the other way, and once one split differs the trees
diverge. In practice all four learners move. For the random forest the movement
shrinks as more trees are averaged but does not disappear, so all four are
measured rather than argued about.
