"""Shared pieces for the descriptor-normalisation analysis.

Descriptors are built exactly as MoleculeACE builds them, before its Gaussian
normalisation is applied. Folds, seeds and hyperparameters are those of
06_model_selection_cv_vs_test.py, so the uncorrected arm of the analysis
reproduces the deposited cross-validated scores exactly.
"""
import os
import pickle

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors, rdmolops

RDLogger.DisableLog("rdApp.*")

SEED = 42
N_FOLDS = 5
SCALED_DESCRIPTORS = ("PHYSCHEM", "WHIM")
SCALE_SENSITIVE_MODELS = ("SVM", "KNN")

HYPERPARAMS = {
    "SVM_PHYSCHEM": dict(C=100, epsilon=0.1, gamma=0.1, kernel="rbf"),
    "KNN_PHYSCHEM": dict(metric="euclidean", n_neighbors=11, weights="distance"),
    "RF_PHYSCHEM": dict(n_estimators=500, random_state=SEED),
    "GBM_PHYSCHEM": dict(learning_rate=0.1, max_depth=6, max_features="sqrt",
                         max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                         n_estimators=200, random_state=SEED),
    "SVM_WHIM": dict(C=1, epsilon=0.1, gamma=0.01, kernel="rbf"),
    "KNN_WHIM": dict(metric="euclidean", n_neighbors=11, weights="distance"),
    "RF_WHIM": dict(n_estimators=500, random_state=SEED),
    "GBM_WHIM": dict(learning_rate=0.1, max_depth=6, max_features="sqrt",
                     max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                     n_estimators=200, random_state=SEED),
}


def physchem(smiles):
    """The eleven MoleculeACE physicochemical descriptors, before normalisation."""
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(np.array([
            Descriptors.ExactMolWt(m), Descriptors.MolLogP(m),
            Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
            Descriptors.NumRotatableBonds(m), m.GetNumAtoms(), m.GetNumHeavyAtoms(),
            Crippen.MolMR(m), QED.properties(m).PSA,
            rdmolops.GetFormalCharge(m), rdMolDescriptors.CalcNumRings(m)]))
    return np.array(out)


def whim(smiles, cache_path):
    """The 114 WHIM descriptors, before normalisation, with the embedding
    settings and seed MoleculeACE uses. Conformers are cached so that a run can
    be resumed and so that repeated calls are consistent."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
    out, new = [], 0
    for s in smiles:
        if s not in cache:
            mh = Chem.AddHs(Chem.MolFromSmiles(s))
            e = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=True,
                                      randomSeed=SEED, maxAttempts=5)
            if e == -1:
                e = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=True,
                                          randomSeed=SEED, clearConfs=False,
                                          enforceChirality=False, maxAttempts=45)
            if e == -1:
                e = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=False,
                                          randomSeed=SEED, clearConfs=True,
                                          enforceChirality=False, maxAttempts=1000)
            if e == -1:
                raise RuntimeError(f"FAILED embedding {s}")
            AllChem.MMFFOptimizeMolecule(mh, maxIters=1000, mmffVariant="MMFF94")
            cache[s] = rdMolDescriptors.CalcWHIM(mh)
            new += 1
            if cache_path and new % 250 == 0:
                with open(cache_path, "wb") as fh:
                    pickle.dump(cache, fh)
        out.append(cache[s])
    if cache_path and new:
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh)
    return np.array(out)


def folds_for(cliff):
    """The fold assignment used by the model-selection script."""
    cliff = np.asarray(cliff).astype(int)
    stratified = cliff.sum() >= N_FOLDS and (len(cliff) - cliff.sum()) >= N_FOLDS
    if stratified:
        sp = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return list(sp.split(np.zeros(len(cliff)), cliff)), stratified
    sp = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(sp.split(np.zeros(len(cliff)))), stratified


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
