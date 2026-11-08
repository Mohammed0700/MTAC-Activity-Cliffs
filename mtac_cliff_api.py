"""Access to the MoleculeACE activity-cliff routines.

In a complete MoleculeACE installation the ordinary import works. The analysis
scripts here need only the similarity and fold-change helpers, which depend on
RDKit and Levenshtein alone, so when the full package cannot be imported (for
example in a minimal environment without the deep-learning extras) the module
file is loaded directly. Either way the functions are MoleculeACE's own and the
criterion is unchanged.
"""
import importlib.util
import os


def _load():
    try:
        from MoleculeACE.benchmark.cliffs import (            # noqa: F401
            get_fc, get_levenshtein_matrix, get_scaffold_matrix,
            get_tanimoto_matrix, moleculeace_similarity)
        import MoleculeACE.benchmark.cliffs as m
        return m
    except Exception:                                          # noqa: BLE001
        path = os.environ.get("MOLECULEACE_CLIFFS")
        if not path:
            try:
                import MoleculeACE
                path = os.path.join(os.path.dirname(MoleculeACE.__file__),
                                    "benchmark", "cliffs.py")
            except Exception:                                  # noqa: BLE001
                raise ImportError(
                    "MoleculeACE is not importable and MOLECULEACE_CLIFFS is not set. "
                    "Set MOLECULEACE_CLIFFS to the path of MoleculeACE/benchmark/cliffs.py.")
        spec = importlib.util.spec_from_file_location("mtac_mace_cliffs", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_M = _load()

get_fc = _M.get_fc
moleculeace_similarity = _M.moleculeace_similarity
get_tanimoto_matrix = _M.get_tanimoto_matrix
get_scaffold_matrix = _M.get_scaffold_matrix
get_levenshtein_matrix = _M.get_levenshtein_matrix

CLIFF_SIMILARITY = 0.9
CLIFF_POTENCY_FOLD = 10
