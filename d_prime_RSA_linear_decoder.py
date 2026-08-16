"""
Pairwise linear-decoder RSA analysis.

For each stimulus type (pureTones, AM, naturalSounds), and for each session /
brain region / response range, we train a pairwise linear SVC decoder for every
pair of stimuli. We take the SVC separating hyperplane normal `w` as a 1D
projection axis, project both stimulus distributions onto it, and compute a
separability metric (Fisher's criterion J by default, or the classic d-prime)
on the two resulting 1D projections. This yields a symmetric RSA matrix whose
entries are the pairwise separability values.

We do this for:
  - raw firing rates per brain region, and
  - CCA-projected firing rates (primary-side canonical variates) from Primary
    auditory area into each secondary auditory area (Ventral, Dorsal).

Kept as a separate file from CCA_RSA.py for research-archiving purposes.
"""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for multiprocessing

import os

# Limit BLAS/OpenMP threads per process so the 9 parallel workers do not
# oversubscribe cores (process-level parallelism is what we want here).
# These must be set before numpy/sklearn import their backends.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import CCA
import pandas as pd

from analysis_class import FiringRateAnalysis

response_ranges = ["onset", "sustained", "offset"]
neuron_threshold = 20
n_splits = 5  # Number of random neuron subsamples to average over

# SVC hyperparameter search settings
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
CV_FOLDS = 5

# Metric to store in the RSA matrices. We compute both regardless; this only
# controls the default "primary" metric key, but both are saved.
METRICS = ("J", "dprime")


# ----------------------------------------------------------------------------
# Hyperparameter cache (JSON) helpers
# ----------------------------------------------------------------------------
def _load_hyperparam_cache(cache_path: str) -> dict:
    """Load the JSON hyperparameter cache, returning {} if it does not exist."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def _save_hyperparam_cache(cache_path: str, cache: dict) -> None:
    """Persist the JSON hyperparameter cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _hyperparam_key(session, brain_region, target_region, stimulus,
                    response_range, stim_a, stim_b) -> str:
    """
    Build a fine-grained cache key. `target_region` is "raw" when decoding on
    raw firing rates (no CCA target). Stimuli are sorted so (a, b) and (b, a)
    map to the same key.
    """
    stim_lo, stim_hi = sorted([str(stim_a), str(stim_b)])
    target = "raw" if target_region is None else str(target_region)
    return "|".join([
        str(session), str(brain_region), target, str(stimulus),
        str(response_range), stim_lo, stim_hi,
    ])


# ----------------------------------------------------------------------------
# Separability metric on 1D projections
# ----------------------------------------------------------------------------
def dprime_from_projections(proj1: np.ndarray, proj2: np.ndarray,
                            metric: str = "J") -> float:
    """
    Compute a separability metric between two 1D projections.

    proj1, proj2: 1D arrays of projected trial values for stimulus 1 and 2.

    metric="J":       Fisher's criterion J = (m1 - m2)^2 / (s1^2 + s2^2),
                      matching calc_fisher_criterion_Christian on 1D data.
    metric="dprime":  Classic d-prime = (m1 - m2) / sqrt((s1^2 + s2^2) / 2).
                      (Related to J by d' = sqrt(2 * J) up to sign.)
    """
    m1, m2 = np.mean(proj1), np.mean(proj2)
    s1, s2 = np.var(proj1), np.var(proj2)
    denom = s1 + s2
    if denom == 0:
        return 0.0
    if metric == "J":
        return float((m1 - m2) ** 2 / denom)
    elif metric == "dprime":
        return float((m1 - m2) / np.sqrt(denom / 2.0))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ----------------------------------------------------------------------------
# Pairwise decoder
# ----------------------------------------------------------------------------
def train_pairwise_decoder(features_a: np.ndarray, features_b: np.ndarray,
                           cached_C=None):
    """
    Train a linear SVC to separate two stimulus distributions and return the
    projection axis `w` plus the projected distributions.

    features_a, features_b: arrays of shape (nTrials, nFeatures). Features are
        neurons (raw) or CCA components. Note: callers pass ALREADY-TRANSPOSED
        arrays here (trials as rows), since incoming firing-rate arrays start as
        (nCells, nTrials).

    cached_C: if not None, skip GridSearchCV and fit directly with this C.

    Returns: (proj_a, proj_b, best_C)
        proj_a, proj_b: 1D projections of each class onto w.
        best_C: the C used (either the searched-optimal or the cached value).
    """
    # X shape (nTrials_total, nFeatures); y is the class label per trial.
    X = np.vstack([features_a, features_b])
    y = np.concatenate([
        np.zeros(features_a.shape[0]),
        np.ones(features_b.shape[0]),
    ])

    if cached_C is None:
        # Cross-validate ONLY to select the optimal SVC hyperparameter.
        search = GridSearchCV(
            SVC(kernel="linear"),
            param_grid={"C": C_GRID},
            cv=CV_FOLDS,
        )
        search.fit(X, y)
        best_C = float(search.best_params_["C"])
    else:
        best_C = float(cached_C)

    # Refit the optimal model on ALL pair trials; d-prime is computed only from
    # this optimal model (not from CV folds).
    model = SVC(kernel="linear", C=best_C)
    model.fit(X, y)

    # w is the separating hyperplane normal: shape (1, nFeatures) -> (nFeatures,)
    w = model.coef_.ravel()
    w_norm = np.linalg.norm(w)
    if w_norm > 0:
        w = w / w_norm  # Unit vector, matching the Fisher-criterion convention.

    # Project each class (trials as rows) onto w -> 1D arrays of length nTrials.
    proj_a = features_a @ w
    proj_b = features_b @ w
    return proj_a, proj_b, best_C


def compute_rsa_matrix(get_features, uniq_stims, stim_row, session,
                       brain_region, target_region, stimulus, response_range,
                       hp_cache, solve_hyperparam):
    """
    Build symmetric RSA matrices (one per metric) for all stimulus pairs.

    get_features(stim_mask) -> array of shape (nTrials_for_stim, nFeatures)
        Callback returning the (trials-as-rows) feature matrix for a stimulus.
        This abstracts over raw vs CCA features.
    stim_row: per-trial stimulus id array, shape (nTrials,).
    """
    n = len(uniq_stims)
    matrices = {m: np.zeros((n, n)) for m in METRICS}

    for i, stim_a in enumerate(uniq_stims):
        for j in range(i + 1, n):
            stim_b = uniq_stims[j]

            feats_a = get_features(stim_row == stim_a)  # (nTrials_a, nFeatures)
            feats_b = get_features(stim_row == stim_b)  # (nTrials_b, nFeatures)

            key = _hyperparam_key(session, brain_region, target_region,
                                  stimulus, response_range, stim_a, stim_b)
            cached_C = None if solve_hyperparam else hp_cache.get(key)

            proj_a, proj_b, best_C = train_pairwise_decoder(
                feats_a, feats_b, cached_C=cached_C)

            # Update cache with the C actually used (solved or reused).
            hp_cache[key] = best_C

            for metric in METRICS:
                val = dprime_from_projections(proj_a, proj_b, metric=metric)
                matrices[metric][i, j] = val
                matrices[metric][j, i] = val  # Symmetric matrix.

    return matrices


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Per-(stimulus, response_range) worker
# ----------------------------------------------------------------------------
def process_stimulus_response(stimulus: str, response_range: str,
                              solve_hyperparam: bool = True) -> str:
    """
    Compute and save the RSA data for a single (stimulus, response_range) cell.

    Runs in its own process. Loads its own FiringRateAnalysis and significant_df
    so workers are fully independent, and writes its own per-cell hyperparameter
    cache (keys already include stimulus/response_range, so files are disjoint
    and safe to merge later).

    Returns the path of the saved pickle (for logging by the driver).
    """
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path

    out_dir = f"{file_path}/dprime_RSA"
    os.makedirs(out_dir, exist_ok=True)

    # Per-worker cache file, disjoint across (stimulus, response_range).
    cache_path = f"{out_dir}/hyperparams_{stimulus}_{response_range}.json"
    hp_cache = _load_hyperparam_cache(cache_path)

    significant_df = pd.read_csv(
        f"{file_path}/CCA_two_region_analysis/cca_primary_auditory_results.csv")

    print(f"[{stimulus}/{response_range}] Computing d-prime RSA")
    stim_arrays = fr_db.return_arrays(stimulus)
    brainRegionArray = stim_arrays["brainRegionArray"]  # per-neuron (rows)
    sessionArray = stim_arrays["sessionIDArray"]        # per-neuron (rows)
    uniqSessions = np.unique(sessionArray)

    # Rename posterior to dorsal instead of treating them as separate areas.
    brainRegionArray[brainRegionArray == "Posterior auditory area"] = \
        "Dorsal auditory area"
    uniq_regions = np.unique(brainRegionArray)
    # Dropping Temporal auditory area.
    uniq_regions = uniq_regions[uniq_regions != "Temporal association areas"]

    stim_row = stim_arrays["stimArray"][0, :]  # per-trial stimulus id
    uniqStims = np.unique(stim_row)

    rsa_data = []
    # respArray: rows are neurons, columns are trials -> (nCells, nTrials)
    respArray = stim_arrays[f"{response_range}fr"]

    for session in uniqSessions:
        session_mask = sessionArray == session
        session_resp_array = respArray[session_mask, :]  # (nCells, nTrials)
        brain_session_array = brainRegionArray[session_mask]

        for brain_region in uniq_regions:
            # primary_array: (nCells_in_region, nTrials)
            primary_array = session_resp_array[
                brain_session_array == brain_region, :]

            if primary_array.shape[0] < neuron_threshold:
                print(f"[{stimulus}/{response_range}] Skipping session "
                      f"{session}: insufficient neurons in {brain_region}")
                continue

            # ----------------------------------------------------------
            # (1) RAW firing rates for this single brain region.
            # ----------------------------------------------------------
            raw_matrix_lists = {m: [] for m in METRICS}
            for _ in range(n_splits):
                # Subsample neurons: (neuron_threshold, nTrials)
                idx = np.random.choice(primary_array.shape[0],
                                       neuron_threshold, replace=False)
                primary_subset = primary_array[idx, :]

                # Features must be (nTrials, nFeatures) -> transpose.
                raw_features = primary_subset.T  # (nTrials, nNeurons)

                def get_raw_features(mask, _f=raw_features):
                    return _f[mask, :]  # (nTrials_for_stim, nNeurons)

                mats = compute_rsa_matrix(
                    get_raw_features, uniqStims, stim_row, session,
                    brain_region, None, stimulus, response_range,
                    hp_cache, solve_hyperparam)
                for m in METRICS:
                    raw_matrix_lists[m].append(mats[m])

            rsa_data.append({
                "stimulus": stimulus,
                "response_range": response_range,
                "session": session,
                "data_source": "raw",
                "brain_region": brain_region,
                "target_region": None,
                "pair_comparison": brain_region,
                "J_matrix": np.mean(raw_matrix_lists["J"], axis=0),
                "dprime_matrix": np.mean(raw_matrix_lists["dprime"], axis=0),
            })

            # ----------------------------------------------------------
            # (2) CCA-projected firing rates: primary-side canonical
            #     variates from this (primary) region into each target.
            # ----------------------------------------------------------
            for target_region in uniq_regions:
                if brain_region == target_region:
                    continue
                # target_array: (nCells_in_target, nTrials)
                target_array = session_resp_array[
                    brain_session_array == target_region, :]
                if target_array.shape[0] < neuron_threshold:
                    print(f"[{stimulus}/{response_range}] Skipping session "
                          f"{session}: insufficient neurons in "
                          f"{target_region}")
                    continue

                mask_n_comps_target = (
                    (significant_df["region1"] == brain_region)
                    & (significant_df["region2"] == target_region)
                    & (significant_df["stimulus"] == stimulus)
                    & (significant_df["response_range"] == response_range)
                    & (significant_df["session"] == session)
                )
                try:
                    n_components_target = significant_df.loc[
                        mask_n_comps_target, "significant_components"].iloc[0]
                    n_components_target = np.int64(n_components_target)
                    if n_components_target <= 1:
                        print(f"[{stimulus}/{response_range}] Skipping "
                              f"session {session}: insufficient significant "
                              f"components for {brain_region} vs "
                              f"{target_region}")
                        continue
                except (IndexError, ValueError):
                    print(f"[{stimulus}/{response_range}] No significant "
                          f"components found for {brain_region} vs "
                          f"{target_region}, session {session}")
                    continue

                cca_matrix_lists = {m: [] for m in METRICS}
                for _ in range(n_splits):
                    # Subsample and zero-mean both regions (as in CCA_RSA).
                    p_idx = np.random.choice(primary_array.shape[0],
                                             neuron_threshold, replace=False)
                    primary_subset = primary_array[p_idx, :]  # (nCells, nTrials)
                    primary_zero_mean = primary_subset - np.mean(primary_subset)

                    t_idx = np.random.choice(target_array.shape[0],
                                             neuron_threshold, replace=False)
                    target_subset = target_array[t_idx, :]  # (nCells, nTrials)
                    target_zero_mean = target_subset - np.mean(target_subset)

                    # CCA expects (nSamples, nFeatures) = (nTrials, nCells),
                    # so transpose. primary_transform is
                    # (nTrials, n_components): the primary-side variates.
                    cca_mod = CCA(n_components=n_components_target)
                    primary_transform, _ = cca_mod.fit_transform(
                        primary_zero_mean.T, target_zero_mean.T)

                    def get_cca_features(mask, _f=primary_transform):
                        return _f[mask, :]  # (nTrials_for_stim, n_components)

                    mats = compute_rsa_matrix(
                        get_cca_features, uniqStims, stim_row, session,
                        brain_region, target_region, stimulus,
                        response_range, hp_cache, solve_hyperparam)
                    for m in METRICS:
                        cca_matrix_lists[m].append(mats[m])

                rsa_data.append({
                    "stimulus": stimulus,
                    "response_range": response_range,
                    "session": session,
                    "data_source": "cca",
                    "brain_region": brain_region,
                    "target_region": target_region,
                    "pair_comparison": f"{brain_region}_{target_region}",
                    "J_matrix": np.mean(cca_matrix_lists["J"], axis=0),
                    "dprime_matrix": np.mean(cca_matrix_lists["dprime"], axis=0),
                })

    # Persist this worker's hyperparameter cache.
    _save_hyperparam_cache(cache_path, hp_cache)

    out_pkl = f"{out_dir}/dprime_rsa_{stimulus}_{response_range}.pkl"
    pickle.dump(rsa_data, open(out_pkl, "wb"))
    print(f"[{stimulus}/{response_range}] d-prime RSA data saved to {out_pkl}")

    rsa_frame = pd.DataFrame(rsa_data)
    rsa_frame.to_csv(
        f"{out_dir}/dprime_rsa_{stimulus}_{response_range}.csv")

    return out_pkl


def _merge_hyperparam_caches(out_dir: str, stim_types, resp_ranges) -> None:
    """
    Merge the disjoint per-worker cache files into a single hyperparams.json for
    convenient reuse on reruns. Per-worker files are kept as-is.
    """
    merged = {}
    for stimulus in stim_types:
        for response_range in resp_ranges:
            per_worker = f"{out_dir}/hyperparams_{stimulus}_{response_range}.json"
            merged.update(_load_hyperparam_cache(per_worker))
    _save_hyperparam_cache(f"{out_dir}/hyperparams.json", merged)


# ----------------------------------------------------------------------------
# Parallel driver
# ----------------------------------------------------------------------------
def main(solve_hyperparam: bool = True, max_workers: int = None):
    """
    Run all (stimulus, response_range) combinations in parallel, one process per
    combination.
    """
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path
    stim_types = fr_db.stim_types
    out_dir = f"{file_path}/dprime_RSA"
    os.makedirs(out_dir, exist_ok=True)
    stored_Errors = []

    # One task per (stimulus, response_range) -> up to 9 parallel workers.
    tasks = [(stimulus, response_range)
             for stimulus in stim_types
             for response_range in response_ranges]

    if max_workers is None:
        max_workers = len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_stimulus_response, stimulus,
                            response_range, solve_hyperparam): (stimulus,
                                                                response_range)
            for stimulus, response_range in tasks
        }
        for future in as_completed(future_to_task):
            stimulus, response_range = future_to_task[future]
            try:
                out_pkl = future.result()
                print(f"Completed {stimulus}/{response_range} -> {out_pkl}")
            except Exception as exc:
                print(f"[{stimulus}/{response_range}] FAILED: {exc!r}")
                stored_Errors.append(f"[{stimulus}/{response_range}] FAILED: {exc!r}")

    # Merge per-worker caches into a single hyperparams.json for reruns.
    _merge_hyperparam_caches(out_dir, stim_types, response_ranges)
    for err in stored_Errors:
        print(err)


if __name__ == "__main__":
    # Set solve_hyperparam=False on reruns to reuse the cached C values from
    # dprime_RSA/hyperparams_*.json and skip GridSearchCV.
    main(solve_hyperparam=True)