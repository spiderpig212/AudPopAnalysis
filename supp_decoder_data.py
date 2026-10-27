"""
Supporting-panel data for final_figures_dprime.py: everything that is CALCULATED.

Writes to figure_params.figure_data_dir("supp_decoder"); supp_decoder_figures.py
only reads these files.

  pair_data.pkl     Per example pair (AM, pure tones, natural within-source,
                    natural between-source): the pair's trials in PC space and a
                    linear SVC fit to them (same recipe as
                    d_prime_RSA_linear_decoder.train_pairwise_decoder).
  heatmap_data.pkl  The example session's RSA matrices for each stimulus type
                    (the matrices the near-diagonal and within/between box-plot
                    points are averaged from), plus the example pairs' positions.

The PC-space fit is a 2-D decoder: PCA on just the two stimuli's trials, then the
linear SVC on PC1 and PC2. It is NOT the 20-neuron decoder behind the heat maps,
so its accuracy / D' will differ from the matrix entry for the same pair.
"""
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import figure_params as fp
import fig1_data as f1
from d_prime_RSA_linear_decoder import C_GRID, CV_FOLDS, dprime_from_projections
from d_prime_RSA_visualization import load_rsa_data

FIG = "supp_decoder"
PAIR_PCA_ZSCORE = False   # False = raw rates, as the decoder sees them
PAIR_SEED = 0
# PAIR_SELECTION_METRIC = "accuracy_matrix"
PAIR_SELECTION_METRIC = "J_matrix"
METRIC_KEYS = ("J_matrix", "dprime_matrix", "accuracy_matrix")

# (key, data_source, target_region); brain_region is always fp.SUPP_REGION.
HEATMAP_SOURCES = [("raw", "raw", None),
                   ("cca_Ventral auditory area", "cca", "Ventral auditory area"),
                   ("cca_Dorsal auditory area", "cca", "Dorsal auditory area")]


# ----------------------------------------------------------------------------
# Example matrices from the RSA pickles
# ----------------------------------------------------------------------------
def example_matrices(rsa_dir, stim, date, window):
    """{source_key: {metric: matrix}} for the example session's date."""
    records = load_rsa_data(rsa_dir, stim)
    out = {}
    for key, source, target in HEATMAP_SOURCES:
        for r in records:
            if (r["data_source"] == source and r["response_range"] == window
                    and str(r["session"]) == str(date)
                    and r["brain_region"] == fp.SUPP_REGION
                    and r["target_region"] == target):
                out[key] = {m: r[m] for m in METRIC_KEYS if r.get(m) is not None}
                break
    return out


def _closest_to_median(values, candidates):
    """Candidate whose value is closest to the median of all `values`."""
    values = np.asarray(values)
    return candidates[int(np.argmin(np.abs(values - np.median(values))))]


def select_pairs(stim, mat):
    """
    {panel_key: (idx_a, idx_b)} for one stimulus type. With a matrix, picks the
    pair closest to the median of the entries feeding the box plots; without one,
    falls back to fixed pairs.
    """
    if stim in ("AM", "pureTones"):
        if mat is None:
            n = 11 if stim == "AM" else 16
            k = (n - 1) // 2
            return {stim: (k, k + 1)}
        n = mat.shape[0]
        cands = [(i, i + 1) for i in range(n - 1)]
        return {stim: _closest_to_median([mat[a, b] for a, b in cands], cands)}

    if mat is None:
        return {"naturalSound_within": (0, 1), "naturalSound_between": (0, fp.EXEMPLARS_PER_CATEGORY)}
    n = mat.shape[0]
    src = np.arange(n) // fp.EXEMPLARS_PER_CATEGORY
    ii, jj = np.triu_indices(n, k=1)
    pairs = list(zip(ii, jj))
    out = {}
    for name, same in (("naturalSound_within", True), ("naturalSound_between", False)):
        cands = [(a, b) for a, b in pairs if (src[a] == src[b]) == same]
        out[name] = _closest_to_median([mat[a, b] for a, b in cands], cands)
    return out


# ----------------------------------------------------------------------------
# 2-D decoder in PC space
# ----------------------------------------------------------------------------
def fit_pair_2d(features, stim_idx, a, b):
    """
    features: (nTrials, nNeurons). PCA on the two stimuli's trials, then the
    pipeline's recipe (GridSearchCV over C with CV_FOLDS folds, refit on all
    trials, unit-normalised w for the D' projection) on PC1 and PC2.
    """
    mask = (stim_idx == a) | (stim_idx == b)
    x = features[mask]
    y = (stim_idx[mask] == b).astype(int)  # class 1 = stimulus b
    if PAIR_PCA_ZSCORE:
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    pca = PCA().fit(x)
    scores = pca.transform(x)[:, :2]

    search = GridSearchCV(SVC(kernel="linear"), {"C": C_GRID}, cv=CV_FOLDS).fit(scores, y)
    best_c = float(search.best_params_["C"])
    model = SVC(kernel="linear", C=best_c).fit(scores, y)
    w = model.coef_.ravel()
    proj = scores @ (w / np.linalg.norm(w))
    return dict(scores=scores, pair_idx=y, evr=pca.explained_variance_ratio_[:2],
                w=w, b=float(model.intercept_[0]), C=best_c,
                cv_accuracy=float(search.best_score_),
                J=dprime_from_projections(proj[y == 0], proj[y == 1], metric="J"),
                n_neurons=features.shape[1], pair=(int(a), int(b)))


def _session_features(d, session, region, rng):
    """(nTrials, nNeurons) sustained rates of PCA_NEURONS random neurons, per-trial
    stimulus index, and the unique stimulus values."""
    idx = np.where(d["valid"] & (d["session"] == session) & (d["region"] == region))[0]
    stim_rows = d["stim"][idx]
    same = np.all(stim_rows == stim_rows[0], axis=1)
    idx, stim_vec = idx[same], stim_rows[0]
    chosen = np.sort(rng.choice(idx, min(f1.PCA_NEURONS, len(idx)), replace=False))
    uniq = np.unique(stim_vec)
    return d["resp"][chosen].T, np.searchsorted(uniq, stim_vec), uniq


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def build_supp_data(data_dir, rsa_dir=None, fr_db=None):
    fr_db = fr_db or f1._fr_db()
    rsa_dir = rsa_dir or f"{fp.FIGDATA_ROOT}/dprime_RSA"
    window = fp.MAIN_RESPONSE_RANGE
    per_stim = {s: f1._load_stim_arrays(fr_db, s, window) for s in fp.STIM_TYPES}
    session = f1.pick_example_session(per_stim)
    mouse, date = session.split("|")
    print(f"[supp] example session (mouse|date): {session}, region: {fp.SUPP_REGION}")

    # The RSA pipeline identifies a session by date alone, so two mice recorded
    # on one date would have been pooled in its matrices.
    mice_that_date = np.unique(np.concatenate([
        d["session"][d["valid"] & np.char.endswith(d["session"], "|" + date)]
        for d in per_stim.values()]))
    if len(mice_that_date) > 1:
        print(f"[supp] WARNING: {len(mice_that_date)} mice share {date} "
              f"({[str(m) for m in mice_that_date]}); the RSA matrices pool their neurons, so "
              f"the heat map and the single-session PC panels describe different cells.")

    rng = np.random.default_rng(PAIR_SEED)
    pair_results = {"meta": dict(session=session, region=fp.SUPP_REGION, window=window,
                                 zscore=PAIR_PCA_ZSCORE, seed=PAIR_SEED)}
    heat_results = {"meta": dict(session=session, region=fp.SUPP_REGION, window=window)}

    for stim in fp.STIM_TYPES:
        try:
            mats = example_matrices(rsa_dir, stim, date, window)
        except Exception as exc:
            print(f"[supp] {stim}: could not read RSA pickles ({exc!r}); using fixed pairs")
            mats = {}
        raw = mats.get("raw", {})
        sel = raw.get(PAIR_SELECTION_METRIC)
        if sel is None:
            print(f"[supp] {stim}: no raw {fp.SUPP_REGION} matrix for {date}; fixed pairs")
        pairs = select_pairs(stim, sel)
        for key in pairs:
            if fp.SUPP_PAIRS.get(key) is not None:
                pairs[key] = tuple(fp.SUPP_PAIRS[key])

        features, stim_idx, uniq = _session_features(per_stim[stim], session, fp.SUPP_REGION, rng)
        for key, (a, b) in pairs.items():
            pair_results[key] = dict(fit_pair_2d(features, stim_idx, a, b), stim=stim, uniq_stims=uniq)
            r = pair_results[key]
            print(f"[supp] {key}: pair {r['pair']}, 2-D SVC accuracy {r['cv_accuracy']:.2f}, "
                  f"D' {r['J']:.2f}, C={r['C']}")
        heat_results[stim] = dict(uniq_stims=uniq, sources=mats, pairs=pairs)

    os.makedirs(data_dir, exist_ok=True)
    pickle.dump(pair_results, open(f"{data_dir}/pair_data.pkl", "wb"))
    pickle.dump(heat_results, open(f"{data_dir}/heatmap_data.pkl", "wb"))
    print(f"[supp] saved pair_data.pkl and heatmap_data.pkl in {data_dir}")


def main(data_dir=None, rsa_dir=None):
    build_supp_data(data_dir or fp.figure_data_dir(FIG), rsa_dir)


if __name__ == "__main__":
    main()
