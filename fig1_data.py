"""
Figure 1 data generation: everything that is CALCULATED for the panels.

Writes to figure_params.figure_data_dir("figure1"); fig1_figures.py only reads
these files. Steps (run any subset via main(steps=...)):

  spectrograms  Spectrograms of every natural / AM / pure-tone .wav file
                (panels 2 and 3). Needs only the sound files.
  pca           PC-space fits for one example session (panel 5): all regions
                together, and each region alone, for every stimulus type.
  rasters       Candidate example cells per stimulus type x region, ranked by
                stimulus selectivity, with their spike times (panel 4). Needs the
                raw ephys data through jaratoolbox. Cells named in
                figure_params.RASTER_CELLS are saved too.

Array orientation: firing-rate arrays are (nCells, nTrials); scikit-learn wants
(nTrials, nFeatures), so arrays are transposed before PCA.
"""
import ast
import glob
import os
import pickle
import re

import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.decomposition import PCA

import figure_params as fp

# ----------------------------------------------------------------------------
# Calculation constants
# ----------------------------------------------------------------------------
PCA_NEURONS = 20      # neurons drawn per region, as in the RSA/CCA pipeline
PCA_SEED = 0
PCA_ZSCORE = True    # False: mean-center only (as in PCA_analysis.py)
PCA_N_SAVED = 3       # leading PC scores stored per fit (all variance ratios kept)

N_RASTER_CANDIDATES = 4                       # per stimulus type x region
RASTER_SCORE_WINDOWS = ("onset", "sustained")
RASTER_MIN_RATE_HZ = 2.0                      # ignore near-silent cells


def _fr_db():
    from analysis_class import FiringRateAnalysis
    return FiringRateAnalysis(db_suffix="coords_updated")


# ----------------------------------------------------------------------------
# Spectrograms (panels 2, 3)
# ----------------------------------------------------------------------------
def compute_spectrogram(wav_path, duration, nperseg, overlap, fmin, fmax):
    """
    Spectrogram of the first `duration` seconds of a wav, in dB relative to its
    own peak (so sounds with different calibrated levels are comparable in
    structure). Returns (freqs_hz, times_s, S_db) with freqs cropped to
    [fmin, fmax].
    """
    sr, x = wavfile.read(wav_path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x[: int(round(duration * sr))]
    # Zero-pad half a window each side so frame centers span 0..duration instead
    # of leaving blank slivers at both edges (the stimuli are gated at 0 / duration).
    pad = nperseg // 2
    f, t, sxx = signal.spectrogram(np.pad(x, pad), fs=sr, window="hann", nperseg=nperseg,
                                   noverlap=int(nperseg * overlap), mode="psd")
    t = t - pad / sr
    keep = (f >= fmin) & (f <= min(fmax, sr / 2))
    s_db = 10 * np.log10(sxx[keep] + 1e-20)
    return f[keep], t, (s_db - s_db.max()).astype(np.float32)


def _wav_list(stim):
    """Return (wav_paths, labels) for a stimulus type, ordered as the stimuli are."""
    if stim == "naturalSound":
        paths = sorted(glob.glob(f"{fp.NATURAL_SOUND_DIR}/*.wav"))
        return paths, np.arange(len(paths))
    if stim == "AM":
        paths = glob.glob(f"{fp.SYNTHETIC_SOUND_DIR}/AM_*.wav")
        rates = []
        for p in paths:
            meta = ast.literal_eval(open(p.replace(".wav", ".txt")).read())
            rates.append(meta["modFrequency"])
    else:  # pureTones
        paths = glob.glob(f"{fp.SYNTHETIC_SOUND_DIR}/pt_*.wav")
        rates = [int(re.search(r"pt_(\d+)\.wav", p).group(1)) for p in paths]
    order = np.argsort(rates)
    return [paths[i] for i in order], np.array(rates)[order]


def build_spectrogram_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for stim in fp.STIM_TYPES:
        paths, labels = _wav_list(stim)
        params = fp.SPEC_PARAMS[stim]
        specs = []
        for p in paths:
            f, t, s_db = compute_spectrogram(
                p, fp.STIM_DURATION[stim], params["nperseg"], params["overlap"],
                params["fmin"], params["fmax"])
            specs.append(s_db)
        out = f"{data_dir}/spectrograms_{stim}.npz"
        np.savez_compressed(out, S_db=np.stack(specs), f=f, t=t, labels=labels,
                            files=np.array([os.path.basename(p) for p in paths]))
        print(f"[spectrograms] {stim}: {len(paths)} sounds -> {out}")


# ----------------------------------------------------------------------------
# Shared loading of the firing-rate arrays
# ----------------------------------------------------------------------------
def _load_stim_arrays(fr_db, stim, window):
    arrs = fr_db.return_arrays(stim)
    region = np.array(arrs["brainRegionArray"], dtype=object)
    region[region == "Posterior auditory area"] = "Dorsal auditory area"
    resp = arrs[f"{window}fr"]  # (nCells, nTrials)
    # Cells skipped during array generation are all-NaN rows.
    valid = ~np.isnan(resp).any(axis=1) & np.isin(region, fp.REGIONS)
    mouse = np.array(arrs["mouseIDArray"], dtype=str)
    date = np.array(arrs["sessionIDArray"], dtype=str)
    return dict(resp=resp, region=region, valid=valid,
                stim=np.asarray(arrs["stimArray"]),
                # sessionIDArray is only the date; pair it with the mouse so two
                # mice recorded on one date are never merged.
                session=np.char.add(np.char.add(mouse, "|"), date))


def _session_counts(d):
    """{session_key: {region: n valid neurons}}"""
    counts = {}
    for key in np.unique(d["session"][d["valid"]]):
        in_s = d["valid"] & (d["session"] == key)
        counts[key] = {r: int(np.sum(in_s & (d["region"] == r))) for r in fp.REGIONS}
    return counts


def pick_example_session(per_stim):
    """Session with the most neurons in its smallest region, over all stimulus types."""
    if fp.EXAMPLE_SESSION is not None:
        return "|".join(fp.EXAMPLE_SESSION)
    counts = {stim: _session_counts(d) for stim, d in per_stim.items()}
    shared = set.intersection(*[set(c) for c in counts.values()])
    if not shared:
        raise RuntimeError("No session has valid neurons for all stimulus types.")
    scored = []
    for key in shared:
        per_cell = [counts[s][key][r] for s in counts for r in fp.REGIONS]
        scored.append((min(per_cell), sum(per_cell), key))
    best_min, _, best = max(scored)
    if best_min < PCA_NEURONS:
        print(f"[pca] WARNING: best session ({best}) has only {best_min} neurons "
              f"in its smallest region (< {PCA_NEURONS}); region fits use fewer.")
    return best


# ----------------------------------------------------------------------------
# PC-space fits (panel 5)
# ----------------------------------------------------------------------------
def _fit_pca(features, stim_idx):
    """features: (nTrials, nNeurons)."""
    if PCA_ZSCORE:
        features = (features - features.mean(0)) / (features.std(0) + 1e-12)
    pca = PCA().fit(features)
    scores = pca.transform(features)[:, :PCA_N_SAVED]
    return dict(scores=scores, evr=pca.explained_variance_ratio_,
                stim_idx=stim_idx, n_neurons=features.shape[1])


def build_pca_data(data_dir, fr_db=None):
    fr_db = fr_db or _fr_db()
    window = fp.MAIN_RESPONSE_RANGE
    per_stim = {s: _load_stim_arrays(fr_db, s, window) for s in fp.STIM_TYPES}
    session = pick_example_session(per_stim)
    print(f"[pca] example session (mouse|date): {session}")

    rng = np.random.default_rng(PCA_SEED)
    results = {"meta": dict(session=session, window=window, zscore=PCA_ZSCORE,
                            seed=PCA_SEED, neurons_per_region=PCA_NEURONS)}
    for stim, d in per_stim.items():
        idx = np.where(d["valid"] & (d["session"] == session))[0]
        stim_rows = d["stim"][idx]
        same_trials = np.all(stim_rows == stim_rows[0], axis=1)
        if not same_trials.all():
            print(f"[pca] {stim}: dropping {np.sum(~same_trials)} cells whose "
                  f"trial order differs from the session's first cell")
        idx, stim_vec = idx[same_trials], stim_rows[0]
        uniq = np.unique(stim_vec)
        stim_idx = np.searchsorted(uniq, stim_vec)

        chosen = {}
        for region in fp.REGIONS:
            r_idx = idx[d["region"][idx] == region]
            n_use = min(PCA_NEURONS, len(r_idx))
            chosen[region] = np.sort(rng.choice(r_idx, n_use, replace=False))

        out = {"uniq_stims": uniq}
        for region in fp.REGIONS:
            out[region] = _fit_pca(d["resp"][chosen[region]].T, stim_idx)
        all_idx = np.concatenate([chosen[r] for r in fp.REGIONS])
        out["all_regions"] = _fit_pca(d["resp"][all_idx].T, stim_idx)
        results[stim] = out
        print(f"[pca] {stim}: neurons per region "
              f"{ {fp.REGION_SHORT[r]: len(chosen[r]) for r in fp.REGIONS} }, "
              f"{len(uniq)} stimuli, {len(stim_vec)} trials")

    path = f"{data_dir}/pca_data.pkl"
    os.makedirs(data_dir, exist_ok=True)
    pickle.dump(results, open(path, "wb"))
    print(f"[pca] saved {path}")


# ----------------------------------------------------------------------------
# Raster candidates (panel 4)
# ----------------------------------------------------------------------------
def _eta_squared(values, group_idx, n_groups):
    """Fraction of response variance explained by stimulus identity."""
    counts = np.bincount(group_idx, minlength=n_groups)
    means = np.bincount(group_idx, weights=values, minlength=n_groups) / np.maximum(counts, 1)
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    if ss_total == 0:
        return 0.0
    return float(np.sum(counts * (means - grand) ** 2) / ss_total)


def rank_cells(fr_db, stim):
    """Return a list of dicts (row, region, score, window, rate), best first."""
    arrs = fr_db.return_arrays(stim)
    region = np.array(arrs["brainRegionArray"], dtype=object)
    region[region == "Posterior auditory area"] = "Dorsal auditory area"
    stim_rows = np.asarray(arrs["stimArray"])
    rows = []
    for i in range(len(region)):
        if region[i] not in fp.REGIONS or np.isnan(stim_rows[i]).any():
            continue
        _, g = np.unique(stim_rows[i], return_inverse=True)
        best = None
        for window in RASTER_SCORE_WINDOWS:
            v = arrs[f"{window}fr"][i]
            if np.isnan(v).any() or v.mean() < RASTER_MIN_RATE_HZ:
                continue
            score = _eta_squared(v, g, g.max() + 1)
            if best is None or score > best["score"]:
                best = dict(row=i, region=region[i], score=score, window=window,
                            rate=float(v.mean()))
        if best:
            rows.append(best)
    return sorted(rows, key=lambda r: -r["score"])


def _check_array_celldb_alignment(fr_db, subset, stim):
    arrs = fr_db.return_arrays(stim)
    same = (len(subset) == len(arrs["brainRegionArray"])
            and np.array_equal(subset["simpleSiteName"].to_numpy(dtype=str),
                               np.array(arrs["brainRegionArray"], dtype=str))
            and np.array_equal(subset["subject"].to_numpy(dtype=str),
                               np.array(arrs["mouseIDArray"], dtype=str))
            and np.array_equal(subset["date"].to_numpy(dtype=str),
                               np.array(arrs["sessionIDArray"], dtype=str)))
    if not same:
        raise RuntimeError(
            f"Firing-rate array rows for {stim} do not line up with "
            f"get_subset_by_areas(); the arrays were probably generated from a "
            f"different database, so row -> cell lookups would be wrong.")


def load_cell_spikes(subset_row, stim, stim_info):
    """Event-locked spike times and sorted-trial positions for one cell."""
    from jaratoolbox import ephyscore, spikesanalysis
    n_trials = stim_info["nTrials"]
    time_range = stim_info["time_range"]
    ephys, bdata = ephyscore.Cell(subset_row).load(stim)
    spike_times = ephys["spikeTimes"]
    onsets = ephys["events"]["stimOn"][:n_trials]
    current = np.asarray(bdata[stim_info["stim_var"]][:n_trials])
    if len(current) == len(onsets) - 1:
        onsets = onsets[:len(current)]
    elif len(current) != len(onsets):
        raise RuntimeError(f"behavior trials ({len(current)}) != ephys trials ({len(onsets)})")

    spk_t, trial_idx, _ = spikesanalysis.eventlocked_spiketimes(
        spike_times, onsets, time_range)
    # Same stable stimulus sort as funcs.calculate_fr_arrays.
    order = np.argsort(current, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    uniq, counts = np.unique(current, return_counts=True)
    return dict(spike_times=np.asarray(spk_t, dtype=np.float32),
                trial_rank=rank[np.asarray(trial_idx)].astype(np.int32),
                n_trials=len(current), uniq_stims=uniq, trials_per_stim=counts,
                time_range=time_range)


def build_raster_data(data_dir, fr_db=None):
    fr_db = fr_db or _fr_db()
    subset = fr_db.get_subset_by_areas()
    results = {}
    for stim in fp.STIM_TYPES:
        _check_array_celldb_alignment(fr_db, subset, stim)
        ranked = rank_cells(fr_db, stim)
        wanted = []
        for region in fp.REGIONS:
            wanted += [dict(c, candidate=True) for c in
                       [c for c in ranked if c["region"] == region][:N_RASTER_CANDIDATES]]
        chosen = fp.RASTER_CELLS.get(stim)
        if chosen is not None and chosen not in [w["row"] for w in wanted]:
            info = next((c for c in ranked if c["row"] == chosen), None)
            wanted.append(dict(info or dict(row=chosen, region=None, score=np.nan,
                                            window=None, rate=np.nan), candidate=False))

        cells = []
        for c in wanted:
            row = subset.iloc[c["row"]]
            try:
                spikes = load_cell_spikes(row, stim, fr_db.stim_info[stim])
            except Exception as exc:
                print(f"[rasters] {stim} row {c['row']}: skipped ({exc!r})")
                continue
            cells.append(dict(c, mouse=str(row["subject"]), date=str(row["date"]),
                              celldb_index=int(row["index"]) if "index" in row else None,
                              **spikes))
            print(f"[rasters] {stim} row {c['row']} ({c['region']}, score {c['score']:.2f})")
        results[stim] = cells

    path = f"{data_dir}/raster_data.pkl"
    os.makedirs(data_dir, exist_ok=True)
    pickle.dump(results, open(path, "wb"))
    print(f"[rasters] saved {path}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(steps=("spectrograms", "pca", "rasters"), data_dir=None):
    data_dir = data_dir or fp.figure_data_dir("figure1")
    fr_db = _fr_db() if {"pca", "rasters"} & set(steps) else None
    if "spectrograms" in steps:
        build_spectrogram_data(data_dir)
    if "pca" in steps:
        build_pca_data(data_dir, fr_db)
    if "rasters" in steps:
        build_raster_data(data_dir, fr_db)


if __name__ == "__main__":
    main()
