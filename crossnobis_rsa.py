import numpy as np
import pandas as pd
import rsatoolbox
from rsatoolbox.data.noise import prec_from_measurements
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

import matplotlib.pyplot as plt
import seaborn as sns

from analysis_class import AnalysisBase, FiringRateAnalysis


response_ranges = ["onset", "sustained", "offset"]
neuron_threshold = 20
n_splits = 5

fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
# stim_types = fr_db.stim_types
stim_types = ['pureTones', 'AM']

significant_df = pd.read_csv(
                    f"{file_path}/CCA_two_region_analysis/cca_primary_auditory_results.csv")

def assign_cv_folds(stim_labels, n_splits, rng=None):
    """Assign each trial to one of ``n_splits`` cross-validation folds,
    balanced within each stimulus condition so every fold sees every stimulus.

    Returns an integer fold array aligned with ``stim_labels`` or None if any
    condition has fewer trials than ``n_splits``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    folds = np.full(stim_labels.shape[0], -1, dtype=int)
    for stim in np.unique(stim_labels):
        idx = np.where(stim_labels == stim)[0]
        if idx.size < n_splits:
            # Not enough repeats of this condition for cross-validation
            return None
        idx = rng.permutation(idx)
        # Cycle fold ids so folds are as balanced as possible
        folds[idx] = np.arange(idx.size) % n_splits
    return folds


def compute_crossnobis_rdm(measurements, stim_labels, session, n_splits,
                           reduce_dim=None, noise_method="shrinkage_eye"):
    """Build an rsatoolbox Dataset and compute the crossnobis RDM.

    measurements : (n_trials, n_neurons) firing rates
    stim_labels  : (n_trials,) stimulus condition per trial
    reduce_dim   : None      -> use rsatoolbox's built-in crossnobis on full space
                   int / float -> number (or explained-variance fraction) of PCs;
                   triggers the leak-free manual cross-validation path.
    """
    folds = assign_cv_folds(stim_labels, n_splits)
    if folds is None:
        return None

    if reduce_dim is not None:
        return _crossnobis_pca_manual(
            measurements, np.asarray(stim_labels), folds, reduce_dim
        )

    dataset = rsatoolbox.data.Dataset(
        measurements=measurements,
        descriptors={"session": str(session)},
        obs_descriptors={
            "stimulus": np.asarray(stim_labels),
            "fold": folds,
        },
        channel_descriptors={"neuron": np.arange(measurements.shape[1])},
    )

    noise = prec_from_measurements(
        dataset,
        obs_desc="stimulus",
        method=noise_method,
    )

    rdm = rsatoolbox.rdm.calc_rdm(
        dataset,
        method="crossnobis",
        descriptor="stimulus",
        noise=noise,
        cv_descriptor="fold",
    )
    return rdm


def _crossnobis_pca_manual(measurements, stim_labels, folds, reduce_dim):
    """Leak-free crossnobis in a PCA-reduced, whitened space.

    For each held-out fold we fit PCA + noise precision on the *training* folds
    only, project train & test condition means into that space, whiten, and
    accumulate the cross-validated inner product. This keeps the crossnobis
    estimator unbiased even though we denoise with dimensionality reduction.
    Returns an rsatoolbox RDMs object so downstream code is identical.
    """
    conds = np.unique(stim_labels)
    n_cond = conds.size
    unique_folds = np.unique(folds)
    n_fold = unique_folds.size

    accum = np.zeros((n_cond, n_cond))
    n_terms = 0

    for test_fold in unique_folds:
        train_mask = folds != test_fold
        test_mask = folds == test_fold

        Xtr, ytr = measurements[train_mask], stim_labels[train_mask]
        Xte, yte = measurements[test_mask], stim_labels[test_mask]

        # Require every condition present in both splits for this fold
        if not (set(conds) <= set(ytr) and set(conds) <= set(yte)):
            continue

        # --- Fit reducer + whitening on TRAIN only ---
        pca = PCA(n_components=reduce_dim, svd_solver="full")
        Ztr = pca.fit_transform(Xtr)          # (n_train, k)
        Zte = pca.transform(Xte)              # project test with train PCA

        # Residual precision (whitening) in the reduced space, train residuals
        resid = np.vstack([
            Ztr[ytr == c] - Ztr[ytr == c].mean(0, keepdims=True) for c in conds
        ])
        cov = LedoitWolf().fit(resid).covariance_
        # symmetric inverse-sqrt whitening
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-8, None)
        whiten = V @ np.diag(1.0 / np.sqrt(w)) @ V.T

        # Whitened condition-mean patterns for train and test folds
        Btr = np.vstack([Ztr[ytr == c].mean(0) for c in conds]) @ whiten
        Bte = np.vstack([Zte[yte == c].mean(0) for c in conds]) @ whiten

        # Cross-fold outer product of pattern differences -> unbiased d^2
        # d_ij = (b_i - b_j)_train . (b_i - b_j)_test / P
        diff_tr = Btr[:, None, :] - Btr[None, :, :]
        diff_te = Bte[:, None, :] - Bte[None, :, :]
        accum += np.einsum("ijk,ijk->ij", diff_tr, diff_te)
        n_terms += 1

    if n_terms == 0:
        return None

    rdm_matrix = accum / (n_terms * measurements.shape[1])
    np.fill_diagonal(rdm_matrix, 0.0)

    rdm = rsatoolbox.rdm.RDMs(
        dissimilarities=rdm_matrix[np.triu_indices(n_cond, k=1)][None, :],
        dissimilarity_measure="crossnobis_pca",
        pattern_descriptors={"stimulus": conds},
    )
    return rdm

def plot_rdm_heatmap(rdm, ordered_stims, title, save_path=None):
    """Heatmap of a single crossnobis RDM, reordered along the stimulus axis."""
    rdm_matrix = rdm.get_matrices()[0]
    rdm_stims = np.asarray(rdm.pattern_descriptors["stimulus"])
    order = np.asarray([np.where(rdm_stims == s)[0][0]
                        for s in ordered_stims if s in rdm_stims])
    rdm_matrix = rdm_matrix[np.ix_(order, order)]
    labels = [str(s) for s in ordered_stims if s in rdm_stims]

    fig, ax = plt.subplots(figsize=(6, 5))
    # crossnobis can be negative -> center the diverging colormap at 0
    vmax = np.nanmax(np.abs(rdm_matrix))
    sns.heatmap(rdm_matrix, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                xticklabels=labels, yticklabels=labels, square=True,
                cbar_kws={"label": "crossnobis $d^2$"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("stimulus")
    ax.set_ylabel("stimulus")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_fisher_across_regions(results_df, title, save_path=None,
                               pair_labels=None):
    """Bar plot of mean Fisher information per region (mean +/- SEM over
    sessions), plus the per-adjacent-pair Fisher information profile.

    pair_labels : optional list of strings labelling each adjacent stimulus
                  pair, e.g. ["4->8", "8->16", ...] for the along-axis panel.
    """
    if results_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (1) Mean Fisher info per region, aggregated over sessions
    agg = (results_df
           .groupby("brain_region")["mean_fisher_info"]
           .agg(["mean", "sem"])
           .reset_index())
    axes[0].bar(agg["brain_region"], agg["mean"], yerr=agg["sem"],
                capsize=4, color="steelblue")
    axes[0].set_ylabel("mean near-diagonal Fisher info")
    axes[0].set_title("Local Fisher information by region")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].axhline(0, color="k", lw=0.8)

    # (2) Fisher info as a function of position along the stimulus axis
    for region, g in results_df.groupby("brain_region"):
        profiles = [p for p in g["fisher_info_per_pair"]
                    if np.ndim(p) == 1 and len(p)]
        if not profiles:
            continue
        lengths = {len(p) for p in profiles}
        if len(lengths) != 1:
            raise ValueError(
                f"Region '{region}' has Fisher profiles of differing lengths "
                f"{lengths}; all sessions must share the same stimulus set."
            )
        stacked = np.vstack(profiles)
        mean_prof = np.nanmean(stacked, axis=0)
        sem_prof = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
        x = np.arange(mean_prof.size)
        axes[1].plot(x, mean_prof, marker="o", label=region)
        axes[1].fill_between(x, mean_prof - sem_prof, mean_prof + sem_prof,
                             alpha=0.2)
    axes[1].set_xlabel("adjacent-stimulus pair index")
    axes[1].set_ylabel("Fisher info (crossnobis $d^2$ / $\\Delta s^2$)")
    axes[1].set_title("Fisher information along stimulus axis")
    axes[1].axhline(0, color="k", lw=0.8)
    if pair_labels is not None:
        axes[1].set_xticks(np.arange(len(pair_labels)))
        axes[1].set_xticklabels(pair_labels, rotation=30, fontsize=7)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig

def reorder_rdm_matrix(rdm, ordered_stims):
    """Return the square RDM matrix reordered along the shared stimulus axis."""
    rdm_matrix = rdm.get_matrices()[0]
    rdm_stims = np.asarray(rdm.pattern_descriptors["stimulus"])
    order = np.asarray([np.where(rdm_stims == s)[0][0]
                        for s in ordered_stims if s in rdm_stims])
    return rdm_matrix[np.ix_(order, order)]


def plot_region_response_grid(grid_rdms, ordered_stims, regions, responses,
                              title, save_path=None):
    """3x3 grid of session-averaged crossnobis RDMs.

    grid_rdms : dict keyed by (region, response) -> list of reordered RDM
                matrices (one per session). Each list is averaged for display.
    regions   : ordered list of brain regions (grid rows)
    responses : ordered list of response ranges (grid columns)
    """
    n_rows, n_cols = len(regions), len(responses)
    labels = [str(s) for s in ordered_stims]

    # Shared symmetric color scale across all panels for comparability
    all_means = []
    for key, mats in grid_rdms.items():
        if mats:
            all_means.append(np.nanmean(np.stack(mats, axis=0), axis=0))
    vmax = np.nanmax([np.nanmax(np.abs(m)) for m in all_means]) if all_means else 1.0

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.6 * n_rows),
                             squeeze=False)

    for r, region in enumerate(regions):
        for c, response in enumerate(responses):
            ax = axes[r][c]
            mats = grid_rdms.get((region, response), [])
            if mats:
                avg = np.nanmean(np.stack(mats, axis=0), axis=0)
                n_sess = len(mats)
                sns.heatmap(
                    avg, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=labels, yticklabels=labels, square=True,
                    cbar=(c == n_cols - 1),
                    cbar_kws={"label": "crossnobis $d^2$"},
                    ax=ax,
                )
                ax.set_title(f"n_sess={n_sess}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])

            # Row labels (regions) on the left column
            if c == 0:
                ax.set_ylabel(region, fontsize=10, fontweight="bold")
            # Column labels (response ranges) on the top row
            if r == 0:
                ax.set_title(f"{response}\n" + (ax.get_title() or ""),
                             fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig

def fisher_information_from_rdm(rdm, ordered_stims, delta_stim=1.0):
    """Estimate linear Fisher information from the near (first off-) diagonal
    of a crossnobis RDM.

    The crossnobis squared distance between two conditions is an unbiased
    estimate of the (linear) Fisher discriminant information separating them.
    For an ordered stimulus axis, dividing the distance between *adjacent*
    conditions by the squared stimulus step gives a local Fisher information
    estimate; averaging over the first off-diagonal summarises the population.
    """
    # Full square RDM, reordered so rows/cols follow the ordered stimulus axis.
    rdm_matrix = rdm.get_matrices()[0]
    rdm_stims = np.asarray(rdm.pattern_descriptors["stimulus"])

    order = [np.where(rdm_stims == s)[0][0]
             for s in ordered_stims if s in rdm_stims]
    order = np.asarray(order)

    # All sessions share the same stimulus set/order for AM & pure tones,
    # so the RDM should contain every ordered stimulus. Fail loudly otherwise.
    recovered = [s for s in ordered_stims if s in rdm_stims]
    if len(recovered) != len(ordered_stims):
        raise ValueError(
            f"RDM is missing stimuli: expected {list(ordered_stims)}, "
            f"got {recovered}"
        )

    rdm_matrix = rdm_matrix[np.ix_(order, order)]

    # First off-diagonal = distances between adjacent stimuli.
    near_diag = np.diag(rdm_matrix, k=1)
    per_pair = near_diag / (delta_stim ** 2)

    return {
        "per_pair": per_pair,
        "mean_fisher_info": float(np.mean(per_pair)) if per_pair.size else np.nan,
    }

for stimulus in stim_types:
    print(f"Computing RSA for {stimulus}")
    stim_arrays = fr_db.return_arrays(stimulus)
    brainRegionArray = stim_arrays["brainRegionArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    uniqSessions = np.unique(sessionArray)
    # Rename posterior to also be dorsal instead of treating them as separate areas
    brainRegionArray[brainRegionArray == "Posterior auditory area"] = "Dorsal auditory area"
    uniq_regions = np.unique(brainRegionArray)
    # Dropping Temporal auditory area
    uniq_regions = uniq_regions[uniq_regions != "Temporal association areas"]
    uniqStims = np.unique(stim_arrays["stimArray"][0, :])

    # Accumulates session-averaged RDM inputs across ALL response ranges
    # for this stimulus, keyed by (brain_region, response_range).
    grid_rdms = {}

    for response_range in response_ranges:
        rsa_data = []
        respArray = stim_arrays[f"{response_range}fr"]

        for session in uniqSessions:
            session_mask = sessionArray == session
            session_resp_array = respArray[session_mask, :]
            brain_session_array = brainRegionArray[session_mask]
            # stimulus label for every trial in this session
            session_stim_array = stim_arrays["stimArray"][session_mask, :]
            session_stim_array = session_stim_array[0, :]

            for brain_region in uniq_regions:
                region_mask = brain_session_array == brain_region
                primary_array = session_resp_array[region_mask, :].T

                # Need enough neurons; skip empty/small populations
                n_neurons = primary_array.shape[1]
                if primary_array.shape[0] == 0 or n_neurons < neuron_threshold:
                    continue

                rdm = compute_crossnobis_rdm(
                    measurements=primary_array,
                    stim_labels=session_stim_array,
                    session=session,
                    n_splits=n_splits,
                    reduce_dim=5,  # e.g. 10 or 0.9 for PCA path. None uses RSA toolkit default
                    noise_method="shrinkage_eye",
                )
                if rdm is None:
                    continue

                fisher_info = fisher_information_from_rdm(
                    rdm, ordered_stims=uniqStims
                )

                plot_rdm_heatmap(
                    rdm, uniqStims,
                    title=f"{stimulus} | {response_range} | "
                          f"{brain_region} | sess {session}",
                    save_path=f"{file_path}/crossnobis_rsa/rdm_"
                              f"{stimulus}_{response_range}_{brain_region}_"
                              f"{session}.png",
                )

                # Collect the reordered matrix for the group-average grid
                grid_rdms.setdefault((brain_region, response_range), []).append(
                    reorder_rdm_matrix(rdm, uniqStims)
                )

                rsa_data.append(
                    {
                        "stimulus": stimulus,
                        "response_range": response_range,
                        "session": session,
                        "brain_region": brain_region,
                        "n_neurons": n_neurons,
                        "mean_fisher_info": fisher_info["mean_fisher_info"],
                        "fisher_info_per_pair": fisher_info["per_pair"],
                        "rdm": rdm,
                    }
                )

            rsa_results_df = pd.DataFrame(rsa_data)
            out_name = (
                f"{file_path}/crossnobis_rsa/"
                f"crossnobis_{stimulus}_{response_range}.pkl"
            )
            rsa_results_df.to_pickle(out_name)
            print(f"  Saved {len(rsa_results_df)} region/session RDMs -> {out_name}")

            pair_labels = [f"{uniqStims[i]}\u2192{uniqStims[i + 1]}"
                           for i in range(len(uniqStims) - 1)]
            plot_fisher_across_regions(
                rsa_results_df,
                title=f"Fisher information | {stimulus} | {response_range}",
                save_path=f"{file_path}/crossnobis_rsa/fisher_"
                          f"{stimulus}_{response_range}.png",
                pair_labels=pair_labels,
            )

            # One 3x3 (regions x response ranges) averaged-RDM figure per stimulus
            plot_region_response_grid(
                grid_rdms,
                ordered_stims=uniqStims,
                regions=list(uniq_regions),
                responses=response_ranges,
                title=f"Session-averaged crossnobis RDMs | {stimulus}",
                save_path=f"{file_path}/crossnobis_rsa/"
                          f"avg_rdm_grid_{stimulus}.png",
            )