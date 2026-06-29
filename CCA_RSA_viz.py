import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
from statsmodels.stats.multitest import multipletests
from analysis_class import FiringRateAnalysis

fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
SOUND_CATEGORIES = ['Frogs', 'Crickets', 'Streamside', 'Bubbling', 'Bees']
EXEMPLARS_PER_CATEGORY = 4

def get_stim_labels(stimulus, uniqStims, sounds_categories=None, exemplars_per_category=None):
    """
    Build tick labels for a given stimulus type.

    Parameters
    ----------
    stimulus : str
        One of the stimulus types (e.g. "AM", "pureTones", "naturalSounds").
    uniqStims : array-like
        The unique stimulus identifiers from stim_arrays["stimArray"][0, :].
    sounds_categories : list of str, optional
        Ordered category names for natural sounds (e.g. ["Frogs", "Crickets", ...]).
    exemplars_per_category : int, optional
        Number of exemplars per category for natural sounds.

    Returns
    -------
    list of str
        Tick labels aligned with the order of uniqStims.
    """
    if stimulus == "naturalSound":
        if sounds_categories is None or exemplars_per_category is None:
            raise ValueError(
                "sounds_categories and exemplars_per_category are required for naturalSound"
            )
        labels = [
            f"{category}_{exemplar}"
            for category in sounds_categories
            for exemplar in range(exemplars_per_category)
        ]
        # natural sound ids are 0-19, so index into the generated labels
        return [labels[int(s)] for s in uniqStims]

    # AM / pureTones: use the raw stimulus values directly
    return [str(s) for s in uniqStims]


def plot_rsa_heatmaps(rsa_frame, stim_labels=None, save_path=None):
    """
    Plot one RSA heatmap per response_range (averaging across sessions).

    Parameters
    ----------
    rsa_frame : pd.DataFrame
        Must contain 'rsa_matrix', 'response_range', 'brain_region',
        'target_region1', 'target_region2'.
    stim_labels : list of str, optional
        Tick labels for both axes. If None, integer indices are used.
    save_path : str, optional
        If provided, the figure is saved to this path.
    """
    response_ranges = rsa_frame["response_range"].unique()
    n_ranges = len(response_ranges)

    fig, axes = plt.subplots(1, n_ranges, figsize=(6 * n_ranges, 5.5), squeeze=False)
    axes = axes.ravel()

    for ax, response_range in zip(axes, response_ranges):
        sub = rsa_frame[rsa_frame["response_range"] == response_range]

        avg_matrix = np.mean(np.stack(sub["rsa_matrix"].values), axis=0)

        row = sub.iloc[0]
        x_label = f"{row['brain_region']}–{row['target_region2']} subspace"
        y_label = f"{row['brain_region']}–{row['target_region1']} subspace"

        sns.heatmap(
            avg_matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=stim_labels if stim_labels is not None else "auto",
            yticklabels=stim_labels if stim_labels is not None else "auto",
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title(f"RSA: {response_range}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig, axes

def create_RDM_plots():
    """
    Pulls correlation matrices for each session and averages across a brain region
    to make a general RDM for each stim type, brain area, and response_range.
    Then plots:
      1. Per-stimulus heatmap grid (rows = response_range, cols = region) of
         the session-averaged RDM.
      2. A per-stimulus boxplot of upper-triangle correlations with
         response_range on the x-axis and boxes colored by brain region.
    """

    response_range_order = ['onset', 'sustained', 'offset']

    for stim in ["AM", "pureTones", "naturalSound"]:
        stim_arrays = fr_db.return_arrays(stim)
        uniqStims = np.unique(stim_arrays["stimArray"][0, :])
        rsa_frame = pd.DataFrame()
        for response_range in response_range_order:
            try:
                with open(f"{file_path}/CCA_RSA/rsa_data_{stim}_{response_range}.pkl", 'rb') as f:
                    rsa_pkl = pickle.load(f)
                    rsa_frame = pd.concat([rsa_frame, pd.DataFrame(rsa_pkl)], ignore_index=True)
            except FileNotFoundError:
                print(f"No data found for {stim}, skipping.")
                continue

        # Drop rows that have NaN for 'rsa_matrix'
        rsa_frame = rsa_frame.dropna(subset=['rsa_matrix'])
        if rsa_frame.empty:
            print(f"No data found for {stim}, skipping.")
            continue

        stim_labels = get_stim_labels(
            stim,
            uniqStims,
            sounds_categories=SOUND_CATEGORIES,
            exemplars_per_category=EXEMPLARS_PER_CATEGORY,
        )

        fig, axes = plot_rsa_heatmaps(
            rsa_frame,
            stim_labels=stim_labels,
            save_path=f"{file_path}/CCA_RSA/rsa_heatmaps_{stim}.png",
        )

if __name__ == "__main__":
    create_RDM_plots()