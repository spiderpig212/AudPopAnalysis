"""
Null-distribution histogram plots for CCA cross-region projection similarity.

For each (brain region pair, stimulus type, response range) condition we plot:

  Approach 1 — Per-session:
      One figure per (session, target_pair, stimulus, response_range). A single
      histogram of that session's z-scored null similarities, with one vertical
      line showing where the actual z_score_diff_from_null falls.

  Approach 2 — Combined:
      One figure per (target_pair, stimulus, response_range). A single pooled
      histogram of ALL sessions' z-scored null similarities (each row z-scored
      against its own null distribution), with one vertical line per session
      showing the actual z_score_diff_from_null value (all lines the same color
      since the focus is spread relative to the null).

In both approaches the x-axis is z-score space. The null similarities are
z-scored per row (each session/pair/stimulus/response_range has its own null
distribution) before pooling or plotting.

Kept as a separate file for research-archiving purposes.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from analysis_class import FiringRateAnalysis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESPONSE_RANGE_ORDER = ["onset", "sustained", "offset"]
STIM_TYPES = ["AM", "pureTones", "naturalSound"]

# Histogram appearance
NULL_COLOR = "#AAAAAA"       # gray fill for null histogram
ACTUAL_LINE_COLOR = "#C44E52"  # red for the vertical actual-value line(s)
N_HIST_BINS = 30


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_feather(
        f"{file_path}/CCA_cross_region_projection_similarity.feather")

    # Drop Temporal association areas.
    for col in ("target1", "target2"):
        df = df[~df[col].isin(["Temporal association areas"])]

    # Add in Pri- to each of the target regions
    df["target1"] = df["target1"].apply(lambda x: "Pri-" + x)
    df["target2"] = df["target2"].apply(lambda x: "Pri-" + x)

    df["target_pair"] = df["target1"] + " vs " + df["target2"]
    return df


def zscore_null_row(null_list: list) -> np.ndarray:
    """
    Z-score a single row's null similarities against their own distribution.
    Returns a 1D array of z-scored null values.
    If the null distribution has zero std, returns an array of zeros.
    """
    arr = np.asarray(null_list, dtype=float)
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def parse_null_similarities(val):
    """
    Safely parse the null_similarities column value into a Python list/array.
    Handles the case where values may be stored as strings.
    """
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if isinstance(val, str):
        # Stored as a string representation of a list.
        import ast
        return ast.literal_eval(val)
    return []


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------
def _setup_histogram_ax(ax, z_null: np.ndarray, title: str, metric_label: str):
    """
    Draw the z-scored null histogram on `ax` and return the computed x-limits
    so vertical lines can be clipped sensibly.
    """
    ax.hist(z_null, bins=N_HIST_BINS, color=NULL_COLOR, edgecolor="white",
            alpha=0.85, density=True, label="Null distribution")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
               label="z = 0")
    ax.set_xlabel("Z-score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=9)
    return ax.get_xlim()


def _save_figure(fig, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Approach 1 — per-session figures
# ---------------------------------------------------------------------------
def make_per_session_figures(df: pd.DataFrame, out_root: str):
    """
    One figure per (session, target_pair, stimulus, response_range).
    Output: out_root/per_session/{stimulus}/{response_range}/{target_pair}/{session}.png
    """
    for stim in STIM_TYPES:
        stim_df = df[df["stimulus"] == stim]
        if stim_df.empty:
            continue

        for response_range in RESPONSE_RANGE_ORDER:
            rr_df = stim_df[stim_df["response_range"] == response_range]
            if rr_df.empty:
                continue

            for target_pair in sorted(rr_df["target_pair"].unique()):
                tp_df = rr_df[rr_df["target_pair"] == target_pair]

                for _, row in tp_df.iterrows():
                    session = row["session"]

                    # Parse and z-score this row's null distribution.
                    raw_null = parse_null_similarities(row["null_similarities"])
                    if len(raw_null) < 3:
                        print(f"Skipping {session}/{target_pair}/{stim}/"
                              f"{response_range}: too few null values.")
                        continue
                    z_null = zscore_null_row(raw_null)

                    # Actual value is already in z-score space.
                    z_actual = float(row["z_score_diff_from_null"])

                    fig, ax = plt.subplots(figsize=(5, 4))
                    title = (f"{target_pair}\n"
                             f"{stim} | {response_range} | session {session}")
                    _setup_histogram_ax(ax, z_null, title,
                                        metric_label="Z-score")

                    ax.axvline(z_actual, color=ACTUAL_LINE_COLOR,
                               linewidth=1.8, linestyle="-",
                               label=f"Actual (z = {z_actual:.2f})")
                    ax.legend(fontsize=8, loc="upper left")
                    fig.tight_layout()

                    # Sanitise target_pair for use as a directory name.
                    pair_dir = target_pair.replace(" ", "_").replace("/", "-")
                    out_path = (
                        f"{out_root}/per_session/{stim}/{response_range}/"
                        f"{pair_dir}/{session}.png"
                    )
                    _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Approach 2 — combined (pooled null, all sessions' actual lines)
# ---------------------------------------------------------------------------
def make_combined_figures(df: pd.DataFrame, out_root: str):
    """
    One figure per (target_pair, stimulus, response_range).
    Pools z-scored nulls across all sessions (each row z-scored against its
    own null), then draws one vertical line per session for the actual value.
    Output: out_root/combined/{stimulus}/{response_range}/{target_pair}.png
    """
    for stim in STIM_TYPES:
        stim_df = df[df["stimulus"] == stim]
        if stim_df.empty:
            continue

        for response_range in RESPONSE_RANGE_ORDER:
            rr_df = stim_df[stim_df["response_range"] == response_range]
            if rr_df.empty:
                continue

            for target_pair in sorted(rr_df["target_pair"].unique()):
                tp_df = rr_df[rr_df["target_pair"] == target_pair]

                # Pool z-scored nulls and collect actual values across sessions.
                pooled_z_null = []
                actual_z_values = []  # one per session
                sessions_used = []

                for _, row in tp_df.iterrows():
                    raw_null = parse_null_similarities(row["null_similarities"])
                    if len(raw_null) < 3:
                        print(f"Skipping session {row['session']} in combined "
                              f"plot ({target_pair}/{stim}/{response_range}): "
                              f"too few null values.")
                        continue
                    # Z-score each row against its own null distribution so
                    # values are comparable across sessions.
                    z_null = zscore_null_row(raw_null)
                    pooled_z_null.append(z_null)
                    actual_z_values.append(float(row["z_score_diff_from_null"]))
                    sessions_used.append(row["session"])

                if len(pooled_z_null) == 0:
                    continue

                # Single pooled null histogram.
                all_z_null = np.concatenate(pooled_z_null)

                fig, ax = plt.subplots(figsize=(6, 4))
                n_sessions = len(sessions_used)
                title = (f"{target_pair}\n"
                         f"{stim} | {response_range} | "
                         f"n = {n_sessions} session(s)")
                _setup_histogram_ax(ax, all_z_null, title,
                                    metric_label="Z-score")

                # One vertical line per session; all the same color since we
                # are interested in spread relative to the null, not per-session
                # differences.
                for z_actual in actual_z_values:
                    ax.axvline(z_actual, color=ACTUAL_LINE_COLOR,
                               linewidth=1.4, linestyle="-", alpha=0.7)

                # Summarise the actual values (median + IQR) in the legend.
                median_z = np.median(actual_z_values)
                q25, q75 = np.percentile(actual_z_values, [25, 75])
                actual_handle = plt.Line2D(
                    [0], [0], color=ACTUAL_LINE_COLOR, linewidth=1.8,
                    label=f"Actual values (n={n_sessions})\n"
                          f"median z = {median_z:.2f} "
                          f"[{q25:.2f}, {q75:.2f}]")
                null_handle = plt.Line2D(
                    [0], [0], color=NULL_COLOR, linewidth=8, alpha=0.85,
                    label="Pooled null distribution")
                zero_handle = plt.Line2D(
                    [0], [0], color="black", linestyle="--", linewidth=1.0,
                    label="z = 0")
                ax.legend(handles=[null_handle, zero_handle, actual_handle],
                          fontsize=8, loc="upper left")
                fig.tight_layout()

                pair_dir = target_pair.replace(" ", "_").replace("/", "-")
                out_path = (
                    f"{out_root}/combined/{stim}/{response_range}/"
                    f"{pair_dir}.png"
                )
                _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path

    df = load_data(file_path)

    out_root = f"{file_path}/CCA_two_region_analysis/null_histograms"
    os.makedirs(out_root, exist_ok=True)

    print("Generating per-session histogram figures (Approach 1)...")
    make_per_session_figures(df, out_root)

    print("Generating combined histogram figures (Approach 2)...")
    make_combined_figures(df, out_root)

    print("Done.")


if __name__ == "__main__":
    main()