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

Sizes, fonts and line widths come from figure_params.STYLES ("paper" / "poster"),
like the other final-figure scripts; output goes to
figure_params.figure_output_dir("cca_null_histograms", style_name) and can be written as
png and/or svg.

Kept as a separate file for research-archiving purposes.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

from analysis_class import FiringRateAnalysis
import figure_params as fp

FIG = "cca_null_histograms"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESPONSE_RANGE_ORDER = ["onset", "sustained", "offset"]
STIM_TYPES = ["AM", "pureTones", "naturalSound"]

# Histogram appearance
NULL_COLOR = "#AAAAAA"       # gray fill for null histogram
ACTUAL_LINE_COLOR = "#C44E52"  # red for the vertical actual-value line(s)
N_HIST_BINS = 30

# Subspace-alignment inset: outside +/-INSET_Z_THRESHOLD the two region ellipses are drawn
# perpendicular, inside it they are drawn strongly overlapping. For the combined figures
# the deciding value is set by INSET_RULE: "median" (median of the sessions' actual z) or
# "any" (any session's actual z lies outside the threshold).
INSET_Z_THRESHOLD = 2.0
INSET_RULE = "median"
INSET_OVERLAP_ANGLE = 10      # degrees between the ellipses' long axes when "aligned"


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
def _short_pair(target_pair: str) -> str:
    """'Pri-Ventral auditory area vs Pri-Dorsal auditory area' -> 'Pri-Ventral vs Pri-Dorsal'."""
    return target_pair.replace(" auditory area", "")


def _setup_histogram_ax(ax, z_null: np.ndarray, title: str, style: dict):
    """Draw the z-scored null histogram and the z = 0 reference line on `ax`."""
    ax.hist(z_null, bins=N_HIST_BINS, color=NULL_COLOR, edgecolor="white",
            alpha=0.85, density=True, label="Null distribution")
    ax.axvline(0, color="black", linestyle="--", linewidth=style["line_width"],
               alpha=0.6, label="z = 0")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.set_title(title)


def _legend_outside(ax, handles=None):
    """Legend beside the axes so it never covers the histogram or the actual-value lines."""
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)


def _pair_regions(target_pair: str):
    """[(short name, color), ...] for the two regions in 'Pri-X auditory area vs Pri-Y auditory area'."""
    out = []
    for part in target_pair.split(" vs "):
        name = part.replace("Pri-", "", 1)
        out.append((fp.REGION_SHORT.get(name, name.replace(" auditory area", "")),
                    fp.REGION_COLORS.get(name, "0.5")))
    return out


def _add_alignment_inset(ax, target_pair: str, orthogonal: bool, style: dict):
    """
    Cartoon of the two regions' subspaces in the top-left corner of the histogram axes:
    perpendicular ellipses when the result is beyond +/-INSET_Z_THRESHOLD, strongly
    overlapping ones otherwise. Colored by region; no text.
    """
    regions = _pair_regions(target_pair)
    inset = ax.inset_axes([0.02, 0.60, 0.30, 0.38])
    inset.set_aspect("equal", adjustable="box")
    inset.set_anchor("NW")
    angles = (0, 90) if orthogonal else (0, INSET_OVERLAP_ANGLE)
    for (_, color), angle in zip(regions, angles):
        inset.add_patch(Ellipse((0, 0), width=2.0, height=0.8, angle=angle, facecolor=color,
                                edgecolor=color, alpha=0.55, linewidth=style["line_width"]))
    inset.set_xlim(-1.1, 1.1)
    inset.set_ylim(-1.1, 1.1)
    inset.axis("off")


def _is_orthogonal(z_values) -> bool:
    z = np.asarray(z_values, dtype=float)
    if INSET_RULE == "any":
        return bool(np.any(np.abs(z) > INSET_Z_THRESHOLD))
    return bool(abs(np.median(z)) > INSET_Z_THRESHOLD)


# ---------------------------------------------------------------------------
# Approach 1 — per-session figures
# ---------------------------------------------------------------------------
def make_per_session_figures(df: pd.DataFrame, out_root: str, style_name="paper",
                             formats=fp.DEFAULT_FORMATS):
    """
    One figure per (session, target_pair, stimulus, response_range).
    Output: out_root/per_session/{stimulus}/{response_range}/{target_pair}/{session}.<fmt>
    """
    style = fp.STYLES[style_name]
    with plt.rc_context(fp.rc_params(style)):
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

                        fig, ax = plt.subplots(figsize=style["panel_figsize"]["null_hist_session"],
                                               layout="constrained")
                        title = (f"{_short_pair(target_pair)}\n"
                                 f"{fp.STIM_LABELS.get(stim, stim)}, {response_range}\n"
                                 f"session {session}")
                        _setup_histogram_ax(ax, z_null, title, style)
                        ax.axvline(z_actual, color=ACTUAL_LINE_COLOR,
                                   linewidth=style["line_width"] * 1.8, linestyle="-",
                                   label=f"Actual (z = {z_actual:.2f})")
                        _legend_outside(ax)
                        _add_alignment_inset(ax, target_pair, _is_orthogonal([z_actual]), style)

                        # Sanitise target_pair for use as a directory name.
                        pair_dir = target_pair.replace(" ", "_").replace("/", "-")
                        fp.save_figure(fig, f"{out_root}/per_session/{stim}/{response_range}/{pair_dir}",
                                       str(session), style, formats)


# ---------------------------------------------------------------------------
# Approach 2 — combined (pooled null, all sessions' actual lines)
# ---------------------------------------------------------------------------
def make_combined_figures(df: pd.DataFrame, out_root: str, style_name="paper",
                          formats=fp.DEFAULT_FORMATS):
    """
    One figure per (target_pair, stimulus, response_range).
    Pools z-scored nulls across all sessions (each row z-scored against its
    own null), then draws one vertical line per session for the actual value.
    Output: out_root/combined/{stimulus}/{response_range}/{target_pair}.<fmt>
    """
    style = fp.STYLES[style_name]
    with plt.rc_context(fp.rc_params(style)):
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

                    fig, ax = plt.subplots(figsize=style["panel_figsize"]["null_hist"],
                                           layout="constrained")
                    n_sessions = len(sessions_used)
                    title = (f"{_short_pair(target_pair)}\n"
                             f"{fp.STIM_LABELS.get(stim, stim)}, {response_range}")
                    _setup_histogram_ax(ax, all_z_null, title, style)

                    # One vertical line per session; all the same color since we
                    # are interested in spread relative to the null, not per-session
                    # differences.
                    for z_actual in actual_z_values:
                        ax.axvline(z_actual, color=ACTUAL_LINE_COLOR,
                                   linewidth=style["line_width"] * 1.4, linestyle="-", alpha=0.7)

                    # Summarise the actual values (median + IQR) in the legend.
                    median_z = np.median(actual_z_values)
                    q25, q75 = np.percentile(actual_z_values, [25, 75])
                    actual_handle = plt.Line2D(
                        [0], [0], color=ACTUAL_LINE_COLOR, linewidth=style["line_width"] * 1.8,
                        label=f"Actual values (n = {n_sessions})\n"
                              f"median z = {median_z:.2f}\n"
                              f"[{q25:.2f}, {q75:.2f}]")
                    null_handle = plt.Line2D(
                        [0], [0], color=NULL_COLOR, linewidth=8, alpha=0.85,
                        label="Pooled null\ndistribution")
                    zero_handle = plt.Line2D(
                        [0], [0], color="black", linestyle="--", linewidth=style["line_width"],
                        label="z = 0")
                    _legend_outside(ax, [null_handle, zero_handle, actual_handle])
                    _add_alignment_inset(ax, target_pair, _is_orthogonal(actual_z_values), style)

                    pair_dir = target_pair.replace(" ", "_").replace("/", "-")
                    fp.save_figure(fig, f"{out_root}/combined/{stim}/{response_range}",
                                   pair_dir, style, formats)


def main(style_name="paper", formats=fp.DEFAULT_FORMATS, steps=("per_session", "combined"),
         df=None, out_root=None):
    if df is None:
        fr_db = FiringRateAnalysis(db_suffix="coords_updated")
        df = load_data(fr_db.figdata_path)
    out_root = out_root or fp.figure_output_dir(FIG, style_name)
    os.makedirs(out_root, exist_ok=True)

    if "per_session" in steps:
        print("Generating per-session histogram figures (Approach 1)...")
        make_per_session_figures(df, out_root, style_name, formats)
    if "combined" in steps:
        print("Generating combined histogram figures (Approach 2)...")
        make_combined_figures(df, out_root, style_name, formats)
    print("Done.")


if __name__ == "__main__":
    # main(style_name="paper", formats=("png", "svg"))
    main(style_name="poster", formats=("svg",), steps=("combined",))
