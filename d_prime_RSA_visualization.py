"""
Visualization for the pairwise-decoder RSA analysis (dprime_RSA.py).

Generates 3x3 panel figures where:
  - columns = response range (onset, sustained, offset)
  - rows    = brain region (raw) or CCA target region (Primary -> target)
  - each panel = session-averaged RSA matrix heatmap

Produces one "raw" figure and one "cca" figure per stimulus type.
"""
import matplotlib
matplotlib.use("Agg")

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_class import FiringRateAnalysis

response_ranges = ["onset", "sustained", "offset"]

# Row ordering. For "raw", rows are the decoded brain region. For "cca", rows
# are the target region (source is always Primary auditory area).
raw_rows = ["Primary auditory area", "Ventral auditory area", "Dorsal auditory area"]
cca_target_rows = ["Ventral auditory area", "Dorsal auditory area"]


def load_rsa_data(out_dir: str, stimulus: str) -> list:
    """Load and concatenate all response-range pickles for a stimulus type."""
    records = []
    for response_range in response_ranges:
        pkl = f"{out_dir}/dprime_rsa_{stimulus}_{response_range}.pkl"
        if not os.path.exists(pkl):
            print(f"Missing {pkl}, skipping.")
            continue
        records.extend(pickle.load(open(pkl, "rb")))
    return records


def average_over_sessions(records, data_source, row_value, response_range,
                          row_field, metric_key):
    """
    Average the RSA matrices over sessions for a single (row, column) panel.

    row_field: "brain_region" for raw, "target_region" for cca.
    Returns the mean matrix (or None if no matching records).
    """
    mats = [
        r[metric_key]
        for r in records
        if r["data_source"] == data_source
        and r["response_range"] == response_range
        and r[row_field] == row_value
        and r[metric_key] is not None
    ]
    if len(mats) == 0:
        return None
    return np.mean(mats, axis=0)


def make_figure(records, stimulus, data_source, uniq_stims, out_dir,
                metric_key="J_matrix"):
    """Build and save one 3x3 figure for a given stimulus type and data source."""
    if data_source == "raw":
        rows = raw_rows
        row_field = "brain_region"
        row_label = lambda r: r
    else:
        rows = cca_target_rows
        row_field = "target_region"
        row_label = lambda r: f"Primary -> {r.replace(' auditory area', '')}"

    # Determine a shared color scale across all panels for comparability.
    all_vals = []
    panels = {}  # (row, col) -> matrix
    for row_value in rows:
        for response_range in response_ranges:
            mat = average_over_sessions(
                records, data_source, row_value, response_range,
                row_field, metric_key)
            panels[(row_value, response_range)] = mat
            if mat is not None:
                all_vals.append(mat)

    if len(all_vals) == 0:
        print(f"No data for {stimulus} / {data_source}, skipping figure.")
        return

    stacked = np.concatenate([m.ravel() for m in all_vals])
    vmin, vmax = np.nanmin(stacked), np.nanmax(stacked)

    n_rows = len(rows)
    n_cols = len(response_ranges)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)

    tick_labels = [str(s) for s in uniq_stims]

    im = None
    for i, row_value in enumerate(rows):
        for j, response_range in enumerate(response_ranges):
            ax = axes[i][j]
            mat = panels[(row_value, response_range)]
            if mat is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis",
                               aspect="auto")
                ax.set_xticks(range(len(tick_labels)))
                ax.set_yticks(range(len(tick_labels)))
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
                ax.set_yticklabels(tick_labels, fontsize=6)

            if i == 0:
                ax.set_title(response_range, fontsize=12)
            if j == 0:
                ax.set_ylabel(row_label(row_value), fontsize=10)

    metric_name = "Fisher's J" if metric_key == "J_matrix" else "d-prime"
    fig.suptitle(f"{stimulus} — {data_source.upper()} RSA ({metric_name})",
                 fontsize=14)

    if im is not None:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=metric_name)

    fig.subplots_adjust(left=0.08, top=0.92, hspace=0.3, wspace=0.3)

    metric_tag = "J" if metric_key == "J_matrix" else "dprime"
    out_path = f"{out_dir}/figures/dprime_rsa_{stimulus}_{data_source}_{metric_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main(metric_key: str = "J_matrix"):
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path
    stim_types = fr_db.stim_types
    out_dir = f"{file_path}/dprime_RSA"

    for stimulus in stim_types:
        records = load_rsa_data(out_dir, stimulus)
        if len(records) == 0:
            continue

        # Stimulus axis labels: recover the stimulus set for this type. Matrices
        # were built over np.unique(stimArray) in dprime_RSA.py.
        stim_arrays = fr_db.return_arrays(stimulus)
        uniq_stims = np.unique(stim_arrays["stimArray"][0, :])

        make_figure(records, stimulus, "raw", uniq_stims, out_dir, metric_key)
        make_figure(records, stimulus, "cca", uniq_stims, out_dir, metric_key)


if __name__ == "__main__":
    # Pass metric_key="dprime_matrix" to plot classic d-prime instead of J.
    main(metric_key="J_matrix")
    # main(metric_key="dprime_matrix")