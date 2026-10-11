"""
Curated poster/paper figures for the pairwise-decoder RSA pipeline.

Unlike d_prime_RSA_visualization.py (the exhaustive/exploratory figure catalog),
this script produces the small, curated set of figures intended for the poster
and paper. Main figures are restricted to the "sustained" response range;
onset/offset versions of the same figures are supplementary and are produced by
calling the same functions with a different response_range.

Two figure families, each split into "raw" (brain regions) and "cca" (Primary ->
target region-pairs) versions:

  - Natural sounds: within-source vs between-source separability.
    X-axis: "within", "between". Bars grouped/colored by region or region-pair.
  - Synthetic sounds: separability by stimulus type, near-diagonal (adjacent-
    stimulus) pairs only. X-axis: "AM", "Pure Tones". Bars grouped/colored by
    region or region-pair.

Every box plot point is one session: that session's own matrix, averaged over
the entries selected for the figure (within-source / between-source entries for
natural sounds, near-diagonal entries for synthetic sounds). So each box holds
at most n_sessions points, not pooled entries or entries of a cross-session-
averaged matrix.

The y-axis is labeled D' throughout, though the plotted values are Fisher's J
(a generic separability label was chosen since the method isn't the classic
d-prime calculation).

Statistics: pairwise Mann-Whitney U between groups (regions / region-pairs)
WITHIN each x-tick only. Comparisons across x-ticks (e.g. within vs between, or
AM vs Pure Tones, for the same region) are not tested here.

Figure sizing/text is controlled entirely by POSTER_STYLE / PAPER_STYLE below --
edit those two dicts to retarget every figure at once.
"""
import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from analysis_class import FiringRateAnalysis
from d_prime_RSA_visualization import (
    load_rsa_data,
    extract_triangle_values,
    split_within_between,
    raw_rows,
    cca_target_rows,
    GROUP_FILL_COLORS,
    STRIP_ALPHA_THRESHOLD,
    _p_to_stars,
)

MAIN_RESPONSE_RANGE = "sustained"

# Output file formats; any extension matplotlib supports ("png", "svg", "pdf").
DEFAULT_FORMATS = ("png",)

# Keep text as real text objects in SVGs (editable in Illustrator/Inkscape)
# instead of converting glyphs to paths. Requires the font to be installed on
# the machine that opens the SVG.
plt.rcParams["svg.fonttype"] = "none"

# Maps the RSA matrix field name to (tag used in output filenames, y-axis
# label). "J_matrix" and "dprime_matrix" are both separability metrics
# computed from the decoder's projection axis; "accuracy_matrix" is the
# binary decoder's own cross-validated classification accuracy.
METRIC_INFO = {
    "J_matrix": ("J", "D′"),
    "dprime_matrix": ("dprime", "D′"),
    "accuracy_matrix": ("accuracy", "Decoding accuracy"),
}

# ----------------------------------------------------------------------------
# Style presets -- the single place to change figure size / text size.
# ----------------------------------------------------------------------------
POSTER_STYLE = dict(
    figsize=(9, 8),
    title_fontsize=22,
    label_fontsize=20,
    tick_fontsize=16,
    legend_fontsize=15,
    star_fontsize=16,
    box_width=0.6,
    dpi=300,
)

PAPER_STYLE = dict(
    figsize=(4.5, 4),
    title_fontsize=11,
    label_fontsize=10,
    tick_fontsize=9,
    legend_fontsize=8,
    star_fontsize=9,
    box_width=0.6,
    dpi=300,
)

STYLES = {"poster": POSTER_STYLE, "paper": PAPER_STYLE}


def _region_groups(data_source):
    """
    Return (groups, labels, colors) for either raw regions or cca region-pairs.

    Colors are index-matched to raw_rows so a region keeps the same color
    whether it's shown as a raw region or as a CCA target (e.g. Ventral is
    always orange, Dorsal is always green).
    """
    if data_source == "raw":
        rows = raw_rows
        row_field = "brain_region"
        labels = list(rows)
    elif data_source == "cca":
        rows = cca_target_rows
        row_field = "target_region"
        labels = [f"Primary → {r.replace(' auditory area', '')}" for r in rows]
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    colors = [GROUP_FILL_COLORS[raw_rows.index(r) % len(GROUP_FILL_COLORS)]
              for r in rows]
    groups = [(data_source, r, row_field) for r in rows]
    return groups, labels, colors


# ----------------------------------------------------------------------------
# Per-session data collection: one point per session (that session's own
# matrix, averaged over the selected entries), rather than pooling entries
# across sessions or averaging matrices across sessions first.
# ----------------------------------------------------------------------------
def collect_session_averaged_values(records, data_source, row_value, row_field,
                                    response_range, metric_key, triangle):
    """One point per session: mean of that session's own selected triangle entries."""
    vals = []
    for r in records:
        if (r["data_source"] == data_source
                and r["response_range"] == response_range
                and r[row_field] == row_value
                and r[metric_key] is not None):
            tri = extract_triangle_values(r[metric_key], triangle)
            if tri.size == 0:
                continue
            vals.append(np.mean(tri))
    return np.array(vals)


def collect_session_averaged_within_between(records, data_source, row_value,
                                            row_field, response_range, metric_key,
                                            exemplars_per_source=4):
    """
    One point per session for each of within/between: mean of that session's own
    within-source entries, and separately the mean of its between-source entries.
    Returns (within_vals, between_vals), each up to length n_sessions.
    """
    within_vals, between_vals = [], []
    for r in records:
        if (r["data_source"] == data_source
                and r["response_range"] == response_range
                and r[row_field] == row_value
                and r[metric_key] is not None):
            w, b = split_within_between(r[metric_key], exemplars_per_source)
            if w.size:
                within_vals.append(np.mean(w))
            if b.size:
                between_vals.append(np.mean(b))
    return np.array(within_vals), np.array(between_vals)


def _draw_grouped_box_strip(tick_labels, values_by_tick, group_labels, colors,
                            style, ylabel, response_range):
    """
    Draw a box+strip figure with `len(tick_labels)` x-ticks, each holding one
    box per group (color-coded). Runs pairwise Mann-Whitney U between groups
    WITHIN each tick only (no comparisons across ticks).

    values_by_tick: dict tick_label -> list of 1D arrays, aligned with
        group_labels / colors.
    Returns (fig, ax, stats_rows).
    """
    n_groups = len(group_labels)
    within_gap = 0.8
    tick_gap = 1.2
    block_width = n_groups * within_gap
    box_width = style["box_width"]

    fig, ax = plt.subplots(figsize=style["figsize"])
    xticks, xticklabels = [], []
    stats_rows = []

    for t_idx, tick_label in enumerate(tick_labels):
        block_start = t_idx * (block_width + tick_gap)
        positions = [block_start + g * within_gap for g in range(n_groups)]
        xticks.append(np.mean(positions))
        xticklabels.append(tick_label)

        group_values = values_by_tick[tick_label]

        box_data = [gv if gv.size > 0 else np.array([np.nan]) for gv in group_values]
        bp = ax.boxplot(box_data, positions=positions, widths=box_width,
                        showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")

        for pos, gv, color in zip(positions, group_values, colors):
            if gv.size == 0:
                continue
            alpha = 0.0 if gv.size > STRIP_ALPHA_THRESHOLD else 0.4
            if alpha > 0:
                jitter = (np.random.rand(gv.size) - 0.5) * (box_width * 0.6)
                ax.scatter(np.full(gv.size, pos) + jitter, gv,
                          s=10, alpha=alpha, color="black", zorder=3)

        # Pairwise Mann-Whitney U between groups at THIS tick only.
        finite = [gv for gv in group_values if gv.size > 0]
        y_max = np.nanmax([np.nanmax(gv) for gv in finite] or [0.0])
        y_min = np.nanmin([np.nanmin(gv) for gv in finite] or [0.0])
        span = (y_max - y_min) if (y_max > y_min) else 1.0
        step = span * 0.08
        level = 0
        for a, b in itertools.combinations(range(n_groups), 2):
            gv_a, gv_b = group_values[a], group_values[b]
            if gv_a.size < 2 or gv_b.size < 2:
                continue
            try:
                u_stat, p_val = stats.mannwhitneyu(gv_a, gv_b, alternative="two-sided")
            except ValueError:
                continue
            stats_rows.append({
                "tick": tick_label,
                "response_range": response_range,
                "group_a": group_labels[a],
                "group_b": group_labels[b],
                "n_a": gv_a.size,
                "n_b": gv_b.size,
                "U": u_stat,
                "p_value": p_val,
            })
            if p_val < 0.05:
                y = y_max + step * (level + 1)
                x1, x2 = positions[a], positions[b]
                ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                        lw=1.0, color="black")
                ax.text((x1 + x2) / 2, y + step * 0.3, _p_to_stars(p_val),
                        ha="center", va="bottom", fontsize=style["star_fontsize"])
                level += 1

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=style["tick_fontsize"])
    ax.tick_params(axis="y", labelsize=style["tick_fontsize"])
    ax.set_ylabel(ylabel, fontsize=style["label_fontsize"])

    # Headroom above the highest bracket/star so annotations never sit flush
    # against the frame (brackets are added after autoscale already ran once).
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(y_bottom, y_top + 0.06 * (y_top - y_bottom))

    # Legend lives outside the axes (right side) so it can never collide with
    # stacked significance brackets, however tall they get.
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                          color=c, label=lbl)
               for c, lbl in zip(colors, group_labels)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=style["legend_fontsize"], frameon=False)

    return fig, ax, stats_rows


def _save_figure(fig, stats_rows, out_dir, base_name, style, formats):
    os.makedirs(out_dir, exist_ok=True)
    base = f"{out_dir}/{base_name}"
    for fmt in formats:
        # dpi only affects raster formats; svg is vector.
        fig.savefig(f"{base}.{fmt}", dpi=style["dpi"], bbox_inches="tight")
        print(f"Saved figure to {base}.{fmt}")
    plt.close(fig)
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(f"{base}_stats.csv", index=False)


# ----------------------------------------------------------------------------
# Natural sounds: within-source vs between-source
# ----------------------------------------------------------------------------
def make_natural_sounds_figure(records, data_source, out_dir, style_name="paper",
                               response_range=MAIN_RESPONSE_RANGE,
                               metric_key="J_matrix", exemplars_per_source=4,
                               formats=DEFAULT_FORMATS):
    style = STYLES[style_name]
    groups, labels, colors = _region_groups(data_source)

    tick_labels = ["within", "between"]
    values_by_tick = {t: [] for t in tick_labels}
    for (ds, row_value, row_field) in groups:
        w, b = collect_session_averaged_within_between(
            records, ds, row_value, row_field, response_range, metric_key,
            exemplars_per_source)
        values_by_tick["within"].append(w)
        values_by_tick["between"].append(b)

    metric_tag, ylabel = METRIC_INFO[metric_key]
    fig, ax, stats_rows = _draw_grouped_box_strip(
        tick_labels, values_by_tick, labels, colors, style, ylabel,
        response_range)

    ax.set_xlabel("Source pairing", fontsize=style["label_fontsize"])
    ax.set_title(f"Natural sounds — {data_source.upper()} — {response_range}",
                fontsize=style["title_fontsize"])

    base_name = f"naturalSound_within_between_{data_source}_{response_range}_{style_name}_{metric_tag}"
    _save_figure(fig, stats_rows, out_dir, base_name, style, formats)


# ----------------------------------------------------------------------------
# Synthetic sounds (AM, pure tones): separability by stimulus type
# ----------------------------------------------------------------------------
def make_synthetic_sounds_figure(records_by_stim, data_source, out_dir,
                                 style_name="paper",
                                 response_range=MAIN_RESPONSE_RANGE,
                                 metric_key="J_matrix", triangle="neardiag",
                                 formats=DEFAULT_FORMATS):
    """
    records_by_stim: dict {"AM": <records>, "pureTones": <records>}, each the
        result of load_rsa_data(out_dir, stimulus) for that stimulus type.

    triangle="neardiag" (default): only adjacent-stimulus pairs. Each box point
    is one session's mean over its own near-diagonal entries (one point per
    session), not pooled entries or a cross-session-averaged matrix.
    """
    style = STYLES[style_name]
    groups, labels, colors = _region_groups(data_source)

    tick_labels = ["AM", "Pure Tones"]
    stim_keys = {"AM": "AM", "Pure Tones": "pureTones"}

    values_by_tick = {t: [] for t in tick_labels}
    for tick_label in tick_labels:
        stim_records = records_by_stim[stim_keys[tick_label]]
        for (ds, row_value, row_field) in groups:
            gv = collect_session_averaged_values(
                stim_records, ds, row_value, row_field, response_range,
                metric_key, triangle)
            values_by_tick[tick_label].append(gv)

    metric_tag, ylabel = METRIC_INFO[metric_key]
    fig, ax, stats_rows = _draw_grouped_box_strip(
        tick_labels, values_by_tick, labels, colors, style, ylabel,
        response_range)

    ax.set_xlabel("Stimulus type", fontsize=style["label_fontsize"])
    ax.set_title(f"Synthetic sounds — {data_source.upper()} — {response_range}",
                fontsize=style["title_fontsize"])

    base_name = f"synthetic_{data_source}_{response_range}_{triangle}_{style_name}_{metric_tag}"
    _save_figure(fig, stats_rows, out_dir, base_name, style, formats)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(style_name="paper", response_ranges=(MAIN_RESPONSE_RANGE,),
        metric_key="J_matrix", formats=DEFAULT_FORMATS):
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path
    rsa_dir = f"{file_path}/dprime_RSA"
    out_dir = f"{rsa_dir}/figures/final/{style_name}"

    records_natural = load_rsa_data(rsa_dir, "naturalSound")
    records_by_stim = {
        "AM": load_rsa_data(rsa_dir, "AM"),
        "pureTones": load_rsa_data(rsa_dir, "pureTones"),
    }

    for response_range in response_ranges:
        for data_source in ("raw", "cca"):
            if records_natural:
                make_natural_sounds_figure(
                    records_natural, data_source, out_dir, style_name=style_name,
                    response_range=response_range, metric_key=metric_key,
                    formats=formats)
            else:
                print("No naturalSound records found, skipping.")

            if records_by_stim["AM"] and records_by_stim["pureTones"]:
                make_synthetic_sounds_figure(
                    records_by_stim, data_source, out_dir, style_name=style_name,
                    response_range=response_range, metric_key=metric_key,
                    formats=formats)
            else:
                print("Missing AM or pureTones records, skipping synthetic figure.")


if __name__ == "__main__":
    # Main poster/paper figures: sustained response range only.
    # main(style_name="paper", response_ranges=("sustained",), metric_key="J_matrix")
    main(style_name="paper", response_ranges=("sustained",), metric_key="accuracy_matrix",
         formats=("png", "svg"))
    # main(style_name="poster", response_ranges=("sustained",), formats=("svg",))

    # Supplementary: other response ranges, same figures.
    # main(style_name="paper", response_ranges=("onset", "offset"))
