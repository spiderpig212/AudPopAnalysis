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
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
    out_path = f"{out_dir}/dprime_rsa_{stimulus}_{data_source}_{metric_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")

# ----------------------------------------------------------------------------
# Distribution (box + strip) visualizations
# ----------------------------------------------------------------------------
# NOTE on statistical independence: entries within a single RSA matrix share the
# same neurons / decoder axes and overlapping stimuli, so the per-entry points
# are NOT independent. The per-entry ("entry") stats are therefore exploratory
# and likely anti-conservative.

STRIP_ALPHA_THRESHOLD = 100  # Above this many points per group, hide the strip.

# Fill colors for the boxplots, applied per group in order. Boxes are the color
# key for which brain region / region-pair a group is; strip points stay gray.
GROUP_FILL_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
]


# Star annotation thresholds for Mann-Whitney p-values.
def _p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def extract_triangle_values(mat, triangle="upper"):
    """
    Return the selected entries of a symmetric RSA matrix as a 1D array.

    triangle="upper":    all entries strictly above the main diagonal.
    triangle="neardiag": only the first superdiagonal (entries [i, i+1]),
                         i.e. adjacent-stimulus pairs.
    """
    if mat is None:
        return np.array([])
    n = mat.shape[0]
    if triangle == "upper":
        i, j = np.triu_indices(n, k=1)
        return mat[i, j]
    elif triangle == "neardiag":
        idx = np.arange(n - 1)
        return mat[idx, idx + 1]
    else:
        raise ValueError(f"Unknown triangle: {triangle}")


def collect_group_values(records, data_source, row_value, row_field,
                         response_range, metric_key, triangle, point_level):
    """
    Collect the plotted values for one (group, response_range) cell.

    point_level="entry":        every selected triangle entry from every session
                                (per-session per-entry; entries are not
                                independent -- exploratory).
    point_level="session_mean": FIRST average the RSA matrices across sessions
                                (as in the heatmaps -> one mean matrix per group /
                                response range / stimulus), THEN take the selected
                                triangle entries of that averaged matrix. Each
                                point is one cell of the session-averaged heatmap.
    Returns a 1D array of values.
    """
    if point_level == "session_mean":
        # Session-averaged matrix, identical to what the heatmaps display.
        mean_mat = average_over_sessions(
            records, data_source, row_value, response_range,
            row_field, metric_key)
        if mean_mat is None:
            return np.array([])
        return extract_triangle_values(mean_mat, triangle)

    if point_level != "entry":
        raise ValueError(f"Unknown point_level: {point_level}")

    # Per-session per-entry: pool every selected triangle entry from every
    # session. Entries share neurons/decoder axes, so they are NOT independent.
    per_session = []
    for r in records:
        if (r["data_source"] == data_source
                and r["response_range"] == response_range
                and r[row_field] == row_value
                and r[metric_key] is not None):
            vals = extract_triangle_values(r[metric_key], triangle)
            if vals.size == 0:
                continue
            per_session.append(vals)

    if len(per_session) == 0:
        return np.array([])
    return np.concatenate(per_session)

    if point_level == "entry":
        return np.concatenate(per_session)
    elif point_level == "session_mean":
        return np.array([np.nanmean(v) for v in per_session])
    else:
        raise ValueError(f"Unknown point_level: {point_level}")


def _draw_box_strip_panel(ax, group_values, group_labels, positions,
                          group_colors, box_width=0.6):
    """Draw box + (conditional) strip for a set of groups at given x positions."""
    box_data = [gv if gv.size > 0 else np.array([np.nan]) for gv in group_values]
    bp = ax.boxplot(box_data, positions=positions, widths=box_width,
                    showfliers=False, patch_artist=True)

    # Color each box body by its group; keep the median line readable.
    for patch, color in zip(bp["boxes"], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")

    for pos, gv in zip(positions, group_values):
        if gv.size == 0:
            continue
        # Hide the strip (alpha -> 0) when there are too many points.
        alpha = 0.0 if gv.size > STRIP_ALPHA_THRESHOLD else 0.4
        if alpha > 0:
            jitter = (np.random.rand(gv.size) - 0.5) * (box_width * 0.6)
            ax.scatter(np.full(gv.size, pos) + jitter, gv,
                       s=8, alpha=alpha, color="black", zorder=3)


def _annotate_pairwise_stats(ax, group_values, positions, stats_rows,
                             response_range):
    """
    Run pairwise Mann-Whitney U tests between groups and annotate significant
    (or all) comparisons with brackets + stars. Appends results to stats_rows.
    """
    finite = [gv for gv in group_values]
    y_max = np.nanmax([np.nanmax(gv) for gv in finite if gv.size > 0]
                      or [0.0])
    y_min = np.nanmin([np.nanmin(gv) for gv in finite if gv.size > 0]
                      or [0.0])
    span = (y_max - y_min) if (y_max > y_min) else 1.0
    step = span * 0.08
    level = 0

    for a, b in itertools.combinations(range(len(group_values)), 2):
        gv_a, gv_b = group_values[a], group_values[b]
        if gv_a.size < 2 or gv_b.size < 2:
            continue
        try:
            u_stat, p_val = stats.mannwhitneyu(gv_a, gv_b,
                                               alternative="two-sided")
        except ValueError:
            continue

        stats_rows.append({
            "response_range": response_range,
            "group_a": a,
            "group_b": b,
            "n_a": gv_a.size,
            "n_b": gv_b.size,
            "U": u_stat,
            "p_value": p_val,
        })

        # Only draw a bracket for significant comparisons to reduce clutter.
        if p_val < 0.05:
            y = y_max + step * (level + 1)
            x1, x2 = positions[a], positions[b]
            ax.plot([x1, x1, x2, x2],
                    [y, y + step * 0.3, y + step * 0.3, y],
                    lw=1.0, color="black")
            ax.text((x1 + x2) / 2, y + step * 0.3, _p_to_stars(p_val),
                    ha="center", va="bottom", fontsize=9)
            level += 1


def make_distribution_figure(records, stimulus, groups, group_labels,
                             row_field, out_dir, metric_key,
                             triangle, point_level, fig_tag):
    """
    Build one box+strip figure. X = response range, groups stacked within each
    response range. Y = metric. Pairwise Mann-Whitney stats per response range.

    groups: list of group identifiers to look up via row_field. For the raw-vs-
        cca comparison (Viz 3), groups is a list of (data_source, row_value,
        row_field) tuples instead; see make_raw_vs_cca_figure.
    """
    metric_name = "Fisher's J" if metric_key == "J_matrix" else "d-prime"
    n_groups = len(groups)

    # One consistent fill color per group (the box color key).
    group_colors = [GROUP_FILL_COLORS[g % len(GROUP_FILL_COLORS)]
                    for g in range(n_groups)]

    fig, ax = plt.subplots(figsize=(3 + 2.5 * len(response_ranges), 6))

    group_gap = 1.0
    within_gap = 0.8
    block_width = n_groups * within_gap
    stats_rows = []

    xticks, xticklabels = [], []
    for r_idx, response_range in enumerate(response_ranges):
        block_start = r_idx * (block_width + group_gap)
        positions = [block_start + g * within_gap for g in range(n_groups)]
        xticks.append(np.mean(positions))
        xticklabels.append(response_range)

        group_values = []
        for grp in groups:
            data_source, row_value, gfield = grp
            gv = collect_group_values(
                records, data_source, row_value, gfield, response_range,
                metric_key, triangle, point_level)
            group_values.append(gv)

        _draw_box_strip_panel(ax, group_values, group_labels, positions,
                              group_colors)
        _annotate_pairwise_stats(ax, group_values, positions, stats_rows,
                                 response_range)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Response range")

    # Build a legend keyed by group fill color so brain regions are identifiable.
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                          color=group_colors[idx], label=lbl)
               for idx, lbl in enumerate(group_labels)]
    ax.legend(handles=handles, title="Group", loc="upper right", fontsize=8)

    caption = ("per-entry (non-independent; exploratory stats)"
               if point_level == "entry"
               else "session-averaged")
    ax.set_title(f"{stimulus} — {fig_tag} — {metric_name}\n{caption}",
                 fontsize=11)

    fig.tight_layout()

    metric_tag = "J" if metric_key == "J_matrix" else "dprime"
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    base = (f"{out_dir}/dprime_dist_{stimulus}_{fig_tag}_"
            f"{triangle}_{point_level}_{metric_tag}")
    fig.savefig(f"{base}.png", dpi=150)
    plt.close(fig)

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(f"{base}_stats.csv", index=False)
    print(f"Saved figure to {base}.png")


def make_within_source_figures(records, stimulus, data_source, out_dir,
                               metric_key, triangle, point_level):
    """
    Viz 1 / Viz 2: one figure per (stimulus, data_source). Groups are the
    brain regions (raw) or target region-pairs (cca).
    """
    if data_source == "raw":
        rows = raw_rows
        row_field = "brain_region"
        labels = [r for r in rows]
    else:
        rows = cca_target_rows
        row_field = "target_region"
        labels = [f"Primary -> {r.replace(' auditory area', '')}" for r in rows]

    groups = [(data_source, row_value, row_field) for row_value in rows]
    fig_tag = f"{data_source}"
    make_distribution_figure(
        records, stimulus, groups, labels, row_field, out_dir, metric_key,
        triangle, point_level, fig_tag)


def make_raw_vs_cca_figure(records, stimulus, out_dir, metric_key,
                           triangle, point_level):
    """
    Viz 3: raw Primary auditory area vs the two CCA region-pairs
    (Primary -> Ventral, Primary -> Dorsal). All represent Primary-area data.
    """
    groups = [
        ("raw", "Primary auditory area", "brain_region"),
        ("cca", "Ventral auditory area", "target_region"),
        ("cca", "Dorsal auditory area", "target_region"),
    ]
    labels = ["Raw Primary", "Primary -> Ventral", "Primary -> Dorsal"]
    make_distribution_figure(
        records, stimulus, groups, labels, "mixed", out_dir, metric_key,
        triangle, point_level, "raw_vs_cca")

# ----------------------------------------------------------------------------
# Within- vs between-sound-source comparison (natural sounds only)
# ----------------------------------------------------------------------------
# Natural sounds are 5 sources x 4 exemplars = 20 stimuli, ordered so that
# consecutive blocks of 4 belong to the same source (source = index // 4).
WITHIN_BETWEEN_COLORS = {"within": "#2C7FB8", "between": "#D95F0E"}


def source_labels_for_matrix(n, exemplars_per_source=4):
    """Return a length-n array of source labels via index // exemplars_per_source."""
    return np.arange(n) // exemplars_per_source


def split_within_between(mat, exemplars_per_source=4):
    """
    Split the upper-triangle entries of an RSA matrix into within-source and
    between-source values.

    An upper-triangle entry (i, j) is "within" if source(i) == source(j),
    otherwise "between". The main diagonal is excluded.

    Returns (within_vals, between_vals) as 1D arrays.
    """
    if mat is None:
        return np.array([]), np.array([])
    n = mat.shape[0]
    src = source_labels_for_matrix(n, exemplars_per_source)
    i, j = np.triu_indices(n, k=1)
    vals = mat[i, j]
    same_source = src[i] == src[j]
    return vals[same_source], vals[~same_source]


def collect_within_between(records, data_source, row_value, row_field,
                           response_range, metric_key, point_level,
                           exemplars_per_source=4):
    """
    Collect within-source and between-source values for one group/response range.

    point_level="session_mean": split the upper triangle of the session-averaged
                                matrix (like the heatmaps).
    point_level="entry":        pool the within / between entries across all
                                sessions (non-independent; exploratory).
    Returns (within_vals, between_vals) as 1D arrays.
    """
    if point_level == "session_mean":
        mean_mat = average_over_sessions(
            records, data_source, row_value, response_range,
            row_field, metric_key)
        return split_within_between(mean_mat, exemplars_per_source)

    if point_level != "entry":
        raise ValueError(f"Unknown point_level: {point_level}")

    within_all, between_all = [], []
    for r in records:
        if (r["data_source"] == data_source
                and r["response_range"] == response_range
                and r[row_field] == row_value
                and r[metric_key] is not None):
            w, b = split_within_between(r[metric_key], exemplars_per_source)
            if w.size:
                within_all.append(w)
            if b.size:
                between_all.append(b)

    within = np.concatenate(within_all) if within_all else np.array([])
    between = np.concatenate(between_all) if between_all else np.array([])
    return within, between


def _annotate_within_between_stats(ax, within_between_pairs, xcenters,
                                   within_offset, between_offset, box_width,
                                   response_range, groups, stats_rows):
    """
    Stats for the within/between figure:
      1) Mann-Whitney U (within vs between) at EACH x-tick (brain region / pair).
      2) Pairwise Mann-Whitney U across regions with Bonferroni correction, run
         separately for "within" across all regions and "between" across all
         regions. Significant (Bonferroni-corrected) comparisons are annotated.
    """
    all_vals = []
    for w, b in within_between_pairs:
        if w.size:
            all_vals.append(w)
        if b.size:
            all_vals.append(b)
    y_max = np.nanmax([np.nanmax(v) for v in all_vals] or [0.0])
    y_min = np.nanmin([np.nanmin(v) for v in all_vals] or [0.0])
    span = (y_max - y_min) if (y_max > y_min) else 1.0
    step = span * 0.08

    # (1) Per-tick within vs between (Mann-Whitney U).
    for idx, (w, b) in enumerate(within_between_pairs):
        if w.size < 2 or b.size < 2:
            continue
        try:
            u_stat, p_val = stats.mannwhitneyu(w, b, alternative="two-sided")
        except ValueError:
            continue
        stats_rows.append({
            "response_range": response_range,
            "test": "within_vs_between",
            "group": groups[idx][3],  # human-readable label (see build below)
            "n_within": w.size,
            "n_between": b.size,
            "statistic": u_stat,
            "p_value": p_val,
            "p_value_bonferroni": p_val,  # single test, no correction
        })
        if p_val < 0.05:
            x1 = xcenters[idx] + within_offset
            x2 = xcenters[idx] + between_offset
            y = y_max + step
            ax.plot([x1, x1, x2, x2],
                    [y, y + step * 0.3, y + step * 0.3, y],
                    lw=1.0, color="black")
            ax.text((x1 + x2) / 2, y + step * 0.3, _p_to_stars(p_val),
                    ha="center", va="bottom", fontsize=9)

    # (2) Pairwise Mann-Whitney U across regions, Bonferroni-corrected, run
    #     separately for the "within" family and the "between" family. Brackets
    #     for these region-vs-region comparisons are stacked above the per-tick
    #     annotations.
    level = 2  # start above the per-tick brackets
    for kind, sel, offset, color in (
            ("within", 0, within_offset, WITHIN_BETWEEN_COLORS["within"]),
            ("between", 1, between_offset, WITHIN_BETWEEN_COLORS["between"])):
        # Region indices that have enough data for this family.
        valid_idx = [i for i, pair in enumerate(within_between_pairs)
                     if pair[sel].size >= 2]
        comparisons = list(itertools.combinations(valid_idx, 2))
        n_comparisons = len(comparisons)
        if n_comparisons == 0:
            continue

        for (a, b) in comparisons:
            va = within_between_pairs[a][sel]
            vb = within_between_pairs[b][sel]
            try:
                u_stat, p_val = stats.mannwhitneyu(va, vb,
                                                   alternative="two-sided")
            except ValueError:
                continue
            # Bonferroni correction within this family.
            p_corrected = min(p_val * n_comparisons, 1.0)
            stats_rows.append({
                "response_range": response_range,
                "test": f"{kind}_across_regions",
                "group": f"{groups[a][3]} vs {groups[b][3]}",
                "n_within": va.size if kind == "within" else None,
                "n_between": vb.size if kind == "between" else None,
                "statistic": u_stat,
                "p_value": p_val,
                "p_value_bonferroni": p_corrected,
            })
            if p_corrected < 0.05:
                x1 = xcenters[a] + offset
                x2 = xcenters[b] + offset
                y = y_max + step * (level + 1)
                ax.plot([x1, x1, x2, x2],
                        [y, y + step * 0.3, y + step * 0.3, y],
                        lw=1.0, color=color)
                ax.text((x1 + x2) / 2, y + step * 0.3,
                        _p_to_stars(p_corrected),
                        ha="center", va="bottom", fontsize=9, color=color)
                level += 1


def make_within_between_figure(records, groups, out_dir, metric_key,
                               response_range, point_level, fig_tag,
                               exemplars_per_source=4):
    """
    One figure (one response range). X = brain region / region pair (per group),
    two boxes per tick (within vs between, color-coded). Y = metric.

    groups: list of (data_source, row_value, row_field, label) tuples.
    """
    metric_name = "Fisher's J" if metric_key == "J_matrix" else "d-prime"
    stimulus = "naturalSounds"

    n_groups = len(groups)
    xcenters = np.arange(n_groups) * 1.5  # spacing between region ticks
    box_width = 0.5
    within_offset = -box_width * 0.55
    between_offset = box_width * 0.55

    fig, ax = plt.subplots(figsize=(3 + 2.0 * n_groups, 6))

    within_between_pairs = []
    for (data_source, row_value, row_field, label) in groups:
        w, b = collect_within_between(
            records, data_source, row_value, row_field, response_range,
            metric_key, point_level, exemplars_per_source)
        within_between_pairs.append((w, b))

    # Draw within and between boxes at each tick.
    for idx, (w, b) in enumerate(within_between_pairs):
        for vals, offset, kind in ((w, within_offset, "within"),
                                    (b, between_offset, "between")):
            data = vals if vals.size > 0 else np.array([np.nan])
            bp = ax.boxplot([data], positions=[xcenters[idx] + offset],
                            widths=box_width, showfliers=False,
                            patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(WITHIN_BETWEEN_COLORS[kind])
                patch.set_alpha(0.7)
            for median in bp["medians"]:
                median.set_color("black")
            # Conditional strip overlay (gray, low alpha).
            if 0 < vals.size <= STRIP_ALPHA_THRESHOLD:
                jitter = (np.random.rand(vals.size) - 0.5) * (box_width * 0.6)
                ax.scatter(np.full(vals.size, xcenters[idx] + offset) + jitter,
                           vals, s=8, alpha=0.4, color="black", zorder=3)

    _annotate_within_between_stats(
        ax, within_between_pairs, xcenters, within_offset, between_offset,
        box_width, response_range, groups, _WB_STATS_ROWS)

    ax.set_xticks(xcenters)
    ax.set_xticklabels([g[3] for g in groups], rotation=15, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Brain region / region pair")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                          color=WITHIN_BETWEEN_COLORS[k], label=k)
               for k in ("within", "between")]
    ax.legend(handles=handles, title="Source pairing", loc="upper right",
              fontsize=8)

    caption = ("per-entry (non-independent; exploratory stats)"
               if point_level == "entry" else "session-averaged")
    ax.set_title(f"{stimulus} — {fig_tag} — {response_range} — {metric_name}\n"
                 f"{caption}", fontsize=11)

    fig.tight_layout()

    metric_tag = "J" if metric_key == "J_matrix" else "dprime"
    os.makedirs(f"{out_dir}", exist_ok=True)
    base = (f"{out_dir}/dprime_within_between_{fig_tag}_"
            f"{response_range}_{point_level}_{metric_tag}")
    fig.savefig(f"{base}.png", dpi=150)
    plt.close(fig)
    print(f"Saved figure to {base}.png")


# Module-level accumulator so all within/between stats (per-tick + omnibus) can
# be written to a single CSV per (fig_tag, point_level, metric). Reset per call
# batch in make_all_within_between_figures.
_WB_STATS_ROWS = []


def make_all_within_between_figures(records, out_dir, metric_key, point_level,
                                    exemplars_per_source=4):
    """
    Build the within/between figures for natural sounds: raw regions, cca region
    pairs, and the raw-Primary-vs-two-CCA-pairs comparison. One figure per
    response range for each of the three flavors.
    """
    raw_groups = [("raw", r, "brain_region", r) for r in raw_rows]
    cca_groups = [
        ("cca", r, "target_region",
         f"Primary -> {r.replace(' auditory area', '')}")
        for r in cca_target_rows
    ]
    raw_vs_cca_groups = [
        ("raw", "Primary auditory area", "brain_region", "Raw Primary"),
        ("cca", "Ventral auditory area", "target_region", "Primary -> Ventral"),
        ("cca", "Dorsal auditory area", "target_region", "Primary -> Dorsal"),
    ]

    flavors = [
        ("raw", raw_groups),
        ("cca", cca_groups),
        ("raw_vs_cca", raw_vs_cca_groups),
    ]

    metric_tag = "J" if metric_key == "J_matrix" else "dprime"
    for fig_tag, groups in flavors:
        global _WB_STATS_ROWS
        _WB_STATS_ROWS = []  # fresh accumulator per flavor
        for response_range in response_ranges:
            make_within_between_figure(
                records, groups, out_dir, metric_key, response_range,
                point_level, fig_tag, exemplars_per_source)
        if _WB_STATS_ROWS:
            os.makedirs(f"{out_dir}", exist_ok=True)
            stats_path = (f"{out_dir}/dprime_within_between_{fig_tag}_"
                          f"{point_level}_{metric_tag}_stats.csv")
            pd.DataFrame(_WB_STATS_ROWS).to_csv(stats_path, index=False)
            print(f"Saved within/between stats to {stats_path}")

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

        # Distribution figures (box + strip) in both triangle modes and both
        # point levels (per-entry and session-averaged).
        for triangle in ("upper", "neardiag"):
            for point_level in ("entry", "session_mean"):
                # Viz 1 / Viz 2: within-source (raw, then cca).
                make_within_source_figures(
                    records, stimulus, "raw", out_dir+f"/figures/{triangle}", metric_key,
                    triangle, point_level)
                make_within_source_figures(
                    records, stimulus, "cca", out_dir+f"/figures/{triangle}", metric_key,
                    triangle, point_level)
                # Viz 3: raw Primary vs the two CCA region-pairs.
                make_raw_vs_cca_figure(
                    records, stimulus, out_dir+f"/figures/{triangle}", metric_key,
                    triangle, point_level)
        # Within- vs between-sound-source figures (natural sounds only), for
        # both per-entry and session-averaged point levels.
        if stimulus == "naturalSound":
            for point_level in ("entry", "session_mean"):
                make_all_within_between_figures(
                    records, out_dir+"/figures/wb", metric_key, point_level)


if __name__ == "__main__":
    # Pass metric_key="dprime_matrix" to plot classic d-prime instead of J.
    main(metric_key="J_matrix")
    # main(metric_key="dprime_matrix")