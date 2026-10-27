"""
Final-figure versions of the Primary-auditory-area CCA summary plots in
Two_region_CCA.py (create_summary_plots / create_d_component_summary_plots).

Changes from the originals:
  - Temporal association areas are excluded everywhere.
  - Only one response range is plotted (sustained by default; pass another for
    supplementary versions).
  - One panel with stimulus type on the x-axis (AM, Pure Tones, Natural sounds);
    bars are colored by the region pair being compared. Y-axis labels are
    unchanged from the originals.
  - One sample per session: the 5 random neuron subsamples run for a session are
    averaged first, so bar heights, SEM, the overlaid points and the statistics
    all use n = number of sessions rather than treating subsamples as independent.
  - Statistics: pairwise Mann-Whitney U between region pairs WITHIN each
    stimulus, Bonferroni-corrected by the number of comparisons in that stimulus

The Primary-vs-Primary bars are the within-area control (Primary split into two
randomly sampled halves).

Input is cca_primary_auditory_results_full.pkl (a list of per-row dicts). Sizes,
fonts and formats come from figure_params.py (paper / poster presets).
"""
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import figure_params as fp

# TODO: Generalize this to be using our normal path imports
PKL_PATH = "/Users/matt/Desktop/Research/Murray/data/cca_primary_auditory_results_full.pkl"
FIG = "cca_correlations"

PRIMARY = "Primary auditory area"
EXCLUDED_REGIONS = ("Temporal association areas",)

STIM_ORDER = ["AM", "pureTones", "naturalSound"]
STIM_TICKLABELS = {"AM": "AM", "pureTones": "Pure Tones", "naturalSound": "Natural sounds"}

# Other region of each Primary pair (Primary itself = the within-area control).
PAIR_ORDER = ["Ventral auditory area", "Dorsal auditory area", PRIMARY]
PAIR_LABELS = {"Ventral auditory area": "Primary vs Ventral",
               "Dorsal auditory area": "Primary vs Dorsal",
               PRIMARY: "Primary vs Primary (within-area split)"}

# Original y-axis labels.
LABEL_SIGNIFICANT = "Number of Significant Components"
LABEL_D_CORR = "Mean Correlation (d components only)"

# Columns read from the pickle; the permutation arrays are most of its 580 MB.
_KEEP = ["region1", "region2", "stimulus", "response_range", "session", "iteration",
         "significant_components", "mean_correlation_d", "analysis_type",
         "test_correlations_d"]


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def load_results(pkl_path=PKL_PATH):
    rows = pickle.load(open(pkl_path, "rb"))
    return pd.DataFrame([{k: r.get(k) for k in _KEEP} for r in rows])


def primary_pairs(df, response_range):
    """Primary-involving rows for one response range, Temporal association dropped,
    with a `pair` column naming the other region (Primary = within-area control)."""
    df = df[(df["region1"] == PRIMARY) | (df["region2"] == PRIMARY)]
    df = df[~df["region1"].isin(EXCLUDED_REGIONS) & ~df["region2"].isin(EXCLUDED_REGIONS)]
    df = df[df["response_range"] == response_range].copy()
    other = np.where(df["region1"] == PRIMARY, df["region2"], df["region1"])
    df["pair"] = np.where(df["analysis_type"] == "within_region", PRIMARY, other)
    return df


def add_d_component_means(df, d_comps):
    """test_correlations_d_{k}_comps: mean over folds and the first k significant
    components of each row (as in create_d_component_summary_plots)."""
    df = df[df["significant_components"] > 0].copy()  # d-component plots drop these
    for k in d_comps:
        df[f"test_correlations_d_{k}_comps"] = [
            float(np.mean(a[:, :k])) if np.size(a) else 0.0 for a in df["test_correlations_d"]]
    return df


def session_values(df, value_column):
    """{(stimulus, pair): per-session means}; each session's subsamples averaged."""
    per_session = df.groupby(["stimulus", "pair", "session"])[value_column].mean()
    return {key: g.to_numpy() for key, g in per_session.groupby(level=["stimulus", "pair"])}


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def draw_cca_bars(values, style, ylabel, title, response_range, show_values=False):
    """
    values: {(stimulus, pair): 1D array of per-session values}.
    Returns (fig, ax, stats_rows).
    """
    rng = np.random.default_rng(0)
    n_pairs = len(PAIR_ORDER)
    width = 0.26
    offsets = (np.arange(n_pairs) - (n_pairs - 1) / 2) * width
    colors = [fp.REGION_COLORS[p] for p in PAIR_ORDER]

    fig, ax = plt.subplots(figsize=style["figsize"])
    stats_rows = []
    top = 0.0

    for s_idx, stim in enumerate(STIM_ORDER):
        positions = s_idx + offsets
        group_vals = [values.get((stim, p), np.array([])) for p in PAIR_ORDER]
        stim_top = 0.0

        for pos, vals, color in zip(positions, group_vals, colors):
            if vals.size == 0:
                continue
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            ax.bar(pos, mean, width, color=color, alpha=0.7, zorder=2)
            ax.errorbar(pos, mean, yerr=sem, color="black", capsize=3, zorder=3,
                        linewidth=style["line_width"])
            ax.scatter(pos + (rng.random(vals.size) - 0.5) * width * 0.5, vals, s=style["pca_marker_size"],
                       color="black", alpha=0.5, linewidths=0, zorder=4)
            if show_values:
                ax.text(pos, mean + sem, f"{mean:.2f}", ha="center", va="bottom",
                        fontsize=style["tick_fontsize"])
            stim_top = max(stim_top, vals.max(), mean + sem)
        top = max(top, stim_top)

        # Pairwise Mann-Whitney U within this stimulus, Bonferroni-corrected.
        present = [i for i, v in enumerate(group_vals) if v.size >= 2]
        combos = list(itertools.combinations(present, 2))
        step = max(stim_top, 1e-9) * 0.08
        level = 0
        for a, b in combos:
            try:
                u, p = stats.mannwhitneyu(group_vals[a], group_vals[b], alternative="two-sided")
            except ValueError:
                continue
            p_corr = min(p * len(combos), 1.0)
            stats_rows.append(dict(stimulus=stim, response_range=response_range,
                                   group_a=PAIR_LABELS[PAIR_ORDER[a]], group_b=PAIR_LABELS[PAIR_ORDER[b]],
                                   n_a=group_vals[a].size, n_b=group_vals[b].size, U=u,
                                   p_value=p, p_bonferroni=p_corr))
            if p_corr < 0.05:
                y = stim_top + step * (level + 1)
                x1, x2 = positions[a], positions[b]
                ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                        lw=1.0, color="black")
                ax.text((x1 + x2) / 2, y + step * 0.3, _stars(p_corr), ha="center",
                        va="bottom", fontsize=style["star_fontsize"])
                level += 1
        top = max(top, stim_top + step * (level + 1.6))

    ax.set_xticks(range(len(STIM_ORDER)))
    ax.set_xticklabels([STIM_TICKLABELS[s] for s in STIM_ORDER], fontsize=style["tick_fontsize"])
    ax.tick_params(axis="y", labelsize=style["tick_fontsize"])
    ax.set_xlabel("Stimulus type", fontsize=style["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=style["label_fontsize"])
    ax.set_title(title, fontsize=style["title_fontsize"])
    ax.set_ylim(0, top * 1.05)
    ax.set_xlim(-0.6, len(STIM_ORDER) - 0.4)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, color=c,
                          label=PAIR_LABELS[p]) for p, c in zip(PAIR_ORDER, colors)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=style["legend_fontsize"], frameon=False)
    return fig, ax, stats_rows


def _make_figure(df, value_column, ylabel, subtitle, tag, style_name, formats, out_dir,
                 response_range, show_values, report=False):
    style = fp.STYLES[style_name]
    values = session_values(df, value_column)
    with plt.rc_context(fp.rc_params(style)):
        fig, ax, stats_rows = draw_cca_bars(
            values, style, ylabel, f"CCA, Primary comparisons ({response_range})\n{subtitle}",
            response_range, show_values)
        base = f"cca_{tag}_{response_range}_{style_name}"
        fp.save_figure(fig, out_dir, base, style, formats)
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(f"{out_dir}/{base}_stats.csv", index=False)
    if report:
        n = {f"{STIM_TICKLABELS[s]}/{PAIR_LABELS[p].split(' (')[0]}": v.size for (s, p), v in values.items()}
        print(f"  sessions per bar: {n}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(style_name="paper", formats=fp.DEFAULT_FORMATS, response_range=fp.MAIN_RESPONSE_RANGE,
         d_comps=(1, 2, 3, 4, 5), include_mean_d=True, show_values=False,
         pkl_path=PKL_PATH, df=None, out_dir=None):
    out_dir = out_dir or fp.figure_output_dir(FIG, style_name)
    df = primary_pairs(load_results(pkl_path) if df is None else df, response_range)
    print(f"{len(df)} Primary-involving rows ({response_range}, Temporal association excluded)")

    _make_figure(df, "significant_components", LABEL_SIGNIFICANT, "Significant components",
                 "significant_components", style_name, formats, out_dir, response_range, show_values,
                 report=True)

    d_df = add_d_component_means(df, d_comps)
    for k in d_comps:
        label = "component" if k == 1 else "components"
        _make_figure(d_df, f"test_correlations_d_{k}_comps", LABEL_D_CORR,
                     f"D-component correlations, first {k} {label}", f"dcorr_first{k}",
                     style_name, formats, out_dir, response_range, show_values)
    if include_mean_d:
        _make_figure(d_df, "mean_correlation_d", LABEL_D_CORR, "D-component correlations, all d components",
                     "dcorr_all_d", style_name, formats, out_dir, response_range, show_values)


if __name__ == "__main__":
    main(style_name="paper", formats=("png", "svg"))
    # main(style_name="poster", formats=("svg",))
    # main(response_range="onset")   # supplementary
