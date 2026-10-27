"""
Paired, stimulus-pair-level version of the final D' box plots (companion to
final_figures_dprime.py, which is unchanged).

Calculation
  For each stimulus type and region (or Primary->target pair), the RSA matrices
  are first AVERAGED ACROSS SESSIONS into one heat map. Every selected entry of
  that averaged map is then one point: near-diagonal (adjacent-stimulus) or
  upper-triangle entries for AM / pure tones; within-source and between-source
  entries for natural sounds (two panels). So n = number of stimulus pairs, not
  sessions, and the inference is over stimulus pairs.

Statistics (per stimulus / panel)
  Entry (i, j) of the averaged map is the same stimulus pair in every region, so
  regions are matched by stimulus pair:
    - Friedman test across the regions (omnibus, 3+ regions only)
    - post-hoc Wilcoxon signed-rank test for every region pair, two-sided,
      corrected for the number of region pairs (Bonferroni by default, or Holm)
  Brackets mark corrected p < 0.05. Faint lines link each stimulus pair across
  regions (hidden when there are many pairs), and a companion panel shows the
  per-pair differences against zero.

Caveats: entries that share a stimulus are not fully independent (more so for
the upper triangle than the near-diagonal); region maps may average different
numbers of sessions (printed when run).

Valid metric keys: "J_matrix", "dprime_matrix", "accuracy_matrix".
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import figure_params as fp
from d_prime_RSA_visualization import load_rsa_data, STRIP_ALPHA_THRESHOLD, _p_to_stars
from final_figures_dprime import _region_groups

PAIRED_LINE_MAX = 40        # draw stimulus-pair lines only up to this many pairs
STIM_TITLES = {"AM": "AM", "pureTones": "Pure Tones", "naturalSound": "Natural sounds"}
KIND_TITLES = {"neardiag": "near-diagonal pairs", "upper": "all pairs",
               "within": "within-source pairs", "between": "between-source pairs"}


# ----------------------------------------------------------------------------
# Calculation
# ----------------------------------------------------------------------------
def entry_indices(kind, n):
    """Index arrays (rows, cols) of the matrix entries for one kind of comparison."""
    if kind == "neardiag":
        i = np.arange(n - 1)
        return i, i + 1
    ii, jj = np.triu_indices(n, k=1)
    if kind == "upper":
        return ii, jj
    same = (ii // fp.EXEMPLARS_PER_CATEGORY) == (jj // fp.EXEMPLARS_PER_CATEGORY)
    if kind == "within":
        return ii[same], jj[same]
    if kind == "between":
        return ii[~same], jj[~same]
    raise ValueError(f"Unknown kind: {kind}")


def averaged_matrix(records, data_source, row_value, row_field, response_range, metric_key):
    """Mean RSA matrix across sessions, and how many sessions went into it."""
    mats = [r[metric_key] for r in records
            if r["data_source"] == data_source and r["response_range"] == response_range
            and r[row_field] == row_value and r.get(metric_key) is not None]
    return (np.mean(mats, axis=0), len(mats)) if mats else (None, 0)


def collect_paired_values(records, groups, response_range, metric_key, kind):
    """
    Returns (values, session_counts): values is (n_groups, n_pairs), row g holding
    group g's selected entries of its session-averaged map, in the same
    stimulus-pair order for every group. None if any group has no data.
    """
    mats, counts = [], []
    for data_source, row_value, row_field in groups:
        m, n_sessions = averaged_matrix(records, data_source, row_value, row_field,
                                        response_range, metric_key)
        if m is None:
            return None
        mats.append(m)
        counts.append(n_sessions)
    if len({m.shape for m in mats}) != 1:
        raise ValueError("Regions have different stimulus sets; cannot pair entries.")
    ii, jj = entry_indices(kind, mats[0].shape[0])
    return np.stack([m[ii, jj] for m in mats]), counts


def _correct(pvals, method):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method == "holm":  # step-down: ascending p times (m - rank), made monotone
        order = np.argsort(p)
        out = np.empty(m)
        out[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
        return out
    raise ValueError(f"Unknown correction: {method}")


def paired_tests(values, correction="bonferroni"):
    """
    Friedman across rows (if 3+) and Wilcoxon signed-rank for every pair of rows.
    Returns (friedman_dict_or_None, list of dicts with a, b, W, p, p_corrected,
    median_diff = median(values[b] - values[a])).
    """
    n_groups, n_pairs = values.shape
    friedman = None
    if n_groups >= 3 and n_pairs >= 2:
        chi2, p = stats.friedmanchisquare(*values)
        friedman = dict(chi2=float(chi2), p=float(p))
    combos = list(itertools.combinations(range(n_groups), 2))
    rows = []
    for a, b in combos:
        try:
            w, p = stats.wilcoxon(values[a], values[b], alternative="two-sided")
        except ValueError:  # every difference is zero
            w, p = 0.0, 1.0
        rows.append(dict(a=a, b=b, W=float(w), p=float(p),
                         median_diff=float(np.median(values[b] - values[a]))))
    for r, pc in zip(rows, _correct([r["p"] for r in rows], correction)):
        r["p_corrected"] = float(pc)
    return friedman, rows


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def _bracket(ax, x1, x2, y, step, text, fontsize):
    ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y], lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y + step * 0.3, text, ha="center", va="bottom", fontsize=fontsize)


def draw_paired_box(ax, values, tick_labels, colors, style, tests):
    n_groups, n_pairs = values.shape
    pos = np.arange(n_groups)
    bp = ax.boxplot(list(values), positions=pos, widths=0.55, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")

    if n_pairs <= PAIRED_LINE_MAX:   # unjittered points, joined per stimulus pair
        ax.plot(pos, values, color="0.5", lw=0.6, alpha=0.4, zorder=2)
        for g in range(n_groups):
            ax.scatter(np.full(n_pairs, pos[g]), values[g], s=style["pca_marker_size"], color="black",
                       alpha=0.55, linewidths=0, zorder=3)
    elif n_pairs <= STRIP_ALPHA_THRESHOLD:
        rng = np.random.default_rng(0)
        for g in range(n_groups):
            ax.scatter(pos[g] + (rng.random(n_pairs) - 0.5) * 0.3, values[g], s=style["pca_marker_size"] * 0.7,
                       color="black", alpha=0.35, linewidths=0, zorder=3)

    span = max(values.max() - values.min(), 1e-9)
    step, level = span * 0.08, 0
    sig = sorted([t for t in tests if t["p_corrected"] < 0.05], key=lambda t: t["b"] - t["a"])
    for t in sig:  # short brackets lowest
        _bracket(ax, pos[t["a"]], pos[t["b"]], values.max() + step * (level + 1), step,
                 _p_to_stars(t["p_corrected"]), style["star_fontsize"])
        level += 1
    ax.set_xticks(pos)
    ax.set_xticklabels(tick_labels, fontsize=style["tick_fontsize"])
    ax.set_xlim(-0.6, n_groups - 0.4)


def draw_differences(ax, values, short, style, tests, ylabel):
    ax.axhline(0, color="0.3", ls="--", lw=1.0, zorder=1)
    rng = np.random.default_rng(1)
    diffs = [values[t["b"]] - values[t["a"]] for t in tests]
    pos = np.arange(len(tests))
    bp = ax.boxplot(diffs, positions=pos, widths=0.5, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("0.75")
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    for p_i, d in zip(pos, diffs):
        if d.size <= STRIP_ALPHA_THRESHOLD:
            ax.scatter(p_i + (rng.random(d.size) - 0.5) * 0.3, d, s=style["pca_marker_size"] * 0.7,
                       color="black", alpha=0.4, linewidths=0, zorder=3)
    top = max(np.max(d) for d in diffs)
    bottom = min(np.min(d) for d in diffs)
    span = max(top - bottom, 1e-9)
    for p_i, t in zip(pos, tests):
        ax.text(p_i, top + span * 0.04, _p_to_stars(t["p_corrected"]), ha="center", va="bottom",
                fontsize=style["star_fontsize"])
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{short[t['b']]} − {short[t['a']]}" for t in tests], fontsize=style["tick_fontsize"])
    ax.set_xlim(-0.6, len(tests) - 0.4)
    ax.set_ylabel(f"Δ {ylabel}", fontsize=style["label_fontsize"])
    ax.set_ylim(min(bottom, 0) - span * 0.1, top + span * 0.22)  # room for the stars


# ----------------------------------------------------------------------------
# Figure builder
# ----------------------------------------------------------------------------
def make_paired_figure(records, stim, data_source, out_dir, style_name="paper",
                       response_range=fp.MAIN_RESPONSE_RANGE, metric_key="J_matrix",
                       synthetic_kind="neardiag", correction="bonferroni",
                       show_differences=True, share_y=False, formats=fp.DEFAULT_FORMATS):
    style = fp.STYLES[style_name]
    groups, labels, colors = _region_groups(data_source)
    short = [fp.REGION_SHORT[g[1]] for g in groups]
    tick_labels = short if data_source == "raw" else labels   # "Primary", ... / "Primary → Ventral"
    kinds = ("within", "between") if stim == "naturalSound" else (synthetic_kind,)
    metric_tag, ylabel = fp.METRIC_INFO[metric_key]

    panels = []
    for kind in kinds:
        got = collect_paired_values(records, groups, response_range, metric_key, kind)
        if got is None:
            print(f"[paired] {stim}/{data_source}: missing region data, skipping")
            return
        values, counts = got
        panels.append((kind, values, counts, *paired_tests(values, correction)))
    print(f"[paired] {stim}/{data_source}: sessions averaged per region "
          f"{dict(zip(short, panels[0][2]))}")

    n_cols, n_rows = len(panels), 2 if show_differences else 1
    fw, fh = style["figsize"]
    with plt.rc_context(fp.rc_params(style)):
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, layout="constrained", sharey="row" if share_y else False,
                                 figsize=(fw * n_cols, fh * (1.45 if show_differences else 1.0)),
                                 gridspec_kw=dict(height_ratios=[2, 1] if show_differences else [1]))
        stats_rows = []
        for c, (kind, values, counts, friedman, tests) in enumerate(panels):
            ax = axes[0][c]
            draw_paired_box(ax, values, tick_labels, colors, style, tests)
            title = f"{STIM_TITLES[stim]}, {KIND_TITLES[kind]} (n = {values.shape[1]})"
            if friedman:
                title += f"\nFriedman p = {friedman['p']:.2g}"
            ax.set_title(title)
            if c == 0 or not share_y:
                ax.set_ylabel(ylabel, fontsize=style["label_fontsize"])
            if show_differences:
                draw_differences(axes[1][c], values, short, style, tests, ylabel)
                if c > 0 and share_y:
                    axes[1][c].set_ylabel("")

            base = dict(stimulus=stim, data_source=data_source, response_range=response_range,
                        metric=metric_key, panel=kind, n_pairs=values.shape[1], correction=correction)
            if friedman:
                stats_rows.append(dict(base, test="friedman", group_a="all regions", group_b="",
                                       statistic=friedman["chi2"], p_value=friedman["p"], p_corrected=np.nan,
                                       median_diff=np.nan))
            for t in tests:
                stats_rows.append(dict(base, test="wilcoxon_signed_rank", group_a=short[t["a"]],
                                       group_b=short[t["b"]], statistic=t["W"], p_value=t["p"],
                                       p_corrected=t["p_corrected"], median_diff=t["median_diff"]))

        name = f"paired_{stim}_{data_source}_{response_range}_{'-'.join(kinds)}_{style_name}_{metric_tag}"
        fp.save_figure(fig, out_dir, name, style, formats)
    pd.DataFrame(stats_rows).to_csv(f"{out_dir}/{name}_stats.csv", index=False)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(style_name="paper", response_ranges=(fp.MAIN_RESPONSE_RANGE,), metric_key="J_matrix",
         synthetic_kind="neardiag", correction="bonferroni", show_differences=True,
         share_y=False, formats=fp.DEFAULT_FORMATS, records_by_stim=None, out_dir=None):
    if records_by_stim is None:
        from analysis_class import FiringRateAnalysis
        rsa_dir = f"{FiringRateAnalysis(db_suffix='coords_updated').figdata_path}/dprime_RSA"
        records_by_stim = {s: load_rsa_data(rsa_dir, s) for s in ("AM", "pureTones", "naturalSound")}
        out_dir = out_dir or f"{rsa_dir}/figures/final_paired/{style_name}"

    for response_range in response_ranges:
        for stim, records in records_by_stim.items():
            if not records:
                print(f"No {stim} records found, skipping.")
                continue
            for data_source in ("raw", "cca"):
                make_paired_figure(records, stim, data_source, out_dir, style_name, response_range,
                                   metric_key, synthetic_kind, correction, show_differences, share_y, formats)


if __name__ == "__main__":
    main(style_name="paper", metric_key="J_matrix", synthetic_kind="upper", formats=("png", "svg"))
    # main(style_name="paper", metric_key="accuracy_matrix", formats=("png", "svg"))
    # main(metric_key="accuracy_matrix", synthetic_kind="upper")   # all pairs instead of near-diagonal
    # main(correction="holm")
    # main(style_name="poster", formats=("svg",))
    # main(response_ranges=("onset", "offset"))   # supplementary
