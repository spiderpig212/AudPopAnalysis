"""
Paired, stimulus-pair-level version of the final D' plots, drawn as violins (companion to
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
  Brackets mark corrected p < 0.05. Each distribution is a violin (kernel density of
  the stimulus-pair values, with a dot at the median and a bar for the quartiles);
  individual points and pairing lines are not drawn. A companion panel shows the
  per-pair differences against zero, also as violins.

Caveats: entries that share a stimulus are not fully independent (more so for
the upper triangle than the near-diagonal); region maps may average different
numbers of sessions (printed when run).

Display options for heavy-tailed values (statistics are always on the raw values):
  yscale="log"        log y-axis on the violins (density estimated in log space); the companion panel then shows the
                      per-pair log2 ratio (region b / region a) instead of the difference
  clip_outliers=True  limit the axes to Q3 + clip_iqr * IQR (and Q1 - clip_iqr * IQR) and
                      draw anything beyond as a triangle on the axis edge labeled with its
                      true value. Works alone or together with yscale="log".
Pairs whose value exceeds OUTLIER_FLAG_FACTOR x the panel median are printed at run time
(with their averaged accuracy), since a huge D' usually means a near-perfectly separable
pair with collapsed variance rather than a real effect.

Text collisions (big poster fonts): before saving, resolve_text_overlap() measures the
rendered tick labels and titles; overlapping x tick labels are first wrapped onto two lines
("Ventral −\nPrimary"), then rotated 30 degrees if that is not enough, and overlapping
titles are wrapped the same way. Any text that still overlaps is reported as a warning.

Valid metric keys: "J_matrix", "dprime_matrix", "accuracy_matrix".
"""
import itertools
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy import stats

import figure_params as fp
from d_prime_RSA_visualization import load_rsa_data, _p_to_stars
from final_figures_dprime import _region_groups

OUTLIER_FLAG_FACTOR = 5.0   # report pairs above this multiple of the panel median
VIOLIN_WIDTH = 0.8          # full width of the widest part of a violin (x units)
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
    Returns (values, session_counts, (rows, cols)): values is (n_groups, n_pairs), row g
    holding group g's selected entries of its session-averaged map, in the same
    stimulus-pair order for every group; rows/cols are those entries' matrix indices.
    None if any group has no data.
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
    return np.stack([m[ii, jj] for m in mats]), counts, (ii, jj)


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
def _bracket(ax, x1, x2, y0, y1, text, fontsize):
    ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0], lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y1, text, ha="center", va="bottom", fontsize=fontsize)


def _fwd(v, log):
    return np.log10(v) if log else np.asarray(v, dtype=float)


def _inv(t, log):
    return 10.0 ** t if log else t


def _clip_bounds(arrays, factor):
    """(lo, hi): limits at Q1 - factor*IQR / Q3 + factor*IQR of the most generous group.
    A side is None when no value lies beyond it (nothing to clip there)."""
    q = [np.percentile(a, [25, 75]) for a in arrays]
    hi = max(q3 + factor * (q3 - q1) for q1, q3 in q)
    lo = min(q1 - factor * (q3 - q1) for q1, q3 in q)
    allv = np.concatenate([np.ravel(a) for a in arrays])
    return (lo if allv.min() < lo else None), (hi if allv.max() > hi else None)


def _clip(arr, lo, hi):
    return np.clip(arr, -np.inf if lo is None else lo, np.inf if hi is None else hi)


def _fmt(v):
    text = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"
    return text.replace("-", "−")


EDGE_LABEL_SCALE = 0.7   # edge-marker value lists use this fraction of legend_fontsize
EDGE_LIST_MAX = 4   # list up to this many clipped values beside a marker; beyond it, a count and range


def _edge_label(values, up):
    """Text for the clipped values at one marker: one per line, most extreme first."""
    vals = sorted(values, reverse=up)
    if len(vals) <= EDGE_LIST_MAX:
        return "\n".join(_fmt(v) for v in vals)
    return f"{len(vals)} values\n{_fmt(vals[0])} to {_fmt(vals[-1])}"


def _edge_markers(ax, x, y, true_values, up, style):
    """
    One triangle per (x, edge) for the values beyond the clip limit; x and y are the
    markers' positions (all clipped values of a group share them), with the true values
    listed beside it, stacked into the panel, so several clipped values never overlap.
    """
    if len(x) == 0:
        return
    ax.scatter(x[:1], y[:1], marker="^" if up else "v", s=style["pca_marker_size"] * 4, color="black",
               linewidths=0, zorder=4, clip_on=False)
    ax.annotate(_edge_label(list(true_values), up), (x[0], y[0]), xytext=(8, 4 if up else -4),
                textcoords="offset points", ha="left", va="top" if up else "bottom",
                fontsize=max(7, style["legend_fontsize"] * EDGE_LABEL_SCALE), linespacing=1.1, annotation_clip=False,
                zorder=5)


def _log_ticks(ax):
    """Plain-number ticks on a log axis, denser when the axis spans less than a decade or two."""
    lo, hi = ax.get_ylim()
    decades = np.log10(hi / lo)
    subs = (1, 2, 5) if decades >= 1.5 else (1, 2, 3, 5, 7) if decades >= 0.8 else tuple(range(1, 10))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _violin(ax, x, t, inside, color, style, log=False):
    """
    One violin at position x. t: all values in axis space (log10 when log); the density
    is estimated from the values that lie inside the clip limits (`inside` mask), so
    clipped outliers don't stretch it. Median dot and quartile bar use every value.
    """
    v = t[inside]
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    half_w = VIOLIN_WIDTH / 2
    face = mcolors.to_rgba(color, 0.7)
    drawn = False
    if v.size >= 3 and np.ptp(v) > 0:
        try:
            grid = np.linspace(v.min(), v.max(), 200)
            dens = stats.gaussian_kde(v)(grid)
            half = half_w * dens / dens.max()
            ax.fill_betweenx(_inv(grid, log), x - half, x + half, facecolor=face, edgecolor="0.25",
                             linewidth=style["line_width"], zorder=2)
            drawn = True
        except np.linalg.LinAlgError:
            pass
    if not drawn:  # too few / identical values: just a short bar at the median
        ax.hlines(_inv(med, log), x - half_w / 2, x + half_w / 2, color="0.25",
                  linewidth=style["line_width"] * 2, zorder=2)
    ax.vlines(x, _inv(q1, log), _inv(q3, log), color="black", linewidth=style["line_width"] * 3, zorder=3)
    ax.scatter([x], [_inv(med, log)], s=style["pca_marker_size"] * 2.5, color="white", edgecolors="black",
               linewidths=style["line_width"] * 0.8, zorder=4)


def draw_paired_violin(ax, values, tick_labels, colors, style, tests, log=False, clip_iqr=None,
                       axes_height_in=5.0):
    n_groups, n_pairs = values.shape
    pos = np.arange(n_groups)
    if log:
        ax.set_yscale("log")

    # Everything below is computed in axis space (log10 when log) so clipping and
    # bracket spacing are uniform on the plotted scale.
    t = _fwd(values, log)
    lo_b, hi_b = _clip_bounds(list(t), clip_iqr) if clip_iqr else (None, None)
    tc = _clip(t, lo_b, hi_b)

    inside = (t >= (-np.inf if lo_b is None else lo_b)) & (t <= (np.inf if hi_b is None else hi_b))
    for g in range(n_groups):
        _violin(ax, pos[g], t[g], inside[g], colors[g], style, log)
    for g in range(n_groups):
        for beyond, limit, up in ((t[g] > hi_b if hi_b is not None else None, hi_b, True),
                                  (t[g] < lo_b if lo_b is not None else None, lo_b, False)):
            if beyond is not None and beyond.any():
                k = beyond.sum()
                _edge_markers(ax, np.full(k, pos[g]), np.full(k, _inv(limit, log)), values[g][beyond], up, style)

    tmin, tmax = tc.min(), tc.max()
    span = max(tmax - tmin, 1e-9)
    level = 0
    sig = sorted([x for x in tests if x["p_corrected"] < 0.05], key=lambda x: x["b"] - x["a"])
    # Bracket spacing: at least 8% of the data span, and enough that a row of stars
    # (star_fontsize) fits between one bracket and the next at this axes height.
    n_sig = len(sig)
    frac = 1.5 * style["star_fontsize"] / 72 / axes_height_in
    step = max(span * 0.08, frac * span * 1.06 / max(1 - frac * (n_sig + 1.9), 0.2))
    for x in sig:  # short brackets lowest
        y0 = tmax + step * (level + 1)
        _bracket(ax, pos[x["a"]], pos[x["b"]], _inv(y0, log), _inv(y0 + step * 0.3, log), 
                 _p_to_stars(x["p_corrected"]), style["star_fontsize"])
        level += 1
    ax.set_ylim(_inv(tmin - span * 0.06, log), _inv(tmax + (step * (level + 1.9) if sig else span * 0.12), log))
    if log:
        _log_ticks(ax)
    ax.set_xticks(pos)
    ax.set_xticklabels(tick_labels, fontsize=style["tick_fontsize"])
    ax.set_xlim(-0.6, n_groups - 0.4)


def draw_differences(ax, values, short, style, tests, ylabel, log=False, clip_iqr=None):
    """Per-pair difference (b - a), or log2 ratio (b / a) when log, for each tested region pair."""
    ax.axhline(0, color="0.3", ls="--", lw=1.0, zorder=1)
    if log:
        diffs = [np.log2(values[x["b"]] / values[x["a"]]) for x in tests]
        sep, ylabel = " / ", f"log₂ {ylabel} ratio"
    else:
        diffs = [values[x["b"]] - values[x["a"]] for x in tests]
        sep, ylabel = " − ", f"Δ {ylabel}"
    pos = np.arange(len(tests))

    lo_b, hi_b = _clip_bounds(diffs, clip_iqr) if clip_iqr else (None, None)
    shown = [_clip(d, lo_b, hi_b) for d in diffs]
    for p_i, d in zip(pos, diffs):
        inside = (d >= (-np.inf if lo_b is None else lo_b)) & (d <= (np.inf if hi_b is None else hi_b))
        _violin(ax, p_i, d, inside, "0.75", style)
        for beyond, limit, up in ((d > hi_b if hi_b is not None else None, hi_b, True),
                                  (d < lo_b if lo_b is not None else None, lo_b, False)):
            if beyond is not None and beyond.any():
                k = beyond.sum()
                _edge_markers(ax, np.full(k, p_i), np.full(k, limit), d[beyond], up, style)

    top = max(np.max(d) for d in shown)
    bottom = min(np.min(d) for d in shown)
    span = max(top - bottom, 1e-9)
    for p_i, x in zip(pos, tests):  # stars pinned to the top of the axes, above any edge-marker labels
        ax.text(p_i, 0.97, _p_to_stars(x["p_corrected"]), ha="center", va="top",
                fontsize=style["star_fontsize"], transform=ax.get_xaxis_transform())
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{short[x['b']]}{sep}{short[x['a']]}" for x in tests], fontsize=style["tick_fontsize"])
    ax.set_xlim(-0.6, len(tests) - 0.4)
    ax.set_ylabel(ylabel, fontsize=style["label_fontsize"])
    ax.set_ylim(min(bottom, 0) - span * 0.1, top + span * 0.32)  # room for the stars


def flag_outliers(values, idx, short, accuracy=None, factor=OUTLIER_FLAG_FACTOR):
    """Print stimulus pairs whose value in any region exceeds factor x the panel median."""
    cols = np.where((values > factor * np.median(values)).any(axis=0))[0]
    for k in cols:
        line = ", ".join(f"{short[g]} {values[g, k]:.3g}" + (f" (acc {accuracy[g, k]:.2f})" if accuracy is not None else "")
                         for g in range(values.shape[0]))
        print(f"[paired]   large value, stimulus pair ({idx[0][k]}, {idx[1][k]}): {line}")


# ----------------------------------------------------------------------------
# Text overlap handling
# ----------------------------------------------------------------------------
def _extents(texts, renderer):
    return [t.get_window_extent(renderer) for t in texts]


def _any_overlap(texts, renderer, gap=0.04):
    """True if any two texts overlap, or sit closer than `gap` x their width (so labels that
    only just touch count as colliding)."""
    bbs = [b.expanded(1 + gap, 1) for b in _extents(texts, renderer)]
    return any(a.overlaps(b) for a, b in itertools.combinations(bbs, 2))


def _wrap_label(text):
    """Break a 'A − B' / 'A / B' / 'A → B' label after the operator."""
    return re.sub(r" ([−/→]) ", r" \1\n", text, count=1)


def _fix_xticklabels(fig, ax, renderer):
    labels = ax.get_xticklabels()
    if len(labels) < 2 or not _any_overlap(labels, renderer):
        return
    original = [l.get_text() for l in labels]
    size = labels[0].get_fontsize()
    ax.set_xticklabels([_wrap_label(t) for t in original])
    for scale in (1.0, 0.9, 0.8):   # wrapped, then wrapped and slightly smaller
        for l in ax.get_xticklabels():
            l.set_fontsize(size * scale)
        fig.canvas.draw()
        if not _any_overlap(ax.get_xticklabels(), renderer):
            return
    ax.set_xticklabels(original, rotation=30, ha="right", rotation_mode="anchor")
    for l in ax.get_xticklabels():
        l.set_fontsize(size)
    fig.canvas.draw()


def _fix_titles(fig, axes, renderer):
    titles = [ax.title for ax in axes if ax.get_title()]
    if len(titles) < 2 or not _any_overlap(titles, renderer):
        return
    for wrap in (lambda t: t.replace(", ", ",\n", 1), lambda t: t.replace(" (n", "\n(n", 1)):
        for ax in axes:
            if ax.get_title():
                ax.set_title(wrap(ax.get_title()))
        fig.canvas.draw()
        if not _any_overlap([ax.title for ax in axes if ax.get_title()], renderer):
            return


def _visible_ticklabels(ax):
    lo, hi = sorted(ax.get_ylim())
    pad = 1e-9 + 0.001 * (hi - lo)
    return [l for l, loc in zip(ax.get_yticklabels(), ax.get_yticks()) if lo - pad <= loc <= hi + pad and l.get_text()]


def resolve_text_overlap(fig, label=""):
    """Wrap or rotate colliding x tick labels, wrap colliding titles, then warn about leftovers."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = fig.axes
    for ax in axes:
        _fix_xticklabels(fig, ax, renderer)
    rows = {}
    for ax in axes:   # panels in one row share the smallest tick font any of them needed
        rows.setdefault(ax.get_subplotspec().rowspan.start, []).append(ax)
    for row_axes in rows.values():
        smallest = min(l.get_fontsize() for ax in row_axes for l in ax.get_xticklabels())
        for ax in row_axes:
            for l in ax.get_xticklabels():
                l.set_fontsize(smallest)
    _fix_titles(fig, [ax for ax in axes if ax.get_subplotspec().rowspan.start == 0], renderer)

    items = []
    for ax in axes:
        items += [(f"x tick '{l.get_text()}'", l) for l in ax.get_xticklabels() if l.get_text()]
        items += [(f"y tick '{l.get_text()}'", l) for l in _visible_ticklabels(ax)]
        items += [(f"text '{t.get_text()}'", t) for t in ax.texts if t.get_text()]
        items += [(f"title '{ax.get_title()}'", ax.title)] if ax.get_title() else []
        items += [(f"ylabel '{ax.get_ylabel()}'", ax.yaxis.label)] if ax.get_ylabel() else []
    fig.canvas.draw()
    bbs = [t.get_window_extent(renderer) for _, t in items]
    for (i, (na, _)), (j, (nb, _)) in itertools.combinations(list(enumerate(items)), 2):
        if bbs[i].overlaps(bbs[j]):
            print(f"[paired] WARNING {label}: overlapping text: {na!r} and {nb!r}".replace("\n", " "))


# ----------------------------------------------------------------------------
# Figure builder
# ----------------------------------------------------------------------------
def make_paired_figure(records, stim, data_source, out_dir, style_name="paper",
                       response_range=fp.MAIN_RESPONSE_RANGE, metric_key="J_matrix",
                       synthetic_kind="neardiag", correction="bonferroni",
                       show_differences=True, share_y=False, yscale="linear",
                       clip_outliers=False, clip_iqr=3.0, formats=fp.DEFAULT_FORMATS):
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
        values, counts, idx = got
        acc = None
        if metric_key != "accuracy_matrix":
            acc_got = collect_paired_values(records, groups, response_range, "accuracy_matrix", kind)
            acc = acc_got[0] if acc_got else None
        flag_outliers(values, idx, short, acc)
        panels.append((kind, values, counts, *paired_tests(values, correction)))
    print(f"[paired] {stim}/{data_source}: sessions averaged per region "
          f"{dict(zip(short, panels[0][2]))}")

    log = yscale == "log"
    if log and min(p[1].min() for p in panels) <= 0:
        print(f"[paired] {stim}/{data_source}: values <= 0, cannot use a log axis; falling back to linear")
        log = False
    clip = clip_iqr if clip_outliers else None

    n_cols, n_rows = len(panels), 2 if show_differences else 1
    fw, fh = style["figsize"]
    fig_h = fh * (1.45 if show_differences else 1.0)
    axes_height_in = fig_h * (2 / 3 if show_differences else 1.0) * 0.72   # approx., before layout
    with plt.rc_context(fp.rc_params(style)):
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, layout="constrained", sharey="row" if share_y else False,
                                 figsize=(fw * n_cols, fh * (1.45 if show_differences else 1.0)),
                                 gridspec_kw=dict(height_ratios=[2, 1] if show_differences else [1]))
        stats_rows = []
        for c, (kind, values, counts, friedman, tests) in enumerate(panels):
            ax = axes[0][c]
            draw_paired_violin(ax, values, tick_labels, colors, style, tests, log, clip, axes_height_in)
            title = f"{STIM_TITLES[stim]}, {KIND_TITLES[kind]} (n = {values.shape[1]})"
            if friedman:
                title += f"\nFriedman p = {friedman['p']:.2g}"
            ax.set_title(title, pad=max(6, 0.35 * style["title_fontsize"]))
            if c == 0 or not share_y:
                ax.set_ylabel(ylabel, fontsize=style["label_fontsize"])
            if show_differences:
                draw_differences(axes[1][c], values, short, style, tests, ylabel, log, clip)
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
        name += ("_log" if log else "") + ("_clip" if clip else "")
        resolve_text_overlap(fig, name)
        fp.save_figure(fig, out_dir, name, style, formats)
    pd.DataFrame(stats_rows).to_csv(f"{out_dir}/{name}_stats.csv", index=False)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(style_name="paper", response_ranges=(fp.MAIN_RESPONSE_RANGE,), metric_key="J_matrix",
         synthetic_kind="neardiag", correction="bonferroni", show_differences=True,
         share_y=False, yscale="linear", clip_outliers=False, clip_iqr=3.0,
         formats=fp.DEFAULT_FORMATS, records_by_stim=None, out_dir=None):
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
                                   metric_key, synthetic_kind, correction, show_differences, share_y,
                                   yscale, clip_outliers, clip_iqr, formats)


if __name__ == "__main__":
    main(style_name="poster", metric_key="J_matrix", synthetic_kind="upper", formats=("svg",), clip_outliers=True)
    # main(yscale="log", formats=("png", "svg"))                       # log axis, log2-ratio differences
    # main(clip_outliers=True, formats=("png", "svg"))                 # linear axis, outliers as edge triangles
    # main(yscale="log", clip_outliers=True, formats=("png", "svg"))   # both
    # main(style_name="paper", metric_key="accuracy_matrix", formats=("png", "svg"))
    # main(metric_key="accuracy_matrix", synthetic_kind="upper")   # all pairs instead of near-diagonal
    # main(correction="holm")
    # main(style_name="poster", formats=("svg",))
    # main(response_ranges=("onset", "offset"))   # supplementary
