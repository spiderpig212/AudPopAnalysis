"""
Supporting panels for final_figures_dprime.py. Reads the files written by
supp_decoder_data.py; nothing is calculated here beyond layout.

  supp_pair_*      Each trial of an example stimulus pair as a point in the pair's
                   PC1/PC2 plane, with a linear SVC's decision boundary (and
                   faint decision regions). One panel each for AM, pure tones,
                   a within-source natural pair and a between-source natural pair.
  supp_heatmap_*   The example session's RSA matrix per stimulus type, with the
                   entries that  feed the box plots outlined: near-diagonal pairs
                   for AM / pure tones, within-source blocks for natural sounds.
                   Open circles mark the example pairs shown in the PC panels.

Size/text come from figure_params.STYLES ("paper" / "poster").
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import figure_params as fp
import studyparams
from fig1_figures import _load_pickle, _load_spec, _natural_label

FIG = "supp_decoder"
FIG1 = "figure1"  # stimulus labels (AM rates, tone frequencies) come from its spectrogram data

PAIR_TITLES = {"AM": "AM noise", "pureTones": "Pure tones",
               "naturalSound_within": "Natural sounds, within source",
               "naturalSound_between": "Natural sounds, between sources"}



_label_dir = None  # set by main(); None = Figure 1's default data directory


def _value_labels(stim, n):
    """Per-stimulus labels in real units, or plain indices if unavailable."""
    if stim == "naturalSound":
        return [_natural_label(i) for i in range(n)]
    spec = _load_spec(_label_dir or fp.figure_data_dir(FIG1), stim)
    if spec is None or len(spec["labels"]) != n:
        return [str(i) for i in range(n)]
    vals = spec["labels"].astype(float)
    return ([f"{v:.1f}" for v in vals] if stim == "AM"
            else [f"{v / 1000:.1f}" for v in vals])


def _unit(stim):
    return {"AM": " Hz", "pureTones": " kHz", "naturalSound": ""}[stim]


# ----------------------------------------------------------------------------
# PC-space pairs with a decision boundary
# ----------------------------------------------------------------------------
def _draw_pair(ax, entry, stim, style):
    scores, cls = entry["scores"], entry["pair_idx"]
    col_a, col_b = fp.PAIR_COLORS
    labels = _value_labels(stim, len(entry["uniq_stims"]))
    lab_a, lab_b = (labels[i] + _unit(stim) for i in entry["pair"])

    lo, hi = scores.min(0), scores.max(0)
    pad = 0.15 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
    xx, yy = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    zz = entry["w"][0] * xx + entry["w"][1] * yy + entry["b"]

    # Faint decision regions (class b where the decision function is positive).
    ax.contourf(xx, yy, zz, levels=[min(zz.min(), 0) - 1, 0, max(zz.max(), 0) + 1],
                colors=[col_a, col_b], alpha=0.12)
    if zz.min() < 0 < zz.max():
        ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=style["line_width"] * 1.5)

    for c, col, lab in ((0, col_a, lab_a), (1, col_b, lab_b)):
        ax.scatter(scores[cls == c, 0], scores[cls == c, 1], s=style["pca_marker_size"] * 1.6,
                   color=col, edgecolors="white", linewidths=0.4, zorder=3, label=lab)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC1 ({entry['evr'][0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({entry['evr'][1] * 100:.0f}%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    dlabel = fp.METRIC_INFO["J_matrix"][1]
    # Beside the axes (under the legend) so it never covers a trial.
    ax.text(1.02, 0.0, f"2-D SVC\naccuracy {entry['cv_accuracy']:.2f}\n{dlabel} {entry['J']:.1f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=style["legend_fontsize"])


def make_pair_panels(data_dir, out_dir, style_name, formats):
    data = _load_pickle(data_dir, "pair_data.pkl")
    if data is None:
        return
    style = fp.STYLES[style_name]
    with plt.rc_context(fp.rc_params(style)):
        for key in PAIR_TITLES:
            if key not in data:
                print(f"[pairs] {key} not in pair_data.pkl; skipping")
                continue
            entry = data[key]
            fig, ax = plt.subplots(figsize=style["panel_figsize"]["pca_pair"], layout="constrained")
            _draw_pair(ax, entry, entry["stim"], style)
            ax.set_title(PAIR_TITLES[key])
            fp.save_figure(fig, out_dir, f"supp_pair_{key}_{style_name}", style, formats)


# ----------------------------------------------------------------------------
# Example heat maps
# ----------------------------------------------------------------------------
def _source_label(key):
    if key == "raw":
        return "raw"
    return "Primary → " + fp.REGION_SHORT[key.split("_", 1)[1]]


def _draw_heatmap(fig, ax, matrix, stim, pairs, style, metric_key):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")  # diagonal is never computed
    m = np.array(matrix, dtype=float)
    np.fill_diagonal(m, np.nan)
    im = ax.imshow(m, cmap=cmap, vmin=np.nanmin(m), vmax=np.nanmax(m), interpolation="nearest")
    n = m.shape[0]
    outline = dict(fill=False, edgecolor="white", linewidth=style["line_width"] * 1.2)

    if stim == "naturalSound":
        e = fp.EXEMPLARS_PER_CATEGORY
        for c in range(n // e):
            ax.add_patch(Rectangle((e * c - 0.5, e * c - 0.5), e, e, **outline))
        centers = [e * c + (e - 1) / 2 for c in range(n // e)]
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(studyparams.SOUND_CATEGORIES, rotation=45, ha="right")
        ax.set_yticklabels(studyparams.SOUND_CATEGORIES)
        ax.set_xlabel("outlined: within-source pairs")
        ax.set_ylabel("Sound source")
    else:
        for i in range(n - 1):
            ax.add_patch(Rectangle((i + 0.5, i - 0.5), 1, 1, **outline))
            ax.add_patch(Rectangle((i - 0.5, i + 0.5), 1, 1, **outline))
        labels = _value_labels(stim, n)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("outlined: near-diagonal (adjacent) pairs")
        ax.set_ylabel("AM rate (Hz)" if stim == "AM" else "Frequency (kHz)")

    # Code for adding circles showing what pairs in the heatmaps are compared for the boundary decisions made below
    # for a, b in pairs:  # example pairs from the PC-space panels
    #     ax.plot([b, a], [a, b], linestyle="", marker="o", markerfacecolor="none",
    #             markeredgecolor=fp.PAIR_COLORS[0], markeredgewidth=style["line_width"] * 1.5,
    #             markersize=style["tick_fontsize"] * 0.7)
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(fp.METRIC_INFO[metric_key][1])


def make_heatmap_panels(data_dir, out_dir, style_name, formats, metric_key="J_matrix",
                        sources=None):
    data = _load_pickle(data_dir, "heatmap_data.pkl")
    if data is None:
        return
    style = fp.STYLES[style_name]
    meta = data["meta"]
    tag = fp.METRIC_INFO[metric_key][0]
    with plt.rc_context(fp.rc_params(style)):
        for stim in fp.STIM_TYPES:
            entry = data.get(stim)
            if entry is None:
                continue
            for key, source in entry["sources"].items():
                if sources is not None and key not in sources:
                    continue
                if metric_key not in source:
                    print(f"[heatmaps] {stim}/{key}: no {metric_key}; skipping")
                    continue
                fig, ax = plt.subplots(figsize=style["panel_figsize"]["heatmap"], layout="constrained")
                _draw_heatmap(fig, ax, source[metric_key], stim, entry["pairs"].values(), style, metric_key)
                ax.set_title(f"{fp.STIM_LABELS[stim]}, {_source_label(key)}")
                name = key.replace(" auditory area", "")
                fp.save_figure(fig, out_dir, f"supp_heatmap_{stim}_{name}_{tag}_{style_name}",
                               style, formats)
    print(f"[heatmaps] example session {meta['session'].replace('|', ' ')}, "
          f"region {meta['region']}, window {meta['window']}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(style_name="paper", formats=fp.DEFAULT_FORMATS, steps=("pairs", "heatmaps"),
         metric_key="J_matrix", heatmap_sources=None, data_dir=None, out_dir=None,
         stim_label_dir=None):
    global _label_dir
    _label_dir = stim_label_dir
    data_dir = data_dir or fp.figure_data_dir(FIG)
    out_dir = out_dir or fp.figure_output_dir(FIG, style_name)
    if "pairs" in steps:
        make_pair_panels(data_dir, out_dir, style_name, formats)
    if "heatmaps" in steps:
        make_heatmap_panels(data_dir, out_dir, style_name, formats, metric_key, heatmap_sources)


if __name__ == "__main__":
    main(style_name="paper", formats=("png", "svg"), metric_key="J_matrix")
    # main(style_name="paper", formats=("png", "svg"), metric_key="accuracy_matrix")
    # main(heatmap_sources=("raw",))   # raw Primary only, no CCA heat maps
