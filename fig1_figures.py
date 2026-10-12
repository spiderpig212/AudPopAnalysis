"""
Figure 1 panels. Reads the files written by fig1_data.py; nothing is calculated
here beyond layout and color mapping.

Panels produced (each saved as its own file so they can be assembled externally):
  2   Natural-sound spectrogram (+ a contact sheet of all 20 to choose from)
  3   AM-noise and pure-tone spectrograms
  4   Sorted spike rasters for the cells in figure_params.RASTER_CELLS, plus
      candidate sheets (per stimulus type, rows = region) to choose them from
  5.1 PC space, all regions fit together, natural sounds
  5.2 PC space, Primary auditory area only, natural sounds
  5.3 PC space, every region x every stimulus type (3x3 grid)

Panels 1 and the sound list in 2 are supplied from lab resources.
Size/text come from figure_params.STYLES ("paper" / "poster").
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import figure_params as fp
import studyparams

FIG = "figure1"
_FREQ_TICKS_KHZ = {"naturalSound": [1, 2, 5, 10, 20], "AM": [2, 4, 8, 16, 32, 64],
                   "pureTones": [2, 4, 8, 16, 32, 64]}


def _load_pickle(data_dir, name):
    path = f"{data_dir}/{name}"
    if not os.path.exists(path):
        print(f"Missing {path}; run fig1_data.py first. Skipping.")
        return None
    return pickle.load(open(path, "rb"))


def _load_spec(data_dir, stim):
    path = f"{data_dir}/spectrograms_{stim}.npz"
    if not os.path.exists(path):
        print(f"Missing {path}; run fig1_data.py first. Skipping.")
        return None
    return dict(np.load(path, allow_pickle=True))


# ----------------------------------------------------------------------------
# Panels 2 and 3: spectrograms
# ----------------------------------------------------------------------------
def _draw_spectrogram(ax, spec, index, stim, style, colorbar=True, small=False):
    im = ax.pcolormesh(spec["t"], spec["f"] / 1000, spec["S_db"][index],
                       cmap=fp.SPEC_CMAP, vmin=-fp.SPEC_DYNAMIC_RANGE_DB, vmax=0,
                       shading="auto", rasterized=True)
    ax.set_yscale("log")
    lo, hi = spec["f"].min() / 1000, spec["f"].max() / 1000
    ticks = [t for t in _FREQ_TICKS_KHZ[stim] if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0, fp.STIM_DURATION[stim])
    ax.set_ylim(lo, hi)
    if not small:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("dB")
    return im


def _natural_label(i):
    cat, ex = divmod(i, fp.EXEMPLARS_PER_CATEGORY)
    return f"{studyparams.SOUND_CATEGORIES[cat]} {ex + 1}"


def make_spectrogram_panels(data_dir, out_dir, style_name, formats):
    style = fp.STYLES[style_name]
    size = style["panel_figsize"]["spectrogram"]
    cat, ex = fp.EXAMPLE_NATURAL_SOUND
    jobs = [("naturalSound", cat * fp.EXEMPLARS_PER_CATEGORY + ex, "fig1_panel2_natural_spectrogram"),
            ("AM", None, "fig1_panel3_AM_spectrogram"),
            ("pureTones", None, "fig1_panel3_pureTone_spectrogram")]
    targets = {"AM": fp.EXAMPLE_AM_RATE, "pureTones": fp.EXAMPLE_PT_FREQ}

    with plt.rc_context(fp.rc_params(style)):
        for stim, index, name in jobs:
            spec = _load_spec(data_dir, stim)
            if spec is None:
                continue
            if index is None:
                index = int(np.argmin(np.abs(spec["labels"].astype(float) - targets[stim])))
            fig, ax = plt.subplots(figsize=size, layout="constrained")
            _draw_spectrogram(ax, spec, index, stim, style)
            if stim == "naturalSound":
                title = _natural_label(index)
            elif stim == "AM":
                title = f"AM noise, {spec['labels'][index]:.1f} Hz modulation"
            else:
                title = f"Pure tone, {spec['labels'][index] / 1000:.1f} kHz"
            ax.set_title(title)
            fp.save_figure(fig, out_dir, f"{name}_{style_name}", style, formats)

        # Contact sheet of all 20 natural sounds, to choose the example from.
        spec = _load_spec(data_dir, "naturalSound")
        if spec is not None:
            fig, axes = plt.subplots(5, 4, figsize=style["panel_figsize"]["raster_sheet"],
                                     layout="constrained", sharex=True, sharey=True)
            for i, ax in enumerate(axes.ravel()):
                _draw_spectrogram(ax, spec, i, "naturalSound", style, colorbar=False, small=True)
                ax.set_title(f"{i}: {_natural_label(i)}", fontsize=style["tick_fontsize"])
            fp.save_figure(fig, out_dir, f"fig1_panel2_natural_spectrogram_sheet_{style_name}",
                           style, formats)


# ----------------------------------------------------------------------------
# Panel 4: rasters
# ----------------------------------------------------------------------------
def _draw_raster(ax, cell, stim, style, label_axes=True):
    n = len(cell["uniq_stims"])
    colors = fp.stim_point_colors(stim, n)
    ax.axvspan(0, fp.STIM_DURATION[stim], color="0.92", zorder=0, lw=0)
    ax.scatter(cell["spike_times"], cell["trial_rank"], s=style["raster_marker_size"],
               marker="|", linewidths=style["line_width"] * 0.6, color="black", zorder=2)

    edges = np.concatenate([[0], np.cumsum(cell["trials_per_stim"])])
    for k in range(n):
        ax.add_patch(Rectangle((1.015, edges[k]), 0.035, edges[k + 1] - edges[k],
                               transform=ax.get_yaxis_transform(), clip_on=False,
                               color=colors[k], lw=0))
    ax.set_xlim(*fp.RASTER_XLIM[stim])
    ax.set_ylim(0, cell["n_trials"])
    ax.set_yticks([0, cell["n_trials"]])
    if label_axes:
        ax.set_xlabel("Time from sound onset (s)")
        ax.set_ylabel("Trials (sorted by stimulus)")


def make_raster_panels(data_dir, out_dir, style_name, formats):
    data = _load_pickle(data_dir, "raster_data.pkl")
    if data is None:
        return
    style = fp.STYLES[style_name]
    with plt.rc_context(fp.rc_params(style)):
        for stim in fp.STIM_TYPES:
            cells = data.get(stim, [])
            chosen = fp.RASTER_CELLS.get(stim)

            # Candidate sheet: rows = region, columns = ranked candidates.
            n_cols = max([sum(c["region"] == r and c.get("candidate") for c in cells)
                          for r in fp.REGIONS] + [1])
            fig, axes = plt.subplots(len(fp.REGIONS), n_cols, squeeze=False,
                                     figsize=style["panel_figsize"]["raster_sheet"],
                                     layout="constrained")
            for i, region in enumerate(fp.REGIONS):
                cands = [c for c in cells if c["region"] == region and c.get("candidate")]
                for j in range(n_cols):
                    ax = axes[i][j]
                    if j >= len(cands):
                        ax.axis("off")
                        continue
                    c = cands[j]
                    _draw_raster(ax, c, stim, style, label_axes=(i == len(fp.REGIONS) - 1 and j == 0))
                    ax.set_title(f"row {c['row']}  {c['mouse']} {c['date']}\n"
                                 f"score {c['score']:.2f}, {c['rate']:.1f} Hz",
                                 fontsize=style["tick_fontsize"],
                                 color=fp.REGION_COLORS[region])
                    if j == 0:
                        ax.annotate(fp.REGION_SHORT[region], xy=(-0.3, 0.5), rotation=90,
                                    xycoords="axes fraction", ha="center", va="center",
                                    fontsize=style["label_fontsize"], color=fp.REGION_COLORS[region])
            fig.suptitle(f"{fp.STIM_LABELS[stim]}: raster candidates", fontsize=style["title_fontsize"])
            fp.save_figure(fig, out_dir, f"fig1_panel4_raster_candidates_{stim}_{style_name}",
                           style, formats)

            if chosen is None:
                print(f"[rasters] no cell chosen for {stim}: set figure_params.RASTER_CELLS")
                continue
            cell = next((c for c in cells if c["row"] == chosen), None)
            if cell is None:
                print(f"[rasters] row {chosen} for {stim} is not in raster_data.pkl; "
                      f"rerun fig1_data.py rasters")
                continue
            fig, ax = plt.subplots(figsize=style["panel_figsize"]["raster"], layout="constrained")
            _draw_raster(ax, cell, stim, style)
            ax.set_title(f"{fp.STIM_LABELS[stim]}", color="black")
            fp.save_figure(fig, out_dir, f"fig1_panel4_raster_{stim}_{style_name}", style, formats)


# ----------------------------------------------------------------------------
# Panel 5: PC space
# ----------------------------------------------------------------------------
def _draw_pca(ax, entry, stim, style, show_labels=True):
    n_stim = int(entry["stim_idx"].max()) + 1
    colors = fp.stim_point_colors(stim, n_stim)
    order = np.random.default_rng(0).permutation(len(entry["stim_idx"]))  # avoid draw-order bias
    ax.scatter(entry["scores"][order, 0], entry["scores"][order, 1],
               c=[colors[k] for k in entry["stim_idx"][order]],
               s=style["pca_marker_size"], linewidths=0, alpha=0.85)
    evr = entry["evr"]
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.0f}%)")
    ax.set_xticks([])
    ax.set_yticks([])


def _natural_legend_handles(style):
    return [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   color=fp.NATURAL_CATEGORY_COLORS[i], label=name)
            for i, name in enumerate(studyparams.SOUND_CATEGORIES)]


def _stim_endpoint_labels(stim, data_dir):
    spec = _load_spec(data_dir, stim)
    if spec is None:
        return "low", "high"
    lo, hi = float(spec["labels"][0]), float(spec["labels"][-1])
    if stim == "AM":
        return f"{lo:.0f} Hz", f"{hi:.0f} Hz"
    return f"{lo / 1000:.1f} kHz", f"{hi / 1000:.1f} kHz"


def make_pca_panels(data_dir, out_dir, style_name, formats):
    data = _load_pickle(data_dir, "pca_data.pkl")
    if data is None:
        return
    style = fp.STYLES[style_name]
    meta = data["meta"]
    session_note = meta["session"].replace("|", " ")

    with plt.rc_context(fp.rc_params(style)):
        # 5.1 and 5.2: natural sounds, all regions together / Primary only.
        for key, base, title in [
                ("all_regions", "fig1_panel5_1_pca_all_regions_naturalSound", "All regions"),
                ("Primary auditory area", "fig1_panel5_2_pca_primary_naturalSound", "Primary auditory area")]:
            entry = data["naturalSound"][key]
            fig, ax = plt.subplots(figsize=style["panel_figsize"]["pca"], layout="constrained")
            _draw_pca(ax, entry, "naturalSound", style)
            ax.set_title(f"{title} ({entry['n_neurons']} neurons)")
            ax.legend(handles=_natural_legend_handles(style), loc="center left",
                      bbox_to_anchor=(1.02, 0.5), frameon=False)
            fp.save_figure(fig, out_dir, f"{base}_{style_name}", style, formats)

        # 5.3: regions (rows) x stimulus types (columns).
        columns = ["AM", "pureTones", "naturalSound"]
        fig, axes = plt.subplots(len(fp.REGIONS), len(columns), squeeze=False,
                                 figsize=style["panel_figsize"]["pca_grid"], layout="constrained")
        for i, region in enumerate(fp.REGIONS):
            for j, stim in enumerate(columns):
                ax = axes[i][j]
                _draw_pca(ax, data[stim][region], stim, style)
                if i == 0:
                    ax.set_title(fp.STIM_LABELS[stim])
                if j == 0:
                    ax.annotate(fp.REGION_SHORT[region], xy=(-0.3, 0.5), rotation=90,
                                xycoords="axes fraction", ha="center", va="center",
                                fontsize=style["label_fontsize"], color=fp.REGION_COLORS[region])
        for j, stim in enumerate(columns):
            if stim == "naturalSound":
                axes[1][j].legend(handles=_natural_legend_handles(style), loc="center left",
                                  bbox_to_anchor=(1.02, 0.5), frameon=False)
            else:
                lo, hi = _stim_endpoint_labels(stim, data_dir)
                sm = ScalarMappable(norm=Normalize(0, 1), cmap=fp.SEQUENTIAL_CMAP)
                cbar = fig.colorbar(sm, ax=axes[:, j], orientation="horizontal",
                                    location="bottom", shrink=0.8, aspect=25, pad=0.03,
                                    ticks=[0, 1])
                cbar.ax.set_xticklabels([lo, hi])
                cbar.set_label("AM rate" if stim == "AM" else "Tone frequency")
        fp.save_figure(fig, out_dir, f"fig1_panel5_3_pca_grid_{style_name}", style, formats)
    print(f"[pca] panels use session {session_note}, window {meta['window']}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(style_name="paper", formats=fp.DEFAULT_FORMATS,
         steps=("spectrograms", "rasters", "pca"), data_dir=None, out_dir=None):
    data_dir = data_dir or fp.figure_data_dir(FIG)
    out_dir = out_dir or fp.figure_output_dir(FIG, style_name)
    if "spectrograms" in steps:
        make_spectrogram_panels(data_dir, out_dir, style_name, formats)
    if "rasters" in steps:
        make_raster_panels(data_dir, out_dir, style_name, formats)
    if "pca" in steps:
        make_pca_panels(data_dir, out_dir, style_name, formats)


if __name__ == "__main__":
    main(style_name="paper", formats=("png", "svg"))
    # main(style_name="poster", formats=("svg",))
