"""
Centralised figure formatting and example selections for the final figures.

Everything that controls how a figure LOOKS (sizes, fonts, colors, file
formats) or WHICH example is shown (sounds, session, raster cells) lives here, so
the data scripts and figure scripts for every figure stay consistent. Edit
PAPER_STYLE / POSTER_STYLE to retarget all figures at once.

Named figure_params.py (not studyparams.py) on purpose: analysis_class.py does
`import studyparams` for STUDY_NAME / SOUND_CATEGORIES, and a studyparams.py in
this repo would shadow the jarapubs/2025acpop one and break it.
"""
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from jaratoolbox import settings
import studyparams

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SOUND_DIR = "/Users/matt/Desktop/Research/Murray/data/soundFiles"
NATURAL_SOUND_DIR = f"{SOUND_DIR}/jaramillo_lab_natural_sounds"
SYNTHETIC_SOUND_DIR = f"{SOUND_DIR}/jaramillo_lab_synthetic_sounds"

FIGDATA_ROOT = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)


def figure_data_dir(figure_name):
    """Where a figure's calculated data is stored (written by the data script)."""
    return f"{FIGDATA_ROOT}/final_figures/{figure_name}/data"


def figure_output_dir(figure_name, style_name):
    """Where a figure's rendered panels are written."""
    return f"{FIGDATA_ROOT}/final_figures/{figure_name}/{style_name}"


# ----------------------------------------------------------------------------
# Shared analysis choices
# ----------------------------------------------------------------------------
MAIN_RESPONSE_RANGE = "sustained"
DEFAULT_FORMATS = ("png",)  # any matplotlib extension: "png", "svg", "pdf"

REGIONS = ["Primary auditory area", "Ventral auditory area", "Dorsal auditory area"]
REGION_SHORT = {"Primary auditory area": "Primary",
                "Ventral auditory area": "Ventral",
                "Dorsal auditory area": "Dorsal"}
# RSA matrix field -> (tag used in output filenames, axis / colorbar label).
# "J_matrix" and "dprime_matrix" are separability metrics computed from the
# decoder's projection axis (labelled D' throughout); "accuracy_matrix" is the
# binary decoder's own cross-validated classification accuracy.
METRIC_INFO = {
    "J_matrix": ("J", "D′"),
    "dprime_matrix": ("dprime", "D′"),
    "accuracy_matrix": ("accuracy", "Decoding accuracy"),
}

STIM_TYPES = ["naturalSound", "AM", "pureTones"]
STIM_LABELS = {"naturalSound": "Natural sounds", "AM": "AM noise",
               "pureTones": "Pure tones"}

# Presented sound durations in seconds (from the stimulus periods in
# analysis_class.FiringRateAnalysis.stim_info).
STIM_DURATION = {"naturalSound": 4.0, "AM": 0.5, "pureTones": 0.1}

# ----------------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------------
# Matches GROUP_FILL_COLORS in d_prime_RSA_visualization.py so a region keeps
# its color across every figure.
REGION_COLORS = {"Primary auditory area": "#4C72B0",
                 "Ventral auditory area": "#DD8452",
                 "Dorsal auditory area": "#55A868"}

# One base color per natural-sound category (order = studyparams.SOUND_CATEGORIES),
# deliberately avoiding the three region colors above.
NATURAL_CATEGORY_COLORS = ["#8172B3", "#C44E52", "#937860", "#008080", "#CCB974"]
EXEMPLARS_PER_CATEGORY = 4
SEQUENTIAL_CMAP = "viridis"  # AM rate / tone frequency, low -> high

# ----------------------------------------------------------------------------
# Style presets -- the single place to change figure size / text size.
# ----------------------------------------------------------------------------
# `figsize` is used by the box-plot figures (final_figures_dprime.py);
# `panel_figsize` sizes each Figure 1 panel type.
POSTER_STYLE = dict(
    figsize=(9, 8),
    title_fontsize=22,
    label_fontsize=20,
    tick_fontsize=16,
    legend_fontsize=15,
    star_fontsize=16,
    box_width=0.6,
    dpi=300,
    line_width=2.0,
    axis_linewidth=1.5,
    pca_marker_size=30,
    raster_marker_size=14,
    panel_figsize=dict(
        spectrogram=(9, 5),
        raster=(8, 7),
        raster_sheet=(18, 14),
        pca=(8, 6.5),
        pca_grid=(16, 14),
        pca_pair=(7, 6),
        heatmap=(8, 7),
    ),
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
    line_width=1.0,
    axis_linewidth=0.8,
    pca_marker_size=8,
    raster_marker_size=6,
    panel_figsize=dict(
        spectrogram=(3.6, 2.3),
        raster=(3.4, 3.0),
        raster_sheet=(10, 8),
        pca=(3.8, 3.0),
        pca_grid=(7.5, 6.8),
        pca_pair=(3.4, 3.0),
        heatmap=(3.6, 3.2),
    ),
)

STYLES = {"poster": POSTER_STYLE, "paper": PAPER_STYLE}


def rc_params(style):
    """matplotlib rc settings for a style; use as `with plt.rc_context(...)`."""
    return {
        "font.size": style["tick_fontsize"],
        "axes.titlesize": style["title_fontsize"],
        "axes.labelsize": style["label_fontsize"],
        "xtick.labelsize": style["tick_fontsize"],
        "ytick.labelsize": style["tick_fontsize"],
        "legend.fontsize": style["legend_fontsize"],
        "axes.linewidth": style["axis_linewidth"],
        "lines.linewidth": style["line_width"],
        "svg.fonttype": "none",  # keep SVG text editable (needs the font installed)
    }


def save_figure(fig, out_dir, base_name, style, formats=DEFAULT_FORMATS):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = f"{out_dir}/{base_name}.{fmt}"
        fig.savefig(path, dpi=style["dpi"], bbox_inches="tight")  # dpi: raster only
        print(f"Saved figure to {path}")
    plt.close(fig)


def stim_point_colors(stim, n_stim):
    """
    One color per stimulus (index into np.unique(stimArray)).

    Natural sounds: category base color, lightened per exemplar (stimuli are
    ordered in blocks of EXEMPLARS_PER_CATEGORY per category). AM / pure tones:
    a sequential colormap from lowest to highest rate / frequency.
    """
    if stim == "naturalSound":
        colors = []
        for i in range(n_stim):
            base = np.array(mcolors.to_rgb(
                NATURAL_CATEGORY_COLORS[i // EXEMPLARS_PER_CATEGORY]))
            lighten = 0.45 *(i % EXEMPLARS_PER_CATEGORY) / (EXEMPLARS_PER_CATEGORY - 1)
            colors.append(tuple(base + (1 - base) * lighten))
        return colors
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    return [cmap(v) for v in np.linspace(0.0, 0.95, n_stim)]


# ----------------------------------------------------------------------------
# Figure 1 -- example selections (edit these to change what is shown)
# ----------------------------------------------------------------------------
# Spectrogram examples (panels 2 and 3).
EXAMPLE_NATURAL_SOUND = (0, 0)   # (category index 0-4, exemplar 0-3); 0 = Frogs
EXAMPLE_AM_RATE = 16.0           # Hz; one of 4, 5.7, 8, 11.3, 16, 22.6, 32, 45.3, 64, 90.5, 128
EXAMPLE_PT_FREQ = 8066           # Hz; one of the 16 tone frequencies

# PC-space example session as (mouse, date). None = auto-pick the session with
# the most neurons in its smallest region across all stimulus types.
EXAMPLE_SESSION = None

# Raster example cells, as row indices into the firing-rate arrays. Leave None
# until you've picked from the candidate sheets; the data script always saves
# these cells too.
RASTER_CELLS = {"naturalSound": 4592, "AM": 1064, "pureTones": 3595}
# AM Cell: feat016 2924094017 17.3 Hz
# Natural sounds: feat019 2024-06-19 (row 4592) (frequency response); feat015 2024-03-20 (row373) (white noise)
# PT: feat018 2024-06-27

# Plotted raster time windows (s from sound onset); the data keeps the full
# analysis window.
RASTER_XLIM = {"naturalSound": (-0.5, 4.5), "AM": (-0.25, 0.75),
               "pureTones": (-0.1, 0.2)}

# Spectrogram display/calculation settings, per stimulus type.
SPEC_PARAMS = {
    "naturalSound": dict(nperseg=1024, overlap=0.75, fmin=500, fmax=22050),
    "AM": dict(nperseg=1024, overlap=0.75, fmin=1000, fmax=64000),
    "pureTones": dict(nperseg=2048, overlap=0.9, fmin=1000, fmax=64000),
}
SPEC_DYNAMIC_RANGE_DB = 60  # color range below each sound's own peak
SPEC_CMAP = "magma"


# ----------------------------------------------------------------------------
# Supporting decoder panels (supp_decoder_data.py / supp_decoder_figures.py)
# ----------------------------------------------------------------------------
# Region whose neurons are decoded in the example pair panels and heat maps.
SUPP_REGION = "Primary auditory area"

# Example stimulus pairs for the PC-space decoder panels, as indices into
# np.unique(stimArray) (so (0, 1) = the two lowest stimuli). None = auto-pick the
# pair whose separability is closest to the median of the entries that feed the
# box plots (near-diagonal for AM / pure tones; within- or between-source for
# natural sounds), using the example session's matrix.
SUPP_PAIRS = {"AM": None, "pureTones": None,
              "naturalSound_within": None, "naturalSound_between": None}

# The two stimuli of a pair; contrasting, colorblind-safe, and distinct from the
# region and natural-category colors.
PAIR_COLORS = ("#D55E00", "#56B4E9")
