"""
Figure 5 panels: how the communication-subspace null is built. Reads the file
written by fig5_data.py (real-data version); the cartoon version needs no data.

Real-data panels (one example session; Primary -> Dorsal vs Primary -> Ventral)
  A  Primary activity in its PC1/PC2 plane with the footprints of the two actual
     subspaces (U: Primary->Dorsal, V: Primary->Ventral)
  B  Null: vectors drawn from N(0, Sigma), orthonormalized -> random subspaces
  C  Cosines of the principal angles, actual vs the null band
  D  Overlap R for the example session: Sigma-shaped null, isotropic null, actual
  E  Pooled z-scored null across sessions with each session's actual z

Cartoon panels (schematic, d = 1 so subspaces are lines)
  1  Two CCA subspaces in Primary space and their overlap
  2  Null: sample from Sigma, then orthonormalize
  3  The overlap measure R, repeated per null draw
  4  Actual overlap compared with the null distribution

A "footprint" is the projection of a subspace's unit ball onto the PC1/PC2 plane
(an ellipse, a segment for d = 1). It reaches the unit circle only if the subspace
lies in the plane, so it shows how much of the subspace points along the
high-variance directions. The data are scaled so PC1's 2 SD reaches the circle.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch, Polygon

import figure_params as fp

FIG = "figure5"
COLOR_U = fp.REGION_COLORS["Dorsal auditory area"]     # target1
COLOR_V = fp.REGION_COLORS["Ventral auditory area"]    # target2
NULL_COLOR = "#AAAAAA"
ACTUAL_COLOR = "#C44E52"
ISO_COLOR = "#4D4D4D"
N_HIST_BINS = 30
N_NULL_SHOWN = 4      # null pairs outlined in the real-data null panel


def _load(data_dir):
    path = f"{data_dir}/null_example.pkl"
    if not os.path.exists(path):
        print(f"Missing {path}; run fig5_data.py first. Skipping real-data panels.")
        return None
    return pickle.load(open(path, "rb"))


# ----------------------------------------------------------------------------
# Shared drawing helpers
# ----------------------------------------------------------------------------
def _footprint(A):
    """Closed outline (N x 2) of the unit-ball shadow of a subspace; A is (2, d)."""
    u, s, _ = np.linalg.svd(A, full_matrices=True)
    s2 = np.zeros(2)
    s2[:len(s)] = s
    t = np.linspace(0, 2 * np.pi, 200)
    return ((u * s2) @ np.vstack([np.cos(t), np.sin(t)])).T


def _unit_frame(ax):
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", lw=0.8, color="0.6", zorder=1))
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _cov_ellipse(ax, ex):
    ev = ex["pc_evals"]
    ax.add_patch(Ellipse((0, 0), 2, 2 * np.sqrt(ev[1] / ev[0]), fill=False, ec="0.35", lw=1.2, zorder=2))


def _pc_labels(ax, ex):
    f = ex["pc_var_frac"] * 100
    ax.set_xlabel(f"PC1 ({f[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({f[1]:.0f}%)")


def _shadow(ax, A, color, alpha, lw=1.2, ls="-", fill=True, zorder=3):
    pts = _footprint(A)
    if fill:
        ax.add_patch(Polygon(pts, closed=True, fc=color, ec=color, alpha=alpha, lw=0, zorder=zorder))
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, alpha=1.0 if fill else alpha, zorder=zorder + 1)


def _legend(ax, handles, outside, style):
    if handles and outside:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                  fontsize=style["legend_fontsize"])


# ----------------------------------------------------------------------------
# Real-data panels
# ----------------------------------------------------------------------------
def draw_real_subspaces(ax, ex, style, legend=True):
    scale = 2 * np.sqrt(ex["pc_evals"][0])
    ax.scatter(ex["scores"][:, 0] / scale, ex["scores"][:, 1] / scale, s=style["pca_marker_size"] * 0.5,
               color="0.55", alpha=0.3, linewidths=0, zorder=1)
    _unit_frame(ax)
    _cov_ellipse(ax, ex)
    _shadow(ax, ex["shadow_u"], COLOR_U, 0.35)
    _shadow(ax, ex["shadow_v"], COLOR_V, 0.35)
    _pc_labels(ax, ex)
    ax.set_title("Actual subspaces")
    return [Patch(fc=COLOR_U, alpha=0.5, label=f"U: Primary→Dorsal (d={ex['d_u']})"),
            Patch(fc=COLOR_V, alpha=0.5, label=f"V: Primary→Ventral (d={ex['d_v']})"),
            Line2D([0], [0], color="0.35", lw=1.2, label="Primary activity (2 SD)")]


def draw_real_null(ax, ex, style, legend=True):
    scale = 2 * np.sqrt(ex["pc_evals"][0])
    _unit_frame(ax)
    _cov_ellipse(ax, ex)
    shown = ex["null_illustrated"][:N_NULL_SHOWN]
    for item in shown:  # outlines only, so overlapping draws stay readable
        _shadow(ax, item["shadow_u"], COLOR_U, 0.5, lw=0.8, fill=False, zorder=2)
        _shadow(ax, item["shadow_v"], COLOR_V, 0.5, lw=0.8, fill=False, zorder=2)
    first = ex["null_illustrated"][0]
    ax.scatter(first["samples_u"][:, 0] / scale, first["samples_u"][:, 1] / scale, s=style["pca_marker_size"] * 2,
               color=COLOR_U, edgecolors="white", linewidths=0.5, zorder=6)
    ax.scatter(first["samples_v"][:, 0] / scale, first["samples_v"][:, 1] / scale, s=style["pca_marker_size"] * 2,
               color=COLOR_V, edgecolors="white", linewidths=0.5, zorder=6)
    _shadow(ax, ex["shadow_u"], COLOR_U, 0.8, lw=0.9, ls="--", fill=False, zorder=7)
    _shadow(ax, ex["shadow_v"], COLOR_V, 0.8, lw=0.9, ls="--", fill=False, zorder=7)
    _pc_labels(ax, ex)
    ax.set_title("Null: draws from Σ")
    return [Line2D([0], [0], marker="o", ls="", color="0.4", label="vectors drawn from N(0, Σ)\n(one draw, before QR)"),
            Line2D([0], [0], color="0.5", lw=0.8, label=f"{N_NULL_SHOWN} null subspace pairs\n(after QR)"),
            Line2D([0], [0], color="0.4", ls="--", lw=0.9, label="actual (from panel A)")]


def draw_real_angles(ax, ex, style, legend=True):
    k = np.arange(1, len(ex["cos_actual"]) + 1)
    ax.fill_between(k, ex["cos_null_lo"], ex["cos_null_hi"], color=NULL_COLOR, alpha=0.5, lw=0)
    ax.plot(k, ex["cos_null_mean"], color="0.4", lw=style["line_width"], marker="o", ms=3)
    ax.plot(k, ex["cos_actual"], color=ACTUAL_COLOR, lw=style["line_width"] * 1.5, marker="o", ms=4)
    ax.set_xticks(k)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Principal angle (ordered)")
    ax.set_ylabel("cos θ")
    ax.set_title(f"R = mean cos²θ = {ex['R_actual']:.2f}")
    return [Line2D([0], [0], color=ACTUAL_COLOR, lw=2, label="actual"),
            Patch(fc=NULL_COLOR, alpha=0.5, label="null mean, 5-95%")]


def draw_real_hist(ax, ex, style, legend=True):
    lo = min(ex["null_R"].min(), ex["R_null_isotropic"].min(), ex["R_actual"])
    hi = max(ex["null_R"].max(), ex["R_null_isotropic"].max(), ex["R_actual"])
    bins = np.linspace(lo, hi, 45)
    ax.hist(ex["null_R"], bins=bins, color=NULL_COLOR, edgecolor="white", density=True)
    ax.hist(ex["R_null_isotropic"], bins=bins, histtype="step", color=ISO_COLOR, ls="--",
            lw=style["line_width"], density=True)
    ax.axvline(ex["R_actual"], color=ACTUAL_COLOR, lw=style["line_width"] * 1.8)
    ax.set_xlabel("Subspace overlap R")
    ax.set_ylabel("Density")
    ax.set_title(f"Example session (z = {ex['z_actual']:.1f})")
    return [Patch(fc=NULL_COLOR, label="null, N(0, Σ)"),
            Line2D([0], [0], color=ISO_COLOR, ls="--", label="null, isotropic"),
            Line2D([0], [0], color=ACTUAL_COLOR, lw=2, label="actual")]


def draw_real_pooled(ax, data, style, legend=True):
    pooled = data["pooled"]
    z_null = np.concatenate([p["z_null"] for p in pooled])
    z_act = np.array([p["z_actual"] for p in pooled])
    ax.hist(z_null, bins=N_HIST_BINS, color=NULL_COLOR, edgecolor="white", alpha=0.85, density=True)
    ax.axvline(0, color="black", ls="--", lw=1.0, alpha=0.6)
    for z in z_act:
        ax.axvline(z, color=ACTUAL_COLOR, lw=style["line_width"] * 1.5, alpha=0.7)
    q = np.percentile(z_act, [25, 75])
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.set_title(f"All sessions (n = {len(pooled)})")
    return [Patch(fc=NULL_COLOR, label="pooled null"),
            Line2D([0], [0], color=ACTUAL_COLOR, lw=2,
                   label=f"actual, each session\nmedian z = {np.median(z_act):.2f}\n[{q[0]:.2f}, {q[1]:.2f}]")]


REAL_PANELS = [("A_subspaces", draw_real_subspaces), ("B_null", draw_real_null),
               ("C_angles", draw_real_angles), ("D_example_hist", draw_real_hist),
               ("E_pooled", draw_real_pooled)]


def make_real_panels(data_dir, out_dir, style_name, formats):
    data = _load(data_dir)
    if data is None:
        return
    style = fp.STYLES[style_name]
    ex = data["example"]
    with plt.rc_context(fp.rc_params(style)):
        for name, draw in REAL_PANELS:
            fig, ax = plt.subplots(figsize=style["panel_figsize"]["method_panel"], layout="constrained")
            handles = draw(ax, data if name.startswith("E") else ex, style)
            _legend(ax, handles, True, style)
            fp.save_figure(fig, out_dir, f"fig5_real_{name}_{style_name}", style, formats)

        fig, axes = plt.subplots(1, len(REAL_PANELS), figsize=style["panel_figsize"]["method_strip_real"],
                                 layout="constrained")
        for ax, (name, draw) in zip(axes, REAL_PANELS):
            draw(ax, data if name.startswith("E") else ex, style)
        fp.save_figure(fig, out_dir, f"fig5_real_strip_{style_name}", style, formats)
    c = data["condition"]
    print(f"[fig5] real-data panels: {c['target1']} vs {c['target2']}, {c['stimulus']}, "
          f"{c['response_range']}; example session {ex['session']}")


# ----------------------------------------------------------------------------
# Cartoon panels (no data)
# ----------------------------------------------------------------------------
_TOY_ANGLE = 28           # degrees; tilt of the Primary activity ellipse
_TOY_AB = (1.0, 0.38)     # semi-axes of the 2 SD ellipse
_TH_U, _TH_V = 22, 66     # degrees; the two actual subspace directions


def _unit_dir(deg):
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])


def _toy_frame(ax, title):
    ax.add_patch(Ellipse((0, 0), 2 * _TOY_AB[0], 2 * _TOY_AB[1], angle=_TOY_ANGLE, fill=False, ec="0.35", lw=1.4, zorder=2))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def _arrow(ax, vec, color, lw=2.2, alpha=1.0, zorder=5, length=1.0):
    ax.add_patch(FancyArrowPatch((0, 0), tuple(np.asarray(vec) * length), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, alpha=alpha, zorder=zorder, shrinkA=0, shrinkB=0))


def draw_cartoon_subspaces(ax, style):
    _toy_frame(ax, "Actual subspaces")
    u, v = _unit_dir(_TH_U), _unit_dir(_TH_V)
    _arrow(ax, u, COLOR_U, length=0.95)
    _arrow(ax, v, COLOR_V, length=0.95)
    proj = (u @ v) * u * 0.95
    ax.plot([proj[0], v[0] * 0.95], [proj[1], v[1] * 0.95], color="0.5", ls=":", lw=1.2, zorder=4)
    t = np.deg2rad(np.linspace(_TH_U, _TH_V, 30))
    ax.plot(0.3 * np.cos(t), 0.3 * np.sin(t), color="0.2", lw=1.0)
    ax.text(0.40 * np.cos(np.deg2rad((_TH_U + _TH_V) / 2)), 0.40 * np.sin(np.deg2rad((_TH_U + _TH_V) / 2)),
            "θ", ha="center", va="center", fontsize=style["label_fontsize"])
    ax.text(*(u * 1.02 + [0.04, -0.16]), "U\nPrimary→Dorsal", color=COLOR_U, ha="left", va="top", fontsize=style["legend_fontsize"])
    ax.text(v[0] * 0.95 - 0.12, v[1] * 0.95, "V\nPrimary→Ventral", color=COLOR_V, ha="right", va="center", fontsize=style["legend_fontsize"])
    ax.text(-0.95, -0.95, "ellipse: Primary\nactivity (Σ)", color="0.35", ha="left", va="bottom", fontsize=style["legend_fontsize"])


def draw_cartoon_null(ax, style):
    _toy_frame(ax, "Null subspaces")
    rng = np.random.default_rng(3)
    ang = np.deg2rad(_TOY_ANGLE)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = (rng.standard_normal((70, 2)) * (np.array(_TOY_AB) / 2)) @ R.T
    ax.scatter(pts[:, 0], pts[:, 1], s=style["pca_marker_size"] * 0.8, color="0.6", alpha=0.6, linewidths=0, zorder=3)
    for k in range(3):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        for vec, color in ((a, COLOR_U), (b, COLOR_V)):
            _arrow(ax, vec / np.linalg.norm(vec), color, lw=1.4, alpha=0.45, zorder=4, length=0.9)
    ax.text(0.0, -1.1, "sample from Σ, then QR:\npairs lean along high-variance\ndirections", color="0.3", ha="center",
            va="top", fontsize=style["legend_fontsize"])


def draw_cartoon_metric(ax, style):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Overlap R")
    ax.text(0.5, 0.78, r"$R=\dfrac{\|U^{\top}V\|_F^{2}}{\min(\|U\|_F^{2},\,\|V\|_F^{2})}$", ha="center",
            va="center", fontsize=style["label_fontsize"] * 1.1)
    ax.text(0.5, 0.52, "= mean cos²θ over the principal angles", ha="center", va="center", fontsize=style["legend_fontsize"])
    ax.text(0.5, 0.40, "R = 1: same subspace      R = 0: orthogonal", ha="center", va="center",
            fontsize=style["legend_fontsize"], color="0.3")
    ax.add_patch(FancyArrowPatch((0.5, 0.30), (0.5, 0.17), arrowstyle="-|>", mutation_scale=12, color="0.3", lw=1.4))
    ax.text(0.5, 0.08, "repeat for each null draw\n→ distribution of R", ha="center", va="center",
            fontsize=style["legend_fontsize"])


def draw_cartoon_compare(ax, style):
    x = np.linspace(-4, 4, 25)
    heights = np.exp(-x ** 2 / 2)
    ax.bar(x, heights, width=x[1] - x[0], color=NULL_COLOR, edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.axvline(-3.4, color=ACTUAL_COLOR, lw=style["line_width"] * 1.8)
    ax.text(-3.4, 1.12, "actual", color=ACTUAL_COLOR, ha="center", va="bottom", fontsize=style["legend_fontsize"])
    ax.text(0, 1.12, "null", color="0.3", ha="center", va="bottom", fontsize=style["legend_fontsize"])
    ax.annotate("", xy=(-3.4, 0.5), xytext=(-0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.2))
    ax.text(-1.9, 0.56, "z", ha="center", va="bottom", fontsize=style["label_fontsize"])
    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("overlap R, in null SDs")
    ax.set_ylabel("Density")
    ax.set_title("Actual vs null")
    ax.spines[["top", "right"]].set_visible(False)


CARTOON_PANELS = [("1_subspaces", draw_cartoon_subspaces), ("2_null", draw_cartoon_null),
                  ("3_metric", draw_cartoon_metric), ("4_compare", draw_cartoon_compare)]


def make_cartoon_panels(out_dir, style_name, formats):
    style = fp.STYLES[style_name]
    with plt.rc_context(fp.rc_params(style)):
        for name, draw in CARTOON_PANELS:
            fig, ax = plt.subplots(figsize=style["panel_figsize"]["method_panel"], layout="constrained")
            draw(ax, style)
            fp.save_figure(fig, out_dir, f"fig5_cartoon_{name}_{style_name}", style, formats)
        fig, axes = plt.subplots(1, len(CARTOON_PANELS), figsize=style["panel_figsize"]["method_strip_cartoon"],
                                 layout="constrained")
        for ax, (name, draw) in zip(axes, CARTOON_PANELS):
            draw(ax, style)
        fp.save_figure(fig, out_dir, f"fig5_cartoon_strip_{style_name}", style, formats)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(style_name="paper", formats=fp.DEFAULT_FORMATS, steps=("real", "cartoon"),
         data_dir=None, out_dir=None):
    data_dir = data_dir or fp.figure_data_dir(FIG)
    out_dir = out_dir or fp.figure_output_dir(FIG, style_name)
    if "real" in steps:
        make_real_panels(data_dir, out_dir, style_name, formats)
    if "cartoon" in steps:
        make_cartoon_panels(out_dir, style_name, formats)


if __name__ == "__main__":
    main(style_name="paper", formats=("png", "svg"))
    # main(style_name="poster", formats=("svg",))
    # main(steps=("cartoon",))   # cartoon only; needs no data
