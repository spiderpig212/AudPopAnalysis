import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
from statsmodels.stats.multitest import multipletests
from analysis_class import FiringRateAnalysis

fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
SOUND_CATEGORIES = ['Frogs', 'Crickets', 'Streamside', 'Bubbling', 'Bees']
EXEMPLARS_PER_CATEGORY = 4

def get_stim_labels(stimulus, uniqStims, sounds_categories=None, exemplars_per_category=None):
    """
    Build tick labels for a given stimulus type.

    Parameters
    ----------
    stimulus : str
        One of the stimulus types (e.g. "AM", "pureTones", "naturalSounds").
    uniqStims : array-like
        The unique stimulus identifiers from stim_arrays["stimArray"][0, :].
    sounds_categories : list of str, optional
        Ordered category names for natural sounds (e.g. ["Frogs", "Crickets", ...]).
    exemplars_per_category : int, optional
        Number of exemplars per category for natural sounds.

    Returns
    -------
    list of str
        Tick labels aligned with the order of uniqStims.
    """
    if stimulus == "naturalSound":
        if sounds_categories is None or exemplars_per_category is None:
            raise ValueError(
                "sounds_categories and exemplars_per_category are required for naturalSound"
            )
        labels = [
            f"{category}_{exemplar}"
            for category in sounds_categories
            for exemplar in range(exemplars_per_category)
        ]
        # natural sound ids are 0-19, so index into the generated labels
        return [labels[int(s)] for s in uniqStims]

    # AM / pureTones: use the raw stimulus values directly
    return [str(s) for s in uniqStims]


def plot_rsa_heatmaps(rsa_frame, stim_labels=None, save_path=None):
    """
    Plot a grid of RSA heatmaps (averaging across sessions).

    Rows correspond to each response_range and columns correspond to each
    unique (brain_region, target_region) pair.

    Parameters
    ----------
    rsa_frame : pd.DataFrame
        Must contain 'rsa_matrix', 'response_range', 'brain_region',
        'target_region'.
    stim_labels : list of str, optional
        Tick labels for both axes. If None, integer indices are used.
    save_path : str, optional
        If provided, the figure is saved to this path.
    """
    response_ranges = list(rsa_frame["response_range"].unique())
    n_rows = len(response_ranges)

    # Build the ordered list of unique (brain_region, target_region) pairs
    region_pairs = (
        rsa_frame[["brain_region", "target_region"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    region_pairs = list(region_pairs)
    n_cols = len(region_pairs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )

    for ri, response_range in enumerate(response_ranges):
        for ci, (brain_region, target_region) in enumerate(region_pairs):
            ax = axes[ri][ci]
            sub = rsa_frame[
                (rsa_frame["response_range"] == response_range)
                & (rsa_frame["brain_region"] == brain_region)
                & (rsa_frame["target_region"] == target_region)
            ]

            if sub.empty:
                ax.set_axis_off()
                ax.set_title(
                    f"{brain_region}–{target_region} | {response_range}\n(no data)"
                )
                continue

            avg_matrix = np.mean(np.stack(sub["rsa_matrix"].values), axis=0)

            x_label = f"{brain_region}–{target_region} subspace"
            y_label = f"{brain_region}–{target_region} subspace"

            sns.heatmap(
                avg_matrix,
                ax=ax,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                xticklabels=stim_labels if stim_labels is not None else "auto",
                yticklabels=stim_labels if stim_labels is not None else "auto",
                cbar_kws={"label": "Pearson r"},
            )
            ax.set_title(f"RSA: {response_range}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.tick_params(axis="x", rotation=90)
            ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig, axes

def plot_subspace_boxplots(
    rsa_frame,
    stim,
    response_range_order=None,
    save_dir=None,
):
    """
    Grouped boxplots of upper-triangle RSA values for one stimulus.

    The x-axis is response_range; within each response_range the
    (brain_region, target_region) subspaces are stacked horizontally and
    color-coded, with a legend mapping colors to subspaces.

    Parameters
    ----------
    rsa_frame : pd.DataFrame
        Must contain 'rsa_matrix', 'response_range', 'brain_region',
        'target_region'.
    stim : str
        Stimulus type, used for titles / filenames.
    response_range_order : list of str, optional
        Desired ordering of response ranges on the x-axis. Any present in the
        data but not listed here are appended at the end.
    save_dir : str, optional
        Directory to save the figure into. If None, the figure is not saved.

    Returns
    -------
    (fig, ax) or None if there is insufficient data.
    """
    # Resolve response_range ordering against what's actually present
    present_ranges = list(rsa_frame["response_range"].unique())
    if response_range_order is None:
        resp_ranges = present_ranges
    else:
        resp_ranges = [r for r in response_range_order if r in present_ranges]
        for r in present_ranges:
            if r not in resp_ranges:
                resp_ranges.append(r)

    # Consistent subspace ordering + color mapping across the whole figure
    subspace_pairs = list(
        rsa_frame[["brain_region", "target_region"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    subspace_labels = [f"{br}–{tr}" for (br, tr) in subspace_pairs]
    cmap = plt.get_cmap("tab10")
    color_map = {
        lab: cmap(i % 10) for i, lab in enumerate(subspace_labels)
    }

    # Collect upper-triangle values per (response_range, subspace)
    group_values = {}  # (resp_range, label) -> np.ndarray
    for resp_range in resp_ranges:
        for (brain_region, target_region), label in zip(
            subspace_pairs, subspace_labels
        ):
            sub = rsa_frame[
                (rsa_frame["response_range"] == resp_range)
                & (rsa_frame["brain_region"] == brain_region)
                & (rsa_frame["target_region"] == target_region)
            ]
            if sub.empty:
                continue
            avg_matrix = np.mean(np.stack(sub["rsa_matrix"].values), axis=0)
            n = avg_matrix.shape[0]
            vals = avg_matrix[np.triu_indices(n, k=1)]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                group_values[(resp_range, label)] = vals

    if not group_values:
        print(f"No subspace data for {stim}, skipping grouped boxplots.")
        return None

    # ── Plotting ─────────────────────────────────────────────────────────────
    n_x = len(resp_ranges)
    n_sub = len(subspace_labels)
    fig_w = max(6, n_x * max(2.0, 0.9 * n_sub))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    total_width = 0.8
    box_width = total_width / max(n_sub, 1)
    x_centers = np.arange(1, n_x + 1)

    for ri, resp_range in enumerate(resp_ranges):
        for gi, label in enumerate(subspace_labels):
            key = (resp_range, label)
            if key not in group_values:
                continue
            vals = group_values[key]
            offset = (gi - (n_sub - 1) / 2) * box_width
            pos = x_centers[ri] + offset
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_width * 0.9,
                patch_artist=True,
                notch=False,
                medianprops=dict(color="black", linewidth=2),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color_map[label])
                patch.set_alpha(0.7)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(r) for r in resp_ranges], fontsize=10)
    ax.set_xlim(0.5, n_x + 0.5)
    ax.set_xlabel("Response range")
    ax.set_ylabel("RSA upper-triangle values (Pearson r)")
    ax.set_title(
        f"{stim} — Upper-triangle RSA values by response range\n"
        f"(colored by communication subspace)"
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map[lab], alpha=0.7, label=lab)
        for lab in subspace_labels
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        title="Subspace (brain_region–target_region)",
        fontsize=8,
    )

    fig.tight_layout()

    if save_dir is not None:
        box_path = f"{save_dir}/rsa_subspace_boxplots_{stim}.png"
        fig.savefig(box_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {box_path}")

    return fig, ax

def plot_subspace_upper_triangle(
    rsa_frame,
    response_range,
    stim,
    save_dir=None,
):
    """
    For a single (stim, response_range), compare communication subspaces by:

      1. Extracting the upper triangle (k=1) of each subspace's
         session-averaged RSA matrix.
      2. Plotting one boxplot per (brain_region, target_region) subspace.
      3. Computing pairwise Spearman correlations between subspaces' upper-
         triangle vectors (how strongly stimulus-similarity structure agrees
         across subspaces), plotting a correlation heatmap, and printing a
         Bonferroni-corrected summary.

    Parameters
    ----------
    rsa_frame : pd.DataFrame
        Must contain 'rsa_matrix', 'response_range', 'brain_region',
        'target_region'.
    response_range : str
        The response range to restrict to (e.g. 'onset').
    stim : str
        Stimulus type, used for titles / filenames.
    save_dir : str, optional
        Directory to save the two figures into. If None, figures are not saved.

    Returns
    -------
    dict
        {
            'labels': list of subspace labels,
            'upper_tri': dict label -> np.ndarray of upper-triangle values,
            'rho': np.ndarray pairwise Spearman rho matrix,
            'pval_corr': np.ndarray Bonferroni-corrected p-values,
        }
        or None if there is insufficient data.
    """
    sub_rr = rsa_frame[rsa_frame["response_range"] == response_range]
    if sub_rr.empty:
        print(f"No data for {stim} / {response_range}, skipping subspace stats.")
        return None

    region_pairs = list(
        sub_rr[["brain_region", "target_region"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    # Collect the upper-triangle vector for each subspace
    labels = []
    upper_tri = {}
    for brain_region, target_region in region_pairs:
        sub = sub_rr[
            (sub_rr["brain_region"] == brain_region)
            & (sub_rr["target_region"] == target_region)
        ]
        if sub.empty:
            continue

        avg_matrix = np.mean(np.stack(sub["rsa_matrix"].values), axis=0)
        n = avg_matrix.shape[0]
        upper_idx = np.triu_indices(n, k=1)
        vals = avg_matrix[upper_idx]

        label = f"{brain_region}–{target_region}"
        labels.append(label)
        upper_tri[label] = vals

    if len(labels) < 1:
        print(f"No subspaces with data for {stim} / {response_range}.")
        return None

    # ── 1. Boxplots of upper-triangle values per subspace ────────────────────
    fig_w = max(6, 1.4 * len(labels))
    fig_box, ax_box = plt.subplots(figsize=(fig_w, 6))
    cmap = plt.get_cmap("tab10")
    box_data = [upper_tri[lab] for lab in labels]
    bp = ax_box.boxplot(
        box_data,
        positions=np.arange(1, len(labels) + 1),
        widths=0.6,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i % 10))
        patch.set_alpha(0.7)

    ax_box.set_xticks(np.arange(1, len(labels) + 1))
    ax_box.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_box.set_ylabel("RSA upper-triangle values (Pearson r)")
    ax_box.set_xlabel("Communication subspace (brain_region–target_region)")
    ax_box.set_title(
        f"{stim} — {response_range}\n"
        f"Upper-triangle RSA values per communication subspace"
    )
    ax_box.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig_box.tight_layout()

    if save_dir is not None:
        box_path = (
            f"{save_dir}/rsa_subspace_boxplots_{stim}_{response_range}.png"
        )
        fig_box.savefig(box_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {box_path}")

    # ── 2. Pairwise Spearman correlations between subspaces ──────────────────
    n_sub = len(labels)
    rho = np.full((n_sub, n_sub), np.nan)
    pval = np.full((n_sub, n_sub), np.nan)

    # Only the unique off-diagonal pairs need a test
    pair_indices = []
    for i in range(n_sub):
        rho[i, i] = 1.0
        for j in range(i + 1, n_sub):
            x = upper_tri[labels[i]]
            y = upper_tri[labels[j]]
            # Spearman needs matched, finite entries
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3 and x.shape == y.shape:
                r, p = stats.spearmanr(x[mask], y[mask])
            else:
                r, p = np.nan, np.nan
            rho[i, j] = rho[j, i] = r
            pval[i, j] = pval[j, i] = p
            pair_indices.append((i, j))

    # Bonferroni correction across the unique pairwise tests
    pval_corr = np.full((n_sub, n_sub), np.nan)
    raw_p = np.array([pval[i, j] for (i, j) in pair_indices], dtype=float)
    valid = np.isfinite(raw_p)
    corr_vals = np.full_like(raw_p, np.nan)
    if valid.sum() > 0:
        _, pcorr, _, _ = multipletests(
            raw_p[valid], alpha=0.05, method="bonferroni"
        )
        corr_vals[valid] = pcorr
    for k, (i, j) in enumerate(pair_indices):
        pval_corr[i, j] = pval_corr[j, i] = corr_vals[k]

    def _sig_str(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    # ── Plot the Spearman correlation heatmap ────────────────────────────────
    fig_corr, ax_corr = plt.subplots(
        figsize=(max(5, 0.9 * n_sub + 2), max(4, 0.9 * n_sub + 1))
    )
    # Annotate each cell with rho and significance stars
    annot = np.empty((n_sub, n_sub), dtype=object)
    for i in range(n_sub):
        for j in range(n_sub):
            if i == j:
                annot[i, j] = "1.00"
            elif np.isnan(rho[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{rho[i, j]:.2f}\n{_sig_str(pval_corr[i, j])}"

    sns.heatmap(
        rho,
        ax=ax_corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 8},
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax_corr.set_title(
        f"{stim} — {response_range}\n"
        f"Spearman correlation between subspace RSA structures\n"
        f"(Bonferroni-corrected; * p<0.05, ** p<0.01, *** p<0.001)"
    )
    ax_corr.tick_params(axis="x", rotation=30)
    ax_corr.tick_params(axis="y", rotation=0)
    for tick in ax_corr.get_xticklabels():
        tick.set_ha("right")
    fig_corr.tight_layout()

    if save_dir is not None:
        corr_path = (
            f"{save_dir}/rsa_subspace_spearman_{stim}_{response_range}.png"
        )
        fig_corr.savefig(corr_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {corr_path}")

    # ── Console summary ──────────────────────────────────────────────────────
    print(
        f"\n{stim} — {response_range}: Spearman correlations between "
        f"communication subspaces (Bonferroni-corrected):"
    )
    header = (
        f"{'subspace A':<24} {'subspace B':<24} "
        f"{'rho':>8} {'p_raw':>12} {'p_corr':>12} {'sig':>5}"
    )
    print(header)
    print("-" * len(header))
    for (i, j) in pair_indices:
        def _fmt(v, width, prec=4):
            if np.isnan(v):
                return f"{'n/a':>{width}}"
            return f"{v:>{width}.{prec}f}"

        print(
            f"{labels[i]:<24} {labels[j]:<24} "
            f"{_fmt(rho[i, j], 8, 3)} {_fmt(pval[i, j], 12)} "
            f"{_fmt(pval_corr[i, j], 12)} {_sig_str(pval_corr[i, j]):>5}"
        )

    return {
        "labels": labels,
        "upper_tri": upper_tri,
        "rho": rho,
        "pval_corr": pval_corr,
    }

def plot_natural_sound_within_between(
    rsa_frame,
    stim,
    response_range_order=None,
    save_dir=None,
):
    """
    Natural-sound within- vs between-category RSA comparison.

    One figure with one panel per response_range (side by side). Within each
    panel: x-axis = brain-region subspace, with within- and between-category
    boxes stacked horizontally and color-coded.

    Statistics (Bonferroni-corrected across all brackets within a panel):
      * within vs between (per region)   -> Mann-Whitney U (unpaired)
      * within A vs within B             -> Spearman on matched within pairs
      * between A vs between B            -> Spearman on matched between pairs

    Matched-pair correlations rely on a fixed ordering of the upper-triangle
    indices, so pair index k refers to the same stimulus pair in every region.

    Parameters
    ----------
    rsa_frame : pd.DataFrame
        Must contain 'rsa_matrix', 'response_range', 'brain_region',
        'target_region'.
    stim : str
        Stimulus type (only 'naturalSound' is meaningful here).
    response_range_order : list of str, optional
        Desired panel ordering. Ranges present but not listed are appended.
    save_dir : str, optional
        Directory to save the figure into. If None, the figure is not saved.

    Returns
    -------
    (fig, axes) or None if there is insufficient data.
    """
    if stim != "naturalSound":
        return None

    expected_n = len(SOUND_CATEGORIES) * EXEMPLARS_PER_CATEGORY

    # Resolve response_range ordering against what's actually present
    present_ranges = list(rsa_frame["response_range"].unique())
    if response_range_order is None:
        resp_ranges = present_ranges
    else:
        resp_ranges = [r for r in response_range_order if r in present_ranges]
        for r in present_ranges:
            if r not in resp_ranges:
                resp_ranges.append(r)

    subspace_pairs = list(
        rsa_frame[["brain_region", "target_region"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    pair_type_order = ["within", "between"]
    pair_type_colors = {
        "within": "#4C72B0",   # blue
        "between": "#DD8452",  # orange
    }

    def _mannwhitney_safe(x, y):
        """Mann-Whitney U (unpaired); return (stat, p) or (nan, nan)."""
        try:
            if len(x) < 1 or len(y) < 1:
                return np.nan, np.nan
            res = stats.mannwhitneyu(x, y, alternative="two-sided")
            return res.statistic, res.pvalue
        except ValueError:
            return np.nan, np.nan

    def _spearman_safe(x, y):
        """Spearman on matched vectors; return (rho, p) or (nan, nan)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            return np.nan, np.nan
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return np.nan, np.nan
        r, p = stats.spearmanr(x[mask], y[mask])
        return r, p

    def _sig_str(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    # ── Collect within/between vectors keyed by (resp_range, region_label) ────
    # Values keep the fixed upper-triangle ordering so pairs stay matched.
    within_records = {}   # (resp_range, label) -> np.ndarray (30 values)
    between_records = {}   # (resp_range, label) -> np.ndarray (160 values)

    for resp_range in resp_ranges:
        for brain_region, target_region in subspace_pairs:
            sub = rsa_frame[
                (rsa_frame["response_range"] == resp_range)
                & (rsa_frame["brain_region"] == brain_region)
                & (rsa_frame["target_region"] == target_region)
            ]
            if sub.empty:
                continue

            avg_matrix = np.mean(np.stack(sub["rsa_matrix"].values), axis=0)
            n = avg_matrix.shape[0]
            if n != expected_n:
                print(
                    f"Skipping within/between split for "
                    f"{brain_region}-{target_region} / {resp_range}: "
                    f"matrix is {n}x{n}, expected "
                    f"{expected_n}x{expected_n}."
                )
                continue

            row_idx, col_idx = np.triu_indices(n, k=1)
            cat_row = row_idx // EXEMPLARS_PER_CATEGORY
            cat_col = col_idx // EXEMPLARS_PER_CATEGORY
            pair_vals = avg_matrix[row_idx, col_idx]

            within_mask = cat_row == cat_col
            between_mask = ~within_mask

            label = f"{brain_region}-{target_region}"
            within_records[(resp_range, label)] = pair_vals[within_mask]
            between_records[(resp_range, label)] = pair_vals[between_mask]

    if not within_records:
        print(f"No natural-sound within/between data for {stim}, skipping.")
        return None

    # ── Figure: one panel per response_range ─────────────────────────────────
    n_panels = len(resp_ranges)

    # Estimate width from the busiest panel
    max_regions = max(
        (len({lab for (rr, lab) in within_records if rr == r})
         for r in resp_ranges),
        default=1,
    )
    panel_w = max(4.0, 1.8 * max(max_regions, 1))
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_w * n_panels, 7),
        squeeze=False,
    )
    axes = axes.ravel()

    kind_colors = {
        "within_vs_between": "black",
        "within_region_vs_region": pair_type_colors["within"],
        "between_region_vs_region": pair_type_colors["between"],
    }

    console_blocks = []  # collected for printing after plotting

    # Accumulate every Spearman/MWU record for CSV export + ρ heatmaps
    spearman_records = []  # tidy rows for CSV
    within_rho_by_rr = {}  # resp_range -> (regions, rho_mat, pcorr_mat)
    between_rho_by_rr = {}  # resp_range -> (regions, rho_mat, pcorr_mat)

    for panel_idx, resp_range in enumerate(resp_ranges):
        ax = axes[panel_idx]

        # Regions with BOTH within and between data in this response_range
        available_regions = [
            f"{br}-{tr}" for (br, tr) in subspace_pairs
            if (resp_range, f"{br}-{tr}") in within_records
            and (resp_range, f"{br}-{tr}") in between_records
        ]
        if not available_regions:
            ax.set_axis_off()
            ax.set_title(f"{resp_range}\n(no data)")
            continue

        # ── Build comparison lists ───────────────────────────────────────
        # 1) within vs between per region (Mann-Whitney U)
        wb_comparisons = []
        for reg in available_regions:
            w = within_records[(resp_range, reg)]
            b = between_records[(resp_range, reg)]
            s, p = _mannwhitney_safe(w, b)
            wb_comparisons.append({
                "kind": "within_vs_between",
                "region": reg, "stat": s, "p_raw": p,
                "n_within": len(w), "n_between": len(b),
            })

        # 2) region vs region for within values (Spearman, matched pairs)
        within_region_comps = []
        for i in range(len(available_regions)):
            for j in range(i + 1, len(available_regions)):
                reg_a, reg_b = available_regions[i], available_regions[j]
                x = within_records[(resp_range, reg_a)]
                y = within_records[(resp_range, reg_b)]
                r, p = _spearman_safe(x, y)
                within_region_comps.append({
                    "kind": "within_region_vs_region",
                    "reg_a": reg_a, "reg_b": reg_b,
                    "stat": r, "p_raw": p,
                    "n_a": len(x), "n_b": len(y),
                })

        # 3) region vs region for between values (Spearman, matched pairs)
        between_region_comps = []
        for i in range(len(available_regions)):
            for j in range(i + 1, len(available_regions)):
                reg_a, reg_b = available_regions[i], available_regions[j]
                x = between_records[(resp_range, reg_a)]
                y = between_records[(resp_range, reg_b)]
                r, p = _spearman_safe(x, y)
                between_region_comps.append({
                    "kind": "between_region_vs_region",
                    "reg_a": reg_a, "reg_b": reg_b,
                    "stat": r, "p_raw": p,
                    "n_a": len(x), "n_b": len(y),
                })

        # Bonferroni across ALL comparisons in this panel
        all_comps = (
            wb_comparisons + within_region_comps + between_region_comps
        )
        raw_p = np.array([c["p_raw"] for c in all_comps], dtype=float)
        corr_p = np.full_like(raw_p, np.nan)
        reject = np.zeros_like(raw_p, dtype=bool)
        valid = np.isfinite(raw_p)
        if valid.sum() > 0:
            rej, pcorr, _, _ = multipletests(
                raw_p[valid], alpha=0.05, method="bonferroni"
            )
            corr_p[valid] = pcorr
            reject[valid] = rej
        for k, c in enumerate(all_comps):
            c["p_corr"] = corr_p[k]
            c["reject"] = bool(reject[k])

        # ── Accumulate records for CSV + ρ heatmaps ──────────────────────
        for c in wb_comparisons:
            spearman_records.append({
                "response_range": resp_range,
                "comparison_type": "within_vs_between",
                "region_a": c["region"],
                "region_b": c["region"],
                "test": "Mann-Whitney U",
                "statistic": c["stat"],
                "n_a": c["n_within"],
                "n_b": c["n_between"],
                "p_raw": c["p_raw"],
                "p_corr": c["p_corr"],
                "significant": c["reject"],
            })

        n_reg = len(available_regions)
        reg_index = {reg: i for i, reg in enumerate(available_regions)}

        for comps, store, ctype in (
                (within_region_comps, within_rho_by_rr, "within_region_vs_region"),
                (between_region_comps, between_rho_by_rr, "between_region_vs_region"),
        ):
            rho_mat = np.full((n_reg, n_reg), np.nan)
            pcorr_mat = np.full((n_reg, n_reg), np.nan)
            np.fill_diagonal(rho_mat, 1.0)
            for c in comps:
                i = reg_index[c["reg_a"]]
                j = reg_index[c["reg_b"]]
                rho_mat[i, j] = rho_mat[j, i] = c["stat"]
                pcorr_mat[i, j] = pcorr_mat[j, i] = c["p_corr"]
                spearman_records.append({
                    "response_range": resp_range,
                    "comparison_type": ctype,
                    "region_a": c["reg_a"],
                    "region_b": c["reg_b"],
                    "test": "Spearman",
                    "statistic": c["stat"],
                    "n_a": c["n_a"],
                    "n_b": c["n_b"],
                    "p_raw": c["p_raw"],
                    "p_corr": c["p_corr"],
                    "significant": c["reject"],
                })
            store[resp_range] = (list(available_regions), rho_mat, pcorr_mat)

        # ── Boxplots ─────────────────────────────────────────────────────
        n_reg = len(available_regions)
        box_width = 0.35
        x_centers = np.arange(1, n_reg + 1)
        box_positions = {}  # (region, pair_type) -> x pos
        all_vals_flat = []

        for ri, reg in enumerate(available_regions):
            for gi, pt in enumerate(pair_type_order):
                vals = (within_records if pt == "within"
                        else between_records).get((resp_range, reg))
                if vals is None or len(vals) == 0:
                    continue
                offset = (gi - 0.5) * box_width
                pos = x_centers[ri] + offset
                box_positions[(reg, pt)] = pos
                all_vals_flat.append(vals)
                bp = ax.boxplot(
                    [vals], positions=[pos],
                    widths=box_width * 0.9,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(pair_type_colors[pt])
                    patch.set_alpha(0.7)

        y_max = max(v.max() for v in all_vals_flat)
        y_min = min(v.min() for v in all_vals_flat)
        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
        bracket_height = y_range * 0.035
        bracket_gap = y_range * 0.015
        base_y = y_max + y_range * 0.04

        # Layer 1: within vs between per region (lowest brackets)
        current_y = base_y
        for c in wb_comparisons:
            key_w = (c["region"], "within")
            key_b = (c["region"], "between")
            if key_w not in box_positions or key_b not in box_positions:
                continue
            x_a = box_positions[key_w]
            x_b = box_positions[key_b]
            y_bottom = current_y
            y_top = y_bottom + bracket_height
            color = kind_colors[c["kind"]]
            ax.plot([x_a, x_a, x_b, x_b],
                    [y_bottom, y_top, y_top, y_bottom],
                    color=color, linewidth=0.8)
            ax.text((x_a + x_b) / 2, y_top, _sig_str(c["p_corr"]),
                    ha="center", va="bottom", fontsize=9,
                    color="#d62728" if c["reject"] else color)
        level_y = current_y + bracket_height + y_range * 0.04

        # Layer 2: stacked region-vs-region for WITHIN values
        for level, c in enumerate(within_region_comps):
            key_a = (c["reg_a"], "within")
            key_b = (c["reg_b"], "within")
            if key_a not in box_positions or key_b not in box_positions:
                continue
            x_a = box_positions[key_a]
            x_b = box_positions[key_b]
            y_bottom = level_y + level * (bracket_height + bracket_gap)
            y_top = y_bottom + bracket_height
            color = kind_colors[c["kind"]]
            ax.plot([x_a, x_a, x_b, x_b],
                    [y_bottom, y_top, y_top, y_bottom],
                    color=color, linewidth=0.8)
            ax.text((x_a + x_b) / 2, y_top, _sig_str(c["p_corr"]),
                    ha="center", va="bottom", fontsize=9,
                    color="#d62728" if c["reject"] else color)
        if within_region_comps:
            level_y = (level_y
                       + len(within_region_comps)
                       * (bracket_height + bracket_gap)
                       + y_range * 0.03)

        # Layer 3: stacked region-vs-region for BETWEEN values
        for level, c in enumerate(between_region_comps):
            key_a = (c["reg_a"], "between")
            key_b = (c["reg_b"], "between")
            if key_a not in box_positions or key_b not in box_positions:
                continue
            x_a = box_positions[key_a]
            x_b = box_positions[key_b]
            y_bottom = level_y + level * (bracket_height + bracket_gap)
            y_top = y_bottom + bracket_height
            color = kind_colors[c["kind"]]
            ax.plot([x_a, x_a, x_b, x_b],
                    [y_bottom, y_top, y_top, y_bottom],
                    color=color, linewidth=0.8)
            ax.text((x_a + x_b) / 2, y_top, _sig_str(c["p_corr"]),
                    ha="center", va="bottom", fontsize=9,
                    color="#d62728" if c["reject"] else color)
        top_y = level_y
        if between_region_comps:
            top_y = (level_y
                     + len(between_region_comps)
                     * (bracket_height + bracket_gap))

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylim(top=max(y_max + y_range * 0.15,
                            top_y + y_range * 0.05))
        ax.set_xticks(x_centers)
        ax.set_xticklabels(available_regions, fontsize=9,
                           rotation=30, ha="right")
        ax.set_xlim(0.5, n_reg + 0.5)
        ax.set_xlabel("Brain-region subspace")
        if panel_idx == 0:
            ax.set_ylabel("RSA values (Pearson r)")
        ax.set_title(f"{resp_range}")

        console_blocks.append({
            "resp_range": resp_range,
            "wb": wb_comparisons,
            "within": within_region_comps,
            "between": between_region_comps,
        })

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=pair_type_colors["within"], alpha=0.7,
              label="Within category"),
        Patch(facecolor=pair_type_colors["between"], alpha=0.7,
              label="Between category"),
        Line2D([0], [0], color="black", lw=1,
               label="Within vs between (Mann-Whitney U)"),
        Line2D([0], [0], color=pair_type_colors["within"], lw=1,
               label="Region vs region, within (Spearman)"),
        Line2D([0], [0], color=pair_type_colors["between"], lw=1,
               label="Region vs region, between (Spearman)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=8, title="Boxes / brackets",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"naturalSound - Within- vs between-category RSA "
        f"(Bonferroni-corrected per panel; "
        f"* p<0.05, ** p<0.01, *** p<0.001)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    if save_dir is not None:
        wb_path = f"{save_dir}/rsa_within_vs_between_{stim}.png"
        fig.savefig(wb_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {wb_path}")

        # ── Export Spearman/MWU table as CSV ─────────────────────────────────────
        if save_dir is not None and spearman_records:
            records_df = pd.DataFrame(spearman_records)
            csv_path = f"{save_dir}/rsa_within_vs_between_{stim}_stats.csv"
            records_df.to_csv(csv_path, index=False)
            print(f"Stats table saved to {csv_path}")

        # ── ρ heatmaps: region × region Spearman per response_range ──────────────
        def _rho_annot(rho_mat, pcorr_mat):
            m = rho_mat.shape[0]
            annot = np.empty((m, m), dtype=object)
            for i in range(m):
                for j in range(m):
                    if i == j:
                        annot[i, j] = "1.00"
                    elif np.isnan(rho_mat[i, j]):
                        annot[i, j] = ""
                    else:
                        annot[i, j] = (
                            f"{rho_mat[i, j]:.2f}\n{_sig_str(pcorr_mat[i, j])}"
                        )
            return annot

        def _plot_rho_grid(store, value_kind):
            """One figure: a ρ heatmap per response_range for within/between."""
            panels = [rr for rr in resp_ranges if rr in store]
            if not panels:
                return
            n_p = len(panels)
            # Size each panel from its region count
            max_m = max(len(store[rr][0]) for rr in panels)
            panel_side = max(4.0, 0.9 * max_m + 1.5)
            fig_r, axes_r = plt.subplots(
                1, n_p,
                figsize=(panel_side * n_p, panel_side),
                squeeze=False,
            )
            axes_r = axes_r.ravel()
            for idx, rr in enumerate(panels):
                ax_r = axes_r[idx]
                regions, rho_mat, pcorr_mat = store[rr]
                sns.heatmap(
                    rho_mat,
                    ax=ax_r,
                    cmap="RdBu_r",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    annot=_rho_annot(rho_mat, pcorr_mat),
                    fmt="",
                    annot_kws={"fontsize": 8},
                    xticklabels=regions,
                    yticklabels=regions,
                    cbar_kws={"label": "Spearman ρ"},
                )
                ax_r.set_title(f"{rr}")
                ax_r.tick_params(axis="x", rotation=30)
                ax_r.tick_params(axis="y", rotation=0)
                for tick in ax_r.get_xticklabels():
                    tick.set_ha("right")

            fig_r.suptitle(
                f"naturalSound - Region-vs-region Spearman ρ "
                f"({value_kind}-category matched pairs)\n"
                f"(Bonferroni-corrected; * p<0.05, ** p<0.01, *** p<0.001)",
                fontsize=13,
            )
            fig_r.tight_layout(rect=[0, 0, 1, 0.95])

            if save_dir is not None:
                rho_path = (
                    f"{save_dir}/rsa_spearman_rho_{value_kind}_{stim}.png"
                )
                fig_r.savefig(rho_path, dpi=300, bbox_inches="tight")
                print(f"Figure saved to {rho_path}")

        _plot_rho_grid(within_rho_by_rr, "within")
        _plot_rho_grid(between_rho_by_rr, "between")

    # ── Console summaries ────────────────────────────────────────────────────
    def _fmt(v, width, prec=4):
        if np.isnan(v):
            return f"{'n/a':>{width}}"
        return f"{v:>{width}.{prec}f}"

    for block in console_blocks:
        resp_range = block["resp_range"]

        print(f"\nnaturalSound - {resp_range}: Within vs between per region "
              f"(Mann-Whitney U, Bonferroni-corrected across panel):")
        header = (f"{'region':<28} {'n_within':>8} {'n_between':>10} "
                  f"{'U':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")
        print(header)
        print("-" * len(header))
        for c in block["wb"]:
            print(f"{str(c['region']):<28} {c['n_within']:>8d} "
                  f"{c['n_between']:>10d} "
                  f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                  f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

        header2 = (f"{'region A':<28} {'region B':<28} "
                   f"{'rho':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")

        print(f"\nnaturalSound - {resp_range}: Region-vs-region for WITHIN "
              f"values (Spearman, matched pairs):")
        print(header2)
        print("-" * len(header2))
        for c in block["within"]:
            print(f"{str(c['reg_a']):<28} {str(c['reg_b']):<28} "
                  f"{_fmt(c['stat'], 10, 3)} {_fmt(c['p_raw'], 12)} "
                  f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

        print(f"\nnaturalSound - {resp_range}: Region-vs-region for BETWEEN "
              f"values (Spearman, matched pairs):")
        print(header2)
        print("-" * len(header2))
        for c in block["between"]:
            print(f"{str(c['reg_a']):<28} {str(c['reg_b']):<28} "
                  f"{_fmt(c['stat'], 10, 3)} {_fmt(c['p_raw'], 12)} "
                  f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

    return fig, axes

# New function to compute null result for RSA where we compute the spearman correlation between two different sessions between the same brain region and different brain regions
def RSA_null():


def create_RDM_plots():
    """
    Pulls correlation matrices for each session and averages across a brain region
    to make a general RDM for each stim type, brain area, and response_range.
    Then plots:
      1. Per-stimulus heatmap grid (rows = response_range, cols = region) of
         the session-averaged RDM.
      2. A per-stimulus boxplot of upper-triangle correlations with
         response_range on the x-axis and boxes colored by brain region.
    """

    response_range_order = ['onset', 'sustained', 'offset']

    for stim in ["AM", "pureTones", "naturalSound"]:
        stim_arrays = fr_db.return_arrays(stim)
        uniqStims = np.unique(stim_arrays["stimArray"][0, :])
        rsa_frame = pd.DataFrame()
        for response_range in response_range_order:
            try:
                with open(f"{file_path}/CCA_RSA/rsa_data_{stim}_{response_range}.pkl", 'rb') as f:
                    rsa_pkl = pickle.load(f)
                    rsa_frame = pd.concat([rsa_frame, pd.DataFrame(rsa_pkl)], ignore_index=True)
            except FileNotFoundError:
                print(f"No data found for {stim}, skipping.")
                continue

        # Drop rows that have NaN for 'rsa_matrix'
        rsa_frame = rsa_frame.dropna(subset=['rsa_matrix'])
        if rsa_frame.empty:
            print(f"No data found for {stim}, skipping.")
            continue

        stim_labels = get_stim_labels(
            stim,
            uniqStims,
            sounds_categories=SOUND_CATEGORIES,
            exemplars_per_category=EXEMPLARS_PER_CATEGORY,
        )

        fig, axes = plot_rsa_heatmaps(
            rsa_frame,
            stim_labels=stim_labels,
            save_path=f"{file_path}/CCA_RSA/heatmaps/rsa_heatmaps_{stim}.png",
        )

        # Compare communication subspaces via upper-triangle boxplots
        # and pairwise Spearman correlations, one set per response_range.
        for response_range in response_range_order:
            plot_subspace_upper_triangle(
                rsa_frame,
                response_range=response_range,
                stim=stim,
                save_dir=f"{file_path}/CCA_RSA/upper_tri",
            )

        # Grouped boxplots: response_range on x-axis, subspaces stacked
        # horizontally and color-coded (one figure per stimulus).
        plot_subspace_boxplots(
            rsa_frame,
            stim=stim,
            response_range_order=response_range_order,
            save_dir=f"{file_path}/CCA_RSA/boxplots",
        )

        # Natural-sound only: within- vs between-category RSA values,
        # one panel per response_range (x-axis = brain-region subspace).
        plot_natural_sound_within_between(
            rsa_frame,
            stim=stim,
            response_range_order=response_range_order,
            save_dir=f"{file_path}/CCA_RSA/wb",
        )


if __name__ == "__main__":
    create_RDM_plots()