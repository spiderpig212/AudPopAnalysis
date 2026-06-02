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

def create_upper_triangle_boxplots():
    """
    For each stimulus type, extract upper-triangle values (excluding diagonal)
    from the pairwise `mean_accuracy` matrices restricted to a single brain
    region (`region1`). Plot grouped boxplots with `response_range` on the
    x-axis and one box per brain region (color-coded). Annotate with
    pairwise Mann-Whitney U tests comparing regions within each
    response_range (unpaired), Bonferroni-corrected across all comparisons.
    """
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            ns_df = pd.read_feather(f"{file_path}/SVC/SVC_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        # Use the single-region column `region1`
        regions = sorted(ns_df['region1'].dropna().unique().tolist())
        response_ranges = list(ns_df['response_range'].unique())

        # Collect upper-triangle values per (response_range, region) group
        all_stims = set()
        for stim_pair in ns_df['stim_pair']:
            all_stims.update(stim_pair)
        unique_stims = sorted(list(all_stims))
        n_stims = len(unique_stims)

        # group_values[(resp_range, region)] -> np.array of upper-tri mean_accuracy
        group_values = {}

        for resp_range in response_ranges:
            for region in regions:
                subset = ns_df[
                    (ns_df['response_range'] == resp_range) &
                    (ns_df['region1'] == region)
                ]

                if len(subset) == 0:
                    continue

                acc_matrix = np.full((n_stims, n_stims), np.nan)
                for _, row in subset.iterrows():
                    stim1, stim2 = row['stim_pair']
                    idx1 = unique_stims.index(stim1)
                    idx2 = unique_stims.index(stim2)
                    val = row['mean_accuracy']
                    acc_matrix[idx1, idx2] = val
                    acc_matrix[idx2, idx1] = val
                np.fill_diagonal(acc_matrix, np.nan)

                upper_idx = np.triu_indices(n_stims, k=1)
                vals = acc_matrix[upper_idx]
                vals = vals[~np.isnan(vals)]

                if len(vals) > 0:
                    group_values[(resp_range, region)] = vals

        if not group_values:
            print(f"No data found for {stim}, skipping.")
            continue

        def _mannwhitney_safe(x, y):
            """Run Mann-Whitney U (unpaired); return (stat, p) or (nan, nan)."""
            try:
                if len(x) < 1 or len(y) < 1:
                    return np.nan, np.nan
                res = stats.mannwhitneyu(x, y, alternative='two-sided')
                return res.statistic, res.pvalue
            except ValueError:
                return np.nan, np.nan

        # Build pairwise comparisons within each response_range
        # comparisons: list of dicts with keys: resp_range, reg_a, reg_b, stat, p
        comparisons = []
        for resp_range in response_ranges:
            available = [r for r in regions if (resp_range, r) in group_values]
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    reg_a, reg_b = available[i], available[j]
                    x = group_values[(resp_range, reg_a)]
                    y = group_values[(resp_range, reg_b)]
                    s, p = _mannwhitney_safe(x, y)
                    comparisons.append({
                        'resp_range': resp_range,
                        'reg_a': reg_a,
                        'reg_b': reg_b,
                        'stat': s,
                        'p_raw': p,
                    })

        # Bonferroni correction across all comparisons
        raw_p_arr = np.array([c['p_raw'] for c in comparisons], dtype=float)
        corr_p = np.full_like(raw_p_arr, np.nan)
        reject = np.zeros_like(raw_p_arr, dtype=bool)
        valid_mask = ~np.isnan(raw_p_arr)
        if valid_mask.sum() > 0:
            rej, pvals_corr, _, _ = multipletests(raw_p_arr[valid_mask],
                                                  alpha=0.05,
                                                  method='bonferroni')
            corr_p[valid_mask] = pvals_corr
            reject[valid_mask] = rej
        for i, c in enumerate(comparisons):
            c['p_corr'] = corr_p[i]
            c['reject'] = bool(reject[i])

        def _sig_str(p):
            if np.isnan(p):
                return "n/a"
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return "ns"

        # ── Plotting ─────────────────────────────────────────────────────────
        n_x = len(response_ranges)
        n_reg = len(regions)
        fig_width = max(6, n_x * max(2.0, 0.9 * n_reg))
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        cmap = plt.get_cmap('tab10')
        region_colors = {reg: cmap(i % 10) for i, reg in enumerate(regions)}

        total_width = 0.8
        box_width = total_width / max(n_reg, 1)
        x_centers = np.arange(1, n_x + 1)

        # Track box positions for annotation brackets
        box_positions = {}  # (resp_range, region) -> x position

        all_vals_flat = []
        for ri, resp_range in enumerate(response_ranges):
            for gi, region in enumerate(regions):
                key = (resp_range, region)
                if key not in group_values:
                    continue
                vals = group_values[key]
                all_vals_flat.append(vals)
                offset = (gi - (n_reg - 1) / 2) * box_width
                pos = x_centers[ri] + offset
                box_positions[key] = pos
                bp = ax.boxplot([vals], positions=[pos], widths=box_width * 0.9,
                                patch_artist=True, notch=False,
                                medianprops=dict(color='black', linewidth=2))
                for patch in bp['boxes']:
                    patch.set_facecolor(region_colors[region])
                    patch.set_alpha(0.7)

        # Y-axis layout
        y_max = max(v.max() for v in all_vals_flat)
        y_min = min(v.min() for v in all_vals_flat)
        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

        # Draw pairwise comparison brackets, stacked vertically within each response_range
        bracket_height = y_range * 0.04
        bracket_gap = y_range * 0.02
        base_y = y_max + y_range * 0.04

        # Group comparisons by response_range for stacking
        comps_by_rr = {}
        for c in comparisons:
            comps_by_rr.setdefault(c['resp_range'], []).append(c)

        max_top = base_y
        for resp_range, comps in comps_by_rr.items():
            for level, c in enumerate(comps):
                key_a = (resp_range, c['reg_a'])
                key_b = (resp_range, c['reg_b'])
                if key_a not in box_positions or key_b not in box_positions:
                    continue
                x_a = box_positions[key_a]
                x_b = box_positions[key_b]
                y_bottom = base_y + level * (bracket_height + bracket_gap)
                y_top = y_bottom + bracket_height
                ax.plot([x_a, x_a, x_b, x_b],
                        [y_bottom, y_top, y_top, y_bottom],
                        color='gray', linewidth=0.8)
                ax.text((x_a + x_b) / 2, y_top, _sig_str(c['p_corr']),
                        ha='center', va='bottom', fontsize=9,
                        color='#d62728' if c['reject'] else 'gray')
                max_top = max(max_top, y_top + y_range * 0.03)

        ax.set_ylim(top=max(y_max + y_range * 0.15, max_top + y_range * 0.05))
        ax.set_xticks(x_centers)
        ax.set_xticklabels([str(r) for r in response_ranges],
                           fontsize=9, rotation=30, ha='right')
        ax.set_xlabel('response_range')
        ax.set_ylabel('mean_accuracy (upper triangle values)')
        ax.set_title(f'{stim} — Upper Triangle mean_accuracy by response_range\n'
                     f'(colored by brain region; pairwise Mann-Whitney U within '
                     f'response_range, Bonferroni-corrected; '
                     f'* p<0.05, ** p<0.01, *** p<0.001)')

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=region_colors[reg], alpha=0.7, label=reg)
                           for reg in regions]
        ax.legend(handles=legend_elements, loc='upper right', title='Region')

        ax.set_xlim(0.5, n_x + 0.5)
        plt.tight_layout()
        plt.savefig(f"{file_path}/SVC/Upper_tri/upper_triangle_boxplots_{stim}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # ── Console summary ──────────────────────────────────────────────────
        print(f"\n{stim} — Pairwise region comparisons within each response_range "
              f"(Mann-Whitney U, Bonferroni-corrected):")
        header = (f"{'response_range':<20} {'region A':<12} {'region B':<12} "
                  f"{'U':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")
        print(header)
        print("-" * len(header))
        for c in comparisons:
            def _fmt(v, width, prec=4):
                if np.isnan(v):
                    return f"{'n/a':>{width}}"
                return f"{v:>{width}.{prec}f}"
            print(f"{str(c['resp_range']):<20} {str(c['reg_a']):<12} "
                  f"{str(c['reg_b']):<12} "
                  f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                  f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

def create_accuracy_heatmaps():
    """
    For each stimulus type, plot pairwise `mean_accuracy` heatmaps across all
    stimulus pairs for every (response_range, region1) combination.

    One figure per stimulus type. Rows = response_range, columns = region1.
    Each panel shows the symmetric stimulus × stimulus accuracy matrix.
    """
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            ns_df = pd.read_feather(f"{file_path}/SVC/SVC_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        regions = sorted(ns_df['region1'].dropna().unique().tolist())
        response_ranges = list(ns_df['response_range'].unique())

        # Build the global stimulus index (consistent ordering across panels)
        all_stims = set()
        for stim_pair in ns_df['stim_pair']:
            all_stims.update(stim_pair)
        unique_stims = sorted(list(all_stims))
        n_stims = len(unique_stims)

        # Build accuracy matrices keyed by (response_range, region)
        matrices = {}
        for resp_range in response_ranges:
            for region in regions:
                subset = ns_df[
                    (ns_df['response_range'] == resp_range) &
                    (ns_df['region1'] == region)
                ]
                if len(subset) == 0:
                    continue

                acc_matrix = np.full((n_stims, n_stims), np.nan)
                for _, row in subset.iterrows():
                    stim1, stim2 = row['stim_pair']
                    idx1 = unique_stims.index(stim1)
                    idx2 = unique_stims.index(stim2)
                    val = row['mean_accuracy']
                    acc_matrix[idx1, idx2] = val
                    acc_matrix[idx2, idx1] = val

                matrices[(resp_range, region)] = acc_matrix

        if not matrices:
            print(f"No data found for {stim}, skipping.")
            continue

        # Shared color scale across all panels of this figure
        all_vals = np.concatenate([m[~np.isnan(m)].ravel()
                                   for m in matrices.values()])
        if all_vals.size == 0:
            print(f"No valid accuracy values for {stim}, skipping.")
            continue
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))

        n_rows = len(response_ranges)
        n_cols = len(regions)
        fig_w = max(4, n_cols * max(4, 0.45 * n_stims))
        fig_h = max(4, n_rows * max(4, 0.45 * n_stims))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)

        last_im = None
        for ri, resp_range in enumerate(response_ranges):
            for ci, region in enumerate(regions):
                ax = axes[ri][ci]
                key = (resp_range, region)
                if key not in matrices:
                    ax.set_axis_off()
                    ax.set_title(f"{region} | {resp_range}\n(no data)",
                                 fontsize=14)
                    continue

                mat = matrices[key]
                im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax,
                               aspect='auto', origin='upper')
                last_im = im

                ax.set_xticks(np.arange(n_stims))
                ax.set_yticks(np.arange(n_stims))
                ax.set_xticklabels(unique_stims, rotation=90, fontsize=14)
                ax.set_yticklabels(unique_stims, fontsize=14)
                ax.set_title(f"{region} | {resp_range}", fontsize=18)

                # Only label axes on the outer panels to keep things clean
                if ci == 0:
                    ax.set_ylabel('stimulus')
                if ri == n_rows - 1:
                    ax.set_xlabel('stimulus')

        # Shared colorbar
        if last_im is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            fig.colorbar(last_im, cax=cbar_ax, label='mean_accuracy')

        fig.suptitle(f'{stim} — Pairwise mean_accuracy heatmaps '
                     f'(rows: response_range, columns: region)',
                     fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 0.9, 0.98])
        plt.savefig(f"{file_path}/SVC/Upper_tri/accuracy_heatmaps_{stim}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

def create_RDM_plots():
    """
    Pulls correlation matrices for each session and averages across a brain region to make a general RDM for each stim
    type and brain area
    """
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            with open(f"{file_path}/RDM/RDM_{stim}.pkl", 'rb') as f:
                rdm_pkl = pickle.load(f)
            rdm_df = pd.DataFrame(rdm_pkl)
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue


print("Creating upper triangle boxplots with stats...")
create_upper_triangle_boxplots()

print("Creating pairwise accuracy heatmaps...")
create_accuracy_heatmaps()