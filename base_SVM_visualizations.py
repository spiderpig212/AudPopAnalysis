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

def create_upper_triangle_boxplots():
    """
    For each stimulus type, extract upper-triangle values (excluding diagonal)
    from the pairwise `mean_accuracy` matrices restricted to a single brain
    region (`region1`). Plot grouped boxplots with `response_range` on the
    x-axis and one box per brain region (color-coded). Annotate with
    pairwise Mann-Whitney U tests comparing regions within each
    response_range (unpaired), Bonferroni-corrected across all comparisons.
    """
    # Canonical response_range ordering for downstream plots
    response_range_order = ['onset', 'sustained', 'offset']
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            ns_df = pd.read_feather(f"{file_path}/SVC/SVC_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        # Use the single-region column `region1`
        regions = sorted(ns_df['region1'].dropna().unique().tolist())
        # Drop Temporal Association Area from regions
        regions = [r for r in regions if r != "Temporal association areas"]
        response_ranges = list(ns_df['response_range'].unique())

        # Collect upper-triangle values per (response_range, region) group
        all_stims = set()
        for stim_pair in ns_df['stim_pair']:
            all_stims.update(stim_pair)
        unique_stims = sorted(list(all_stims))
        n_stims = len(unique_stims)

        # group_values[(resp_range, region)] -> np.array of upper-tri mean_accuracy
        group_values = {}

        # For naturalSound: within_between_records[(resp_range, region, pair_type)]
        within_between_records = {}

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

                # Within- vs between-category split (naturalSound only)
                if stim == "naturalSound":
                    expected_n = len(SOUND_CATEGORIES) * EXEMPLARS_PER_CATEGORY
                    if n_stims == expected_n:
                        row_idx, col_idx = np.triu_indices(n_stims, k=1)
                        cat_row = row_idx // EXEMPLARS_PER_CATEGORY
                        cat_col = col_idx // EXEMPLARS_PER_CATEGORY
                        pair_vals = acc_matrix[row_idx, col_idx]

                        within_mask = (cat_row == cat_col)
                        between_mask = ~within_mask

                        w = pair_vals[within_mask]
                        b = pair_vals[between_mask]
                        w = w[~np.isnan(w)]
                        b = b[~np.isnan(b)]

                        if len(w) > 0:
                            within_between_records[(resp_range, region, 'within')] = w
                        if len(b) > 0:
                            within_between_records[(resp_range, region, 'between')] = b
                    else:
                        print(f"Skipping within/between split for "
                              f"{region}/{resp_range}: matrix is "
                              f"{n_stims}×{n_stims}, expected "
                              f"{expected_n}×{expected_n}.")

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
        ax.legend(handles=legend_elements, loc='lower right', title='Region')

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

        # ── Natural-sound within- vs between-category boxplots ───────────
        if stim != "naturalSound" or not within_between_records:
            continue

        wb_regions = sorted({k[1] for k in within_between_records})
        wb_resp_ranges_present = {k[0] for k in within_between_records}
        wb_resp_ranges = [r for r in response_range_order
                          if r in wb_resp_ranges_present]
        for r in wb_resp_ranges_present:
            if r not in wb_resp_ranges:
                wb_resp_ranges.append(r)

        pair_type_order = ['within', 'between']
        pair_type_colors = {
            'within': '#4C72B0',  # blue
            'between': '#DD8452',  # orange
        }

        for resp_range in wb_resp_ranges:
            available_regions = [
                reg for reg in wb_regions
                if (resp_range, reg, 'within') in within_between_records
                   and (resp_range, reg, 'between') in within_between_records
            ]
            if not available_regions:
                continue

            # ── Build comparison lists ──────────────────────────────────
            # 1) within-vs-between per region (paired in spirit; unpaired test)
            wb_comparisons = []
            for reg in available_regions:
                w = within_between_records[(resp_range, reg, 'within')]
                b = within_between_records[(resp_range, reg, 'between')]
                s, p = _mannwhitney_safe(w, b)
                wb_comparisons.append({
                    'kind': 'within_vs_between',
                    'region': reg, 'stat': s, 'p_raw': p,
                    'n_within': len(w), 'n_between': len(b),
                })

            # 2) region-vs-region for within values
            within_region_comps = []
            for i in range(len(available_regions)):
                for j in range(i + 1, len(available_regions)):
                    reg_a, reg_b = available_regions[i], available_regions[j]
                    x = within_between_records[(resp_range, reg_a, 'within')]
                    y = within_between_records[(resp_range, reg_b, 'within')]
                    s, p = _mannwhitney_safe(x, y)
                    within_region_comps.append({
                        'kind': 'within_region_vs_region',
                        'reg_a': reg_a, 'reg_b': reg_b,
                        'stat': s, 'p_raw': p,
                        'n_a': len(x), 'n_b': len(y),
                    })

            # 3) region-vs-region for between values
            between_region_comps = []
            for i in range(len(available_regions)):
                for j in range(i + 1, len(available_regions)):
                    reg_a, reg_b = available_regions[i], available_regions[j]
                    x = within_between_records[(resp_range, reg_a, 'between')]
                    y = within_between_records[(resp_range, reg_b, 'between')]
                    s, p = _mannwhitney_safe(x, y)
                    between_region_comps.append({
                        'kind': 'between_region_vs_region',
                        'reg_a': reg_a, 'reg_b': reg_b,
                        'stat': s, 'p_raw': p,
                        'n_a': len(x), 'n_b': len(y),
                    })

            # Bonferroni-correct across ALL comparisons in this figure
            all_comps = wb_comparisons + within_region_comps + between_region_comps
            raw_p = np.array([c['p_raw'] for c in all_comps], dtype=float)
            corr_p = np.full_like(raw_p, np.nan)
            reject = np.zeros_like(raw_p, dtype=bool)
            valid = ~np.isnan(raw_p)
            if valid.sum() > 0:
                rej, pcorr, _, _ = multipletests(raw_p[valid], alpha=0.05,
                                                 method='bonferroni')
                corr_p[valid] = pcorr
                reject[valid] = rej
            for i, c in enumerate(all_comps):
                c['p_corr'] = corr_p[i]
                c['reject'] = bool(reject[i])

            # ── Plot ────────────────────────────────────────────────────
            n_reg_wb = len(available_regions)
            fig_w = max(6, n_reg_wb * 1.8)
            fig, ax = plt.subplots(figsize=(fig_w, 7))

            box_width = 0.35
            x_centers = np.arange(1, n_reg_wb + 1)
            box_positions = {}  # (region, pair_type) -> x pos
            all_vals_flat = []

            for ri, reg in enumerate(available_regions):
                for gi, pt in enumerate(pair_type_order):
                    vals = within_between_records.get(
                        (resp_range, reg, pt))
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
                        medianprops=dict(color='black', linewidth=2),
                    )
                    for patch in bp['boxes']:
                        patch.set_facecolor(pair_type_colors[pt])
                        patch.set_alpha(0.7)

            y_max = max(v.max() for v in all_vals_flat)
            y_min = min(v.min() for v in all_vals_flat)
            y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
            bracket_height = y_range * 0.035
            bracket_gap = y_range * 0.015
            base_y = y_max + y_range * 0.04

            # Bracket colors per comparison kind
            kind_colors = {
                'within_vs_between': 'black',
                'within_region_vs_region': pair_type_colors['within'],
                'between_region_vs_region': pair_type_colors['between'],
            }

            current_y = base_y

            # Layer 1: within-vs-between brackets (lowest, per-region pairs)
            for c in wb_comparisons:
                reg = c['region']
                key_w = (reg, 'within')
                key_b = (reg, 'between')
                if key_w not in box_positions or key_b not in box_positions:
                    continue
                x_a = box_positions[key_w]
                x_b = box_positions[key_b]
                y_bottom = current_y
                y_top = y_bottom + bracket_height
                color = kind_colors[c['kind']]
                ax.plot([x_a, x_a, x_b, x_b],
                        [y_bottom, y_top, y_top, y_bottom],
                        color=color, linewidth=0.8)
                ax.text((x_a + x_b) / 2, y_top, _sig_str(c['p_corr']),
                        ha='center', va='bottom', fontsize=9,
                        color='#d62728' if c['reject'] else color)
            max_top_layer1 = current_y + bracket_height + y_range * 0.04

            # Layer 2: stacked region-vs-region for within values
            level_y = max_top_layer1
            for level, c in enumerate(within_region_comps):
                key_a = (c['reg_a'], 'within')
                key_b = (c['reg_b'], 'within')
                if key_a not in box_positions or key_b not in box_positions:
                    continue
                x_a = box_positions[key_a]
                x_b = box_positions[key_b]
                y_bottom = level_y + level * (bracket_height + bracket_gap)
                y_top = y_bottom + bracket_height
                color = kind_colors[c['kind']]
                ax.plot([x_a, x_a, x_b, x_b],
                        [y_bottom, y_top, y_top, y_bottom],
                        color=color, linewidth=0.8)
                ax.text((x_a + x_b) / 2, y_top, _sig_str(c['p_corr']),
                        ha='center', va='bottom', fontsize=9,
                        color='#d62728' if c['reject'] else color)
            if within_region_comps:
                level_y = (level_y
                           + len(within_region_comps)
                           * (bracket_height + bracket_gap)
                           + y_range * 0.03)

            # Layer 3: stacked region-vs-region for between values
            for level, c in enumerate(between_region_comps):
                key_a = (c['reg_a'], 'between')
                key_b = (c['reg_b'], 'between')
                if key_a not in box_positions or key_b not in box_positions:
                    continue
                x_a = box_positions[key_a]
                x_b = box_positions[key_b]
                y_bottom = level_y + level * (bracket_height + bracket_gap)
                y_top = y_bottom + bracket_height
                color = kind_colors[c['kind']]
                ax.plot([x_a, x_a, x_b, x_b],
                        [y_bottom, y_top, y_top, y_bottom],
                        color=color, linewidth=0.8)
                ax.text((x_a + x_b) / 2, y_top, _sig_str(c['p_corr']),
                        ha='center', va='bottom', fontsize=9,
                        color='#d62728' if c['reject'] else color)
            top_y = level_y
            if between_region_comps:
                top_y = (level_y
                         + len(between_region_comps)
                         * (bracket_height + bracket_gap))

            ax.set_ylim(top=max(y_max + y_range * 0.15,
                                top_y + y_range * 0.05))
            ax.set_xticks(x_centers)
            ax.set_xticklabels(available_regions, fontsize=10,
                               rotation=30, ha='right')
            ax.set_xlim(0.5, n_reg_wb + 0.5)
            ax.set_xlabel('Brain region')
            ax.set_ylabel('mean_accuracy (upper triangle values)')
            ax.set_title(
                f'naturalSound — Within- vs between-category SVC accuracy '
                f'({resp_range})\n'
                f'(Mann-Whitney U; Bonferroni-corrected across all brackets; '
                f'* p<0.05, ** p<0.01, *** p<0.001)'
            )

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=pair_type_colors['within'], alpha=0.7,
                      label='Within category'),
                Patch(facecolor=pair_type_colors['between'], alpha=0.7,
                      label='Between category'),
                Line2D([0], [0], color='black', lw=1,
                       label='Within vs between (per region)'),
                Line2D([0], [0], color=pair_type_colors['within'], lw=1,
                       label='Region vs region (within)'),
                Line2D([0], [0], color=pair_type_colors['between'], lw=1,
                       label='Region vs region (between)'),
            ]
            ax.legend(handles=legend_elements, loc='lower right',
                      fontsize=8, title='Boxes / brackets')

            plt.tight_layout()
            plt.savefig(
                f"{file_path}/SVC/Upper_tri/"
                f"within_vs_between_naturalSound_{resp_range}.png",
                dpi=300, bbox_inches='tight',
            )
            plt.show()

            # ── Console summaries ───────────────────────────────────────
            def _fmt(v, width, prec=4):
                if np.isnan(v):
                    return f"{'n/a':>{width}}"
                return f"{v:>{width}.{prec}f}"

            print(f"\nnaturalSound — Within vs between category per region "
                  f"({resp_range}; Mann-Whitney U, Bonferroni-corrected "
                  f"across all brackets in this figure):")
            header = (f"{'region':<28} {'n_within':>8} {'n_between':>10} "
                      f"{'U':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")
            print(header)
            print("-" * len(header))
            for c in wb_comparisons:
                print(f"{str(c['region']):<28} {c['n_within']:>8d} "
                      f"{c['n_between']:>10d} "
                      f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                      f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

            print(f"\nnaturalSound — Region-vs-region for WITHIN-category "
                  f"values ({resp_range}):")
            header2 = (f"{'region A':<28} {'region B':<28} "
                       f"{'U':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")
            print(header2)
            print("-" * len(header2))
            for c in within_region_comps:
                print(f"{str(c['reg_a']):<28} {str(c['reg_b']):<28} "
                      f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                      f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

            print(f"\nnaturalSound — Region-vs-region for BETWEEN-category "
                  f"values ({resp_range}):")
            print(header2)
            print("-" * len(header2))
            for c in between_region_comps:
                print(f"{str(c['reg_a']):<28} {str(c['reg_b']):<28} "
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
        regions = [r for r in regions if r != "Temporal association areas"]
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
    Pulls correlation matrices for each session and averages across a brain region
    to make a general RDM for each stim type, brain area, and response_range.
    Then plots:
      1. Per-stimulus heatmap grid (rows = response_range, cols = region) of
         the session-averaged RDM.
      2. A per-stimulus boxplot of upper-triangle correlations with
         response_range on the x-axis and boxes colored by brain region.
    """
    # Long-format container for boxplots
    boxplot_records = []  # {stimulus, response_range, region, corr}

    # Container for natural-sound within/between records keyed by
    # (response_range, region, pair_type) -> list of corr values
    within_between_records = {}

    # Keep a consistent color mapping for regions across figures
    region_color_map = {}
    cmap = plt.get_cmap('tab10')

    # Desired response_range ordering
    response_range_order = ['onset', 'sustained', 'offset']

    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            with open(f"{file_path}/RDM/RDM_{stim}.pkl", 'rb') as f:
                rdm_pkl = pickle.load(f)
            rdm_df = pd.DataFrame(rdm_pkl)
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        required = {'corr_mat', 'region1', 'response_range'}
        if not required.issubset(rdm_df.columns):
            print(f"Required columns missing for {stim}, skipping.")
            continue

        regions = sorted(rdm_df['region1'].dropna().unique().tolist())
        regions = [r for r in regions if r != "Temporal association areas"]
        # Preserve canonical order but only keep what exists in the data
        present_ranges = rdm_df['response_range'].dropna().unique().tolist()
        resp_ranges = [r for r in response_range_order if r in present_ranges]
        # Append any unexpected response ranges so we still plot them
        for r in present_ranges:
            if r not in resp_ranges:
                resp_ranges.append(r)

        n_sessions = rdm_df['session'].nunique()

        # Register any new regions in the global color map
        for reg in regions:
            if reg not in region_color_map:
                region_color_map[reg] = cmap(len(region_color_map) % 10)

        # Build averaged matrices keyed by (response_range, region)
        avg_matrices = {}
        for resp_range in resp_ranges:
            for region in regions:
                subset = rdm_df[
                    (rdm_df['region1'] == region) &
                    (rdm_df['response_range'] == resp_range)
                    ]
                mats = []
                for m in subset['corr_mat']:
                    if m is None:
                        continue
                    arr = np.asarray(m, dtype=float)
                    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                        mats.append(arr)

                if not mats:
                    continue

                # Only stack matrices that share the most common shape
                shapes = [m.shape for m in mats]
                common_shape = max(set(shapes), key=shapes.count)
                mats = [m for m in mats if m.shape == common_shape]
                if not mats:
                    continue

                stacked = np.stack(mats, axis=0)
                avg_mat = np.nanmean(stacked, axis=0)
                avg_matrices[(resp_range, region)] = avg_mat

                # Collect upper-triangle values for the boxplot
                upper_idx = np.triu_indices(common_shape[0], k=1)
                vals = avg_mat[upper_idx]
                vals = vals[~np.isnan(vals)]
                for v in vals:
                    boxplot_records.append({
                        'stimulus': stim,
                        'response_range': resp_range,
                        'region': region,
                        'corr': float(v),
                    })

                # For natural sounds, also split into within- vs between-category
                if stim == "naturalSound":
                    n = common_shape[0]
                    expected_n = len(SOUND_CATEGORIES) * EXEMPLARS_PER_CATEGORY
                    if n == expected_n:
                        row_idx, col_idx = np.triu_indices(n, k=1)
                        cat_row = row_idx // EXEMPLARS_PER_CATEGORY
                        cat_col = col_idx // EXEMPLARS_PER_CATEGORY
                        pair_vals = avg_mat[row_idx, col_idx]

                        within_mask = (cat_row == cat_col)
                        between_mask = ~within_mask

                        within_vals = pair_vals[within_mask]
                        between_vals = pair_vals[between_mask]
                        within_vals = within_vals[~np.isnan(within_vals)]
                        between_vals = between_vals[~np.isnan(between_vals)]

                        within_between_records[(resp_range, region, 'within')] = within_vals
                        within_between_records[(resp_range, region, 'between')] = between_vals
                    else:
                        print(f"Skipping within/between split for "
                              f"{region}/{resp_range}: matrix is {n}×{n}, "
                              f"expected {expected_n}×{expected_n}.")

        if not avg_matrices:
            print(f"No valid RDM matrices for {stim}, skipping.")
            continue

        # Shared color scale across all panels of this figure
        all_vals = np.concatenate([m[~np.isnan(m)].ravel()
                                   for m in avg_matrices.values()])
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))

        # Heatmap grid: rows = response_range, cols = region
        n_rows = len(resp_ranges)
        n_cols = len(regions)
        sample_shape = next(iter(avg_matrices.values())).shape[0]
        fig_w = max(5, n_cols * max(4, 0.4 * sample_shape))
        fig_h = max(4, n_rows * max(4, 0.45 * sample_shape))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)

        last_im = None
        for ri, resp_range in enumerate(resp_ranges):
            for ci, region in enumerate(regions):
                ax = axes[ri][ci]
                key = (resp_range, region)
                if key not in avg_matrices:
                    ax.set_axis_off()
                    ax.set_title(f"{region} | {resp_range}\n(no data)",
                                 fontsize=12)
                    continue

                mat = avg_matrices[key]
                im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax,
                               aspect='auto', origin='upper')
                last_im = im
                ax.set_title(f"{region} | {resp_range}", fontsize=14)
                ax.set_xticks(np.arange(mat.shape[0]))
                ax.set_yticks(np.arange(mat.shape[0]))
                ax.tick_params(axis='x', labelrotation=90, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)

                if ci == 0:
                    ax.set_ylabel(f"{resp_range}\nstimulus", fontsize=12)
                if ri == n_rows - 1:
                    ax.set_xlabel('stimulus', fontsize=12)

        if last_im is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            fig.colorbar(last_im, cax=cbar_ax, label='Pearson r')

        fig.suptitle(f'{stim} — Session-averaged RDM '
                     f'({n_sessions} sessions; rows: response_range, '
                     f'cols: region)',
                     fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 0.9, 0.97])
        plt.savefig(f"{file_path}/RDM/RDM_heatmaps_{stim}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # ── Per-stimulus boxplots ────────────────────────────────────────────
        if not boxplot_records:
            print("No RDM data collected; skipping boxplots.")
            return

        box_df = pd.DataFrame(boxplot_records)
        region_order = sorted(box_df['region'].unique().tolist())
        palette = {reg: region_color_map[reg] for reg in region_order}

        def _mannwhitney_safe(x, y):
            """Run Mann-Whitney U (unpaired); return (stat, p) or (nan, nan)."""
            try:
                if len(x) < 1 or len(y) < 1:
                    return np.nan, np.nan
                res = stats.mannwhitneyu(x, y, alternative='two-sided')
                return res.statistic, res.pvalue
            except ValueError:
                return np.nan, np.nan

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

        for stim in ["AM", "pureTones", "naturalSound"]:
            stim_df = box_df[box_df['stimulus'] == stim]
            if stim_df.empty:
                continue

            present_ranges = stim_df['response_range'].unique().tolist()
            x_order = [r for r in response_range_order if r in present_ranges]
            for r in present_ranges:
                if r not in x_order:
                    x_order.append(r)

            # Build group_values[(resp_range, region)] -> np.array of corr vals
            group_values = {}
            for resp_range in x_order:
                for region in region_order:
                    vals = stim_df[
                        (stim_df['response_range'] == resp_range) &
                        (stim_df['region'] == region)
                        ]['corr'].to_numpy()
                    if vals.size > 0:
                        group_values[(resp_range, region)] = vals

            # Pairwise comparisons within each response_range
            comparisons = []
            for resp_range in x_order:
                available = [r for r in region_order
                             if (resp_range, r) in group_values]
                for i in range(len(available)):
                    for j in range(i + 1, len(available)):
                        reg_a, reg_b = available[i], available[j]
                        s, p = _mannwhitney_safe(
                            group_values[(resp_range, reg_a)],
                            group_values[(resp_range, reg_b)],
                        )
                        comparisons.append({
                            'resp_range': resp_range,
                            'reg_a': reg_a,
                            'reg_b': reg_b,
                            'stat': s,
                            'p_raw': p,
                        })

            # Bonferroni correction across all comparisons for this stimulus
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

            # ── Plot ─────────────────────────────────────────────────────────
            n_x = len(x_order)
            n_reg = len(region_order)
            fig_w = max(6, n_x * max(2.0, 0.9 * n_reg))
            fig, ax = plt.subplots(figsize=(fig_w, 6))

            total_width = 0.8
            box_width = total_width / max(n_reg, 1)
            x_centers = np.arange(1, n_x + 1)

            # Manual boxplots so we know exact x positions for brackets
            box_positions = {}  # (resp_range, region) -> x position
            all_vals_flat = []
            for ri, resp_range in enumerate(x_order):
                for gi, region in enumerate(region_order):
                    key = (resp_range, region)
                    if key not in group_values:
                        continue
                    vals = group_values[key]
                    all_vals_flat.append(vals)
                    offset = (gi - (n_reg - 1) / 2) * box_width
                    pos = x_centers[ri] + offset
                    box_positions[key] = pos
                    bp = ax.boxplot([vals], positions=[pos],
                                    widths=box_width * 0.9,
                                    patch_artist=True, notch=False,
                                    medianprops=dict(color='black', linewidth=2))
                    for patch in bp['boxes']:
                        patch.set_facecolor(palette[region])
                        patch.set_alpha(0.7)

            # Y-axis range for bracket placement
            y_max = max(v.max() for v in all_vals_flat)
            y_min = min(v.min() for v in all_vals_flat)
            y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

            bracket_height = y_range * 0.04
            bracket_gap = y_range * 0.02
            base_y = y_max + y_range * 0.04

            # Stack brackets per response_range
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
            ax.set_xticklabels(x_order, fontsize=10)
            ax.set_xlim(0.5, n_x + 0.5)
            ax.set_xlabel('Response range')
            ax.set_ylabel('Pearson correlation (upper triangle)')
            ax.set_title(f'{stim} — RDM upper-triangle correlations by '
                         f'response range and brain region\n'
                         f'(pairwise Mann-Whitney U within response_range, '
                         f'Bonferroni-corrected; * p<0.05, ** p<0.01, *** p<0.001)')

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=palette[reg], alpha=0.7, label=reg)
                               for reg in region_order]
            ax.legend(handles=legend_elements, loc='lower right', title='Region')

            plt.tight_layout()
            plt.savefig(f"{file_path}/RDM/RDM_upper_triangle_boxplot_{stim}_stats.png",
                        dpi=300, bbox_inches='tight')
            plt.show()

            # ── Console summary ──────────────────────────────────────────────
            print(f"\n{stim} — Pairwise region comparisons within each "
                  f"response_range (Mann-Whitney U, Bonferroni-corrected):")
            header = (f"{'response_range':<14} {'region A':<12} {'region B':<12} "
                      f"{'U':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")
            print(header)
            print("-" * len(header))
            for c in comparisons:
                def _fmt(v, width, prec=4):
                    if np.isnan(v):
                        return f"{'n/a':>{width}}"
                    return f"{v:>{width}.{prec}f}"

                print(f"{str(c['resp_range']):<14} {str(c['reg_a']):<12} "
                      f"{str(c['reg_b']):<12} "
                      f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                      f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

        # ── Natural-sound within- vs between-category boxplots ───────────
        if not within_between_records:
            continue

        wb_regions = sorted({k[1] for k in within_between_records})
        wb_resp_ranges_present = {k[0] for k in within_between_records}
        wb_resp_ranges = [r for r in response_range_order
                          if r in wb_resp_ranges_present]

        pair_type_order = ['within', 'between']
        pair_type_colors = {
            'within': '#4C72B0',  # blue
            'between': '#DD8452',  # orange
        }

        for resp_range in wb_resp_ranges:
            # Filter to regions that have both within and between data here
            available_regions = [
                reg for reg in wb_regions
                if (resp_range, reg, 'within') in within_between_records
                   and (resp_range, reg, 'between') in within_between_records
            ]
            if not available_regions:
                continue

            # ── Build comparison lists ──────────────────────────────────
            # 1) within-vs-between per region
            wb_comparisons = []
            for reg in available_regions:
                w = within_between_records[(resp_range, reg, 'within')]
                b = within_between_records[(resp_range, reg, 'between')]
                s, p = _mannwhitney_safe(w, b)
                wb_comparisons.append({
                    'kind': 'within_vs_between',
                    'region': reg, 'stat': s, 'p_raw': p,
                    'n_within': len(w), 'n_between': len(b),
                })

            # 2) region-vs-region for within values
            within_region_comps = []
            for i in range(len(available_regions)):
                for j in range(i + 1, len(available_regions)):
                    reg_a, reg_b = available_regions[i], available_regions[j]
                    x = within_between_records[(resp_range, reg_a, 'within')]
                    y = within_between_records[(resp_range, reg_b, 'within')]
                    s, p = _mannwhitney_safe(x, y)
                    within_region_comps.append({
                        'kind': 'within_region_vs_region',
                        'reg_a': reg_a, 'reg_b': reg_b,
                        'stat': s, 'p_raw': p,
                        'n_a': len(x), 'n_b': len(y),
                    })

            # 3) region-vs-region for between values
            between_region_comps = []
            for i in range(len(available_regions)):
                for j in range(i + 1, len(available_regions)):
                    reg_a, reg_b = available_regions[i], available_regions[j]
                    x = within_between_records[(resp_range, reg_a, 'between')]
                    y = within_between_records[(resp_range, reg_b, 'between')]
                    s, p = _mannwhitney_safe(x, y)
                    between_region_comps.append({
                        'kind': 'between_region_vs_region',
                        'reg_a': reg_a, 'reg_b': reg_b,
                        'stat': s, 'p_raw': p,
                        'n_a': len(x), 'n_b': len(y),
                    })

            # Bonferroni-correct across ALL comparisons in this figure
            all_comps = wb_comparisons + within_region_comps + between_region_comps
            raw_p = np.array([c['p_raw'] for c in all_comps], dtype=float)
            corr_p = np.full_like(raw_p, np.nan)
            reject = np.zeros_like(raw_p, dtype=bool)
            valid = ~np.isnan(raw_p)
            if valid.sum() > 0:
                rej, pcorr, _, _ = multipletests(raw_p[valid], alpha=0.05,
                                                 method='bonferroni')
                corr_p[valid] = pcorr
                reject[valid] = rej
            for i, c in enumerate(all_comps):
                c['p_corr'] = corr_p[i]
                c['reject'] = bool(reject[i])

            # ── Plot ────────────────────────────────────────────────────
            n_reg = len(available_regions)
            fig_w = max(6, n_reg * 1.8)
            fig, ax = plt.subplots(figsize=(fig_w, 7))

            box_width = 0.35
            x_centers = np.arange(1, n_reg + 1)
            box_positions = {}  # (region, pair_type) -> x pos
            all_vals_flat = []

            for ri, reg in enumerate(available_regions):
                for gi, pt in enumerate(pair_type_order):
                    vals = within_between_records[(resp_range, reg, pt)]
                    if len(vals) == 0:
                        continue
                    offset = (gi - 0.5) * box_width
                    pos = x_centers[ri] + offset
                    box_positions[(reg, pt)] = pos
                    all_vals_flat.append(vals)
                    bp = ax.boxplot(
                        [vals], positions=[pos],
                        widths=box_width * 0.9,
                        patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2),
                    )
                    for patch in bp['boxes']:
                        patch.set_facecolor(pair_type_colors[pt])
                        patch.set_alpha(0.7)

            y_max = max(v.max() for v in all_vals_flat)
            y_min = min(v.min() for v in all_vals_flat)
            y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
            bracket_height = y_range * 0.035
            bracket_gap = y_range * 0.015
            base_y = y_max + y_range * 0.04

            kind_colors = {
                'within_vs_between': 'black',
                'within_region_vs_region': pair_type_colors['within'],
                'between_region_vs_region': pair_type_colors['between'],
            }

            current_y = base_y

            # Layer 1: within-vs-between per region (one row of brackets)
            for c in wb_comparisons:
                reg = c['region']
                key_w = (reg, 'within')
                key_b = (reg, 'between')
                if key_w not in box_positions or key_b not in box_positions:
                    continue
                x_a = box_positions[key_w]
                x_b = box_positions[key_b]
                y_bottom = current_y
                y_top = y_bottom + bracket_height
                color = kind_colors[c['kind']]
                ax.plot([x_a, x_a, x_b, x_b],
                        [y_bottom, y_top, y_top, y_bottom],
                        color=color, linewidth=0.8)
                ax.text((x_a + x_b) / 2, y_top, _sig_str(c['p_corr']),
                        ha='center', va='bottom', fontsize=9,
                        color='#d62728' if c['reject'] else color)
            max_top_layer1 = current_y + bracket_height + y_range * 0.04

            # Layer 2: stacked region-vs-region for within values
            level_y = max_top_layer1
            for level, c in enumerate(within_region_comps):
                key_a = (c['reg_a'], 'within')
                key_b = (c['reg_b'], 'within')
                if key_a not in box_positions or key_b not in box_positions:
                    continue
                x_a = box_positions[key_a]
                x_b = box_positions[key_b]
                y_bottom = level_y + level * (bracket_height + bracket_gap)
                y_top = y_bottom + bracket_height
                color = kind_colors[c['kind']]
                ax.plot([x_a, x_a, x_b, x_b],
                        [y_bottom, y_top, y_top, y_bottom],
                        color=color, linewidth=0.8)
                ax.text((x_a + x_b) / 2, y_top, _sig_str(c['p_corr']),
                        ha='center', va='bottom', fontsize=9,
                        color='#d62728' if c['reject'] else color)
            if within_region_comps:
                level_y = (level_y
                           + len(within_region_comps)
                           * (bracket_height + bracket_gap)
                           + y_range * 0.03)

            # Layer 3: stacked region-vs-region for between values
            for level, c in enumerate(between_region_comps):
                key_a = (c['reg_a'], 'between')
                key_b = (c['reg_b'], 'between')
                if key_a not in box_positions or key_b not in box_positions:
                    continue
                x_a = box_positions[key_a]
                x_b = box_positions[key_b]
                y_bottom = level_y + level * (bracket_height + bracket_gap)
                y_top = y_bottom + bracket_height
                color = kind_colors[c['kind']]
                ax.plot([x_a, x_a, x_b, x_b],
                        [y_bottom, y_top, y_top, y_bottom],
                        color=color, linewidth=0.8)
                ax.text((x_a + x_b) / 2, y_top, _sig_str(c['p_corr']),
                        ha='center', va='bottom', fontsize=9,
                        color='#d62728' if c['reject'] else color)
            top_y = level_y
            if between_region_comps:
                top_y = (level_y
                         + len(between_region_comps)
                         * (bracket_height + bracket_gap))

            ax.set_ylim(top=max(y_max + y_range * 0.15,
                                top_y + y_range * 0.05))
            ax.set_xticks(x_centers)
            ax.set_xticklabels(available_regions, fontsize=10,
                               rotation=30, ha='right')
            ax.set_xlim(0.5, n_reg + 0.5)
            ax.set_xlabel('Brain region')
            ax.set_ylabel('Pearson correlation (upper triangle)')
            ax.set_title(
                f'naturalSound — Within- vs between-category correlations '
                f'({resp_range})\n'
                f'(Mann-Whitney U; Bonferroni-corrected across all brackets; '
                f'* p<0.05, ** p<0.01, *** p<0.001)'
            )

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=pair_type_colors['within'], alpha=0.7,
                      label='Within category'),
                Patch(facecolor=pair_type_colors['between'], alpha=0.7,
                      label='Between category'),
                Line2D([0], [0], color='black', lw=1,
                       label='Within vs between (per region)'),
                Line2D([0], [0], color=pair_type_colors['within'], lw=1,
                       label='Region vs region (within)'),
                Line2D([0], [0], color=pair_type_colors['between'], lw=1,
                       label='Region vs region (between)'),
            ]
            ax.legend(handles=legend_elements, loc='lower right',
                      fontsize=8, title='Boxes / brackets')

            plt.tight_layout()
            plt.savefig(
                f"{file_path}/RDM/RDM_within_vs_between_naturalSound_"
                f"{resp_range}.png",
                dpi=300, bbox_inches='tight',
            )
            plt.show()

            # ── Console summaries ───────────────────────────────────────
            def _fmt(v, width, prec=4):
                if np.isnan(v):
                    return f"{'n/a':>{width}}"
                return f"{v:>{width}.{prec}f}"

            print(f"\nnaturalSound — Within vs between category per region "
                  f"({resp_range}; Mann-Whitney U, Bonferroni-corrected "
                  f"across all brackets in this figure):")
            header = (f"{'region':<28} {'n_within':>8} {'n_between':>10} "
                      f"{'U':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")
            print(header)
            print("-" * len(header))
            for c in wb_comparisons:
                print(f"{str(c['region']):<28} {c['n_within']:>8d} "
                      f"{c['n_between']:>10d} "
                      f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                      f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

            print(f"\nnaturalSound — Region-vs-region for WITHIN-category "
                  f"values ({resp_range}):")
            header2 = (f"{'region A':<28} {'region B':<28} "
                       f"{'U':>10} {'p_raw':>12} {'p_corr':>12} {'sig':>5}")
            print(header2)
            print("-" * len(header2))
            for c in within_region_comps:
                print(f"{str(c['reg_a']):<28} {str(c['reg_b']):<28} "
                      f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                      f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")

            print(f"\nnaturalSound — Region-vs-region for BETWEEN-category "
                  f"values ({resp_range}):")
            print(header2)
            print("-" * len(header2))
            for c in between_region_comps:
                print(f"{str(c['reg_a']):<28} {str(c['reg_b']):<28} "
                      f"{_fmt(c['stat'], 10, 2)} {_fmt(c['p_raw'], 12)} "
                      f"{_fmt(c['p_corr'], 12)} {_sig_str(c['p_corr']):>5}")




print("Creating upper triangle boxplots with stats...")
create_upper_triangle_boxplots()

print("Creating pairwise accuracy heatmaps...")
create_accuracy_heatmaps()

print("Creating RDM heatmaps and combined boxplot...")
create_RDM_plots()