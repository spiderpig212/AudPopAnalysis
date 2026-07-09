import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from itertools import combinations

from analysis_class import FiringRateAnalysis

neuron_threshold = 40
fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path

null_results_df = pd.read_feather(f"{file_path}/CCA_cross_region_projection_similarity.feather")

# If target1 or target2 is Temporal association areas, drop that row from the data frame
if 'Temporal association areas' in null_results_df['target1'].unique() or 'Temporal association areas' in null_results_df['target2'].unique():
    null_results_df = null_results_df[~null_results_df['target1'].isin(['Temporal association areas'])]
    null_results_df = null_results_df[~null_results_df['target2'].isin(['Temporal association areas'])]

stim_types = ['AM', 'pureTones', 'naturalSound']

null_results_df['target_pair'] = null_results_df['target1'] + ' vs ' + null_results_df['target2']


def p_value_to_stars(p):
    """Convert a p-value into the conventional significance star notation."""
    if p <= 1e-4:
        return '****'
    elif p <= 1e-3:
        return '***'
    elif p <= 1e-2:
        return '**'
    elif p <= 5e-2:
        return '*'
    else:
        return 'ns'


def compute_pairwise_stats(subset, response_ranges, target_pairs):
    """
    Perform Mann-Whitney U tests between every pair of target_pair groups
    within each response_range, then apply Bonferroni correction across all
    tests in this subset.

    Returns a list of dicts, each describing one comparison:
        {
            'response_range': rr,
            'group1': tp1,
            'group2': tp2,
            'U': U statistic,
            'p_raw': uncorrected p-value,
            'p_corrected': Bonferroni-corrected p-value,
            'stars': significance annotation,
        }
    """
    results = []

    for rr in response_ranges:
        rr_data = subset[subset['response_range'] == rr]

        # Collect the comparisons for THIS response range only, so the
        # Bonferroni family size is the number of region-pair comparisons
        # within a single response range (3 pairs -> divide by 3).
        rr_results = []
        for tp1, tp2 in combinations(target_pairs, 2):
            group1 = rr_data[rr_data['target_pair'] == tp1]['z_score_diff_from_null'].dropna().values
            group2 = rr_data[rr_data['target_pair'] == tp2]['z_score_diff_from_null'].dropna().values

            # Skip comparisons where either group has no data
            if len(group1) == 0 or len(group2) == 0:
                continue

            U, p_raw = stats.mannwhitneyu(group1, group2, alternative='two-sided')

            rr_results.append({
                'response_range': rr,
                'group1': tp1,
                'group2': tp2,
                'n1': len(group1),
                'n2': len(group2),
                'U': U,
                'p_raw': p_raw,
            })

        # Bonferroni correction within this response range only
        n_tests = len(rr_results)
        for res in rr_results:
            p_corr = min(res['p_raw'] * n_tests, 1.0) if n_tests > 0 else res['p_raw']
            res['p_corrected'] = p_corr
            res['stars'] = p_value_to_stars(p_corr)

        results.extend(rr_results)

    return results


def annotate_comparisons(ax, subset, stat_results, response_ranges, target_pairs):
    """
    Draw significance brackets above the boxes based on pre-computed stats.

    Bracket x-positions are derived from seaborn's grouped boxplot geometry:
    each response_range occupies an integer tick, and hues (target_pairs) are
    evenly spread around that tick within a total width of 0.8.
    """
    n_hues = len(target_pairs)
    total_width = 0.8
    box_width = total_width / n_hues

    # Fixed baseline for significance brackets (independent of data range)
    bracket_baseline = 4.0

    # Map each (response_range, target_pair) to its x center on the axis
    def box_center(rr_index, tp_index):
        # Offset of the leftmost hue box relative to the tick center
        left_edge = -total_width / 2 + box_width / 2
        return rr_index + left_edge + tp_index * box_width

    rr_to_index = {rr: i for i, rr in enumerate(response_ranges)}
    tp_to_index = {tp: i for i, tp in enumerate(target_pairs)}

    # Determine spacing for the brackets relative to the data range
    y_max = subset['z_score_diff_from_null'].max()
    y_min = subset['z_score_diff_from_null'].min()
    y_range = y_max - y_min if y_max > y_min else 1.0

    line_offset = y_range * 0.05  # vertical gap between stacked brackets
    bracket_height = y_range * 0.02

    # Track how many brackets we've stacked per response_range so they don't overlap
    stack_count = {rr: 0 for rr in response_ranges}

    for res in stat_results:
        # Only draw significant comparisons to keep the plot readable;
        # remove this check if you want every comparison labeled.
        if res['stars'] == 'ns':
            continue

        rr = res['response_range']
        rr_idx = rr_to_index[rr]
        x1 = box_center(rr_idx, tp_to_index[res['group1']])
        x2 = box_center(rr_idx, tp_to_index[res['group2']])

        level = stack_count[rr]
        y = bracket_baseline + line_offset * (level + 1)
        stack_count[rr] += 1

        # Draw the bracket
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_height, y + bracket_height, y],
            lw=1.0,
            c='black',
        )
        # Draw the significance text centered above the bracket
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height,
            res['stars'],
            ha='center',
            va='bottom',
            fontsize=9,
        )


fig, axes = plt.subplots(1, len(stim_types), figsize=(7 * len(stim_types), 8), sharey=True)

# Collect all stats so they can be inspected/saved after plotting
all_stats = []

for ax, stim in zip(axes, stim_types):
    subset = null_results_df[null_results_df['stimulus'] == stim]

    if subset.empty:
        ax.set_title(f'{stim}\n(No data)')
        continue

    response_ranges = sorted(subset['response_range'].unique())
    target_pairs = sorted(subset['target_pair'].unique())

    sns.boxplot(
        data=subset,
        x='response_range',
        y='z_score_diff_from_null',
        hue='target_pair',
        order=response_ranges,
        hue_order=target_pairs,
        ax=ax,
        palette='Set2',
        linewidth=1.2,
        legend=(ax is axes[0]),  # only the first axis builds a legend
    )

    ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axhline(-2, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_title(stim, fontsize=13, fontweight='bold')
    ax.set_xlabel('Response Range', fontsize=11)
    ax.set_ylabel('Z-score (diff from null)' if ax == axes[0] else '', fontsize=11)
    ax.tick_params(axis='x', rotation=30)

    # --- Manual statistics ---
    stat_results = compute_pairwise_stats(subset, response_ranges, target_pairs)

    # Keep a record tagged with the stimulus for later inspection/export
    for res in stat_results:
        record = dict(res)
        record['stimulus'] = stim
        all_stats.append(record)

    # Draw the significance brackets ourselves
    annotate_comparisons(ax, subset, stat_results, response_ranges, target_pairs)

# Grab handles/labels from the first axis (the only one that has a legend)
handles, labels = axes[0].get_legend_handles_labels()

# Remove the per-axis legend so it doesn't appear twice
leg = axes[0].get_legend()
if leg is not None:
    leg.remove()

null_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.0, alpha=0.7)
handles.append(null_handle)
labels.append('Null (z=0)')
# labels.append('0.95 (z=-2)')

plt.ylim(-12, 6)

fig.legend(
    handles, labels,
    title='Target Pair',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=len(labels),
    fontsize=9,
    title_fontsize=10,
    frameon=True,
)

plt.suptitle('Subspace Alignment vs Null Distribution', fontsize=15, fontweight='bold')
# plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # reserve room for legend + suptitle
plt.savefig(f"{file_path}/CCA_two_region_analysis/CCA_null_distribution_boxplots_test3.png", dpi=300, bbox_inches='tight')
plt.show()

# Optionally, save the full statistics table so the numbers can be verified
stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(f"{file_path}/CCA_two_region_analysis/CCA_null_pairwise_stats.csv", index=False)
print(stats_df)