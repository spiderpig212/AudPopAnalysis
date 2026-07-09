import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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


def rank_biserial_from_u(U, n1, n2):
    """
    Rank-biserial correlation as an effect size for the Mann-Whitney U test.

    r = 1 - (2U) / (n1 * n2)

    Ranges from -1 to +1. The sign depends on which group's scipy chose as the
    reference for U; the magnitude is what matters for reporting effect size.
    A magnitude near 1 indicates near-complete separation of the two groups,
    while 0 indicates complete overlap.
    """
    if n1 == 0 or n2 == 0:
        return np.nan
    return 1.0 - (2.0 * U) / (n1 * n2)


def compute_pairwise_stats(subset, response_ranges, target_pairs):
    """
    Perform Mann-Whitney U tests between every pair of target_pair groups
    within each response_range. Bonferroni correction is applied *within*
    each response_range (family size = number of region-pair comparisons in
    that response range).

    Adds sample sizes and rank-biserial effect size to each record so results
    can be reported descriptively even when significance is unreachable at
    small n.
    """
    results = []

    for rr in response_ranges:
        rr_data = subset[subset['response_range'] == rr]

        rr_results = []
        for tp1, tp2 in combinations(target_pairs, 2):
            group1 = rr_data[rr_data['target_pair'] == tp1]['z_score_diff_from_null'].dropna().values
            group2 = rr_data[rr_data['target_pair'] == tp2]['z_score_diff_from_null'].dropna().values

            # Skip comparisons where either group has no data
            if len(group1) == 0 or len(group2) == 0:
                continue

            n1, n2 = len(group1), len(group2)
            U, p_raw = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            effect_size = rank_biserial_from_u(U, n1, n2)

            rr_results.append({
                'response_range': rr,
                'group1': tp1,
                'group2': tp2,
                'n1': n1,
                'n2': n2,
                'U': U,
                'p_raw': p_raw,
                'rank_biserial': effect_size,
            })

        # Bonferroni correction within this response range only
        n_tests = len(rr_results)
        for res in rr_results:
            p_corr = min(res['p_raw'] * n_tests, 1.0) if n_tests > 0 else res['p_raw']
            res['p_corrected'] = p_corr
            res['stars'] = p_value_to_stars(p_corr)

        results.extend(rr_results)

    return results


def overlay_mean_sem(ax, subset, response_ranges, target_pairs):
    """
    Overlay mean +/- SEM markers on top of a seaborn stripplot.

    Recreates seaborn's dodge geometry so the mean/SEM markers line up with
    each hue group. seaborn spreads the hue groups across a total width of 0.8
    centered on each categorical tick.
    """
    n_hues = len(target_pairs)
    total_width = 0.8
    group_width = total_width / n_hues

    def group_center(rr_index, tp_index):
        left_edge = -total_width / 2 + group_width / 2
        return rr_index + left_edge + tp_index * group_width

    for rr_idx, rr in enumerate(response_ranges):
        rr_data = subset[subset['response_range'] == rr]
        for tp_idx, tp in enumerate(target_pairs):
            vals = rr_data[rr_data['target_pair'] == tp]['z_score_diff_from_null'].dropna().values
            if len(vals) == 0:
                continue

            x = group_center(rr_idx, tp_idx)
            mean = np.mean(vals)
            # SEM is undefined for a single point; guard against n==1
            sem = stats.sem(vals) if len(vals) > 1 else 0.0

            # Mean marker (short horizontal bar) + SEM error bar
            ax.errorbar(
                x, mean,
                yerr=sem,
                fmt='_',
                color='black',
                markersize=18,
                markeredgewidth=2.0,
                elinewidth=1.5,
                capsize=5,
                capthick=1.5,
                zorder=10,
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

    # Stripplot shows every individual data point -- honest for small n
    sns.stripplot(
        data=subset,
        x='response_range',
        y='z_score_diff_from_null',
        hue='target_pair',
        order=response_ranges,
        hue_order=target_pairs,
        ax=ax,
        palette='Set2',
        dodge=True,
        jitter=0.15,
        size=7,
        alpha=0.8,
        edgecolor='gray',
        linewidth=0.5,
        legend=(ax is axes[0]),  # only the first axis builds a legend
    )

    # Overlay mean +/- SEM on top of the points
    overlay_mean_sem(ax, subset, response_ranges, target_pairs)

    ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axhline(-2, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_title(stim, fontsize=13, fontweight='bold')
    ax.set_xlabel('Response Range', fontsize=11)
    ax.set_ylabel('Z-score (diff from null)' if ax == axes[0] else '', fontsize=11)
    ax.tick_params(axis='x', rotation=30)

    # --- Manual statistics (Mann-Whitney U + rank-biserial effect size) ---
    stat_results = compute_pairwise_stats(subset, response_ranges, target_pairs)

    # Keep a record tagged with the stimulus for later inspection/export
    for res in stat_results:
        record = dict(res)
        record['stimulus'] = stim
        all_stats.append(record)

# Grab handles/labels from the first axis (the only one that has a legend)
handles, labels = axes[0].get_legend_handles_labels()

# Remove the per-axis legend so it doesn't appear twice
leg = axes[0].get_legend()
if leg is not None:
    leg.remove()

null_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.0, alpha=0.7)
handles.append(null_handle)
labels.append('Null (z=0)')

# Add a legend entry explaining the mean +/- SEM marker
mean_handle = plt.Line2D(
    [0], [0], color='black', marker='_', linestyle='None',
    markersize=12, markeredgewidth=2.0,
)
handles.append(mean_handle)
labels.append('Mean \u00b1 SEM')

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
plt.savefig(f"{file_path}/CCA_two_region_analysis/CCA_null_distribution_stripplots.png", dpi=300, bbox_inches='tight')
plt.show()

# Save the full statistics table (including effect sizes) for verification
stats_df = pd.DataFrame(all_stats)

# Reorder columns for readability
column_order = [
    'stimulus', 'response_range', 'group1', 'group2',
    'n1', 'n2', 'U', 'rank_biserial',
    'p_raw', 'p_corrected', 'stars',
]
stats_df = stats_df[[c for c in column_order if c in stats_df.columns]]

stats_df.to_csv(f"{file_path}/CCA_two_region_analysis/CCA_null_pairwise_stats.csv", index=False)
print(stats_df.to_string(index=False))