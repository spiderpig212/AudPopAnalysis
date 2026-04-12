import os
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from statannotations.Annotator import Annotator
from itertools import combinations

from analysis_class import FiringRateAnalysis

# neuron_threshold = 40
# fr_db = FiringRateAnalysis(db_suffix="coords_updated")
# file_path = fr_db.figdata_path
file_path = "/Users/matt/Desktop/Research/Murray/data/ahmadian/figuresdata/2025acpop"

null_results_df = pd.read_feather(f"{file_path}/CCA_cross_region_projection_similarity.feather")
stim_types = ['AM', 'pureTones', 'naturalSound']

null_results_df['target_pair'] = null_results_df['target1'] + ' vs ' + null_results_df['target2']

fig, axes = plt.subplots(1, len(stim_types), figsize=(7 * len(stim_types), 6), sharey=True)

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
        legend=False,
    )

    ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_title(stim, fontsize=13, fontweight='bold')
    ax.set_xlabel('Response Range', fontsize=11)
    ax.set_ylabel('Z-score (diff from null)' if ax == axes[0] else '', fontsize=11)
    ax.tick_params(axis='x', rotation=30)

    # Build pairwise comparison list: all target_pair combos within each response_range
    annotation_pairs = [
        ((rr, tp1), (rr, tp2))
        for rr in response_ranges
        for tp1, tp2 in combinations(target_pairs, 2)
    ]

    # Annotate with Mann-Whitney U + Bonferroni correction
    annotator = Annotator(
        ax,
        annotation_pairs,
        data=subset,
        x='response_range',
        y='z_score_diff_from_null',
        hue='target_pair',
        order=response_ranges,
        hue_order=target_pairs,
    )
    annotator.configure(
        test='Mann-Whitney',
        comparisons_correction='bonferroni',
        text_format='star',       # shows ns / * / ** / ***
        loc='outside',            # brackets above the boxes
        verbose=0,
    )
    annotator.apply_and_annotate()

# Shared legend from the last non-empty axis
# last_ax = next(ax for ax in reversed(axes) if ax.get_legend_handles_labels()[0])
# handles, labels = last_ax.get_legend_handles_labels()

# null_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.0, alpha=0.7)
# handles.append(null_handle)
# labels.append('Null (z=0)')

fig.legend(
    # handles, labels,
    title='Target Pair',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.12),
    # ncol=len(labels),
    fontsize=9,
    title_fontsize=10,
    frameon=True,
)

plt.suptitle('Subspace Alignment vs Null Distribution', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{file_path}/CCA_two_region_analysis/CCA_null_distribution_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()

