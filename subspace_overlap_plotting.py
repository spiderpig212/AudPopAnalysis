"""
Script to visualize SSA overlap analysis results with boxplots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from analysis_class import FiringRateAnalysis

# Load the data
fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path

# Load the SSA overlap analysis dataframe
df = pd.read_csv(f"{file_path}/subspace_overlap_analysis/SSA_overlap_analysis.csv")

print(f"Loaded dataframe with {len(df)} rows")
print(f"Unique response ranges: {df['response_range'].unique()}")
print(f"Unique region comparisons: {df['region_comparison'].unique()}")

# Create the boxplot
plt.figure(figsize=(14, 8))

# Create boxplot with response_range on x-axis, SSA_overlap on y-axis, colored by region_comparison
sns.boxplot(data=df, x='response_range', y='SSA_overlap', hue='region_comparison')

# Customize the plot
plt.title('SSA Overlap Analysis by Response Range and Region Comparison', fontsize=16, pad=20)
plt.xlabel('Response Range', fontsize=14)
plt.ylabel('SSA Overlap Value', fontsize=14)

# Rotate legend if there are many region comparisons
plt.legend(title='Region Comparison', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid for better readability
plt.grid(True, alpha=0.3, axis='y')

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save the plot
plt.savefig(f"{file_path}/subspace_overlap_analysis/SSA_overlap_boxplot.png",
            dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print some summary statistics
print("\nSummary statistics by response range:")
print(df.groupby('response_range')['SSA_overlap'].describe())

print("\nSummary statistics by region comparison:")
print(df.groupby('region_comparison')['SSA_overlap'].describe())

# Create a more detailed plot with individual data points
plt.figure(figsize=(16, 8))

# Create boxplot with swarm plot overlay to show individual points
ax = sns.boxplot(data=df, x='response_range', y='SSA_overlap', hue='region_comparison')
sns.stripplot(data=df, x='response_range', y='SSA_overlap', hue='region_comparison',
              dodge=True, size=4, alpha=0.7, ax=ax)

# Remove duplicate legend entries from stripplot
handles, labels = ax.get_legend_handles_labels()
n_regions = len(df['region_comparison'].unique())
ax.legend(handles[:n_regions], labels[:n_regions], title='Region Comparison',
          bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('SSA Overlap Analysis by Response Range and Region Comparison\n(with individual data points)',
          fontsize=16, pad=20)
plt.xlabel('Response Range', fontsize=14)
plt.ylabel('SSA Overlap Value', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Save the detailed plot
plt.savefig(f"{file_path}/subspace_overlap_analysis/SSA_overlap_boxplot_detailed.png",
            dpi=300, bbox_inches='tight')
plt.show()

# Optional: Create separate plots for each stimulus type if there are multiple
if 'stimulus' in df.columns and len(df['stimulus'].unique()) > 1:
    stimuli = df['stimulus'].unique()

    fig, axes = plt.subplots(1, len(stimuli), figsize=(6 * len(stimuli), 8))
    if len(stimuli) == 1:
        axes = [axes]

    for i, stim in enumerate(stimuli):
        stim_data = df[df['stimulus'] == stim]

        sns.boxplot(data=stim_data, x='response_range', y='SSA_overlap',
                    hue='region_comparison', ax=axes[i])

        axes[i].set_title(f'SSA Overlap - {stim}', fontsize=14)
        axes[i].set_xlabel('Response Range', fontsize=12)
        axes[i].set_ylabel('SSA Overlap Value', fontsize=12)
        axes[i].grid(True, alpha=0.3, axis='y')

        # Only show legend on first plot to avoid clutter
        if i == 0:
            axes[i].legend(title='Region Comparison', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[i].get_legend().remove()

    plt.suptitle('SSA Overlap Analysis by Stimulus Type', fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig(f"{file_path}/subspace_overlap_analysis/SSA_overlap_by_stimulus.png",
                dpi=300, bbox_inches='tight')
    plt.show()

print("\nPlots saved to subspace_overlap_analysis directory")