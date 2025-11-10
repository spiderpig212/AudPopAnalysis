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

# Create abbreviated labels for the legend to save space
region_abbreviations = {}
unique_regions = df['region_comparison'].unique()
for i, region in enumerate(unique_regions):
    # Create shorter labels - you can customize this mapping
    parts = region.replace(' vs ', '_vs_').replace(' and ', '_and_').split()
    if len(parts) > 10:
        # Use first letters of each word for very long names
        abbrev = ''.join([word[0] for word in parts[:6]])  # Take first 6 words
    else:
        abbrev = region
    region_abbreviations[region] = f"R{i+1}: {abbrev}"

# Add abbreviated column for plotting
df['region_abbrev'] = df['region_comparison'].map(region_abbreviations)

# Create the boxplot with larger figure size
plt.figure(figsize=(16, 10))

# Create boxplot with response_range on x-axis, SSA_overlap on y-axis, colored by region_comparison
sns.boxplot(data=df, x='response_range', y='SSA_overlap', hue='region_abbrev')

# Customize the plot
plt.title('SSA Overlap Analysis by Response Range and Region Comparison', fontsize=18, pad=30)
plt.xlabel('Response Range', fontsize=16)
plt.ylabel('SSA Overlap Value', fontsize=16)

# Position legend below the plot
plt.legend(title='Region Comparison', bbox_to_anchor=(0.5, -0.15), loc='upper center',
          ncol=2, fontsize=12, title_fontsize=14)

# Add grid for better readability
plt.grid(True, alpha=0.3, axis='y')

# Adjust layout to accommodate legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Make room for legend

# Save the plot
plt.savefig(f"{file_path}/subspace_overlap_analysis/SSA_overlap_boxplot.png",
           dpi=300, bbox_inches='tight')
plt.show()

# Create a more detailed plot with individual data points
plt.figure(figsize=(18, 10))

# Create boxplot with swarm plot overlay to show individual points
ax = sns.boxplot(data=df, x='response_range', y='SSA_overlap', hue='region_abbrev')
sns.stripplot(data=df, x='response_range', y='SSA_overlap', hue='region_abbrev',
              dodge=True, size=4, alpha=0.7, ax=ax)

# Remove duplicate legend entries from stripplot
handles, labels = ax.get_legend_handles_labels()
n_regions = len(df['region_abbrev'].unique())
ax.legend(handles[:n_regions], labels[:n_regions], title='Region Comparison',
         bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=3,
         fontsize=12, title_fontsize=14)

plt.title('SSA Overlap Analysis by Response Range and Region Comparison\n(with individual data points)',
          fontsize=18, pad=30)
plt.xlabel('Response Range', fontsize=16)
plt.ylabel('SSA Overlap Value', fontsize=16)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)  # Make room for legend

# Save the detailed plot
plt.savefig(f"{file_path}/subspace_overlap_analysis/SSA_overlap_boxplot_detailed.png",
           dpi=300, bbox_inches='tight')
plt.show()

# Optional: Create separate plots for each stimulus type if there are multiple
if 'stimulus' in df.columns and len(df['stimulus'].unique()) > 1:
    stimuli = df['stimulus'].unique()

    # Make plots much larger
    fig, axes = plt.subplots(1, len(stimuli), figsize=(12*len(stimuli), 10))
    if len(stimuli) == 1:
        axes = [axes]

    for i, stim in enumerate(stimuli):
        stim_data = df[df['stimulus'] == stim]

        sns.boxplot(data=stim_data, x='response_range', y='SSA_overlap',
                   hue='region_abbrev', ax=axes[i])

        axes[i].set_title(f'SSA Overlap - {stim}', fontsize=16)
        axes[i].set_xlabel('Response Range', fontsize=14)
        axes[i].set_ylabel('SSA Overlap Value', fontsize=14)
        axes[i].grid(True, alpha=0.3, axis='y')

        # Remove individual legends from each subplot
        axes[i].get_legend().remove()

    # Add a single legend for the entire figure, positioned at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Region Comparison',
              bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol=3,
              fontsize=12, title_fontsize=14)

    plt.suptitle('SSA Overlap Analysis by Stimulus Type', fontsize=20, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for legend at top

    plt.savefig(f"{file_path}/subspace_overlap_analysis/SSA_overlap_by_stimulus.png",
               dpi=300, bbox_inches='tight')
    plt.show()

# Print the abbreviation mapping for reference
print("\nRegion abbreviation mapping:")
for original, abbrev in region_abbreviations.items():
    print(f"{abbrev} = {original}")

# Print some summary statistics
print("\nSummary statistics by response range:")
print(df.groupby('response_range')['SSA_overlap'].describe())

print("\nSummary statistics by region comparison:")
print(df.groupby('region_comparison')['SSA_overlap'].describe())

print("\nPlots saved to subspace_overlap_analysis directory")