"""
This file contains code for running canonical correlation analysis. Compares responses from different conditions in
various brain areas to see if there is any relationship between primary auditory area and either secondary cortical area.
"""

import os
import sys
import numpy as np
from scipy import stats
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import studyparams
from jaratoolbox import settings, celldatabase
from funcs import add_significance_stars, add_statistical_brackets, participation_ratio

#%% Data import
file_path = settings.FIGURES_DATA_PATH + "/" + studyparams.STUDY_NAME
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["pureTones", "AM", "naturalSound"]  # For now only start with pure tones to try and understand analysis meaning
analysis_attempts = ["correlation", "mean_corr", "PR"]
# stim_types = ["naturalSound", "AM", "pureTones"]

neuron_threshold = 20
figdataPath = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)

for i_stim, stim in enumerate(stim_types):

    if stim == 'AM':
        nTrials = 220
        nCategories = 11
    elif stim == 'naturalSound':
        nTrials = 200
        nCategories = len(studyparams.SOUND_CATEGORIES)
        soundCats = studyparams.SOUND_CATEGORIES
        nInstances = 4
        stimVals = np.empty(nInstances*nCategories, dtype=object)
        for i in range(nCategories):
            for j in range(nInstances):
                stimVals[i*nInstances+j] = soundCats[i] + f"_{j+1}"
    elif stim == 'pureTones':
        nTrials = 320
        nCategories = 16

    stim_arrays = np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    mouseIDArray = stim_arrays["mouseIDArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # Stored the trials for each neuron to make sure they were all the same, but only need one now
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)
    uniqSessions = np.unique(sessionArray)
    correlation_data = []

    for respRange in response_ranges:
        respArray = stim_arrays[f"{respRange}fr"]

        for session in uniqSessions:
            session_mask = sessionArray == session
            session_resp_array = respArray[session_mask, :]
            brain_session_array = brainRegionArray[session_mask]

            for i, brainRegion in enumerate(uniqRegions):
                brainRegion_mask = brain_session_array == brainRegion
                brain_resp_array = session_resp_array[brainRegion_mask, :].T  # Make the array (nTrials, nNeurons)
                region1_sess_count = brain_resp_array.shape[1]
                if region1_sess_count < neuron_threshold:
                    print(f"Skipping region 1: {brainRegion} because it has fewer than {neuron_threshold} neurons (n = {region1_sess_count}), session {session}")
                    continue
                # Grab a random 20 neurons from the array
                region1_neurons = np.random.choice(brain_resp_array.shape[1], size=neuron_threshold, replace=False)
                brain_resp_array = brain_resp_array[:, region1_neurons]

                for brainRegion2 in uniqRegions[i+1:]:
                    brainRegion2_mask = brain_session_array == brainRegion2
                    brain2_resp_array = session_resp_array[brainRegion2_mask, :].T  # Make the array (nTrials, nNeurons)
                    region2_sess_count = brain2_resp_array.shape[1]
                    if region2_sess_count < neuron_threshold:
                        print(f"Skipping region 2: {brainRegion2} because it has fewer than {neuron_threshold} neurons (n = {region2_sess_count}), session {session}")
                        continue
                    region2_neurons = np.random.choice(brain2_resp_array.shape[1], size=neuron_threshold, replace=False)
                    brain2_resp_array = brain2_resp_array[:, region2_neurons]

                    n_components = np.min([brain_resp_array.shape[1], brain2_resp_array.shape[1]])  # Whichever region has fewer neurons (should always be equal to neuron threshold now)
                    cca = CCA(n_components=n_components)
                    response_transform = cca.fit_transform(brain_resp_array, brain2_resp_array)

                    correlation_val = np.corrcoef(response_transform[0][:, 0], response_transform[1][:, 0])[0, 1]

                    all_corr_vals = []
                    for cca_component in range(neuron_threshold):
                        all_corr_vals.append(np.corrcoef(response_transform[0][:, cca_component], response_transform[1][:, cca_component])[0, 1])
                    corr_array = np.array(all_corr_vals)
                    partic_ratio = participation_ratio(corr_array)
                    mean_corr_val = np.mean(corr_array[:int(np.ceil(partic_ratio))])  # Doing the ceiling because the last value is not inclusive so this will grab lowest whole number less than PR value

                    correlation_data.append({
                        'region_pair': f"{brainRegion}_vs_{brainRegion2}",
                        'region1': brainRegion,
                        'region2': brainRegion2,
                        'correlation': correlation_val,
                        'mean_corr': mean_corr_val,
                        'corr_array': corr_array,
                        'PR': partic_ratio,
                        'response_range': respRange,
                        'stimulus': stim,
                        'session': session
                    })

                    plt.scatter(response_transform[0][:, 0], response_transform[1][:, 0])
                    plt.xlabel(f'{brainRegion}_canonical_dimension_0')
                    plt.ylabel(f'{brainRegion2}_canonical_dimension_0')
                    plt.title(f'First pair of canonical variables, Pearson correlation = {correlation_val:.2f}')
                    plt.savefig(f"{file_path}/CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}.png")
                    plt.show()

                    print(f"Weight norms = {np.linalg.norm(cca.y_weights_[0])} and weight shape is {cca.y_weights_.shape}")

                    print("Now plotting the canonical dimension in original data space")
                    origin = [0, 0]
                    br1_fr = brain_resp_array[:, :2]
                    br1_weights = cca.x_weights_[:, 0]
                    br2_fr = brain2_resp_array[:, :2]
                    br2_weights = cca.x_weights_[:, 1]

                    # Making scatters
                    plt.scatter(br1_fr[:, 0], br1_fr[:, 1], c=stimArray, cmap='viridis', alpha=0.3, s=2)
                    plt.xlabel(f'{brainRegion}_neuron_0')
                    plt.ylabel(f'{brainRegion}_neuron_1')
                    plt.title(f'{brainRegion} Neuron 0 and 1 vs CCA component 0')
                    plt.colorbar(label='Stimulus')
                    plt.quiver(*origin, br1_weights[0], br1_weights[1], scale_units='xy', scale=0.001, color='black',
                               angles='xy', label='CCA separation')
                    plt.savefig(f"{file_path}/CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}_original_space.png")
                    plt.show()

                    plt.scatter(br2_fr[:, 0], br2_fr[:, 1], c=stimArray, cmap='viridis', alpha=0.3, s=2)
                    plt.xlabel(f'{brainRegion2}_neuron_0')
                    plt.ylabel(f'{brainRegion2}_neuron_1')
                    plt.title(f'{brainRegion2} Neuron 0 and 1 vs CCA component 0')
                    plt.colorbar(label='Stimulus')
                    plt.quiver(*origin, br2_weights[0], br2_weights[1], scale_units='xy', scale=0.001, color='black',
                               angles='xy', label='CCA separation')
                    plt.savefig(f"{file_path}/CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}_original_space2.png")
                    plt.show()

    df_correlations = pd.DataFrame(correlation_data)
    df_correlations.to_feather(f"{file_path}/CCA_correlations_{stim}.feather")
    df_correlations.to_csv(f"{file_path}/CCA_correlations_{stim}.csv", index=False)

    for analysis_type in analysis_attempts:
        # Stats comparisons with a Kruskal-Wallis test (ANOVA but non-parametric)
        groups = [group[f'{analysis_type}'].values for name, group in df_correlations.groupby('region_pair')]
        kruskal_stat, kruskal_p = kruskal(*groups)
        print(f"Kruskal-Wallis test: H = {kruskal_stat:.4f}, p = {kruskal_p:.4f}")

        if kruskal_p < 0.05:
            print("Significant differences found between region pairs")
        else:
            print("No significant differences found between region pairs")

        region_pairs = df_correlations['region_pair'].unique()

        # Perform all pairwise comparisons
        pairwise_results = []
        for pair1, pair2 in itertools.combinations(region_pairs, 2):
            group1 = df_correlations[df_correlations['region_pair'] == pair1][f'{analysis_type}']
            group2 = df_correlations[df_correlations['region_pair'] == pair2][f'{analysis_type}']

            stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            pairwise_results.append({
                'comparison': f"{pair1} vs {pair2}",
                'statistic': stat,
                'p_value': p_val,
                'group1_median': group1.median(),
                'group2_median': group2.median()
            })
        pairwise_df = pd.DataFrame(pairwise_results)

        # Apply multiple comparisons correction (Benjamini-Hochberg)
        # _, corrected_p, _, _ = multipletests(pairwise_df['p_value'], method='fdr_bh')
        _, corrected_p, _, _ = multipletests(pairwise_df['p_value'], method='bonferroni')
        pairwise_df['corrected_p'] = corrected_p
        pairwise_df['significant'] = corrected_p < 0.05

        significant_pairs = pairwise_df[pairwise_df['significant']]
        print(f"\nSignificant pairwise comparisons (corrected p < 0.05):")
        print(significant_pairs[['comparison', 'group1_median', 'group2_median', 'corrected_p']])
        pairwise_df.to_feather(f"{file_path}/CCA_pairwise_{analysis_type}{stim}.feather")
        pairwise_df.to_csv(f"{file_path}/CCA_pairwise_{analysis_type}_{stim}.csv")

        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(data=df_correlations, x='region_pair', y=f'{analysis_type}')
        plt.xticks(rotation=45, ha='right')
        add_significance_stars(ax, df_correlations, 'region_pair', f'{analysis_type}')
        plt.title(f'CCA {analysis_type} by Brain Region Pair - {stim}')
        plt.ylabel('Canonical Correlation')
        plt.tight_layout()
        plt.savefig(f"{file_path}/CCA_summary_plots/CCA_{analysis_type}_boxplot_{stim}.png", dpi=300, bbox_inches='tight')
        plt.show()

        if len(response_ranges) > 1:  # Create a boxplot of the various response ranges. No stats for these yet
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=df_correlations, x='region_pair', y=f'{analysis_type}', hue='response_range')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'CCA {analysis_type} by Brain Region Pair and Response Range - {stim}')
            plt.ylabel('Canonical Correlation')
            plt.legend(title='Response Range')
            plt.tight_layout()
            plt.savefig(f"{file_path}/CCA_summary_plots/CCA_{analysis_type}_by_response_range_{stim}.png", dpi=300, bbox_inches='tight')
            plt.show()

        # Create individual plots for each response range with statistical comparisons
        if len(response_ranges) > 1:
            # Create output directory for individual response range plots
            individual_plots_dir = f"{file_path}/CCA_individual_response_plots_{analysis_type}"
            os.makedirs(individual_plots_dir, exist_ok=True)

            print(f"\n=== Statistical Analysis by Response Range for {stim} ===")

            for respRange in response_ranges:
                print(f"\n--- {respRange.title()} Response Range ---")

                # Filter data for this response range
                range_data = df_correlations[df_correlations['response_range'] == respRange]

                if len(range_data) == 0:
                    print(f"No data found for {respRange} response range")
                    continue

                # Overall Kruskal-Wallis test
                groups = [group['correlation'].values for name, group in range_data.groupby('region_pair')]
                group_names = [name for name, group in range_data.groupby('region_pair')]

                if len(groups) < 2:
                    print(f"Not enough groups for statistical comparison in {respRange}")
                    continue

                kruskal_stat, kruskal_p = kruskal(*groups)
                print(f"Kruskal-Wallis test: H = {kruskal_stat:.4f}, p = {kruskal_p:.4f}")

                if kruskal_p < 0.05:
                    print("Significant differences found between region pairs")
                else:
                    print("No significant differences found between region pairs")

                # Create the boxplot
                plt.figure(figsize=(14, 10))
                ax = sns.boxplot(data=range_data, x='region_pair', y=f'{analysis_type}')
                plt.xticks(rotation=45, ha='right')

                # Add statistical brackets if overall test is significant
                if kruskal_p < 0.05:
                    statistical_results = add_statistical_brackets(ax, range_data, 'region_pair', f'{analysis_type}')

                    # Print significant pairwise comparisons
                    significant_pairs = [r for r in statistical_results if r['significant']]
                    if significant_pairs:
                        print(f"Significant pairwise comparisons (FDR corrected p < 0.05):")
                        for result in significant_pairs:
                            effect_size = abs(result['median1'] - result['median2'])
                            print(f"  {result['group1_name']} vs {result['group2_name']}: "
                                  f"p = {result['corrected_p']:.4f}, "
                                  f"medians: {result['median1']:.3f} vs {result['median2']:.3f}, "
                                  f"effect size: {effect_size:.3f}")
                    else:
                        print("No significant pairwise comparisons after FDR correction")

                plt.title(f'CCA {analysis_type} - {respRange.title()} Response Range - {stim}\n'
                          f'Kruskal-Wallis: H = {kruskal_stat:.3f}, p = {kruskal_p:.4f}')
                plt.ylabel('Canonical Correlation')
                plt.tight_layout()

                # Save the plot
                plt.savefig(f"{individual_plots_dir}/CCA_{analysis_type}_{respRange}_{stim}.png",
                            dpi=300, bbox_inches='tight')
                plt.show()

                # Create a summary statistics table for this response range
                summary_stats = range_data.groupby('region_pair')[f'{analysis_type}'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(4)

                print(f"\nSummary statistics for {respRange} response range:")
                print(summary_stats)

                # Save summary statistics
                summary_stats.to_csv(f"{individual_plots_dir}/summary_stats_{analysis_type}_{respRange}_{stim}.csv")

        # Additional analysis: Compare response ranges for each region pair
        print(f"\n=== Comparing Response Ranges Within Region Pairs for {stim} ===")

        response_range_comparison_dir = f"{file_path}/CCA_response_range_comparisons_{analysis_type}"
        os.makedirs(response_range_comparison_dir, exist_ok=True)

        # Get unique region pairs
        unique_region_pairs = df_correlations['region_pair'].unique()

        for region_pair in unique_region_pairs:
            print(f"\n--- {region_pair} ---")

            # Filter data for this region pair
            pair_data = df_correlations[df_correlations['region_pair'] == region_pair]

            # Check if we have data for multiple response ranges
            available_ranges = pair_data['response_range'].unique()
            if len(available_ranges) < 2:
                print(f"Only one response range available for {region_pair}")
                continue

            # Kruskal-Wallis test across response ranges
            range_groups = [group[f'{analysis_type}'].values for name, group in pair_data.groupby('response_range')]
            kruskal_stat, kruskal_p = kruskal(*range_groups)
            print(f"Kruskal-Wallis test across response ranges: H = {kruskal_stat:.4f}, p = {kruskal_p:.4f}")

            # Create boxplot comparing response ranges for this region pair
            plt.figure(figsize=(10, 8))
            ax = sns.boxplot(data=pair_data, x='response_range', y=f'{analysis_type}')

            # Add statistical brackets if significant
            if kruskal_p < 0.05:
                statistical_results = add_statistical_brackets(ax, pair_data, 'response_range', f'{analysis_type}')

                # Print significant comparisons
                significant_pairs = [r for r in statistical_results if r['significant']]
                if significant_pairs:
                    print(f"Significant response range comparisons (FDR corrected p < 0.05):")
                    for result in significant_pairs:
                        effect_size = abs(result['median1'] - result['median2'])
                        print(f"  {result['group1_name']} vs {result['group2_name']}: "
                              f"p = {result['corrected_p']:.4f}, "
                              f"medians: {result['median1']:.3f} vs {result['median2']:.3f}")

            plt.title(f'CCA {analysis_type} Across Response Ranges\n{region_pair} - {stim}\n'
                      f'Kruskal-Wallis: H = {kruskal_stat:.3f}, p = {kruskal_p:.4f}')
            plt.ylabel('Canonical Correlation')
            plt.xlabel('Response Range')
            plt.tight_layout()

            # Save the plot
            safe_region_pair = region_pair.replace('/', '_').replace(' ', '_')
            plt.savefig(f"{response_range_comparison_dir}/response_ranges_{analysis_type}_{safe_region_pair}_{stim}.png",
                        dpi=300, bbox_inches='tight')
            plt.show()

            # Summary statistics for this region pair across response ranges
            pair_summary = pair_data.groupby('response_range')[f'{analysis_type}'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(4)

            print(f"Summary statistics for {region_pair}:")
            print(pair_summary)

            # Save summary
            pair_summary.to_csv(f"{response_range_comparison_dir}/summary_{analysis_type}_{safe_region_pair}_{stim}.csv")


