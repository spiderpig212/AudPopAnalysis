
"""
CCA Analysis Script: Compare two brain regions using 5-fold cross-validation
Creates scree plots showing correlation distributions for each CCA component
and determines optimal dimensionality using permutation tests.
"""

import os
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from analysis_class import FiringRateAnalysis
import studyparams
from jaratoolbox import settings, celldatabase


class TwoRegionCCAAnalysis:
    def __init__(self, neuron_threshold=20, n_splits=5, random_state=42, n_permutations=1000):
        """
        Initialize the two-region CCA analysis

        Parameters:
        -----------
        neuron_threshold : int
            Minimum number of neurons required per region
        n_splits : int
            Number of folds for cross-validation
        random_state : int
            Random seed for reproducibility
        n_permutations : int
            Number of permutations for statistical testing
        """
        self.neuron_threshold = neuron_threshold
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_permutations = n_permutations
        self.fr_db = FiringRateAnalysis(db_suffix="coords_updated")
        self.file_path = self.fr_db.figdata_path
        self.stim_types = self.fr_db.stim_types
        self.response_ranges = ["onset", "sustained", "offset"]

        # Create output directory
        self.output_dir = os.path.join(self.file_path, "CCA_two_region_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_cca_cross_validation(self, region1_data, region2_data):
        """
        Run CCA with 5-fold cross-validation

        Parameters:
        -----------
        region1_data : np.ndarray
            Neural data from region 1 (trials x neurons)
        region2_data : np.ndarray
            Neural data from region 2 (trials x neurons)

        Returns:
        --------
        test_correlation_matrix : np.ndarray
            Test correlation values for each component across CV folds
        train_correlation_matrix : np.ndarray
            Training correlation values for each component across CV folds
        """
        n_components = min(region1_data.shape[1], region2_data.shape[1])
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        test_correlation_matrix = []
        train_correlation_matrix = []

        for train_idx, test_idx in kf.split(region1_data):
            r1_train, r1_test = region1_data[train_idx], region1_data[test_idx]
            r2_train, r2_test = region2_data[train_idx], region2_data[test_idx]

            cca = CCA(n_components=n_components)
            cca.fit(r1_train, r2_train)

            # Transform test data
            r1_test_transform, r2_test_transform = cca.transform(r1_test, r2_test)

            # Transform training data
            r1_train_transform, r2_train_transform = cca.transform(r1_train, r2_train)

            # Calculate test correlations for each component
            test_fold_correlations = []
            for comp in range(n_components):
                corr = np.corrcoef(r1_test_transform[:, comp], r2_test_transform[:, comp])[0, 1]
                test_fold_correlations.append(corr)
            test_correlation_matrix.append(test_fold_correlations)

            # Calculate training correlations for each component
            train_fold_correlations = []
            for comp in range(n_components):
                corr = np.corrcoef(r1_train_transform[:, comp], r2_train_transform[:, comp])[0, 1]
                train_fold_correlations.append(corr)
            train_correlation_matrix.append(train_fold_correlations)

        return np.array(test_correlation_matrix), np.array(train_correlation_matrix)

    def permutation_test(self, region1_data, region2_data):
        """
        Perform permutation test for statistical significance

        Parameters:
        -----------
        region1_data : np.ndarray
            Neural data from region 1 (trials x neurons)
        region2_data : np.ndarray
            Neural data from region 2 (trials x neurons)

        Returns:
        --------
        permutation_correlations : np.ndarray
            Distribution of permuted correlations for each component
            Shape: (n_permutations, n_components)
        """
        n_components = min(region1_data.shape[1], region2_data.shape[1])
        permutation_correlations = []

        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        print(f"Running {self.n_permutations} permutations...")
        for perm in tqdm(range(self.n_permutations)):
            # Shuffle one region (region2) to break correlations
            perm_indices = np.random.permutation(region2_data.shape[0])
            region2_permuted = region2_data[perm_indices]

            # Run cross-validation on permuted data
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=perm)
            perm_correlations = []

            for train_idx, test_idx in kf.split(region1_data):
                r1_train, r1_test = region1_data[train_idx], region1_data[test_idx]
                r2_train, r2_test = region2_permuted[train_idx], region2_permuted[test_idx]

                cca = CCA(n_components=n_components)
                cca.fit(r1_train, r2_train)
                r1_transform, r2_transform = cca.transform(r1_test, r2_test)

                # Calculate correlations for each component
                fold_correlations = []
                for comp in range(n_components):
                    corr = np.corrcoef(r1_transform[:, comp], r2_transform[:, comp])[0, 1]
                    fold_correlations.append(corr)
                perm_correlations.append(fold_correlations)

            # Average across folds for this permutation
            perm_r_bar = np.mean(perm_correlations, axis=0)
            permutation_correlations.append(perm_r_bar)

        return np.array(permutation_correlations)

    def permutation_test_components(self, r_bar, permutation_correlations, alpha=0.05):
        """
        Test each component using permutation test results

        Parameters:
        -----------
        r_bar : np.ndarray
            Average test correlations for each component
        permutation_correlations : np.ndarray
            Distribution of permuted correlations for each component
        alpha : float
            Significance level

        Returns:
        --------
        significant_components : int
            Number of significant components
        p_values : np.ndarray
            P-values for each component
        """
        n_components = len(r_bar)
        p_values = []

        for comp in range(n_components):
            # Calculate p-value as proportion of permutations >= observed r_bar
            p_val = np.mean(permutation_correlations[:, comp] >= r_bar[comp])
            p_values.append(p_val)

        p_values = np.array(p_values)

        # Find first non-significant component
        significant_mask = p_values < alpha
        if np.any(significant_mask):
            # Count consecutive significant components from the beginning
            significant_components = 0
            for i, is_sig in enumerate(significant_mask):
                if is_sig:
                    significant_components = i + 1
                else:
                    break
        else:
            significant_components = 0

        return significant_components, p_values

    def create_scree_plot(self, test_correlation_matrix, train_correlation_matrix,
                         permutation_correlations, p_values, region_pair,
                         stimulus, response_range, session):
        """
        Create scree plot showing r_bar values and training correlations

        Parameters:
        -----------
        test_correlation_matrix : np.ndarray
            Test correlation values for each component across CV folds
        train_correlation_matrix : np.ndarray
            Training correlation values for each component across CV folds
        permutation_correlations : np.ndarray
            Distribution of permuted correlations for each component
        p_values : np.ndarray
            P-values for each component
        region_pair : str
            Name of the region pair
        stimulus : str
            Stimulus type
        response_range : str
            Response time window
        session : str
            Session identifier
        """
        n_components = test_correlation_matrix.shape[1]

        # Calculate r_bar (average test correlations across folds)
        r_bar = np.mean(test_correlation_matrix, axis=0)
        r_bar_train = np.mean(train_correlation_matrix, axis=0)

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Top left: Test r_bar scree plot
        ax1.plot(range(1, n_components + 1), r_bar, 'bo-', linewidth=2, markersize=8, label='Test r_bar')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Null hypothesis (r=0)')
        ax1.set_xlabel('CCA Component')
        ax1.set_ylabel('Average Correlation (r_bar)')
        ax1.set_title(f'Test Correlations: {region_pair}\n{stimulus} - {response_range} - Session {session}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add significance indicators
        for comp in range(n_components):
            if p_values[comp] < 0.05:
                ax1.plot(comp + 1, r_bar[comp], 'r*', markersize=15, label='Significant' if comp == 0 else "")

        # Top right: Training r_bar scree plot
        ax2.plot(range(1, n_components + 1), r_bar_train, 'go-', linewidth=2, markersize=8, label='Train r_bar')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Null hypothesis (r=0)')
        ax2.set_xlabel('CCA Component')
        ax2.set_ylabel('Average Correlation (r_bar)')
        ax2.set_title(f'Training Correlations: {region_pair}\n{stimulus} - {response_range} - Session {session}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Bottom left: Permutation distributions for first few components
        n_components_to_show = min(4, n_components)
        for comp in range(n_components_to_show):
            ax3.hist(permutation_correlations[:, comp], bins=50, alpha=0.6,
                    label=f'Comp {comp+1}', density=True)
            # Mark observed r_bar
            ax3.axvline(r_bar[comp], color=f'C{comp}', linestyle='-', linewidth=2)

        ax3.set_xlabel('Correlation Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Permutation Distributions (First 4 Components)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom right: P-values
        bars = ax4.bar(range(1, n_components + 1), -np.log10(p_values), alpha=0.7)
        ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='α = 0.05')

        # Color significant bars differently
        for i, p_val in enumerate(p_values):
            if p_val < 0.05:
                bars[i].set_color('red')

        ax4.set_xlabel('CCA Component')
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title('Statistical Significance (Permutation Test)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        safe_region_pair = region_pair.replace('_vs_', '_').replace(' ', '_')
        filename = f"scree_plot_{safe_region_pair}_{stimulus}_{response_range}_session_{session}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def analyze_region_pair(self, region1_name, region2_name):
        """
        Analyze a specific pair of brain regions across all conditions

        Parameters:
        -----------
        region1_name : str
            Name of first brain region
        region2_name : str
            Name of second brain region
        """
        results = []

        print(f"\nAnalyzing region pair: {region1_name} vs {region2_name}")

        for stimulus in self.stim_types:
            print(f"Processing stimulus: {stimulus}")

            # Get stimulus data
            stim_arrays = self.fr_db.return_arrays(stimulus)
            brainRegionArray = stim_arrays["brainRegionArray"]
            sessionArray = stim_arrays["sessionIDArray"]
            uniqSessions = np.unique(sessionArray)

            for response_range in self.response_ranges:
                respArray = stim_arrays[f"{response_range}fr"]

                for session in uniqSessions:
                    session_mask = sessionArray == session
                    session_resp_array = respArray[session_mask, :]
                    brain_session_array = brainRegionArray[session_mask]

                    # Get data for region 1
                    region1_mask = brain_session_array == region1_name
                    region1_data = session_resp_array[region1_mask, :].T

                    # Get data for region 2
                    region2_mask = brain_session_array == region2_name
                    region2_data = session_resp_array[region2_mask, :].T

                    # Check neuron counts
                    if (region1_data.shape[1] < self.neuron_threshold or
                            region2_data.shape[1] < self.neuron_threshold):
                        print(f"Skipping session {session}: insufficient neurons "
                              f"(R1: {region1_data.shape[1]}, R2: {region2_data.shape[1]})")
                        continue

                    # Sample neurons if we have more than threshold
                    np.random.seed(self.random_state)
                    if region1_data.shape[1] > self.neuron_threshold:
                        r1_neurons = np.random.choice(region1_data.shape[1],
                                                      size=self.neuron_threshold,
                                                      replace=False)
                        region1_data = region1_data[:, r1_neurons]

                    if region2_data.shape[1] > self.neuron_threshold:
                        r2_neurons = np.random.choice(region2_data.shape[1],
                                                      size=self.neuron_threshold,
                                                      replace=False)
                        region2_data = region2_data[:, r2_neurons]

                    print(f"Session {session}, Response: {response_range} - "
                          f"R1: {region1_data.shape[1]} neurons, R2: {region2_data.shape[1]} neurons")

                    # Run CCA cross-validation
                    test_correlation_matrix, train_correlation_matrix = self.run_cca_cross_validation(
                        region1_data, region2_data)

                    # Calculate r_bar (average across folds)
                    r_bar = np.mean(test_correlation_matrix, axis=0)
                    r_bar_train = np.mean(train_correlation_matrix, axis=0)

                    # Run permutation test
                    permutation_correlations = self.permutation_test(region1_data, region2_data)

                    # Statistical testing using permutation results
                    significant_components, p_values = self.permutation_test_components(
                        r_bar, permutation_correlations)

                    # Create scree plot
                    region_pair = f"{region1_name}_vs_{region2_name}"
                    self.create_scree_plot(test_correlation_matrix, train_correlation_matrix,
                                         permutation_correlations, p_values, region_pair,
                                         stimulus, response_range, session)

                    # Store results
                    result = {
                        'region1': region1_name,
                        'region2': region2_name,
                        'region_pair': region_pair,
                        'stimulus': stimulus,
                        'response_range': response_range,
                        'session': session,
                        'significant_components': significant_components,
                        'total_components': test_correlation_matrix.shape[1],
                        'r_bar': r_bar,  # Average test correlations
                        'r_bar_train': r_bar_train,  # Average training correlations
                        'test_correlations_std': test_correlation_matrix.std(axis=0),
                        'train_correlations_std': train_correlation_matrix.std(axis=0),
                        'p_values': p_values,
                        'permutation_correlations': permutation_correlations
                    }
                    results.append(result)

                    print(f"Found {significant_components} significant components out of "
                          f"{test_correlation_matrix.shape[1]} total")

        return results

    def run_analysis(self, region_pairs=None):
        """
        Run the full analysis for specified region pairs

        Parameters:
        -----------
        region_pairs : list of tuples, optional
            List of (region1, region2) tuples to analyze
            If None, analyzes all possible pairs
        """
        # Get available regions
        stim_arrays = self.fr_db.return_arrays(list(self.stim_types)[0])  # Use first stimulus to get regions
        uniq_regions = np.unique(stim_arrays["brainRegionArray"])

        if region_pairs is None:
            # Generate all possible pairs
            region_pairs = []
            for i, region1 in enumerate(uniq_regions):
                for region2 in uniq_regions[i + 1:]:
                    region_pairs.append((region1, region2))

        print(f"Analyzing {len(region_pairs)} region pairs...")

        all_results = []
        for region1, region2 in region_pairs:
            pair_results = self.analyze_region_pair(region1, region2)
            all_results.extend(pair_results)

        # Convert to DataFrame and save (excluding large arrays)
        results_for_df = []
        for result in all_results:
            result_copy = result.copy()
            # Remove large arrays for CSV storage
            result_copy.pop('permutation_correlations', None)
            results_for_df.append(result_copy)

        results_df = pd.DataFrame(results_for_df)

        # Save detailed results
        results_df.to_csv(os.path.join(self.output_dir, "cca_two_region_results.csv"), index=False)
        results_df.to_feather(os.path.join(self.output_dir, "cca_two_region_results.feather"))

        # Save full results with arrays as pickle
        import pickle
        with open(os.path.join(self.output_dir, "cca_two_region_results_full.pkl"), 'wb') as f:
            pickle.dump(all_results, f)

        # Create summary plots
        self.create_summary_plots(results_df)

        # Create new detailed summary visualizations
        print("Creating detailed summary visualizations...")
        self.create_three_panel_summary(results_df)
        summary_table = self.create_detailed_summary_table(results_df)
        self.create_heatmap_summary(results_df)

        print(f"Summary table shape: {summary_table.shape}")
        print("Top 10 conditions by average significant components:")
        top_conditions = summary_table.nlargest(10, 'significant_components_mean')
        print(top_conditions[['stimulus', 'response_range', 'region_pair', 'significant_components_mean']])

        return results_df

    def create_summary_plots(self, results_df):
        """
        Create summary plots of the analysis results
        """
        # Summary of significant components by region pair
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=results_df, x='region_pair', y='significant_components')
        plt.xticks(rotation=45, ha='right')
        plt.title('Number of Significant CCA Components by Region Pair (Permutation Test)')
        plt.ylabel('Number of Significant Components')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary_significant_components.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Summary by stimulus and response range
        if len(results_df['stimulus'].unique()) > 1:
            plt.figure(figsize=(15, 10))
            sns.boxplot(data=results_df, x='region_pair', y='significant_components',
                        hue='stimulus')
            plt.xticks(rotation=45, ha='right')
            plt.title('Number of Significant CCA Components by Region Pair and Stimulus (Permutation Test)')
            plt.ylabel('Number of Significant Components')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "summary_by_stimulus.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def create_three_panel_summary(self, results_df):
        """
        Create three-panel summary plot showing significant components by stimulus, response range, and region pair

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe from the CCA analysis
        """
        # Get unique values
        unique_stimuli = sorted(results_df['stimulus'].unique())
        unique_response_ranges = sorted(results_df['response_range'].unique())
        unique_region_pairs = sorted(results_df['region_pair'].unique())

        # Create color palette for region pairs
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_region_pairs)))
        region_pair_colors = dict(zip(unique_region_pairs, colors))

        # Create figure with three panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        # Set bar width for grouped bars
        bar_width = 0.8 / len(unique_region_pairs)

        for i, stimulus in enumerate(unique_stimuli):
            ax = axes[i]

            # Filter data for this stimulus
            stimulus_data = results_df[results_df['stimulus'] == stimulus]

            # Create grouped bar plot
            x_positions = np.arange(len(unique_response_ranges))

            for j, region_pair in enumerate(unique_region_pairs):
                # Get data for this region pair
                pair_data = stimulus_data[stimulus_data['region_pair'] == region_pair]

                # Calculate mean and std for each response range
                means = []
                stds = []

                for response_range in unique_response_ranges:
                    range_data = pair_data[pair_data['response_range'] == response_range]
                    if len(range_data) > 0:
                        means.append(range_data['significant_components'].mean())
                        stds.append(range_data['significant_components'].std())
                    else:
                        means.append(0)
                        stds.append(0)

                # Plot bars
                x_pos = x_positions + j * bar_width - (len(unique_region_pairs) - 1) * bar_width / 2
                bars = ax.bar(x_pos, means, bar_width, yerr=stds,
                              color=region_pair_colors[region_pair],
                              alpha=0.7, label=region_pair,
                              capsize=3)

                # Add value labels on bars
                for k, (bar, mean) in enumerate(zip(bars, means)):
                    if mean > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[k] + 0.1,
                                f'{mean:.1f}', ha='center', va='bottom', fontsize=8)

            # Customize subplot
            ax.set_xlabel('Response Range', fontsize=12)
            if i == 0:
                ax.set_ylabel('Number of Significant Components', fontsize=12)
            ax.set_title(f'{stimulus}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(unique_response_ranges)
            ax.grid(True, alpha=0.3, axis='y')

            # Add legend only to the last subplot to avoid clutter
            if i == len(unique_stimuli) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Overall title
        fig.suptitle('CCA Significant Components by Stimulus Type, Response Range, and Region Pair',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(self.output_dir, "three_panel_summary.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def create_detailed_summary_table(self, results_df):
        """
        Create a detailed summary table with statistics for each condition

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe from the CCA analysis

        Returns:
        --------
        summary_table : pd.DataFrame
            Summary statistics table
        """
        # Group by all relevant factors and calculate statistics
        summary_stats = results_df.groupby(['stimulus', 'response_range', 'region_pair']).agg({
            'significant_components': ['mean', 'std', 'count', 'max'],
            'total_components': ['mean', 'std'],
        }).round(2)

        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

        # Reset index to make grouping variables regular columns
        summary_table = summary_stats.reset_index()

        # Add proportion of significant components
        summary_table['proportion_significant'] = (
                summary_table['significant_components_mean'] /
                summary_table['total_components_mean']
        ).round(3)

        # Save summary table
        summary_table.to_csv(os.path.join(self.output_dir, "detailed_summary_table.csv"), index=False)

        return summary_table

    def create_heatmap_summary(self, results_df):
        """
        Create heatmap showing average significant components across conditions

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe from the CCA analysis
        """
        # Create pivot table for heatmap
        for stimulus in results_df['stimulus'].unique():
            stimulus_data = results_df[results_df['stimulus'] == stimulus]

            # Create pivot table: response_range x region_pair
            pivot_data = stimulus_data.pivot_table(
                values='significant_components',
                index='response_range',
                columns='region_pair',
                aggfunc='mean'
            )

            # Create heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.1f',
                        cbar_kws={'label': 'Number of Significant Components'})
            plt.title(f'Average Significant CCA Components - {stimulus}',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Region Pair', fontsize=12)
            plt.ylabel('Response Range', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save individual heatmaps
            safe_stimulus = stimulus.replace(' ', '_').replace('/', '_')
            plt.savefig(os.path.join(self.output_dir, f"heatmap_summary_{safe_stimulus}.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    # Initialize analysis
    analyzer = TwoRegionCCAAnalysis(neuron_threshold=40, n_splits=5, random_state=42,
                                   n_permutations=100)

    # Example: analyze specific region pairs
    # Uncomment and modify these lines to analyze specific pairs:
    # region_pairs = [('AudP', 'AudV'), ('AudP', 'TeA')]
    # results_df = analyzer.run_analysis(region_pairs)

    # Or analyze all possible pairs:
    results_df = analyzer.run_analysis()

    print("\nAnalysis complete! Results saved to:", analyzer.output_dir)
    print(f"Analyzed {len(results_df)} conditions across all region pairs")