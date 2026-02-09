"""
CCA Analysis Script: Compare two brain regions using 5-fold cross-validation
Creates scree plots showing correlation distributions for each CCA component
and determines optimal dimensionality using non-parametric statistical tests.
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
    def __init__(self, neuron_threshold=20, n_splits=5, random_state=42):
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
        """
        self.neuron_threshold = neuron_threshold
        self.n_splits = n_splits
        self.random_state = random_state
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
        correlation_matrix : np.ndarray
            Correlation values for each component across CV folds
        """
        n_components = min(region1_data.shape[1], region2_data.shape[1])
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        correlation_matrix = []

        for train_idx, test_idx in kf.split(region1_data):
            r1_train, r1_test = region1_data[train_idx], region1_data[test_idx]
            r2_train, r2_test = region2_data[train_idx], region2_data[test_idx]

            cca = CCA(n_components=n_components)
            cca.fit(r1_train, r2_train)

            r1_transform, r2_transform = cca.transform(r1_test, r2_test)

            # Calculate correlations for each component
            fold_correlations = []
            for comp in range(n_components):
                corr = np.corrcoef(r1_transform[:, comp], r2_transform[:, comp])[0, 1]
                fold_correlations.append(corr)

            correlation_matrix.append(fold_correlations)

        return np.array(correlation_matrix)  # Shape: (n_folds, n_components)

    def statistical_test_components(self, correlation_matrix, alpha=0.05):
        """
        Test each component against null hypothesis of mean correlation = 0
        using Wilcoxon signed-rank test (non-parametric one-sided test)

        Parameters:
        -----------
        correlation_matrix : np.ndarray
            Correlation values for each component across CV folds
        alpha : float
            Significance level

        Returns:
        --------
        significant_components : int
            Number of significant components
        p_values : np.ndarray
            P-values for each component
        """
        n_components = correlation_matrix.shape[1]
        p_values = []

        for comp in range(n_components):
            correlations = correlation_matrix[:, comp]
            # One-sided test: H0: median = 0, H1: median > 0
            stat, p_val = stats.wilcoxon(correlations, alternative='greater')
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

    def create_scree_plot(self, correlation_matrix, p_values, region_pair,
                          stimulus, response_range, session):
        """
        Create scree plot showing correlation distributions for each component

        Parameters:
        -----------
        correlation_matrix : np.ndarray
            Correlation values for each component across CV folds
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
        n_components = correlation_matrix.shape[1]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top panel: Box plots of correlation distributions
        plot_data = []
        for comp in range(n_components):
            for corr_val in correlation_matrix[:, comp]:
                plot_data.append({
                    'Component': f'Comp {comp + 1}',
                    'Correlation': corr_val,
                    'Component_num': comp + 1
                })

        plot_df = pd.DataFrame(plot_data)

        # Create box plot
        sns.boxplot(data=plot_df, x='Component', y='Correlation', ax=ax1)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Null hypothesis (r=0)')
        ax1.set_title(f'CCA Component Correlations: {region_pair}\n'
                      f'{stimulus} - {response_range} - Session {session}')
        ax1.set_ylabel('Pearson Correlation')
        ax1.legend()

        # Add significance indicators
        for comp in range(n_components):
            if p_values[comp] < 0.05:
                # Add star above box plot
                y_pos = correlation_matrix[:, comp].max() + 0.05
                ax1.text(comp, y_pos, '*', ha='center', va='bottom',
                         fontsize=16, color='red', weight='bold')

        # Bottom panel: P-values
        ax2.bar(range(1, n_components + 1), -np.log10(p_values), alpha=0.7)
        ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--',
                    label='α = 0.05')
        ax2.set_xlabel('CCA Component')
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_title('Statistical Significance of Components')
        ax2.legend()

        # Set x-axis ticks
        ax1.set_xticks(range(n_components))
        ax1.set_xticklabels([f'Comp {i + 1}' for i in range(n_components)], rotation=45)
        ax2.set_xticks(range(1, n_components + 1))

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
                    correlation_matrix = self.run_cca_cross_validation(region1_data, region2_data)

                    # Statistical testing
                    significant_components, p_values = self.statistical_test_components(correlation_matrix)

                    # Create scree plot
                    region_pair = f"{region1_name}_vs_{region2_name}"
                    self.create_scree_plot(correlation_matrix, p_values, region_pair,
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
                        'total_components': correlation_matrix.shape[1],
                        'mean_correlations': correlation_matrix.mean(axis=0),  # Averaging across folds, so ends up being size of n_components
                        'std_correlations': correlation_matrix.std(axis=0),
                        'p_values': p_values
                    }
                    results.append(result)

                    print(
                        f"Found {significant_components} significant components out of {correlation_matrix.shape[1]} total")

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

        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)

        # Save detailed results
        results_df.to_csv(os.path.join(self.output_dir, "cca_two_region_results.csv"), index=False)
        results_df.to_feather(os.path.join(self.output_dir, "cca_two_region_results.feather"))

        # Create summary plot
        self.create_summary_plots(results_df)

        return results_df

    def create_summary_plots(self, results_df):
        """
        Create summary plots of the analysis results
        """
        # Summary of significant components by region pair
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=results_df, x='region_pair', y='significant_components')
        plt.xticks(rotation=45, ha='right')
        plt.title('Number of Significant CCA Components by Region Pair')
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
            plt.title('Number of Significant CCA Components by Region Pair and Stimulus')
            plt.ylabel('Number of Significant Components')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "summary_by_stimulus.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    # Initialize analysis
    analyzer = TwoRegionCCAAnalysis(neuron_threshold=40, n_splits=5, random_state=42)

    # Example: analyze specific region pairs
    # Uncomment and modify these lines to analyze specific pairs:
    # region_pairs = [('AudP', 'AudV'), ('AudP', 'TeA')]
    # results_df = analyzer.run_analysis(region_pairs)

    # Or analyze all possible pairs:
    results_df = analyzer.run_analysis()

    print("\nAnalysis complete! Results saved to:", analyzer.output_dir)
    print(f"Analyzed {len(results_df)} conditions across all region pairs")