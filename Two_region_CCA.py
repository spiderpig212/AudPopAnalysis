
"""
CCA Analysis Script: Compare two brain regions using 5-fold cross-validation
Creates scree plots showing correlation distributions for each CCA component
and determines optimal dimensionality using permutation tests.
"""

import os
import sys
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from analysis_class import FiringRateAnalysis

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
                try:
                    cca.fit(r1_train, r2_train)
                except ValueError as e:
                    print(f"Error fitting CCA for permutation {perm}: {e}")
                    continue
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
                         stimulus, response_range, session, verbose=True):
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
        filename = f"scree_plots/scree_plot_{safe_region_pair}_{stimulus}_{response_range}_session_{session}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved scree plot: {filename}")
            plt.show()

        return fig

    def analyze_primary_auditory_within_region(self, region1_data, session, stimulus, response_range,
                                               n_iterations=10, verbose=True):
        """
        Perform within-region analysis for Primary auditory area by randomly splitting neurons

        Parameters:
        -----------
        region1_data : np.ndarray
            Neural data from Primary auditory area (trials x neurons)
        session : str
            Session identifier
        stimulus : str
            Stimulus type
        response_range : str
            Response range
        n_iterations : int
            Number of random split iterations

        Returns:
        --------
        list : List of results from multiple iterations
        """
        total_neurons = region1_data.shape[1]
        half_size = self.neuron_threshold

        print(f"Performing within-Primary auditory area analysis: {total_neurons} neurons, "
              f"splitting into 2 halves of {half_size} neurons each, {n_iterations} iterations")

        iteration_results = []

        for iteration in range(n_iterations):
            # Set seed for reproducibility within each iteration
            np.random.seed(self.random_state + iteration)

            # Randomly shuffle neuron indices and split into two equal halves
            neuron_indices = np.random.permutation(total_neurons)
            half1_indices = neuron_indices[:half_size]
            half2_indices = neuron_indices[half_size:2 * half_size]

            # Create two subregion datasets
            subregion1_data = region1_data[:, half1_indices]
            subregion2_data = region1_data[:, half2_indices]

            print(f"Iteration {iteration + 1}: Subregion1: {subregion1_data.shape[1]} neurons, "
                  f"Subregion2: {subregion2_data.shape[1]} neurons")

            # Run CCA cross-validation on the two subregions
            test_correlation_matrix, train_correlation_matrix = self.run_cca_cross_validation(
                subregion1_data, subregion2_data)

            # Calculate r_bar (average across folds)
            r_bar = np.mean(test_correlation_matrix, axis=0)
            r_bar_train = np.mean(train_correlation_matrix, axis=0)

            # Run permutation test
            permutation_correlations = self.permutation_test(subregion1_data, subregion2_data)

            # Statistical testing using permutation results
            significant_components, p_values = self.permutation_test_components(
                r_bar, permutation_correlations)

            # Calculate correlations using only significant components
            test_correlations_d, train_correlations_d, mean_correlation_d = (
                self.calculate_significant_component_correlations(
                    subregion1_data, subregion2_data, significant_components))

            # Create scree plot for this iteration
            region_pair = f"Primary_auditory_area_subregion1_vs_subregion2"
            self.create_scree_plot(test_correlation_matrix, train_correlation_matrix,
                                   permutation_correlations, p_values, region_pair,
                                   stimulus, response_range, f"{session}_iter{iteration + 1}",
                                   verbose=verbose)

            # Store results for this iteration
            result = {
                'region1': 'Primary auditory area',
                'region2': 'Primary auditory area',
                'region_pair': 'Primary_auditory_area_vs_Primary_auditory_area',
                'stimulus': stimulus,
                'response_range': response_range,
                'session': session,
                'iteration': iteration + 1,
                'total_neurons_available': total_neurons,
                'subregion1_neurons': half_size,
                'subregion2_neurons': half_size,
                'significant_components': significant_components,
                'total_components': test_correlation_matrix.shape[1],
                'r_bar': r_bar,
                'r_bar_train': r_bar_train,
                'test_correlations_std': test_correlation_matrix.std(axis=0),
                'train_correlations_std': train_correlation_matrix.std(axis=0),
                'p_values': p_values,
                'permutation_correlations': permutation_correlations,
                'test_correlations_d': test_correlations_d,
                'train_correlations_d': train_correlations_d,
                'mean_correlation_d': mean_correlation_d,
                'r_max_d': np.max(test_correlations_d) if test_correlations_d.size > 0 else np.array([]),  # TODO: Choose something else to save if if array is empty
                'r_max_train_d': np.max(train_correlations_d) if train_correlations_d.size > 0 else np.array([]),
                'analysis_type': 'within_region'
            }
            iteration_results.append(result)

            print(f"Iteration {iteration + 1}: Found {significant_components} significant components, "
                  f"mean d-component correlation: {mean_correlation_d:.3f}")

        return iteration_results

    def analyze_region_pair(self, region1_name, region2_name, n_iterations=10, verbose=True):
        """
        Analyze a specific pair of brain regions across all conditions
        With special handling for Primary auditory area self-comparison

        Parameters:
        -----------
        n_iterations : int
            Number of iterations for sampling neurons (applies to between-region analysis)
        """
        results = []

        print(f"\nAnalyzing region pair: {region1_name} vs {region2_name}")

        # Check if this is a Primary auditory area self-comparison
        is_primary_self_comparison = (region1_name == "Primary auditory area" and
                                      region2_name == "Primary auditory area")

        for stimulus in self.stim_types:
            print(f"Processing stimulus: {stimulus}")

            # Get stimulus data
            stim_arrays = self.fr_db.return_arrays(stimulus)
            brainRegionArray = stim_arrays["brainRegionArray"]
            brainRegionArray[brainRegionArray == "Posterior auditory area"] = "Dorsal auditory area"
            sessionArray = stim_arrays["sessionIDArray"]
            uniqSessions = np.unique(sessionArray)

            for response_range in self.response_ranges:
                respArray = stim_arrays[f"{response_range}fr"]

                for session in uniqSessions:
                    session_mask = sessionArray == session
                    session_resp_array = respArray[session_mask, :]
                    brain_session_array = brainRegionArray[session_mask]

                    if is_primary_self_comparison:
                        # Special handling for Primary auditory area self-comparison
                        region_mask = brain_session_array == region1_name
                        region_data = session_resp_array[region_mask, :].T  # Transpose it to be [Trials, Nuerons]

                        # Check if we have enough neurons for within-region analysis
                        min_neurons_needed = 2 * self.neuron_threshold
                        if region_data.shape[1] < min_neurons_needed:
                            print(f"Skipping Primary auditory area within-region analysis for session {session}: "
                                  f"only {region_data.shape[1]} neurons available "
                                  f"(need at least {min_neurons_needed})")
                            continue

                        print(f"Primary auditory area within-region analysis: {region_data.shape[1]} total neurons")

                        # Perform within-region analysis with multiple random splits
                        within_region_results = self.analyze_primary_auditory_within_region(
                            region_data, session, stimulus, response_range, n_iterations=n_iterations, verbose=verbose)

                        results.extend(within_region_results)

                    else:
                        # Between-region analysis with multiple iterations
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

                        print(f"Session {session}, Response: {response_range} - "
                              f"R1: {region1_data.shape[1]} neurons, R2: {region2_data.shape[1]} neurons, "
                              f"{n_iterations} iterations")

                        # Perform multiple iterations with different neuron sampling
                        iteration_results = []

                        for iteration in range(n_iterations):
                            # Set seed for reproducibility within each iteration
                            np.random.seed(self.random_state + iteration)

                            # Sample neurons for this iteration
                            region1_sampled = region1_data.copy()
                            region2_sampled = region2_data.copy()

                            if region1_sampled.shape[1] > self.neuron_threshold:
                                r1_neurons = np.random.choice(region1_sampled.shape[1],
                                                              size=self.neuron_threshold,
                                                              replace=False)
                                region1_sampled = region1_sampled[:, r1_neurons]

                            if region2_sampled.shape[1] > self.neuron_threshold:
                                r2_neurons = np.random.choice(region2_sampled.shape[1],
                                                              size=self.neuron_threshold,
                                                              replace=False)
                                region2_sampled = region2_sampled[:, r2_neurons]

                            print(f"Iteration {iteration + 1}: R1: {region1_sampled.shape[1]} neurons, "
                                  f"R2: {region2_sampled.shape[1]} neurons")

                            # Run CCA cross-validation
                            test_correlation_matrix, train_correlation_matrix = self.run_cca_cross_validation(
                                region1_sampled, region2_sampled)

                            # Calculate r_bar (average across folds)
                            r_bar = np.mean(test_correlation_matrix, axis=0)
                            r_bar_train = np.mean(train_correlation_matrix, axis=0)

                            # Run permutation test
                            permutation_correlations = self.permutation_test(region1_sampled, region2_sampled)

                            # Statistical testing using permutation results
                            significant_components, p_values = self.permutation_test_components(
                                r_bar, permutation_correlations)

                            # Calculate correlations using only significant components
                            test_correlations_d, train_correlations_d, mean_correlation_d = (
                                self.calculate_significant_component_correlations(
                                    region1_sampled, region2_sampled, significant_components))

                            # Create scree plot for this iteration
                            region_pair = f"{region1_name}_vs_{region2_name}"
                            self.create_scree_plot(test_correlation_matrix, train_correlation_matrix,
                                                   permutation_correlations, p_values, region_pair,
                                                   stimulus, response_range, f"{session}_iter{iteration + 1}",
                                                   verbose=verbose)

                            # Store results for this iteration
                            result = {
                                'region1': region1_name,
                                'region2': region2_name,
                                'region_pair': region_pair,
                                'stimulus': stimulus,
                                'response_range': response_range,
                                'session': session,
                                'iteration': iteration + 1,
                                'region1_neurons': region1_sampled.shape[1],
                                'region2_neurons': region2_sampled.shape[1],
                                'significant_components': significant_components,
                                'total_components': test_correlation_matrix.shape[1],
                                'r_bar': r_bar,
                                'r_bar_train': r_bar_train,
                                'test_correlations_std': test_correlation_matrix.std(axis=0),
                                'train_correlations_std': train_correlation_matrix.std(axis=0),
                                'p_values': p_values,
                                'permutation_correlations': permutation_correlations,
                                'test_correlations_d': test_correlations_d,
                                'train_correlations_d': train_correlations_d,
                                'mean_correlation_d': mean_correlation_d,
                                'r_max_d': np.max(test_correlations_d) if test_correlations_d.size > 0 else np.array([]),
                                'r_max_train_d': np.max(train_correlations_d) if train_correlations_d.size > 0 else np.array([]),
                                'analysis_type': 'between_region'
                            }
                            iteration_results.append(result)

                            print(f"Iteration {iteration + 1}: Found {significant_components} significant components, "
                                  f"mean d-component correlation: {mean_correlation_d:.3f}")

                        # Add all iteration results
                        results.extend(iteration_results)

                        with open(os.path.join(self.output_dir, f"cca_primary_auditory_results_partial_intermediate_pairs.pkl"), 'wb') as f:
                            pickle.dump(results, f)

        return results

    def run_analysis(self, region_pairs=None, n_iterations=10, verbose=True):
        """
        Run the full analysis for specified region pairs (filtered for Primary auditory area)
        With special handling for Primary auditory area within-region analysis

        Parameters:
        -----------
        n_iterations : int
            Number of iterations for sampling neurons
        """
        # Get available regions
        stim_arrays = self.fr_db.return_arrays(list(self.stim_types)[0])
        brain_regions = stim_arrays["brainRegionArray"]
        # Rename posterior to also be dorsal instead of treating them as separate areas
        brain_regions[brain_regions == "Posterior auditory area"] = "Dorsal auditory area"
        uniq_regions = np.unique(brain_regions)

        if region_pairs is None:
            # Generate pairs involving Primary auditory area
            region_pairs = []
            primary_region = "Primary auditory area"

            if primary_region in uniq_regions:
                # Add Primary auditory area with itself for within-region analysis
                region_pairs.append((primary_region, primary_region))

                # Add Primary auditory area with other regions
                for region in uniq_regions:
                    if region != primary_region:
                        region_pairs.append((primary_region, region))
            else:
                print("Primary auditory area not found in available regions:")
                print(uniq_regions)
                return None

        print(f"Analyzing {len(region_pairs)} region pairs involving Primary auditory area...")
        print(f"Using {n_iterations} iterations per condition")
        print("Note: Primary auditory area vs itself will use within-region split analysis")

        all_results = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.analyze_region_pair, region1, region2, n_iterations, verbose): (region1, region2)
                for region1, region2 in region_pairs
            }

            for future in as_completed(futures):
                region1, region2 = futures[future]
                try:
                    pair_results = future.result()
                    all_results.extend(pair_results)
                    with open(os.path.join(self.output_dir, "cca_primary_auditory_results_full_intermediate_pairs.pkl"),
                              'wb') as f:
                        pickle.dump(all_results, f)
                except Exception as e:
                    print(f"Region pair ({region1}, {region2}) failed with error: {e}")

        if not all_results:
            print("No results generated. Check neuron thresholds and data availability.")
            return None

        # Save full results with arrays as pickle
        with open(os.path.join(self.output_dir, "cca_primary_auditory_results_full.pkl"), 'wb') as f:
            pickle.dump(all_results, f)

        # Convert to DataFrame and save (excluding large arrays)
        results_for_df = []
        for result in all_results:
            result_copy = result.copy()
            # Remove large arrays for CSV storage
            result_copy.pop('permutation_correlations', None)
            result_copy.pop('test_correlations_d', None)
            result_copy.pop('train_correlations_d', None)
            results_for_df.append(result_copy)

        results_save_df = pd.DataFrame(results_for_df)


        # Save detailed results
        results_save_df.to_csv(os.path.join(self.output_dir, "cca_primary_auditory_results.csv"), index=False)
        try:
            results_save_df.to_feather(os.path.join(self.output_dir, "cca_primary_auditory_results.feather"))
        except Exception as e:
            print(f"Failed to save feather file: {e}")

        results_df = pd.DataFrame(all_results)

        # # Save full results with arrays as pickle
        # with open(os.path.join(self.output_dir, "cca_primary_auditory_results_full.pkl"), 'wb') as f:
        #     pickle.dump(all_results, f)

        # Create summary plots (now filtered for Primary auditory area)
        self.create_summary_plots(results_df)

        # Create within-region vs between-region comparison
        print("Creating within-region vs between-region comparison...")
        self.create_within_region_summary(results_df)

        # Create detailed summary table
        summary_table = self.create_detailed_summary_table(results_df)

        # Create d-component specific visualizations
        print("Creating d-component correlation visualizations...")
        self.create_d_component_summary_plots(results_df)

        print(f"Summary table shape: {summary_table.shape}")

        # Print detailed statistical summary
        within_region_results = results_df[results_df.get('analysis_type', '') == 'within_region']
        between_region_results = results_df[results_df.get('analysis_type', '') != 'within_region']

        print(f"\nAnalysis Summary:")
        print(f"Within-region analyses (Primary auditory area split): {len(within_region_results)} total iterations")
        print(f"Between-region analyses: {len(between_region_results)} total iterations")

        if len(within_region_results) > 0:
            within_d_comp = within_region_results[within_region_results['significant_components'] > 0]
            if len(within_d_comp) > 0:
                print(f"\nWithin-Primary auditory area (d-component correlations):")
                print(
                    f"Mean ± SEM: {within_d_comp['mean_correlation_d'].mean():.3f} ± {within_d_comp['mean_correlation_d'].sem():.3f}")
                print(
                    f"Range: {within_d_comp['mean_correlation_d'].min():.3f} - {within_d_comp['mean_correlation_d'].max():.3f}")

        if len(between_region_results) > 0:
            between_d_comp = between_region_results[between_region_results['significant_components'] > 0]
            if len(between_d_comp) > 0:
                print(f"\nBetween-region (d-component correlations):")
                print(
                    f"Mean ± SEM: {between_d_comp['mean_correlation_d'].mean():.3f} ± {between_d_comp['mean_correlation_d'].sem():.3f}")
                print(
                    f"Range: {between_d_comp['mean_correlation_d'].min():.3f} - {between_d_comp['mean_correlation_d'].max():.3f}")

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

    ########## D-component analysis code ##########

    def calculate_significant_component_correlations(self, region1_data, region2_data, n_significant_components):
        """
        Calculate correlations using only the first d (significant) components

        Parameters:
        -----------
        region1_data : np.ndarray
            Neural data from region 1 (trials x neurons)
        region2_data : np.ndarray
            Neural data from region 2 (trials x neurons)
        n_significant_components : int
            Number of significant components to use

        Returns:
        --------
        test_correlations_d : np.ndarray
            Test correlations for significant components only (n_folds, d_components)
        train_correlations_d : np.ndarray
            Training correlations for significant components only (n_folds, d_components)
        mean_correlation_d : float
            Mean correlation across significant components and folds
        """
        if n_significant_components == 0:
            return np.array([]), np.array([]), 0.0

        n_components = min(region1_data.shape[1], region2_data.shape[1], n_significant_components)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        test_correlations_d = []
        train_correlations_d = []

        for train_idx, test_idx in kf.split(region1_data):
            r1_train, r1_test = region1_data[train_idx], region1_data[test_idx]
            r2_train, r2_test = region2_data[train_idx], region2_data[test_idx]

            # Fit CCA with only significant components
            cca = CCA(n_components=n_components)
            cca.fit(r1_train, r2_train)

            # Transform data
            r1_test_transform, r2_test_transform = cca.transform(r1_test, r2_test)
            r1_train_transform, r2_train_transform = cca.transform(r1_train, r2_train)

            # Calculate correlations for significant components only
            test_fold_correlations = []
            train_fold_correlations = []

            for comp in range(n_components):
                test_corr = np.corrcoef(r1_test_transform[:, comp], r2_test_transform[:, comp])[0, 1]
                train_corr = np.corrcoef(r1_train_transform[:, comp], r2_train_transform[:, comp])[0, 1]

                test_fold_correlations.append(test_corr)
                train_fold_correlations.append(train_corr)

            test_correlations_d.append(test_fold_correlations)
            train_correlations_d.append(train_fold_correlations)

        test_correlations_d = np.array(test_correlations_d)
        train_correlations_d = np.array(train_correlations_d)

        # Calculate mean correlation across all significant components and folds
        mean_correlation_d = np.mean(test_correlations_d) if test_correlations_d.size > 0 else 0.0

        return test_correlations_d, train_correlations_d, mean_correlation_d

    def create_summary_plots(self, results_df):
        """
        Create summary plots of the analysis results (filtered for Primary auditory area)
        """
        # Filter for Primary auditory area comparisons
        filtered_df = self.filter_primary_auditory_results(results_df)

        if len(filtered_df) == 0:
            print("No results found for Primary auditory area comparisons.")
            return

        # Create enhanced three-panel plot with statistics
        self.create_three_panel_summary_with_stats(
            filtered_df,
            value_column='significant_components',
            ylabel='Number of Significant Components',
            title_suffix=' - Significant Components'
        )

        # Summary by stimulus and response range (if multiple stimuli)
        if len(filtered_df['stimulus'].unique()) > 1:
            plt.figure(figsize=(15, 10))
            sns.boxplot(data=filtered_df, x='region_pair', y='significant_components',
                        hue='stimulus')
            plt.xticks(rotation=45, ha='right')
            plt.title('CCA Significant Components - Primary Auditory Area Comparisons',
                      fontsize=14, fontweight='bold')
            plt.ylabel('Number of Significant Components')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "summary_by_stimulus_primary_auditory.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def create_d_component_summary_plots(self, results_df):
        """
        Create summary plots showing correlations calculated using only significant (d) components
        """
        # Filter for Primary auditory area and significant components
        primary_df = self.filter_primary_auditory_results(results_df)
        filtered_df = primary_df[primary_df['significant_components'] > 0].copy()

        if len(filtered_df) == 0:
            print("No significant components found for Primary auditory area comparisons.")
            return
        # TODO: Use the test_d correlation to calculate first, and mean of first 2-5 components and make plots of all
        #  as the value column below
        # Column name is test_correlations_d
        for d_comps in range(1, 6):
            for row_i, row in filtered_df.iterrows():
                test_correlations_d = row['test_correlations_d']
                test_subset = test_correlations_d[:, :d_comps]
                mean_test_d = np.mean(test_subset) if test_subset.size > 0 else 0.0
                filtered_df.loc[row_i, f'test_correlations_d_{d_comps}_comps'] = mean_test_d

            # Create three-panel plot for d-component correlations with statistics
            self.create_three_panel_summary_with_stats(
                filtered_df,
                value_column=f'test_correlations_d_{d_comps}_comps',
                ylabel='Mean Correlation (d components only)',
                title_suffix=f' - D-Component Correlations - First {d_comps} Components'
            )

        # Create three-panel plot for d-component correlations with statistics
        self.create_three_panel_summary_with_stats(
            filtered_df,
            value_column='mean_correlation_d',  # TODO: Could change to only plot the first or average of first 3 correlation values
            ylabel='Mean Correlation (d components only)',
            title_suffix=' - D-Component Correlations'
        )

        # Comparison plot: all components vs d components (Primary auditory area only)
        self.create_comparison_all_vs_d_components_primary(filtered_df)

        # Heatmaps for d-component correlations (Primary auditory area only)
        self.create_d_component_heatmaps_primary(filtered_df)

    def create_comparison_all_vs_d_components_primary(self, results_df, verbose=True):
        """
        Create comparison plots for Primary auditory area showing all vs d components correlations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Scatter plot: all vs d components
        ax1 = axes[0, 0]
        all_comp_means = [np.mean(r_bar) for r_bar in results_df['r_bar']]
        ax1.scatter(results_df['mean_correlation_d'], all_comp_means, alpha=0.6, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect agreement')
        ax1.set_xlabel('Mean Correlation (d components)', fontsize=12)
        ax1.set_ylabel('Mean Correlation (all components)', fontsize=12)
        ax1.set_title('All vs D Components - Primary Auditory Area', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Difference plot
        ax2 = axes[0, 1]
        differences = np.array(results_df['mean_correlation_d']) - np.array(all_comp_means)
        ax2.hist(differences, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Difference (d - all components)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Correlation Difference Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Box plot by region pair
        ax3 = axes[1, 0]
        unique_pairs = sorted(results_df['region_pair'].unique())
        box_data = [results_df[results_df['region_pair'] == pair]['mean_correlation_d'].values
                    for pair in unique_pairs]

        bp = ax3.boxplot(box_data, labels=unique_pairs, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax3.set_xticklabels(unique_pairs, rotation=45, ha='right')
        ax3.set_ylabel('Mean Correlation (d components)', fontsize=12)
        ax3.set_title('D-Component Correlations by Region Pair', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Statistical summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')

        # Create summary statistics table
        summary_stats = []
        for pair in unique_pairs:
            pair_data = results_df[results_df['region_pair'] == pair]['mean_correlation_d']
            stats_row = [
                pair,
                f"{pair_data.mean():.3f}",
                f"{pair_data.std():.3f}",
                f"{pair_data.std() / np.sqrt(len(pair_data)):.3f}",  # SEM
                f"{len(pair_data)}"
            ]
            summary_stats.append(stats_row)

        table = ax4.table(cellText=summary_stats,
                          colLabels=['Region Pair', 'Mean', 'Std', 'SEM', 'N'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_all_vs_d_primary_auditory.png"),
                    dpi=300, bbox_inches='tight')
        if verbose:
            plt.show()

    def create_d_component_heatmaps_primary(self, results_df):
        """
        Create heatmaps for Primary auditory area showing d-component correlations
        """
        for stimulus in results_df['stimulus'].unique():
            stimulus_data = results_df[results_df['stimulus'] == stimulus]

            # Create pivot table for heatmap
            pivot_data = stimulus_data.pivot_table(
                values='mean_correlation_d',
                index='response_range',
                columns='region_pair',
                aggfunc='mean'
            )

            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f',
                        cbar_kws={'label': 'Mean Correlation (d components)'})
            plt.title(f'D-Component Correlations - Primary Auditory Area - {stimulus}',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Region Pair', fontsize=12)
            plt.ylabel('Response Range', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            safe_stimulus = stimulus.replace(' ', '_').replace('/', '_')
            plt.savefig(os.path.join(self.output_dir,
                                     f"heatmap_d_components_primary_auditory_{safe_stimulus}.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def create_comparison_all_vs_d_components_primary(self, results_df):
        """
        Create comparison plots for Primary auditory area showing all vs d components correlations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Scatter plot: all vs d components
        ax1 = axes[0, 0]
        all_comp_means = [np.mean(r_bar) for r_bar in results_df['r_bar']]
        ax1.scatter(results_df['mean_correlation_d'], all_comp_means, alpha=0.6, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect agreement')
        ax1.set_xlabel('Mean Correlation (d components)', fontsize=12)
        ax1.set_ylabel('Mean Correlation (all components)', fontsize=12)
        ax1.set_title('All vs D Components - Primary Auditory Area', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Difference plot
        ax2 = axes[0, 1]
        differences = np.array(results_df['mean_correlation_d']) - np.array(all_comp_means)
        ax2.hist(differences, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Difference (d - all components)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Correlation Difference Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Box plot by region pair
        ax3 = axes[1, 0]
        unique_pairs = sorted(results_df['region_pair'].unique())
        box_data = [results_df[results_df['region_pair'] == pair]['mean_correlation_d'].values
                    for pair in unique_pairs]

        bp = ax3.boxplot(box_data, labels=unique_pairs, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax3.set_xticklabels(unique_pairs, rotation=45, ha='right')
        ax3.set_ylabel('Mean Correlation (d components)', fontsize=12)
        ax3.set_title('D-Component Correlations by Region Pair', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Statistical summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')

        # Create summary statistics table
        summary_stats = []
        for pair in unique_pairs:
            pair_data = results_df[results_df['region_pair'] == pair]['mean_correlation_d']
            stats_row = [
                pair,
                f"{pair_data.mean():.3f}",
                f"{pair_data.std():.3f}",
                f"{pair_data.std() / np.sqrt(len(pair_data)):.3f}",  # SEM
                f"{len(pair_data)}"
            ]
            summary_stats.append(stats_row)

        table = ax4.table(cellText=summary_stats,
                          colLabels=['Region Pair', 'Mean', 'Std', 'SEM', 'N'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_all_vs_d_primary_auditory.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_d_component_heatmaps_primary(self, results_df):
        """
        Create heatmaps for Primary auditory area showing d-component correlations
        """
        for stimulus in results_df['stimulus'].unique():
            stimulus_data = results_df[results_df['stimulus'] == stimulus]

            # Create pivot table for heatmap
            pivot_data = stimulus_data.pivot_table(
                values='mean_correlation_d',
                index='response_range',
                columns='region_pair',
                aggfunc='mean'
            )

            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f',
                        cbar_kws={'label': 'Mean Correlation (d components)'})
            plt.title(f'D-Component Correlations - Primary Auditory Area - {stimulus}',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Region Pair', fontsize=12)
            plt.ylabel('Response Range', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            safe_stimulus = stimulus.replace(' ', '_').replace('/', '_')
            plt.savefig(os.path.join(self.output_dir,
                                     f"heatmap_d_components_primary_auditory_{safe_stimulus}.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()


    def create_three_panel_d_component_plot(self, results_df):
        """
        Create three-panel plot showing d-component correlations by stimulus, response range, and region pair
        """
        unique_stimuli = sorted(results_df['stimulus'].unique())
        unique_response_ranges = sorted(results_df['response_range'].unique())
        unique_region_pairs = sorted(results_df['region_pair'].unique())

        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_region_pairs)))
        region_pair_colors = dict(zip(unique_region_pairs, colors))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        bar_width = 0.8 / len(unique_region_pairs)

        for i, stimulus in enumerate(unique_stimuli):
            ax = axes[i]
            stimulus_data = results_df[results_df['stimulus'] == stimulus]
            x_positions = np.arange(len(unique_response_ranges))

            for j, region_pair in enumerate(unique_region_pairs):
                pair_data = stimulus_data[stimulus_data['region_pair'] == region_pair]

                means = []
                sems = []

                for response_range in unique_response_ranges:
                    range_data = pair_data[pair_data['response_range'] == response_range]
                    if len(range_data) > 0:
                        means.append(range_data['mean_correlation_d'].mean())
                        sems.append(range_data['mean_correlation_d'].std()/np.sqrt(len(range_data['mean_correlation_d'])))
                    else:
                        means.append(0)
                        sems.append(0)

                x_pos = x_positions + j * bar_width - (len(unique_region_pairs) - 1) * bar_width / 2
                bars = ax.bar(x_pos, means, bar_width, yerr=sems,
                              color=region_pair_colors[region_pair],
                              alpha=0.7, label=region_pair, capsize=3)

                # Add value labels
                for k, (bar, mean) in enumerate(zip(bars, means)):
                    if mean > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sems[k] + 0.01,
                                f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

            ax.set_xlabel('Response Range', fontsize=12)
            if i == 0:
                ax.set_ylabel('Mean Correlation (d components only)', fontsize=12)
            ax.set_title(f'{stimulus}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(unique_response_ranges)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, None)

            if i == len(unique_stimuli) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        fig.suptitle('CCA Correlations Using Only Significant (d) Components',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "three_panel_d_components.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_comparison_all_vs_d_components(self, results_df):
        """
        Create comparison plots showing all components vs d components correlations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Scatter plot: all vs d components
        ax1 = axes[0, 0]
        ax1.scatter(results_df['mean_correlation_d'], [np.mean(r_bar) for r_bar in results_df['r_bar']],
                    alpha=0.6, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect agreement')
        ax1.set_xlabel('Mean Correlation (d components)', fontsize=12)
        ax1.set_ylabel('Mean Correlation (all components)', fontsize=12)
        ax1.set_title('All Components vs D Components Correlation', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Difference plot
        ax2 = axes[0, 1]
        all_comp_means = [np.mean(r_bar) for r_bar in results_df['r_bar']]
        differences = np.array(results_df['mean_correlation_d']) - np.array(all_comp_means)
        ax2.hist(differences, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Difference (d components - all components)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Correlation Differences', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Box plot by number of significant components
        ax3 = axes[1, 0]
        sig_comp_groups = results_df.groupby('significant_components')['mean_correlation_d'].apply(list)
        box_data = [group for _, group in sig_comp_groups.items()]
        box_labels = [f'd={k}' for k in sig_comp_groups.keys()]

        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax3.set_xlabel('Number of Significant Components', fontsize=12)
        ax3.set_ylabel('Mean Correlation (d components)', fontsize=12)
        ax3.set_title('Correlation by Number of Significant Components', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Region pair comparison
        ax4 = axes[1, 1]
        region_means = results_df.groupby('region_pair')['mean_correlation_d'].mean().sort_values(ascending=True)
        bars = ax4.barh(range(len(region_means)), region_means.values, alpha=0.7)
        ax4.set_yticks(range(len(region_means)))
        ax4.set_yticklabels(region_means.index, fontsize=10)
        ax4.set_xlabel('Mean Correlation (d components)', fontsize=12)
        ax4.set_title('Average d-Component Correlation by Region Pair', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, region_means.values)):
            ax4.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                     ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_all_vs_d_components.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_d_component_heatmaps(self, results_df):
        """
        Create heatmaps showing d-component correlations across conditions
        """
        for stimulus in results_df['stimulus'].unique():
            stimulus_data = results_df[results_df['stimulus'] == stimulus]

            # Create pivot table for heatmap
            pivot_data = stimulus_data.pivot_table(
                values='mean_correlation_d',
                index='response_range',
                columns='region_pair',
                aggfunc='mean'
            )

            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f',
                        cbar_kws={'label': 'Mean Correlation (d components)'})
            plt.title(f'Mean d-Component Correlations - {stimulus}',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Region Pair', fontsize=12)
            plt.ylabel('Response Range', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            safe_stimulus = stimulus.replace(' ', '_').replace('/', '_')
            plt.savefig(os.path.join(self.output_dir, f"heatmap_d_components_{safe_stimulus}.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()

    def filter_primary_auditory_results(self, results_df):
        """
        Filter results to only include comparisons involving Primary auditory area

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe from the CCA analysis

        Returns:
        --------
        filtered_df : pd.DataFrame
            Filtered dataframe with only Primary auditory area comparisons
        """
        primary_mask = (
                (results_df['region1'] == 'Primary auditory area') |
                (results_df['region2'] == 'Primary auditory area')
        )

        filtered_df = results_df[primary_mask].copy()
        print(f"Filtered to {len(filtered_df)} results involving Primary auditory area "
              f"(from {len(results_df)} total results)")

        return filtered_df

    def perform_wilcoxon_tests_with_bonferroni(self, data_df, value_column, grouping_column):
        """
        Perform pairwise Wilcoxon tests with Bonferroni correction

        Parameters:
        -----------
        data_df : pd.DataFrame
            Data for statistical testing
        value_column : str
            Column name containing values to test
        grouping_column : str
            Column name for grouping (e.g., 'region_pair')

        Returns:
        --------
        results_dict : dict
            Dictionary with statistical test results
        """
        from scipy.stats import mannwhitneyu
        from itertools import combinations

        groups = data_df[grouping_column].unique()
        n_comparisons = len(list(combinations(groups, 2)))

        if n_comparisons == 0:
            return {}

        alpha = 0.05
        bonferroni_alpha = alpha / n_comparisons

        results = []

        for group1, group2 in combinations(groups, 2):
            data1 = data_df[data_df[grouping_column] == group1][value_column].values
            data2 = data_df[data_df[grouping_column] == group2][value_column].values

            if len(data1) > 0 and len(data2) > 0:
                try:
                    stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')

                    results.append({
                        'group1': group1,
                        'group2': group2,
                        'statistic': stat,
                        'p_value': p_val,
                        'bonferroni_corrected_p': p_val * n_comparisons,
                        'significant_bonferroni': (p_val * n_comparisons) < alpha,
                        'mean1': np.mean(data1),
                        'mean2': np.mean(data2),
                        'n1': len(data1),
                        'n2': len(data2)
                    })
                except ValueError:
                    # Handle cases where all values are identical
                    continue

        return {
            'results': results,
            'n_comparisons': n_comparisons,
            'bonferroni_alpha': bonferroni_alpha,
            'alpha': alpha
        }

    def add_significance_brackets_to_plot(self, ax, data_df, value_column, x_positions,
                                          region_pairs, y_max_offset=0.1):
        """
        Add significance brackets to plots based on statistical tests

        Parameters:
        -----------
        ax : matplotlib axis
            Axis to add brackets to
        data_df : pd.DataFrame
            Data used for plotting
        value_column : str
            Column name containing values
        x_positions : array
            X positions of bars
        region_pairs : list
            List of region pairs in order
        y_max_offset : float
            Offset for bracket positioning
        """
        # Perform statistical tests
        stats_results = self.perform_wilcoxon_tests_with_bonferroni(
            data_df, value_column, 'region_pair')

        if not stats_results or len(stats_results.get('results', [])) == 0:
            return

        # Find significant pairs
        significant_pairs = [
            r for r in stats_results['results']
            if r['significant_bonferroni']
        ]

        if not significant_pairs:
            return

        # Get y-axis limits for bracket positioning
        y_max = ax.get_ylim()[1]
        bracket_height = y_max * 0.05  # Height of significance brackets

        # Sort pairs by p-value for better bracket positioning
        significant_pairs.sort(key=lambda x: x['bonferroni_corrected_p'])

        for i, pair in enumerate(significant_pairs):
            group1, group2 = pair['group1'], pair['group2']

            try:
                x1 = region_pairs.index(group1)
                x2 = region_pairs.index(group2)

                # Position brackets at different heights to avoid overlap
                bracket_y = y_max + (y_max_offset * y_max) + (i * bracket_height)

                # Draw bracket
                ax.plot([x1, x1, x2, x2],
                        [bracket_y, bracket_y + bracket_height / 2,
                         bracket_y + bracket_height / 2, bracket_y],
                        'k-', linewidth=1)

                # Add significance stars
                n_stars = min(3, max(1, int(-np.log10(pair['bonferroni_corrected_p']))))
                stars = '*' * n_stars
                ax.text((x1 + x2) / 2, bracket_y + bracket_height / 2, stars,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            except ValueError:
                # Handle cases where region pair not found in list
                continue

        # Adjust y-axis limits to accommodate brackets
        if significant_pairs:
            new_y_max = y_max + (y_max_offset * y_max) + (len(significant_pairs) * bracket_height) * 1.2
            ax.set_ylim(ax.get_ylim()[0], new_y_max)

    def create_three_panel_summary_with_stats(self, results_df, value_column='significant_components',
                                              ylabel='Number of Significant Components',
                                              title_suffix=''):
        """
        Create three-panel summary plot with SEM and statistical testing

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe (should already be filtered for Primary auditory area)
        value_column : str
            Column to plot
        ylabel : str
            Y-axis label
        title_suffix : str
            Suffix for plot title
        """
        unique_stimuli = sorted(results_df['stimulus'].unique())
        unique_response_ranges = sorted(results_df['response_range'].unique())
        unique_region_pairs = sorted(results_df['region_pair'].unique())

        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_region_pairs)))
        region_pair_colors = dict(zip(unique_region_pairs, colors))

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
        bar_width = 0.8 / len(unique_region_pairs)

        for i, stimulus in enumerate(unique_stimuli):
            ax = axes[i]
            stimulus_data = results_df[results_df['stimulus'] == stimulus]

            for j, response_range in enumerate(unique_response_ranges):
                response_data = stimulus_data[stimulus_data['response_range'] == response_range]

                if len(response_data) == 0:
                    continue

                x_positions = []
                means = []
                sems = []
                region_pairs_present = []

                for k, region_pair in enumerate(unique_region_pairs):
                    pair_data = response_data[response_data['region_pair'] == region_pair]

                    if len(pair_data) > 0:
                        mean_val = pair_data[value_column].mean()
                        sem_val = pair_data[value_column].std() / np.sqrt(len(pair_data))  # SEM calculation

                        x_pos = j + k * bar_width - (len(unique_region_pairs) - 1) * bar_width / 2

                        bar = ax.bar(x_pos, mean_val, bar_width, yerr=sem_val,
                                     color=region_pair_colors[region_pair],
                                     alpha=0.7, label=region_pair if j == 0 else "",
                                     capsize=3)

                        x_positions.append(x_pos)
                        means.append(mean_val)
                        sems.append(sem_val)
                        region_pairs_present.append(region_pair)

                        # Add value labels
                        if mean_val > 0:
                            ax.text(x_pos, mean_val + sem_val + 0.01 * ax.get_ylim()[1],
                                    f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)

                # Add statistical testing for this response range
                if len(region_pairs_present) > 1:
                    # Create bracket positions relative to response range
                    range_x_positions = [(j + k * bar_width - (len(unique_region_pairs) - 1) * bar_width / 2)
                                         for k in range(len(region_pairs_present))]

                    # Perform statistical tests for this response range
                    stats_results = self.perform_wilcoxon_tests_with_bonferroni(
                        response_data, value_column, 'region_pair')

                    if stats_results and stats_results.get('results'):
                        significant_pairs = [
                            r for r in stats_results['results']
                            if r['significant_bonferroni']
                        ]

                        # Add brackets for significant comparisons
                        y_max = max(means) + max(sems) if means else 1
                        bracket_height = y_max * 0.1

                        for bracket_idx, pair in enumerate(significant_pairs):
                            try:
                                idx1 = region_pairs_present.index(pair['group1'])
                                idx2 = region_pairs_present.index(pair['group2'])

                                x1, x2 = range_x_positions[idx1], range_x_positions[idx2]
                                bracket_y = y_max + bracket_height * (1 + bracket_idx * 0.5)

                                # Draw bracket
                                ax.plot([x1, x1, x2, x2],
                                        [bracket_y, bracket_y + bracket_height / 4,
                                         bracket_y + bracket_height / 4, bracket_y],
                                        'k-', linewidth=1)

                                # Add significance indication
                                p_val = pair['bonferroni_corrected_p']
                                if p_val < 0.001:
                                    sig_text = '***'
                                elif p_val < 0.01:
                                    sig_text = '**'
                                elif p_val < 0.05:
                                    sig_text = '*'
                                else:
                                    sig_text = 'ns'

                                ax.text((x1 + x2) / 2, bracket_y + bracket_height / 4, sig_text,
                                        ha='center', va='bottom', fontsize=10, fontweight='bold')

                            except ValueError:
                                continue

            # Customize subplot
            ax.set_xlabel('Response Range', fontsize=12)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{stimulus}', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(unique_response_ranges)))
            ax.set_xticklabels(unique_response_ranges)
            ax.grid(True, alpha=0.3, axis='y')

            # Add legend only to the last subplot
            if i == len(unique_stimuli) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Overall title
        main_title = f'CCA Analysis - Primary Auditory Area Comparisons{title_suffix}'
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save plot
        safe_suffix = title_suffix.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"three_panel_primary_auditory{safe_suffix}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def create_within_region_summary(self, results_df):
        """
        Create summary comparing within-region vs between-region correlations
        """
        # Separate within-region and between-region results
        within_region_df = results_df[results_df.get('analysis_type', '') == 'within_region'].copy()
        between_region_df = results_df[results_df.get('analysis_type', '') != 'within_region'].copy()

        if len(within_region_df) == 0:
            print("No within-region analysis results found.")
            return

        print(f"Within-region results: {len(within_region_df)}")
        print(f"Between-region results: {len(between_region_df)}")

        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Significant components comparison
        ax1 = axes[0, 0]

        # Prepare data for plotting
        within_sig_comp = within_region_df['significant_components']
        between_sig_comp = between_region_df['significant_components']

        box_data = [within_sig_comp, between_sig_comp]
        labels = ['Within Primary\nAuditory Area', 'Between Regions']

        bp1 = ax1.boxplot(box_data, labels=labels, patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightcoral')
        bp1['boxes'][1].set_facecolor('lightblue')

        ax1.set_ylabel('Number of Significant Components', fontsize=12)
        ax1.set_title('Significant Components Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add statistical test
        from scipy.stats import mannwhitneyu
        if len(within_sig_comp) > 0 and len(between_sig_comp) > 0:
            stat, p_val = mannwhitneyu(within_sig_comp, between_sig_comp, alternative='two-sided')
            ax1.text(0.5, 0.95, f'Mann-Whitney U p = {p_val:.4f}',
                     transform=ax1.transAxes, ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: D-component correlations comparison
        ax2 = axes[0, 1]

        # Filter for cases with significant components
        within_d_corr = within_region_df[within_region_df['significant_components'] > 0]['mean_correlation_d']
        between_d_corr = between_region_df[between_region_df['significant_components'] > 0]['mean_correlation_d']

        if len(within_d_corr) > 0 and len(between_d_corr) > 0:
            box_data_d = [within_d_corr, between_d_corr]

            bp2 = ax2.boxplot(box_data_d, labels=labels, patch_artist=True)
            bp2['boxes'][0].set_facecolor('lightcoral')
            bp2['boxes'][1].set_facecolor('lightblue')

            # Statistical test
            stat_d, p_val_d = mannwhitneyu(within_d_corr, between_d_corr, alternative='two-sided')
            ax2.text(0.5, 0.95, f'Mann-Whitney U p = {p_val_d:.4f}',
                     transform=ax2.transAxes, ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_ylabel('Mean Correlation (d components)', fontsize=12)
        ax2.set_title('D-Component Correlations Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distribution by stimulus type
        ax3 = axes[0, 2]

        if len(within_region_df) > 0:
            within_by_stim = within_region_df.groupby('stimulus')['mean_correlation_d'].mean()
            between_by_stim = between_region_df.groupby('stimulus')['mean_correlation_d'].mean()

            x_pos = np.arange(len(within_by_stim))
            width = 0.35

            bars1 = ax3.bar(x_pos - width / 2, within_by_stim.values, width,
                            label='Within Primary Auditory', color='lightcoral', alpha=0.7)
            bars2 = ax3.bar(x_pos + width / 2, between_by_stim.values, width,
                            label='Between Regions', color='lightblue', alpha=0.7)

            ax3.set_xlabel('Stimulus Type', fontsize=12)
            ax3.set_ylabel('Mean Correlation (d components)', fontsize=12)
            ax3.set_title('Correlations by Stimulus Type', fontsize=12, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(within_by_stim.index)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        # Plot 4: Within-region iteration variability
        ax4 = axes[1, 0]

        if 'iteration' in within_region_df.columns:
            # Group by session and response range to show iteration variability
            within_grouped = within_region_df.groupby(['session', 'response_range', 'stimulus'])

            variability_data = []
            for name, group in within_grouped:
                if len(group) > 1:  # Multiple iterations
                    std_val = group['mean_correlation_d'].std()
                    mean_val = group['mean_correlation_d'].mean()
                    variability_data.append({'group': f"{name[0]}_{name[1]}_{name[2]}",
                                             'std': std_val, 'mean': mean_val})

            if variability_data:
                var_df = pd.DataFrame(variability_data)
                ax4.scatter(var_df['mean'], var_df['std'], alpha=0.6, s=50)
                ax4.set_xlabel('Mean Correlation (d components)', fontsize=12)
                ax4.set_ylabel('Std Deviation Across Iterations', fontsize=12)
                ax4.set_title('Within-Region Iteration Variability', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)

        # Plot 5: Summary statistics table
        ax5 = axes[1, 1]
        ax5.axis('off')

        summary_stats = []

        # Within-region stats
        within_stats = [
            'Within Primary Auditory',
            f"{within_sig_comp.mean():.2f} ± {within_sig_comp.std():.2f}",
            f"{len(within_sig_comp)}",
            f"{within_d_corr.mean():.3f} ± {within_d_corr.std():.3f}" if len(within_d_corr) > 0 else "N/A",
            f"{len(within_d_corr)}" if len(within_d_corr) > 0 else "0"
        ]
        summary_stats.append(within_stats)

        # Between-region stats
        between_stats = [
            'Between Regions',
            f"{between_sig_comp.mean():.2f} ± {between_sig_comp.std():.2f}",
            f"{len(between_sig_comp)}",
            f"{between_d_corr.mean():.3f} ± {between_d_corr.std():.3f}" if len(between_d_corr) > 0 else "N/A",
            f"{len(between_d_corr)}" if len(between_d_corr) > 0 else "0"
        ]
        summary_stats.append(between_stats)

        table = ax5.table(cellText=summary_stats,
                          colLabels=['Analysis Type', 'Sig. Components', 'N (Sig)', 'D-Corr Mean±SD', 'N (D-Corr)'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold')

        # Plot 6: Response range comparison
        ax6 = axes[1, 2]

        if len(within_region_df) > 0 and len(between_region_df) > 0:
            response_ranges = sorted(within_region_df['response_range'].unique())
            within_by_range = within_region_df.groupby('response_range')['mean_correlation_d'].mean()
            between_by_range = between_region_df.groupby('response_range')['mean_correlation_d'].mean()

            x_pos = np.arange(len(response_ranges))
            width = 0.35

            ax6.bar(x_pos - width / 2, [within_by_range.get(r, 0) for r in response_ranges],
                    width, label='Within Primary Auditory', color='lightcoral', alpha=0.7)
            ax6.bar(x_pos + width / 2, [between_by_range.get(r, 0) for r in response_ranges],
                    width, label='Between Regions', color='lightblue', alpha=0.7)

            ax6.set_xlabel('Response Range', fontsize=12)
            ax6.set_ylabel('Mean Correlation (d components)', fontsize=12)
            ax6.set_title('Correlations by Response Range', fontsize=12, fontweight='bold')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(response_ranges)
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "within_vs_between_region_comparison.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

        return fig


if __name__ == "__main__":
    # Initialize analysis
    analyzer = TwoRegionCCAAnalysis(neuron_threshold=40, n_splits=5, random_state=42,
                                   n_permutations=10000)  # TODO: Will want to do with n_permutations=10000 at some point

    # Example: analyze specific region pairs
    # Uncomment and modify these lines to analyze specific pairs:
    # region_pairs = [('AudP', 'AudV'), ('AudP', 'TeA')]
    # results_df = analyzer.run_analysis(region_pairs)

    # Or analyze all possible pairs:
    results_df = analyzer.run_analysis(n_iterations=5, verbose=False)

    print("\nAnalysis complete! Results saved to:", analyzer.output_dir)
    print(f"Analyzed {len(results_df)} conditions across all region pairs")