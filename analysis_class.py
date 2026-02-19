"""
This will contain a new class for processing the npz files. The idea is it can load in and return the arrays indexed in
common ways as a lot of my files are starting with the same 30 lines. This should also just make it easier for others to
work with the data in the future as it will be much cleaner and hopefully clearer what is being done. I will need to
think on what kind of flexibility I will want to include for the data loading and how the self attributes are defined.
"""

import numpy as np
import os
import pandas as pd
from jaratoolbox import celldatabase
import settings
import studyparams
import funcs
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy import signal
from sklearn.cross_decomposition import CCA
import warnings
from tqdm import tqdm


class AnalysisBase:
    """Base class for neural data analysis with common initialization and utilities."""

    def __init__(self, study_name=None, db_suffix='coords'):
        """
        Initialize analysis with common paths and database loading.

        Args:
            study_name: Name of the study (defaults to studyparams.STUDY_NAME)
            db_suffix: Suffix for database filename (e.g., 'coords', 'responsive_all_stims_index_new'). Assumes the
                database is named 'celldb_{study_name}_{db_suffix}.h5' in the database path.
        """
        self.study_name = study_name or studyparams.STUDY_NAME
        self.db_suffix = db_suffix

        # Initialize paths
        self._setup_paths()

        # Load database
        self.celldb = self._load_database()

        # Initialize general information for analysis
        self.areas_of_interest = [
            "Dorsal auditory area",
            "Posterior auditory area",
            "Primary auditory area",
            "Ventral auditory area"
        ]

        nCategories = len(studyparams.SOUND_CATEGORIES)
        soundCats = studyparams.SOUND_CATEGORIES
        nInstances = 4
        stimVals = np.empty(nInstances * nCategories, dtype=object)

        for i in range(nCategories):
            for j in range(nInstances):
                stimVals[i * nInstances + j] = soundCats[i] + f"_{j + 1}"

        self.stim_info = {
            'naturalSound': {
            'stim_type': 'naturalSound',
            'stim_var': 'soundID',
            'time_range': [-2, 6],
            'all_periods': [[-1, 0], [0, 0.5], [1, 4], [4, 4.5]],
            'nTrials': 200,
            'nCategories': nCategories,
            'soundCats': soundCats,
            'nInstances': nInstances,
            'stimVals': stimVals
            },

            'pureTones': {
                'stim_type': 'pureTones',
                'stim_var': 'currentFreq',
                'time_range': [-0.1, 0.3],
                'all_periods': [[-0.1, 0], [0, 0.05], [0.05, 0.1], [0.1, 0.15]],
                'nTrials': 320,
                'nCategories': 16,
                },

            'AM': {
                'stim_type': 'AM',
                'stim_var': 'currentFreq',
                'time_range': [-0.5, 1.5],
                'all_periods': [[-0.5, 0], [0, 0.2], [0.2, 0.5], [0.5, 0.7]],
                'nTrials': 220,
                'nCategories': 11,
            }
        }

        self.response_regions = ["baseline", "onset", "sustained", "offset"]
        self.stim_types = self.stim_info.keys()
        # Add simplified site names
        self._process_site_names()

    def _setup_paths(self):
        """Setup common file paths."""
        self.db_path = os.path.join(settings.DATABASE_PATH, self.study_name)
        self.figdata_path = os.path.join(settings.FIGURES_DATA_PATH, self.study_name)
        os.makedirs(self.figdata_path, exist_ok=True)

    def _load_database(self):
        """Load the cell database."""
        db_filename = os.path.join(self.db_path, f'celldb_{self.study_name}_{self.db_suffix}.h5')
        return celldatabase.load_hdf(db_filename)

    def _process_site_names(self):
        """Add simplified site names to the database."""
        # TODO: Monitor this to change away from "_updated" once we roll out the actual correction everywhere
        simple_site_names = self.celldb['recordingSiteName_updated'].str.split(',').apply(lambda x: x[0])
        simple_site_names.name = 'simpleSiteName'
        self.celldb = pd.concat([self.celldb, simple_site_names], axis=1)

    def get_subset_by_areas(self, areas=None):
        """Get database subset filtered by brain areas."""
        areas = areas or self.areas_of_interest
        return self.celldb[self.celldb['simpleSiteName'].isin(areas)].reset_index()

    def save_arrays(self, filename, **arrays):
        """Save numpy arrays to the figures data path."""
        filepath = os.path.join(self.figdata_path, filename)
        np.savez(filepath, **arrays)
        print(f"Saved arrays to {filepath}")
        return filepath

    def load_arrays(self, filename):
        """Load numpy arrays from the figures data path."""
        filepath = os.path.join(self.figdata_path, filename)
        return np.load(filepath, allow_pickle=True)


class FiringRateProcessing(AnalysisBase):
    """Class for array generation."""

    def __init__(self, study_name=None, db_suffix='coords'):
        super().__init__(study_name, db_suffix=db_suffix)
        self.celldb_subset = self.get_subset_by_areas()

    def calculate_stimulus_firing_rates(self, stim_type, stim_var, time_range, all_periods):
        """
        Calculate firing rates for a specific stimulus type.

        Args:
            stim_type: Type of stimulus (e.g., 'naturalSound', 'AM', 'pureTones')
            stim_var: Stimulus variable (e.g., 'soundID', 'currentFreq')
            time_range: Time range for analysis
            all_periods: List of time periods for analysis

        Returns:
            Tuple of firing rate arrays
        """
        print(f"Calculating firing rates for {stim_type}")
        fr_arrays = funcs.calculate_fr_arrays(
            self.celldb_subset, stim_type, stim_var, time_range, all_periods
        )
        return fr_arrays

    def process_and_save_stimulus(self, stim_type, stim_var, time_range, all_periods):
        """Process and save firing rates for a stimulus type."""
        fr_arrays = self.calculate_stimulus_firing_rates(stim_type, stim_var, time_range, all_periods)

        # Save arrays
        filename = f'fr_arrays_{stim_type}.npz'
        self.save_arrays(
            filename,
            basefr=fr_arrays[0],
            onsetfr=fr_arrays[1],
            sustainedfr=fr_arrays[2],
            offsetfr=fr_arrays[3],
            stimArray=fr_arrays[4],
            brainRegionArray=fr_arrays[5],
            mouseIDArray=fr_arrays[6],
            sessionIDArray=fr_arrays[7]
        )

        return fr_arrays

    def process_natural_sounds(self):
        """Process natural sounds with predefined parameters."""
        return self.process_and_save_stimulus(
            stim_type='naturalSound',
            stim_var='soundID',
            time_range=[-2, 6],
            all_periods=[[-1, 0], [0, 0.5], [1, 4], [4, 4.5]]
        )

    def process_am_sounds(self):
        """Process AM sounds with predefined parameters."""
        return self.process_and_save_stimulus(
            stim_type='AM',
            stim_var='currentFreq',
            time_range=[-0.5, 1.5],
            all_periods=[[-0.5, 0], [0, 0.2], [0.2, 0.5], [0.5, 0.7]]
        )

    def process_pure_tones(self):
        """Process pure tones with predefined parameters."""
        return self.process_and_save_stimulus(
            stim_type='pureTones',
            stim_var='currentFreq',
            time_range=[-0.1, 0.3],
            all_periods=[[-0.1, 0], [0, 0.05], [0.05, 0.1], [0.1, 0.15]]
        )

    def process_all_stimuli(self):
        """Process all stimulus types."""
        results = {}
        results['natural_sounds'] = self.process_natural_sounds()
        results['am_sounds'] = self.process_am_sounds()
        results['pure_tones'] = self.process_pure_tones()
        return results


class FiringRateAnalysis(AnalysisBase):
    """Class for firing rate analysis and plotting."""
    def __init__(self, study_name=None, db_suffix='coords'):
        super().__init__(study_name, db_suffix=db_suffix)
        self.naturalSound_arrays = self.load_arrays(f'fr_arrays_naturalSound.npz')
        self.pureTone_arrays = self.load_arrays(f'fr_arrays_pureTones.npz')
        self.AM_arrays = self.load_arrays(f'fr_arrays_AM.npz')

    def return_arrays(self, stim_type):
        """Return firing rate arrays for a specific stimulus type."""
        if stim_type == 'naturalSound':
            return self.naturalSound_arrays
        elif stim_type == 'pureTones':
            return self.pureTone_arrays
        elif stim_type == 'AM':
            return self.AM_arrays
        else:
            raise ValueError(f"Invalid stimulus type: {stim_type}")

    def get_firing_rate_array(self, region, session, stim, response_range, neuron_threshold, random_state):
        npz_array = self.return_arrays(stim)
        brainRegionArray = npz_array['brainRegionArray']
        sessionArray = npz_array['sessionIDArray']
        response_range_array = npz_array[f"{response_range}fr"]
        mask = (brainRegionArray == region) & (sessionArray == session)
        response_range_array = response_range_array[mask, :].T  # Change to now be [trials, neurons]
        np.random.seed(random_state)
        if response_range_array.shape[1] > neuron_threshold:
            neurons = np.random.choice(response_range_array.shape[1],
                                          size=neuron_threshold,
                                          replace=False)
            filtered_frs = response_range_array[:, neurons]

        return filtered_frs


class CCAAnalysis(FiringRateAnalysis):
    """Class for Canonical Correlation Analysis and subspace similarity analysis."""

    def __init__(self, study_name=None, db_suffix='coords_updated', neuron_threshold=20):
        super().__init__(study_name, db_suffix=db_suffix)
        self.neuron_threshold = neuron_threshold
        self.response_ranges = ["onset", "sustained", "offset"]  # baseline excluded for CCA
        self.analysis_types = ["correlation", "mean_corr", "PR"]

        # Create output directories
        self._setup_cca_directories()

    def _setup_cca_directories(self):
        """Create directories for CCA analysis outputs."""
        dirs_to_create = [
            "CCA_plots",
            "CCA_summary_plots",
            "subspace_similarity_plots"
        ]

        for dir_name in dirs_to_create:
            os.makedirs(os.path.join(self.figdata_path, dir_name), exist_ok=True)

    def compute_principal_angles(self, U1, U2):
        """
        Compute principal angles between two subspaces defined by orthonormal matrices U1 and U2.

        Parameters:
        -----------
        U1, U2 : ndarray
            Orthonormal matrices defining the subspaces (columns are basis vectors)
            Both should have the same number of rows (ambient dimension)

        Returns:
        --------
        angles : ndarray
            Principal angles in radians, sorted in ascending order
        similarity_index : float
            Subspace similarity index (mean cosine of principal angles)
        """
        # Ensure inputs are numpy arrays
        U1 = np.array(U1)
        U2 = np.array(U2)

        # Compute the matrix M = U1^T * U2
        M = U1.T @ U2

        # Compute SVD of M
        _, sigma, _ = np.linalg.svd(M, full_matrices=False)

        # Clamp sigma values to [0, 1] to handle numerical errors
        sigma = np.clip(sigma, 0, 1)

        # Principal angles are arccos of singular values
        angles = np.arccos(sigma)

        # Similarity index: mean cosine of principal angles (higher = more similar)
        similarity_index = np.mean(sigma)

        return angles, similarity_index

    def compute_cca_subspace_similarity(self, cca1, cca2, n_components=None):
        """
        Compute subspace similarity between two CCA analyses using principal angles.

        Parameters:
        -----------
        cca1, cca2 : CCA objects
            Fitted CCA objects to compare
        n_components : int, optional
            Number of components to compare (default: all available)

        Returns:
        --------
        results : dict
            Dictionary containing similarity metrics
        """
        if n_components is None:
            n_components = min(cca1.x_weights_.shape[1], cca2.x_weights_.shape[1])

        # Get the canonical weights (these define the subspaces)
        U1_region1 = cca1.x_weights_[:, :n_components]  # Region 1 subspace from CCA 1
        U2_region1 = cca2.x_weights_[:, :n_components]  # Region 1 subspace from CCA 2

        U1_region2 = cca1.y_weights_[:, :n_components]  # Region 2 subspace from CCA 1
        U2_region2 = cca2.y_weights_[:, :n_components]  # Region 2 subspace from CCA 2

        # Compute principal angles for each brain region
        angles_r1, sim_r1 = self.compute_principal_angles(U1_region1, U2_region1)
        angles_r2, sim_r2 = self.compute_principal_angles(U1_region2, U2_region2)

        return {
            'angles_region1': angles_r1,
            'angles_region2': angles_r2,
            'similarity_region1': sim_r1,
            'similarity_region2': sim_r2,
            'mean_similarity': (sim_r1 + sim_r2) / 2,
            'angles_degrees_region1': np.degrees(angles_r1),
            'angles_degrees_region2': np.degrees(angles_r2)
        }

    def run_cca_for_session(self, stim, resp_range, session, region1, region2):
        """
        Helper function to run CCA for a specific session and region pair.
        Returns CCA object and data arrays.
        """
        try:
            # Get arrays for this stimulus type
            stim_arrays = self.return_arrays(stim)
            respArray = stim_arrays[f"{resp_range}fr"]
            sessionArray = stim_arrays["sessionIDArray"]
            brainRegionArray = stim_arrays["brainRegionArray"]

            # Filter for specific session
            session_mask = sessionArray == session
            session_resp_array = respArray[session_mask, :]
            brain_session_array = brainRegionArray[session_mask]

            # Get data for region 1
            region1_mask = brain_session_array == region1
            brain1_resp_array = session_resp_array[region1_mask, :].T

            if brain1_resp_array.shape[1] < self.neuron_threshold:
                return None

            region1_neurons = np.random.choice(brain1_resp_array.shape[1],
                                               size=self.neuron_threshold, replace=False)
            brain1_resp_array = brain1_resp_array[:, region1_neurons]

            # Get data for region 2
            region2_mask = brain_session_array == region2
            brain2_resp_array = session_resp_array[region2_mask, :].T

            if brain2_resp_array.shape[1] < self.neuron_threshold:
                return None

            region2_neurons = np.random.choice(brain2_resp_array.shape[1],
                                               size=self.neuron_threshold, replace=False)
            brain2_resp_array = brain2_resp_array[:, region2_neurons]

            # Run CCA
            n_components = min(brain1_resp_array.shape[1], brain2_resp_array.shape[1])
            cca = CCA(n_components=n_components)
            cca.fit(brain1_resp_array, brain2_resp_array)

            return {
                'cca': cca,
                'X1': brain1_resp_array,
                'X2': brain2_resp_array
            }

        except Exception as e:
            print(f"Error in run_cca_for_session: {str(e)}")
            return None

    def analyze_subspace_consistency(self, stim):
        """
        Analyze consistency of CCA subspaces across sessions for the same region pairs.

        Parameters:
        -----------
        stim : str
            Stimulus type ('naturalSound', 'pureTones', 'AM')

        Returns:
        --------
        subspace_similarity_data : list
            List of dictionaries containing subspace similarity results
        """

        subspace_similarity_data = []

        # Get stimulus arrays and basic info
        stim_arrays = self.return_arrays(stim)
        brainRegionArray = stim_arrays["brainRegionArray"]
        sessionArray = stim_arrays["sessionIDArray"]
        uniqRegions = np.unique(brainRegionArray)
        uniqSessions = np.unique(sessionArray)

        print(f"\n=== Analyzing Subspace Consistency for {stim} ===")
        print(f"Found {len(uniqRegions)} regions and {len(uniqSessions)} sessions")

        # Iterate through all combinations
        for resp_range in self.response_ranges:
            for i, region1 in enumerate(uniqRegions):
                for region2 in uniqRegions[i + 1:]:
                    region_pair = f"{region1}_vs_{region2}"

                    # Find sessions where both regions have sufficient neurons
                    valid_sessions = []
                    for session in uniqSessions:
                        session_mask = sessionArray == session
                        brain_session_array = brainRegionArray[session_mask]

                        region1_count = np.sum(brain_session_array == region1)
                        region2_count = np.sum(brain_session_array == region2)

                        if region1_count >= self.neuron_threshold and region2_count >= self.neuron_threshold:
                            valid_sessions.append(session)

                    if len(valid_sessions) < 2:
                        print(f"Skipping {region_pair}, {resp_range}: insufficient sessions ({len(valid_sessions)})")
                        continue

                    print(f"Processing {region_pair}, {resp_range}: {len(valid_sessions)} valid sessions")

                    # Compare all pairs of sessions
                    for idx1, session1 in enumerate(valid_sessions):
                        for session2 in valid_sessions[idx1 + 1:]:

                            try:
                                # Run CCA for both sessions
                                cca1_data = self.run_cca_for_session(stim, resp_range, session1, region1, region2)
                                cca2_data = self.run_cca_for_session(stim, resp_range, session2, region1, region2)

                                if cca1_data is None or cca2_data is None:
                                    continue

                                # Compute subspace similarity
                                similarity_results = self.compute_cca_subspace_similarity(
                                    cca1_data['cca'], cca2_data['cca']
                                )

                                # Store results
                                subspace_similarity_data.append({
                                    'stimulus': stim,
                                    'response_range': resp_range,
                                    'region_pair': region_pair,
                                    'region1': region1,
                                    'region2': region2,
                                    'session1': session1,
                                    'session2': session2,
                                    'session_pair': f"{session1}_vs_{session2}",
                                    'similarity_region1': similarity_results['similarity_region1'],
                                    'similarity_region2': similarity_results['similarity_region2'],
                                    'mean_similarity': similarity_results['mean_similarity'],
                                    'angles_region1': similarity_results['angles_degrees_region1'],
                                    'angles_region2': similarity_results['angles_degrees_region2']
                                })

                            except Exception as e:
                                print(f"Error comparing sessions {session1} and {session2}: {str(e)}")
                                continue

        return subspace_similarity_data

    def plot_subspace_similarity_results(self, subspace_data, stim):
        """
        Create plots to visualize subspace similarity results.
        """
        if not subspace_data:
            print("No subspace similarity data to plot")
            return

        df = pd.DataFrame(subspace_data)

        # Plot 1: Distribution of similarity indices
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(df['similarity_region1'], bins=20, alpha=0.7, label='Region 1')
        plt.hist(df['similarity_region2'], bins=20, alpha=0.7, label='Region 2')
        plt.xlabel('Subspace Similarity Index')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Subspace Similarity - {stim}')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.scatter(df['similarity_region1'], df['similarity_region2'], alpha=0.6)
        plt.xlabel(f'Region 1 Similarity')
        plt.ylabel(f'Region 2 Similarity')
        plt.title('Region 1 vs Region 2 Similarity')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)

        plt.subplot(1, 3, 3)
        import seaborn as sns
        sns.boxplot(data=df, x='region_pair', y='mean_similarity')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Mean Subspace Similarity')
        plt.title('Similarity by Region Pair')

        plt.tight_layout()
        plt.savefig(f"{self.figdata_path}/subspace_similarity_plots/similarity_overview_{stim}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Plot 2: Similarity by response range
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='region_pair', y='mean_similarity', hue='response_range')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Mean Subspace Similarity')
        plt.title(f'Subspace Similarity by Region Pair and Response Range - {stim}')
        plt.legend(title='Response Range')
        plt.tight_layout()
        plt.savefig(f"{self.figdata_path}/subspace_similarity_plots/similarity_by_response_{stim}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Summary statistics
        print("\n=== Subspace Similarity Summary ===")
        summary = df.groupby(['region_pair', 'response_range'])['mean_similarity'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(summary)

        return df

    def run_full_subspace_analysis(self, stimuli=None):
        """
        Run complete subspace similarity analysis for specified stimuli.

        Parameters:
        -----------
        stimuli : list, optional
            List of stimulus types to analyze (default: all available)
        """
        if stimuli is None:
            stimuli = list(self.stim_types)

        all_results = {}

        for stim in stimuli:
            print(f"\n{'=' * 50}")
            print(f"Processing stimulus: {stim}")
            print(f"{'=' * 50}")

            # Run subspace consistency analysis
            subspace_data = self.analyze_subspace_consistency(stim)

            if subspace_data:
                # Save results
                df_subspace = pd.DataFrame(subspace_data)
                df_subspace.to_feather(f"{self.figdata_path}/CCA_subspace_similarity_{stim}.feather")
                df_subspace.to_csv(f"{self.figdata_path}/CCA_subspace_similarity_{stim}.csv", index=False)

                # Create plots
                df_plotted = self.plot_subspace_similarity_results(subspace_data, stim)
                all_results[stim] = df_plotted

                print(f"\nSubspace analysis complete for {stim}.")
                print(f"Found {len(subspace_data)} session pairs with subspace similarity data.")
            else:
                print(f"No subspace similarity data generated for {stim}")
                all_results[stim] = None

        return all_results


"""
Temporal Dynamics Analysis for Information Flow Between Brain Regions
This module implements various temporal analysis methods to understand how
information flows between brain regions over time.
"""
warnings.filterwarnings('ignore')

class TemporalDynamicsAnalysis:
    """
    Class for analyzing temporal dynamics of information flow between brain regions.
    """

    def __init__(self, fr_db=None, neuron_threshold=20):
        """
        Initialize the temporal dynamics analysis.

        Parameters:
        -----------
        fr_db : FiringRateAnalysis object
            Database containing firing rate data
        neuron_threshold : int
            Minimum number of neurons required per region
        """
        if fr_db is None:
            self.fr_db = FiringRateAnalysis(db_suffix="coords_updated")
        else:
            self.fr_db = fr_db

        self.neuron_threshold = neuron_threshold
        self.file_path = self.fr_db.figdata_path

        # Create output directories
        self.temporal_plots_dir = f"{self.file_path}/temporal_dynamics_plots"
        os.makedirs(self.temporal_plots_dir, exist_ok=True)

    def cross_correlation_analysis(self, region1_data, region2_data, max_lag=50,
                                   region1_name="Region1", region2_name="Region2"):
        """
        Compute cross-correlation between two brain regions with time lags.

        Parameters:
        -----------
        region1_data : np.array
            Neural activity data for region 1 (trials x neurons)
        region2_data : np.array
            Neural activity data for region 2 (trials x neurons)
        max_lag : int
            Maximum time lag to consider
        region1_name, region2_name : str
            Names of the brain regions

        Returns:
        --------
        dict : Results including cross-correlation values and optimal lag
        """
        # Average across neurons to get population activity
        pop1 = np.mean(region1_data, axis=1)
        pop2 = np.mean(region2_data, axis=1)

        # Compute cross-correlation
        cross_corr = signal.correlate(pop1, pop2, mode='full')
        lags = signal.correlation_lags(len(pop1), len(pop2), mode='full')

        # Limit to desired lag range
        valid_indices = (lags >= -max_lag) & (lags <= max_lag)
        cross_corr = cross_corr[valid_indices]
        lags = lags[valid_indices]

        # Find optimal lag
        max_corr_idx = np.argmax(np.abs(cross_corr))
        optimal_lag = lags[max_corr_idx]
        max_correlation = cross_corr[max_corr_idx]

        return {
            'cross_correlation': cross_corr,
            'lags': lags,
            'optimal_lag': optimal_lag,
            'max_correlation': max_correlation,
            'region1_name': region1_name,
            'region2_name': region2_name
        }

    def sliding_window_cca(self, region1_data, region2_data, window_size=20,
                           step_size=5, n_components=5):
        """
        Perform CCA in sliding time windows to track temporal dynamics.

        Parameters:
        -----------
        region1_data : np.array
            Neural activity data for region 1 (trials x neurons)
        region2_data : np.array
            Neural activity data for region 2 (trials x neurons)
        window_size : int
            Size of sliding window
        step_size : int
            Step size for sliding window
        n_components : int
            Number of CCA components to compute

        Returns:
        --------
        dict : Results including correlations over time
        """
        n_trials = region1_data.shape[0]
        n_windows = (n_trials - window_size) // step_size + 1

        window_centers = []
        correlations = []
        all_components_corr = []

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            window_data1 = region1_data[start_idx:end_idx, :]
            window_data2 = region2_data[start_idx:end_idx, :]

            # Skip if window is too small
            if window_data1.shape[0] < n_components:
                continue

            try:
                # Fit CCA
                cca = CCA(n_components=min(n_components, window_data1.shape[1],
                                           window_data2.shape[1]))
                X_c, Y_c = cca.fit_transform(window_data1, window_data2)

                # Calculate correlation for first component
                corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]

                # Calculate correlations for all components
                component_corrs = []
                for comp in range(X_c.shape[1]):
                    comp_corr = np.corrcoef(X_c[:, comp], Y_c[:, comp])[0, 1]
                    component_corrs.append(comp_corr)

                window_centers.append(start_idx + window_size // 2)
                correlations.append(corr)
                all_components_corr.append(component_corrs)

            except Exception as e:
                print(f"CCA failed for window {i}: {e}")
                continue

        return {
            'window_centers': np.array(window_centers),
            'correlations': np.array(correlations),
            'all_components_corr': np.array(all_components_corr),
            'window_size': window_size,
            'step_size': step_size
        }

    def information_transfer_analysis(self, region1_data, region2_data, lag_range=10):
        """
        Analyze information transfer directionality between regions.
        Uses lagged correlations to infer directionality.

        Parameters:
        -----------
        region1_data : np.array
            Neural activity data for region 1 (trials x neurons)
        region2_data : np.array
            Neural activity data for region 2 (trials x neurons)
        lag_range : int
            Range of lags to test for directionality

        Returns:
        --------
        dict : Results including transfer measures in both directions
        """
        pop1 = np.mean(region1_data, axis=1)
        pop2 = np.mean(region2_data, axis=1)

        # Forward direction: Region1 -> Region2
        forward_correlations = []
        # Backward direction: Region2 -> Region1
        backward_correlations = []

        for lag in range(1, lag_range + 1):
            # Forward: correlate region1(t) with region2(t+lag)
            if lag < len(pop1):
                forward_corr, _ = pearsonr(pop1[:-lag], pop2[lag:])
                forward_correlations.append(forward_corr)

            # Backward: correlate region2(t) with region1(t+lag)
            if lag < len(pop2):
                backward_corr, _ = pearsonr(pop2[:-lag], pop1[lag:])
                backward_correlations.append(backward_corr)

        # Calculate average transfer strength
        forward_transfer = np.mean(np.abs(forward_correlations))
        backward_transfer = np.mean(np.abs(backward_correlations))

        # Directionality index: positive means region1 -> region2
        directionality = forward_transfer - backward_transfer

        return {
            'forward_correlations': forward_correlations,
            'backward_correlations': backward_correlations,
            'forward_transfer': forward_transfer,
            'backward_transfer': backward_transfer,
            'directionality': directionality,
            'lags': list(range(1, len(forward_correlations) + 1))
        }

    def run_temporal_analysis(self, stim_type="pureTones", response_ranges=None):
        """
        Run comprehensive temporal dynamics analysis.

        Parameters:
        -----------
        stim_type : str
            Type of stimulus to analyze
        response_ranges : list
            List of response ranges to analyze

        Returns:
        --------
        dict : Comprehensive results from all temporal analyses
        """
        if response_ranges is None:
            response_ranges = ["onset", "sustained", "offset"]

        print(f"Starting temporal dynamics analysis for {stim_type}")

        # Get data arrays
        stim_arrays = self.fr_db.return_arrays(stim_type)
        brainRegionArray = stim_arrays["brainRegionArray"]
        sessionArray = stim_arrays["sessionIDArray"]
        uniqRegions = np.unique(brainRegionArray)
        uniqSessions = np.unique(sessionArray)

        temporal_results = []

        for session in tqdm(uniqSessions, desc="Processing sessions"):
            session_mask = sessionArray == session
            session_brain_array = brainRegionArray[session_mask]

            for respRange in response_ranges:
                respArray = stim_arrays[f"{respRange}fr"]
                session_resp_array = respArray[session_mask, :]

                for i, region1 in enumerate(uniqRegions):
                    region1_mask = session_brain_array == region1
                    region1_data = session_resp_array[region1_mask, :].T

                    if region1_data.shape[1] < self.neuron_threshold:
                        continue

                    # Subsample neurons
                    region1_neurons = np.random.choice(region1_data.shape[1],
                                                       size=self.neuron_threshold,
                                                       replace=False)
                    region1_data = region1_data[:, region1_neurons]

                    for region2 in uniqRegions[i + 1:]:
                        region2_mask = session_brain_array == region2
                        region2_data = session_resp_array[region2_mask, :].T

                        if region2_data.shape[1] < self.neuron_threshold:
                            continue

                        # Subsample neurons
                        region2_neurons = np.random.choice(region2_data.shape[1],
                                                           size=self.neuron_threshold,
                                                           replace=False)
                        region2_data = region2_data[:, region2_neurons]

                        # Cross-correlation analysis
                        xcorr_results = self.cross_correlation_analysis(
                            region1_data, region2_data, region1, region2)

                        # Sliding window CCA
                        sliding_cca_results = self.sliding_window_cca(
                            region1_data, region2_data)

                        # Information transfer analysis
                        transfer_results = self.information_transfer_analysis(
                            region1_data, region2_data)

                        # Store results
                        result_entry = {
                            'session': session,
                            'region1': region1,
                            'region2': region2,
                            'region_pair': f"{region1}_vs_{region2}",
                            'response_range': respRange,
                            'stimulus': stim_type,
                            'xcorr_optimal_lag': xcorr_results['optimal_lag'],
                            'xcorr_max_correlation': xcorr_results['max_correlation'],
                            'sliding_cca_mean_corr': np.mean(sliding_cca_results['correlations']),
                            'sliding_cca_std_corr': np.std(sliding_cca_results['correlations']),
                            'forward_transfer': transfer_results['forward_transfer'],
                            'backward_transfer': transfer_results['backward_transfer'],
                            'directionality': transfer_results['directionality'],
                            'xcorr_data': xcorr_results,
                            'sliding_cca_data': sliding_cca_results,
                            'transfer_data': transfer_results
                        }

                        temporal_results.append(result_entry)

        # Convert to DataFrame
        df_temporal = pd.DataFrame([{k: v for k, v in result.items()
                                     if k not in ['xcorr_data', 'sliding_cca_data', 'transfer_data']}
                                    for result in temporal_results])

        # Save results
        df_temporal.to_csv(f"{self.file_path}/temporal_dynamics_{stim_type}.csv", index=False)
        df_temporal.to_feather(f"{self.file_path}/temporal_dynamics_{stim_type}.feather")

        return temporal_results, df_temporal

    def plot_temporal_results(self, temporal_results, df_temporal, stim_type="pureTones"):
        """
        Create comprehensive plots of temporal dynamics results.
        """
        print(f"Creating temporal dynamics plots for {stim_type}")

        # 1. Cross-correlation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot optimal lag distribution
        axes[0, 0].hist(df_temporal['xcorr_optimal_lag'], bins=20, alpha=0.7)
        axes[0, 0].set_xlabel('Optimal Lag (trials)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Optimal Cross-Correlation Lags')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)

        # Plot max correlation distribution
        axes[0, 1].hist(df_temporal['xcorr_max_correlation'], bins=20, alpha=0.7)
        axes[0, 1].set_xlabel('Max Cross-Correlation')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Max Cross-Correlations')

        # Plot directionality
        axes[1, 0].hist(df_temporal['directionality'], bins=20, alpha=0.7)
        axes[1, 0].set_xlabel('Directionality Index')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Information Transfer Directionality')
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)

        # Plot sliding CCA variability
        axes[1, 1].scatter(df_temporal['sliding_cca_mean_corr'],
                           df_temporal['sliding_cca_std_corr'], alpha=0.6)
        axes[1, 1].set_xlabel('Mean Sliding CCA Correlation')
        axes[1, 1].set_ylabel('Std Sliding CCA Correlation')
        axes[1, 1].set_title('Temporal Stability of CCA')

        plt.tight_layout()
        plt.savefig(f"{self.temporal_plots_dir}/temporal_overview_{stim_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Region pair comparisons
        unique_pairs = df_temporal['region_pair'].unique()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Optimal lag by region pair
        sns.boxplot(data=df_temporal, x='region_pair', y='xcorr_optimal_lag', ax=axes[0, 0])
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
        axes[0, 0].set_title('Optimal Lag by Region Pair')
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.7)

        # Directionality by region pair
        sns.boxplot(data=df_temporal, x='region_pair', y='directionality', ax=axes[0, 1])
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
        axes[0, 1].set_title('Directionality by Region Pair')
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)

        # Temporal stability by region pair
        sns.boxplot(data=df_temporal, x='region_pair', y='sliding_cca_std_corr', ax=axes[1, 0])
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        axes[1, 0].set_title('Temporal Variability by Region Pair')

        # Mean correlation by region pair
        sns.boxplot(data=df_temporal, x='region_pair', y='sliding_cca_mean_corr', ax=axes[1, 1])
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        axes[1, 1].set_title('Mean Temporal Correlation by Region Pair')

        plt.tight_layout()
        plt.savefig(f"{self.temporal_plots_dir}/region_pair_comparison_{stim_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Individual cross-correlation examples
        n_examples = min(6, len(temporal_results))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, result in enumerate(temporal_results[:n_examples]):
            xcorr_data = result['xcorr_data']
            axes[i].plot(xcorr_data['lags'], xcorr_data['cross_correlation'])
            axes[i].axvline(xcorr_data['optimal_lag'], color='red', linestyle='--')
            axes[i].set_xlabel('Lag (trials)')
            axes[i].set_ylabel('Cross-correlation')
            axes[i].set_title(f"{result['region_pair']}\n{result['response_range']}")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.temporal_plots_dir}/xcorr_examples_{stim_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Sliding window examples
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, result in enumerate(temporal_results[:n_examples]):
            sliding_data = result['sliding_cca_data']
            axes[i].plot(sliding_data['window_centers'], sliding_data['correlations'], 'o-')
            axes[i].set_xlabel('Trial (window center)')
            axes[i].set_ylabel('CCA Correlation')
            axes[i].set_title(f"{result['region_pair']}\n{result['response_range']}")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.temporal_plots_dir}/sliding_cca_examples_{stim_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Temporal dynamics plots saved to {self.temporal_plots_dir}")
