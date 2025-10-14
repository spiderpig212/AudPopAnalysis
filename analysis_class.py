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
