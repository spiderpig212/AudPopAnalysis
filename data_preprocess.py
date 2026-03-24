import sys
import os

# Add the 2022paspeech directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import funcs
import params
import numpy as np
from copy import deepcopy
from jaratoolbox import celldatabase, ephyscore

# Initialize data dictionary for session-level storage
data = {}
for sound_type in ['speech', 'AM', 'PT']:
    data[sound_type] = {}
    for window_name in ['onset', 'sustained', 'offset']:
        data[sound_type][window_name] = {
            'X': [],
            'Y_brain': [],
            'Y_freq': [],
            'mouseID': [],
            'sessionID': []
        }

# Initialize dictionaries for population-level arrays
X_all_by_sound = {
    'speech': {'onset': [], 'sustained': [], 'offset': []},
    'AM': {'onset': [], 'sustained': [], 'offset': []},
    'PT': {'onset': [], 'sustained': [], 'offset': []}
}

Y_brain_all_by_sound = {
    'speech': [],
    'AM': [],
    'PT': []
}

Y_freq_by_sound = {
    'speech': None,
    'AM': None,
    'PT': None
}

indices_by_sound = {
    'speech': None,
    'AM': None,
    'PT': None
}

previous_freq_by_sound = {
    'speech': None,
    'AM': None,
    'PT': None
}


# %% Initialize plot and subset dataframe
def load_data(subject, date, targetSiteName, sound_type_load):
    fullDb = celldatabase.load_hdf(params.fullPath)
    simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
    # simpleSiteNames = simpleSiteNames.replace("Posterior auditory area", "Dorsal auditory area")

    fullDb["recordingSiteName"] = simpleSiteNames
    celldb = fullDb[(fullDb.subject == subject)]
    celldbSubset = celldb[(celldb.date == date)]
    celldbSubset = celldbSubset[(celldbSubset.recordingSiteName == targetSiteName)]

    if celldbSubset.empty:
        print(f"No data in {targetSiteName} on {date} for Speech, AM, and PT.")
        return None, None, None

    ensemble = ephyscore.CellEnsemble(celldbSubset)
    try:
        ephysData, bdata = ensemble.load(sound_type_load)
    except IndexError:
        print(f"No sound data for {targetSiteName} on {date} for {subject}")
        return None, None, None
    return ensemble, ephysData, bdata


def calculate_firing_rate(sound_label, window, ensemble, bdata):
    """Calculate firing rate for a specific sound type and time window.

    Assumes ensemble.eventlocked_spiketimes() has already been called for this session.
    Uses the window edges as a single bin to count spikes within the exact window.
    """
    binEdges = [window[0], window[1]]  # Two edges = one bin covering exactly this window
    spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)
    sumEvokedFR = spikeCounts.sum(axis=2)  # (nCells, nTrials)
    spikesPerSecEvoked = sumEvokedFR / (window[1] - window[0])

    if sound_label == "speech":
        FTParamsEachTrial = bdata['targetFTpercent']
        VOTParamsEachTrial = bdata['targetVOTpercent']
        nTrials = len(bdata['targetFTpercent'])
        Y_frequency = np.array([(FTParamsEachTrial[i], VOTParamsEachTrial[i]) for i in range(nTrials)])
    else:  # AM or PT
        nTrials = len(bdata['currentFreq'])
        Y_frequency = np.array(bdata['currentFreq'])

    return spikesPerSecEvoked, Y_frequency


def normalize_firing_rate(spikesPerSecEvoked, targetSiteName):
    """Normalize firing rates and subsample neurons if needed

    Returns:
        spikeRateNormalized: (neurons × trials)
        Y_brain_area_array: list of brain area labels (one per neuron)
    """
    # Normalize by trial means
    trialMeans = spikesPerSecEvoked.mean(axis=1)
    spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans
    spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T

    # Subsample if needed
    if spikesPerSecEvokedNormalized.shape[1] > params.leastCellsArea:
        subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1], params.leastCellsArea, replace=False)
        spikeRateNormalized = spikesPerSecEvokedNormalized[:, subsetIndex]
    else:
        spikeRateNormalized = spikesPerSecEvokedNormalized

    Y_brain_area_array = [targetSiteName] * spikeRateNormalized.shape[0]

    return spikeRateNormalized, Y_brain_area_array


def clean_data(subject, date, targetSiteName, sound_type, X, Y_freq, previous_freq):
    """Clean and filter data, ensuring consistent trial counts and frequency ordering

    Args:
        X: (neurons × trials) array
        Y_freq: array of frequency labels (wrapped in list from normalize step)

    Returns:
        X_clean: (neurons × trials) array, sorted by frequency
        Y_freq_clean: sorted frequency array
        previous_freq: updated previous frequency for consistency checking
        indices: sorting indices used
    """

    if sound_type == "speech":
        trials_per_freq = 500 // len(params.unique_labels)  # 41 for 12 unique labels
        total_trials = trials_per_freq * len(params.unique_labels)  # 492

        # Initialize valid indices to keep only the trials matching the minimum occurrences
        valid_indices = []
        freq_kept_counts = {tuple(freq): 0 for freq in params.unique_labels}

        # Y_freq comes in as a list with one element, unwrap it
        Y_freq_array = Y_freq[0] if isinstance(Y_freq, list) else Y_freq

        # Filter the trials for each frequency, keeping up to trials_per_freq per combo
        for i, freq in enumerate(Y_freq_array):
            freq_tuple = tuple(freq)
            if freq_kept_counts[freq_tuple] < trials_per_freq:
                valid_indices.append(i)
                freq_kept_counts[freq_tuple] += 1

        # Skip session if any frequency combo didn't have enough trials
        if any(count < trials_per_freq for count in freq_kept_counts.values()):
            print(f'Not enough speech trials for subject {subject}, on {date} in brain area {targetSiteName}')
            return None

        # Transpose to (trials × neurons), filter trials, transpose back to (neurons × trials)
        X_transposed = X.T
        X_filtered = X_transposed[valid_indices[:total_trials]]
        X_filtered = X_filtered.T
        Y_freq_filtered = Y_freq_array[valid_indices[:total_trials]]

        if len(X_filtered) == 0:
            return None

        # Sort by frequency using lexsort (second element, then first element)
        indices_speech = np.lexsort((Y_freq_filtered[:, 1], Y_freq_filtered[:, 0]))
        Y_freq_sorted = Y_freq_filtered[indices_speech]

        # Sort X by applying indices to columns (trial axis)
        X_sorted = X_filtered[:, indices_speech]

        # Check frequency consistency
        if previous_freq is not None:
            assert np.array_equal(Y_freq_sorted, previous_freq), (
                f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}\n"
                f"Previous: {previous_freq}\nCurrent: {Y_freq_sorted}")

        previous_freq = deepcopy(Y_freq_sorted)

        return X_sorted, Y_freq_sorted, previous_freq, indices_speech

    else:  # AM or PT
        # Y_freq comes in as a list with one element, unwrap it
        Y_freq_array = Y_freq[0] if isinstance(Y_freq, list) else Y_freq

        # Check if we have enough trials
        if X.shape[1] < params.max_trials[sound_type]:
            print(f"Not enough {sound_type} trials for subject {subject}, on {date} in brain area {targetSiteName}.")
            return None

        # Truncate to max_trials
        X_truncated = X[:, :params.max_trials[sound_type]]
        Y_freq_truncated = Y_freq_array[:params.max_trials[sound_type]]

        # Sort by frequency
        sorted_indices = np.argsort(Y_freq_truncated)
        Y_freq_sorted = Y_freq_truncated[sorted_indices]

        # Sort X by applying indices to columns (trial axis)
        X_sorted = X_truncated[:, sorted_indices]

        # Check frequency consistency
        if previous_freq is not None:
            assert np.array_equal(Y_freq_sorted, previous_freq), (
                f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")

        previous_freq = deepcopy(Y_freq_sorted)

        return X_sorted, Y_freq_sorted, previous_freq, sorted_indices


def sort_x_arrays(X_list, indices, sound_type):
    """Sort a list of (neurons × trials) arrays by trial order

    Args:
        X_list: list of (neurons × trials) arrays
        indices: indices to sort trials
        sound_type: 'speech', 'AM', or 'PT'

    Returns:
        sorted_x_list: list of (neurons × trials) arrays with trials sorted
    """
    sorted_x_list = []
    for x in X_list:
        # x is (neurons × trials), sort along axis 1 (trials)
        sorted_x = x[:, indices]
        sorted_x_list.append(sorted_x)
    return sorted_x_list


def save_data():
    """Save population data for each sound type and time window"""
    # Save each sound type separately
    for sound_label in ['speech', 'AM', 'PT']:
        # Collect data for all windows
        onset_X = []
        sustained_X = []
        offset_X = []
        brain_regions = []
        stim_array = []
        mouseIDs = []
        sessionIDs = []

        for window_name in ['onset', 'sustained', 'offset']:
            if len(data[sound_label][window_name]['X']) == 0:
                print(f"No data for {sound_label} - {window_name}, skipping...")
                continue

            # Concatenate all sessions for this window (stack neurons)
            X_array = np.concatenate(data[sound_label][window_name]['X'], axis=0)

            if window_name == 'onset':
                onset_X = X_array
                # Get metadata from onset (same for all windows)
                brain_regions = np.array(data[sound_label][window_name]['Y_brain'])
                stim_array = np.concatenate(data[sound_label][window_name]['Y_freq'], axis=0) if len(
                    data[sound_label][window_name]['Y_freq']) > 0 else np.array([])
                mouseIDs = np.array(data[sound_label][window_name]['mouseID'])
                sessionIDs = np.array(data[sound_label][window_name]['sessionID'])
            elif window_name == 'sustained':
                sustained_X = X_array
            elif window_name == 'offset':
                offset_X = X_array

        # Check if we have data
        if len(onset_X) == 0 and len(sustained_X) == 0 and len(offset_X) == 0:
            print(f"No data for {sound_label}, skipping...")
            continue

        # Save to file
        fr_arrays_filename = os.path.join(params.dbSavePath, f'fr_arrays_{sound_label}.npz')
        print(f"Saving {sound_label} data to {fr_arrays_filename}")
        print(f"  Onset shape: {onset_X.shape if len(onset_X) > 0 else 'N/A'}")
        print(f"  Sustained shape: {sustained_X.shape if len(sustained_X) > 0 else 'N/A'}")
        print(f"  Offset shape: {offset_X.shape if len(offset_X) > 0 else 'N/A'}")
        print(f"  Brain regions: {brain_regions.shape}")
        print(f"  Stim: {stim_array.shape}")
        print(f"  Unique mice: {np.unique(mouseIDs)}")
        print(f"  Unique sessions: {len(np.unique(sessionIDs))}")

        np.savez(fr_arrays_filename,
                 onsetfr=onset_X,
                 sustainedfr=sustained_X,
                 offsetfr=offset_X,
                 brainRegionArray=brain_regions,
                 stimArray=stim_array,
                 mouseIDArray=mouseIDs,
                 sessionIDArray=sessionIDs)
        print(f"Saved {sound_label}!")


# Main processing loop
print("Loading all sessions...")
for subject in params.subject_list:
    print(f"\nProcessing subject: {subject}")

    for date in params.recordingDate_list[subject]:
        for targetSiteName in params.targetSiteNames:
            print(f"  Date: {date}, Brain area: {targetSiteName}")

            for sound_type, sound_type_load in params.sound_type_load:
                print(f"    Sound type: {sound_type}")

                # Load data for this session
                ensemble, ephys, bdata = load_data(subject, date, targetSiteName, sound_type_load)
                if ensemble is None:
                    print(f"      No data for {subject}, {date}, {targetSiteName}, {sound_type}")
                    continue

                # Storage for this session's data across windows
                session_data = {'onset': None, 'sustained': None, 'offset': None}

                # Lock spikes to events once per session, covering all windows
                full_time_range = {'speech': [0.0, 0.7], 'AM': [0.0, 0.7], 'PT': [0.0, 0.13]}
                time_range = full_time_range[sound_type]
                if sound_type == 'speech':
                    eventOnsetTimes = ephys['events']['stimOn']
                else:
                    nTrials = len(bdata['currentFreq'])
                    eventOnsetTimes = ephys['events']['stimOn'][:nTrials]
                ensemble.eventlocked_spiketimes(eventOnsetTimes, time_range)

                # Process each time window for this sound type
                for window_key, window in params.spike_windows.items():
                    # Check if this window belongs to the current sound_type
                    if sound_type.lower() not in window_key.lower():
                        continue

                    # Extract window name (onset, sustained, or offset)
                    window_name = window_key.split(' - ')[1]  # e.g., 'speech - onset' -> 'onset'

                    print(f"      Window: {window_name}")

                    # Calculate firing rate for this window
                    spikesPerSecEvoked, Y_frequency = calculate_firing_rate(
                        sound_type, window, ensemble, bdata)

                    # Normalize firing rate (returns neurons × trials)
                    X, y_brain = normalize_firing_rate(spikesPerSecEvoked, targetSiteName)
                    print(X.shape[1])

                    # Clean and filter data
                    result = clean_data(subject, date, targetSiteName, sound_type, X, Y_frequency,
                                        previous_freq_by_sound[sound_type])

                    # Skip if insufficient data
                    if result is None:
                        print(f"        Skipping due to insufficient data")
                        # If we skip any window, we need to skip the whole session
                        session_data = {'onset': None, 'sustained': None, 'offset': None}
                        break

                    X_clean, Y_freq_clean, previous_freq, indices = result

                    # Update sound-specific tracking variables
                    previous_freq_by_sound[sound_type] = previous_freq
                    indices_by_sound[sound_type] = indices
                    Y_freq_by_sound[sound_type] = Y_freq_clean

                    # Store cleaned data for this session/window
                    session_data[window_name] = X_clean

                    print(f"        Processed {X_clean.shape[0]} neurons × {X_clean.shape[1]} trials")

                # After processing all windows for this session, store if we have data
                if session_data['onset'] is not None:
                    n_neurons = session_data['onset'].shape[0]

                    # Store in session-level dictionary for each window
                    for window_name in ['onset', 'sustained', 'offset']:
                        if session_data[window_name] is not None:
                            data[sound_type][window_name]['X'].append(session_data[window_name])
                            data[sound_type][window_name]['Y_brain'].extend(y_brain)
                            data[sound_type][window_name]['Y_freq'].append(Y_freq_by_sound[sound_type])
                            data[sound_type][window_name]['mouseID'].extend([subject] * n_neurons)
                            data[sound_type][window_name]['sessionID'].extend(
                                [f"{subject}_{date}_{targetSiteName}"] * n_neurons)

                            # Store for population arrays
                            X_all_by_sound[sound_type][window_name].append(session_data[window_name])

                    # Store Y_brain once per session (same for all windows)
                    Y_brain_all_by_sound[sound_type].extend(y_brain)

# Build population-level arrays
print("\nBuilding population arrays...")
population_data = {}

for sound_type in ['speech', 'AM', 'PT']:
    if len(X_all_by_sound[sound_type]['onset']) == 0:
        print(f"No data collected for {sound_type}, skipping population arrays")
        continue

    population_data[sound_type] = {}

    for window_name in ['onset', 'sustained', 'offset']:
        population_data[sound_type][window_name] = {}

        # Sort the X arrays for this window
        X_sorted = sort_x_arrays(
            X_all_by_sound[sound_type][window_name],
            indices_by_sound[sound_type],
            sound_type
        )

        # Concatenate across sessions (stack neurons, axis=0)
        X_array = np.concatenate(X_sorted, axis=0)  # Shape: (total_neurons × trials)
        Y_brain_array = np.array(Y_brain_all_by_sound[sound_type])
        Y_freq_array = Y_freq_by_sound[sound_type]

        print(f"\n{sound_type} - {window_name}: Combined shape {X_array.shape}, Brain areas: {len(Y_brain_array)}")

        # Split by brain area
        for brain_area in params.targetSiteNames:
            brain_area_mask = Y_brain_array == brain_area
            X_brain_area = X_array[brain_area_mask]  # Filter neurons by brain area

            if len(X_brain_area) > 0:
                # Transpose to get (trials × neurons) to match reference code
                X_brain_area = X_brain_area.T

                # Apply neuron selection if needed
                # X_brain_area = funcs.select_neurons(X_brain_area.T, brain_area, params.min_neuron_dict)
                # X_brain_area = X_brain_area.T  # Back to (trials × neurons)

                # Create Y_brain for this brain area (one label per neuron)
                n_neurons = X_brain_area.shape[1]
                Y_brain_this_area = [brain_area] * n_neurons

                population_data[sound_type][window_name][brain_area] = {
                    'X': X_brain_area,  # (trials × neurons)
                    'Y_freq': Y_freq_array,  # (trials,)
                    'Y_brain': Y_brain_this_area  # (neurons,)
                }

                print(f"  {brain_area}: {X_brain_area.shape}")

print("\nDone with population arrays!")

# Save session-level data
print("\nSaving session-level data...")
save_data()
print("Done!")