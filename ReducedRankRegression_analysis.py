"""
This script will calculate pairwise correlations utilizing reduced rank regression and save the results in a feather file
for plotting and other visualization purposes.
"""

import numpy as np
import pandas as pd

from funcs import ReducedRankRegression, cross_validate_rank, preprocess_neural_data
from analysis_class import FiringRateAnalysis
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def perform_within_region_analysis(brain_resp_array, brainRegion, session, respRange, stim, n_iterations=100):
    """
    Perform within-region analysis by randomly splitting cells into two halves
    and running RRR analysis between the halves.

    Parameters:
    -----------
    brain_resp_array : np.ndarray
        Neural response array (nTrials, nNeurons) - BEFORE thresholding
    brainRegion : str
        Name of the brain region
    session : str
        Session identifier
    respRange : str
        Response range (onset, sustained, offset)
    stim : str
        Stimulus type
    n_iterations : int
        Number of random splits to perform (default: 100)

    Returns:
    --------
    dict : Dictionary with averaged results
    """
    r2_values = []
    r2_score_values = []
    explained_variance_values = []
    best_ranks = []

    total_neurons = brain_resp_array.shape[1]
    half_size = total_neurons // 2

    print(
        f"Performing within-region analysis for {brainRegion} with {total_neurons} neurons, {n_iterations} iterations")

    for iteration in tqdm(range(n_iterations)):
        # Randomly shuffle neuron indices and split into two halves
        neuron_indices = np.random.permutation(total_neurons)
        half1_indices = neuron_indices[:half_size]
        half2_indices = neuron_indices[half_size:2 * half_size]  # Ensure equal sizes

        # Create two halves
        half1_data = brain_resp_array[:, half1_indices]
        half2_data = brain_resp_array[:, half2_indices]

        # Preprocess the data
        half1_processed, half2_processed = preprocess_neural_data(half1_data, half2_data, verbose=False)

        # Cross-validate to find best rank
        cv_df, best_rank = cross_validate_rank(half1_processed, half2_processed, verbose=False)

        # Fit RRR model with best rank
        rrr_model = ReducedRankRegression(rank=best_rank, standardize=True)
        rrr_model.fit(half1_processed, half2_processed)

        # Get predictions and calculate correlation
        Y_pred = rrr_model.predict(half1_processed)
        correlation_val = np.corrcoef(Y_pred.flatten(), half2_processed.flatten())[0, 1]

        # Store results
        r2_values.append(correlation_val ** 2)
        r2_score_values.append(rrr_model.score(half1_processed, half2_processed))
        explained_variance_values.append(np.sum(rrr_model.explained_variance_ratio_[:best_rank]))
        best_ranks.append(best_rank)

    # Calculate averages and standard deviations
    avg_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    avg_r2_score = np.mean(r2_score_values)
    avg_explained_variance = np.mean(explained_variance_values)
    avg_best_rank = np.mean(best_ranks)

    print(f"Within-region analysis complete. Average R² = {avg_r2:.4f} ± {std_r2:.4f}")

    return {
        'brain_regions_compared': f"{brainRegion}_within_region",
        'brain_region_1': brainRegion,
        'br_1_cell_count': half_size,
        'brain_region_2': brainRegion,
        'br_2_cell_count': half_size,
        'best_rank': avg_best_rank,
        'r2': avg_r2,
        'r2_std': std_r2,
        'response_range': respRange,
        'stimulus': stim,
        'session': session,
        'r2_score': avg_r2_score,
        'explained_variance_ratio': avg_explained_variance,
        'n_iterations': n_iterations,
        'total_neurons_before_split': total_neurons,
        'analysis_type': 'within_region_split'
    }


#%% Data import
response_ranges = ["onset", "sustained", "offset"]
# response_ranges = ["sustained"]
stim_types = ["pureTones", "AM", "naturalSound"]  # For now only start with pure tones to try and understand analysis meaning
# analysis_attempts = ["correlation", "mean_corr", "PR"]
# stim_types = ["naturalSound", "AM", "pureTones"]

neuron_threshold = 30
fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
stim_types = fr_db.stim_types

ridge_alpha = 1.0

for i_stim, stim in enumerate(stim_types):

    stim_info = fr_db.stim_info[stim]
    nTrials = stim_info['nTrials']
    nCategories = stim_info['nCategories']
    if stim == 'naturalSound':
        soundCats = stim_info['soundCats']
        nInstances = stim_info['nInstances']
        stimVals = stim_info['stimVals']

    stim_info = fr_db.stim_info[stim]
    stim_arrays = fr_db.return_arrays(stim)
    brainRegionArray = stim_arrays["brainRegionArray"]
    mouseIDArray = stim_arrays["mouseIDArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # Stored the trials for each neuron to make sure they were all the same, but only need one now
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)
    uniqSessions = np.unique(sessionArray)
    rrr_results = []

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

                # Special handling for Primary Auditory Area for comparing with itself
                if brainRegion == "Primary auditory area":
                    # Check if we have enough neurons for within-region analysis (need at least 2x N for two halves of 30 each)
                    if region1_sess_count >= 2*neuron_threshold:
                        within_region_result = perform_within_region_analysis(
                            brain_resp_array, brainRegion, session, respRange, stim, n_iterations=10
                        )
                        rrr_results.append(within_region_result)
                    else:
                        print(f"Skipping within-region analysis for {brainRegion} in session {session}: "
                              f"only {region1_sess_count} neurons available (need at least 60)")


                if region1_sess_count < neuron_threshold:
                    print(f"Skipping region 1: {brainRegion} because it has fewer than {neuron_threshold} neurons (n = {region1_sess_count}), session {session}")
                    continue
                # Grab a random N neurons from the array
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

                    brain_resp_array, brain2_resp_array = preprocess_neural_data(brain_resp_array, brain2_resp_array)

                    # RRR model fitting to find the best rank for each region
                    cv_df, best_rank = cross_validate_rank(brain_resp_array, brain2_resp_array, ridge_alpha=ridge_alpha)

                    # Refit now using best rank to store results
                    rrr_model = ReducedRankRegression(rank=best_rank, standardize=True, ridge_alpha=ridge_alpha)
                    rrr_model.fit(brain_resp_array, brain2_resp_array)

                    Y_pred = rrr_model.predict(brain_resp_array)
                    correlation_val = np.corrcoef(Y_pred.flatten(), brain2_resp_array.flatten())[0, 1]

                    rrr_results.append({
                        'brain_regions_compared': f"{brainRegion}_vs_{brainRegion2}",
                        'brain_region_1': brainRegion,
                        'br_1_cell_count': brain_resp_array.shape[1],
                        'brain_region_2': brainRegion2,
                        'br_2_cell_count': brain2_resp_array.shape[1],
                        'best_rank': best_rank,
                        'r2': correlation_val**2,
                        'response_range': respRange,
                        'stimulus': stim,
                        'session': session,
                        'r2_score': rrr_model.score(brain_resp_array, brain2_resp_array),
                        'explained_variance_ratio': np.sum(rrr_model.explained_variance_ratio_[:best_rank])
                    })

    rrr_df = pd.DataFrame(rrr_results)

    # Save as feather file
    rrr_df.to_feather(f"{file_path}/RRR_results_{stim}.feather")

    # Also save as CSV for backup/readability
    # rrr_df.to_csv(f"{file_path}/RRR_results_{stim}.csv", index=False)

    print(f"Saved RRR results for {stim}: {len(rrr_results)} comparisons")


