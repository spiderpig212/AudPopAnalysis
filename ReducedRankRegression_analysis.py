"""
This script will calculate pairwise correlations utilizing reduced rank regression and save the results in a feather file
for plotting and other visualization purposes.
"""

import numpy as np
import pandas as pd

from funcs import ReducedRankRegression, cross_validate_rank, preprocess_neural_data
from analysis_class import FiringRateAnalysis
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#%% Data import
response_ranges = ["onset", "sustained", "offset"]
# response_ranges = ["sustained"]
stim_types = ["pureTones", "AM", "naturalSound"]  # For now only start with pure tones to try and understand analysis meaning
analysis_attempts = ["correlation", "mean_corr", "PR"]
# stim_types = ["naturalSound", "AM", "pureTones"]

neuron_threshold = 30
fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
stim_types = fr_db.stim_types


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

                    brain_resp_array, brain2_resp_array = preprocess_neural_data(brain_resp_array, brain2_resp_array)

                    # RRR model fitting to find the best rank for each region
                    cv_df, best_rank = cross_validate_rank(brain_resp_array, brain2_resp_array)

                    # Refit now using best rank to store results
                    rrr_model = ReducedRankRegression(rank=best_rank, standardize=True)
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
    rrr_df.to_csv(f"{file_path}/RRR_results_{stim}.csv", index=False)

    print(f"Saved RRR results for {stim}: {len(rrr_results)} comparisons")


