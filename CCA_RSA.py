"""
Goes through Primary plus each other brain area and computes RSA matrices comparing stimuli correlations
"""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for multiprocessing

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
import pandas as pd

from analysis_class import FiringRateAnalysis

response_ranges = ["onset", "sustained", "offset"]
neuron_threshold = 20
n_splits = 5

fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
stim_types = fr_db.stim_types

significant_df = pd.read_csv(
                    f"{file_path}/CCA_two_region_analysis/cca_primary_auditory_results.csv")


for stimulus in stim_types:
    print(f"Computing RSA for {stimulus}")
    stim_arrays = fr_db.return_arrays(stimulus)
    brainRegionArray = stim_arrays["brainRegionArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    uniqSessions = np.unique(sessionArray)
    # Rename posterior to also be dorsal instead of treating them as separate areas
    brainRegionArray[brainRegionArray == "Posterior auditory area"] = "Dorsal auditory area"
    uniq_regions = np.unique(brainRegionArray)
    uniqStims = np.unique(stim_arrays["stimArray"][0, :])

    for response_range in response_ranges:
        rsa_data = []
        respArray = stim_arrays[f"{response_range}fr"]

        for session in uniqSessions:
            session_mask = sessionArray == session
            session_resp_array = respArray[session_mask, :]
            brain_session_array = brainRegionArray[session_mask]

            for brain_region in uniq_regions:
                primary_array = session_resp_array[brain_session_array == brain_region, :]

                # Check to see if we have enough neurons for each region
                if primary_array.shape[0] < neuron_threshold:
                    print(
                        f"[{stimulus}/{response_range}] Skipping session {session}: insufficient neurons in {brain_region}")
                    continue

                for target_region in uniq_regions:
                    if brain_region == target_region:
                        continue
                    target_array = session_resp_array[brain_session_array == target_region, :]
                    if target_array.shape[1] < neuron_threshold:
                        print(f"[{stimulus}/{response_range}] Skipping session {session}: insufficient neurons in {target_region}")
                        continue

                    mask_n_comps_target = (
                            (significant_df["region1"] == brain_region)
                            & (significant_df["region2"] == target_region)
                            & (significant_df["stimulus"] == stimulus)
                            & (significant_df["response_range"] == response_range)
                            & (significant_df["session"] == session)
                    )
                    try:
                        n_components_target = significant_df.loc[mask_n_comps_target, "significant_components"].iloc[
                            0]
                        n_components_target = np.int64(n_components_target)
                        if n_components_target < 1:
                            print(
                                f"[{stimulus}/{response_range}] Skipping session {session}: insufficient significant components for {brain_region} vs {target_region}")
                            continue
                    except (IndexError, ValueError):
                        print(
                            f"[{stimulus}/{response_range}] No significant components found for {brain_region} vs {target_region}, "
                            f"session {session}")
                        continue

                    rsa_matrices = []
                    for _ in range(n_splits):
                        # Randomly sample neuron_threshold neurons from each region
                        primary_array_subset = primary_array[np.random.choice(primary_array.shape[0], neuron_threshold, replace=False), :]
                        primary_zero_mean = primary_array_subset - np.mean(primary_array_subset)

                        target_array_subset = target_array[np.random.choice(target_array.shape[0], neuron_threshold, replace=False), :]
                        target_zero_mean = target_array_subset - np.mean(target_array_subset)

                        cca_mod = CCA(n_components=n_components_target)
                        primary_transform, target_transform = cca_mod.fit_transform(primary_zero_mean.T, target_zero_mean.T)
                        # Transpose to make the shape (n_trials, n_neurons) for the cca so the end result is (n_trials, n_components)

                        rsa_matrix = np.ones((len(uniqStims), len(uniqStims)))
                        for stim_idx, stim in enumerate(uniqStims):
                            for stim_idx2, stim2 in enumerate(uniqStims):
                                stim_mask = stim_arrays["stimArray"][0, :] == stim
                                stim_mask2 = stim_arrays["stimArray"][0, :] == stim2
                                stim1_trial_mean = primary_transform[stim_mask, :].mean(axis=0)  # Average over trials so shape is (n_dim)
                                stim2_trial_mean = primary_transform[stim_mask2, :].mean(axis=0)
                                corr_val = stats.pearsonr(stim1_trial_mean, stim2_trial_mean)[0]
                                rsa_matrix[stim_idx, stim_idx2] = corr_val

                        rsa_matrices.append(rsa_matrix)

                    rsa_matrix_averaged = np.mean(rsa_matrices, axis=0)
                    rsa_data.append({
                        "stimulus": stimulus,
                        "response_range": response_range,
                        "session": session,
                        "rsa_matrix": rsa_matrix_averaged,
                        "brain_region": brain_region,
                        "target_region1": target_region,
                        "pair_comparison": f"{brain_region}_{target_region}",
                    })

        pickle.dump(rsa_data, open(f"{file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.pkl", "wb"))
        print(f"RSA data saved to {file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.pkl")
        rsa_frame = pd.DataFrame(rsa_data)
        rsa_frame.to_csv(f"{file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.csv")


