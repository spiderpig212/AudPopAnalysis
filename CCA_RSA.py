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

brain_region_1 = "Primary auditory area"
target_region_1 = "Dorsal auditory area"
target_region_2 = "Ventral auditory area"

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

            primary_array = session_resp_array[brain_session_array == brain_region_1, :]
            dorsal_array = session_resp_array[brain_session_array == target_region_1, :]
            ventral_array = session_resp_array[brain_session_array == target_region_2, :]

            # Check to see if we have enough neurons for each region
            if primary_array.shape[0] < neuron_threshold or dorsal_array.shape[0] < neuron_threshold or ventral_array.shape[0] < neuron_threshold:
                print(f"[{stimulus}/{response_range}] Skipping session {session}: insufficient neurons in Primary, Dorsal, or Ventral")
                continue
            rsa_matrices = []
            for _ in range(n_splits):
                # Randomly sample neuron_threshold neurons from each region
                primary_array_subset = primary_array[np.random.choice(primary_array.shape[0], neuron_threshold, replace=False), :]
                dorsal_array_subset = dorsal_array[np.random.choice(dorsal_array.shape[0], neuron_threshold, replace=False), :]
                ventral_array_subset = ventral_array[np.random.choice(ventral_array.shape[0], neuron_threshold, replace=False), :]

                mask_n_comps_dorsal = (
                        (significant_df["region1"] == brain_region_1)
                        & (significant_df["region2"] == target_region_1)
                        & (significant_df["stimulus"] == stimulus)
                        & (significant_df["response_range"] == response_range)
                        & (significant_df["session"] == session)
                )
                try:
                    n_components_dorsal = significant_df.loc[mask_n_comps_dorsal, "significant_components"].iloc[0]
                    n_components_dorsal = np.int64(n_components_dorsal)
                    if n_components_dorsal < 1:
                        continue
                except (IndexError, ValueError):
                    print(f"[{stimulus}/{response_range}] No significant components found for {brain_region_1} vs {target_region_1}, "
                          f"session {session}")
                    continue

                mask_n_comps_ventral = (
                        (significant_df["region1"] == brain_region_1)
                        & (significant_df["region2"] == target_region_2)
                        & (significant_df["stimulus"] == stimulus)
                        & (significant_df["response_range"] == response_range)
                        & (significant_df["session"] == session)
                )
                try:
                    n_components_ventral = significant_df.loc[mask_n_comps_ventral, "significant_components"].iloc[0]
                    n_components_ventral = np.int64(n_components_ventral)
                    if n_components_ventral < 1:
                        continue
                except (IndexError, ValueError):
                    print(
                        f"[{stimulus}/{response_range}] No significant components found for {brain_region_1} vs {target_region_2}, "
                        f"session {session}")
                    continue
                n_components_test = min(n_components_dorsal, n_components_ventral)  # Have to use the minimum number of components for pearsonr
                if n_components_test <= 1:
                    print(f"[{stimulus}/{response_range}] Skipping session {session}: insufficient components for pearsonr")
                    continue
                cca_dorsal = CCA(n_components=n_components_test)
                pd_transform, dorsal_transform = cca_dorsal.fit_transform(primary_array_subset.T, dorsal_array_subset.T)
                # Transpose to make the shape (n_trials, n_neurons) for the cca so the end result is (n_trials, n_components)
                cca_ventral = CCA(n_components=n_components_test)
                pv_transform, ven_transform = cca_ventral.fit_transform(primary_array_subset.T, ventral_array_subset.T)

                rsa_matrix = np.ones((len(uniqStims), len(uniqStims)))
                for stim_idx, stim in enumerate(uniqStims):
                    for stim_idx2, stim2 in enumerate(uniqStims):
                        stim_mask = stim_arrays["stimArray"][0, :] == stim
                        stim_mask2 = stim_arrays["stimArray"][0, :] == stim2
                        corr_val = stats.pearsonr(pd_transform[stim_mask, :].mean(axis=0), pv_transform[stim_mask2, :].mean(axis=0))[0]
                        rsa_matrix[stim_idx, stim_idx2] = corr_val

                rsa_matrices.append(rsa_matrix)

            rsa_matrix_averaged = np.mean(rsa_matrices, axis=0)
            rsa_data.append({
                "stimulus": stimulus,
                "response_range": response_range,
                "session": session,
                "rsa_matrix": rsa_matrix_averaged,
                "target_region1": target_region_1,
                "target_region2": target_region_2,
            })

        pickle.dump(rsa_data, open(f"{file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.pkl", "wb"))
        print(f"RSA data saved to {file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.pkl")
        rsa_frame = pd.DataFrame(rsa_data)
        rsa_frame.to_csv(f"{file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.csv")
        # try:
        #     rsa_frame.to_feather(f"{file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.feather")
        #     print(f"RSA data saved to {file_path}/CCA_RSA/rsa_data_{stimulus}_{response_range}.feather")
        # except:
        #     print("Failed to save to feather")


