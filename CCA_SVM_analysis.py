"""
This file contains code for running canonical correlation analysis. Compares responses from different conditions in
various brain areas to see if there is any relationship between primary auditory area and either secondary cortical area.
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from analysis_class import AnalysisBase, FiringRateAnalysis

#%% Data import
response_ranges = ["onset", "sustained", "offset"]
# stim_types = ["pureTones", "AM", "naturalSound"]  # For now only start with pure tones to try and understand analysis meaning
# analysis_attempts = ["correlation", "mean_corr", "PR"]
# stim_types = ["naturalSound", "AM", "pureTones"]

neuron_threshold = 20
n_splits = 5
fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
stim_types = fr_db.stim_types

# Dictionary to store decision boundaries for visualization
decision_boundaries_data = {}

for i_stim, stim in enumerate(stim_types):

    stim_info = fr_db.stim_info[stim]
    nTrials = stim_info['nTrials']
    nCategories = stim_info['nCategories']
    if stim == 'naturalSound':
        soundCats = stim_info['soundCats']
        nInstances = stim_info['nInstances']
        stimVals = stim_info['stimVals']

    stim_arrays = fr_db.return_arrays(stim)
    brainRegionArray = stim_arrays["brainRegionArray"]
    brainRegionArray[brainRegionArray == "Posterior auditory area"] = "Dorsal auditory area"
    mouseIDArray = stim_arrays["mouseIDArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # Stored the trials for each neuron to make sure they were all the same, but only need one now
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)
    reorder = np.array([1,0,2,3])
    uniqRegions = uniqRegions[reorder]
    uniqSessions = np.unique(sessionArray)
    correlation_data = []

    C_values = np.logspace(-3, 1.2, 20)

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
                # TODO: Move this into brainRegion2 so we can include a method for pulling neurons multiple times and averaging
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

                    # TODO: this is where we should average neurons

                    br1_train, br1_test, br2_train, br2_test = train_test_split(brain_resp_array, brain2_resp_array, test_size=0.2, random_state=42)
                    n_components = np.min([brain_resp_array.shape[1], brain2_resp_array.shape[1]])  # Whichever region has fewer neurons (should always be equal to neuron threshold now)
                    cca = CCA(n_components=n_components)
                    response_transform_source, response_transform_target = cca.fit_transform(brain_resp_array, brain2_resp_array) # Should be (nTrials, nDims)

                    if stim == 'naturalSound' or stim == 'AM' or stim == 'pureTones':
                        # Natural sounds can't be linearized, so we must instead do pairwise A vs B SVM fits and store accuracies in matrix
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        for stim_idx, stim_val in tqdm(enumerate(uniqStims)):
                            for stim_idx2, stim_val2 in enumerate(uniqStims):
                                if stim_idx == stim_idx2:
                                    continue
                                stim_mask = stimArray == stim_val
                                stim_mask2 = stimArray == stim_val2
                                combined_mask = stim_mask | stim_mask2
                                combined_resp_array_source = response_transform_source[combined_mask, :]
                                combined_resp_array_target = response_transform_target[combined_mask, :]
                                combined_masked_resp_array = brain_resp_array[combined_mask, :]
                                masked_stims = stimArray[combined_mask].astype(int)

                                C_results = []
                                for C_val in C_values:
                                    svm_cca = SVC(C=C_val, random_state=42, kernel='linear')
                                    svm = SVC(C=C_val, random_state=42, kernel='linear')
                                    cv_scores = cross_val_score(svm_cca, combined_resp_array_source, masked_stims,
                                                                cv=cv, scoring='accuracy')
                                    chance_level = 0.5  # Binary classification
                                    cv_scores_untransformed = cross_val_score(svm, combined_masked_resp_array,
                                                                              masked_stims, cv=cv, scoring='accuracy')
                                    C_results.append({
                                        'C': C_val,
                                        'cv_scores_cca': cv_scores,
                                        'cv_scores': cv_scores_untransformed,
                                        'mean_accuracy_cca': np.mean(cv_scores),
                                        'mean_accuracy': np.mean(cv_scores_untransformed),
                                        'std_accuracy_cca': np.std(cv_scores),
                                        'std_accuracy': np.std(cv_scores_untransformed),
                                    })

                                C_df = pd.DataFrame(C_results)

                                # Select the result with the highest mean CCA accuracy
                                best = max(C_results, key=lambda x: x['mean_accuracy_cca'])
                                best_C_val = best['C']

                                # Plotting the hyper parameter sweep
                                plt.figure(figsize=(10, 6))
                                plt.subplot(1, 2, 1)
                                plt.plot(C_values, C_df['mean_accuracy_cca'], 'bo-')
                                plt.xlabel('C')
                                plt.ylabel('Mean Accuracy')
                                plt.title('Hyperparameter Sweep for C')
                                plt.subplot(1, 2, 2)
                                plt.plot(C_values, C_df['std_accuracy_cca'], 'bo-')
                                plt.xlabel('C')
                                plt.ylabel('Standard Deviation of Accuracy')
                                plt.title('Hyperparameter Sweep for C')
                                plt.tight_layout()
                                plt.savefig(f"{file_path}/CCA_SVC/CCA_SVC_C_plots/CCA_SVC_hyperparameter_sweep_{session}_{stim}_{respRange}.png")
                                # plt.show()
                                plt.close()

                                # Store decision boundary data using the best C
                                boundary_key = f"{stim}_{respRange}_{brainRegion}_vs_{brainRegion2}_{session}_{stim_val}_{stim_val2}_C{best_C_val}"
                                if boundary_key not in decision_boundaries_data and len(decision_boundaries_data) < 5:
                                    fold_iterator = cv.split(combined_resp_array_source, masked_stims)
                                    train_idx, test_idx = next(fold_iterator)
                                    X_train_fold = combined_resp_array_source[train_idx]
                                    y_train_fold = masked_stims[train_idx]
                                    X_test_fold = combined_resp_array_source[test_idx]
                                    y_test_fold = masked_stims[test_idx]

                                    # TODO: Change SVC kernal to linear
                                    # TODO: Also add in calcualtion for dorsal v ventral and ventral v dorsal
                                    svm_boundary = SVC(C=best_C_val, random_state=42, kernel='linear')
                                    svm_boundary.fit(X_train_fold, y_train_fold)
                                    y_pred_fold = svm_boundary.predict(X_test_fold)

                                    decision_boundaries_data[boundary_key] = {
                                        'X_train': X_train_fold,
                                        'y_train': y_train_fold,
                                        'X_test': X_test_fold,
                                        'y_test': y_test_fold,
                                        'y_pred': y_pred_fold,
                                        'svm_model': svm_boundary,
                                        'region_pair': f"{brainRegion}_vs_{brainRegion2}",
                                        'stim_type': stim,
                                        'response_range': respRange,
                                        'session': session,
                                        'stim_pair': (stimVals[int(stim_val)], stimVals[int(stim_val2)]),
                                        'C': best_C_val
                                    }

                                correlation_data.append({
                                    'region_pair': f"{brainRegion}_vs_{brainRegion2}",
                                    'region1': brainRegion,
                                    'region2': brainRegion2,
                                    'mean_accuracy_cca': best['mean_accuracy_cca'],
                                    'mean_accuracy': best['mean_accuracy'],
                                    'std_accuracy_cca': best['std_accuracy_cca'],
                                    'std_accuracy': best['std_accuracy'],
                                    'cv_scores_cca': best['cv_scores_cca'],
                                    'cv_scores': best['cv_scores'],
                                    'n_trials_cca': len(combined_resp_array_source),
                                    'n_classes': 2,
                                    'stim_pair': (stim_val, stim_val2),
                                    'chance_level': chance_level,
                                    'response_range': respRange,
                                    'stimulus': stim,
                                    'session': session,
                                    'C': best_C_val
                                })

    df_correlations = pd.DataFrame(correlation_data)
    df_correlations.to_feather(f"{file_path}/CCA_SVM_{stim}.feather")
    df_correlations.to_csv(f"{file_path}/CCA_SVM_{stim}.csv", index=False)

# Save decision boundaries data
import pickle
with open(f"{file_path}/decision_boundaries_data.pkl", 'wb') as f:
    pickle.dump(decision_boundaries_data, f)

