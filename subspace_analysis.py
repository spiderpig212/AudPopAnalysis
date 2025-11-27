"""
Calculates how similar subsapces are for different brain regions of interest across stimuli
"""

import os
import sys
import numpy as np
import pandas as pd
from argon2 import PasswordHasher
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from analysis_class import FiringRateAnalysis
from sklearn.cross_decomposition import CCA
import funcs as funcs

neuron_threshold = 20
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["pureTones", "AM", "naturalSound"]  # For now only start with pure tones to try and understand analysis meaning
analysis_attempts = ["correlation", "mean_corr", "PR"]
# stim_types = ["naturalSound", "AM", "pureTones"]

fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path
stim_types = fr_db.stim_types

ssa_analysis = []

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
    # uniqRegions = np.unique(brainRegionArray)
    uniqRegions = np.array(['Primary auditory area', 'Dorsal auditory area', 'Ventral auditory area', 'Posterior auditory area'])  # Manually setting this ordering to make plotting comparisons for SSA easier by making Primary the first region compared to others
    uniqSessions = np.unique(sessionArray)
    correlation_data = []

    for respRange in response_ranges:
        respArray = stim_arrays[f"{respRange}fr"]

        for session in uniqSessions:
            session_mask = sessionArray == session
            session_resp_array = respArray[session_mask, :]
            brain_session_array = brainRegionArray[session_mask]
            min_threshold = 100000
            for i, brainRegion in enumerate(uniqRegions):
                brainRegion_mask = brain_session_array == brainRegion
                resp_array = session_resp_array[brainRegion_mask, :].T  # Make the array (nTrials, nNeurons)
                region_sess_count = resp_array.shape[1]
                if region_sess_count < min_threshold and region_sess_count >= neuron_threshold:
                    min_threshold = region_sess_count
            for i, brainRegion in enumerate(uniqRegions):

                brainRegion_mask = brain_session_array == brainRegion
                brain_resp_array = session_resp_array[brainRegion_mask, :].T  # Make the array (nTrials, nNeurons)
                brain_resp_array = brain_resp_array - brain_resp_array.mean()
                region1_sess_count = brain_resp_array.shape[1]
                if region1_sess_count < neuron_threshold:
                    print(
                        f"Skipping region 1: {brainRegion} because it has fewer than {neuron_threshold} neurons (n = {region1_sess_count}), session {session}")
                    continue
                # Grab a random min_threshold neurons from the array so we can sample as many as possible
                region1_neurons = np.random.choice(brain_resp_array.shape[1], size=min_threshold, replace=False)
                brain_resp_array = brain_resp_array[:, region1_neurons]

                for brainRegion2 in uniqRegions[i + 1:]:
                    brainRegion2_mask = brain_session_array == brainRegion2
                    brain2_resp_array = session_resp_array[brainRegion2_mask, :].T  # Make the array (nTrials, nNeurons)
                    brain2_resp_array = brain2_resp_array - brain2_resp_array.mean()
                    region2_sess_count = brain2_resp_array.shape[1]
                    if region2_sess_count < neuron_threshold:
                        print(
                            f"Skipping region 2: {brainRegion2} because it has fewer than {neuron_threshold} neurons (n = {region2_sess_count}), session {session}")
                        continue
                    region2_neurons = np.random.choice(brain2_resp_array.shape[1], size=min_threshold, replace=False)
                    brain2_resp_array = brain2_resp_array[:, region2_neurons]

                    n_components = np.min([brain_resp_array.shape[1], brain2_resp_array.shape[
                        1]]) - 1  # Whichever region has fewer neurons (should always be equal to min threshold now). Subtract one component as the final dimension is 0's
                    cca = CCA(n_components=n_components)
                    response_transform = cca.fit_transform(brain_resp_array, brain2_resp_array)

                    all_corr_vals = []
                    for cca_component in range(n_components):
                        corr_val = np.corrcoef(response_transform[0][:, cca_component], response_transform[1][:, cca_component])[0, 1]
                        if np.sum(np.isnan(corr_val)) > 0:
                            print(f"NaN found in correlation calculation for {brainRegion} vs {brainRegion2} in session {session}")
                            corr_val = 0
                        all_corr_vals.append(corr_val)
                    corr_array = np.array(all_corr_vals)**2

                    br1_weights = cca.x_weights_  # Shape is (n_features n_components) from documentation, is the left singular vectors of the CC matrices of each iteration (u^A->B in our written notation)
                    br2_weights = cca.y_weights_  # Shape is (n_targets, n_components) from documentation, ^ but for the right singular vectors (u^B->A in our written notation)
                    print(f"X dot of first two components is: {np.dot(cca.x_weights_[0, 0], cca.x_weights_[0, 1])}")
                    print(f"Y dot of first two components is: {np.dot(cca.y_weights_[0, 0], cca.y_weights_[0, 1])}")

                    # Plotting the CCA dimensions in PC space
                    pca1, pca2, pc_data1, pc_data2, cca_weights_pc1, cca_weights_pc2 = funcs.plot_pca_with_cca_weights(
                        brain_resp_array, brain2_resp_array, br1_weights, br2_weights,
                        stimArray, brainRegion, brainRegion2, n_pc_components=10,
                        save_path=f"{file_path}/subspace_overlap_analysis/PCA_CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}"
                    )

                    print(f"PC1 dot of first two components is: {np.dot(cca_weights_pc1[0, 0], cca_weights_pc1[0, 1])}")
                    print(f"PC2 dot of first two components is: {np.dot(cca_weights_pc2[0, 0], cca_weights_pc2[0, 1])}")

                    # Comparing CCA weights/angles
                    # funcs.plot_cca_weights_comparison(pc_data1, pc_data2, cca_weights_pc1, cca_weights_pc2,
                    #                             brainRegion, brainRegion2,
                    #                             save_path=f"{file_path}/subspace_overlap_analysis/PCA_CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}")


                    cca_cov_mat = np.einsum('ij,j,jk->ik',br1_weights, corr_array, br1_weights.T)  # Should be the same as br1_weights @ np.diag(corr_array) @ br1_weights.T, shape (min_neuron, min_neuron) as we sum over components
                    ssa_analysis.append({
                        'region_pair': f"{brainRegion}_vs_{brainRegion2}",
                        'region1': brainRegion,
                        #'region1_resp_array_shape': brain_resp_array.shape,
                        'region1_resp_array': brain_resp_array,
                        'region2': brainRegion2,
                        #'region2_resp_array_shape': brain2_resp_array.shape, Remnants from wanting to save to feather file since feather cant store 2d arrays in frames
                        'region2_resp_array': brain2_resp_array,
                        'corr_array': corr_array,
                        'br1_weights': br1_weights,
                        'br2_weights': br2_weights,
                        'cca_cov_mat': cca_cov_mat,
                        'session': session,
                        'stimulus': stim,
                        'n_neurons': min_threshold,
                        'response_range': respRange,
                    })

ssa_frame = pd.DataFrame(ssa_analysis)
ssa_frame.to_csv(f"{file_path}/subspace_overlap_analysis/SSA_analysis.csv", index=False)
# ssa_frame.to_feather(f"{file_path}/SSA_analysis.feather")

# Begin calculations for our subspace analysis metric
ss_overlap_analysis = []
for stim in stim_types:
    ssa_stim_frame = ssa_frame[ssa_frame['stimulus'] == stim]
    for respRange in response_ranges:
        ssa_respRange_frame = ssa_stim_frame[ssa_stim_frame['response_range'] == respRange]
        for session in uniqSessions:
            ssa_session_frame = ssa_respRange_frame[ssa_stim_frame['session'] == session]
            for i, brainRegion in enumerate(uniqRegions):
                brainRegion_frame = ssa_session_frame[ssa_session_frame['region1'] == brainRegion]
                # Check to see if we have multiple brain regions in the same session we can compare brain region subspaces between as we need at least 2
                if len(np.unique(brainRegion_frame['region2'])) > 1:
                    regions_to_compare = []
                    cca_cov_mats = []
                    brain_region_A_weights = []
                    for j, row in brainRegion_frame.iterrows():
                        region_pair = row.region_pair.split('_')
                        region_pair.pop(1)  # Remove the vs as we just want region names
                        regions_to_compare += region_pair
                        cca_cov_mats.append(row.cca_cov_mat)
                        response_data = row.region1_resp_array
                        n_neurons = row.n_neurons
                        brain_region_A_weights.append(row.br1_weights)
                    regions_to_compare.pop(2)
                    subspace_comp_brain_region_string = f"{regions_to_compare[0]} vs {regions_to_compare[1]} and {regions_to_compare[2]}"
                    u_ab = brain_region_A_weights[0]
                    u_ac = brain_region_A_weights[1]
                    cca_cov1 = cca_cov_mats[0]
                    cca_cov2 = cca_cov_mats[1]
                    d = funcs.subspace_overlap_analysis(cca_cov1, cca_cov2)
                    fig1_ellipses, fig2_directions = funcs.visualize_subspace_overlap(response_data, u_ab, u_ac, d, subspace_comp_brain_region_string, f"{file_path}/subspace_overlap_analysis/SSA_plots/{brainRegion}_{respRange}_{session}.png")

                    ss_overlap_analysis.append({
                        'region_comparison': subspace_comp_brain_region_string,
                        'regionA': regions_to_compare[0],
                        'regionB': regions_to_compare[1],
                        'regionC': regions_to_compare[2],
                        'n_neurons': n_neurons,
                        'SSA_overlap': d,
                        'session': session,
                        'stimulus': stim,
                        'response_range': respRange,
                    })

ss_overlap_analysis_frame = pd.DataFrame(ss_overlap_analysis)
ss_overlap_analysis_frame.to_csv(f"{file_path}/subspace_overlap_analysis/SSA_overlap_analysis.csv", index=False)



