"""
This file contains code for running canonical correlation analysis. Compares responses from different conditions in
various brain areas to see if there is any relationship between primary auditory area and either secondary cortical area.
"""

import os
import sys
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import studyparams
from jaratoolbox import settings, celldatabase

#%% Data import
file_path = settings.FIGURES_DATA_PATH + "/" + studyparams.STUDY_NAME
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["pureTones"]  # For now only start with pure tones to try and understand analysis meaning
# stim_types = ["naturalSound", "AM", "pureTones"]

neuron_threshold = 20
figdataPath = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)

for i_stim, stim in enumerate(stim_types):

    if stim == 'AM':
        nTrials = 220
        nCategories = 11
    elif stim == 'naturalSound':
        nTrials = 200
        nCategories = len(studyparams.SOUND_CATEGORIES)
        soundCats = studyparams.SOUND_CATEGORIES
        nInstances = 4
        stimVals = np.empty(nInstances*nCategories, dtype=object)
        for i in range(nCategories):
            for j in range(nInstances):
                stimVals[i*nInstances+j] = soundCats[i] + f"_{j+1}"
    elif stim == 'pureTones':
        nTrials = 320
        nCategories = 16

    stim_arrays = np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    mouseIDArray = stim_arrays["mouseIDArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # Stored the trials for each neuron to make sure they were all the same, but only need one now
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)
    uniqSessions = np.unique(sessionArray)

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
                region1_neurons = np.random.choice(brain_resp_array.shape[1], size=20, replace=False)
                brain_resp_array = brain_resp_array[:, region1_neurons]

                for brainRegion2 in uniqRegions[i+1:]:
                    brainRegion2_mask = brain_session_array == brainRegion2
                    brain2_resp_array = session_resp_array[brainRegion2_mask, :].T  # Make the array (nTrials, nNeurons)
                    region2_sess_count = brain2_resp_array.shape[1]
                    if region2_sess_count < neuron_threshold:
                        print(f"Skipping region 2: {brainRegion2} because it has fewer than {neuron_threshold} neurons (n = {region2_sess_count}), session {session}")
                        continue
                    region2_neurons = np.random.choice(brain2_resp_array.shape[1], size=20, replace=False)
                    brain2_resp_array = brain2_resp_array[:, region2_neurons]

                    n_components = np.min([brain_resp_array.shape[1], brain2_resp_array.shape[1]])  # Whichever region has fewer neurons (should always be equal to neuron threshold now)
                    cca = CCA(n_components=n_components)
                    response_transform = cca.fit_transform(brain_resp_array, brain2_resp_array)

                    correlation_val = np.corrcoef(response_transform[0][:, 0], response_transform[1][:, 0])[0, 1]
                    plt.scatter(response_transform[0][:, 0], response_transform[1][:, 0])
                    plt.xlabel(f'{brainRegion}_canonical_dimension_0')
                    plt.ylabel(f'{brainRegion2}_canonical_dimension_0')
                    plt.title(f'First pair of canonical variables, Pearson correlation = {correlation_val:.2f}')
                    plt.savefig(f"{file_path}/CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}.png")
                    plt.show()

                    print(f"Weight norms = {np.linalg.norm(cca.y_weights_[0])} and weight shape is {cca.y_weights_.shape}")

                    print("Now plotting the canonical dimension in original data space")
                    origin = [0, 0]
                    br1_fr = brain_resp_array[:, :2]
                    br1_weights = cca.x_weights_[:, 0]
                    br2_fr = brain2_resp_array[:, :2]
                    br2_weights = cca.x_weights_[:, 1]

                    # Making scatters
                    plt.scatter(br1_fr[:, 0], br1_fr[:, 1], c=stimArray, cmap='viridis', alpha=0.3, s=2)
                    plt.xlabel(f'{brainRegion}_neuron_0')
                    plt.ylabel(f'{brainRegion}_neuron_1')
                    plt.title(f'{brainRegion} Neuron 0 and 1 vs CCA component 0')
                    plt.colorbar(label='Stimulus')
                    plt.quiver(*origin, br1_weights[0], br1_weights[1], scale_units='xy', scale=0.001, color='black',
                               angles='xy', label='CCA separation')
                    plt.savefig(f"{file_path}/CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}_original_space.png")
                    plt.show()

                    plt.scatter(br2_fr[:, 0], br2_fr[:, 1], c=stimArray, cmap='viridis', alpha=0.3, s=2)
                    plt.xlabel(f'{brainRegion2}_neuron_0')
                    plt.ylabel(f'{brainRegion2}_neuron_1')
                    plt.title(f'{brainRegion2} Neuron 0 and 1 vs CCA component 0')
                    plt.colorbar(label='Stimulus')
                    plt.quiver(*origin, br2_weights[0], br2_weights[1], scale_units='xy', scale=0.001, color='black',
                               angles='xy', label='CCA separation')
                    plt.savefig(f"{file_path}/CCA_plots/{brainRegion}_{brainRegion2}_{respRange}_{stim}_{session}_original_space2.png")
                    plt.show()





