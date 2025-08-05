import numpy as np
import pandas as pd
import plotly.graph_objects as go
import funcs
from jaratoolbox import settings
import studyparams

file_path = settings.FIGURES_DATA_PATH + "/" + studyparams.STUDY_NAME
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
# TODO: Add in boxplots where we are making two different ones. One where each plot is one response range, separated by brain area like in my summary boxplots for firing rates, and then one where each plot is one brain area to comapre diff in each response range
#  Store the upper diagonals in a dictionary perhaps to plot later for the boxplot?
for stim in stim_types:

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
    stimArray = stim_arrays["stimArray"][0, :]  # Stored the trials for each neuron to make sure they were all the same, but only need one now
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)


    for respRange in response_ranges:
        respArray = stim_arrays[f"{respRange}fr"]

        for brainRegion in uniqRegions:
            brainRegion_mask = brainRegionArray == brainRegion
            brain_resp_array = respArray[brainRegion_mask, :]

            dprime_stim_vals = np.empty((len(uniqStims), len(uniqStims)))

            for i1, stimType in enumerate(uniqStims):
                stim_mask = stimArray == stimType
                resp_mask = brain_resp_array[:, stim_mask]

                for i2, stimType2 in enumerate(uniqStims):
                    stim_mask2 = stimArray == stimType2
                    resp_mask2 = brain_resp_array[:, stim_mask2]

                    # dprime = funcs.calc_d_prime(resp_mask, resp_mask2)
                    J, w = funcs.calc_fisher_criterion(resp_mask, resp_mask2)
                    dprime_stim_vals[i1, i2] = J  # Need to take the mean for the fisher_criterion as it returns an array of fisher values for each neuron

            upper_indices = np.triu_indices(dprime_stim_vals.shape[0], k=1)
            upper_tri_values = dprime_stim_vals[upper_indices]


            fig = go.Figure(data=go.Heatmap(
                z=dprime_stim_vals,
                colorscale='Viridis'  # You can change the color scale as needed
            ))

            fig.update_layout(
                title=f"Fisher-criterion for {brainRegion} - {stim} - {respRange} responses - {resp_mask.shape[0]} neurons",
                xaxis=dict(title="stimVals",
                           tickvals=list(range(len(uniqStims))),
                           ticktext=stimVals if stim == 'naturalSound' else uniqStims),
                yaxis=dict(title="stimVals",
                           tickvals=list(range(len(uniqStims))),
                           ticktext=stimVals if stim == 'naturalSound' else uniqStims,
                           autorange='reversed',
            ))

            # fig.show()
            fig.write_html(f"{file_path}/fisher_criterion_plots/FCriterion_{brainRegion}_{stim}_{respRange}.html")


