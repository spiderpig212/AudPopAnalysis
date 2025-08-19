import os
import numpy as np
from sklearn.decomposition import PCA

import funcs
from jaratoolbox import celldatabase
import matplotlib.pyplot as plt
from jaratoolbox import settings
from jaratoolbox import extraplots
from jaratoolbox import ephyscore
from jaratoolbox import behavioranalysis
import sys
import studyparams
import studyutils
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors
from importlib import reload

#%%
neuron_threshold = 65
figdataPath = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)

dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath,f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)
simpleSiteNames = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])#.value_counts()
simpleSiteNames.name = 'simpleSiteName'
celldb = pd.concat([celldb, simpleSiteNames], axis=1)

areas_of_interest = [
    "Dorsal auditory area",
    "Primary auditory area",
    "Ventral auditory area"
]

stim_types = ["naturalSound", "AM", "pureTones"]
response_ranges = ["onset", "sustained", "offset"]

aud_db = celldb[celldb['simpleSiteName'].isin(areas_of_interest)].reset_index()

grouped_data = aud_db.groupby(['simpleSiteName', 'date']).size()
grouped_bools = grouped_data > neuron_threshold
print(f"With {neuron_threshold} neurons, we have: {grouped_bools.groupby(['simpleSiteName']).sum()}")
session_list = grouped_bools[grouped_bools == True]
# Can maybe use the indeces of session_list to then use the fr_arrays to grab the specific dates and sessions needed
#  Store each resulting one in a new array for our actual PCA and participation ratio calculations

num_sessions =len(session_list)
total_neurons = grouped_data[grouped_bools == True].sum()
data = np.array([])
area_array = np.empty(total_neurons, dtype=object)
session_array = np.empty(total_neurons, dtype=object)
# TODO: I think I should sort all the sessions by the stim presentation and then change the three arrays below to just
#  store one of each for each session for coloring the PCAs later. Eventual goals will be PCA plots of every session
#  so we make 3x3 plots for each session,  with the grid being brain area and stim type. Store all the eigenvalues and
#  then calculate participation ratio with SEM and then also do a stats test comparing brain area
transformed_act_onset = {session: [] for session in session_list.index}
transformed_act_sustained = {session: [] for session in session_list.index}
transformed_act_offset = {session: [] for session in session_list.index}
explained_vars_onset = {session: [] for session in session_list.index}
explained_vars_sustained = {session: [] for session in session_list.index}
explained_vars_offset = {session: [] for session in session_list.index}

brain_plot_map = {"Dorsal auditory area": 1, "Primary auditory area": 2, "Ventral auditory area": 3}
response_map = {"onset": 1, "sustained": 2, "offset": 3}

for stim in stim_types:
    stim_arrays = np.load(f"{figdataPath}/fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    mouseIDArray = stim_arrays["mouseIDArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    sorted_stim_ind = np.argsort(stimArray)
    sorted_stim_array = stimArray[sorted_stim_ind]

    for respRange in response_ranges:
        respArray = stim_arrays[f"{respRange}fr"]
        sortedRespArray = respArray[:, sorted_stim_ind]
        for area, session in session_list.index:
            area_ind = np.where(brainRegionArray == area)[0]
            session_ind = np.where(sessionArray == session)[0]
            target_data_ind = np.intersect1d(area_ind, session_ind)
            target_data = sortedRespArray[target_data_ind, :]
            mean_zero_target_data = target_data - target_data.mean(axis=1, keepdims=True)
            random_indices = np.random.choice(mean_zero_target_data.shape[0], size=neuron_threshold, replace=False)
            mean_zero_target_data = mean_zero_target_data[random_indices]

            pc = PCA()
            pc.fit(mean_zero_target_data)
            transformed_activity = pc.transform(mean_zero_target_data)
            variances = pc.explained_variance_ratio_
            if respRange == "onset":
                explained_vars_onset[(area, session)].append(variances)  # Will contain 3 arrays in order of stim_types
                transformed_act_onset[(area, session)].append(transformed_activity)
            elif respRange == "sustained":
                explained_vars_sustained[(area, session)].append(variances)
                transformed_act_sustained[(area, session)].append(transformed_activity)
            elif respRange == "offset":
                explained_vars_offset[(area, session)].append(variances)
                transformed_act_offset[(area, session)].append(transformed_activity)

#%% Plotting skree plots with participation ratios
for stim in stim_types:
    fig_skree = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
            [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
            [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
        ],
        subplot_titles=[
            f"Dorsal - onset", f"Dorsal - sustained", f"Dorsal - offset",
            f"Primary - onset", f"Primary - sustained", f"Primary - offset",
            f"Ventral - onset", f"Ventral - sustained", f"Ventral - offset",
        ],
        horizontal_spacing=0.08,  # Adjust spacing between columns
        vertical_spacing=0.08  # Adjust spacing between rows
    )

    fig_skree.update_layout(
        title=f"{stim} Stimulus - Skree Plot {neuron_threshold} Neurons",
    )

    for respRange in response_ranges:
        plot_col = response_map[respRange]
        if respRange == "onset":
            explained_vars = explained_vars_onset
        elif respRange == "sustained":
            explained_vars = explained_vars_sustained
        elif respRange == "offset":
            explained_vars = explained_vars_offset
        dorsal_data = []
        primary_data = []
        ventral_data = []

        for area, session in session_list.index:
            sesh_vars = explained_vars[(area, session)][plot_col - 1]
            if area == "Dorsal auditory area":
                dorsal_data.append(sesh_vars)
            elif area == "Primary auditory area":
                primary_data.append(sesh_vars)
            elif area == "Ventral auditory area":
                ventral_data.append(sesh_vars)

        dorsal = np.column_stack(dorsal_data)
        primary = np.column_stack(primary_data)
        ventral = np.column_stack(ventral_data)
        dorsal_mean = dorsal.mean(axis=1)
        primary_mean = primary.mean(axis=1)
        ventral_mean = ventral.mean(axis=1)
        dorsal_sem = dorsal.std(axis=1) / np.sqrt(dorsal.shape[0])
        primary_sem = primary.std(axis=1) / np.sqrt(primary.shape[0])
        ventral_sem = ventral.std(axis=1) / np.sqrt(ventral.shape[0])

        dorsal_pr_array = np.sum(dorsal, axis=0)**2 / np.sum(dorsal**2, axis=0)
        primary_pr_array = np.sum(primary, axis=0)**2 / np.sum(primary**2, axis=0)
        ventral_pr_array = np.sum(ventral, axis=0)**2 / np.sum(ventral**2, axis=0)
        dorsal_pr_mean = dorsal_pr_array.mean()
        primary_pr_mean = primary_pr_array.mean()
        ventral_pr_mean = ventral_pr_array.mean()
        dorsal_pr_sem = dorsal_pr_array.std() / np.sqrt(dorsal_pr_array.shape[0])
        primary_pr_sem = primary_pr_array.std() / np.sqrt(primary_pr_array.shape[0])
        ventral_pr_sem = ventral_pr_array.std() / np.sqrt(ventral_pr_array.shape[0])

        fig_skree.add_trace(go.Scatter(
            x=np.arange(0, 12, 1),  # Plotting first 12 PCs
            y=dorsal_mean,
            error_y=dict(type='data', array=dorsal_sem, visible=True),
            name=f"Dorsal",
            legendgroup="Dorsal",
            marker_color="blue",
            mode="lines+markers",
                ),
            row=1, col=plot_col)

        fig_skree.add_annotation(
            text=f" PR = {dorsal_pr_mean:.2f} +/- {dorsal_pr_sem:.2f}",  # Text for the annotation
            #row=1, col=plot_col,
            xref=f"x{plot_col}",  # Handle multiple subplots for x-axis
            yref=f"y{1}",  # Handle multiple subplots for y-axis
            x=0.95,  # Position (relative, 0 is left, 1 is right)
            y=0.95,  # Position (relative, 0 is bottom, 1 is top)
            #xanchor="right",  # Anchors text to the right
            #yanchor="top",  # Anchors text to the top
            font=dict(size=12, color="black"),  # Customize font size and color
            showarrow=False  # Hide the arrow
        )

        fig_skree.add_trace(go.Scatter(
            x=np.arange(0, 12, 1),  # Plotting first 12 PCs
            y=primary_mean,
            error_y=dict(type='data', array=primary_sem, visible=True),
            name=f"Primary, PR = {primary_pr_mean:.2f} +/- {primary_pr_sem:.2f}",
            legendgroup="Primary",
            marker_color="black",
            mode="lines+markers",
                ),
            row=2, col=plot_col)

        fig_skree.add_trace(go.Scatter(
            x=np.arange(0, 12, 1),  # Plotting first 12 PCs
            y=ventral_mean,
            error_y=dict(type='data', array=ventral_sem, visible=True),
            name=f"Ventral, PR = {ventral_pr_mean:.2f} +/- {ventral_pr_sem:.2f}",
            legendgroup="Ventral",
            marker_color="orange",
            mode="lines+markers",
                ),
            row=3, col=plot_col)

        fig_skree.update_layout(showlegend=False)
    fig_skree.write_html(f"{figdataPath}/PCA_and_PR_plots/skree_{stim}.html")


#%% Plotting session PCAs
# TODO: Finsh adding code to plot all the PCA plots in the fig defined above. Then use the saved variances
#  to calculate participation ratio and plot skree plots with error around the means of each bar and PR +-
#  error inlayed as text


fig_all_trials = make_subplots(
    rows=3,
    cols=3,
    specs=[
        [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
        [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
        [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
    ],
    subplot_titles=[
        f"Dorsal - onset", f"Dorsal - sustained", f"Dorsal - offset",
        f"Primary - onset", f"Primary - sustained", f"Primary - offset",
        f"Ventral - onset", f"Ventral - sustained", f"Ventral - offset",
    ],
    horizontal_spacing=0.08,  # Adjust spacing between columns
    vertical_spacing=0.08  # Adjust spacing between rows
)

fig_all_trials.update_layout(
    title=f"{stim} Stimulus - PC 2 vs PC 1",
)

plot_row = brain_plot_map[area]
plot_col = response_map[respRange]

scatter_plot = go.Scatter(
                x=transformed_activity[:, 0],  # First principal component
                y=transformed_activity[:, 1],  # Second principal component
                mode='markers',  # Use markers for a scatter plot
                marker=dict(size=5,
                            color=sorted_stim_array,
                            colorscale='Viridis',
                            opacity=0.7,
                            colorbar=dict(title="Stim Vals")),  # Marker size, color, and transparency
                # name="First 2 PCs"  # Legend entry
            )

fig_all_trials.add_trace(scatter_plot, row=plot_row, col=plot_col)

fig_all_trials.update_layout(showlegend=False)
fig_all_trials.write_html(f"{figdataPath}/PCA_and_PR_plots/PCA_{stim}.html")