import os
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

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
for stim_ind, stim in enumerate(stim_types):
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

    fig_PR = make_subplots(
        rows=3,
        cols=1,
        specs=[
            [{"colspan": 1}],
            [{"colspan": 1}],
            [{"colspan": 1}],
        ],
        subplot_titles=[
            f"Dorsal",
            f"Primary",
            f"Ventral",
        ],
        vertical_spacing=0.18  # Adjust spacing between rows
    )

    fig_skree.update_layout(
        title=f"{stim} Stimulus - Skree Plot {neuron_threshold} Neurons",
    )

    dorsal_x = []
    dorsal_points = []
    dorsal_means = []
    dorsal_stats = []
    primary_x = []
    primary_points = []
    primary_means = []
    primary_stats = []
    ventral_x = []
    ventral_points = []
    ventral_means = []
    ventral_stats = []

    for respRange in response_ranges:
        plot_col = response_map[respRange]
        if respRange == "onset":
            explained_vars = explained_vars_onset
            pr_color = "blue"
            pr_point = 0
        elif respRange == "sustained":
            explained_vars = explained_vars_sustained
            pr_color = "black"
            pr_point = 1
        elif respRange == "offset":
            explained_vars = explained_vars_offset
            pr_color = "orange"
            pr_point = 2
        dorsal_data = []
        primary_data = []
        ventral_data = []

        for area, session in session_list.index:
            sesh_vars = explained_vars[(area, session)][stim_ind]
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

        dorsal_x.append(np.repeat(pr_point, len(dorsal_pr_array)))
        dorsal_points.append(dorsal_pr_array)
        dorsal_means.append(dorsal_pr_mean)
        primary_x.append(np.repeat(pr_point, len(primary_pr_array)))
        primary_points.append(primary_pr_array)
        primary_means.append(primary_pr_mean)
        ventral_x.append(np.repeat(pr_point, len(ventral_pr_array)))
        ventral_points.append(ventral_pr_array)
        ventral_means.append(ventral_pr_mean)

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
            row=1, col=plot_col,
            # xref=f"x{plot_col}",  # Handle multiple subplots for x-axis
            # yref=f"y{1}",  # Handle multiple subplots for y-axis
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
            name=f"Primary",
            legendgroup="Primary",
            marker_color="black",
            mode="lines+markers",
                ),
            row=2, col=plot_col)

        fig_skree.add_annotation(
            text=f" PR = {primary_pr_mean:.2f} +/- {primary_pr_sem:.2f}",  # Text for the annotation
            row=2, col=plot_col,
            # xref=f"x{plot_col}",  # Handle multiple subplots for x-axis
            # yref=f"y{2}",  # Handle multiple subplots for y-axis
            x=0.95,  # Position (relative, 0 is left, 1 is right)
            y=0.95,  # Position (relative, 0 is bottom, 1 is top)
            # xanchor="right",  # Anchors text to the right
            # yanchor="top",  # Anchors text to the top
            font=dict(size=12, color="black"),  # Customize font size and color
            showarrow=False  # Hide the arrow
        )

        fig_skree.add_trace(go.Scatter(
            x=np.arange(0, 12, 1),  # Plotting first 12 PCs
            y=ventral_mean,
            error_y=dict(type='data', array=ventral_sem, visible=True),
            name=f"Ventral, PR = {ventral_pr_mean:.2f} +/- {ventral_pr_sem:.2f}",
            legendgroup="Ventral",
            marker_color="orange",
            mode="lines+markers",
                ),
            row=3, col=plot_col,)

        fig_skree.add_annotation(
            text=f" PR = {ventral_pr_mean:.2f} +/- {ventral_pr_sem:.2f}",  # Text for the annotation
            row=3, col=plot_col,
            # xref=f"x{plot_col}",  # Handle multiple subplots for x-axis
            # yref=f"y{3}",  # Handle multiple subplots for y-axis
            x=0.95,  # Position (relative, 0 is left, 1 is right)
            y=0.95,  # Position (relative, 0 is bottom, 1 is top)
            # xanchor="right",  # Anchors text to the right
            # yanchor="top",  # Anchors text to the top
            font=dict(size=12, color="black"),  # Customize font size and color
            showarrow=False  # Hide the arrow
        )

        fig_skree.update_layout(showlegend=False)

    y_offset = 2  # Initial offset for the brackets
    p_threshold = 0.05/len(response_ranges)  # Bonferonni correction
    for i in range(len(response_ranges)):
        dorsal_stat1 = dorsal_points[i]
        primary_stat1 = primary_points[i]
        ventral_stat1 = ventral_points[i]
        for j in range(1, len(response_ranges)):
            dorsal_stat2 = dorsal_points[j]
            primary_stat2 = primary_points[j]
            ventral_stat2 = ventral_points[j]
            if i != j and j > i:
                bracket_y = 20 + y_offset
                dorsal_pval = stats.mannwhitneyu(dorsal_stat1, dorsal_stat2, alternative='two-sided').pvalue
                primary_pval = stats.mannwhitneyu(primary_stat1, primary_stat2, alternative='two-sided').pvalue
                ventral_pval = stats.mannwhitneyu(ventral_stat1, ventral_stat2, alternative='two-sided').pvalue
                dorsal_stats.append(dorsal_pval)
                primary_stats.append(primary_pval)
                ventral_stats.append(ventral_pval)

                sig_label_primary = '*' if primary_pval < p_threshold else 'ns'
                sig_label_dorsal = '*' if dorsal_pval < p_threshold else 'ns'
                sig_label_ventral = '*' if ventral_pval < p_threshold else 'ns'

                fig_PR.add_shape(
                    type="line",
                    x0=i,  # Start near one box
                    x1=j,  # End near the other box
                    y0=bracket_y,
                    y1=bracket_y,
                    xref=f"x{1}",  # column
                    yref=f"y{1}",  # row
                    line=dict(color="black", width=1),
                )
                fig_PR.add_shape(
                    type="line",
                    x0=i,
                    x1=i,  # Vertical tick for the first group
                    y0=bracket_y,
                    y1=bracket_y - 1,
                    xref=f"x{1}",
                    yref=f"y{1}",
                    line=dict(color="black", width=1),
                )
                fig_PR.add_shape(
                    type="line",
                    x0=j,
                    x1=j,  # Vertical tick for the second group
                    y0=bracket_y,
                    y1=bracket_y - 1,
                    xref=f"x{1}",
                    yref=f"y{1}",
                    line=dict(color="black", width=1),
                )

                # Add text label (* or ns) above the bracket
                fig_PR.add_annotation(
                    x=(i + j) / 2,  # Midpoint of the bracket
                    y=bracket_y + 0.1,  # Slightly above the bracket
                    text=sig_label_dorsal,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xref=f"x{1}",
                    yref=f"y{1}",
                )

                fig_PR.add_shape(
                    type="line",
                    x0=i,  # Start near one box
                    x1=j,  # End near the other box
                    y0=bracket_y,
                    y1=bracket_y,
                    xref=f"x{1}",  # column
                    yref=f"y{2}",  # row
                    line=dict(color="black", width=1),
                )
                fig_PR.add_shape(
                    type="line",
                    x0=i,
                    x1=i,  # Vertical tick for the first group
                    y0=bracket_y,
                    y1=bracket_y - 1,
                    xref=f"x{1}",
                    yref=f"y{2}",
                    line=dict(color="black", width=1),
                )
                fig_PR.add_shape(
                    type="line",
                    x0=j,
                    x1=j,  # Vertical tick for the second group
                    y0=bracket_y,
                    y1=bracket_y - 1,
                    xref=f"x{1}",
                    yref=f"y{2}",
                    line=dict(color="black", width=1),
                )

                # Add text label (* or ns) above the bracket
                fig_PR.add_annotation(
                    x=(i + j) / 2,  # Midpoint of the bracket
                    y=bracket_y + 0.1,  # Slightly above the bracket
                    text=sig_label_primary,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xref=f"x{1}",
                    yref=f"y{2}",
                )

                fig_PR.add_shape(
                    type="line",
                    x0=i,  # Start near one box
                    x1=j,  # End near the other box
                    y0=bracket_y,
                    y1=bracket_y,
                    xref=f"x{1}",  # column
                    yref=f"y{3}",  # row
                    line=dict(color="black", width=1),
                )
                fig_PR.add_shape(
                    type="line",
                    x0=i,
                    x1=i,  # Vertical tick for the first group
                    y0=bracket_y,
                    y1=bracket_y - 1,
                    xref=f"x{1}",
                    yref=f"y{3}",
                    line=dict(color="black", width=1),
                )
                fig_PR.add_shape(
                    type="line",
                    x0=j,
                    x1=j,  # Vertical tick for the second group
                    y0=bracket_y,
                    y1=bracket_y - 1,
                    xref=f"x{1}",
                    yref=f"y{3}",
                    line=dict(color="black", width=1),
                )

                # Add text label (* or ns) above the bracket
                fig_PR.add_annotation(
                    x=(i + j) / 2,  # Midpoint of the bracket
                    y=bracket_y + 0.1,  # Slightly above the bracket
                    text=sig_label_ventral,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xref=f"x{1}",
                    yref=f"y{3}",
                )
                y_offset += 2  # Increment the offset for the next bracket to avoid overlap



    dorsal_x = np.concatenate(dorsal_x)
    dorsal_points = np.concatenate(dorsal_points)
    primary_x = np.concatenate(primary_x)
    primary_points = np.concatenate(primary_points)
    ventral_x = np.concatenate(ventral_x)
    ventral_points = np.concatenate(ventral_points)

    fig_PR.add_trace(go.Scatter(
        x=dorsal_x,
        y=dorsal_points,
        name=f"{respRange} PR",
        legendgroup=f"{respRange} PR",
        marker_color=pr_color,
        mode="markers",
        opacity=0.5,
    ), row=1, col=1, )

    fig_PR.add_trace(go.Scatter(
        x=np.arange(0, 3, 1),
        y=dorsal_means,
        name=f"{respRange} PR - Mean",
        legendgroup=f"{respRange} PR",
        mode="lines+markers",
        marker_color=pr_color,
    ), row=1, col=1, )

    fig_PR.add_trace(go.Scatter(
        x=primary_x,
        y=primary_points,
        name=f"{respRange} PR",
        legendgroup=f"{respRange} PR",
        marker_color=pr_color,
        mode="markers",
        opacity=0.5,
        showlegend=False,
    ), row=2, col=1, )

    fig_PR.add_trace(go.Scatter(
        x=np.arange(0, 3, 1),
        y=primary_means,
        name=f"{respRange} PR",
        legendgroup=f"{respRange} PR",
        mode="lines+markers",
        marker_color=pr_color,
        showlegend=False,
    ), row=2, col=1, )

    fig_PR.add_trace(go.Scatter(
        x=ventral_x,
        y=ventral_points,
        name=f"{respRange} PR",
        legendgroup=f"{respRange} PR",
        marker_color=pr_color,
        mode="markers",
        opacity=0.5,
        showlegend=False,
    ), row=3, col=1, )

    fig_PR.add_trace(go.Scatter(
        x=np.arange(0, 3, 1),
        y=ventral_means,
        name=f"{respRange} PR",
        legendgroup=f"{respRange} PR",
        mode="lines+markers",
        marker_color=pr_color,
        showlegend=False,
    ), row=3, col=1, )

    fig_PR.update_xaxes(title_text="Response", range=[-0.2, 2.2],
                        tickmode='array',
                        tickvals=[0, 1, 2],
                        ticktext=response_ranges,
                        )
    fig_PR.update_yaxes(title_text="Participation Ratio", range=[0, 30])
    fig_PR.update_layout(title=f"{stim} Stimulus - Participation Ratio,  {neuron_threshold} Neurons",
                         showlegend=False,)

    fig_PR.write_html(f"{figdataPath}/PCA_and_PR_plots/PR_{stim}.html")
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