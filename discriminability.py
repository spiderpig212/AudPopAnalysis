import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.colors as pcolors
import funcs
from jaratoolbox import settings
import studyparams

import os

file_path = settings.FIGURES_DATA_PATH + "/" + studyparams.STUDY_NAME
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]

projection_plots = False

boxplot_data = {}

colors = {
    'Dorsal auditory area': '#1f77b4',  # Blue
    'Posterior auditory area': '#ff7f0e',     # Orange
    'Primary auditory area': '#2ca02c',        # Green
    'Ventral auditory area': '#d62728'         # Red
}

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
                    J, w = funcs.calc_fisher_criterion_Christian(resp_mask, resp_mask2)
                    dprime_stim_vals[i1, i2] = J
                    if projection_plots:
                        proj_fig = funcs.plot_scatter_and_histogram(resp_mask, resp_mask2, w)
                        # proj_fig.legend(loc='best')
                        proj_fig.savefig(f"{file_path}/fisher_projections/FCriterion_{brainRegion}_{stimType}_{stimType2}_{respRange}.png")
                        # proj_fig.show()
                        plt.close(proj_fig)


            upper_indices = np.triu_indices(dprime_stim_vals.shape[0], k=1)
            upper_tri_values = dprime_stim_vals[upper_indices]

            boxplot_data[f"{brainRegion}_{stim}_{respRange}"] = upper_tri_values.flatten()


            fig = go.Figure(data=go.Heatmap(
                z=dprime_stim_vals,
                colorscale=pcolors.sequential.Viridis  # You can change the color scale as needed
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


# -------------------------- Plotting by response type to compare the same response across regions -------------------
for respRange in response_ranges:
    fig_boxplot = go.Figure()
    for i, stimulus in enumerate(stim_types):
        region_data = {}
        for i_new, area in enumerate(uniqRegions):
            data = boxplot_data[f"{area}_{stimulus}_{respRange}"]
            region_data[area] = [data, stimulus]

            fig_boxplot.add_trace(go.Box(
                y=data,
                x0=stimulus,
                name=area,
                legendgroup=area,
                marker_color=colors[area],
                boxmean='sd',
                offsetgroup=i+i_new,
                showlegend=(i == 0),
            ))

        regions = list(region_data.keys())
        y_offset = 0.1  # Initial offset for the brackets
        x_offsets = [-0.20, -0.1, 0.1, 0.20]
        for j in range(len(regions)):
            for k in range(j + 1, len(regions)):
                [data1, stat_stim] = region_data[regions[j]]
                [data2, stat_stim] = region_data[regions[k]]

                # Perform the Wilcoxon rank-sum test
                stat, p_value = stats.ranksums(data1, data2)

                # Determine significance label
                significance_label = '*' if p_value < 0.05/len(regions) else 'ns'

                # Calculate position for the bracket
                max_y = max(max(data1), max(data2))
                bracket_y = max_y + y_offset

                # Add a bracket (horizontal line) with vertical ticks
                fig_boxplot.add_shape(
                    type="line",
                    x0=i + x_offsets[j],  # Start near one box
                    x1=i + x_offsets[k],  # End near the other box
                    y0=bracket_y,
                    y1=bracket_y,
                    line=dict(color="black", width=1),
                )
                fig_boxplot.add_shape(
                    type="line",
                    x0=i + x_offsets[j],
                    x1=i + x_offsets[j],  # Vertical tick for the first group
                    y0=bracket_y,
                    y1=bracket_y - 0.01,
                    line=dict(color="black", width=1),
                )
                fig_boxplot.add_shape(
                    type="line",
                    x0=i + x_offsets[k],
                    x1=i + x_offsets[k],  # Vertical tick for the second group
                    y0=bracket_y,
                    y1=bracket_y - 0.01,
                    line=dict(color="black", width=1),
                )

                # Add text label (* or ns) above the bracket
                fig_boxplot.add_annotation(
                    x=(i + x_offsets[j] + i + x_offsets[k]) / 2,  # Midpoint of the bracket
                    y=bracket_y + 0.02,  # Slightly above the bracket
                    text=significance_label,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                )
                y_offset += 0.05  # Increment the offset for the next bracket to avoid overlap

    fig_boxplot.update_layout(
        title=f'{respRange} Responses',
        xaxis=dict(
            title='Stimulus Type',
            tickvals=list(range(len(stim_types))),  # Position ticks
            ticktext=stim_types,  # Add corresponding x-axis labels
        ),
        yaxis=dict(
            title='Fisher Criterion',
        ),
        boxmode='group',  # Group plots by x-axis labels
        template='plotly_white',  # Clean background style
    )
    # for annotation in significance_annotations:
    #     fig_boxplot.add_annotation(annotation)

    file_path_save = os.path.join(file_path, f'boxplot_summaries/')
    fig_boxplot.write_html(f"{file_path_save}{respRange}_FisherCriterion_boxplot_.html")
    fig_boxplot.show()

# -------------- Plotting each region individually to compare response types within region ---------------
colors_resp = {
    'onset': '#1f77b4',  # Blue
    'sustained': '#ff7f0e',     # Orange
    'offset': '#2ca02c',        # Green
}

for area in uniqRegions:
    fig_boxplot_brain = go.Figure()
    for i, stimulus in enumerate(stim_types):
        response_data = {}
        for i_new, respRange in enumerate(response_ranges):
            data = boxplot_data[f"{area}_{stimulus}_{respRange}"]
            response_data[respRange] = [data, stimulus]
            fig_boxplot_brain.add_trace(go.Box(
                y=data,
                x0=stimulus,
                name=respRange,
                legendgroup=respRange,
                marker_color=colors_resp[respRange],
                boxmean='sd',
                offsetgroup=i+i_new,
                showlegend=(i == 0),
            ))

        respRanges = list(response_data.keys())
        y_offset = 0.03  # Initial offset for the brackets
        x_offsets = [-0.15, 0, 0.15]
        for j in range(len(respRanges)):
            for k in range(j + 1, len(respRanges)):
                [data1, stat_stim] = response_data[respRanges[j]]
                [data2, stat_stim] = response_data[respRanges[k]]

                # Perform the Wilcoxon rank-sum test
                stat, p_value = stats.ranksums(data1, data2)

                # Determine significance label
                significance_label = '*' if p_value < 0.05/len(respRanges) else 'ns'

                # Calculate position for the bracket
                max_y = max(max(data1), max(data2))
                bracket_y = max_y + y_offset

                # Add a bracket (horizontal line) with vertical ticks
                fig_boxplot_brain.add_shape(
                    type="line",
                    x0=i + x_offsets[j],  # Start near one box
                    x1=i + x_offsets[k],  # End near the other box
                    y0=bracket_y,
                    y1=bracket_y,
                    line=dict(color="black", width=1),
                )
                fig_boxplot_brain.add_shape(
                    type="line",
                    x0=i + x_offsets[j],
                    x1=i + x_offsets[j],  # Vertical tick for the first group
                    y0=bracket_y,
                    y1=bracket_y - 0.01,
                    line=dict(color="black", width=1),
                )
                fig_boxplot_brain.add_shape(
                    type="line",
                    x0=i + x_offsets[k],
                    x1=i + x_offsets[k],  # Vertical tick for the second group
                    y0=bracket_y,
                    y1=bracket_y - 0.01,
                    line=dict(color="black", width=1),
                )

                # Add text label (* or ns) above the bracket
                fig_boxplot_brain.add_annotation(
                    x=(i + x_offsets[j] + i + x_offsets[k]) / 2,  # Midpoint of the bracket
                    y=bracket_y + 0.01,  # Slightly above the bracket
                    text=significance_label,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                )
                y_offset += 0.05  # Increment the offset for the next bracket to avoid overlap

    fig_boxplot_brain.update_layout(
        title=f'{area} Responses',
        xaxis=dict(
            title='Stimulus Type',
            tickvals=list(range(len(stim_types))),  # Position ticks
            ticktext=stim_types,  # Add corresponding x-axis labels
        ),
        yaxis=dict(
            title='Fisher Criterion',
        ),
        boxmode='group',  # Group plots by x-axis labels
        template='plotly_white',  # Clean background style
    )

    file_path_save = os.path.join(file_path, f'boxplot_summaries/')
    fig_boxplot_brain.write_html(f"{file_path_save}{area}_FisherCriterion_boxplot.html")
    fig_boxplot_brain.show()
