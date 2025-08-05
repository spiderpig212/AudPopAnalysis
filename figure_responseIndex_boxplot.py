import os
import numpy as np

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
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
figdataPath = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)

dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath,f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)
simpleSiteNames = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])#.value_counts()
simpleSiteNames.name = 'simpleSiteName'
celldb = pd.concat([celldb, simpleSiteNames], axis=1)

areas_of_interest = [
    "Dorsal auditory area",
    "Posterior auditory area",
    "Primary auditory area",
    "Ventral auditory area"
]

aud_db = celldb[celldb['simpleSiteName'].isin(areas_of_interest)].reset_index()

# plotting_db = aud_db[['simpleSiteName', 'subject', 'date', 'ptFRIndexOnset', 'ptFRIndexSustain', 'ptFRIndexOffset',
#                      'amFRIndexOnset', 'amFRIndexSustain', 'amFRIndexOffset', 'nsFRIndexOnset', 'nsFRIndexSustain',
#                      'nsFRIndexOffset']]

stimulus_types = ['Pure Tones', 'AM', 'Natural Sounds']

# ---------------------------------------------------- Onset ----------------------------------------------
column_mapping = {
    'Pure Tones': 'ptFiringRateBestOnset',
    'AM': 'amFiringRateBestOnset',
    'Natural Sounds': 'nsFiringRateBestOnset',
}

colors = {
    'Dorsal auditory area': '#1f77b4',  # Blue
    'Posterior auditory area': '#ff7f0e',     # Orange
    'Primary auditory area': '#2ca02c',        # Green
    'Ventral auditory area': '#d62728'         # Red
}

fig = go.Figure()

# Loop through each area of interest and stimulus type to create box plots
for i, stimulus in enumerate(stimulus_types):
    # Loop over each brain area in areas_of_interest
    for i_new, area in enumerate(areas_of_interest):
        # Filter dataframe by area
        filtered_data = aud_db[aud_db['simpleSiteName'] == area][column_mapping[stimulus]]

        # Add box plot for this area and stimulus
        fig.add_trace(go.Box(
            y=filtered_data,
            x0=stimulus,
            name=area,  # All box plots will group by the "name"
            legendgroup=area,  # To group the same area in the legend
            marker_color=colors[area],  # Assign the color
            boxmean='sd',  # Show the mean and standard deviation
            offsetgroup=i+i_new,  # Offset to group box plots for each stimulus type
            showlegend=(i == 0)  # Only show legend once for each area
        ))

# Update layout
fig.update_layout(
    title='Onset Responses',
    xaxis=dict(
        title='Stimulus Type',
        tickvals=list(range(len(stimulus_types))),  # Position ticks
        ticktext=stimulus_types,  # Add corresponding x-axis labels
    ),
    yaxis=dict(
        title='Best FR',
    ),
    boxmode='group',  # Group plots by x-axis labels
    template='plotly_white',  # Clean background style
)

file_path = os.path.join(figdataPath, f'boxplot_summaries/')
fig.write_html(f"{file_path}onset_Responses_best_responses.html")
fig.show()

# ------------------------------------------- Sustain ----------------------------------------------------------
column_mapping = {
    'Pure Tones': 'ptFiringRateBestSustain',
    'AM': 'amFiringRateBestSustain',
    'Natural Sounds': 'nsFiringRateBestSustain',
}

colors = {
    'Dorsal auditory area': '#1f77b4',  # Blue
    'Posterior auditory area': '#ff7f0e',     # Orange
    'Primary auditory area': '#2ca02c',        # Green
    'Ventral auditory area': '#d62728'         # Red
}

fig = go.Figure()

# Loop through each area of interest and stimulus type to create box plots
for i, stimulus in enumerate(stimulus_types):
    # Loop over each brain area in areas_of_interest
    for i_new, area in enumerate(areas_of_interest):
        # Filter dataframe by area
        filtered_data = aud_db[aud_db['simpleSiteName'] == area][column_mapping[stimulus]]

        # Add box plot for this area and stimulus
        fig.add_trace(go.Box(
            y=filtered_data,
            x0=stimulus,
            name=area,  # All box plots will group by the "name"
            legendgroup=area,  # To group the same area in the legend
            marker_color=colors[area],  # Assign the color
            boxmean='sd',  # Show the mean and standard deviation
            offsetgroup=i+i_new,  # Offset to group box plots for each stimulus type
            showlegend=(i == 0)  # Only show legend once for each area
        ))

# Update layout
fig.update_layout(
    title='Sustain Responses',
    xaxis=dict(
        title='Stimulus Type',
        tickvals=list(range(len(stimulus_types))),  # Position ticks
        ticktext=stimulus_types,  # Add corresponding x-axis labels
    ),
    yaxis=dict(
        title='Best FR',
    ),
    boxmode='group',  # Group plots by x-axis labels
    template='plotly_white',  # Clean background style
)

fig.write_html(f"{file_path}sustain_Responses_best_responses.html")
fig.show()

# -------------------------------------------------------- Offset ---------------------------------------------
column_mapping = {
    'Pure Tones': 'ptFiringRateBestOffset',
    'AM': 'amFiringRateBestOffset',
    'Natural Sounds': 'nsFiringRateBestOffset',
}

colors = {
    'Dorsal auditory area': '#1f77b4',  # Blue
    'Posterior auditory area': '#ff7f0e',     # Orange
    'Primary auditory area': '#2ca02c',        # Green
    'Ventral auditory area': '#d62728'         # Red
}

fig = go.Figure()

# Loop through each area of interest and stimulus type to create box plots
for i, stimulus in enumerate(stimulus_types):
    # Loop over each brain area in areas_of_interest
    for i_new, area in enumerate(areas_of_interest):
        # Filter dataframe by area
        filtered_data = aud_db[aud_db['simpleSiteName'] == area][column_mapping[stimulus]]

        # Add box plot for this area and stimulus
        fig.add_trace(go.Box(
            y=filtered_data,
            x0=stimulus,
            name=area,  # All box plots will group by the "name"
            legendgroup=area,  # To group the same area in the legend
            marker_color=colors[area],  # Assign the color
            boxmean='sd',  # Show the mean and standard deviation
            offsetgroup=i+i_new,  # Offset to group box plots for each stimulus type
            showlegend=(i == 0)  # Only show legend once for each area
        ))

# Update layout
fig.update_layout(
    title='Offset Responses',
    xaxis=dict(
        title='Stimulus Type',
        tickvals=list(range(len(stimulus_types))),  # Position ticks
        ticktext=stimulus_types,  # Add corresponding x-axis labels
    ),
    yaxis=dict(
        title='Best FR',
    ),
    boxmode='group',  # Group plots by x-axis labels
    template='plotly_white',  # Clean background style
)

fig.write_html(f"{file_path}offset_Responses_best_response.html")
fig.show()


