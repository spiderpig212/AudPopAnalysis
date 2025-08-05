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
dbCoordsFilename = os.path.join(dbPath,f'celldb_{studyparams.STUDY_NAME}_coords.h5')
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
counts_per_area = celldb[celldb['simpleSiteName'].isin(areas_of_interest)]['simpleSiteName'].value_counts()

print(counts_per_area)

cells_to_plot_per_session = 1
unique_mice = celldb.subject.unique()
for mouse in unique_mice:
    mouse_db = celldb.query('subject==@mouse')
    unique_sessions = mouse_db.date.unique()
    for session in unique_sessions:
        session_db = mouse_db.query('date==@session')
        aud_db = session_db[session_db['simpleSiteName'].isin(areas_of_interest)].reset_index()

        fig = make_subplots(
            rows=6,
            cols=2,
            specs=[
                [{"colspan": 1}, {"colspan": 1}],  # Row 1: Two half-width plots
                [{"colspan": 1}, {"colspan": 1}],  # Row 2: Two half-width plots
                [{"colspan": 1}, {"colspan": 1}],  # Row 3: Two half-width plots
                [{"colspan": 1}, {"colspan": 1}],  # Row 4: Two half-width plots
                [{"colspan": 1}, {"colspan": 1}],  # Row 5: Two half-width plots
                [{"colspan": 2}, None],  # Row 6: One full-width plot
            ],
            subplot_titles=[
                f"Pure Tones Raster", f"Pure Tones PSTH",
                f"AM Raster", f"AM PSTH",
                f"Natural sounds Raster", f"Natural sounds {studyparams.SOUND_CATEGORIES[0]} PSTH",
                f"Natural sounds {studyparams.SOUND_CATEGORIES[1]} PSTH", f"Natural sounds {studyparams.SOUND_CATEGORIES[2]} PSTH",
                f"Natural sounds {studyparams.SOUND_CATEGORIES[3]} PSTH", f"Natural sounds {studyparams.SOUND_CATEGORIES[4]} PSTH",
                f"Natural sounds PSTH - All"
            ],
            horizontal_spacing=0.05,  # Adjust spacing between columns
            vertical_spacing=0.05  # Adjust spacing between rows
        )

        try:
            someCells = aud_db.sample(n=cells_to_plot_per_session).index.tolist()
        except ValueError:
            print(f"Not enough cells to plot for {mouse} {session} within target brain regions")
            continue
        ensemble = ephyscore.CellEnsemble(aud_db)
        for CASE in [0, 1, 2]:
            if CASE == 2:
                stimAbv = 'ns'
                stimType = 'naturalSound'
                stimVar = 'soundID'
                timeRange = [-2, 6]  # In seconds
                allPeriods = [[-1, 0], [0, 0.5], [1, 4], [4, 4.5]]
                nCategories = len(studyparams.SOUND_CATEGORIES)
            elif CASE == 1:
                stimABV = 'am'
                stimType = 'AM'
                stimVar = 'currentFreq'
                timeRange = [-0.5, 1.5]
                allPeriods = [[-0.5, 0], [0, 0.2], [0.2, 0.5], [0.5, 0.7]]
                nCategories = 11
            elif CASE == 0:
                stimAbv = 'pt'
                stimType = 'pureTones'
                stimVar = 'currentFreq'
                timeRange = [-0.1, 0.3]
                allPeriods = [[-0.1, 0], [0, 0.05], [0.05, 0.1], [0.1, 0.15]]
                nCategories = 16

            periodsName = ['base', 'respOnset', 'respSustained', 'respOffset']
            periodDuration = [x[1] - x[0] for x in allPeriods]
            ephysData, bdata = ensemble.load(stimType)
            currentStim = bdata[stimVar]

            nTrials = len(currentStim)
            eventOnsetTimes = ephysData['events']['stimOn'][:nTrials] # Ignore trials not in bdata

            spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
                ensemble.eventlocked_spiketimes(eventOnsetTimes, timeRange)

            binWidth = 0.01
            binEdges = np.arange(timeRange[0], timeRange[-1] + binWidth, binWidth)
            spikeCountMatAll = ensemble.spiketimes_to_spikecounts(binEdges=binEdges)

            possibleStim = np.unique(currentStim)
            trialsEachCond = behavioranalysis.find_trials_each_type(currentStim, possibleStim)
            condEachSortedTrial, sortedTrials = np.nonzero(trialsEachCond.T)
            sortingInds = np.argsort(sortedTrials)

            if CASE == 0:  # Pure tones
                pColors = colors.sequential.Viridis
                pRaster = funcs.plot_rasters(aud_db, sortingInds, trialIndexForEachSpikeAll, spikeTimesFromEventOnsetAll,
                                     rows=1, cols=1, specificInds=someCells, subplot_titles=None, plot=False, random=False)
                for trace in pRaster.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=1, col=1)
                fig.add_vline(x=0, line_width=1, line_color="red", row=1, col=1)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=1, col=1)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=1, col=1)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=1, col=1)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCond, rows=1, cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=possibleStim, random=False)
                for trace in ppSTH.data:
                    trace.legend = 'legend'
                    fig.add_trace(trace, row=1, col=2)
                fig.add_vline(x=0, line_width=1, line_color="red", row=1, col=2)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=1, col=2)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=1, col=2)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=1, col=2)

                fig.update_layout(
                    height=2400,
                    width=1000,
                    title_text=f"Mouse: {mouse} Session: {session}, Region: {aud_db.iloc[someCells]['simpleSiteName'].values[0]}",
                    legend=dict(
                        title=dict(
                            text="PT"
                        ),
                        font=dict(size=12),
                        xref="container",
                        yref="container",
                        y=0.96,
                        bgcolor="White")
                )

            if CASE == 1:  # AM
                pColors = colors.sequential.Viridis
                pRaster = funcs.plot_rasters(aud_db, sortingInds, trialIndexForEachSpikeAll,
                                             spikeTimesFromEventOnsetAll,
                                             rows=1, cols=1, specificInds=someCells, subplot_titles=None,
                                             plot=False, random=False)
                for trace in pRaster.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=2, col=1)
                fig.add_vline(x=0, line_width=1, line_color="red", row=2, col=1)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=2, col=1)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=2, col=1)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=2, col=1)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCond, rows=1,
                                        cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=possibleStim, random=False)

                for trace in ppSTH.data:
                    trace.legend = 'legend2'
                    fig.add_trace(trace, row=2, col=2)
                fig.add_vline(x=0, line_width=1, line_color="red", row=2, col=2)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=2, col=2)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=2, col=2)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=2, col=2)


            if CASE == 2:  # Natural Sounds
                pColors = colors.sequential.Viridis_r
                repeat_sounds = np.repeat(np.array(studyparams.SOUND_CATEGORIES), 4)  # 4 instances of each sound
                pRaster = funcs.plot_rasters(aud_db, sortingInds, trialIndexForEachSpikeAll,
                                             spikeTimesFromEventOnsetAll,
                                             rows=1, cols=1, specificInds=someCells, subplot_titles=None,
                                             plot=False, random=False)
                for trace in pRaster.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=3, col=1)
                fig.add_vline(x=0, line_width=1, line_color="red", row=3, col=1)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=3, col=1)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=3, col=1)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=3, col=1)

                # TODO: Solve how to feed only the indexes in for each repeat sound so we can plot the different varients
                #  together to see if there is large chanegs in response, or if they can be averaged together
                (trialsEachCond_new, nTrialsEachCond_new, nCond_new) = extraplots.trials_each_cond_inds(trialsEachCond, nTrials)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCond_new[0:4], rows=1,
                                        cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=repeat_sounds[0:4], repeat_sound=True, random=False)
                for trace in ppSTH.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=3, col=2)
                fig.add_vline(x=0, line_width=1, line_color="red", row=3, col=2)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=3, col=2)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=3, col=2)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=3, col=2)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCond_new[4:8], rows=1,
                                        cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=repeat_sounds[4:8], repeat_sound=True, random=False)
                for trace in ppSTH.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=4, col=1)
                fig.add_vline(x=0, line_width=1, line_color="red", row=4, col=1)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=4, col=1)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=4, col=1)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=4, col=1)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCond_new[8:12], rows=1,
                                        cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=repeat_sounds[8:12], repeat_sound=True, random=False)
                for trace in ppSTH.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=4, col=2)
                fig.add_vline(x=0, line_width=1, line_color="red", row=4, col=2)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=4, col=2)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=4, col=2)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=4, col=2)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCond_new[12:16], rows=1,
                                        cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=repeat_sounds[12:16], repeat_sound=True, random=False)
                for trace in ppSTH.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=5, col=1)
                fig.add_vline(x=0, line_width=1, line_color="red", row=5, col=1)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=5, col=1)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=5, col=1)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=5, col=1)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCond_new[16:20], rows=1,
                                        cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=repeat_sounds[16:20], repeat_sound=True, random=False)
                for trace in ppSTH.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=5, col=2)
                fig.add_vline(x=0, line_width=1, line_color="red", row=5, col=2)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=5, col=2)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=5, col=2)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=5, col=2)

                # Now averaging all sounds of the same category together
                nInstances = len(possibleStim) // nCategories
                trialsEachCateg = np.zeros((trialsEachCond.shape[0], nCategories), dtype=bool)
                for indc in range(len(possibleStim)):
                    trialsEachCateg[:, indc // nInstances] |= trialsEachCond[:, indc]
                nTrialsEachCateg = trialsEachCateg.sum(axis=0)

                ppSTH = funcs.plot_psth(aud_db, binEdges, spikeCountMatAll, timeRange, trialsEachCateg,
                                        rows=1,
                                        cols=1,
                                        specificInds=someCells, subplot_titles=None, plot=False, colors=pColors,
                                        possibleStim=studyparams.SOUND_CATEGORIES, random=False)

                for trace in ppSTH.data:
                    trace.legend = 'legend3'
                    fig.add_trace(trace, row=6, col=1)
                fig.add_vline(x=0, line_width=1, line_color="red", row=6, col=1)
                fig.add_vline(x=allPeriods[1][1], line_width=1, line_color="black", row=6, col=1)
                fig.add_vline(x=allPeriods[2][1], line_width=1, line_color="blue", row=6, col=1)
                fig.add_vline(x=allPeriods[3][1], line_width=1, line_color="cyan", row=6, col=1)

        fig.update_layout(
            height=2400,
            width=1000,
            title_text=f"Mouse: {mouse} Session: {session}, Region: {aud_db.iloc[someCells]['simpleSiteName'].values[0]}"
                       f", Cell: {someCells[0]}",
            legend=dict(
                title=dict(
                    text="PT"
                ),
                font=dict(size=12),
                xref="container",
                yref="container",
                y=0.96,
                bgcolor="White"),
            legend2=dict(
                title=dict(
                    text="AM"
                ),
                font=dict(size=12),
                xref="container",
                yref="container",
                y=0.80,
                bgcolor="White"),
            legend3=dict(
                title=dict(
                    text="Natural sounds"
                ),
                font=dict(size=12),
                xref="container",
                yref="container",
                y=0.1,
                bgcolor="White")
        )

        # fig.show()
        file_path = os.path.join(figdataPath, f'cell_reports/')
        fig.write_html(f"{file_path}cell_{someCells[0]}_session_{session}.html")
