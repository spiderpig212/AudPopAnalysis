import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jaratoolbox import ephyscore
from jaratoolbox import settings
from jaratoolbox import celldatabase
from jaratoolbox import behavioranalysis
from jaratoolbox import spikesanalysis
from jaratoolbox import histologyanalysis as ha  # This requires the Allen SDK
from jaratoolbox import extraplots
import studyparams
from tqdm import tqdm

# TODO: Go through session by session and plot rasters for all the best channel of each neuron to see if the rasters also
#  show the sound responsive nature we see in the LFPs, so as to guide our decision on how to best use the LFP

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
    "Ventral auditory area",
    "Temporal association areas"
]

# stimAbv = 'ns'
# stimType = 'naturalSound'
# stimVar = 'soundID'
# timeRange = [-2, 6]
# xlims = [-0.3, 3]

stimAbv = 'pt'
stimType = 'pureTones'
stimVar = 'currentFreq'
timeRange = [-0.1, 0.35]
allPeriods = [[-0.1, 0], [0, 0.05], [0.05, 0.1], [0.1, 0.15]]
xlims = [-0.1, 0.35]

unique_mice = celldb.subject.unique()
for mouse in unique_mice:
    mouse_db = celldb.query('subject==@mouse')
    unique_sessions = mouse_db.date.unique()
    for session in unique_sessions:
        session_db = mouse_db.query('date==@session')
        # aud_db = session_db[session_db['simpleSiteName'].isin(areas_of_interest)].reset_index()
        # Sorting the db by best channel number in the sessions and then plotting rasters
        sorted_db = session_db.sort_values(by="bestChannel")
        cellsToPlot = sorted_db.index

        fig = plt.figure(figsize=(12, 160))
        axs = fig.subplots(int(np.ceil(len(cellsToPlot) / 3)), 3, sharex=True, sharey=True)
        current_axis = 0

        for indRow, dbRow in tqdm(sorted_db.iterrows(), total=len(sorted_db), desc=f"Plotting rasters for session {session} for mouse {mouse}"):
            oneCell = ephyscore.Cell(dbRow, useModifiedClusters=False)
            ephysData, bdata = oneCell.load(stimType)
            currentStim = bdata[stimVar]

            nTrials = len(currentStim)
            spikeTimes = ephysData['spikeTimes']
            onsetTimes = ephysData['events']['stimOn'][:nTrials]
            (spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial) = \
                spikesanalysis.eventlocked_spiketimes(spikeTimes, onsetTimes, timeRange)
            spikeCountMat = spikesanalysis.spiketimes_to_spikecounts(spikeTimesFromEventOnset,
                                                                          indexLimitsEachTrial,
                                                                          timeRange)

            possibleStim = np.unique(currentStim)
            trialsEachCond = behavioranalysis.find_trials_each_type(currentStim, possibleStim)
            condEachSortedTrial, sortedTrials = np.nonzero(trialsEachCond.T)
            sortingInds = np.argsort(sortedTrials)

            probeDepth = celldb.pdepth.iloc[cellsToPlot[0]]
            sortedIndexForEachSpike = sortingInds[trialIndexForEachSpike]

            plt.sca(axs.flat[current_axis])
            pRaster, hcond, zline = extraplots.raster_plot(spikeTimesFromEventOnset,
                                                           indexLimitsEachTrial,
                                                           timeRange,
                                                           trialsEachCond=trialsEachCond,
                                                           colorEachCond=None)
            plt.setp(pRaster, ms=1)
            plt.setp(hcond, zorder=3)
            plt.xlabel('Time (s)')
            plt.ylabel(f'Channel {dbRow.bestChannel}')
            plt.xlim(xlims)
            plt.axvline(0.02, color='b', lw=1)
            # plt.title(f'Channel {dbRow.bestChannel}')

            # axs.flat[current_axis].plot(spikeTimesFromEventOnset, sortedIndexForEachSpike, '.k', ms=1)
            # axs.flat[current_axis].set_xlabel('Time (s)')
            # axs.flat[current_axis].set_ylabel(f'[{indRow}] Sorted trials')
            # recSiteName = dbRow.SimpleSiteName
            # axs.flat[current_axis].set_title(f'{recSiteName}')
            current_axis += 1

        plt.tight_layout()
        plt.savefig(os.path.join(figdataPath, f'bestChannelPlots/best_channel_rasters_{mouse}_{session}_{stimAbv}.png'))
        print(f"Saved {mouse}_{session}_{stimAbv}")