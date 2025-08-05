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
from importlib import reload
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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

subject = 'feat018'
session = '2024-06-14'
celldb_subset = celldb.query('subject==@subject & date==@session')
celldb_subset_new = celldb_subset[celldb_subset['simpleSiteName'].isin(areas_of_interest)].reset_index()
counts_per_area = celldb_subset_new['simpleSiteName'].value_counts()
print(counts_per_area)

stimType = 'naturalSound'
stimVar = 'soundID'
timeRange = [-2, 6]  # In seconds

#%% Reproducing Santiago's results
ensemble = ephyscore.CellEnsemble(celldb_subset)
ephysData, bdata = ensemble.load(stimType)
currentStim = bdata[stimVar]

nTrials = len(currentStim)
eventOnsetTimes = ephysData['events']['stimOn'][:nTrials] # Ignore trials not in bdata

spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
    ensemble.eventlocked_spiketimes(eventOnsetTimes, timeRange)

possibleStim = np.unique(currentStim)
trialsEachCond = behavioranalysis.find_trials_each_type(currentStim, possibleStim)
condEachSortedTrial, sortedTrials = np.nonzero(trialsEachCond.T)
sortingInds = np.argsort(sortedTrials)

someCells = [16, 87, 283,  145, 151, 137,  186, 187, 188, 189, 197, 303,  225, 232, 373]

probeDepth = celldb.pdepth.iloc[someCells[0]]

title = f"{subject} {session} {probeDepth}um"

fig = funcs.plot_rasters(celldb, sortingInds, trialIndexForEachSpikeAll, spikeTimesFromEventOnsetAll, rows=5,
                         title=title, random=False, specificInds=someCells)

#%% Random plots for auditory regions
ensemble = ephyscore.CellEnsemble(celldb_subset_new)
ephysData, bdata = ensemble.load(stimType)
currentStim = bdata[stimVar]

nTrials = len(currentStim)
eventOnsetTimes = ephysData['events']['stimOn'][:nTrials] # Ignore trials not in bdata

spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
    ensemble.eventlocked_spiketimes(eventOnsetTimes, timeRange)

possibleStim = np.unique(currentStim)
trialsEachCond = behavioranalysis.find_trials_each_type(currentStim, possibleStim)
condEachSortedTrial, sortedTrials = np.nonzero(trialsEachCond.T)
sortingInds = np.argsort(sortedTrials)

fig = funcs.plot_rasters(celldb_subset_new, sortingInds=sortingInds, trialIndexForEachSpikeAll=trialIndexForEachSpikeAll,
                         spikeTimesFromEventOnsetAll=spikeTimesFromEventOnsetAll)