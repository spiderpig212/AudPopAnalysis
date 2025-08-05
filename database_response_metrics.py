"""
Add columns to database with responsiveness to natural sounds.

It took about 7 min to process all 10486 cells.
"""

import os
import sys
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

from jaratoolbox import settings
from jaratoolbox import celldatabase
from jaratoolbox import ephyscore
from jaratoolbox import behavioranalysis
from jaratoolbox import spikesanalysis
from jaratoolbox import extraplots
import studyparams
from importlib import reload

reload(studyparams)

# TODO: Calculate the peak response for each time region, store that response as well as the stim value that created it,
#  and then also calculate our index value, which is I = (A - B) / (A + B) where A is avg FR during stim that evokes
#  highest response and B is avg FR during stim that evokes lowest FR.

# -- Load database of cells --
dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_coords.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)

# -- Define what stimulus to use --
# TODO: Could probably make these into dictionaries in studyparams.py and load into variables as needed
for CASE in [0, 1, 2]:
    if CASE == 0:
        stimAbv = 'ns'
        stimType = 'naturalSound'
        stimVar = 'soundID'
        timeRange = [-2, 6]  # In seconds
        allPeriods = [[-1, 0], [0, 0.5], [1, 4], [4, 4.5]]
        nCategories = len(studyparams.SOUND_CATEGORIES)
    elif CASE == 1:
        stimAbv = 'am'
        stimType = 'AM'
        stimVar = 'currentFreq'
        timeRange = [-0.5, 1.5]
        allPeriods = [[-0.5, 0], [0, 0.2], [0.2, 0.5], [0.5, 0.7]]
        nCategories = 11
    elif CASE == 2:
        stimAbv = 'pt'
        stimType = 'pureTones'
        stimVar = 'currentFreq'
        timeRange = [-0.1, 0.3]
        allPeriods = [[-0.1, 0], [0, 0.05], [0.05, 0.1], [0.1, 0.15]]
        nCategories = 16


    nCells = len(celldb)

    periodsName = ['base', 'respOnset', 'respSustained', 'respOffset']
    periodDuration = [x[1] - x[0] for x in allPeriods]
    # meanFiringEachPeriodEachCell = np.empty((nCells, len(allPeriods)))

    pValsEachCellOnset = np.empty((nCells, nCategories))
    pValsEachCellSustain = np.empty((nCells, nCategories))
    pValsEachCellOffset = np.empty((nCells, nCategories))
    minPvalEachCellOnset = np.full(nCells, np.nan)
    minPvalEachCellSustain = np.full(nCells, np.nan)
    minPvalEachCellOffset = np.full(nCells, np.nan)
    minPvalIndexEachCellOnset = np.full(nCells, -1)
    minPvalIndexEachCellSustain = np.full(nCells, -1)
    minPvalIndexEachCellOffset = np.full(nCells, -1)

    firingRateEachCellBase = np.full(nCells, np.nan)
    bestFiringRateEachCellOnset = np.full(nCells, np.nan)
    bestFiringRateEachCellSustain = np.full(nCells, np.nan)
    bestFiringRateEachCellOffset = np.full(nCells, np.nan)
    bestIndexEachCellOnset = np.full(nCells, -1)
    bestIndexEachCellSustain = np.full(nCells, -1)
    bestIndexEachCellOffset = np.full(nCells, -1)
    maxFiringRateEachCellOnset = np.full(nCells, np.nan)
    minFiringRateEachCellOnset = np.full(nCells, np.nan)
    maxFiringRateEachCellSustain = np.full(nCells, np.nan)
    minFiringRateEachCellSustain = np.full(nCells, np.nan)
    maxFiringRateEachCellOffset = np.full(nCells, np.nan)
    minFiringRateEachCellOffset = np.full(nCells, np.nan)

    minStimValOnset = np.full(nCells, 'a')
    minStimValSustain = np.full(nCells, 'a')
    minStimValOffset = np.full(nCells, 'a')
    maxStimValOnset = np.full(nCells, 'a')
    maxStimValSustain = np.full(nCells, 'a')
    maxStimValOffset = np.full(nCells, 'a')

    num_iterations = len(celldb)
    indCell = -1
    for indRow, dbRow in tqdm(celldb.iterrows(), total=num_iterations, desc=f"Calculating firing rates for {stimType}"):
        indCell += 1
        # if indCell != 12: continue  # DEBUG (using only one cell)
        oneCell = ephyscore.Cell(dbRow)
        ephysData, bdata = oneCell.load(stimType)

        spikeTimes = ephysData['spikeTimes']
        eventOnsetTimes = ephysData['events']['stimOn']
        currentStim = bdata[stimVar]

        # -- Test if trials from behavior don't match ephys --
        if (len(currentStim) > len(eventOnsetTimes)) or \
                (len(currentStim) < len(eventOnsetTimes) - 1):
            print(f'[{indRow}] Warning! BevahTrials ({len(currentStim)}) and ' +
                  f'EphysTrials ({len(eventOnsetTimes)})')
            continue
        if len(currentStim) == len(eventOnsetTimes) - 1:
            eventOnsetTimes = eventOnsetTimes[:len(currentStim)]

        possibleStim = np.unique(currentStim)
        trialsEachInstance = behavioranalysis.find_trials_each_type(currentStim, possibleStim)
        nTrialsEachInstance = trialsEachInstance.sum(axis=0)  # Not used, but in case you need it
        stimLabels = np.argmax(trialsEachInstance, axis=1)

        # -- Identify trials per category --
        nInstances = len(possibleStim) // nCategories
        trialsEachCateg = np.zeros((trialsEachInstance.shape[0], nCategories), dtype=bool)
        for indc in range(len(possibleStim)):
            trialsEachCateg[:, indc // nInstances] |= trialsEachInstance[:, indc]
        nTrialsEachCateg = trialsEachCateg.sum(axis=0)
        stimLabelsCateg = np.argmax(trialsEachCateg, axis=1)

        (spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial) = \
            spikesanalysis.eventlocked_spiketimes(spikeTimes, eventOnsetTimes, timeRange)

        meanFiringEachPeriod = np.empty(len(allPeriods))
        spikesEachTrialEachPeriod = []
        for indPeriod, period in enumerate(allPeriods):
            spikeCountMat = spikesanalysis.spiketimes_to_spikecounts(spikeTimesFromEventOnset,
                                                                     indexLimitsEachTrial, period)
            spikesEachTrial = spikeCountMat[:, 0]
            spikesEachTrialEachPeriod.append(spikesEachTrial)

        firingRateEachCellBase[indCell] = spikesEachTrialEachPeriod[0].mean() / periodDuration[0]

        meanFiringRateBase = np.empty(nCategories)
        meanFiringRateOnset = np.empty(nCategories)
        pValEachCondOnset = np.empty(nCategories)
        meanFiringRateSustain = np.empty(nCategories)
        pValEachCondSustain = np.empty(nCategories)
        meanFiringRateOffset = np.empty(nCategories)
        pValEachCondOffset = np.empty(nCategories)
        for indcond in range(nCategories):
            trialsThisCond = trialsEachCateg[:, indcond]
            firingRateBase = spikesEachTrialEachPeriod[0][trialsThisCond] / periodDuration[0]
            firingRateOnset = spikesEachTrialEachPeriod[1][trialsThisCond] / periodDuration[1]
            firingRateSustain = spikesEachTrialEachPeriod[2][trialsThisCond] / periodDuration[2]
            firingRateOffset = spikesEachTrialEachPeriod[3][trialsThisCond] / periodDuration[3]
            try:
                wStat, pValThisCond = stats.wilcoxon(firingRateBase, firingRateOnset)
            except ValueError:
                pValThisCond = 1
            pValEachCondOnset[indcond] = pValThisCond
            try:
                wStat, pValThisCond = stats.wilcoxon(firingRateBase, firingRateSustain)
            except ValueError:
                pValThisCond = 1
            pValEachCondSustain[indcond] = pValThisCond
            try:
                wStat, pValThisCond = stats.wilcoxon(firingRateBase, firingRateOffset)
            except ValueError:
                pValThisCond = 1
            pValEachCondOffset[indcond] = pValThisCond
            meanFiringRateOnset[indcond] = firingRateOnset.mean()
            meanFiringRateSustain[indcond] = firingRateSustain.mean()
            meanFiringRateOffset[indcond] = firingRateOffset.mean()

        indMinPvalOnset = np.argmin(pValEachCondOnset)
        minPvalIndexEachCellOnset[indCell] = indMinPvalOnset
        minPvalEachCellOnset[indCell] = pValEachCondOnset[indMinPvalOnset]
        # "BEST" indicates the maximum absolute value difference between the mean firing
        #  rate for a given modulation rate and baseline firing rate for each cell.
        indBestOnset = np.argmax(np.abs(meanFiringRateOnset - firingRateEachCellBase[indCell]))
        bestIndexEachCellOnset[indCell] = indBestOnset
        bestFiringRateEachCellOnset[indCell] = meanFiringRateOnset[indBestOnset]
        maxFiringRateEachCellOnset[indCell] = np.max(meanFiringRateOnset)
        maxStimValOnset[indCell] = studyparams.SOUND_CATEGORIES[stimLabelsCateg[np.argmax(meanFiringRateOnset)]] if (
            stimType == 'naturalSound') else possibleStim[stimLabelsCateg[np.argmax(meanFiringRateOnset)]]
        minFiringRateEachCellOnset[indCell] = np.min(meanFiringRateOnset)
        minStimValOnset[indCell] = studyparams.SOUND_CATEGORIES[stimLabelsCateg[np.argmin(meanFiringRateOnset)]] if (
            stimType == 'naturalSound') else possibleStim[stimLabelsCateg[np.argmin(meanFiringRateOnset)]]

        indMinPvalSustain = np.argmin(pValEachCondSustain)
        minPvalIndexEachCellSustain[indCell] = indMinPvalSustain
        minPvalEachCellSustain[indCell] = pValEachCondSustain[indMinPvalSustain]
        indBestSustain = np.argmax(np.abs(meanFiringRateSustain - firingRateEachCellBase[indCell]))
        bestIndexEachCellSustain[indCell] = indBestSustain
        bestFiringRateEachCellSustain[indCell] = meanFiringRateSustain[indBestSustain]
        maxFiringRateEachCellSustain[indCell] = np.max(meanFiringRateSustain)
        maxStimValSustain[indCell] = studyparams.SOUND_CATEGORIES[stimLabelsCateg[np.argmax(meanFiringRateSustain)]] if (
            stimType == 'naturalSound') else possibleStim[stimLabelsCateg[np.argmax(meanFiringRateSustain)]]
        minFiringRateEachCellSustain[indCell] = np.min(meanFiringRateSustain)
        minStimValSustain[indCell] = studyparams.SOUND_CATEGORIES[stimLabelsCateg[np.argmin(meanFiringRateSustain)]] if (
            stimType == 'naturalSound') else possibleStim[stimLabelsCateg[np.argmin(meanFiringRateSustain)]]

        indMinPvalOffset = np.argmin(pValEachCondOffset)
        minPvalIndexEachCellOffset[indCell] = indMinPvalOffset
        minPvalEachCellOffset[indCell] = pValEachCondOffset[indMinPvalOffset]
        indBestOffset = np.argmax(np.abs(meanFiringRateOffset - firingRateEachCellBase[indCell]))
        bestIndexEachCellOffset[indCell] = indBestOffset
        bestFiringRateEachCellOffset[indCell] = meanFiringRateOffset[indBestOffset]
        maxFiringRateEachCellOffset[indCell] = np.max(meanFiringRateOffset)
        maxStimValOffset[indCell] = studyparams.SOUND_CATEGORIES[stimLabelsCateg[np.argmax(meanFiringRateOffset)]] if (
                stimType == 'naturalSound') else possibleStim[stimLabelsCateg[np.argmax(meanFiringRateOffset)]]
        minFiringRateEachCellOffset[indCell] = np.min(meanFiringRateOffset)
        minStimValOffset[indCell] = studyparams.SOUND_CATEGORIES[stimLabelsCateg[np.argmin(meanFiringRateOffset)]] if (
                stimType == 'naturalSound') else possibleStim[stimLabelsCateg[np.argmin(meanFiringRateOffset)]]

    celldb[f'{stimAbv + "MinPvalOnset"}'] = minPvalEachCellOnset
    celldb[f'{stimAbv + "IndexMinPvalOnset"}'] = minPvalIndexEachCellOnset
    celldb[f'{stimAbv + "MinPvalSustain"}'] = minPvalEachCellSustain
    celldb[f'{stimAbv + "IndexMinPvalSustain"}'] = minPvalIndexEachCellSustain

    celldb[f'{stimAbv + "FiringRateBaseline"}'] = firingRateEachCellBase
    celldb[f'{stimAbv + "FiringRateBestOnset"}'] = bestFiringRateEachCellOnset
    celldb[f'{stimAbv + "IndexBestOnset"}'] = bestIndexEachCellOnset
    celldb[f'{stimAbv + "FiringRateBestSustain"}'] = bestFiringRateEachCellSustain
    celldb[f'{stimAbv + "IndexBestSustain"}'] = bestIndexEachCellSustain
    celldb[f'{stimAbv + "FiringRateBestOffset"}'] = bestFiringRateEachCellOffset
    celldb[f'{stimAbv + "IndexBestOffset"}'] = bestIndexEachCellOffset

    celldb[f'{stimAbv + "FiringRateMaxOnset"}'] = maxFiringRateEachCellOnset
    celldb[f'{stimAbv + "FiringRateMinOnset"}'] = minFiringRateEachCellOnset
    celldb[f'{stimAbv + "FiringRateMaxSustain"}'] = maxFiringRateEachCellSustain
    celldb[f'{stimAbv + "FiringRateMinSustain"}'] = minFiringRateEachCellSustain
    celldb[f'{stimAbv + "FiringRateMaxOffset"}'] = maxFiringRateEachCellOffset
    celldb[f'{stimAbv + "FiringRateMinOffset"}'] = minFiringRateEachCellOffset
    celldb[f'{stimAbv + "StimValMaxOnset"}'] = maxStimValOnset
    celldb[f'{stimAbv + "StimValMinOnset"}'] = minStimValOnset
    celldb[f'{stimAbv + "StimValMaxSustain"}'] = maxStimValSustain
    celldb[f'{stimAbv + "StimValMinSustain"}'] = minStimValSustain
    celldb[f'{stimAbv + "StimValMaxOffset"}'] = maxStimValOffset
    celldb[f'{stimAbv + "StimValMinOffset"}'] = minStimValOffset

    frIndexOnset = ((maxFiringRateEachCellOnset - minFiringRateEachCellOnset) /
                    (maxFiringRateEachCellOnset + minFiringRateEachCellOnset))
    frIndexSustain = ((maxFiringRateEachCellSustain - minFiringRateEachCellSustain) /
                       (maxFiringRateEachCellSustain + minFiringRateEachCellSustain))
    frIndexOffset = ((maxFiringRateEachCellOffset - minFiringRateEachCellOffset) /
                      (maxFiringRateEachCellOffset + minFiringRateEachCellOffset))
    celldb[f'{stimAbv + "FRIndexOnset"}'] = frIndexOnset
    celldb[f'{stimAbv + "FRIndexSustain"}'] = frIndexSustain
    celldb[f'{stimAbv + "FRIndexOffset"}'] = frIndexOffset

    print(f'Successfully added metrics for {stimType}')

dbResponsiveFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5')
celldatabase.save_hdf(celldb, dbResponsiveFilename)
