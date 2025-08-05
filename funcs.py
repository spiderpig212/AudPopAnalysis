import os
import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d
import plotly.colors as colors

from jaratoolbox import celldatabase
import matplotlib.pyplot as plt
from jaratoolbox import settings
from jaratoolbox import extraplots
from jaratoolbox import spikesanalysis
from jaratoolbox import ephyscore
from jaratoolbox import behavioranalysis
import sys
import studyparams
import studyutils
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


def hex_to_rgba(hex_color, alpha=1):
    """
    Converts a hexadecimal color string (e.g., "#RRGGBB" or "#RRGGBBAA")
    to an RGBA tuple (R, G, B, A), where A is between 0 and 1.
    """
    hex_color = hex_color.lstrip('#')  # Remove '#' if present

    if len(hex_color) == 6:
        # Assume full opacity if no alpha component is provided
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = alpha
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = alpha
    else:
        raise ValueError("Invalid hex color format. Expected 6 or 8 characters.")

    return (r, g, b, a)

def plot_rasters(celldb:pd.DataFrame, sortingInds: np.array, trialIndexForEachSpikeAll: list,
                 spikeTimesFromEventOnsetAll: list, rows: int=3, cols: int=3,
                 random: bool=True, specificInds: list=None, title: str=None, plot: bool=True,
                 subplot_titles: bool=True) -> plotly.graph_objs._figure.Figure:

    n_cells = rows*cols
    if random:
        someCells = celldb.sample(n=n_cells).index.tolist()
    elif random == False and specificInds is not None:
        someCells = specificInds
    elif random == False and specificInds is None:
        someCells = np.arange(n_cells)

    # Initialize subplots with shared X and Y axes
    fig = make_subplots(
        rows=rows, cols=cols,
        shared_xaxes=False, shared_yaxes=True,
        subplot_titles=celldb.loc[someCells]['simpleSiteName'].to_list() if subplot_titles else None,
    )

    for count, indcell in enumerate(someCells):
        row = count // 3 + 1
        col = count % 3 + 1

        sortedIndexForEachSpike = sortingInds[trialIndexForEachSpikeAll[indcell]]
        fig.add_trace(
            go.Scatter(
                x=spikeTimesFromEventOnsetAll[indcell],
                y=sortedIndexForEachSpike,
                mode='markers',
                marker=dict(size=2, color='black'),
                name=f'Cell {indcell}'
            ),
            row=row, col=col
        )
        # Add axis labels
        fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        fig.update_yaxes(title_text=f"[{indcell}] Sorted trials", row=row, col=col)

    fig.add_vline(
        x=0,
        line=dict(color='red', width=2),  # Customize color and width
    )

    if title is not None:
        fig.update_layout(
            title=title,
            showlegend=False,
            height=400 * rows,
            width=1200,
            title_font=dict(size=16, family='Arial', color='black')
        )
    else:
        fig.update_layout(
            showlegend=False,
            height=400 * rows,
            width=1200,
        )

    if plot:
        fig.show();

    return fig

def plot_psth(celldb:pd.DataFrame, binsStartTime: np.array,
              spikeCountMatAll: list, timeRange: list, trialsEachCond: list, smoothWinSize: int=2,
              rows: int=3, cols: int=3, colors: list=None,
              random: bool=True, specificInds: list=None, title: str=None, plot: bool=True,
              subplot_titles: bool=True, possibleStim: ndarray=None, downsamplefactor: int=1,
              hidesamples: int=0, repeat_sound: bool=False) -> plotly.graph_objs._figure.Figure:
    n_cells = rows*cols
    if random:
        someCells = celldb.sample(n=n_cells).index.tolist()
    elif random == False and specificInds is not None:
        someCells = specificInds
    elif random == False and specificInds is None:
        someCells = np.arange(n_cells)

    fig = make_subplots(
            rows=rows, cols=cols,
            shared_xaxes=False, shared_yaxes=True,
            subplot_titles=celldb.loc[someCells]['simpleSiteName'].to_list() if subplot_titles else None,
        )

    nTrials = spikeCountMatAll.shape[1]
    winShape = np.concatenate((np.zeros(smoothWinSize), np.ones(smoothWinSize)))  # Square (causal)
    winShape = winShape / np.sum(winShape)
    if not repeat_sound:
        (trialsEachCond, nTrialsEachCond, nCond) = extraplots.trials_each_cond_inds(trialsEachCond, nTrials)
    elif repeat_sound:
        nCond = len(trialsEachCond)

    for index_cond in range(nCond):
        thisCondCounts = spikeCountMatAll[someCells, trialsEachCond[index_cond], :]
        thisPSTH = np.mean(thisCondCounts, axis=0)
        smoothPSTH = np.convolve(thisPSTH, winShape, mode='same')
        sSlice = slice(hidesamples, len(smoothPSTH) - hidesamples, downsamplefactor)

        if possibleStim is not None:
            stimVal = possibleStim[index_cond]
            if repeat_sound:
                stimVal = f"{stimVal + '_' + str(index_cond % 4)}"
        if colors:
            hex_color = colors[index_cond % len(colors)]
            color = 'rgba' + str(hex_to_rgba(hex_color, 0.7))
        else:
            color = "blue"
        fig.add_trace(go.Scatter(
            x=binsStartTime[sSlice],
            y=smoothPSTH[sSlice],
            name=f"{stimVal}",
            mode='lines',
            line=dict(color=color, width=1),
        ))

    fig.update_layout(showlegend=False)
    if plot:
        fig.show();

    return fig


def calculate_fr_arrays(celldb:pd.DataFrame, stimType:str, stimVar:str, timeRange:list, allPeriods:list) -> list:
    """
    Calculates the firing rate arrays for given cell data, stimulus type, and time range.

    This function processes a given set of cell data from a DataFrame alongside the
    corresponding stimulus type and a specified time range. It calculates the firing
    rate arrays for different response periods, such as baseline, onset response, and
    sustained response periods. The calculations take into account the number of cells,
    categories, and specified time ranges provided.

    Args:
        celldb (pd.DataFrame): A DataFrame containing cell information for tracking relevant ephys data.
        stimType (str): The type of stimulus used in the experiment. Valid values are 'Sine', 'naturalSound', and
        'AM'.
        stimVar (str): The name of the variable in the behavior data representing the stimulus.
        timeRange (list): A list specifying the time range to analyze, in the form
            [start_time, end_time].
        allPeriods (list): A list of tuples/lists specifying the periods to analyze. Assumes order of
        [baseline, onset, sustained, offset].

    Returns:
        list: A list containing the calculated firing rate arrays for all periods and an array of trial stims
        [baseline, onset, sustained, offset, stimArray]
    """
    nCells = len(celldb)
    periodDuration = [x[1] - x[0] for x in allPeriods]

    if stimType == 'AM':
        nTrials = 220
        nCategories = 11
    elif stimType == 'naturalSound':
        nTrials = 200
        nCategories = len(studyparams.SOUND_CATEGORIES)
    elif stimType == 'pureTones':
        nTrials = 320
        nCategories = 16
    else:
        raise ValueError(f"Unrecognized stimulus type: {stimType}. Should be in ['AM', 'naturalSound', 'pureTones']")

    basefr = np.full((nCells, nTrials), np.nan)
    onsetfr = np.full((nCells, nTrials), np.nan)
    sustainedfr = np.full((nCells, nTrials), np.nan)
    offsetfr = np.full((nCells, nTrials), np.nan)
    stimArray = np.full((nCells, nTrials), np.nan)
    brainRegion = np.empty(nCells, object)
    mouseID = np.empty(nCells, object)

    num_iterations = len(celldb)
    indCell = -1
    for indRow, dbRow in tqdm(celldb.iterrows(), total=num_iterations, desc=f"Calculating firing rates for {stimType}"):
        indCell += 1
        oneCell = ephyscore.Cell(dbRow)
        ephysData, bdata = oneCell.load(stimType)

        spikeTimes = ephysData['spikeTimes']
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        currentStim = bdata[stimVar][:nTrials]

        # -- Test if trials from behavior don't match ephys -- Shouldn't matter since I am manually subsetting both above
        if (len(currentStim) > len(eventOnsetTimes)) or \
                (len(currentStim) < len(eventOnsetTimes) - 1):
            print(f'[{indRow}] Warning! BevahTrials ({len(currentStim)}) and ' +
                  f'EphysTrials ({len(eventOnsetTimes)})')
            continue
        if len(currentStim) == len(eventOnsetTimes) - 1:
            eventOnsetTimes = eventOnsetTimes[:len(currentStim)]

        (spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial) = \
            spikesanalysis.eventlocked_spiketimes(spikeTimes, eventOnsetTimes, timeRange)

        spikesEachTrialEachPeriod = []
        for indPeriod, period in enumerate(allPeriods):
            spikeCountMat = spikesanalysis.spiketimes_to_spikecounts(spikeTimesFromEventOnset,
                                                                     indexLimitsEachTrial, period)
            spikesEachTrial = spikeCountMat[:, 0]
            spikesEachTrialEachPeriod.append(spikesEachTrial)

        # for indcond in range(nCategories):
            # trialsThisCond = trialsEachCateg[:, indcond]
        firingRateBase = spikesEachTrialEachPeriod[0] / periodDuration[0]
        firingRateOnset = spikesEachTrialEachPeriod[1] / periodDuration[1]
        firingRateSustain = spikesEachTrialEachPeriod[2] / periodDuration[2]
        firingRateOffset = spikesEachTrialEachPeriod[3] / periodDuration[3]

        basefr[indCell, :] = firingRateBase
        onsetfr[indCell, :] = firingRateOnset
        sustainedfr[indCell, :] = firingRateSustain
        offsetfr[indCell, :] = firingRateOffset
        stimArray[indCell, :] = currentStim
        brainRegion[indCell] = dbRow['simpleSiteName']
        mouseID[indCell] = dbRow['subject']

    return [basefr, onsetfr, sustainedfr, offsetfr, stimArray, brainRegion, mouseID]

def calc_d_prime(array1: np.array, array2: np.array) -> np.float32:
    """
    Adapted from Sparse Coding in Temporal Association Cortex Improves Complex Sound Discriminability by Feigin et al.
    2021 in J neurosci. Takes in two arrays of shape (nCells, nTrials) where each array is a different stimulus type e.g.
    for pure tones array1 may be 12 kHz and array2 may be 20 kHz. d' is defined as

    d' = euc_distance(mean(array1), mean(array2)) / mean(inner_distance(array1), inner_distance(array2)))

    where means are trial-averages and inner_distance is the Euclidean distance between each trial firing rate and the
    mean stimulus firing rate, followed by averaging over the differences (size nTrials) to get a scaler.

    Args:
        array1: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2 should be a different stimulus type than array1, otherwise d' will be zero.

    Returns:

    """
    mean1 = np.mean(array1, axis=1)
    mean2 = np.mean(array2, axis=1)
    mean_inner_distance = np.mean((np.mean(np.sqrt(np.square(array1 - mean1[:, np.newaxis])), axis=1),
                                   np.mean(np.sqrt(np.square(array2 - mean2[:, np.newaxis])), axis=1)))  # Creates an array of shape nNeurons for each input, then averages them to single value
    dprime = np.mean(np.sqrt(np.square(mean1 - mean2))) / mean_inner_distance
    return dprime

def calc_fisher_criterion(array1: np.array, array2: np.array) -> np.array:
    """
    Defined as

    J(w) = (m_1 - m_2)^2 / (s_1^2 + s_2^2)

    where
    m_1 = np.mean(x_1 @ w), which is mean along projection w,
    s_1 = np.var(x_1 @ w), which is the scatter along projection w,
    and x_1 is first data set/array.

    For higher dimensions, it can be instead defined as:

    J(w) = (w^T * s_B * w) / (w^T * s_W * w)

    where
    s_B = between-class scatter matrix
    s_W = within-class scatter matrix

    Returns J(w)

    Args:
        array1: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2 should be a different stimulus type than array1, otherwise result will be zero

    Returns:

    """
    m1 = np.mean(array1, axis=0)
    m2 = np.mean(array2, axis=0)

    # Compute within-class
    s1 = np.cov(array1, rowvar=False)
    s2 = np.cov(array2, rowvar=False)
    s_W = s1 + s2

    # Compute between-class
    mean_diff = (m1 - m2).reshape(-1, 1)
    s_B = mean_diff @ mean_diff.T

    # Solve generalized eigenvalue problem for w
    # Equivalent to maximizing J(w) = w^T s_B w / w^T s_W w, which should be same as what is defined above
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(s_W) @ s_B)

    w = eigvecs[:, np.argmax(eigvals)]  # Grab the eigenvec corresponding with the largest eigenval

    # Compute Fisher's criterion value
    J = (w.T @ s_B @ w) / (w.T @ s_W @ w)

    return J.real, w.real