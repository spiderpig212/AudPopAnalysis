import os
from numpy import ndarray
import plotly.colors as colors
from sklearn.decomposition import PCA
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools

from jaratoolbox import settings
from jaratoolbox import extraplots
from jaratoolbox import spikesanalysis
from jaratoolbox import ephyscore
import sys
import studyparams
import studyutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
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

def plot_rasters(celldb:pd.DataFrame, sortingInds: np.ndarray, trialIndexForEachSpikeAll: list,
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

def plot_psth(celldb:pd.DataFrame, binsStartTime: np.ndarray,
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
    sessionID = np.empty(nCells, object)

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

        sorted_stim_ind = np.argsort(currentStim)
        sorted_stim_array = currentStim[sorted_stim_ind]
        sorted_fr_base = firingRateBase[sorted_stim_ind]
        sorted_fr_onset = firingRateOnset[sorted_stim_ind]
        sorted_fr_sustained = firingRateSustain[sorted_stim_ind]
        sorted_fr_offset = firingRateOffset[sorted_stim_ind]


        basefr[indCell, :] = sorted_fr_base
        onsetfr[indCell, :] = sorted_fr_onset
        sustainedfr[indCell, :] = sorted_fr_sustained
        offsetfr[indCell, :] = sorted_fr_offset
        stimArray[indCell, :] = sorted_stim_array
        brainRegion[indCell] = dbRow['simpleSiteName']
        mouseID[indCell] = dbRow['subject']
        sessionID[indCell] = dbRow['date']

    return [basefr, onsetfr, sustainedfr, offsetfr, stimArray, brainRegion, mouseID, sessionID]

def calc_d_prime(array1: np.ndarray, array2: np.ndarray) -> np.float32:
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

def calc_fisher_criterion_Christian(array1: np.ndarray, array2: np.ndarray) -> (np.float32, np.ndarray):
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

    Returns J(w) and w

    Args:
        array1: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2 should be a different stimulus type than array1, otherwise result will be zero

    Returns:

    """
    m1 = np.mean(array1, axis=1).reshape(-1, 1)
    m2 = np.mean(array2, axis=1).reshape(-1, 1)

    # Compute within-class
    s1 = np.cov(array1, rowvar=True)
    s2 = np.cov(array2, rowvar=True)
    s_W = s1 + s2

    # Compute between-class
    mean_diff = (m2 - m1)#.reshape(-1, 1)  # Was adding in new column to make (n_neurons, 1)
    s_B = np.dot(mean_diff, mean_diff.T)

    # Solve generalized eigenvalue problem for w
    # Equivalent to maximizing J(w) = w^T s_B w / w^T s_W w, which should be same as what is defined above
    # eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(s_W) @ mean_diff)

    # w = eigvecs[:, np.argmax(eigvals)]  # Grab the eigenvec corresponding with the largest eigenval
    w = np.dot(np.linalg.pinv(s_W, rcond=1E-6, hermitian=True), mean_diff)  # W Shape (nCells)

    # Compute Fisher's criterion value as it should be defined in the docstring
    # J = (w.T @ s_B @ w) / (w.T @ s_W @ w)
    if w.sum() == 0:
        J = 0
    else:
        w = w / np.linalg.norm(w)  # Normalzie to unit vector!
        J = (np.dot(np.dot(w.T, s_B), w)) / ((np.dot(np.dot(w.T, s_W), w)))
        J = J[0, 0]  # Extract it from the matrix that is shape 1,1

    return J, w

def calc_fisher_criterion(array1: np.ndarray, array2: np.ndarray) -> (np.float32, np.ndarray):
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

    Returns J(w) and w

    Args:
        array1: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2: Array of shape (nCells, nTrials) where each row is a different cell and each column is a different trial.
        array2 should be a different stimulus type than array1, otherwise result will be zero

    Returns:

    """
    # TODO: Concat arrays to be n_neurons x 2*nTrials, do PCA on concat version (19, 20), keep # of PCs = # 2*nTrials - 1, Divide
    #  PC array to get (19, 10)
    concat_array = np.concatenate((array1, array2), axis=1).T
    pcd = PCA()
    pcd.fit(concat_array)

    pc_array_data = pcd.transform(concat_array)
    array1 = pc_array_data[:-1, :array1.shape[1]]
    array2 = pc_array_data[:-1, array1.shape[1]:]

    m1 = np.mean(array1, axis=1).reshape(-1, 1)
    m2 = np.mean(array2, axis=1).reshape(-1, 1)

    # Compute within-class
    s1 = np.cov(array1, rowvar=True)  # Do I need to make ddof=0? That would line up with the math on the website
    s2 = np.cov(array2, rowvar=True)
    # s1 = np.dot((array1 - m1), (array1 - m1).T)
    # s2 = np.dot((array2 - m2), (array2 - m2).T)
    s_W = s1 + s2

    # Compute between-class
    mean_diff = (m2 - m1)#.reshape(-1, 1)  # Was adding in new column to make (n_neurons, 1)
    s_B = np.dot(mean_diff, mean_diff.T)

    # Solve generalized eigenvalue problem for w
    # Equivalent to maximizing J(w) = w^T s_B w / w^T s_W w, which should be same as what is defined above
    # eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(s_W) @ mean_diff)

    # w = eigvecs[:, np.argmax(eigvals)]  # Grab the eigenvec corresponding with the largest eigenval
    w = np.dot(np.linalg.pinv(s_W), mean_diff)  # Not using pinv gives singular matrix error
    # TODO: Normalize w to unit vector for the purposes of graphing later

    # Compute Fisher's criterion value as it should be defined in the docstring
    # J = (w.T @ s_B @ w) / (w.T @ s_W @ w)
    if w.sum() == 0:
        J = 0
    else:
        w = w / np.linalg.norm(w)  # Normalzie to unit vector!
        J = (np.dot(np.dot(w.T, s_B), w)) / ((np.dot(np.dot(w.T, s_W), w)))

    return J, w

def plot_scatter_and_histogram(array1, array2, w, title="Scatter Plot with Projections and Histogram"):
    """
    Plot a scatterplot of two datasets along with their projections and a histogram of projections inlayed.

    Args:
        array1: Original data points for dataset 1 (shape: nCells x nTrials).
        array2: Original data points for dataset 2 (shape: nCells x nTrials).
        w: Projection vector defining the new axis.
        title: Title for the plot.
    """
    # Reshape w to ensure compatibility
    w = w.flatten()

    # Project data onto the vector w
    projections1 = np.dot(array1.T, w)  # shape (nCells)
    projections2 = np.dot(array2.T, w)  # shape (nCells)

    # Normalize w for visualization
    w_normalized = w / np.linalg.norm(w)

    # Prepare scatter plot canvas
    fig, scatter_ax = plt.subplots(figsize=(10, 7))
    scatter_ax.set_title(title)

    # Scatter original data
    scatter_ax.scatter(array1[0, :], array1[1, :], color='blue', alpha=0.5, label='Dataset 1')
    scatter_ax.scatter(array2[0, :], array2[1, :], color='orange', alpha=0.5, label='Dataset 2')

    # Scatter projections (scaled to line defined by w)
    scatter_ax.scatter(projections1 * w_normalized[0], projections1 * w_normalized[1],
                       color='darkblue', label='Projections (Dataset 1)')
    scatter_ax.scatter(projections2 * w_normalized[0], projections2 * w_normalized[1],
                       color='darkorange', label='Projections (Dataset 2)')

    # Draw projection vectors (optional, for clarity)
    for original, proj in zip(array1.T, projections1):
        proj_point = proj * w_normalized
        scatter_ax.plot([original[0], proj_point[0]], [original[1], proj_point[1]], color='blue', alpha=0.3, linestyle='--')
    for original, proj in zip(array2.T, projections2):
        proj_point = proj * w_normalized
        scatter_ax.plot([original[0], proj_point[0]], [original[1], proj_point[1]], color='orange', alpha=0.3, linestyle='--')

    # Add projection line (w) for reference
    line_xs = np.array([-100, 100]) * w_normalized[0]
    line_ys = np.array([-100, 100]) * w_normalized[1]
    scatter_ax.plot(line_xs, line_ys, color='black', linestyle='--', label='Projection Axis (w)')

    # Add labels and grid
    scatter_ax.set_xlabel('Original Dimension 1')
    scatter_ax.set_ylabel('Original Dimension 2')
    scatter_ax.grid(alpha=0.3)
    scatter_ax.legend(loc='upper left')

    # Inlay histogram for projections
    hist_ax = scatter_ax.inset_axes([0.65, 0.65, 0.3, 0.3])  # Position inset within main plot
    hist_ax.hist(projections1, bins=20, color='blue', alpha=0.7, label='Dataset 1', density=True)
    hist_ax.hist(projections2, bins=20, color='orange', alpha=0.7, label='Dataset 2', density=True)
    hist_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    hist_ax.set_title('Projection Histogram')
    hist_ax.set_xlabel('Projections')
    hist_ax.set_ylabel('Density')
    # hist_ax.legend(fontsize=8, loc='best')

    return fig


def add_significance_stars(ax, df, x_col, y_col, pairs_to_test=None):
    """Add significance stars above boxplot pairs"""

    # Get unique groups
    groups = df[x_col].unique()

    # If no specific pairs provided, test all combinations
    if pairs_to_test is None:
        pairs_to_test = list(itertools.combinations(range(len(groups)), 2))

    # Perform pairwise tests
    p_values = []
    for i, j in pairs_to_test:
        group1 = df[df[x_col] == groups[i]][y_col]
        group2 = df[df[x_col] == groups[j]][y_col]
        _, p = mannwhitneyu(group1, group2, alternative='two-sided')
        p_values.append(p)

    # Correct for multiple comparisons
    # _, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
    _, corrected_p, _, _ = multipletests(p_values, method='bonferroni')

    # Add significance annotations
    y_max = df[y_col].max()
    y_range = df[y_col].max() - df[y_col].min()

    for idx, (i, j) in enumerate(pairs_to_test):
        if corrected_p[idx] < 0.001:
            sig_text = '***'
        elif corrected_p[idx] < 0.01:
            sig_text = '**'
        elif corrected_p[idx] < 0.05:
            sig_text = '*'
        else:
            continue  # Skip non-significant comparisons

        # Calculate bracket height
        bracket_height = y_max + y_range * (0.05 + 0.05 * idx)

        # Draw bracket
        ax.plot([i, j], [bracket_height, bracket_height], 'k-', linewidth=1)
        ax.plot([i, i], [bracket_height, bracket_height - y_range * 0.01], 'k-', linewidth=1)
        ax.plot([j, j], [bracket_height, bracket_height - y_range * 0.01], 'k-', linewidth=1)

        # Add significance text
        ax.text((i + j) / 2, bracket_height + y_range * 0.01, sig_text,
                ha='center', va='bottom', fontsize=12, fontweight='bold')


def add_statistical_brackets(ax, df, x_col, y_col, test_pairs=None):
    """Add statistical comparison brackets to boxplot"""

    groups = df[x_col].unique()
    n_groups = len(groups)

    if test_pairs is None:
        test_pairs = list(itertools.combinations(range(n_groups), 2))

    # Perform statistical tests
    results = []
    for i, j in test_pairs:
        group1 = df[df[x_col] == groups[i]][y_col]
        group2 = df[df[x_col] == groups[j]][y_col]

        stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
        results.append({
            'group1_idx': i,
            'group2_idx': j,
            'group1_name': groups[i],
            'group2_name': groups[j],
            'p_value': p_val,
            'median1': group1.median(),
            'median2': group2.median()
        })

    # Correct for multiple comparisons
    p_values = [r['p_value'] for r in results]
    # _, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
    _, corrected_p, _, _ = multipletests(p_values, method='bonferroni')


    # Add corrected p-values to results
    for i, result in enumerate(results):
        result['corrected_p'] = corrected_p[i]
        result['significant'] = corrected_p[i] < 0.05

    # Draw brackets for significant comparisons
    y_max = df[y_col].max()
    y_min = df[y_col].min()
    y_range = y_max - y_min

    # Sort by distance between groups to avoid overlapping brackets
    significant_results = [r for r in results if r['significant']]
    significant_results.sort(key=lambda x: abs(x['group1_idx'] - x['group2_idx']))

    bracket_heights = {}
    base_height = y_max + y_range * 0.05

    for result in significant_results:
        i, j = result['group1_idx'], result['group2_idx']
        p_val = result['corrected_p']

        # Determine significance level
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        else:
            sig_text = '*'

        # Calculate bracket height to avoid overlaps
        max_height_in_range = base_height
        for existing_i, existing_j in bracket_heights.keys():
            if (min(i, j) <= max(existing_i, existing_j) and
                    max(i, j) >= min(existing_i, existing_j)):
                max_height_in_range = max(max_height_in_range,
                                          bracket_heights[(existing_i, existing_j)] + y_range * 0.08)

        bracket_height = max_height_in_range
        bracket_heights[(i, j)] = bracket_height

        # Draw the bracket
        ax.plot([i, j], [bracket_height, bracket_height], 'k-', linewidth=1.5)
        ax.plot([i, i], [bracket_height, bracket_height - y_range * 0.02], 'k-', linewidth=1.5)
        ax.plot([j, j], [bracket_height, bracket_height - y_range * 0.02], 'k-', linewidth=1.5)

        # Add significance text
        ax.text((i + j) / 2, bracket_height + y_range * 0.02, sig_text,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Adjust plot limits to accommodate brackets
    if bracket_heights:
        max_bracket = max(bracket_heights.values()) + y_range * 0.1
        ax.set_ylim(y_min - y_range * 0.05, max_bracket)

    return results

def participation_ratio(array):
    """
    Calculate participation ratio which is the sum squared divided by the sum of the squares
    """

    numerator = np.sum(array)**2
    denominator = np.sum(array**2)
    return numerator / denominator
