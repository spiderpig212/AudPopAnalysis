import numpy as np
import funcs
from jaratoolbox import celldatabase
import settings
import studyparams
import os
import pandas as pd
from analysis_class import FiringRateProcessing

fr_obj = FiringRateProcessing(db_suffix="coords_updated")
fr_obj.process_all_stimuli()
# dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
# figdataPath = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)
# os.makedirs(figdataPath, exist_ok=True)
# dbCoordsFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_coords.h5')
# celldb = celldatabase.load_hdf(dbCoordsFilename)
# simpleSiteNames = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])#.value_counts()
# simpleSiteNames.name = 'simpleSiteName'
# celldb = pd.concat([celldb, simpleSiteNames], axis=1)
#
# areas_of_interest = [
#     "Dorsal auditory area",
#     "Posterior auditory area",
#     "Primary auditory area",
#     "Ventral auditory area"
# ]
# celldb_subset = celldb[celldb['simpleSiteName'].isin(areas_of_interest)].reset_index()
#
# if 1:
#     stimType = 'naturalSound'
#     stimVar = 'soundID'
#     timeRange = [-2, 6]
#     allPeriods = [[-1, 0], [0, 0.5], [1, 4], [4, 4.5]]
#
#     print("Calculating firing rates for natural sounds")
#     fr_arrays = funcs.calculate_fr_arrays(celldb_subset, stimType, stimVar, timeRange, allPeriods)
#
#     fr_arrays_filename = os.path.join(figdataPath, f'fr_arrays_{stimType}.npz')
#     print(f"Saving firing rate arrays to {fr_arrays_filename}")
#     np.savez(fr_arrays_filename, basefr=fr_arrays[0], onsetfr=fr_arrays[1], sustainedfr=fr_arrays[2], offsetfr=fr_arrays[3],
#              stimArray=fr_arrays[4], brainRegionArray=fr_arrays[5], mouseIDArray=fr_arrays[6], sessionIDArray=fr_arrays[7])
#     print("Saved!")
#
# if 1:
#     stimType = 'AM'
#     stimVar = 'currentFreq'
#     timeRange = [-0.5, 1.5]
#     allPeriods = [[-0.5, 0], [0, 0.2], [0.2, 0.5], [0.5, 0.7]]
#
#     print("Calculating firing rates for AM")
#     fr_arrays = funcs.calculate_fr_arrays(celldb_subset, stimType, stimVar, timeRange, allPeriods)
#     fr_arrays_filename = os.path.join(figdataPath, f'fr_arrays_{stimType}.npz')
#     print(f"Saving firing rate arrays to {fr_arrays_filename}")
#     np.savez(fr_arrays_filename, basefr=fr_arrays[0], onsetfr=fr_arrays[1], sustainedfr=fr_arrays[2], offsetfr=fr_arrays[3],
#              stimArray=fr_arrays[4], brainRegionArray=fr_arrays[5], mouseIDArray=fr_arrays[6], sessionIDArray=fr_arrays[7])
#     print("Saved!")
#
# if 1:
#     stimType = 'pureTones'
#     stimVar = 'currentFreq'
#     timeRange = [-0.1, 0.3]
#     allPeriods = [[-0.1, 0], [0, 0.05], [0.05, 0.1], [0.1, 0.15]]
#
#     print("Calculating firing rates for pure tone sounds")
#     fr_arrays = funcs.calculate_fr_arrays(celldb_subset, stimType, stimVar, timeRange, allPeriods)
#     fr_arrays_filename = os.path.join(figdataPath, f'fr_arrays_{stimType}.npz')
#     print(f"Saving firing rate arrays to {fr_arrays_filename}")
#     np.savez(fr_arrays_filename, basefr=fr_arrays[0], onsetfr=fr_arrays[1], sustainedfr=fr_arrays[2], offsetfr=fr_arrays[3],
#              stimArray=fr_arrays[4], brainRegionArray=fr_arrays[5], mouseIDArray=fr_arrays[6], sessionIDArray=fr_arrays[7])
#     print("Saved!")