"""
Goes through each brain area, response range, and stim type to compute the SVM classification accuracy for firing rate data.
For pureTones and AM we also use SVR since they are linearly separable.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for multiprocessing

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.svm import SVC, LinearSVR
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from analysis_class import FiringRateAnalysis

def process_stim_resp(task, file_path):
    """
    Run the full SVM pipeline for a single (stim, respRange) combination.

    Parameters
    ----------
    task : dict with keys:
        stim, respRange, file_path, neuron_threshold, n_splits,
        C_values, stim_info, stim_arrays, uniqRegions, uniqSessions,
        uniqStims, stimVals (or None)

    Returns
    -------
    (stim, respRange, correlation_data_list, decision_boundaries_dict)
    """
    stim = task["stim"]
    respRange = task["respRange"]
    file_path = task["file_path"]
    neuron_threshold = task["neuron_threshold"]
    n_splits = task["n_splits"]
    C_values = task["C_values"]
    stim_arrays = task["stim_arrays"]
    uniqRegions = task["uniqRegions"]
    uniqSessions = task["uniqSessions"]
    uniqStims = task["uniqStims"]
    stimVals = task["stimVals"]

    brainRegionArray = stim_arrays["brainRegionArray"].copy()
    brainRegionArray[brainRegionArray == "Posterior auditory area"] = "Dorsal auditory area"
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    respArray = stim_arrays[f"{respRange}fr"]

    # Deterministic seed per task so results are reproducible across runs
    rng = np.random.default_rng(abs(hash((stim, respRange))) % (2**32))

    correlation_data = []
    decision_boundaries_data = {}
    correlation_data_svr = []
    correlation_data_RDM = []

    for session in uniqSessions:
        session_mask = sessionArray == session
        session_resp_array = respArray[session_mask, :]
        brain_session_array = brainRegionArray[session_mask]

        for i, brainRegion in enumerate(uniqRegions):
            brainRegion_mask = brain_session_array == brainRegion
            brain_resp_array = session_resp_array[brainRegion_mask, :].T
            region1_sess_count = brain_resp_array.shape[1]
            if region1_sess_count < neuron_threshold:
                print(f"[{stim}/{respRange}] Skipping region 1: {brainRegion} (n={region1_sess_count}), session {session}")
                continue
            region1_neurons = rng.choice(brain_resp_array.shape[1], size=neuron_threshold, replace=False)
            brain_resp_array = brain_resp_array[:, region1_neurons]
            if stim in ('naturalSound', 'AM', 'pureTones'):
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_cont = KFold(n_splits=n_splits, shuffle=True, random_state=42)

                if stim in ('AM', 'pureTones'):
                    # Have to take the log of the stim values to make it linear
                    C_results_svr = []
                    for c in C_values:
                        log_stims = np.log(stimArray)
                        svr = LinearSVR(C=c, random_state=42)
                        cv_scores_svr_untransformed = cross_val_score(svr, brain_resp_array,
                                                                      log_stims, cv=cv_cont,
                                                                      scoring='neg_mean_squared_error')
                        C_results_svr.append({
                            'C': c,
                            'cv_scores_svr_untransformed': cv_scores_svr_untransformed*-1,  # Multiply by -1 since sklearn returns negative MSE
                            'mean_accuracy_svr_untransformed': np.mean(cv_scores_svr_untransformed),
                            'std_accuracy_svr_untransformed': np.std(cv_scores_svr_untransformed),
                        })
                    best_svr = max(C_results_svr, key=lambda x: x['mean_accuracy_svr_untransformed'])

                    correlation_data_svr.append({
                        'region1': brainRegion,
                        'mean_accuracy_svr_untransformed': best_svr['mean_accuracy_svr_untransformed'],
                        'std_accuracy_svr_untransformed': best_svr['std_accuracy_svr_untransformed'],
                        'cv_scores_svr_untransformed': best_svr['cv_scores_svr_untransformed'],
                        'response_range': respRange,
                        'stimulus': stim,
                        'session': session,
                        'best_C': best_svr['C'],
                    })

                stim_length = len(uniqStims)
                corr_mat = np.ones((stim_length, stim_length))
                for stim_idx, stim_val in tqdm(list(enumerate(uniqStims)),
                                               desc=f"{stim}/{respRange}/{session}",
                                               leave=False):
                    for stim_idx2, stim_val2 in enumerate(uniqStims):
                        if stim_idx == stim_idx2:
                            continue
                        stim_mask = stimArray == stim_val
                        stim_mask2 = stimArray == stim_val2
                        combined_mask = stim_mask | stim_mask2
                        combined_masked_resp_array = brain_resp_array[combined_mask, :]
                        masked_stims = stimArray[combined_mask].astype(int)

                        C_results = []
                        for C_val in C_values:
                            svm = SVC(C=C_val, random_state=42, kernel='linear')
                            chance_level = 0.5
                            cv_scores_untransformed = cross_val_score(svm, combined_masked_resp_array,
                                                                      masked_stims, cv=cv, scoring='accuracy')
                            C_results.append({
                                'C': C_val,
                                'cv_scores': cv_scores_untransformed,
                                'mean_accuracy': np.mean(cv_scores_untransformed),
                                'std_accuracy': np.std(cv_scores_untransformed),
                            })

                        C_df = pd.DataFrame(C_results)
                        best = max(C_results, key=lambda x: x['mean_accuracy'])
                        best_C_val = best['C']

                        # Hyperparameter sweep plot
                        fig = plt.figure(figsize=(10, 6))
                        plt.subplot(1, 2, 1)
                        plt.plot(C_values, C_df['mean_accuracy'], 'bo-')
                        plt.xlabel('C'); plt.ylabel('Mean Accuracy')
                        plt.title('Hyperparameter Sweep for C')
                        plt.subplot(1, 2, 2)
                        plt.plot(C_values, C_df['std_accuracy'], 'bo-')
                        plt.xlabel('C'); plt.ylabel('Std of Accuracy')
                        plt.title('Hyperparameter Sweep for C')
                        plt.tight_layout()
                        plt.savefig(f"{file_path}/SVC/SVC_C_plots/"
                                    f"SVC_hyperparameter_sweep_{session}_{stim}_{respRange}.png")
                        plt.close(fig)


                        correlation_data.append({
                            'region1': brainRegion,
                            'mean_accuracy': best['mean_accuracy'],
                            'std_accuracy': best['std_accuracy'],
                            'cv_scores': best['cv_scores'],
                            'n_classes': 2,
                            'stim_pair': (stim_val, stim_val2),
                            'chance_level': chance_level,
                            'response_range': respRange,
                            'stimulus': stim,
                            'session': session,
                            'C': best_C_val,
                        })

                        # RDM calculations for per-stim work. We go through stim A and stim B, calculate the average firing
                        # rate per neuron, end up with r_vec which is shape (n_neurons), so averaged over trials with
                        # one each for A and B, and then calculate Pearson correlation between r_vec_a and r_vec_b. Store in
                        # matrix for stim pairs where the matrix vals will be symmetrical so only really need upper_tri

                        r_a = brain_resp_array[stim_mask, :]
                        r_b = brain_resp_array[stim_mask2, :]
                        r_vec_a = np.mean(r_a, axis=0)
                        r_vec_b = np.mean(r_b, axis=0)
                        pearson_r = stats.pearsonr(r_vec_a, r_vec_b)[0]
                        corr_mat[stim_idx, stim_idx2] = pearson_r
                    correlation_data_RDM.append({
                        'region1': brainRegion,
                        'session': session,
                        'stimulus': stim,
                        'corr_mat': corr_mat,
                        'corr_dims': (stim_length, stim_length),
                        'corr_mat_flattened': corr_mat.flatten(),
                    })


    return stim, respRange, correlation_data, correlation_data_svr, correlation_data_RDM

def main():
    response_ranges = ["onset", "sustained", "offset"]
    neuron_threshold = 20
    n_splits = 5
    C_values = np.logspace(-3, 1.2, 20)

    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path
    stim_types = fr_db.stim_types

    tasks = []
    per_stim_meta = {}
    for stim in stim_types:
        stim_info = fr_db.stim_info[stim]
        stim_arrays = fr_db.return_arrays(stim)

        brainRegionArray = stim_arrays["brainRegionArray"].copy()
        brainRegionArray[brainRegionArray == "Posterior auditory area"] = "Dorsal auditory area"
        uniqRegions = np.unique(brainRegionArray)
        reorder = np.array([1, 0, 2, 3])
        uniqRegions = uniqRegions[reorder]
        uniqSessions = np.unique(stim_arrays["sessionIDArray"])
        uniqStims = np.unique(stim_arrays["stimArray"][0, :])
        stimVals = stim_info.get('stimVals') if stim == 'naturalSound' else None

        per_stim_meta[stim] = {"correlation_data": [],
                               "correlation_data_svr": [],
                               "RDM_data": [],
                               }

        for respRange in response_ranges:
            tasks.append({
                "stim": stim,
                "respRange": respRange,
                "file_path": file_path,
                "neuron_threshold": neuron_threshold,
                "n_splits": n_splits,
                "C_values": C_values,
                "stim_arrays": stim_arrays,   # numpy arrays pickle cheaply
                "uniqRegions": uniqRegions,
                "uniqSessions": uniqSessions,
                "uniqStims": uniqStims,
                "stimVals": stimVals,
            })

    # Cap workers — sklearn already uses BLAS threads internally, so don't oversubscribe.
    max_workers = min(len(tasks), max(1, (os.cpu_count() or 2) // 2))
    print(f"Launching {len(tasks)} tasks across {max_workers} worker processes")

    # TODO: Look at removing raise call so that way one process failing doesn't kill the others before they finish running
    #  Viz file right now loads in each stim which contains all response ranges so may need to adjust for what response ranges we do have
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_stim_resp, t, file_path=file_path): (t["stim"], t["respRange"]) for t in
                   tasks}
        for fut in as_completed(futures):
            stim, respRange = futures[fut]
            try:
                stim_out, resp_out, corr_data, corr_data_svr, corr_data_rdm = fut.result()
            except Exception as e:
                print(f"Task {stim}/{respRange} failed: {e!r}")
                raise
            per_stim_meta[stim_out]["correlation_data"].extend(corr_data)
            per_stim_meta[stim_out]["correlation_data_svr"].extend(corr_data_svr)
            per_stim_meta[stim_out]["RDM_data"].extend(corr_data_rdm)
            print(f"Finished {stim_out}/{resp_out} ({len(corr_data)} rows)")

    # --- Persist per-stim outputs (parent only — no concurrent writes) ---
    for stim, meta in per_stim_meta.items():
        df = pd.DataFrame(meta["correlation_data"])
        df.to_feather(f"{file_path}/SVC/SVC_{stim}.feather")
        df.to_csv(f"{file_path}/SVC/SVC_{stim}.csv", index=False)
        df_svr = pd.DataFrame(meta["correlation_data_svr"])
        df_svr.to_feather(f"{file_path}/SVR/SVR_{stim}.feather")
        df_svr.to_csv(f"{file_path}/SVR/SVR_{stim}.csv", index=False)
        try:
            df_rdm = pd.DataFrame(meta["RDM_data"])
            df_rdm.to_csv(f"{file_path}/RDM/RDM_{stim}.csv", index=False)
            df_rdm.to_feather(f"{file_path}/RDM/RDM_{stim}.feather")
        except:
            # If RDM can't be put into a dataframe just drop a pickle file
            with open(f"{file_path}/RDM/RDM_{stim}.pkl", 'wb') as f:
                pickle.dump(meta["RDM_data"], f)

if __name__ == "__main__":
    main()