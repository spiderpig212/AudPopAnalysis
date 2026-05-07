"""
This file contains code for running canonical correlation analysis. Compares responses from different conditions in
various brain areas to see if there is any relationship between primary auditory area and either secondary cortical area.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for multiprocessing

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.svm import SVC, LinearSVR
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from analysis_class import FiringRateAnalysis


# ---------------------------------------------------------------------------
# Worker function: processes ONE (stim, respRange) combination.
# Must be defined at module level so it can be pickled and sent to workers.
# ---------------------------------------------------------------------------
def process_stim_resp(task, file_path):
    """
    Run the full CCA + SVM pipeline for a single (stim, respRange) combination.

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

            i_local = i
            if brainRegion == "Ventral auditory area":
                i_local -= 1

            for brainRegion2 in uniqRegions[i_local + 1:]:
                if brainRegion == "Ventral auditory area":
                    brainRegion2 = "Dorsal auditory area"
                    i_local += 2
                brainRegion2_mask = brain_session_array == brainRegion2
                brain2_resp_array = session_resp_array[brainRegion2_mask, :].T
                region2_sess_count = brain2_resp_array.shape[1]
                if region2_sess_count < neuron_threshold:
                    print(f"[{stim}/{respRange}] Skipping region 2: {brainRegion2} (n={region2_sess_count}), session {session}")
                    continue
                region2_neurons = rng.choice(brain2_resp_array.shape[1], size=neuron_threshold, replace=False)
                brain2_resp_array = brain2_resp_array[:, region2_neurons]

                significant_df = pd.read_csv(
                    f"{file_path}/CCA_two_region_analysis/cca_primary_auditory_results_backup.csv")
                # Get the significant components for the region pair, stimulus, response range, and session
                mask_n_comps = (
                        (significant_df["region1"] == brainRegion)
                        & (significant_df["region2"] == brainRegion2)
                        & (significant_df["stimulus"] == stim)
                        & (significant_df["response_range"] == respRange)
                        & (significant_df["session"] == session)
                )
                try:
                    n_components = significant_df.loc[mask_n_comps, "significant_components"].iloc[0]
                    n_components = np.int64(n_components)
                    if n_components < 1:
                        continue
                except (IndexError, ValueError):
                    print(f"[{stim}/{respRange}] No significant components found for {brainRegion} vs {brainRegion2}, "
                          f"session {session}")
                    continue
                br1_train, br1_test, br2_train, br2_test = train_test_split(
                    brain_resp_array, brain2_resp_array, test_size=0.2, random_state=42
                )
                # n_components = min(brain_resp_array.shape[1], brain2_resp_array.shape[1])  # TODO: This should be pulled from optimal fit as it should be d components
                cca = CCA(n_components=n_components)
                response_transform_source, response_transform_target = cca.fit_transform(
                    brain_resp_array, brain2_resp_array
                )

                if stim in ('naturalSound', 'AM', 'pureTones'):
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

                    if stim in ('AM', 'pureTones'):
                        # Have to take the log of the stim values to make it linear
                        C_results_svr = []
                        for c in C_values:
                            log_stims = np.log(stimArray)
                            svr_cca = LinearSVR(C=C_val, random_state=42)
                            svr = LinearSVR(C=C_val, random_state=42)
                            cv_scores_svr = cross_val_score(svr_cca, response_transform_source,
                                                            log_stims, cv=cv, scoring='neg_mean_squared_error')
                            cv_scores_svr_untransformed = cross_val_score(svr, brain_resp_array,
                                                                          log_stims, cv=cv,
                                                                          scoring='neg_mean_squared_error')
                            C_results_svr.append({
                                'C': C_val,
                                'cv_scores_svr_cca': cv_scores_svr,
                                'cv_scores_svr_untransformed': cv_scores_svr_untransformed,
                                'mean_accuracy_svr_cca': np.mean(cv_scores_svr),
                                'mean_accuracy_svr_untransformed': np.mean(cv_scores_svr_untransformed),
                                'std_accuracy_svr_cca': np.std(cv_scores_svr),
                                'std_accuracy_svr_untransformed': np.std(cv_scores_svr_untransformed),
                            })
                        best_svr = max(C_results_svr, key=lambda x: x['mean_accuracy_cca'])

                        correlation_data_svr.append({
                            'region_pair': f"{brainRegion}_vs_{brainRegion2}",
                            'region1': brainRegion,
                            'region2': brainRegion2,
                            'mean_accuracy_svr_cca': best_svr['mean_accuracy_svr_cca'],
                            'std_accuracy_svr_cca': best_svr['std_accuracy_svr_cca'],
                            'cv_scores_svr_cca': best_svr['cv_scores_svr_cca'],
                            'cv_scores_svr': best_svr['cv_scores_svr_untransformed'],
                            'cv_scores_svr_untransformed': best_svr['cv_scores_svr_untransformed'],
                            'n_trials_cca': len(response_transform_source),
                            'response_range': respRange,
                            'stimulus': stim,
                            'session': session,
                            'best_C': best_svr['C'],
                        })

                    for stim_idx, stim_val in tqdm(list(enumerate(uniqStims)),
                                                   desc=f"{stim}/{respRange}/{session}",
                                                   leave=False):
                        for stim_idx2, stim_val2 in enumerate(uniqStims):
                            if stim_idx == stim_idx2:
                                continue
                            stim_mask = stimArray == stim_val
                            stim_mask2 = stimArray == stim_val2
                            combined_mask = stim_mask | stim_mask2
                            combined_resp_array_source = response_transform_source[combined_mask, :]
                            combined_resp_array_target = response_transform_target[combined_mask, :]
                            combined_masked_resp_array = brain_resp_array[combined_mask, :]
                            masked_stims = stimArray[combined_mask].astype(int)

                            C_results = []
                            for C_val in C_values:
                                svm_cca = SVC(C=C_val, random_state=42, kernel='linear')
                                svm = SVC(C=C_val, random_state=42, kernel='linear')
                                cv_scores = cross_val_score(svm_cca, combined_resp_array_source,
                                                            masked_stims, cv=cv, scoring='accuracy')
                                chance_level = 0.5
                                cv_scores_untransformed = cross_val_score(svm, combined_masked_resp_array,
                                                                          masked_stims, cv=cv, scoring='accuracy')
                                C_results.append({
                                    'C': C_val,
                                    'cv_scores_cca': cv_scores,
                                    'cv_scores': cv_scores_untransformed,
                                    'mean_accuracy_cca': np.mean(cv_scores),
                                    'mean_accuracy': np.mean(cv_scores_untransformed),
                                    'std_accuracy_cca': np.std(cv_scores),
                                    'std_accuracy': np.std(cv_scores_untransformed),
                                })

                            C_df = pd.DataFrame(C_results)
                            best = max(C_results, key=lambda x: x['mean_accuracy_cca'])
                            best_C_val = best['C']
                            pca = PCA(n_components=n_components, random_state=42)
                            transformed_resp_data = pca.fit_transform(combined_masked_resp_array)
                            svm_pca = SVC(C=best_C_val, random_state=42, kernel='linear')
                            pca_scores = cross_val_score(svm_pca, transformed_resp_data, masked_stims, cv=cv,
                                                         scoring='accuracy')
                            # pca_train, pca_test, stim_train, stim_test = train_test_split(transformed_resp_data,
                            #                                                               masked_stims, test_size=0.2,
                            #                                                               random_state=42)
                            # pca_accuracy = svc_pca.score(pca_test, stim_test)

                            # Hyperparameter sweep plot
                            fig = plt.figure(figsize=(10, 6))
                            plt.subplot(1, 2, 1)
                            plt.plot(C_values, C_df['mean_accuracy_cca'], 'bo-')
                            plt.xlabel('C'); plt.ylabel('Mean Accuracy')
                            plt.title('Hyperparameter Sweep for C')
                            plt.subplot(1, 2, 2)
                            plt.plot(C_values, C_df['std_accuracy_cca'], 'bo-')
                            plt.xlabel('C'); plt.ylabel('Std of Accuracy')
                            plt.title('Hyperparameter Sweep for C')
                            plt.tight_layout()
                            plt.savefig(f"{file_path}/CCA_SVC/CCA_SVC_C_plots/"
                                        f"CCA_SVC_hyperparameter_sweep_{session}_{stim}_{respRange}.png")
                            plt.close(fig)

                            boundary_key = (f"{stim}_{respRange}_{brainRegion}_vs_{brainRegion2}_"
                                            f"{session}_{stim_val}_{stim_val2}_C{best_C_val}")
                            if boundary_key not in decision_boundaries_data and len(decision_boundaries_data) < 5:
                                fold_iterator = cv.split(combined_resp_array_source, masked_stims)
                                train_idx, test_idx = next(fold_iterator)
                                X_train_fold = combined_resp_array_source[train_idx]
                                y_train_fold = masked_stims[train_idx]
                                X_test_fold = combined_resp_array_source[test_idx]
                                y_test_fold = masked_stims[test_idx]

                                svm_boundary = SVC(C=best_C_val, random_state=42, kernel='linear')
                                svm_boundary.fit(X_train_fold, y_train_fold)
                                y_pred_fold = svm_boundary.predict(X_test_fold)

                                decision_boundaries_data[boundary_key] = {
                                    'X_train': X_train_fold,
                                    'y_train': y_train_fold,
                                    'X_test': X_test_fold,
                                    'y_test': y_test_fold,
                                    'y_pred': y_pred_fold,
                                    'svm_model': svm_boundary,
                                    'region_pair': f"{brainRegion}_vs_{brainRegion2}",
                                    'stim_type': stim,
                                    'response_range': respRange,
                                    'session': session,
                                    'stim_pair': (stimVals[int(stim_val)], stimVals[int(stim_val2)])
                                                   if stimVals is not None else (stim_val, stim_val2),
                                    'C': best_C_val,
                                }

                            correlation_data.append({
                                'region_pair': f"{brainRegion}_vs_{brainRegion2}",
                                'region1': brainRegion,
                                'region2': brainRegion2,
                                'mean_accuracy_cca': best['mean_accuracy_cca'],
                                'mean_accuracy_pca': np.mean(pca_scores),
                                'mean_accuracy': best['mean_accuracy'],
                                'std_accuracy_cca': best['std_accuracy_cca'],
                                'std_accuracy_pca': np.std(pca_scores),
                                'std_accuracy': best['std_accuracy'],
                                'cv_scores_cca': best['cv_scores_cca'],
                                'cv_scores': best['cv_scores'],
                                'cv_scores_pca': pca_scores,
                                'n_trials_cca': len(combined_resp_array_source),
                                'n_classes': 2,
                                'stim_pair': (stim_val, stim_val2),
                                'chance_level': chance_level,
                                'response_range': respRange,
                                'stimulus': stim,
                                'session': session,
                                'C': best_C_val,
                            })

    return stim, respRange, correlation_data, decision_boundaries_data, correlation_data_svr


# ---------------------------------------------------------------------------
# Driver: must be guarded by __main__ for multiprocessing on Windows/macOS.
# ---------------------------------------------------------------------------
def main():
    response_ranges = ["onset", "sustained", "offset"]
    neuron_threshold = 20
    n_splits = 5
    C_values = np.logspace(-3, 1.2, 20)

    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path
    stim_types = fr_db.stim_types

    # --- Build all tasks, loading data ONCE per stim in the parent process ---
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

        per_stim_meta[stim] = {"correlation_data": []}
        per_stim_meta[stim] = {"correlation_data_svr": []}

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

    decision_boundaries_data = {}

    # Cap workers — sklearn already uses BLAS threads internally, so don't oversubscribe.
    max_workers = min(len(tasks), max(1, (os.cpu_count() or 2) // 2))
    print(f"Launching {len(tasks)} tasks across {max_workers} worker processes")

    # TODO: Look at removing raise call so that way one process failing doesn't kill the others before they finish running
    #  Viz file right now loads in each stim which contains all response ranges so may need to adjust for what response ranges we do have
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_stim_resp, t, file_path=file_path): (t["stim"], t["respRange"]) for t in tasks}
        for fut in as_completed(futures):
            stim, respRange = futures[fut]
            try:
                stim_out, resp_out, corr_data, db_data, corr_data_svr = fut.result()
            except Exception as e:
                print(f"Task {stim}/{respRange} failed: {e!r}")
                raise
            per_stim_meta[stim_out]["correlation_data"].extend(corr_data)
            per_stim_meta[stim_out]["correlation_data_svr"].extend(corr_data_svr)
            # Merge boundary dict (keys already include stim/respRange so no collisions)
            for k, v in db_data.items():
                if k not in decision_boundaries_data and len(decision_boundaries_data) < 5:
                    decision_boundaries_data[k] = v
            print(f"Finished {stim_out}/{resp_out} ({len(corr_data)} rows)")

    # --- Persist per-stim outputs (parent only — no concurrent writes) ---
    for stim, meta in per_stim_meta.items():
        df = pd.DataFrame(meta["correlation_data"])
        df.to_feather(f"{file_path}/CCA_SVC/CCA_SVM_{stim}.feather")
        df.to_csv(f"{file_path}/CCA_SVC/CCA_SVM_{stim}.csv", index=False)
        df_svr = pd.DataFrame(meta["correlation_data_svr"])
        df_svr.to_feather(f"{file_path}/CCA_SVR/CCA_SVR_{stim}.feather")
        df_svr.to_csv(f"{file_path}/CCA_SVR/CCA_SVR_{stim}.csv", index=False)

    with open(f"{file_path}/decision_boundaries_data.pkl", 'wb') as f:
        pickle.dump(decision_boundaries_data, f)


if __name__ == "__main__":
    main()
