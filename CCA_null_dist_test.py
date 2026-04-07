"""
Uses the number of significant components found in Two_region_CCA.py to select the first d components to calculate a
null distribution. The first d U and V components of each brain region are compared with a cosine similarity test,
followed by using a normal distribution with variance equal to covariance matrix of primary AC. For the null, we sample
U_hat d times and then perform QR decomposition on the resulting matrix to create orthogonal U's. Do the same for second
brain region to generate V's. Then do cosine similarity test and record similarity value. Do this 1000 times and compare
our true similarity value with the null distribution.
"""

import os
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from itertools import combinations

from analysis_class import FiringRateAnalysis
from Two_region_CCA import TwoRegionCCAAnalysis


def calculate_cca_cosine_similarity_with_null(analyzer, session_sig_df, n_null_samples=100):
    """
    Compare CCA projections from the same source region to different target regions
    within the same session. For example, compare "Primary vs Ventral" and "Primary vs Dorsal"
    to see how similarly the Primary region projects to different areas.

    Parameters:
    -----------
    analyzer : TwoRegionCCAAnalysis
        The analyzer instance
    session_sig_df : pandas.DataFrame
        DataFrame with columns: region1, region2, session, significant_components
    n_null_samples : int
        Number of null samples to generate for significance testing

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with cosine similarity results and null distribution statistics
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.stats import percentileofscore

    results = []

    # Group by session and source region (region1) to find pairs
    grouped = session_sig_df.groupby(['date', 'region1'])

    for (session, source_region), group in tqdm(grouped, desc="Processing session-region groups"):
        # Get all target regions for this source region in this session
        target_regions = group['region2'].unique()

        if len(target_regions) < 2:
            # Need at least 2 target regions to compare
            print("Skipping session:", session, "Source region:", source_region, "No enough target regions.")
            continue

        # Get all possible pairs of target regions
        for target1, target2 in combinations(target_regions, 2):
            for stimulus in ['pureTones', 'naturalSound', 'AM']:
                for response_range in ['onset', 'sustained', 'offset']:
                    try:
                        # Get data for both region pairs
                        frame_1 = group[(group['region2'] == target1) & (group['stimulus'] == stimulus) & (
                                    group['response_range'] == response_range)]
                        frame_2 = group[(group['region2'] == target2) & (group['stimulus'] == stimulus) & (
                                    group['response_range'] == response_range)]
                        # pair1_row = group[group['region2'] == target1].iloc[0]
                        # pair2_row = group[group['region2'] == target2].iloc[0]

                        d_components_u = int(np.floor(frame_1['significant_components'].mean()))
                        d_components_v = int(np.floor(frame_2['significant_components'].mean()))

                        if d_components_u == 0 or d_components_v == 0:
                            print("Skipping {source_region} -> {target1} vs {target2}, session {session}: No significant components")
                            continue

                        # # Use the minimum number of components between the two pairs
                        # d_components = min(d_components_u, d_components_v)
                        #
                        # if d_components == 0:
                        #     continue

                        # Get stimulus and response_range info (assuming they're the same for both pairs)
                        # stimulus = pair1_row.get('stimulus', 'naturalSound')  # Default fallback
                        # response_range = pair1_row.get('response_range', 'sustainedfr')  # Default fallback

                        # Load data for first region pair (source_region vs target1)
                        source_data_u = analyzer.fr_db.get_firing_rate_array(
                            source_region, session, stimulus, response_range, analyzer.neuron_threshold, analyzer.random_state
                        )
                        if target1 == 'Dorsal auditory area':
                            try:
                                target1_data_u = analyzer.fr_db.get_firing_rate_array(
                                    target1, session, stimulus, response_range, analyzer.neuron_threshold, analyzer.random_state
                                )
                            except UnboundLocalError:  # This is for handling cases where we renamed posterior to dorsal
                                target1_data_u = analyzer.fr_db.get_firing_rate_array(
                                    'Posterior auditory area', session, stimulus, response_range, analyzer.neuron_threshold,
                                    analyzer.random_state
                                )
                        else:
                            target1_data_u = analyzer.fr_db.get_firing_rate_array(
                                target1, session, stimulus, response_range, analyzer.neuron_threshold,
                                analyzer.random_state
                            )

                        # Load data for second region pair (source_region vs target2)
                        source_data_v = analyzer.fr_db.get_firing_rate_array(
                            source_region, session, stimulus, response_range, analyzer.neuron_threshold, analyzer.random_state
                        )
                        if target2 == 'Dorsal auditory area':
                            try:
                                target2_data_v = analyzer.fr_db.get_firing_rate_array(
                                    target2, session, stimulus, response_range, analyzer.neuron_threshold,
                                    analyzer.random_state
                                )
                            except UnboundLocalError:  # This is for handling cases where we renamed posterior to dorsal
                                target2_data_v = analyzer.fr_db.get_firing_rate_array(
                                    'Posterior auditory area', session, stimulus, response_range,
                                    analyzer.neuron_threshold,
                                    analyzer.random_state
                                )
                        else:
                            target2_data_v = analyzer.fr_db.get_firing_rate_array(
                                target2, session, stimulus, response_range, analyzer.neuron_threshold,
                                analyzer.random_state
                            )

                        if any(data is None for data in [source_data_u, target1_data_u, source_data_v, target2_data_v]):
                            print(f"Missing data for {source_region} comparisons in session {session}")
                            continue

                        # Ensure consistent trial numbers
                        min_trials_u = min(source_data_u.shape[0], target1_data_u.shape[0])
                        min_trials_v = min(source_data_v.shape[0], target2_data_v.shape[0])
                        min_trials = min(min_trials_u, min_trials_v)

                        X_u = source_data_u[:min_trials, :]
                        Y_u = target1_data_u[:min_trials, :]
                        X_v = source_data_v[:min_trials, :]
                        Y_v = target2_data_v[:min_trials, :]

                        # Fit CCA for both region pairs
                        cca_u = CCA(n_components=d_components_u, max_iter=1000)
                        cca_u.fit(X_u, Y_u)

                        cca_v = CCA(n_components=d_components_v, max_iter=1000)
                        cca_v.fit(X_v, Y_v)

                        # Get the x_weights (source region projections) from both models
                        U_actual = cca_u.x_weights_  # Shape is (n,d_u) Source region -> target1 projections
                        V_actual = cca_v.x_weights_  # Shape is (n,d_v)Source region -> target2 projections

                        # Check if same number of neurons in source region
                        if U_actual.shape[0] != V_actual.shape[0]:
                            print(f"Different neuron counts for {source_region}: {U_actual.shape[0]} vs {V_actual.shape[0]}")
                            continue

                        U_norm_sq = np.linalg.norm(U_actual)**2  # Default is axis=None which returns frobenius nmatrix norm
                        V_norm_sq = np.linalg.norm(V_actual)**2
                        subspace_numerator = np.linalg.norm(U_actual.T @ V_actual)**2
                        R_actual = subspace_numerator / np.min([U_norm_sq, V_norm_sq])


                        # Calculate actual cosine similarity between source region projections
                        # actual_similarities = []
                        # for i in range(d_components):
                        #     u_component = U_actual[:, i].reshape(1, -1)
                        #     v_component = V_actual[:, i].reshape(1, -1)
                        #     cos_sim = cosine_similarity(u_component, v_component)[0, 0]
                        #     actual_similarities.append(cos_sim)

                        # R_actual = np.mean(actual_similarities)

                        # Calculate covariance matrix for the source region (same for both pairs)
                        # Use the data from the first pair (X_u) as reference
                        cov_source = np.cov(X_u.T)  # Shape: (n_neurons_source, n_neurons_source)

                        # Generate null distribution
                        null_similarities = []

                        for null_iter in range(n_null_samples):
                            # Sample d vectors from multivariate normal distribution for source region
                            # Generate two independent sets of random vectors
                            U_hat = np.random.multivariate_normal(
                                mean=np.zeros(cov_source.shape[0]),
                                cov=cov_source,
                                size=d_components_u
                            ).T  # Shape: (n_neurons_source, d_components)

                            V_hat = np.random.multivariate_normal(
                                mean=np.zeros(cov_source.shape[0]),
                                cov=cov_source,
                                size=d_components_v
                            ).T  # Shape: (n_neurons_source, d_components)

                            # QR decomposition to orthonormalize (ignore the upper triangle, R)
                            U_null, _ = np.linalg.qr(U_hat)  # Shape: (n_neurons_source, d_components)
                            V_null, _ = np.linalg.qr(V_hat)  # Shape: (n_neurons_source, d_components)

                            U_null_norm_sq = np.linalg.norm(
                                U_null) ** 2  # Default is axis=None which returns frobenius nmatrix norm
                            V_null_norm_sq = np.linalg.norm(V_null) ** 2
                            null_subspace_numerator = np.linalg.norm(U_null.T @ V_null) ** 2
                            R_null = null_subspace_numerator / np.min([U_null_norm_sq, V_null_norm_sq])
                            # # Calculate cosine similarity for this null sample
                            # null_cos_similarities = []
                            # for i in range(d_components):
                            #     u_null_component = U_null[:, i].reshape(1, -1)
                            #     v_null_component = V_null[:, i].reshape(1, -1)
                            #     null_cos_sim = cosine_similarity(u_null_component, v_null_component)[0, 0]
                            #     null_cos_similarities.append(null_cos_sim)
                            #
                            # R_null = np.mean(null_cos_similarities)
                            null_similarities.append(R_null)

                        # Calculate p-value and statistics
                        null_similarities = np.array(null_similarities)
                        # p_value = (np.sum(null_similarities >= R_actual)) / n_null_samples
                        null_center = np.mean(null_similarities)
                        p_value = np.sum(np.abs(null_similarities - null_center) >= np.abs(R_actual - null_center)) / n_null_samples
                        # p_value = (np.sum(np.abs(null_similarities) >= np.abs(R_actual))) / n_null_samples

                        null_mean = np.mean(null_similarities)
                        null_std = np.std(null_similarities)
                        # null_95_percentile = np.percentile(null_similarities, 95)
                        # null_99_percentile = np.percentile(null_similarities, 99)
                        z = (R_actual - null_mean) / null_std

                        # Store results
                        results.append({
                            'source_region': source_region,
                            'target1': target1,
                            'target2': target2,
                            'session': session,
                            'stimulus': stimulus,
                            'response_range': response_range,
                            'significant_components': np.min([d_components_u, d_components_v]),
                            'R_actual': R_actual,
                            'null_mean': null_mean,
                            'null_std': null_std,
                            'z_score_diff_from_null': z,
                            # 'null_95_percentile': null_95_percentile,
                            # 'null_99_percentile': null_99_percentile,
                            'p_value': p_value,
                            'is_significant_95': p_value < 0.05,
                            'is_significant_99': p_value < 0.01,
                            'null_similarities': null_similarities.tolist(),
                            'U_shape': U_actual.shape,
                            'V_shape': V_actual.shape,
                            'pair1_components': d_components_u,
                            'pair2_components': d_components_v
                        })

                    except Exception as e:
                        print(f"Error processing {source_region} -> {target1} vs {target2}, session {session}: {e}")
                        continue

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    # Initialize analysis
    neuron_threshold = 40
    analyzer = TwoRegionCCAAnalysis(neuron_threshold=neuron_threshold, n_splits=5, random_state=42,
                                    n_permutations=10000)
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path

    # significant_df = pd.read_feather(f"{file_path}/CCA_two_region_analysis/cca_primary_auditory_results.feather")
    significant_df = pd.read_csv(f"{file_path}/CCA_two_region_analysis/cca_primary_auditory_results.csv")
    session_sig_df = significant_df # [
    #     significant_df["region2"] != "Primary auditory area"]  # Filter out the primary vs primary results

    # Need to average the different session iterations together
    session_sig_df['date'] = significant_df['session'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0]

    # Calculate cosine similarities with null distribution
    print("Starting cross-region projection similarity analysis...")
    null_results = calculate_cca_cosine_similarity_with_null(analyzer, session_sig_df, n_null_samples=100)

    # Display results
    print(f"\nProcessed {len(null_results)} region pair comparisons")

    if len(null_results) > 0:
        significant_results = null_results[null_results['is_significant_95']]
        print(f"Found {len(significant_results)} significant projection similarity comparisons (p < 0.05)")

        if len(significant_results) > 0:
            print("\nSignificant Results:")
            print(significant_results[['source_region', 'target1', 'target2', 'session', 'R_actual', 'p_value']])

        # Save results
        null_results.to_feather(f"{file_path}/CCA_cross_region_projection_similarity.feather")
        print(f"\nResults saved to: {file_path}/CCA_cross_region_projection_similarity.feather")
        null_results.to_csv(f"{file_path}/CCA_cross_region_projection_similarity.csv")
    else:
        print("No valid comparisons found. Check data availability and session groupings.")