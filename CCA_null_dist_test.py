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

from analysis_class import FiringRateAnalysis
from Two_region_CCA import TwoRegionCCAAnalysis


def calculate_cca_cosine_similarity_with_null(self, session_sig_df, n_null_samples=100):
    """
    For each region pair, perform CCA and calculate cosine similarity between
    the first 'd' components of U (x_weights_) and V (y_weights_) matrices,
    along with a null distribution for significance testing.

    Parameters:
    -----------
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

    for idx, row in tqdm(session_sig_df.iterrows(), total=len(session_sig_df)):
        region1 = row['region1']
        region2 = row['region2']
        session = row['session']
        stimulus = row['stimulus']
        response_range = row['response_range']
        d_components = int(row['significant_components'])

        if d_components == 0:
            # Skip if no significant components
            continue

        try:
            # Get firing rate data for both regions
            region1_data = self.fr_db.get_firing_rate_array(
                region1, session, stimulus, response_range, self.neuron_threshold, self.random_state
            )  # Shape is [trials, neurons]
            region2_data = self.fr_db.get_firing_rate_array(
                region2, session, stimulus, response_range, self.neuron_threshold, self.random_state
            )  # Shape is [trials, neurons]

            if region1_data is None or region2_data is None:
                print(f"Missing data for {region1} vs {region2}, session {session}")
                continue

            # Ensure both datasets have the same number of time points
            min_timepoints = min(region1_data.shape[0], region2_data.shape[0])
            X = region1_data[:min_timepoints, :]
            Y = region2_data[:min_timepoints, :]

            # Transpose for CCA (samples x features)
            # X = region1_data  # Shape: (n_samples, n_neurons_region1)
            # Y = region2_data  # Shape: (n_samples, n_neurons_region2)

            # Fit CCA with d components to get actual similarity
            cca = CCA(n_components=d_components, max_iter=1000)
            cca.fit(X, Y)

            # Get the first d components of U and V weight matrices
            U_actual = cca.x_weights_[:, :d_components]  # Shape: (n_neurons_region1, d_components)
            V_actual = cca.y_weights_[:, :d_components]  # Shape: (n_neurons_region2, d_components)

            # Calculate actual cosine similarity
            if U_actual.shape[0] == V_actual.shape[0]:  # Same number of neurons
                actual_similarities = []
                for i in range(d_components):
                    u_component = U_actual[:, i].reshape(1, -1)
                    v_component = V_actual[:, i].reshape(1, -1)
                    cos_sim = cosine_similarity(u_component, v_component)[0, 0]
                    actual_similarities.append(cos_sim)

                R_actual = np.mean(actual_similarities)

                # Calculate covariance matrices for null distribution
                # Covariance of region1 data
                # np.cov assumes observations are in the columns so I think we need to transpose again
                cov_region1 = np.cov(X.T)  # Shape: (n_neurons_region1, n_neurons_region1)
                # Covariance of region2 data
                cov_region2 = np.cov(Y.T)  # Shape: (n_neurons_region2, n_neurons_region2)

                # Generate null distribution
                null_similarities = []

                for null_iter in range(n_null_samples):
                    # Sample d vectors from multivariate normal distribution for region1
                    U_hat = np.random.multivariate_normal(
                        mean=np.zeros(cov_region1.shape[0]),
                        cov=cov_region1,
                        size=d_components
                    ).T  # Shape: (n_neurons_region1, d_components)

                    # Sample d vectors from multivariate normal distribution for region2
                    V_hat = np.random.multivariate_normal(
                        mean=np.zeros(cov_region2.shape[0]),
                        cov=cov_region2,
                        size=d_components
                    ).T  # Shape: (n_neurons_region2, d_components)

                    # QR decomposition to orthogonalize
                    U_null, _ = np.linalg.qr(U_hat)  # Shape: (n_neurons_region1, d_components)
                    V_null, _ = np.linalg.qr(V_hat)  # Shape: (n_neurons_region2, d_components)

                    # Calculate cosine similarity for this null sample
                    null_cos_similarities = []
                    for i in range(d_components):
                        u_null_component = U_null[:, i].reshape(1, -1)
                        v_null_component = V_null[:, i].reshape(1, -1)
                        null_cos_sim = cosine_similarity(u_null_component, v_null_component)[0, 0]
                        null_cos_similarities.append(null_cos_sim)

                    R_null = np.mean(null_cos_similarities)
                    null_similarities.append(R_null)

                # Calculate p-value (percentage of null values >= actual value)
                null_similarities = np.array(null_similarities)
                p_value = (np.sum(null_similarities >= R_actual) + 1) / (n_null_samples + 1)

                # Calculate null distribution statistics
                null_mean = np.mean(null_similarities)
                null_std = np.std(null_similarities)
                null_95_percentile = np.percentile(null_similarities, 95)
                null_99_percentile = np.percentile(null_similarities, 99)

                # Store results
                results.append({
                    'region1': region1,
                    'region2': region2,
                    'session': session,
                    'significant_components': d_components,
                    'R_actual': R_actual,
                    'actual_component_similarities': actual_similarities,
                    'null_mean': null_mean,
                    'null_std': null_std,
                    'null_95_percentile': null_95_percentile,
                    'null_99_percentile': null_99_percentile,
                    'p_value': p_value,
                    'is_significant_95': R_actual > null_95_percentile,
                    'is_significant_99': R_actual > null_99_percentile,
                    'null_similarities': null_similarities.tolist(),
                    'U_shape': U_actual.shape,
                    'V_shape': V_actual.shape
                })

            else:
                print(
                    f"Different dimensions for {region1} ({U_actual.shape[0]}) vs {region2} ({V_actual.shape[0]}) - skipping")
                continue

        except Exception as e:
            print(f"Error processing {region1} vs {region2}, session {session}: {e}")
            continue

    results_df = pd.DataFrame(results)
    return results_df

# Should probably properly do inheritance for adding this, or add both of these to my analysis_class file, but hey
# monkey patching exists for a reason right?
TwoRegionCCAAnalysis.calculate_cca_cosine_similarity_with_null = calculate_cca_cosine_similarity_with_null

if __name__ == "__main__":
    # Initialize analysis
    neuron_threshold = 40
    analyzer = TwoRegionCCAAnalysis(neuron_threshold=neuron_threshold, n_splits=5, random_state=42,
                                   n_permutations=100)
    fr_db = FiringRateAnalysis(db_suffix="coords_updated")
    file_path = fr_db.figdata_path

    significant_df = pd.read_feather(f"{file_path}/CCA_two_region_analysis/cca_primary_auditory_results.feather")
    session_sig_df = significant_df[significant_df["region2"] != "Primary auditory area"]  # Filter out the primary vs primary results for now

    # Calculate cosine similarities with null distribution
    null_results = analyzer.calculate_cca_cosine_similarity_with_null(session_sig_df, n_null_samples=1000)

    # Display significant results
    significant_results = null_results[null_results['is_significant_95']]
    print(f"Found {len(significant_results)} significant region pairs (p < 0.05)")
    print(significant_results[['region1', 'region2', 'session', 'R_actual', 'p_value']])

    # Save results
    null_results.to_feather(f"{analyzer.file_path}/CCA_cosine_similarity_with_null.feather")