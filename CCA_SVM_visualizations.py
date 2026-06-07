import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
from statsmodels.stats.multitest import multipletests
from analysis_class import FiringRateAnalysis

fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path

SOUND_CATEGORIES = ['Frogs', 'Crickets', 'Streamside', 'Bubbling', 'Bees']
EXEMPLARS_PER_CATEGORY = 4

def plot_decision_boundary_scatter(boundary_data, ax=None):
    """Plot scatter plot with decision boundary for 2D visualization"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    X_train = boundary_data['X_train']
    y_train = boundary_data['y_train']
    X_test = boundary_data['X_test']
    y_test = boundary_data['y_test']
    y_pred = boundary_data['y_pred']
    svm_model = boundary_data['svm_model']

    # Use first 2 components for visualization
    X_train_2d = X_train[:, :2]
    X_test_2d = X_test[:, :2]

    # Create a mesh for decision boundary
    h = 0.02
    x_min, x_max = np.min([X_train_2d[:, 0].min(), X_test_2d[:, 0].min()]) - 1, \
                   np.max([X_train_2d[:, 0].max(), X_test_2d[:, 0].max()]) + 1
    y_min, y_max = np.min([X_train_2d[:, 1].min(), X_test_2d[:, 1].min()]) - 1, \
                   np.max([X_train_2d[:, 1].max(), X_test_2d[:, 1].max()]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # For decision boundary, we need to pad the mesh points with zeros for higher dimensions
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    if X_train.shape[1] > 2:
        # Pad with zeros for higher dimensions
        padding = np.zeros((mesh_points.shape[0], X_train.shape[1] - 2))
        mesh_points_full = np.hstack([mesh_points, padding])
    else:
        mesh_points_full = mesh_points

    Z = svm_model.predict(mesh_points_full)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    # Plot training points
    unique_classes = np.unique(y_train)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    for i, cls in enumerate(unique_classes):
        mask_train = y_train == cls
        ax.scatter(X_train_2d[mask_train, 0], X_train_2d[mask_train, 1],
                   c=[colors[i]], label=f'Train {cls:.2f}', alpha=0.7, s=50, marker='o')

    # Plot test points (correct vs incorrect predictions)
    for i, cls in enumerate(unique_classes):
        mask_test = y_test == cls
        correct_mask = (y_test == y_pred) & mask_test
        incorrect_mask = (y_test != y_pred) & mask_test

        if np.any(correct_mask):
            ax.scatter(X_test_2d[correct_mask, 0], X_test_2d[correct_mask, 1],
                       c=[colors[i]], label=f'Test {cls:.2f} (correct)', alpha=0.9, s=80, marker='s')
        if np.any(incorrect_mask):
            ax.scatter(X_test_2d[incorrect_mask, 0], X_test_2d[incorrect_mask, 1],
                       c=[colors[i]], label=f'Test {cls:.2f} (incorrect)', alpha=0.9, s=80, marker='x')

    ax.set_xlabel('CCA Component 1')
    ax.set_ylabel('CCA Component 2')
    ax.set_title(f"{boundary_data['region_pair']} - {boundary_data['response_range']}\n{boundary_data['stim_type']}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    return ax


# def create_boxplots_am_puretones():
#     """Create box plots for AM and pureTones comparing brain region pairs accuracy values"""
#     # Load data for AM and pureTones
#     for stim in ["AM", "pureTones"]:
#         try:
#             stim_df = pd.read_feather(f"{file_path}/CCA_SVC/CCA_SVM_{stim}.feather")
#         except FileNotFoundError:
#             print(f"No data found for {stim}, skipping boxplot creation.")
#             continue
#
#         # pt_df = pd.read_feather(f"{file_path}/CCA_correlations_pureTones.feather")
#
#         # combined_df = pd.concat([am_df, pt_df], ignore_index=True)
#
#         # Get unique region pairs and response ranges
#         region_pairs = stim_df['region_pair'].unique()
#         response_ranges = stim_df['response_range'].unique()
#
#         # Create figure with subplots
#         n_pairs = len(region_pairs)
#         n_ranges = len(response_ranges)
#         fig, axes = plt.subplots(n_ranges, n_pairs, figsize=(5 * n_pairs, 4 * n_ranges))
#
#         if n_ranges == 1:
#             axes = axes.reshape(1, -1)
#         if n_pairs == 1:
#             axes = axes.reshape(-1, 1)
#
#         for i, resp_range in enumerate(response_ranges):
#             for j, region_pair in enumerate(region_pairs):
#                 ax = axes[i, j]
#
#                 # Filter data for this combination
#                 subset = stim_df[(stim_df['response_range'] == resp_range) &
#                                      (stim_df['region_pair'] == region_pair)]
#
#                 if len(subset) > 0:
#                     # Create box plot
#                     sns.boxplot(data=subset, x='stimulus', y='mean_accuracy', ax=ax, palette='Set2')
#
#                     # Add chance level line
#                     chance_level = subset['chance_level'].iloc[0]
#                     ax.axhline(y=chance_level, color='red', linestyle='--', alpha=0.7, label='Chance')
#
#                     ax.set_title(f'{region_pair}\n{resp_range}')
#                     ax.set_xlabel('Stimulus Type')
#                     ax.set_ylabel('Classification Accuracy')
#                     ax.legend()
#                 else:
#                     ax.set_title(f'{region_pair}\n{resp_range} (No data)')
#                     ax.set_xlabel('Stimulus Type')
#                     ax.set_ylabel('Classification Accuracy')
#
#         plt.tight_layout()
#         plt.savefig(f"{file_path}/CCA_SVC/boxplots/boxplots_{stim}.png", dpi=300, bbox_inches='tight')
#         plt.show()


# def create_heatmap_natural_sounds():
#     """Create heatmap for natural sounds with upper triangle only"""
#     # Load natural sounds data
#     try:
#         ns_df = pd.read_feather(f"{file_path}/CCA_SVC/CCA_SVM_naturalSound.feather")
#     except FileNotFoundError:
#         print("No data found for natural sounds, skipping heatmap creation.")
#         return
#
#     stim_info = fr_db.stim_info['naturalSound']
#     stimVals = stim_info['stimVals']
#
#     # Get unique values
#     region_pairs = ns_df['region_pair'].unique()
#     response_ranges = ns_df['response_range'].unique()
#
#     # Get unique stimuli from the stim_pair column
#     all_stims = set()
#     for stim_pair in ns_df['stim_pair']:
#         all_stims.update(stim_pair)
#     unique_stims = sorted(list(all_stims))
#
#     # Create figure with subplots
#     n_pairs = len(region_pairs)
#     n_ranges = len(response_ranges)
#     fig, axes = plt.subplots(n_ranges, n_pairs, figsize=(6 * n_pairs, 5 * n_ranges))
#
#     if n_ranges == 1:
#         axes = axes.reshape(1, -1)
#     if n_pairs == 1:
#         axes = axes.reshape(-1, 1)
#
#     for i, resp_range in enumerate(response_ranges):
#         for j, region_pair in enumerate(region_pairs):
#             ax = axes[i, j]
#
#             # Filter data for this combination
#             subset = ns_df[(ns_df['response_range'] == resp_range) &
#                            (ns_df['region_pair'] == region_pair)]
#
#             if len(subset) > 0:
#                 # Create accuracy matrix
#                 n_stims = len(unique_stims)
#                 accuracy_matrix = np.full((n_stims, n_stims), np.nan)
#
#                 # Fill the matrix with accuracy values
#                 for _, row in subset.iterrows():
#                     stim1, stim2 = row['stim_pair']
#                     idx1 = unique_stims.index(stim1)
#                     idx2 = unique_stims.index(stim2)
#                     accuracy_matrix[idx1, idx2] = row['mean_accuracy']
#                     # Since accuracies should be symmetric, fill both positions
#                     accuracy_matrix[idx2, idx1] = row['mean_accuracy']
#
#                 # Set diagonal to NaN (no self-comparison)
#                 np.fill_diagonal(accuracy_matrix, np.nan)
#
#                 # Create mask for upper triangle only
#                 mask = np.tril(np.ones_like(accuracy_matrix, dtype=bool))
#
#                 # Create heatmap
#                 sns.heatmap(accuracy_matrix, mask=mask, annot=True, fmt='.2f',
#                             cmap='viridis', ax=ax, cbar_kws={'label': 'Accuracy'},
#                             xticklabels=[f'{stimVals[int(s)]}' for s in unique_stims],
#                             yticklabels=[f'{stimVals[int(s)]}' for s in unique_stims])
#
#                 ax.set_title(f'{region_pair}\n{resp_range}')
#                 ax.set_xlabel('Stimulus Value')
#                 ax.set_ylabel('Stimulus Value')
#             else:
#                 ax.set_title(f'{region_pair}\n{resp_range} (No data)')
#                 ax.set_xlabel('Stimulus Value')
#                 ax.set_ylabel('Stimulus Value')
#
#     plt.tight_layout()
#     plt.savefig(f"{file_path}/CCA_SVC/heatmap_naturalSounds.png", dpi=400, bbox_inches='tight')
#     plt.show()

def create_heatmap_stims_pairwise():
    """Create heatmap for natural sounds with upper triangle only"""
    # Load natural sounds data\
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            ns_df = pd.read_feather(f"{file_path}/CCA_SVC/CCA_SVM_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        if stim == "naturalSound":
            stim_info = fr_db.stim_info[f'{stim}']
            stimVals = stim_info['stimVals']

        # Get unique values
        region_pairs = ns_df['region_pair'].unique()
        response_ranges = ns_df['response_range'].unique()

        # Get unique stimuli from the stim_pair column
        all_stims = set()
        for stim_pair in ns_df['stim_pair']:
            all_stims.update(stim_pair)
        unique_stims = sorted(list(all_stims))

        # Create figure with subplots
        n_pairs = len(region_pairs)
        n_ranges = len(response_ranges)
        fig, axes = plt.subplots(n_ranges, n_pairs, figsize=(6 * n_pairs, 5 * n_ranges))

        if n_ranges == 1:
            axes = axes.reshape(1, -1)
        if n_pairs == 1:
            axes = axes.reshape(-1, 1)

        for i, resp_range in enumerate(response_ranges):
            for j, region_pair in enumerate(region_pairs):
                ax = axes[i, j]

                # Filter data for this combination
                subset = ns_df[(ns_df['response_range'] == resp_range) &
                               (ns_df['region_pair'] == region_pair)]

                if len(subset) > 0:
                    # Create accuracy matrix
                    n_stims = len(unique_stims)
                    accuracy_matrix = np.full((n_stims, n_stims), np.nan)

                    # Fill the matrix with accuracy values
                    for _, row in subset.iterrows():
                        stim1, stim2 = row['stim_pair']
                        idx1 = unique_stims.index(stim1)
                        idx2 = unique_stims.index(stim2)
                        accuracy_matrix[idx1, idx2] = row['mean_accuracy_cca']
                        # Since accuracies should be symmetric, fill both positions
                        accuracy_matrix[idx2, idx1] = row['mean_accuracy_cca']

                    # Set diagonal to NaN (no self-comparison)
                    np.fill_diagonal(accuracy_matrix, np.nan)

                    # Create mask for upper triangle only
                    mask = np.tril(np.ones_like(accuracy_matrix, dtype=bool))

                    # Create heatmap
                    if stim == "naturalSound":
                        sns.heatmap(accuracy_matrix, mask=mask, #annot=True, fmt='.2f',
                                    vmin=0, vmax=1,
                                    cmap='viridis', ax=ax, cbar_kws={'label': 'Accuracy CCA'},
                                    xticklabels=[f'{stimVals[int(s)]}' for s in unique_stims],
                                    yticklabels=[f'{stimVals[int(s)]}' for s in unique_stims])
                    else:
                        sns.heatmap(accuracy_matrix, mask=mask, #annot=True, fmt='.2f',
                                    vmin=0, vmax=1,
                                    cmap='viridis', ax=ax, cbar_kws={'label': 'Accuracy CCA'},
                                    xticklabels=unique_stims, yticklabels=unique_stims)

                    ax.set_title(f'{region_pair}\n{resp_range}')
                    ax.set_xlabel('Stimulus Value')
                    ax.set_ylabel('Stimulus Value')
                else:
                    ax.set_title(f'{region_pair}\n{resp_range} (No data)')
                    ax.set_xlabel('Stimulus Value')
                    ax.set_ylabel('Stimulus Value')

        plt.tight_layout()
        plt.savefig(f"{file_path}/CCA_SVC/heatmaps/heatmap_{stim}.png", dpi=400, bbox_inches='tight')
        plt.show()


def create_delta_heatmap_stims_pairwise():
    """Create heatmap for natural sounds with upper triangle only"""
    # Load natural sounds data\
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            ns_df = pd.read_feather(f"{file_path}/CCA_SVC/CCA_SVM_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        if stim == "naturalSound":
            stim_info = fr_db.stim_info[f'{stim}']
            stimVals = stim_info['stimVals']

        # Get unique values
        region_pairs = ns_df['region_pair'].unique()
        response_ranges = ns_df['response_range'].unique()

        # Get unique stimuli from the stim_pair column
        all_stims = set()
        for stim_pair in ns_df['stim_pair']:
            all_stims.update(stim_pair)
        unique_stims = sorted(list(all_stims))

        # Create figure with subplots
        n_pairs = len(region_pairs)
        n_ranges = len(response_ranges)
        fig, axes = plt.subplots(n_ranges, n_pairs, figsize=(6 * n_pairs, 5 * n_ranges))

        if n_ranges == 1:
            axes = axes.reshape(1, -1)
        if n_pairs == 1:
            axes = axes.reshape(-1, 1)

        for i, resp_range in enumerate(response_ranges):
            for j, region_pair in enumerate(region_pairs):
                ax = axes[i, j]

                # Filter data for this combination
                subset = ns_df[(ns_df['response_range'] == resp_range) &
                               (ns_df['region_pair'] == region_pair)]

                if len(subset) > 0:
                    # Create accuracy matrix
                    n_stims = len(unique_stims)
                    accuracy_matrix = np.full((n_stims, n_stims), np.nan)

                    # Fill the matrix with accuracy values
                    for _, row in subset.iterrows():
                        stim1, stim2 = row['stim_pair']
                        idx1 = unique_stims.index(stim1)
                        idx2 = unique_stims.index(stim2)
                        accuracy_matrix[idx1, idx2] = row['mean_accuracy_cca'] - row['mean_accuracy']
                        # Since accuracies should be symmetric, fill both positions
                        accuracy_matrix[idx2, idx1] = row['mean_accuracy_cca'] - row['mean_accuracy']

                    # Set diagonal to NaN (no self-comparison)
                    np.fill_diagonal(accuracy_matrix, np.nan)

                    # Create mask for upper triangle only
                    mask = np.tril(np.ones_like(accuracy_matrix, dtype=bool))

                    # Create heatmap
                    if stim == "naturalSound":
                        sns.heatmap(accuracy_matrix, mask=mask, #annot=True, fmt='.2f',
                                    cmap='RdBu', ax=ax, center=0, cbar_kws={'label': 'Accuracy CCA - Accuracy'},
                                    xticklabels=[f'{stimVals[int(s)]}' for s in unique_stims],
                                    yticklabels=[f'{stimVals[int(s)]}' for s in unique_stims])
                    else:  # Use center to adjust the colorbar? Annotate to False to clean up the appearance
                        sns.heatmap(accuracy_matrix, mask=mask, #annot=True, fmt='.2f',
                                    cmap='RdBu', ax=ax, center=0, cbar_kws={'label': 'Accuracy CCA - Accuracy'},
                                    xticklabels=unique_stims, yticklabels=unique_stims)

                    ax.set_title(f'{region_pair}\n{resp_range}')
                    ax.set_xlabel('Stimulus Value')
                    ax.set_ylabel('Stimulus Value')
                else:
                    ax.set_title(f'{region_pair}\n{resp_range} (No data)')
                    ax.set_xlabel('Stimulus Value')
                    ax.set_ylabel('Stimulus Value')

        plt.tight_layout()
        plt.savefig(f"{file_path}/CCA_SVC/delta_heatmaps/heatmap_delta_{stim}.png", dpi=400, bbox_inches='tight')
        plt.show()

def create_delta_heatmap_stims_pairwise_pca():
    """Create heatmap for natural sounds with upper triangle only"""
    # Load natural sounds data\
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            ns_df = pd.read_feather(f"{file_path}/CCA_SVC/CCA_SVM_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        if stim == "naturalSound":
            stim_info = fr_db.stim_info[f'{stim}']
            stimVals = stim_info['stimVals']

        # Get unique values
        region_pairs = ns_df['region_pair'].unique()
        response_ranges = ns_df['response_range'].unique()

        # Get unique stimuli from the stim_pair column
        all_stims = set()
        for stim_pair in ns_df['stim_pair']:
            all_stims.update(stim_pair)
        unique_stims = sorted(list(all_stims))

        # Create figure with subplots
        n_pairs = len(region_pairs)
        n_ranges = len(response_ranges)
        fig, axes = plt.subplots(n_ranges, n_pairs, figsize=(6 * n_pairs, 5 * n_ranges))

        if n_ranges == 1:
            axes = axes.reshape(1, -1)
        if n_pairs == 1:
            axes = axes.reshape(-1, 1)

        for i, resp_range in enumerate(response_ranges):
            for j, region_pair in enumerate(region_pairs):
                ax = axes[i, j]

                # Filter data for this combination
                subset = ns_df[(ns_df['response_range'] == resp_range) &
                               (ns_df['region_pair'] == region_pair)]

                if len(subset) > 0:
                    # Create accuracy matrix
                    n_stims = len(unique_stims)
                    accuracy_matrix = np.full((n_stims, n_stims), np.nan)

                    # Fill the matrix with accuracy values
                    for _, row in subset.iterrows():
                        stim1, stim2 = row['stim_pair']
                        idx1 = unique_stims.index(stim1)
                        idx2 = unique_stims.index(stim2)
                        accuracy_matrix[idx1, idx2] = row['mean_accuracy_pca'] - row['mean_accuracy']
                        # Since accuracies should be symmetric, fill both positions
                        accuracy_matrix[idx2, idx1] = row['mean_accuracy_pca'] - row['mean_accuracy']

                    # Set diagonal to NaN (no self-comparison)
                    np.fill_diagonal(accuracy_matrix, np.nan)

                    # Create mask for upper triangle only
                    mask = np.tril(np.ones_like(accuracy_matrix, dtype=bool))

                    # Create heatmap
                    if stim == "naturalSound":
                        sns.heatmap(accuracy_matrix, mask=mask, #annot=True, fmt='.2f',
                                    cmap='RdBu', ax=ax, center=0, cbar_kws={'label': 'Accuracy PCA - Accuracy'},
                                    xticklabels=[f'{stimVals[int(s)]}' for s in unique_stims],
                                    yticklabels=[f'{stimVals[int(s)]}' for s in unique_stims])
                    else:  # Use center to adjust the colorbar? Annotate to False to clean up the appearance
                        sns.heatmap(accuracy_matrix, mask=mask, #annot=True, fmt='.2f',
                                    cmap='RdBu', ax=ax, center=0, cbar_kws={'label': 'Accuracy PCA - Accuracy'},
                                    xticklabels=unique_stims, yticklabels=unique_stims)

                    ax.set_title(f'{region_pair}\n{resp_range}')
                    ax.set_xlabel('Stimulus Value')
                    ax.set_ylabel('Stimulus Value')
                else:
                    ax.set_title(f'{region_pair}\n{resp_range} (No data)')
                    ax.set_xlabel('Stimulus Value')
                    ax.set_ylabel('Stimulus Value')

        plt.tight_layout()
        plt.savefig(f"{file_path}/CCA_SVC/delta_heatmaps_pca/heatmap_delta_{stim}.png", dpi=400, bbox_inches='tight')
        plt.show()

def plot_example_decision_boundaries():
    """Plot example decision boundaries from stored data"""
    # Load decision boundaries
    with open(f"{file_path}/CCA_SVC/decision_boundaries_data.pkl", 'rb') as f:
        boundaries_data = pickle.load(f)

    # Plot a few examples
    n_examples = min(4, len(boundaries_data))
    if n_examples == 0:
        print("No decision boundary data found!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (key, data) in enumerate(list(boundaries_data.items())[:n_examples]):
        plot_decision_boundary_scatter(data, ax=axes[i])

    # Hide unused subplots
    for i in range(n_examples, 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{file_path}/CCA_SVC/decision_boundaries_examples.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_upper_triangle_boxplots():
    """
    For each stimulus type, extract the upper-triangle values (excluding diagonal)
    from the pairwise accuracy matrices for both CCA and PCA (each relative to the
    baseline 'mean_accuracy'). Plot them as paired boxplots grouped by region_pair
    and response_range, and annotate with non-parametric Wilcoxon signed-rank
    tests, all Bonferroni-corrected together:
        - CCA-delta vs 0  (one-sample, per group)
        - PCA-delta vs 0  (one-sample, per group)
        - CCA-delta vs PCA-delta  (paired, per group)
    """
    for stim in ["AM", "pureTones", "naturalSound"]:
        try:
            ns_df = pd.read_feather(f"{file_path}/CCA_SVC/CCA_SVM_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        region_pairs = ns_df['region_pair'].unique()
        response_ranges = ns_df['response_range'].unique()

        # Collect upper-triangle values per (region_pair, response_range) group
        all_stims = set()
        for stim_pair in ns_df['stim_pair']:
            all_stims.update(stim_pair)
        unique_stims = sorted(list(all_stims))
        n_stims = len(unique_stims)

        # Build dicts: key -> list of upper-triangle values for CCA and PCA deltas
        group_values_cca = {}
        group_values_pca = {}
        group_labels = []

        for resp_range in response_ranges:
            for region_pair in region_pairs:
                subset = ns_df[
                    (ns_df['response_range'] == resp_range) &
                    (ns_df['region_pair'] == region_pair)
                ]

                if len(subset) == 0:
                    continue

                # Build accuracy matrices for CCA-delta and PCA-delta
                cca_matrix = np.full((n_stims, n_stims), np.nan)
                pca_matrix = np.full((n_stims, n_stims), np.nan)
                for _, row in subset.iterrows():
                    stim1, stim2 = row['stim_pair']
                    idx1 = unique_stims.index(stim1)
                    idx2 = unique_stims.index(stim2)
                    cca_val = row['mean_accuracy_cca'] - row['mean_accuracy']
                    pca_val = row['mean_accuracy_pca'] - row['mean_accuracy']
                    cca_matrix[idx1, idx2] = cca_val
                    cca_matrix[idx2, idx1] = cca_val
                    pca_matrix[idx1, idx2] = pca_val
                    pca_matrix[idx2, idx1] = pca_val
                np.fill_diagonal(cca_matrix, np.nan)
                np.fill_diagonal(pca_matrix, np.nan)

                # Extract upper triangle (k=1 excludes diagonal); keep paired entries
                upper_idx = np.triu_indices(n_stims, k=1)
                cca_vals = cca_matrix[upper_idx]
                pca_vals = pca_matrix[upper_idx]
                # Keep only pairs where both are non-NaN (preserves pairing)
                valid = ~np.isnan(cca_vals) & ~np.isnan(pca_vals)
                cca_vals = cca_vals[valid]
                pca_vals = pca_vals[valid]

                if len(cca_vals) > 0:
                    label = f"{region_pair}\n{resp_range}"
                    group_values_cca[label] = cca_vals
                    group_values_pca[label] = pca_vals
                    group_labels.append(label)

        if not group_labels:
            print(f"No data found for {stim}, skipping.")
            continue

        # ── Statistical testing (Wilcoxon signed-rank, non-parametric) ───────
        # Three tests per group:
        #   (1) CCA delta vs 0    (one-sample)
        #   (2) PCA delta vs 0    (one-sample)
        #   (3) CCA delta vs PCA delta  (paired)
        # All p-values pooled into a single Bonferroni correction.
        def _wilcoxon_safe(x, y=None):
            """Run Wilcoxon; return (stat, p) or (nan, nan) on failure."""
            try:
                if y is None:
                    if len(x) < 2 or not np.any(x != 0):
                        return np.nan, np.nan
                    res = stats.wilcoxon(x, alternative='two-sided',
                                         zero_method='wilcox')
                else:
                    diffs = x - y
                    if len(diffs) < 2 or not np.any(diffs != 0):
                        return np.nan, np.nan
                    res = stats.wilcoxon(x, y, alternative='two-sided',
                                         zero_method='wilcox')
                return res.statistic, res.pvalue
            except ValueError:
                return np.nan, np.nan

        cca_vs0_stats, cca_vs0_p = [], []
        pca_vs0_stats, pca_vs0_p = [], []
        paired_stats, paired_p = [], []

        for label in group_labels:
            cca_vals = group_values_cca[label]
            pca_vals = group_values_pca[label]

            s, p = _wilcoxon_safe(cca_vals)
            cca_vs0_stats.append(s); cca_vs0_p.append(p)

            s, p = _wilcoxon_safe(pca_vals)
            pca_vs0_stats.append(s); pca_vs0_p.append(p)

            s, p = _wilcoxon_safe(cca_vals, pca_vals)
            paired_stats.append(s); paired_p.append(p)

        # Pool all p-values for a single Bonferroni correction
        all_raw_p = np.array(cca_vs0_p + pca_vs0_p + paired_p, dtype=float)
        all_corr_p = np.full_like(all_raw_p, np.nan)
        all_reject = np.zeros_like(all_raw_p, dtype=bool)

        valid_mask = ~np.isnan(all_raw_p)
        if valid_mask.sum() > 0:
            reject, pvals_corr, _, _ = multipletests(all_raw_p[valid_mask],
                                                    alpha=0.05,
                                                    method='bonferroni')
            all_corr_p[valid_mask] = pvals_corr
            all_reject[valid_mask] = reject

        n_groups = len(group_labels)
        cca_vs0_corr = all_corr_p[:n_groups]
        cca_vs0_rej = all_reject[:n_groups]
        pca_vs0_corr = all_corr_p[n_groups:2 * n_groups]
        pca_vs0_rej = all_reject[n_groups:2 * n_groups]
        paired_corr = all_corr_p[2 * n_groups:]
        paired_rej = all_reject[2 * n_groups:]

        def _sig_str(p):
            if np.isnan(p):
                return "n/a"
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return "ns"

        # ── Plotting ─────────────────────────────────────────────────────────
        fig_width = max(6, n_groups * 1.8)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # Side-by-side paired boxplots
        width = 0.35
        positions = np.arange(1, n_groups + 1)
        cca_positions = positions - width / 2
        pca_positions = positions + width / 2

        cca_box_data = [group_values_cca[lbl] for lbl in group_labels]
        pca_box_data = [group_values_pca[lbl] for lbl in group_labels]

        bp_cca = ax.boxplot(cca_box_data, positions=cca_positions, widths=width,
                            patch_artist=True, notch=False,
                            medianprops=dict(color='black', linewidth=2))
        bp_pca = ax.boxplot(pca_box_data, positions=pca_positions, widths=width,
                            patch_artist=True, notch=False,
                            medianprops=dict(color='black', linewidth=2))

        cca_color = '#1f77b4'  # blue
        pca_color = '#ff7f0e'  # orange
        for patch in bp_cca['boxes']:
            patch.set_facecolor(cca_color)
            patch.set_alpha(0.7)
        for patch in bp_pca['boxes']:
            patch.set_facecolor(pca_color)
            patch.set_alpha(0.7)

        # Reference line at 0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6)

        # Compute y-axis layout for annotations
        all_vals = [v for v in cca_box_data + pca_box_data if len(v) > 0]
        y_max = max(v.max() for v in all_vals)
        y_min = min(v.min() for v in all_vals)
        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

        # Per-box vs-0 annotations (just above each individual box)
        per_box_y = y_max + y_range * 0.04
        for idx in range(n_groups):
            ax.text(cca_positions[idx], per_box_y, _sig_str(cca_vs0_corr[idx]),
                    ha='center', va='bottom', fontsize=9,
                    color='#d62728' if cca_vs0_rej[idx] else 'gray')
            ax.text(pca_positions[idx], per_box_y, _sig_str(pca_vs0_corr[idx]),
                    ha='center', va='bottom', fontsize=9,
                    color='#d62728' if pca_vs0_rej[idx] else 'gray')

        # Paired-comparison bracket above the per-box annotations
        bracket_base = y_max + y_range * 0.12
        bracket_top = y_max + y_range * 0.16
        for idx in range(n_groups):
            ax.plot([cca_positions[idx], cca_positions[idx],
                     pca_positions[idx], pca_positions[idx]],
                    [bracket_base, bracket_top, bracket_top, bracket_base],
                    color='gray', linewidth=0.8)
            ax.text(positions[idx], bracket_top, _sig_str(paired_corr[idx]),
                    ha='center', va='bottom', fontsize=10,
                    color='#d62728' if paired_rej[idx] else 'gray')

        # Give the y-axis a bit of headroom for annotations
        ax.set_ylim(top=y_max + y_range * 0.25)

        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Accuracy delta (vs baseline) — upper triangle values')
        ax.set_title(f'{stim} — Upper Triangle Accuracy Deltas: CCA vs PCA\n'
                     f'(Wilcoxon signed-rank, Bonferroni-corrected across all tests; '
                     f'* p<0.05, ** p<0.01, *** p<0.001)')

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cca_color, alpha=0.7, label='CCA - baseline'),
            Patch(facecolor=pca_color, alpha=0.7, label='PCA - baseline'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_xlim(0.5, n_groups + 0.5)

        plt.tight_layout()
        plt.savefig(f"{file_path}/CCA_SVC/Upper_tri/upper_triangle_boxplots_{stim}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # ── Console summary ──────────────────────────────────────────────────
        print(f"\n{stim} — Statistical summary "
              f"(Wilcoxon signed-rank, Bonferroni-corrected across all tests):")
        header = (f"{'Group':<30} "
                  f"{'CCA W':>8} {'CCA p_corr':>11} "
                  f"{'PCA W':>8} {'PCA p_corr':>11} "
                  f"{'Pair W':>8} {'Pair p_corr':>12}")
        print(header)
        print("-" * len(header))
        for i, label in enumerate(group_labels):
            label_flat = label.replace("\n", " | ")
            def _fmt(v, width, prec=4):
                if np.isnan(v):
                    return f"{'n/a':>{width}}"
                return f"{v:>{width}.{prec}f}"
            print(f"{label_flat:<30} "
                  f"{_fmt(cca_vs0_stats[i], 8, 2)} {_fmt(cca_vs0_corr[i], 11)} "
                  f"{_fmt(pca_vs0_stats[i], 8, 2)} {_fmt(pca_vs0_corr[i], 11)} "
                  f"{_fmt(paired_stats[i], 8, 2)} {_fmt(paired_corr[i], 12)}")


def create_SVR_boxplots():
    """
    Loads in the support vector regression data and makes a boxplot comparing the accuracies and performing statistics
    """
    from itertools import combinations

    for stim in ["AM", "pureTones"]:
        try:
            svr_df = pd.read_feather(f"{file_path}/CCA_SVR/CCA_SVR_{stim}.feather")
        except FileNotFoundError:
            print(f"No data found for {stim}, skipping.")
            continue

        svr_df["mean_accuracy_svr_untransformed"] = np.array(
            [np.mean(val) for val in svr_df["cv_scores_svr_untransformed"].values]) * -1
        svr_df["mean_std_svr_untransformed"] = np.array(
            [np.std(val) for val in svr_df["cv_scores_svr_untransformed"].values])
        svr_df["mean_accuracy_svr_cca"] = svr_df["mean_accuracy_svr_cca"] * -1.  # sklearn uses neg mse to optimize

        # Reshape to long format so CCA & untransformed sit side-by-side per response_range
        long_df = svr_df.melt(
            id_vars=["response_range", "region_pair"],
            value_vars=["mean_accuracy_svr_cca", "mean_accuracy_svr_untransformed"],
            var_name="transform",
            value_name="mse",
        )
        long_df["transform"] = long_df["transform"].map({
            "mean_accuracy_svr_cca": "CCA",
            "mean_accuracy_svr_untransformed": "Untransformed",
        })

        # Build a combined x-axis category: "<response_range> | <transform>"
        # so CCA and Untransformed appear adjacent for each response_range.
        response_ranges = list(svr_df["response_range"].unique())
        transform_order = ["Untransformed", "CCA"]
        x_order = [f"{rr} | {t}" for rr in response_ranges for t in transform_order]
        long_df["x_group"] = (
                long_df["response_range"].astype(str) + " | " + long_df["transform"]
        )

        region_pairs = sorted(long_df["region_pair"].unique())

        # Larger figure
        fig_width = max(14, 2.2 * len(x_order))
        fig_height = 10
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.boxplot(
            data=long_df,
            x="x_group",
            y="mse",
            hue="region_pair",
            hue_order=region_pairs,
            order=x_order,
            ax=ax,
        )

        # Visually separate response_range groups with vertical dividers
        n_transforms = len(transform_order)
        for i in range(1, len(response_ranges)):
            ax.axvline(i * n_transforms - 0.5, color="gray",
                       linestyle="--", linewidth=0.8, alpha=0.6)

        # ── Pairwise statistics across region_pairs within each x_group ──────
        # Mann–Whitney U (non-parametric, unpaired). All p-values pooled into
        # a single Bonferroni correction per stimulus.
        def _mwu_safe(a, b):
            try:
                if len(a) < 2 or len(b) < 2:
                    return np.nan, np.nan
                res = stats.mannwhitneyu(a, b, alternative="two-sided")
                return res.statistic, res.pvalue
            except ValueError:
                return np.nan, np.nan

        raw_records = []  # (x_group, rp_a, rp_b, stat, p)
        for xg in x_order:
            for rp_a, rp_b in combinations(region_pairs, 2):
                a = long_df.loc[
                    (long_df["x_group"] == xg) & (long_df["region_pair"] == rp_a),
                    "mse",
                ].values
                b = long_df.loc[
                    (long_df["x_group"] == xg) & (long_df["region_pair"] == rp_b),
                    "mse",
                ].values
                s, p = _mwu_safe(a, b)
                raw_records.append((xg, rp_a, rp_b, s, p))

        raw_p = np.array([r[4] for r in raw_records], dtype=float)
        corr_p = np.full_like(raw_p, np.nan)
        reject = np.zeros_like(raw_p, dtype=bool)
        valid_mask = ~np.isnan(raw_p)
        if valid_mask.sum() > 0:
            rej, pcorr, _, _ = multipletests(raw_p[valid_mask],
                                             alpha=0.05, method="bonferroni")
            corr_p[valid_mask] = pcorr
            reject[valid_mask] = rej

        def _sig_str(p):
            if np.isnan(p):
                return "n/a"
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return "ns"

        # Compute hue x-offsets used by seaborn (matches default dodge layout)
        n_hue = len(region_pairs)
        # seaborn box widths default to 0.8 of the category; per-hue width is 0.8/n_hue
        full_width = 0.8
        hue_width = full_width / n_hue
        hue_offsets = {
            rp: -full_width / 2 + hue_width / 2 + i * hue_width
            for i, rp in enumerate(region_pairs)
        }

        # Headroom for brackets
        y_max = long_df["mse"].max()
        y_min = long_df["mse"].min()
        y_range = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
        step = y_range * 0.05  # vertical gap between stacked brackets

        x_index = {xg: i for i, xg in enumerate(x_order)}

        # Group records by x_group to stack brackets per category
        for xg in x_order:
            xi = x_index[xg]
            # Only annotate significant comparisons to keep the plot readable;
            # change `if rej:` -> `if True:` to show all (incl. ns).
            level = 0
            for (xg_r, rp_a, rp_b, s, p), pc, rej in zip(
                    raw_records, corr_p, reject
            ):
                if xg_r != xg:
                    continue
                if not rej:
                    continue
                xa = xi + hue_offsets[rp_a]
                xb = xi + hue_offsets[rp_b]
                y_bracket = y_max + step * (1 + level)
                y_text = y_bracket + step * 0.15
                ax.plot([xa, xa, xb, xb],
                        [y_bracket - step * 0.2, y_bracket,
                         y_bracket, y_bracket - step * 0.2],
                        color="black", linewidth=0.9)
                ax.text((xa + xb) / 2, y_text, _sig_str(pc),
                        ha="center", va="bottom", fontsize=9, color="black")
                level += 1

        # Give y-axis headroom for the brackets
        # Estimate max level used (cap to avoid runaway when many sig pairs)
        max_level = 0
        for xg in x_order:
            lvl = sum(
                1 for (xg_r, *_), rej in zip(raw_records, reject)
                if xg_r == xg and rej
            )
            max_level = max(max_level, lvl)
        ax.set_ylim(top=y_max + step * (1 + max(1, max_level)) + step * 0.6)

        ax.set_xlabel("Response range | Transform", fontsize=12)
        ax.set_ylabel("MSE (SVR)", fontsize=12)
        ax.set_title(
            f"{stim} — SVR MSE: CCA vs Untransformed by region pair\n"
            f"(Mann–Whitney U between region pairs, Bonferroni-corrected; "
            f"* p<0.05, ** p<0.01, *** p<0.001)",
            fontsize=12,
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)

        # Legend INSIDE the figure
        ax.legend(title="Region pair", loc="upper left",
                  frameon=True, framealpha=0.9, fontsize=10, title_fontsize=11)

        plt.tight_layout()
        plt.savefig(
            f"{file_path}/CCA_SVR/boxplots_svr_{stim}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.show()

        # ── Console summary ──────────────────────────────────────────────────
        print(f"\n{stim} — Region-pair comparisons "
              f"(Mann–Whitney U, Bonferroni-corrected):")
        header = (f"{'x_group':<40} {'rp A':<20} {'rp B':<20} "
                  f"{'U':>10} {'p_corr':>12} {'sig':>5}")
        print(header)
        print("-" * len(header))
        for (xg, rp_a, rp_b, s, p), pc, rej in zip(raw_records, corr_p, reject):
            u_str = "n/a" if np.isnan(s) else f"{s:>10.2f}"
            p_str = "n/a" if np.isnan(pc) else f"{pc:>12.4f}"
            print(f"{xg:<40} {str(rp_a):<20} {str(rp_b):<20} "
                  f"{u_str} {p_str} {_sig_str(pc):>5}")

# Create all visualizations
# print("Creating box plots for AM and pureTones...")
# create_boxplots_am_puretones()

# print("Creating heatmap for natural sounds...")
# create_heatmap_natural_sounds()
print("Creating heatmap for stims...")
create_heatmap_stims_pairwise()

print("Creating delta heatmap for stims...")
create_delta_heatmap_stims_pairwise()
create_delta_heatmap_stims_pairwise_pca()

print("Creating upper triangle boxplots with stats...")
create_upper_triangle_boxplots()

print("Creating SVR boxplots...")
create_SVR_boxplots()

print("Plotting example decision boundaries...")
# plot_example_decision_boundaries()

print("All visualizations completed!")

