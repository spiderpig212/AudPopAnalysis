import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from analysis_class import FiringRateAnalysis

fr_db = FiringRateAnalysis(db_suffix="coords_updated")
file_path = fr_db.figdata_path

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


def create_boxplots_am_puretones():
    """Create box plots for AM and pureTones comparing brain region pairs accuracy values"""
    # Load data for AM and pureTones
    for stim in ["AM", "pureTones"]:
        stim_df = pd.read_feather(f"{file_path}/CCA_SVM_{stim}.feather")
        # pt_df = pd.read_feather(f"{file_path}/CCA_correlations_pureTones.feather")

        # combined_df = pd.concat([am_df, pt_df], ignore_index=True)

        # Get unique region pairs and response ranges
        region_pairs = stim_df['region_pair'].unique()
        response_ranges = stim_df['response_range'].unique()

        # Create figure with subplots
        n_pairs = len(region_pairs)
        n_ranges = len(response_ranges)
        fig, axes = plt.subplots(n_ranges, n_pairs, figsize=(5 * n_pairs, 4 * n_ranges))

        if n_ranges == 1:
            axes = axes.reshape(1, -1)
        if n_pairs == 1:
            axes = axes.reshape(-1, 1)

        for i, resp_range in enumerate(response_ranges):
            for j, region_pair in enumerate(region_pairs):
                ax = axes[i, j]

                # Filter data for this combination
                subset = stim_df[(stim_df['response_range'] == resp_range) &
                                     (stim_df['region_pair'] == region_pair)]

                if len(subset) > 0:
                    # Create box plot
                    sns.boxplot(data=subset, x='stimulus', y='mean_accuracy', ax=ax, palette='Set2')

                    # Add chance level line
                    chance_level = subset['chance_level'].iloc[0]
                    ax.axhline(y=chance_level, color='red', linestyle='--', alpha=0.7, label='Chance')

                    ax.set_title(f'{region_pair}\n{resp_range}')
                    ax.set_xlabel('Stimulus Type')
                    ax.set_ylabel('Classification Accuracy')
                    ax.legend()
                else:
                    ax.set_title(f'{region_pair}\n{resp_range} (No data)')
                    ax.set_xlabel('Stimulus Type')
                    ax.set_ylabel('Classification Accuracy')

        plt.tight_layout()
        plt.savefig(f"{file_path}/CCA_SVM/boxplots_{stim}.png", dpi=300, bbox_inches='tight')
        plt.show()


def create_heatmap_natural_sounds():
    """Create heatmap for natural sounds with upper triangle only"""
    # Load natural sounds data
    ns_df = pd.read_feather(f"{file_path}/CCA_SVM_naturalSound.feather")
    stim_info = fr_db.stim_info['naturalSound']
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
                    accuracy_matrix[idx1, idx2] = row['mean_accuracy']
                    # Since accuracies should be symmetric, fill both positions
                    accuracy_matrix[idx2, idx1] = row['mean_accuracy']

                # Set diagonal to NaN (no self-comparison)
                np.fill_diagonal(accuracy_matrix, np.nan)

                # Create mask for upper triangle only
                mask = np.tril(np.ones_like(accuracy_matrix, dtype=bool))

                # Create heatmap
                sns.heatmap(accuracy_matrix, mask=mask, annot=True, fmt='.2f',
                            cmap='viridis', ax=ax, cbar_kws={'label': 'Accuracy'},
                            xticklabels=[f'{stimVals[int(s)]}' for s in unique_stims],
                            yticklabels=[f'{stimVals[int(s)]}' for s in unique_stims])

                ax.set_title(f'{region_pair}\n{resp_range}')
                ax.set_xlabel('Stimulus Value')
                ax.set_ylabel('Stimulus Value')
            else:
                ax.set_title(f'{region_pair}\n{resp_range} (No data)')
                ax.set_xlabel('Stimulus Value')
                ax.set_ylabel('Stimulus Value')

    plt.tight_layout()
    plt.savefig(f"{file_path}/heatmap_naturalSounds.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_example_decision_boundaries():
    """Plot example decision boundaries from stored data"""
    # Load decision boundaries
    with open(f"{file_path}/decision_boundaries_data.pkl", 'rb') as f:
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
    plt.savefig(f"{file_path}/decision_boundaries_examples.png", dpi=300, bbox_inches='tight')
    plt.show()


# Create all visualizations
print("Creating box plots for AM and pureTones...")
# create_boxplots_am_puretones()

print("Creating heatmap for natural sounds...")
create_heatmap_natural_sounds()

print("Plotting example decision boundaries...")
plot_example_decision_boundaries()

print("All visualizations completed!")