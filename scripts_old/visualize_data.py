# imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser(description='visualize the data')
parser.add_argument('--pkl_file_path', type=str, required=True, help='Base path to pkl file')
parser.add_argument('--base_output_path', type=str, required=True, help='Base path to save the output file')
parser.add_argument('--color_by_action', action='store_true', help='Flag to color the dots by action')
parser.add_argument('--n_samples', type=int, default=25000, help='take n sample from pickle')
args = parser.parse_args()

with open(args.pkl_file_path, 'rb') as f:
    data = pickle.load(f)


data = data[:args.n_samples]
# Unpack the data: assuming each sample is (state, action, cost)
features = []
labels = []
"""
# feature - (state,action) concatenated
for state, action, cost in data:
    state_flat = state.flatten()
    feature = np.concatenate((state_flat, action))
    features.append(feature)
    binary_cost = (cost >= 1).float()
    labels.append(binary_cost)

# feature - only state
"""
for state, action, cost in data:
    state_flat = state.flatten()
    features.append(state_flat)
    binary_cost = (cost >= 1).float()
    labels.append(binary_cost)

features = np.array(features, dtype=np.float64)  # Shape: (num_samples, 26)
labels = np.array(labels,  dtype=np.float64)      # Shape: (num_samples,)

# Dimensionality Reduction (Choose one: PCA, t-SNE, or UMAP)
# PCA

"""
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)
"""
# t-SNE
tsne = TSNE(n_components=2, random_state=42)

features_tsne = tsne.fit_transform(features)
"""
# UMAP
umap = UMAP(n_components=2, random_state=42)
features_umap = umap.fit_transform(features)

"""
# Visualization

def plot_2d(data, labels, title, plot_path=None):
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f"Class {label}", alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    if plot_path:
        plt.savefig(plot_path + f"_{title}.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")

# Plot using PCA
#plot_2d(features_pca, labels, "pca_visualization_25k_samples_only_states", plot_path = args.base_output_path)

# Plot using t-SNE

#plot_2d(features_tsne, labels, f"tsne_visualization_samples={args.n_samples}_only_states", plot_path = args.base_output_path)

# Plot using UMAP
#plot_2d(features_umap, labels, "UMAP Visualization")


# Visualization
def plot_2d_by_action(data, actions, title, plot_path=None):
    plt.figure(figsize=(8, 6))
    actions = actions.flatten()
    # Assuming actions is a numerical value or can be converted to categories
    unique_actions = np.unique(actions)

    for action in unique_actions:
        plt.scatter(data[actions == action, 0],
                    data[actions == action, 1],
                    label=f"Action {action}",
                    alpha=0.7)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()

    if plot_path:
        plt.savefig(plot_path + f"_{title}.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}_{title}.png")
    else:
        plt.show()


# Extracting actions from data
actions = np.array([action for _, action, _ in data]).flatten()

# Call the function with t-SNE features and actions
#plot_2d_by_action(features_tsne, actions, f"samples={args.n_samples}_t-SNE Visualization Colored by Actions_only_state", args.base_output_path)


# Visualization
def plot_2d_by_action_and_label(data, actions, labels, title, plot_path=None):
    plt.figure(figsize=(10, 8))
    actions = actions.flatten()
    # Get unique actions and labels
    unique_actions = np.unique(actions)
    unique_labels = np.unique(labels)  # Assuming binary labels, e.g., 0 and 1

    # Define a color map for actions
    color_map = plt.cm.get_cmap("tab10", len(unique_actions))  # Adjust the number of colors
    action_colors = {action: color_map(i) for i, action in enumerate(unique_actions)}

    # Define specific markers for binary labels
    label_markers = {unique_labels[0]: 'o', unique_labels[1]: 's'}  # Circle and square markers

    # Plot data points
    for action in unique_actions:
        for label in unique_labels:
            # Filter points by action and label
            mask = (actions == action) & (labels == label)
            plt.scatter(data[mask, 0],
                        data[mask, 1],
                        label=f"Action {action}, Label {label}",
                        color=action_colors[action],
                        marker=label_markers[label],
                        edgecolor='black',  # Add black edges for clarity
                        alpha=0.8,
                        s=50)  # Adjust marker size

    # Plot details
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))  # Adjust legend position for clarity

    # Save or show the plot
    if plot_path:
        plt.savefig(plot_path + f"_{title}.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}_{title}.png")
    else:
        plt.show()


# Extracting actions and labels
# Call the function with t-SNE features, actions, and labels
plot_2d_by_action_and_label(features_tsne, actions, labels, f"samples={args.n_samples}_t-SNE Visualization_with_label_markers", args.base_output_path)

