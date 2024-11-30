# src/visualization.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples


def visualize_idc_scores(idc_scores, filename='./logs/idc_scores.pdf'):
    methods = list(idc_scores.keys())
    scores = [idc_scores[method] for method in methods]
    
    plt.figure(figsize=(12, 6))
    plt.bar(methods, scores, color='skyblue')
    plt.xlabel('Attribution Method')
    plt.ylabel('IDC Score (%)')
    plt.title('Comparison of IDC Scores for Different Attribution Methods')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=1200)
    # plt.show()

def visualize_activation(activation_values, selected_activations, layer_name, threshold=0, mode='fc'):
    saved_file = f'./images/mean_activation_values_{layer_name}.pdf'
    if mode == 'fc':
        mean_activation = activation_values.mean(dim=0)
        mean_selected_activation = selected_activations.mean(dim=0)
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(activation_values[1])), mean_activation.numpy(), marker='o')
        plt.plot(range(len(selected_activations[1])), mean_selected_activation.numpy(), marker='p')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation Value')
        plt.title('Mean Activation Values of Neurons')
        plt.grid(True)
        plt.legend(['All Neurons', 'Selected Neurons'])
        plt.savefig(saved_file, dpi=1500)
        print("Mean Activation Values plotted, saved to {}".format(saved_file))
        
    elif mode == 'conv':
        mean_activation = torch.mean(selected_activations, dim=[2, 3]).mean(dim=0)
        plt.plot(range(len(mean_activation)), mean_activation.numpy(), marker='o')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation Value')
        plt.title('Mean Activation Values of Neurons')
        plt.grid(True)
        plt.savefig(saved_file, dpi=1500)
        print("Mean Activation Values plotted, saved to {}".format(saved_file))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

## TODO: bugs here, scatter plot is not working, for the shape of X
def plot_cluster_info(n_clusters, silhouette_avg, X, clusterer, cluster_labels):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The silhouette coefficient can range from -0.1, 1
    ax1.set_xlim([-0.1, 1])
    y_lower = 10
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # [5000, 1] for CIFAR-10 here
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )
    
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    
    plt.savefig('./images/silhouette_n_{}.pdf'.format(n_clusters), dpi=1500)