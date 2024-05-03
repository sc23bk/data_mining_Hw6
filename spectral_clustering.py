"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.cluster.vq import kmeans2
from scipy.linalg import eigh
from scipy.sparse import csgraph
from typing import Tuple,Optional

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def proximity_measure(x, y, sigma):
    """
    Calculate the Gaussian (RBF) kernel similarity between two points.

    Arguments:
    - x: First data point (array-like).
    - y: Second data point (array-like).
    - sigma: Width of the Gaussian kernel.

    Returns:
    - Gaussian kernel similarity as a float.
    """
    dist_squared = np.linalg.norm(x - y) ** 2
    return np.exp(-dist_squared / (2 * sigma ** 2))

def plot_clustering(data, labels, title):
    """
    Plot clustering results with a scatter plot.

    Arguments:
    - data: A 2D numpy array where rows represent samples and columns represent features.
    - labels: Cluster labels for each sample in data.
    - title: The title for the plot.

    Displays:
    - A scatter plot of the clustering results.
    """
    # Create a scatter plot of the data, coloring the points by their cluster label
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6, edgecolors='w')

    # Set plot properties
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label').set_label('Cluster', rotation=270, labelpad=15)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Show the plot
    plt.show()

def adjusted_rand_index(true_labels, pred_labels):
    """
    Compute the adjusted Rand index using a vectorized approach.

    Arguments:
    - true_labels: true labels of the data
    - pred_labels: predicted labels by the clustering algorithm

    Return value:
    - Adjusted Rand index
    """
    # Create a contingency matrix by cross-tabulating predicted and true labels
    unique_true, inverse_true = np.unique(true_labels, return_inverse=True)
    unique_pred, inverse_pred = np.unique(pred_labels, return_inverse=True)
    contingency_matrix = np.bincount(inverse_true * len(unique_pred) + inverse_pred, minlength=len(unique_true) * len(unique_pred)).reshape(len(unique_true), len(unique_pred))

    # Compute the sums over rows, columns, and the grand total
    sum_rows = np.sum(contingency_matrix, axis=1)
    sum_cols = np.sum(contingency_matrix, axis=0)
    sum_all = np.sum(contingency_matrix)

    # Compute combinatorial terms for the sums
    sum_comb_rows = np.sum(sum_rows * (sum_rows - 1)) / 2
    sum_comb_cols = np.sum(sum_cols * (sum_cols - 1)) / 2
    sum_comb = np.sum(contingency_matrix * (contingency_matrix - 1)) / 2

    # Compute expected and maximum indices for the Rand index
    expected_index = sum_comb_rows * sum_comb_cols / (sum_all * (sum_all - 1))
    max_index = (sum_comb_rows + sum_comb_cols) / 2

    # Calculate the adjusted Rand index
    return (sum_comb - expected_index) / (max_index - expected_index)

def compute_SSE(data, labels):
    """
    Compute the Sum of Squared Errors (SSE) for cluster validation.

    Arguments:
    - data: A numpy array where each row represents a data point.
    - labels: A numpy array of cluster labels for each data point.

    Return value:
    - SSE: The total sum of squared errors for all clusters.
    """
    sse = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    Optional[NDArray[np.int32]], Optional[float], Optional[float], Optional[NDArray[np.floating]]
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """

    sigma = params_dict['sigma']
    k = params_dict['k']

    # Efficient vectorized computation of the similarity matrix
    norm_sq = np.sum(data**2, axis=1)
    dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2 * np.dot(data, data.T)
    similarity_matrix = np.exp(-dist_sq / (2 * sigma ** 2))

    # Construct the Laplacian matrix
    laplacian_matrix = csgraph.laplacian(similarity_matrix, normed=True)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    # Perform k-means clustering on the first k eigenvectors
    _, computed_labels = kmeans2(eigenvectors[:, :k], k, minit='++')

    # Compute SSE and ARI
    SSE = compute_SSE(data, computed_labels)
    ARI = adjusted_rand_index(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues

def spectral_hyperparameter_study(data, labels):
    """
    Perform hyperparameter study for spectral clustering on the given data.

    Arguments:
    - data: input data array of shape (n_samples, n_features)
    - labels: true labels of the data

    Return values:
    - sigmas: Array of sigma values
    - ari_scores: Array of ARI scores for each sigma value
    - sse_scores: Array of SSE scores for each sigma value
    """
    # Define the range and number of sigma values to test
    sigmas = np.logspace(-1, 1, num=10)  # Sigma values from 0.1 to 10
    k = 5  # Number of clusters to use in spectral clustering
    ari_scores, sse_scores = [], []

    # Iterate over each sigma value to perform spectral clustering
    for sigma in sigmas:
        # Perform spectral clustering with the current sigma and k values
        _, sse, ari, _ = spectral(data, labels, {'sigma': sigma, 'k': k})
        # Append the results for SSE and ARI to their respective lists
        sse_scores.append(sse)
        ari_scores.append(ari)
        # Output the ARI score for monitoring progress
        print(f"ARI: {ari}")

    return sigmas, np.array(ari_scores), np.array(sse_scores)


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    
    cluster_data = np.load('question1_cluster_data.npy')
    cluster_labels = np.load('question1_cluster_labels.npy')
    
    required_data = cluster_data[:1000]
    required_labels = cluster_labels[:1000]
    
    sigmas, ari_scores, sse_scores = spectral_hyperparameter_study(required_data, required_labels)
    
    
    # After hyperparameter study
    best_sigma = 0.1
    best_k = 5
    
    plots_values = {}

    # Apply best hyperparameters on five slices of data
    for i in range(5):  # Use range for clarity and Pythonic style
        # Slice the data and labels for each segment
        data_slice = cluster_data[i * 1000: (i + 1) * 1000]
        labels_slice = cluster_labels[i * 1000: (i + 1) * 1000]
        
        # Perform spectral clustering on each slice with the best hyperparameters
        computed_labels, sse, ari, eig_values = spectral(data_slice, labels_slice, {'sigma': best_sigma, 'k': best_k})
        
        # Store results in groups and plots_values dictionaries
        groups[i] = {"sigma": best_sigma, "ARI": ari, "SSE": sse}
        plots_values[i] = {"computed_labels": computed_labels, "ARI": ari, "SSE": sse, "eig_values": eig_values}

    # Identify the dataset with the highest ARI
    highest_ari = -1
    best_dataset_index = None
    for i, group_info in plots_values.items():
        if group_info['ARI'] > highest_ari:
            highest_ari = group_info['ARI']
            best_dataset_index = i  # Correct the variable to lowercase 'i'
            
    
    # Plot the clusters for the dataset with the highest ARI
    # Start by setting up the figure with a specific size
    plt.figure(figsize=(8, 6))

    # Extract the relevant data slice based on the best dataset index
    data_slice_x = cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 0]
    data_slice_y = cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 1]
    computed_labels = plots_values[best_dataset_index]["computed_labels"]

    # Create a scatter plot of the clustering results for the best dataset
    plot_ARI = plt.scatter(data_slice_x, data_slice_y, c=computed_labels, cmap='viridis', s=10, edgecolor='k')

    # Add a color bar to the plot to show cluster labels
    plt.colorbar(label='Cluster ID')

    # Enhance the plot with a title, axis labels, and a grid
    plt.title(f'Best Clustering Result with Highest ARI (k={best_k})', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot and then close it to free up memory
    plt.savefig('Spec_Dataset with high ARI.png')
    plt.close()
    
    # Find the dataset with the lowest SSE
    # Initialize variables for finding the dataset with the lowest SSE
    lowest_sse = float('inf')
    best_dataset_index_sse = None

    # Iterate over each group's information to find the one with the lowest SSE
    for i, group_info in plots_values.items():
        if group_info['SSE'] < lowest_sse:
            lowest_sse = group_info['SSE']
            best_dataset_index_sse = i

    # Set up the plot for the dataset with the lowest SSE
    plt.figure(figsize=(8, 6))

    # Extract coordinates and labels for the dataset with the lowest SSE
    data_slice_x = cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 0]
    data_slice_y = cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 1]
    labels = plots_values[best_dataset_index_sse]["computed_labels"]

    # Create a scatter plot of the clusters
    plot_SSE = plt.scatter(data_slice_x, data_slice_y, c=labels, cmap='viridis', edgecolor='k')

    # Add a color bar, title, labels, and grid to the plot
    plt.colorbar(label='Cluster ID')
    plt.title(f'Clustering with Lowest SSE (k={best_k})', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save and close the plot
    plt.savefig('Spec_Dataset with low SSE.png')
    plt.close()
    
    
    # Initialize the figure for plotting eigenvalues
    plt.figure(figsize=(8, 6))

    # Iterate over each dataset's information stored in plots_values
    for i, group_info in plots_values.items():
        # Sort eigenvalues before plotting and store the plot object in plot_eig
        sorted_eigenvalues = np.sort(group_info["eig_values"])
        plot_eig = plt.plot(sorted_eigenvalues, label=f'Dataset {i+1}')

    # Set the plot title, labels, and grid for better visualization
    plt.title('Eigenvalues Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Eigenvalue Index', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.legend(loc='upper right')  # Add a legend to help distinguish between datasets
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot to a file and then close the plot to free up resources
    plt.savefig('Spec_Eigenvalues Plot.png')
    plt.close()
    
 
    

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"] #{}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    #plot_ARI = plt.scatter([1,2,3], [4,5,6])
    #plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    #plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    
    ari_values = [group_info["ARI"] for group_info in groups.values()]
    mean_ari = np.mean(ari_values)
    std_dev_ari = np.std(ari_values)

    # A single float
    answers["mean_ARIs"] = mean_ari

    # A single float
    answers["std_ARIs"] = std_dev_ari
    
    sse_values = [group_info["SSE"] for group_info in groups.values()]
    mean_sse = np.mean(sse_values)
    std_dev_sse = np.std(sse_values)

    # A single float
    answers["mean_SSEs"] = mean_sse

    # A single float
    answers["std_SSEs"] = std_dev_sse

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
