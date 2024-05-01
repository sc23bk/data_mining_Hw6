"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
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
      
    def calculate_distance_matrix(data):
        points_count = len(data)
        matrix = np.zeros((points_count, points_count))
        for i in range(points_count):
            for j in range(points_count):
                matrix[i, j] = np.linalg.norm(data[i] - data[j])
        return matrix

    
    def convert_distance_to_similarity(matrix, sigma):
        similarities = np.exp(-(matrix ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(similarities, 0)
        return similarities

    matrix_distance = compute_distance_matrix(data)
    matrix_similarity = convert_distance_to_similarity(matrix_distance, sigma=params_dict['sigma'])


    matrix_diagonal = np.zeros_like(matrix_similarity)
    sum_rows = np.sum(matrix_similarity, axis=1)

    for index in range(matrix_diagonal.shape[0]):
      matrix_diagonal[index][index] = sum_rows[index]

    laplacian = matrix_diagonal - matrix_similarity

    eigen_values, eigenvectors = np.linalg.eigh(laplacian)

    diagonal_eigenvalues = np.diag(eigen_values)

    first_vectors = eigenvectors[:, :params_dict['k']]
    unit_vectors = first_vectors / np.linalg.norm(first_vectors, axis=1, keepdims=True)

    def choose_initial_centroids(data, k):
        """Selects k random data points as initial centroids."""
        random_indices = np.random.choice(len(data), k, replace=False)
        return data[random_indices]

    def find_clusters(data, centroids):
        """Determine the closest centroid for each data point."""
        centroids_reshaped = centroids.reshape(1, -1, centroids.shape[-1])
        all_distances = np.sqrt(((data[:, np.newaxis] - centroids_reshaped)**2).sum(axis=2))
        return np.argmin(all_distances, axis=1)

    def recalculate_centroids(data, cluster_labels, k):
        """Recalculate centroids by averaging points in each cluster."""
        updated_centroids = np.array([data[cluster_labels == cluster].mean(axis=0) for cluster in range(k) if np.any(cluster_labels == cluster)])
        return updated_centroids

    def calculate_sse(data, centroids, cluster_labels):
        """Calculate the total squared error for the clusters."""
        total_sse = sum(np.sum((data[cluster_labels == idx] - centroids[idx])**2) for idx in range(centroids.shape[0]))
        return total_sse

    def execute_k_means(data, k, max_iterations=300):
        """Performs k-means clustering."""
        centroids = choose_initial_centroids(data, k)
        for iteration in range(max_iterations):
            cluster_labels = find_clusters(data, centroids)
            updated_centroids = recalculate_centroids(data, cluster_labels, k)
            current_sse = calculate_sse(data, updated_centroids, cluster_labels)
            if np.array_equal(centroids, updated_centroids):
                break
            centroids = updated_centroids
        return centroids, cluster_labels, current_sse

    centroids, cluster_labels, total_sse = execute_k_means(unit_vectors, params_dict['k'])

    def compute_adjusted_rand_index(true_labels, predicted_labels):
        unique_classes = np.unique(true_labels)
        unique_clusters = np.unique(predicted_labels)
        contingency = np.zeros((len(unique_classes), len(unique_clusters)), dtype=int)
        for i, class_val in enumerate(unique_classes):
            for j, cluster_val in enumerate(unique_clusters):
                contingency[i, j] = np.sum((true_labels == class_val) & (predicted_labels == cluster_val))

        rows_sum = np.sum(contingency, axis=1)
        cols_sum = np.sum(contingency, axis=0)

        combination_sum = sum(comb * (comb - 1) / 2 for comb in contingency.flatten())
        rows_comb_sum = sum(r * (r - 1) / 2 for r in rows_sum)
        cols_comb_sum = sum(c * (c - 1) / 2 for c in cols_sum)

        total_combinations = true_labels.size * (true_labels.size - 1) / 2
        expected_combinations = rows_comb_sum * cols_comb_sum / total_combinations
        max_combinations = (rows_comb_sum + cols_comb_sum) / 2

        if max_combinations - expected_combinations == 0:
            return 1 if combination_sum == expected_combinations else 0

        adjusted_index = (combination_sum - expected_combinations) / (max_combinations - expected_combinations)
        return adjusted_index

    true_labels = labels
    predicted_labels = cluster_labels

    ari = compute_adjusted_rand_index(true_labels, predicted_labels)


    computed_labels: NDArray[np.int32] | None = cluster_labels
    SSE: float | None = total_sse
    ARI: float | None = ari
    eigenvalues: NDArray[np.floating] | None = eigen_values
    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}
    data = np.load("question1_cluster_data.npy")
    true_labels = np.load("question1_cluster_labels.npy")
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    sse_final = []
    preds_final = []
    ari_final = []
    eigen_final = []
    for idx in range(5):
        datav = data[idx * 1000:(idx + 1) * 1000]
        true_labelsv = true_labels[idx * 1000:(idx + 1) * 1000]
        params_dict = {'k': 5, 'sigma': 0.1}
        preds, sse_hyp, ari_hyp, eigen_val = spectral(datav, true_labelsv, params_dict)
        sse_final.append(sse_hyp)
        ari_final.append(ari_hyp)
        preds_final.append(preds)
        eigen_final.append(eigen_val)
      if idx not in groups:
        groups[idx]={'sigma':0.1,'ARI':ari_hyp,"SSE":sse_hyp}
        
      else:
        pass

    sse_numpy = np.array(sse_final)
    ari_numpy = np.array(ari_final)
    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']
    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.
    least_sse_index=np.argmin(sse_numpy)
    highest_ari_index=np.argmax(ari_numpy)
    lowest_ari_index=np.argmin(ari_numpy)
    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter(data[1000 * highest_ari_index:(highest_ari_index + 1) * 1000, 0],
                           data[1000 * highest_ari_index:(highest_ari_index + 1) * 1000, 1],
                           c=preds_final[highest_ari_index], cmap='viridis', marker='.')
    plt.title('Largest ARI')
    plt.xlabel(f'Feature 1 for Dataset{highest_ari_index + 1}')
    plt.ylabel(f'Feature 2 for Dataset{highest_ari_index + 1}')
    plt.grid(True)

    
    plot_SSE = plt.scatter(data[1000 * least_sse_index:(least_sse_index + 1) * 1000, 0],
                           data[1000 * least_sse_index:(least_sse_index + 1) * 1000, 1],
                           c=preds_final[least_sse_index], cmap='viridis', marker='.')
    plt.title('Least SSE')
    plt.xlabel(f'Feature 1 for Dataset{least_sse_index + 1}')
    plt.ylabel(f'Feature 2 for Dataset{least_sse_index + 1}')
    plt.grid(True)
    plt.savefig('SpectralClusteringARI.png')
    
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # # Plot of the eigenvalues (smallest to largest) as a line plot.
    # # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.

    value_to_plot_eva = [val for sublist in eigen_final for val in sublist]
    plt.title('Eigen Values Sorted')
    plot_eig = plt.plot(sorted(value_to_plot_eva))
    plt.xlabel(f'Eigen Values Sorted in Ascending')
    plt.grid(True)
    plt.savefig('SpectralClustering.png')
    
    answers['eigenvalue plot']=plot_eig
    plt.close()

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    ARI_sum=[]
    SSE_sum=[]
    for i in groups:
      if 'ARI' in groups[i]:
        ARI_sum.append(groups[i]['ARI'])
        SSE_sum.append(groups[i]['SSE'])
    
    # A single float
    answers["mean_ARIs"] = float(np.mean(ari_numpy))

    # A single float
    answers["std_ARIs"] = float(np.std(ari_numpy))

    # A single float
    answers["mean_SSEs"] = float(np.mean(sse_numpy))

    # A single float
    answers["std_SSEs"] = float(np.std(sse_numpy))

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
