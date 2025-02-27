from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd


def silhouette_score_clusters(data, dims, plot=False):
    # Range of clusters to try
    k_values = range(2, 11)  # Start from 2 since k = 1 is not useful for clustering

    silhouette_scores = []

    # Run K-means for each value of K and calculate silhouette score
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data[dims])
        score = silhouette_score(data[dims], labels)
        silhouette_scores.append(score)
        # print(f"For k = {k}, Silhouette Score = {score:.4f}")
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='b')
        plt.title('Silhouette Score for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.show()
    else:
        pass
    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"The optimal number of clusters is: {best_k}")

    # Check if k = 1 was the optimal value
    if best_k == 1:
        print("Warning: The optimal number of clusters is 1, meaning the data might not have distinct clusters.")

    kmeans = KMeans(n_clusters=best_k)
    labels = kmeans.fit_predict(data[dims])
    return labels


"""
Function to concatenate a certain number of PCs describing dimensions of the data per population
"""
def concat_dims(path_data, nr_PCs_per_dim, pop):
    PCs_labels = []
    for i in range(nr_PCs_per_dim):
        PCs_labels.append(f"PC{i+1}")
    dims = [f for f in os.listdir(path_data) if f.startswith('PCs')]
    dims_df = pd.DataFrame()
    for dim  in dims:
        path_dim = f"{path_data}/{dim}"
        dim = dim.split('PCs_')[1].split('.pkl')[0]
        PCs = pd.read_pickle(path_dim)
        for label in PCs_labels:
            PCs.rename(columns={f"{label}": f"{label}_{dim}_pop_{pop}"}, inplace=True)
            dims_df[f"{label}_{dim}_pop_{pop}"] = PCs[f"{label}_{dim}_pop_{pop}"]
    return dims_df


"""
make for every dims a concatenated dataframe so we have a df with all the dims for every pop, sort of one hot encoded

"""
def concat_dims_one_hot(path_iterations, nr_PCs_per_dim):


    iterations = [f for f in os.listdir(f"{path_iterations}") if f.startswith('iteration')]
    iterations.sort()
    for iteration in iterations:
        path_iteration = f"{path_iterations}/{iteration}"
        ids = pd.read_pickle(f"{path_iteration}/ids.pkl")
        dims_dfs = []
        for pop in [f for f in os.listdir(path_iteration) if f.startswith('pop')]:
            pop = pop.split('pop_')[1]
            temp_ids = ids[ids[f"cluster_{iteration.split('_')[1]}"] == pop]
            path_pop = f"{path_iteration}/pop_{pop}/"
            dims = [f for f in os.listdir(path_pop) if f.startswith('PCs')]
            
            if iteration == 'iteration_1':
                PCs_labels = ['PC1']
            else:
                nr_PCs = nr_PCs_per_dim
                PCs_labels = []
                for i in range(nr_PCs):
                    PCs_labels.append(f"PC{i+1}")

            dims_df = pd.DataFrame()
            for dim  in dims:
                path_dim = f"{path_pop}/{dim}"
                dim = dim.split('PCs_')[1].split('.pkl')[0]
                
                PCs = pd.read_pickle(path_dim)
                for label in PCs_labels:
                    PCs.rename(columns={f"{label}": f"pop_{pop}_{label}_{dim}"}, inplace=True)
                    dims_df[f"pop_{pop}_{label}_{dim}"] = list(PCs[f"pop_{pop}_{label}_{dim}"])
            dims_df.index = temp_ids.index
            dims_dfs.append(dims_df)
            

        # Concatenate DataFrames, filling missing columns with NaN
        dims_dfs = pd.concat(dims_dfs, ignore_index=False, sort=True)
        dims_dfs = dims_dfs.fillna(0)
        scaler = StandardScaler()
        scaler = MaxAbsScaler()
        dims_dfs[:] = scaler.fit_transform(dims_dfs)
        dims_dfs.to_pickle(f"{path_iteration}/concated_one_hot_dims.pkl")
