from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

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