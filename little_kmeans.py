import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Function to generate and plot dataset
def generate_and_plot_data(centers):
    X, _ = make_blobs(n_samples=600, centers=centers, cluster_std=0.60, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.title('Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    return X

# Function to perform K-means clustering and plot the results
def plot_kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(f'K-means Clustering with {n_clusters} Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Function to determine optimal number of clusters using Silhouette Score
def find_optimal_clusters(X):
    silhouette_scores = []
    for n_clusters in range(2, 20):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_clusters = np.argmax(silhouette_scores) + 2
    plt.plot(range(2, 20), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    return optimal_clusters


# Generate and plot the dataset (you can input an integer as centers of initial data)
print("Please input an integer as n for centers: ")
user_input1 = input()
user_centers = int(user_input1)
X = generate_and_plot_data(user_centers)

# User-defined number of clusters for K-means 
print("Please input an integer as n for n-clusters: ")
user_input2 = input()
user_clusters = int(user_input2)

print("Performing K-means with user's input clusters integer...")
plot_kmeans_clustering(X, user_clusters)


# Calculate and visualize the optimal number of clusters
print("Calculating the optimal number of clusters...")
optimal_clusters = find_optimal_clusters(X)
print(f"Optimal number of clusters based on Silhouette Score: {optimal_clusters}")

# Perform K-means with optimal clusters
print(f"Performing K-means with the optimal number of clusters ({optimal_clusters})...")
plot_kmeans_clustering(X, optimal_clusters)
