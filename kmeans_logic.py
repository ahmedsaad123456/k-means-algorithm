import pandas as pd
import numpy as np
import random


# calculate the euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Randomly initialize centroids from the dataset
def initialize_centroids(data, num_clusters):
    indices = random.sample(range(len(data)), num_clusters)
    return data[indices]

# def initialize_centroids(data, num_clusters):
#     return data[:num_clusters]  # Take the first `num_clusters` points

# assign clusters to the data points based on the closest centroid
# and calculate the distance of each point from its assigned centroid
def assign_clusters_to_centroids(data, centroids):
    # To Store the cluster labels and distances
    cluster_labels = []
    point_distances = []  

    # For each point in the dataset, find the closest centroid
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]

        # Find the index of the closest centroid
        min_distance_index = np.argmin(distances)

        # Append the cluster label and distance to the lists
        cluster_labels.append(min_distance_index)
        point_distances.append(distances[min_distance_index])  
    return np.array(cluster_labels), np.array(point_distances)  # Return both labels and distances


# update the centroids by calculating the mean of all points assigned to each cluster
def update_centroids(data, cluster_labels, num_clusters):
    updated_centroids = []
    for cluster_id in range(num_clusters):
        # Get all points assigned to the current cluster
        cluster_points = data[cluster_labels == cluster_id]
        updated_centroids.append(cluster_points.mean(axis=0))
    return np.array(updated_centroids)


# K-means clustering algorithm
def kmeans_clustering(data, num_clusters):
    centroids = initialize_centroids(data, num_clusters)
    converged = False

    while not converged:
        cluster_labels, point_distances = assign_clusters_to_centroids(data, centroids)  
        new_centroids = update_centroids(data, cluster_labels, num_clusters)
        
        # Check for convergence by comparing old and new centroids
        # If the centroids do not change significantly, we can stop the algorithm
        if np.allclose(centroids, new_centroids):
            converged = True
        
        centroids = new_centroids

    return cluster_labels, centroids, point_distances


# Detect outliers in each cluster using the IQR method
def detect_cluster_outliers(cluster_labels, distances):
    outlier_indices = []
    for cluster_id in np.unique(cluster_labels):
        cluster_distances = distances[cluster_labels == cluster_id]
        Q1 = np.percentile(cluster_distances, 25)
        Q3 = np.percentile(cluster_distances, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR  
        upper_bound = Q3 + 1.5 * IQR 
        cluster_data_indices = np.where(cluster_labels == cluster_id)[0]

        # Identify outliers based on the calculated bounds
        # Append the indices of outliers to the list
        for idx, dist in zip(cluster_data_indices, cluster_distances):
            if dist < lower_bound or dist > upper_bound:
                outlier_indices.append(idx)
    return outlier_indices

def process_data(file_path, percentage, num_clusters):
    df = pd.read_csv(file_path)

    # Sample the dataset based on the percentage before preprocessing
    if 0 < percentage <= 100:
        df = df.sample(frac=percentage / 100, random_state=42)

    # map categorical variables to numerical values
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    feature_columns = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']
    feature_data = df[feature_columns]

    # Normalize the feature data to the range [0, 1]
    normalized_features = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
    normalized_array = normalized_features.values

    cluster_labels, final_centroids, point_distances = kmeans_clustering(normalized_array, num_clusters)
    outliers = detect_cluster_outliers(cluster_labels, point_distances)
    
    df['Cluster'] = cluster_labels
    df['DistanceFromCentroid'] = point_distances
    df['IsOutlier'] = df.index.isin(outliers)

    return df, outliers, cluster_labels, final_centroids
