from sklearn.cluster import (
    MiniBatchKMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
    KMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cluster(data, n_clusters=None, algo="minibatch_kmeans", add_start=False, add_end=False, **kwargs):
    """
    Generic clustering entry point.
    Supported algos: minibatch_kmeans, gmm, agglomerative, spectral, dbscan
    """
    logging.info(f"Begin clustering with {algo} algorithm")
    
    if algo == "kmeans":
        labels, clusters = kmeans(data, n_clusters)
    elif algo == "minibatch_kmeans":
        labels, clusters = minibatch_kmeans(data, n_clusters)
    elif algo == "gmm":
        labels, clusters = gmm_cluster(data, n_clusters)
    elif algo == "agglomerative":
        labels, clusters = agglomerative_cluster(data, n_clusters)
    elif algo == "spectral":
        labels, clusters = spectral_cluster(data, n_clusters)
    elif algo == "dbscan":
        labels, clusters = dbscan(data, **kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    if add_start:
        # Add a cluster at start euclidean point, no velocity
        start_cluster = np.array([[0,1.5,0,0,0,0,0,0]])
        clusters = np.concatenate([clusters, start_cluster], axis=0)
    
    if add_end:
        # Add a cluster at start euclidean point, no velocity
        end_cluster = np.array([[0,0,0,0,0,0,1,1]])
        clusters = np.concatenate([clusters, end_cluster], axis=0)
        
    return labels, clusters
   
def minibatch_kmeans(data, n_clusters):
    """
    Faster version of K-Means for large datasets.
    """
    data = np.array([x.flatten() for x in data])
    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
    mbk.fit(data)
    return mbk.labels_, mbk.cluster_centers_


def gmm_cluster(data, n_clusters):
    """
    Gaussian Mixture Model clustering (soft, elliptical clusters).
    """
    data = np.array([x.flatten() for x in data])
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(data)
    labels = gmm.predict(data)
    centers = gmm.means_
    return labels, centers


def agglomerative_cluster(data, n_clusters):
    """
    Hierarchical (Agglomerative) clustering.
    """
    data = np.array([x.flatten() for x in data])
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(data)
    centers = np.array([data[labels == i].mean(axis=0) for i in np.unique(labels)])
    return labels, centers


def spectral_cluster(data, n_clusters):
    """
    Spectral clustering (non-convex clusters).
    """
    data = np.array([x.flatten() for x in data])
    spec = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    labels = spec.fit_predict(data)
    centers = np.array([data[labels == i].mean(axis=0) for i in np.unique(labels)])
    return labels, centers


def dbscan(data, eps=0.5, min_samples=5):
    """
    DBSCAN clustering (density-based, no need for k).
    """
    data = np.array([x.flatten() for x in data])
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)

    # Estimate cluster centers as mean of member points
    centers = []
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]
    for lbl in unique_labels:
        cluster_points = data[labels == lbl]
        centers.append(cluster_points.mean(axis=0))
    centers = np.array(centers)

    return labels, centers

 
def kmeans(data, n_clusters):
    data = np.array([x.flatten() if isinstance(x, np.ndarray) else np.array(x) for x in data])
    if data.ndim == 1:
        data = data.reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)

    # Get cluster assignments and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    return labels, centers

def obs_to_cluster(obs, centers):
    # Ensure inputs are numpy arrays
    obs = np.array(obs)
    centers = np.array(centers)

    # Flatten the observation if multi-dimensional
    if obs.ndim > 1:
        obs = obs.flatten()
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)

    # Handle case where obs is scalar or 1D but centers are multi-D
    if obs.ndim == 1 and centers.shape[1] != obs.shape[0]:
        obs = obs.reshape(-1)

    # Compute Euclidean distances
    difs = centers - obs
    euclidean_dist = np.sqrt(np.sum(np.square(difs), axis=1))

    # Return nearest cluster index and all distances
    return np.argmin(euclidean_dist), euclidean_dist


def analyze_k_clusters(obss, k_range=range(1, 20)):
    inertias = []

    # Flatten each observation once
    data = [x.flatten() for x in obss]

    print(f"Running k-means with k = ", end='')
    for k in k_range:
        print("", k, end=',', flush=True)
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        inertias.append(km.inertia_)  # total within-cluster sum of squares
    print("")

    # Plot elbow curve
    plt.figure(figsize=(7, 5))
    plt.plot(list(k_range), inertias, 'bo-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()

    return inertias

def plot_clusters(obss, cluster_centers, labels=None):
    """
    Visualizes clustered data in 3D space.

    Args:
        obss : list or np.ndarray
            List or array of observations (each of shape (3,) or (n_features=3)).
        cluster_centers : np.ndarray
            Array of shape (n_clusters, 3) containing cluster centers.
        labels : list or np.ndarray, optional
            Cluster assignment for each observation. If None, all points use same color.
    """
    obss = np.array([x.flatten() for x in obss])
    cluster_centers = np.array(cluster_centers)

    # Create 3D figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points
    if labels is not None:
        ax.scatter(
            obss[:, 0], obss[:, 1], obss[:, 2],
            c=labels, cmap="viridis", s=5, alpha=0.8
        )
    else:
        ax.scatter(
            obss[:, 0], obss[:, 1], obss[:, 2],
            color="gray", s=5, alpha=0.8
        )

    # Plot cluster centers
    ax.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
        c="red", s=200, marker="X", edgecolor="k", label="Centers"
    )

    ax.set_title("3D Cluster Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()
