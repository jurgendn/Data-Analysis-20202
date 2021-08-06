
import numpy as np
from sklearn.cluster import KMeans


def kmeans_clustering(embb_vec: np.ndarray,
                    num_clusters: int):
    """
    Using Kmeans to cluster embbeding vector from raw images
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=0)
    clusters = kmeans.fit_predict(embb_vec)
    sum_squared = kmeans.inertia_
    return clusters, sum_squared

def get_cluster_class(all_labels: np.ndarray,
                        clusters: np.ndarray) -> np.ndarray:
    """
    Get the class that refer to each cluster.
    Class that have the most instances in a cluster will be
    assign as cluster's class reference. 
    """
    ref_classes = {}
    for i in range(len(np.unique(clusters))):
        cluster_idx = np.where(clusters == i,1,0)
        cluster_cls = np.bincount(all_labels[cluster_idx==1]).argmax()
        ref_classes[i] = cluster_cls
    return ref_classes

def get_class(ref_classes: np.ndarray,
                clusters: np.ndarray) -> np.ndarray:
    """
    Get actual class for each instances
    """
    pred_classes = np.zeros(len(clusters))
    for i in range(len(clusters)):
        pred_classes[i] = ref_classes[clusters[i]]
    return pred_classes