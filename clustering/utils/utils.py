
import matplotlib.pyplot as plt
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

    distances = (kmeans.transform(embb_vec)**2).sum(axis=1)
    return clusters, sum_squared, distances

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

def plot_nearest_samples(images, clusters, distances, ref_classes, class_names):
    """
    Plot 100 nearest samples for each cluster
    """
    fig_list = []
    for i in range(len(np.unique(clusters))):
        cluster_idx = np.where(clusters == i,1,0)
        topk_distances_idx = distances[cluster_idx==1].argsort()[:100]
        topk_images = images[topk_distances_idx, :, :, :]
        
        mainly_cluster_name = class_names[ref_classes[i]]

        fig, axs = plt.subplots(10, 10)
        fig.suptitle(f'Cluster {i}: Mainly {mainly_cluster_name}', weight='bold', size=14)
        counter = 0
        for j in range(10):
            for k in range(10):
                axs[j][k].tick_params(axis='x', labelsize=12)
                axs[j][k].tick_params(axis='y', labelsize=12)
                axs[j][k].axis('off')
                axs[j][k].imshow(topk_images[counter])
                counter += 1

        fig_list.append(fig)

    return fig_list


        