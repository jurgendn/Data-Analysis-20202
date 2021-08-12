
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

def Jaccard_score(clusters,all_labels):
    nber_real_class = len(set(all_labels))
    nber_clusters = len(set(clusters))
    dct_class={}
    for i in list(set(all_labels)):
        temp=[j for j,label in enumerate(all_labels) if label == i]
        dct_class[i] = temp
    dct_cluster={}
    for i in list(set(clusters)):
        temp=[j for j,label in enumerate(clusters) if label == i]
        dct_cluster[i] = temp
    matrix_Jaccard=np.zeros((nber_real_class,nber_clusters))
    for i in range(len(list(set(all_labels)))):
        for j in range(len(list(set(clusters)))):
            matrix_Jaccard[i,j] = Jaccard_index_formular(dct_class[list(set(all_labels))[i]],
            dct_cluster[list(set(clusters))[j]])
    cluster_closest_labels = [list(set(clusters))[j] for j in list(np.argmax(matrix_Jaccard,axis = 1))]
    label_closest_cluster =  [list(set(all_labels))[j] for j in list(np.argmax(matrix_Jaccard,axis = 0))]
    print("thu tu label:",list(set(all_labels)))
    print("cac label gan voi cluster theo thu tu:",cluster_closest_labels)
    print("JC index tuong ung:",list(np.max(matrix_Jaccard,axis = 1)))
    print("-----000------")
    print("thu tu cluster:",list(set(clusters)))
    print("cac cluster gan voi cac label theo thu tu:",label_closest_cluster)
    print("JC index tuong ung:",list(np.max(matrix_Jaccard,axis = 0)))
    print("*********")
    print("matrix Jaccard:",matrix_Jaccard)
    return 1
def Jaccard_index_formular(set1,set2):
    a=set(set1)
    b=set(set2)
    T= len(a & b)
    M= len(a | b)
    return T/M 
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
