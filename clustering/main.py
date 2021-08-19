import os
import argparse
import logging
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

from utils.log import plot_confusion_matrix
from utils.utils import get_class, get_cluster_class, kmeans_clustering, plot_nearest_samples

from tensorboardX import SummaryWriter

logger = logging.getLogger()

def main(args):
    # Setting up logging tools
    exp_dir = os.path.join(os.getcwd(), 'exps', args.exp_name)
    writer = SummaryWriter(exp_dir)
    
    # Get embedding vectors and images
    test_dataset = np.load(args.images_path, allow_pickle=True)
    images = test_dataset[:, :-1].reshape(10000,32,32,3)
    
    codes = np.load(args.codes_path, allow_pickle=True)
    flatten_images = codes[:, :-1]
    all_labels = codes[:, -1].astype('int8')

    class_names = ['plane', 'car', 'bird', 'car', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    # Run KMeans
    for n in range(args.min_cluster, args.max_cluster + 1):
        print(f'KMeans with {n} clusters')
        clusters, sum_squared, distances = kmeans_clustering(embb_vec=flatten_images, num_clusters=n)
        ref_classes = get_cluster_class(all_labels, clusters)
        predicted = get_class(ref_classes, clusters)

        acc = accuracy_score(all_labels, predicted)
        cm = confusion_matrix(all_labels, predicted)
        cm_fig = plot_confusion_matrix(cm, class_names)

        fig_list = plot_nearest_samples(images, clusters, distances, ref_classes, class_names)
        
        import ipdb
        writer.add_figure(tag='test', figure=fig_list, global_step=n)
        # for i in range(len(fig_list)):
        #     writer.add_figure(tag=f'Cluster {i}', figure=fig_list[i], global_step=n)
        
        writer.add_scalar('Accuracy', acc, n)
        writer.add_scalar('Elbow', sum_squared, n)
        writer.add_figure(tag='Confusion Matrix', figure=cm_fig, global_step=n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes_path', type=str, default=None,
                        help='Path to embedding vectors npy file')
    parser.add_argument('--images_path', type=str, default=None,
                        help='Path to images npy file')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Set experiment directory name')
    parser.add_argument('--min_cluster', type=int, default=8,
                        help='Min number of clusters')
    parser.add_argument('--max_cluster', type=int, default=10,
                        help='Max number of clusters')
    args = parser.parse_args()

    main(args)