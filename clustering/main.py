import os
import argparse
import logging
from torchvision.datasets import cifar
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

from extractor import get_extractor

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import get_datamodule

from utils.log import plot_confusion_matrix
from utils.cluster_utils import get_class, get_cluster_class, kmeans_clustering

from tensorboardX import SummaryWriter

logger = logging.getLogger()

def data_process(dataset: Dataset,
                model: nn.Module):
    """
    Loop over dataset to gather all images and its labels then flatten
    """
    all_feats = []
    all_labels = []
    for idx in tqdm(range(len(dataset))):
        image, label = dataset[idx]
        if model is not None:
            with torch.no_grad():
                pseudo_batching = torch.unsqueeze(image, 0) 
                feat = model(pseudo_batching).flatten().numpy()
        else:
            feat = image.numpy()
        # make sure that feature vector is a numpy array
        all_feats.append(feat)
        all_labels.append(label)
    
    all_feats = np.array(all_feats)
    all_labels = np.array(all_labels)
    # flatten image shape
    flatten_images = all_feats.reshape(len(all_feats), -1)
    return flatten_images, all_labels


def main(args):
    # Setting up logging tools
    exp_dir = os.path.join(os.getcwd(), 'exps', args.exp_name)
    writer = SummaryWriter(exp_dir)
    
    # TODO: Load data samples through DataLoader to take advance of Batching
    # Setting up Model
    model = get_extractor(model_name = args.feature_extractor, checkpoint=args.ckpt)
    model.eval()

    # Setting up Dataloader
    cifar10_dm = get_datamodule()
    cifar10_dm.setup('test')
    dataset = cifar10_dm.test_dataloader().dataset
    
    flatten_images, all_labels = data_process(dataset, model)
    class_name = ['plane', 'car', 'bird', 'car', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    # Run KMeans
    for n in range(args.min_cluster, args.max_cluster + 1):
        print(f'KMeans with {n} clusters')
        clusters, sum_squared = kmeans_clustering(embb_vec=flatten_images, num_clusters=n)
        ref_classes = get_cluster_class(all_labels, clusters)
        predicted = get_class(ref_classes, clusters)
        
        acc = accuracy_score(all_labels, predicted)
        cm = confusion_matrix(all_labels, predicted)
        cm_fig = plot_confusion_matrix(cm, class_name)

        writer.add_scalar('Accuracy', acc, n)
        writer.add_scalar('Elbow', sum_squared, n)
        writer.add_figure(tag='Confusion Matrix', figure=cm_fig, global_step=n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to data directory')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Set experiment directory name')
    parser.add_argument('--min_cluster', type=int, default=8,
                        help='Min number of clusters')
    parser.add_argument('--max_cluster', type=int, default=10,
                        help='Max number of clusters')
    parser.add_argument('--feature_extractor', type=str, default=None,
                        help='Name of Feature extractor. Remeber to comment GrayScale transform in dataset')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to ckeckpoint. Only supported with vgg16_bn.')
    args = parser.parse_args()

    main(args)