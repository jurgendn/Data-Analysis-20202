import numpy as np

def pickling_dataset(dataset):
    num_samples = len(dataset)
    dataset_flatten = dataset.data.reshape(num_samples,-1)
    targets_flatten = np.array(dataset.targets).reshape(num_samples,-1)
    dataset_with_label = np.hstack((dataset_flatten, targets_flatten))
    np.save('../clustering/data/test_dataset.npy', dataset_with_label)