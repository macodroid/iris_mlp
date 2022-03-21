import numpy as np
import random

def ordinary_encoding(y):
    """
    Convert
    :param y: class labels. Correct shape (n,)
    :return: ['A','A','B','C'] -> [0,0,1,2]
    """


def split_train_validation(X, y, train_ratio=0.8):
    """
    Split data to train and validation.\n
    :param X: features
    :param y: labels
    :param train_ratio: (default: 0.8) ratio between train and validation dataset.
    :return: X_train, y_train, X_val, y_val
    """
    indices = np.arange(X.shape[0], dtype=int)
    random.shuffle(indices)
    split = int(train_ratio * X.shape[0])
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_features = X[train_indices]
    train_labels = y[train_indices]

    val_features = X[val_indices]
    val_labels = y[val_indices]
    return train_features, train_labels, val_features, val_labels


def add_bias(X):
    '''
    Add bias term to vector, or to every (column) vector in a matrix.
    '''
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)
