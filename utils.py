import numpy as np
import random


def onehot_encode(labels):
    """
    :param labels: label vector
    :param c: count of unique categories
    :return: matrix with onehot encoding
    """
    onehot_matrix = np.zeros((labels.size, labels.max() + 1))
    onehot_matrix[np.arange(labels.size), labels] = 1
    return onehot_matrix


def ordinal_encoding(y):
    """
    Convert categorical data to numerical.\n
    Example: ['A','A','B','C','C'] -> [0,0,1,2,2].\n
    :param y: class labels. Correct shape (n,)
    :return: numpy array with shape (n,)
    """
    i = 0
    unique_categories = np.unique(y)
    for cat in unique_categories:
        y[y == cat] = i
        i += 1
    return y.astype(int)


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
        pad = np.ones((X.shape[0],1))
        return np.concatenate((pad,X), axis=1)
