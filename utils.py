import numpy as np
import random

from matplotlib import pyplot as plt


def onehot_encode(labels):
    """
    :param labels: label vector
    :return: matrix with onehot encoding
    """
    onehot_matrix = np.zeros((labels.size, labels.max() + 1))
    onehot_matrix[np.arange(labels.size), labels] = 1
    return onehot_matrix


def onehot_decode(X):
    return np.argmax(X, axis=1)


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
    """
    Add bias term to vector, or to every (column) vector in a matrix.
    """
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((X.shape[0], 1))
        return np.concatenate((pad, X), axis=1)


def plot_train_val_error(trainCE, trainRE, valCE, valRE):
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'hspace': 0.55})
    ax1.set_title("Test vs Validation classification error")
    ax1.plot(trainCE, '-r', label='train classification error')
    ax1.plot(valCE, '-g', label='validation classification error')
    ax1.set(xlabel='epoch', ylabel='error')
    ax1.legend(loc='best', shadow=True, fontsize='small')
    ax2.set_title("Test vs Validation regression error")
    ax2.plot(trainRE, '-r', label='train regression error')
    ax2.plot(valRE, '-g', label='validation regression error')
    ax2.set(xlabel='epoch', ylabel='error')
    ax2.legend(loc='best', shadow=True, fontsize='small')
    plt.legend()
    plt.show()


def compute_confusion_matrix(true, pred):
    K = len(np.unique(true))
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def plot_confusion_matrix(matrix, v):
    classes = ['A', 'B', 'C']
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(matrix, cmap=plt.cm.Greens, alpha=0.3)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def create_batches(X, y, batch_size):
    """
    Create mini batches from dataset.
    """
    mini_batches_X = []
    mini_batches_y = []
    batch_count = X.shape[0] // batch_size
    indices = np.arange(X.shape[0], dtype=int)
    random.shuffle(indices)
    indices = np.array_split(indices, batch_count)
    for i in indices:
        mini_batches_X.append(X[i])
        mini_batches_y.append(y[i])
    return batch_count, mini_batches_X, mini_batches_y


def print_errors(error_type, CE, RE, ep, eps):
    print(error_type + 'Error: Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep + 1, eps, CE, RE))
