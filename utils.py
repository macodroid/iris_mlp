import os

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


def plot_dots(inputs, labels=None, predicted=None, test_inputs=None, test_labels=None, test_predicted=None, s=60, i_x=0,
              i_y=1, title=None, block=True):
    plt.figure(title or 3)
    use_keypress()
    plt.clf()

    if inputs is not None:
        if labels is None:
            plt.gcf().canvas.set_window_title('Data distribution')
            plt.scatter(inputs[i_x, :], inputs[i_y, :], s=s, c=palette[-1], edgecolors=[0.4] * 3, alpha=0.5,
                        label='train data')

        elif predicted is None:
            plt.gcf().canvas.set_window_title('Class distribution')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x, labels == c], inputs[i_y, labels == c], s=s, c=palette[i], edgecolors=[0.4] * 3,
                            label='train cls {}'.format(c))

        else:
            plt.gcf().canvas.set_window_title('Predicted vs. actual')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x, labels == c], inputs[i_y, labels == c], s=2.0 * s, c=palette[i],
                            edgecolors=None, alpha=0.333, label='train cls {}'.format(c))

            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x, predicted == c], inputs[i_y, predicted == c], s=0.5 * s, c=palette[i],
                            edgecolors=None, label='predicted {}'.format(c))

        plt.xlim(limits(inputs[i_x, :]))
        plt.ylim(limits(inputs[i_y, :]))

    if test_inputs is not None:
        if test_labels is None:
            plt.scatter(test_inputs[i_x, :], test_inputs[i_y, :], marker='s', s=s, c=palette[-1], edgecolors=[0.4] * 3,
                        alpha=0.5, label='test data')

        elif test_predicted is None:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x, test_labels == c], test_inputs[i_y, test_labels == c], marker='s', s=s,
                            c=palette[i], edgecolors=[0.4] * 3, label='test cls {}'.format(c))

        else:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x, test_labels == c], test_inputs[i_y, test_labels == c], marker='s',
                            s=2.0 * s, c=palette[i], edgecolors=None, alpha=0.333, label='test cls {}'.format(c))

            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x, test_predicted == c], test_inputs[i_y, test_predicted == c], marker='s',
                            s=0.5 * s, c=palette[i], edgecolors=None, label='predicted {}'.format(c))

        if inputs is None:
            plt.xlim(limits(test_inputs[i_x, :]))
            plt.ylim(limits(test_inputs[i_y, :]))

    plt.legend()
    if title is not None:
        plt.gcf().canvas.set_window_title(title)
    plt.tight_layout()
    plt.show(block=block)


def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0 - xg, x1 + xg))


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close()  # skip blocking figures

palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']