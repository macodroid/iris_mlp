import numpy as np

from mlp import *
from utils import *


class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid, n_classes):
        self.n_classes = n_classes
        super().__init__(dim_in, dim_hid, dim_out=n_classes)

    # @private
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # @private
    def _derivation_sigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    # @private
    def _relu(self, x):
        return x * (x > 0)

    # @private
    def _derivation_relu(self, x):
        return 1. * (x > 0)

    # @override
    def f_hid(self, x):
        """
        Activation function for hidden layer.
        """
        return self._sigmoid(x)
        # return self._relu(x)

    # @override
    def df_hid(self, x):
        """
        Hidden layer.
        """
        return self._derivation_sigmoid(x)
        # return self._derivation_relu(x)

    # @override
    def f_out(self, x):
        """
        Activation function for output layer.
        """
        return self._sigmoid(x)
        # return self._relu(x)

    # @override
    def df_out(self, x):
        """
        Derivation of sigmoid
        """
        return self._derivation_sigmoid(x)
        # return self._derivation_relu(x)

    def error(self, targets, outputs):  # new
        '''
        Cost / loss / error function
        '''
        return np.sum((targets - outputs) ** 2, axis=0)

    def train(self, train_X, train_y, val_X, val_y, batch_size, alpha=0.1, eps=500, compute_accuracy=True):
        p = [1, 1, 1, 1]

        CEs = []
        REs = []

        for ep in range(eps):
            len_batches, X, y = create_batches(train_X, train_y, batch_size)
            CE = 0
            RE = 0
            for batch in range(len_batches):
                x = X[batch]
                d = onehot_encode(y[batch])
                a, h, b, yhat = self.forward(x)
                dW_hid, dW_out = self.backpropagation(x, a, h, b, yhat, d)
                self.W_hid += alpha * dW_hid
                self.W_out += alpha * dW_out
                CE += (np.not_equal(y[batch], onehot_decode(yhat))).sum()
                RE += np.sum(self.error(d, yhat), axis=0)

            CE /= train_X.shape[0]
            RE /= train_X.shape[0]
            CEs.append(CE)
            REs.append(RE)
            if (ep + 1) % 5 == 0:
                print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep + 1, eps, CE, RE))
        return p[0], p[0], p[0], p[0]

    # def train(self, train_X, train_y, val_X, val_y, batch_size, alpha=0.1, eps=500, compute_accuracy=True):
    #     train_count = train_X.shape[0]
    #     val_count = val_X.shape[0]
    #     y_train_encoded = onehot_encode(train_y)
    #     y_val_encoded = onehot_encode(val_y)
    #
    #     CEs = []
    #     REs = []
    #     batches = create_batches(train_X, y_train_encoded, batch_size)
    #     for ep in range(eps):
    #         CE = 0
    #         RE = 0
    #         sample_train = np.random.choice(train_count, batch_size)
    #         sample_val = np.random.choice(val_count, batch_size)
    #         x = train_X[sample_train]
    #         d = y_train_encoded[sample_train]
    #         a, h, b, yhat = self.forward(x)
    #         dW_hid, dW_out = self.backpropagation(x, a, h, b, yhat, d)
    #
    #         self.W_hid += alpha * dW_hid
    #         self.W_out += alpha * dW_out
    #
    #         CE = (np.not_equal(train_y[sample_train], onehot_decode(yhat))).sum()
    #
    #         RE = np.sum(self.error(d, yhat), axis=0)
    #
    #         CE /= batch_size
    #         RE /= batch_size
    #         CEs.append(CE)
    #         REs.append(RE)
    #         if (ep + 1) % 5 == 0:
    #             print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep + 1, eps, CE, RE))
    #
    #     print("a")


def predict(self):
    pass


def test(self):
    pass


def create_batches(X, y, batch_size):
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
