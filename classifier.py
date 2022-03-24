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

    def train(self, inputs, labels, alpha=0.1, eps=100, compute_accuracy=True):
        (_, count) = inputs.shape
        targets = onehot_encode(labels)

        CEs = []
        REs = []

        for ep in range(eps):
            CE = 0
            RE = 0


    def predict(self):
        pass

    def test(self):
        pass
