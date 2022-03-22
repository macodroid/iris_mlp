from mlp import *


class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid, n_classes):
        self.n_classes = n_classes
        super().__init__(dim_in, dim_hid, dim_out=n_classes)

    
    # @override
    def f_hid(self, x):
        """
        Activation function for hidden layer.
        Function sigmoid
        """
        return 1 / (1 + np.exp(-x))

    # @override
    def df_hid(self, x):
        """
        Derivation of sigmoid
        """
        return self.f_out(x) * (1 - self.f_out(x))

    # @override
    def f_out(self, x):
        """
        Activation function for output layer.
        Function sigmoid
        """
        return 1 / (1 + np.exp(-x))

    # @override
    def df_out(self, x):
        """
        Derivation of sigmoid
        """
        return self.f_out(x) * (1 - self.f_out(x))

    def error(self, targets, outputs):  # new
        '''
        Cost / loss / error function
        '''
        return np.sum((targets - outputs) ** 2, axis=0)

    def train(self):
        pass

    def predict(self):
        pass

    def test(self):
        pass