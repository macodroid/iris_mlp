from mlp import *
from utils import *


class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid,
                 n_classes, activation_hid,
                 activation_out, inti_weights_type,
                 lambd=None):
        self.lambd = lambd
        self.n_classes = n_classes
        self.activation_hid = activation_hid
        self.activation_out = activation_out
        self.lamb = lambd
        super().__init__(dim_in, dim_hid, dim_out=n_classes, init_type=inti_weights_type)

    # @private
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # @private
    def df_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # @private
    def relu(self, x):
        return np.maximum(0, x)

    # @private
    def df_relu(self, x):
        return 1 * (x > 0)

    # @override
    def f_hid(self, x):
        """
        Activation function for hidden layer.
        """
        if self.activation_hid == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_hid == 'relu':
            return self.relu(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_hid}')

    # @override
    def df_hid(self, x):
        """
        Hidden layer.
        """
        if self.activation_hid == 'sigmoid':
            return self.df_sigmoid(x)
        elif self.activation_hid == 'relu':
            return self.df_relu(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_hid}')

    # @override
    def f_out(self, x):
        """
        Activation function for output layer.
        """
        if self.activation_out == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_out == 'relu':
            return self.relu(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_out}')

    # @override
    def df_out(self, x):
        """
        Derivation of sigmoid
        """
        if self.activation_out == 'sigmoid':
            return self.df_sigmoid(x)
        elif self.activation_out == 'relu':
            return self.df_relu(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_out}')

    def error(self, targets, outputs):  # new
        """
        Cost / loss / error function
        """
        return np.sum((targets - outputs) ** 2, axis=0)

    def train(self, train_X, train_y,
              val_X, val_y,
              optimizer,
              batch_size=None,
              alpha=0.1,
              eps=200,
              gamma=None,
              display_error=True):

        test_CEs = []
        test_REs = []
        val_CEs = []
        val_REs = []

        # Mini-batch gd
        if optimizer == 'mini-batch':
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
                test_CEs.append(CE)
                test_REs.append(RE)
                val_CE, val_RE = self.test(val_X, val_y)
                val_CEs.append(val_CE)
                val_REs.append(val_RE)
                if (ep + 1) % 5 == 0 and display_error:
                    print_errors('Train', CE, RE, ep, eps)
                    print_errors('Valid', val_CE, val_RE, ep, eps)
        # Mini-batch sgd using momentum
        elif optimizer == 'momentum':
            for ep in range(eps):
                len_batches, X, y = create_batches(train_X, train_y, batch_size)
                CE = 0
                RE = 0
                v_hid = np.zeros(self.W_hid.shape)
                v_out = np.zeros(self.W_out.shape)
                for batch in range(len_batches):
                    x = X[batch]
                    d = onehot_encode(y[batch])
                    a, h, b, yhat = self.forward(x)
                    dW_hid, dW_out = self.backpropagation(x, a, h, b, yhat, d)

                    v_hid = gamma * v_hid + alpha * dW_hid
                    v_out = gamma * v_out + alpha * dW_out

                    self.W_hid += dW_hid - v_hid
                    self.W_out += dW_out - v_out

                    CE += (np.not_equal(y[batch], onehot_decode(yhat))).sum()
                    RE += np.sum(self.error(d, yhat), axis=0)

                CE /= train_X.shape[0]
                RE /= train_X.shape[0]
                test_CEs.append(CE)
                test_REs.append(RE)
                val_CE, val_RE = self.test(val_X, val_y)
                val_CEs.append(val_CE)
                val_REs.append(val_RE)
                if (ep + 1) % 5 == 0 and display_error:
                    print_errors('Train', CE, RE, ep, eps)
                    print_errors('Valid', val_CE, val_RE, ep, eps)
        # SGD
        elif optimizer == 'sgd':
            for ep in range(eps):
                d = onehot_encode(train_y)
                a, h, b, yhat = self.forward(train_X)
                dW_hid, dW_out = self.backpropagation(train_X, a, h, b, yhat, d)

                self.W_hid += alpha * dW_hid
                self.W_out += alpha * dW_out

                CE = (np.not_equal(train_y, onehot_decode(yhat))).sum()
                RE = np.sum(self.error(d, yhat), axis=0)

                CE /= train_X.shape[0]
                RE /= train_X.shape[0]

                test_CEs.append(CE)
                test_REs.append(RE)

                val_CE, val_RE = self.test(val_X, val_y)

                val_CEs.append(val_CE)
                val_REs.append(val_RE)
                if (ep + 1) % 5 == 0 and display_error:
                    print_errors('Train', CE, RE, ep, eps)
                    print_errors('Valid', val_CE, val_RE, ep, eps)

        return test_CEs, test_REs, val_CEs, val_REs

    def test(self, inputs, labels, confusion_matrix=False, l2=True):
        """
        Test model: forward pass on given inputs, and compute errors
        """
        targets = onehot_encode(labels)
        outputs, predicted = self.predict(inputs)
        CE = (np.not_equal(labels, predicted)).sum() / inputs.shape[0]
        RE = np.mean((self.error(targets, outputs)))
        if confusion_matrix:
            conf_matrix = compute_confusion_matrix(labels, predicted)
            return CE, RE, conf_matrix
        return CE, RE

    def predict(self, inputs):
        """
        Prediction = forward pass
        """
        _, _, _, outputs = self.forward(inputs)
        return outputs, onehot_decode(outputs)
