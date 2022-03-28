from mlp import *
from utils import *
from activation_functions import *


class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid,
                 n_classes, activation_hid,
                 activation_out, inti_weights_type):
        self.n_classes = n_classes
        self.activation_hid = activation_hid
        self.activation_out = activation_out
        super().__init__(dim_in, dim_hid, dim_out=n_classes, init_type=inti_weights_type)

    # @override
    def f_hid(self, x):
        """
        Activation function for hidden layer.
        """
        if self.activation_hid == Activation.Sigmoid:
            return sigmoid(x)
        elif self.activation_hid == Activation.ReLu:
            return relu(x)
        elif self.activation_hid == Activation.Tanh:
            return tanh(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_hid}')

    # @override
    def df_hid(self, x):
        """
        Derivation of activation function on hidden layer.
        """
        if self.activation_hid == Activation.Sigmoid:
            return df_sigmoid(x)
        elif self.activation_hid == Activation.ReLu:
            return df_relu(x)
        elif self.activation_hid == Activation.Tanh:
            return df_tanh(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_hid}')

    # @override
    def f_out(self, x):
        """
        Activation function for output layer.
        """
        if self.activation_out == Activation.Sigmoid:
            return sigmoid(x)
        elif self.activation_out == Activation.ReLu:
            return relu(x)
        elif self.activation_hid == Activation.Tanh:
            return tanh(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_out}')

    # @override
    def df_out(self, x):
        """
        Derivation of activation function on output layer.
        """
        if self.activation_out == Activation.Sigmoid:
            return df_sigmoid(x)
        elif self.activation_out == Activation.ReLu:
            return df_relu(x)
        elif self.activation_hid == Activation.Tanh:
            return df_tanh(x)
        else:
            raise Exception(f'Not supported activation function: {self.activation_out}')

    def error(self, targets, outputs):
        """
        Cost / loss / error function
        """
        return np.sum((targets - outputs) ** 2, axis=0)

    def train(self,
              train_X, train_y,
              val_X, val_y,
              optimizer,
              alpha,
              epochs,
              momentum=False,
              epsilon=None,
              batch_size=None,
              beta1=None,
              beta2=None,
              display_error=True):
        """
        Args:
            train_X:
            train_y:
            val_X:
            val_y:
            optimizer:
            alpha:
            epochs:
            batch_size:
            momentum:
            epsilon:
            beta1:
            beta2:
            display_error:
        Returns:
        """
        test_CEs = []
        test_REs = []
        val_CEs = []
        val_REs = []
        amount = train_X.shape[0]

        v_hid = np.zeros(self.W_hid.shape)
        v_out = np.zeros(self.W_out.shape)

        # Mini-batch gd
        if optimizer == Optimizer.MiniBatch:
            for ep in range(epochs):
                len_batches, X, y = create_batches(train_X, train_y, batch_size)
                CE = 0
                RE = 0
                for batch in range(len_batches):
                    x = X[batch]
                    d = onehot_encode(y[batch])

                    a, h, b, yhat = self.forward(x)
                    dW_hid, dW_out = self.backpropagation(x, a, h, b, yhat, d)
                    # using momentum
                    if momentum:
                        self.W_hid = self.W_hid + alpha * dW_hid
                        self.W_out = self.W_out + alpha * dW_out

                        self.W_hid = self.W_hid + (alpha * v_hid)
                        self.W_out = self.W_out + (alpha * v_out)
                    # not using momentum
                    else:
                        self.W_hid = self.W_hid + alpha * dW_hid
                        self.W_out = self.W_out + alpha * dW_out

                    CE += (np.not_equal(y[batch], onehot_decode(yhat))).sum()
                    RE += np.sum(self.error(d, yhat), axis=0)

                CE /= amount
                RE /= amount
                test_CEs.append(CE)
                test_REs.append(RE)
                val_CE, val_RE = self.test(val_X, val_y)
                val_CEs.append(val_CE)
                val_REs.append(val_RE)
                if (ep + 1) % 5 == 0 and display_error:
                    print_errors('Train', CE, RE, ep, epochs)
                    print_errors('Valid', val_CE, val_RE, ep, epochs)
        # Batch Gradient descent
        elif optimizer == Optimizer.Batch:
            for ep in range(epochs):
                d = onehot_encode(train_y)
                a, h, b, yhat = self.forward(train_X)
                dW_hid, dW_out = self.backpropagation(train_X, a, h, b, yhat, d)

                if momentum:
                    self.W_hid += alpha * dW_hid
                    self.W_out += alpha * dW_out

                    self.W_hid = self.W_hid + (alpha * v_hid)
                    self.W_out = self.W_out + (alpha * v_out)
                else:
                    self.W_hid += alpha * dW_hid
                    self.W_out += alpha * dW_out

                CE = (np.not_equal(train_y, onehot_decode(yhat))).sum()
                RE = np.sum(self.error(d, yhat), axis=0)

                CE /= amount
                RE /= amount

                test_CEs.append(CE)
                test_REs.append(RE)

                val_CE, val_RE = self.test(val_X, val_y)

                val_CEs.append(val_CE)
                val_REs.append(val_RE)
                if (ep + 1) % 5 == 0 and display_error:
                    print_errors('Train', CE, RE, ep, epochs)
                    print_errors('Valid', val_CE, val_RE, ep, epochs)
        # (Page 2)Algorithm1: from papier [Link: https://arxiv.org/pdf/1412.6980.pdf]
        elif optimizer == Optimizer.Adam:
            v_dw = np.zeros(self.W_hid.shape)
            v_do = np.zeros(self.W_out.shape)
            s_dw = np.zeros(self.W_hid.shape)
            s_do = np.zeros(self.W_out.shape)
            for ep in range(epochs):
                len_batches, X, y = create_batches(train_X, train_y, batch_size)
                CE = 0
                RE = 0
                for batch in range(len_batches):
                    t = batch + 1
                    x = X[batch]
                    d = onehot_encode(y[batch])
                    a, h, b, yhat = self.forward(x)
                    dW_hid, dW_out = self.backpropagation(x, a, h, b, yhat, d)

                    # Momentum
                    v_dw = beta1 * v_dw + (1 - beta1) * dW_hid
                    v_do = beta1 * v_do + (1 - beta1) * dW_out
                    # RMSprop
                    s_dw = beta2 * s_dw + (1 - beta2) * (dW_hid ** 2)
                    s_do = beta2 * s_do + (1 - beta2) * (dW_out ** 2)
                    # Correct values
                    v_dw_correct = (v_dw / (1 - (beta1 ** t)))
                    v_do_correct = (v_do / (1 - (beta1 ** t)))
                    s_dw_correct = (s_dw / (1 - (beta2 ** t)))
                    s_do_correct = (s_do / (1 - (beta2 ** t)))
                    # update
                    self.W_hid += alpha * (v_dw_correct / (np.sqrt(s_dw_correct) + epsilon))
                    self.W_out += alpha * (v_do_correct / (np.sqrt(s_do_correct) + epsilon))

                    CE += (np.not_equal(y[batch], onehot_decode(yhat))).sum()
                    RE += np.sum(self.error(d, yhat), axis=0)

                CE /= amount
                RE /= amount
                test_CEs.append(CE)
                test_REs.append(RE)
                val_CE, val_RE = self.test(val_X, val_y)
                val_CEs.append(val_CE)
                val_REs.append(val_RE)
                if (ep + 1) % 5 == 0 and display_error:
                    print_errors('Train', CE, RE, ep, epochs)
                    print_errors('Valid', val_CE, val_RE, ep, epochs)

        return test_CEs, test_REs, val_CEs, val_REs

    def calculate_momentum_velocity(self, beta1, dW_hid, dW_out, v_hid, v_out):
        """
        Calculate velocity for momentum
        Args:
            beta1: hyper-param
            dW_hid: weight from backprop hidden layer
            dW_out: weight from backprop output layer
            v_hid: velocity hidden layer
            v_out: velocity output layer

        Returns: v_hid, v_out
        """
        v_hid = beta1 * v_hid + (1 - beta1) * dW_hid
        v_out = beta1 * v_out + (1 - beta1) * dW_out
        return v_hid, v_out

    def test(self, inputs, labels, confusion_matrix=False):
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
