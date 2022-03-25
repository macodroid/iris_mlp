from utils import *


class MLP:
    """
    Multi-Layer Perceptron (abstract base class)
    """

    def __init__(self, dim_in, dim_hid, dim_out):
        """
        Initialize model, set initial weights
        """
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.W_hid = np.random.randn(dim_hid, dim_in + 1)
        self.W_out = np.random.randn(dim_out, dim_hid + 1)

    # Activation functions & derivations
    # (not implemented, to be overriden in derived classes)
    def f_hid(self, x):
        raise NotImplementedError

    def df_hid(self, x):
        raise NotImplementedError

    def f_out(self, x):
        raise NotImplementedError

    def df_out(self, x):
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass - compute output of network
        x: single input vector (without bias, size=dim_in)
        """
        x = add_bias(x)
        a = x @ self.W_hid.T
        h = self.f_hid(a)
        h = add_bias(h)
        b = h @ self.W_out.T
        y = self.f_out(b)

        return h, y

    def backpropagation(self, x, h, y, d, batch_size):
        """
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        h: activation of hidden layer (without bias, size=dim_hid)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        https://stackoverflow.com/questions/50105249/implementing-back-propagation-using-numpy-and-python-for-cleveland-dataset
        """
        # backpropagation
        output_layer_error = y - d
        output_layer_delta = output_layer_error * y * (1 - y)

        hidden_layer_error = np.dot(output_layer_delta, self.W_out)
        hidden_layer_delta = hidden_layer_error * h * (1 - h)
        h = add_bias(h)
        # dW_hid = np.dot(add_bias(h), output_layer_delta) / batch_size
        dW_hid = (output_layer_delta.T @ h) / batch_size
        x = add_bias(x)
        # dW_out = np.dot(add_bias(x), hidden_layer_delta) / batch_size
        dW_out = (hidden_layer_delta.T @ x) / batch_size
        return dW_hid.T, dW_out.T
