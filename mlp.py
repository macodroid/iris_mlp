from utils import *
from Enums import *


class MLP:
    """
    Multi-Layer Perceptron (abstract base class)
    """

    def __init__(self, dim_in, dim_hid, dim_out, init_type):
        """
        Initialize model, set initial weights
        """
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        # generate random weights from standard/normal distribution
        if init_type == InitWeights.Normal:
            self.W_hid = np.random.randn(dim_hid, dim_in + 1)
            self.W_out = np.random.randn(dim_out, dim_hid + 1)
        # generate random weights from uniform distribution
        elif init_type == InitWeights.Uniform:
            self.W_hid = np.random.rand(dim_hid, dim_in + 1)
            self.W_out = np.random.rand(dim_out, dim_hid + 1)

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
        a = add_bias(x) @ self.W_hid.T
        h = self.f_hid(a)
        b = add_bias(h) @ self.W_out.T
        yhat = self.f_out(b)

        return a, h, b, yhat

    def backpropagation(self, x, a, h, b, y, d):
        """
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        h: activation of hidden layer (without bias, size=dim_hid)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        """
        output_error = (d - y) * self.df_out(b)
        hidden_error = output_error @ self.W_out
        error_h = hidden_error[:, :-1] * self.df_hid(a)
        dW_out = output_error.T @ add_bias(h)
        dW_hid = error_h.T @ add_bias(x)
        return dW_hid, dW_out
