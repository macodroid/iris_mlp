import numpy as np
from utils import *


class MLP:
    '''
    Multi-Layer Perceptron (abstract base class)
    '''
    def __init__(self, dim_in, dim_hid, dim_out):
        '''
        Initialize model, set initial weights
        '''
        self.dim_in  = dim_in
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
        '''
        Forward pass - compute output of network
        x: single input vector (without bias, size=dim_in)
        '''
        a = self.W_hid @ add_bias(x)
        h = self.f_hid(a)
        b = self.W_out @ add_bias(h)
        y = self.f_out(b)

        return a, h, b, y

    def backpropagation(self, x, a, h, b, y, d):
        '''
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        a: net vector on hidden layer (size=dim_hid)
        h: activation of hidden layer (without bias, size=dim_hid)
        b: net vector on output layer (size=dim_out)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        '''
        