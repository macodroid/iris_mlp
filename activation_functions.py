import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function
    Args:
        x: matrix or number

    Returns: numpy array x.shape

    """
    return 1 / (1 + np.exp(-x))


def df_sigmoid(x):
    """
        Derivation of sigmoid activation function
        Args:
            x: matrix or number

        Returns: numpy array x.shape or number

        """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """
    ReLu activation function
    Args:
        x: matrix or number

    Returns: numpy array x.shape or number

    """
    return np.maximum(0, x)


def df_relu(x):
    """
    Derivation of ReLu activation function
    Args:
        x: matrix or number

    Returns: numpy array x.shape or number

    """
    return np.array(x > 0, dtype=float)


def tanh(x):
    """
    Tanh activation function
    Args:
        x: matrix or number

    Returns: numpy array x.shape or number

    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def df_tanh(x):
    """
    Derivation of Tanh activation function
    Args:
        x: matrix or number

    Returns: numpy array x.shape or number

    """
    return 1 - (tanh(x) ** 2)
