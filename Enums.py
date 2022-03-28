import enum


class Optimizer(enum.Enum):
    MiniBatch = 'mini-batch'
    Momentum = 'momentum'
    SGD = 'sgd'
    Adam = 'adam'


class Activation(enum.Enum):
    Sigmoid = 'sigmoid'
    ReLu = 'relu'
    Tanh = 'tanh'
    Softmax = 'softmax'


class InitWeights(enum.Enum):
    Normal = 'normal'
    Uniform = 'uniform'
