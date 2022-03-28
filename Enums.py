import enum


class Optimizer(enum.Enum):
    MiniBatch = 'mini-batch'
    Batch = 'batch'
    Adam = 'adam'


class Activation(enum.Enum):
    Sigmoid = 'sigmoid'
    ReLu = 'relu'
    Tanh = 'tanh'
    Softmax = 'softmax'


class InitWeights(enum.Enum):
    Normal = 'normal'
    Uniform = 'uniform'
