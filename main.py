import enum

from classifier import *


class Optimizer(enum.Enum):
    MiniBatch = 'mini-batch'
    Momentum = 'momentum'
    SGD = 'sgd'
    L2MiniBatch = 'L2_mini-batch'


class Activation(enum.Enum):
    Sigmoid = 'sigmoid'
    ReLu = 'relu'


# load train data
train_data = np.loadtxt('2d.trn.dat', dtype=str)
test_data = np.loadtxt('2d.tst.dat', dtype=str)

train_data = train_data[1:]
test_data = test_data[1:]

train_features = train_data[:, :-1].astype(float)
train_labels = train_data[:, -1:].flatten()  # ['A','B','C' .... 'A']
test_features = test_data[:, :-1].astype(float)
test_labels = test_data[:, -1:].flatten()  # ['A','B','C' .... 'A']

(count, dim) = train_features.shape

train_labels = ordinal_encoding(train_labels)
test_labels = ordinal_encoding(test_labels)

mean = np.mean(train_features, axis=0)
std = np.std(train_features, axis=0)

train_features -= mean
train_features /= std

test_features -= mean
test_features /= std

train_features, train_labels, val_features, val_labels = split_train_validation(train_features, train_labels)

"""
Hidden layer neurons: 20
Activation function hidden layer: sigmoid
Activation function output layer: sigmoid
Optimizer alg: mini batch gd and momentum
batch size: 128
"""
model1 = MLPClassifier(dim_in=dim,
                       dim_hid=20,
                       n_classes=np.max(train_labels) + 1,
                       activation_hid=Activation.Sigmoid.value,
                       activation_out=Activation.Sigmoid.value,
                       inti_weights_type='uniform',
                       lambd=0.08)

trainCEs, trainREs, valCEs, valREs = model1.train(train_features, train_labels,
                                                  val_features, val_labels,
                                                  optimizer=Optimizer.L2MiniBatch.value,
                                                  batch_size=128)

testCE, testRE, confusion_matrix = model1.test(test_features, test_labels, confusion_matrix=True)
print('Final testing error model1: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

# plot test and validation error
plot_train_val_error(trainCEs, trainREs, valCEs, valREs)
plot_confusion_matrix(confusion_matrix)
