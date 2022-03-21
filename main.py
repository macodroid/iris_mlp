import numpy as np

from utils import *

# load train data
train_data = np.loadtxt('2d.trn.dat', dtype=str)
test_data = np.loadtxt('2d.tst.dat', dtype=str)

train_data = train_data[1:]
test_data = test_data[1:]

(count, dim) = train_data.shape

train_features = train_data[:, :-1].astype(float)
train_labels = train_data[:, -1:].flatten()  # ['A','B','C' .... 'A']
test_features = test_data[:, :-1].astype(float)
test_labels = test_data[:, -1:].flatten()  # ['A','B','C' .... 'A']



train_features, train_labels, val_features, val_labels = split_train_validation(train_features, train_labels)

print("a")
