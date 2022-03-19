import numpy as np


# load train data
train_data = np.loadtxt('2d.trn.dat', dtype=str)
test_data = np.loadtxt('2d.tst.dat')

features_train = train_data[0]