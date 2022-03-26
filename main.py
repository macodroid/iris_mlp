from classifier import *

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

train_features -= np.mean(train_features, axis=0)
train_features /= np.std(train_features, axis=0)

train_features, train_labels, val_features, val_labels = split_train_validation(train_features, train_labels)

model = MLPClassifier(dim_in=dim, dim_hid=20, n_classes=np.max(train_labels) + 1)

trainCE, trainRE, valCE, valRE = model.train(train_features, train_labels, val_features, val_labels, batch_size=256)

print("a")
