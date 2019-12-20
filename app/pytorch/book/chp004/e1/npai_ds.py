#
import numpy as np

class NpaiDs(object):
    @staticmethod
    def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
        """ Split the data into train and test sets """
        if shuffle:
            X, y = NpaiDs.shuffle_data(X, y, seed)
        # Split the training data from test data in the ratio specified in
        # test_size
        split_i = len(y) - int(len(y) // (1 / test_size))
        X_train, X_test = X[:split_i], X[split_i:]
        y_train, y_test = y[:split_i], y[split_i:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def shuffle_data(X, y, seed=None):
        """ Random shuffle of the samples in X and y """
        if seed:
            np.random.seed(seed)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    @staticmethod
    def make_diagonal(x):
        """ Converts a vector into an diagonal matrix """
        m = np.zeros((len(x), len(x)))
        for i in range(len(m[0])):
            m[i, i] = x[i]
        return m

    @staticmethod
    def normalize(X, axis=-1, order=2):
        """ Normalize the dataset X """
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1
        return X / np.expand_dims(l2, axis)

    @staticmethod
    def standardize(X):
        """ Standardize the dataset X """
        X_std = X
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        for col in range(np.shape(X)[1]):
            if std[col]:
                X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
        # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        return X_std

    @staticmethod
    def batch_iterator(X, y=None, batch_size=64):
        """ Simple batch generator """
        n_samples = X.shape[0]
        for i in np.arange(0, n_samples, batch_size):
            begin, end = i, min(i+batch_size, n_samples)
            if y is not None:
                yield X[begin:end], y[begin:end]
            else:
                yield X[begin:end]