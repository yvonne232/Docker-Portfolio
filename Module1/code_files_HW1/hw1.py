# you can import hw1
# vissualization --- look at the data, learning test, x data, for each x data point(2-d), corresponding to [0,1] in y, scatter the poingt 
# color the two type y data. compare it with the real y. 

# in c, differnece in neural network, non-trival, you should experiment.
# with or without basis.

from abc import abstractmethod, abstractstaticmethod
from typing import Tuple
import numpy as np

from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def binary_accuracy(ypred, y):
    return sum(ypred.round() == y)/float(y.shape[0])


def sklearn_logreg(X_train, y_train, X_test, y_test):
    sk_logr = LogisticRegression(fit_intercept=False, penalty='none')
    sk_logr.fit(X_train, y_train)
    sk_logr.fit(X_train, y_train)
    print(type(sk_logr.predict(X_test)))
    return binary_accuracy(sk_logr.predict(X_test), y_test)


class HW1Data():
    @abstractmethod
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def data_split(self, test_size=0.33) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class SkLearnGenerator(HW1Data):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    @abstractstaticmethod
    def _generator(n_samples) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def data(self):
        return type(self)._generator(self.n_samples)

    def data_split(self, test_size=0.33):
        X, y = self.data()
        return train_test_split(X, y, test_size=test_size)


class Make_classification(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_classification(n_samples, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)

class Make_blobs(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_classification(n_samples, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)



class Make_moons(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_moons(n_samples, noise=0.05)


class Make_circles(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_circles(n_samples, factor=0.5, noise=0.05)
