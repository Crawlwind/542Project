import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.std = None
        self.mean = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-z/10))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        N, D = X_train.shape
        self.w = np.zeros((1, D))
        self.mean = np.mean(X_train, axis=0, keepdims=True)
        self.std = np.std(X_train, axis=0, keepdims=True)
        assert(self.mean.shape[0] == 1)
        assert(self.mean.shape[1] == D)
        X_train = X_train.astype(float)
        X_train = X_train - self.mean
        X_train /= (self.std + 10e-5)
        for epoch in range(1, self.epochs + 1):

            self.w += self.lr*((self.sigmoid(-y_train.reshape((N, 1)).T *
                               (self.w @ X_train.T))) @ (X_train*y_train.reshape((N, 1))))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = X_test.astype(float)
        X_test -= self.mean
        X_test /= (self.std+10e-5)
        y_predicted = self.w @ X_test.T
        y_predicted[y_predicted > self.threshold] = 1
        y_predicted[y_predicted <= self.threshold] = 0
        return y_predicted
