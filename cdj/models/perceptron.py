import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # y_train: a numpy array of shape (N,) containing training labels
        # w shape(n_class, D)
        self.w = np.zeros(shape=(self.n_class, X_train.shape[1]))
        for epoch in range(self.epochs):
            y_hat = np.argmax(X_train @ (self.w).T, axis=1)
            if epoch % 100 == 0:
                self.lr /= 2
            for idx in range(y_hat.size):
                pred = y_hat[idx]
                true = y_train[idx]
                if pred != true:
                    self.w[true] += self.lr * X_train[idx]
                    self.w[pred] -= self.lr * X_train[idx]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.argmax(X_test @ (self.w).T, axis=1)
