import numpy as np


class SVM:
    def __init__(self, 
                 n_class: int,
                 lr: float,
                 epochs: int,
                 reg_const: float,  # regularization constant
                 batch_num: int):
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_num = batch_num

    def cal_gradient(self,
                     X_train: np.ndarray,       # (N, D) N samples in one mini-batches, dimension D
                     y_train: np.ndarray):      # (N, C) class number
        N, D = X_train.shape
        for i in range(N):
            for c in range(self.n_class):
                if c != y_train[i] and self.w[y_train[i]].reshape((1, -1)) @ X_train[i].reshape((D, 1)) - self.w[c].reshape((1, -1)) @ X_train[i].reshape((D, 1)) < 1:
                    self.w[c] -= self.lr * X_train[i]
                    self.w[y_train[i]] += self.lr * X_train[i]
            self.w -= self.lr * self.reg_const * self.w / N

    def train(self,
              X_train: np.ndarray,       # (N, D) N samples in one mini-batches, dimension D
              y_train: np.ndarray):      # (N, )  vector of class
        N, D = X_train.shape
        X_train = X_train.astype(float)
        X_train_batches = np.split(X_train, self.batch_num)
        y_train_batches = np.split(y_train, self.batch_num)

        self.w = np.zeros((self.n_class, D))
        for epoch in range(1+self.epochs):
            # lr decays every 100 epochs
            if epoch % 100 == 0:
                self.lr /= 5
            i = np.random.randint(len(X_train_batches))
            self.cal_gradient(X_train_batches[i], y_train_batches[i])
