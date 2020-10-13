import numpy as np

from . import functions as F


class Linear:
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.params = {"w": None, "b": None}
        self.grads = {"w": None, "b": None}

    def forward(self, x):
        self.x = x
        return self.x @ self.params["w"] + self.params["b"]

    def backward(self, dy, l1=0.0, l2=0.0):
        self.grads["w"] = (
            self.x.T @ dy + l1 * np.sign(self.params["w"]) + l2 * self.params["w"]
        )
        self.grads["b"] = np.sum(dy, axis=0, keepdims=True)
        return dy @ self.params["w"].T


class ReLU:
    def __init__(self):
        self.idx_greater_than_zero = None

    def forward(self, x):
        self.idx_greater_than_zero = x > 0.0
        x[~self.idx_greater_than_zero] = 0.0
        return x

    def backward(self, dy, *args):
        return dy * self.idx_greater_than_zero


class BatchNormalization:
    def __init__(self, n_out):
        self.eps = 1e-8
        self.n_out = n_out
        self.params = {"gamma": None, "beta": None}
        self.grads = {"gamma": None, "beta": None}

    def forward(self, x):
        self.mu = np.mean(x, axis=0)
        self.var = np.var(x, axis=0, ddof=0)
        self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.params["gamma"] * self.x_hat + self.params["beta"]

    def backward(self, dy, *args):
        self.grads["gamma"] = np.sum(dy * self.x_hat, axis=0)
        self.grads["beta"] = np.sum(dy, axis=0)
        coefficient = self.params["gamma"] / np.sqrt(self.var + self.eps)
        x_hat_centered = self.x_hat - np.mean(self.x_hat, axis=0)
        grad_block = (
            dy
            - (self.grads["beta"] + self.grads["gamma"] * x_hat_centered) / dy.shape[0]
        )
        return coefficient * grad_block


class Dropout:
    def __init__(self, drop_ratio=0.5):
        assert drop_ratio >= 0.0 and drop_ratio <= 1.0
        self.is_training = True
        self.drop_ratio = drop_ratio
        self.idx_drop = None

    def forward(self, x):
        self.mask_to_drop = np.random.binomial(
            1, 1.0 - self.drop_ratio, size=x.shape[0]
        )[:, np.newaxis]
        if self.is_training:
            return x * self.mask_to_drop
        else:
            return x * (1.0 - self.drop_ratio)

    def backward(self, dy, *args):
        return dy * self.mask_to_drop


class SoftmaxCrossEntropy:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = F.softmax(x)
        self.t = t
        return F.cross_entropy_error(self.y, self.t)

    def backward(self, dy=1.0):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size
