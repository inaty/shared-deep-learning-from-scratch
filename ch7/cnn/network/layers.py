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

class Convolution:
    def __init__(self, n_in, n_out, filter_size=3, stride=1, pad=0):
        self.n_in = n_in
        self.n_out = n_out
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.params = {"w": None, "b": None}
        self.grads = {"w": None, "b": None}

    def forward(self, x):
        FN, C, FH, FW = self.params["w"].shape
        N, C, H, W = x.shape
        out_h = conv_output_size(input_size=H, filter_size=FH, stride=self.stride, pad=self.pad)
        out_w = conv_output_size(input_size=W, filter_size=FW, stride=self.stride, pad=self.pad)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.params["w"].reshape(FN, -1).T
        out = np.dot(col, col_w) + self.params["b"]
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_w = col_w
        return out

    def backward(self, dy, *args):
        FN, C, FH, FW = self.params["w"].shape
        dy = dy.transpose(0,2,3,1).reshape(-1, FN)

        self.grads["b"] = np.sum(dy, axis=0)
        self.grads["w"] = np.dot(self.col.T, dy).transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dy, self.col_w.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = conv_output_size(input_size=H, filter_size=self.pool_h, stride=self.stride)
        out_w = conv_output_size(input_size=W, filter_size=self.pool_w, stride=self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dy, *args):
        dy = dy.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dy.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (pool_size,))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

class Flatten:
    def __init__(self):
        self.input_size = None
        self.output_size = None

    def forward(self, x):
        self.input_size = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy, *args):
        return dy.reshape(*self.input_size)

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return int((input_size + 2 * pad - filter_size) / stride + 1)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
