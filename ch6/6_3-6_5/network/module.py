import numpy as np


class SequenceNet:
    def __init__(self, layers, loss, l1=0.0, l2=0.0):
        self.layers = layers
        self.loss = loss
        self.l1 = l1
        self.l2 = l2
        for layer in self.layers:
            if layer.__class__.__name__ == "Linear":
                layer.params["w"] = np.random.randn(layer.n_in, layer.n_out)
                layer.params["b"] = np.zeros([1, layer.n_out])
            if layer.__class__.__name__ == "BatchNormalization":
                layer.params["gamma"] = np.ones(layer.n_out)
                layer.params["beta"] = np.zeros(layer.n_out)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def calculate_loss(self, x, t):
        self.sum_weights_abs = 0.0
        self.sum_weights_pow2 = 0.0
        self.num_regularized_params = 0
        for layer in self.layers:
            if layer.__class__.__name__ == "Linear":
                self.sum_weights_abs += np.sum(np.abs(layer.params["w"]))
                self.sum_weights_pow2 += np.sum(layer.params["w"] ** 2.0)
                self.num_regularized_params += np.prod(layer.params["w"].shape)
        return (
            self.loss.forward(self.predict(x), t)
            + self.l1 * self.sum_weights_abs
            + self.l2 * self.sum_weights_pow2 / 2.
        )

    def calculate_accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.calculate_loss(x, t)

        # backward
        dy = self.loss.backward(1.0)
        for layer in reversed(self.layers):
            dy = layer.backward(dy, self.l1, self.l2)
