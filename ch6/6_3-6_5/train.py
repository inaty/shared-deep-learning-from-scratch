from collections import OrderedDict

import numpy as np
np.random.seed(7)
import matplotlib.pyplot as plt
import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split

from network import trainer
from network import visualizer
from network import layers as L
from network import functions as F
from network.module import SequenceNet


if __name__ == "__main__":

    # loading training data
    load = datasets.load_digits
    X, y = load(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    feature_size = X_train.shape[1]
    num_classes = int(np.max(y)) + 1
    Y_train = np.eye(num_classes)[y_train]
    Y_test = np.eye(num_classes)[y_test]

    # training parameters
    num_epoch = 10
    batch_size = 8
    drop_ratio = 0.25
    lr = 0.02
    l1 = 0.001
    l2 = 0.001

    # network definition
    sequence = [
        L.BatchNormalization(n_out=feature_size),
        L.Linear(n_in=feature_size, n_out=100),
        L.ReLU(),
        L.Dropout(drop_ratio=drop_ratio),
        L.BatchNormalization(n_out=100),
        L.Linear(n_in=100, n_out=100),
        L.ReLU(),
        L.Dropout(drop_ratio=drop_ratio),
        L.Linear(n_in=100, n_out=num_classes),
    ]
    net = SequenceNet(layers=sequence, loss=L.SoftmaxCrossEntropy(), l1=l1, l2=l2)

    # train network
    loss_train, acc_train, loss_test, acc_test, histories = trainer.train(
        net=net, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
        num_epoch=num_epoch, batch_size=batch_size, learning_rate=lr
    )

    print(f"sum_weights_abs: {net.sum_weights_abs:.3f}")
    print(f"sum_weights_pow2: {net.sum_weights_pow2:.3f}")

    # visualize
    visualizer.visualize_histories(histories)
