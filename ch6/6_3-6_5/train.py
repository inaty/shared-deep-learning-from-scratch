from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split

from network import layers as L
from network import functions as F
from network.module import SequenceNet

np.random.seed(7)

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
    num_epoch = 100
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

    # training
    loss_train_history = []
    loss_test_history = []
    acc_train_history = []
    acc_test_history = []
    with tqdm.tqdm(range(num_epoch)) as progress_bar:
        for epoch in progress_bar:
            for _ in range(X_train.shape[0] // batch_size):
                idxes_sample = np.random.choice(X_train.shape[0], batch_size)
                x_batch = X_train[idxes_sample]
                t_batch = Y_train[idxes_sample]

                for layer in net.layers:
                    if layer.__class__.__name__ == "Dropout":
                        layer.is_training = True
                net.gradient(x_batch, t_batch)

                for layer in net.layers:
                    if hasattr(layer, "params"):
                        for key in layer.params:
                            # optimizer with simple SGD
                            layer.params[key] -= lr * layer.grads[key] / batch_size

            # testing
            for layer in net.layers:
                if layer.__class__.__name__ == "Dropout":
                    layer.is_training = False

            loss_train = net.calculate_loss(X_train, Y_train)
            acc_train = net.calculate_accuracy(X_train, Y_train)
            loss_test = net.calculate_loss(X_test, Y_test)
            acc_test = net.calculate_accuracy(X_test, Y_test)

            loss_train_history.append(loss_train)
            acc_train_history.append(acc_train)
            loss_test_history.append(loss_test)
            acc_test_history.append(acc_test)

            # setting for tqdm display parameters
            progress_bar.set_description_str(f"epoch {epoch:5d}")
            progress_bar.set_postfix(
                OrderedDict(
                    loss_train=loss_train,
                    acc_train=acc_train,
                    loss_test=loss_test,
                    acc_test=acc_test,
                )
            )

    print(f"sum_weights_abs: {net.sum_weights_abs:.3f}")
    print(f"sum_weights_pow2: {net.sum_weights_pow2:.3f}")

    # visualization
    plt.figure()
    plt.plot(loss_train_history, marker="o", label="train")
    plt.plot(loss_test_history, marker="o", label="test")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0, np.max(np.concatenate((loss_test_history, loss_test_history))))
    plt.grid()
    plt.tight_layout()
    plt.savefig("loss.png", dpi=200)
    # plt.show()

    plt.figure()
    plt.plot(acc_train_history, marker="o", label="train")
    plt.plot(acc_test_history, marker="o", label="test")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.grid()
    plt.tight_layout()
    plt.savefig("accuracy.png", dpi=200)
    # plt.show()
