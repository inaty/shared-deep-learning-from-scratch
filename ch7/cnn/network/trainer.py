from collections import OrderedDict

import numpy as np
np.random.seed(7)
import tqdm

def train(net, X_train, Y_train, X_test, Y_test, num_epoch, batch_size, learning_rate):

    # training
    histories = {}
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
                            layer.params[key] -= learning_rate * layer.grads[key]

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

    histories["loss_train"] = loss_train_history
    histories["acc_train"] = acc_train_history
    histories["loss_test"] = loss_test_history
    histories["acc_test"] = acc_test_history

    return loss_train, acc_train, loss_test, acc_test, histories