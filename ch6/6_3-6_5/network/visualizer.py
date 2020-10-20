import numpy as np
import matplotlib.pyplot as plt

def visualize_histories(histories, loss_filename="loss.png", acc_filename="acc.png"):

    # visualization
    plt.figure()
    plt.plot(histories["loss_train"], marker="o", label="train")
    plt.plot(histories["loss_test"], marker="o", label="test")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0, np.max(np.concatenate((histories["loss_test"], histories["loss_test"]))))
    plt.grid()
    plt.tight_layout()
    plt.savefig(loss_filename, dpi=200)
    plt.show()

    plt.figure()
    plt.plot(histories["acc_train"], marker="o", label="train")
    plt.plot(histories["acc_test"], marker="o", label="test")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(acc_filename, dpi=200)
    plt.show()