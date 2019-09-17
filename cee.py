# cross entropy error (交叉熵误差)
import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_batch_not_onehot(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# t = [2, 4, 1] #not one hot 0.1053604045467214
# t = [[0,0,1,0,0],[0,0,0,0,1],[0,1,0,0,0]] #onehot 0.1053604045467214
# y = [[0.1, 0.05, 0.9, 0.0, 0.05], [0.1, 0.05, 0.06, 0.0, 0.9],[0.09, 0.9, 0.1, 0.0, 0.05]]
# print(cross_entropy_error_batch(np.array(y), np.array(t)))