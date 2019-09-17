import numpy as np
import handwriting as hw
import cee


def mini_batch():
    (x_train, t_train), (x_test, t_test) = hw.get_data()
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(x_batch)
    print(t_batch)
    network = hw.init_network()
    y = hw.predict(network, x_batch)
    # print(y)
    # print(t_batch)
    cross_error = cee.cross_entropy_error_batch_not_onehot(y, t_batch)
    return cross_error


print(mini_batch())
