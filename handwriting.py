import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
import sigmoid as sig
import softmax as sm


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    # x_train: train_img; t_train: train_label; x_test: test_img; t_test: test_label
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return (x_train, t_train), (x_test, t_test)


def init_network():
    with open("dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sig.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sig.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sm.softmax(a3)

    return y


def accuracy_cnt():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    # print(predict(network, x[1]), np.argmax(predict(network, x[1])))
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 获得概率最高的元素的索引5
        if p == t[i]:
            accuracy_cnt += 1

    print('Accuracy:' + str(float(accuracy_cnt/len(x))))
    return accuracy_cnt


def accuracy_cnt_batch():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    batch_size = 100
    for i in range(0, len(x), batch_size):
        y = predict(network, x[i:i+batch_size])
        p = np.argmax(y, axis=1)
        # print(y)
        print(p)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print('Accuracy:' + str(float(accuracy_cnt / len(x))))
    return accuracy_cnt


# print(accuracy_cnt_batch())