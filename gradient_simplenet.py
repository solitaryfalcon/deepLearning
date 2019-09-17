import sys, os
sys.path.append(os.pardir)
import numpy as np
# np.set_printoptions(formatter={'float':'{:.8f}'.format})
from common.functions import cross_entropy_error, softmax
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(3, 5)  # 用高斯分布进行初始化
        # self.W = np.array([[0.47355323,0.9977393,0.84668094],[0.85557411,0.03563661,0.69422093]])

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9, 0.8])
p = net.predict(x)
print(p)
t = np.array([0, 0, 1])
loss = net.loss(x, t)
print(loss)
print('--------------gradient----------')
f = lambda w: net.loss(x, t)
dw = numerical_gradient(f, net.W)
print(dw)
