import numpy as np
import matplotlib.pyplot as plt


def ReLu(x):
    return np.maximum(0, x)


def graph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = ReLu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.0)
    plt.show()


graph()