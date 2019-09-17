import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def step_function_graph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def graph():        
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = step_function(x)
    plt.plot(x, y1, label='sigmoid')
    plt.plot(x, y2, linestyle='--', label='step')
    plt.legend()
    plt.show()


# step_function_graph()
# sigmoid_graph()
# graph()