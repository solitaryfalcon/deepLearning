import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def sin():
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tanh(x)
    plt.plot(x, y1, label = 'sin')
    plt.plot(x, y2, linestyle = '--',label = 'cos')
    plt.plot(x, y3, label = 'tanh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('sin & cos')
    plt.legend()
    # plt.savefig('/Users/yangbowen/desktop/plt/sinCos')
    plt.show()

def readImg():
    img = imread('/Users/yangbowen/desktop/imread.jpg')
    plt.imshow(img)
    plt.show()


# readImg()
sin()