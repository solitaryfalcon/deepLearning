import numpy as np


def np_test():
    x = np.array([1, 2, 3])
    y = np.array([1.0, 8.0, 27.0])
    # print(x)
    # print(type(x))
    # print(x + y)
    # print(x - y)
    # print(x * y)
    # print(y / x)
    # binaryArray = np.array([[1, 2], [4, 5]])
    # print(binaryArray)
    # print(binaryArray.shape)
    # print(binaryArray.dtype)
    binaryArray2 = np.array([[3, 0],[2, 9]])
    # binaryArray3 = np.array([10, 5])
    # print(binaryArray + binaryArray2)
    # print(binaryArray * binaryArray2)
    # print(binaryArray / 2)
    # print(binaryArray + binaryArray3)
    # print(binaryArray[1])
    # binaryArray = binaryArray.flatten()
    # print(binaryArray)
    print(binaryArray2[binaryArray2 > 2])
    print(binaryArray2 > 2)

class Man:
    def __init__(self, name):
        self.name = name
        print("initialized")


m = Man("Bower")
np_test()

x00 = np.array([[0,0,0,],[0,0,0],[0,0,0]])
x01 = np.array([[0,0,0,],[0,2,1],[0,1,1]])
x02 = np.array([[0,0,0],[0,2,2],[0,0,2]])
w00 = np.array([[-1,1,-1],[-1,1,1],[-1,-1,1]])
w01 = np.array([[-1,-1,1],[-1,1,0],[1,1,-1]])
w02 = np.array([[-1,1,-1],[0,0,0],[-1,0,0]])
print(sum(np.dot(x00, w00).flatten()))
print(sum(np.dot(x01, w01).flatten()))
print(sum(np.dot(x02, w02).flatten()))