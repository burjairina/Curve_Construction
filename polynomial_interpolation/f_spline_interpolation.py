import numpy as np

class FSI:
    def __init__(self, ex):
        self.__ex = ex

    def f(self, x):
        if self.__ex == 1:
            if x < (1 / 3):
                y = 7 / 12 + x * (x - 1 /3)
            else:
                y = 3 * abs(x - 1 / 2) + abs(x - 1 / 4)
        else:
            if x < 1:
                y = 5 * abs(2 * x - 1) + 6 * abs(4 * x - 3)
            else:
                y = x ** 2 + 10
        return y

    def division(self):
        if self.__ex == 1:
            d = np.array([0, 1/3, 4/9, 2/3, 1])
        else:
            d = np.array([0, 1/2, 2/3, 1, 2])
        return d