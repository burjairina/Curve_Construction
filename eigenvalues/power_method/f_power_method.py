import numpy as np

class FPM:
    def __init__(self, ex):
        self.__ex = ex

    def f(self):
        if self.__ex == 1:
            a = np.array([[-4, 14, 0],
                          [-5, 13, 0],
                          [-1, 0, 2]], dtype=float)
            valmax = 6
            x0 = np.array([1, 1, 1], dtype=float)
        elif self.__ex == 2:
            a = np.array([[15, -2, 2],
                          [1, 10, -3],
                          [-2, 1, 0]], dtype=float)
            valmax = 14.103
            x0 = np.array([1, 1, 1], dtype=float)
        else:
            a = np.array([[1, 3, 4],
                          [3, 1, 2],
                          [4, 2, 1]], dtype=float)
            valmax = 7.047
            x0 = np.array([1, 1, 1], dtype=float)
        return a, valmax, x0