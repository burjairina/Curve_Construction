import numpy as np

class FIM:
    def __init__(self, __ex):
        self.__ex = __ex

    def f(self):
        if self.__ex == 1:
            a = np.array([[62, 24, 1, 8, 15],
                          [23, 50, 7, 14, 16],
                          [4, 6, 58, 20, 22],
                          [10, 12, 19, 66, 3],
                          [11, 18, 25, 2, 54]])
            b = np.array([110, 110, 110, 110, 110])
            exact = np.array([1, 1, 1, 1, 1])
            xo = np.array([0, 0, 0, 0, 0])
        else:
            a = np.array([[5, 7, 6, 5],
                          [7, 10, 8, 7],
                          [6, 8, 10, 9],
                          [5, 7, 9, 10]])
            b = np.array([23, 32, 33, 31])
            exact = np.array([1, 1, 1, 1])
            xo = np.array([0, 0, 0, 0])
        return a, b, exact, xo