import numpy as np

class FDM:
    def __init__(self, __ex):
        self.__ex = __ex

    def get_ex(self):
        return self.__ex

    def f(self):
        if self.__ex == 1:
            a = np.array([
                [2, 0, 0, 0],
                [2, 3, 0, 0],
                [-1, 2, 3, 0],
                [4, 3, 2, 1]
            ], dtype=float)
            b = np.array([2, 5, 4, 10], dtype=float)
            exact = np.array([1, 1, 1, 1], dtype=float)
        elif self.__ex == 2:
            a = np.array([
                [1, 2, 3, 4],
                [0, 2, 3, 5],
                [0, 0, 2, -4],
                [0, 0, 0, 3]
            ], dtype=float)
            b = np.array([10,10,-2,3], dtype=float)
            exact = np.array([1, 1, 1, 1], dtype=float)
        elif self.__ex == 3:
            a = np.array([
                [62, 24, 1, 8, 15],
                [23, 50, 7, 14, 16],
                [4, 6, 58, 20, 22],
                [10, 12, 19, 66, 3],
                [11, 18, 25, 2, 54]
            ], dtype=float)
            b = np.array([110,110,110,110,110], dtype=float)
            exact = np.array([1, 1, 1, 1, 1], dtype=float)
        elif self.__ex == 4:
            a = np.array([
                [5, 7, 6, 5],
                [7, 10, 8, 7],
                [6, 8, 10, 9],
                [5, 7, 9, 10]
            ], dtype=float)
            b = np.array([23, 32, 33, 31], dtype=float)
            exact = np.array([1, 1, 1, 1], dtype=float)
        elif self.__ex == 5:
            e = np.finfo(float).eps
            a = np.array([
                [e, 1, 2],
                [1, 2, 1],
                [2, 2, 3]
            ], dtype=float)
            b = np.array([e + 3, 4, 7], dtype=float)
            exact = np.array([1, 1, 1], dtype=float)
        else:
            e = np.finfo(float).eps / 4
            a = np.array([
                [e, 1, 2],
                [1, 2, 1],
                [2, 2, 3]
            ], dtype= float)
            b = np.array([e + 3, 4, 7], dtype=float)
            exact = np.array([1, 1, 1], dtype=float)
        return a, b, exact
