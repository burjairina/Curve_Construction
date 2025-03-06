import numpy as np

class FEE:
    def __init__(self, ex, lambdaa):
        self.__ex = ex
        self.__lambdaa = lambdaa

    def f(self, x, y):
        if self.__ex == 1:
            rez = -self.__lambdaa * y
        else:
            rez = 2 - np.exp(-4 * x) - 2 * y
        return rez

    def exact(self, x):
        if self.__ex == 1:
            rez = np.exp(-self.__lambdaa * x)
        else:
            rez = 1 + 0.5 * np.exp(-4 * x) - 0.5 * np.exp(-2 * x)
        return rez

    def lim_int(self):
        if self.__ex == 1:
            a = 0
            b = 2
        else:
            a = 0
            b = 3
        return a, b

    def initial_val(self):
        return 1