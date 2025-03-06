import numpy as np


class FNE:
    def __init__(self, __ex):
        self.__ex = __ex

    def f(self, x):
        if self.__ex == 1:
            y = x ** 3 + 4 * x ** 2 - 10
        elif self.__ex == 2:
            y = np.cos(x) - x ** 2 / 2
        elif self.__ex == 3:
            y = np.exp(-x ** 2) - 2 * x
        elif self.__ex == 4:
            y = np.exp(- np.sin(x) ** 2 + x) - 2 * x
        elif self.__ex == 5:
            y = x ** 3 - 2 * x + 1
        else:
            y = x ** 4 + 2 * x ** 2 - x - 3
        return y

    def fd(self, x):
        if self.__ex == 1:
            y = 3 * x ** 2 + 8 * x
        elif self.__ex == 2:
            y = -1 * np.sin(x) - x
        elif self.__ex == 3:
            y = -2 * x * np.exp(-x ** 2) - 2
        elif self.__ex == 4:
            y = np.exp(- np.sin(x) ** 2 + x) * (1 - 2 * np.sin(x) * np.cos(x)) - 2
        elif self.__ex == 5:
            y = 3 * x ** 2 - 2
        else:
            y = 4 * x ** 3 + 4 * x - 1
        return y

    def lim_int(self):
        if self.__ex == 1:
            a = 1
            b = 2
            x0n = 2
            x0s = 1
            x1s = 2
        elif self.__ex == 2:
            a = 0.5
            b = 1.5
            x0n = 1.75
            x0s = 0.5
            x1s = 2
        elif self.__ex == 3:
            a = 0
            b = 1
            x0n = 1
            x0s = 0
            x1s = 1
        elif self.__ex == 4:
            a = 0
            b = 1.5
            x0n = 0
            x0s = 0
            x1s = 1.5
        elif self.__ex == 5:
            a = 3 / 4
            b = 2
            x0n = 1.5
            x0s = 3 / 4
            x1s = 1.5
        else:
            a = 0
            b = 2
            x0n = 1
            x0s = 1
            x1s = 2
        return a, b, x0n, x0s, x1s

    def f_title(self):
        if self.__ex == 1:
            q = '$f(x)=x^3+4 x^2-10$'
        elif self.__ex == 2:
            q = '$f(x)=cos(x)-\\frac{x^2}{2}$'
        elif self.__ex == 3:
            q = '$f(x)=e^{-x^2}-2x$'
        elif self.__ex == 4:
            q = '$f(x)=e^{-sin^2 x+x}-2x$'
        elif self.__ex == 5:
            q = '$f(x)=x^3-2x+1$'
        else:
            q = '$f(x)=x^4+2x^2-x-3$'
        return q
