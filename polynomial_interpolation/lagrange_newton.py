import numpy as np

class LagrangeNewton:
    def __init__(self, xv, fv, x):
        self.__xv = xv
        self.__fv = fv
        self.__x = x

    def divided_diff(self):
        n = len(self.__xv) - 1
        a = np.zeros((n, n))
        for i in range(n):
            a[i, 0] = (self.__fv[i + 1] - self.__fv[i]) / (self.__xv[i + 1] - self.__xv[i])
        for j in range(1, n):
            for i in range(n - j ):
                a[i, j] = (a[i + 1, j - 1] - a[i, j - 1]) / (self.__xv[i + j + 1] - self.__xv[i])
        return a, n

    def lagrange_newton(self):
        if self.__x in self.__xv:
            index = np.where(self.__xv == self.__x)[0][0]
            n1 = self.__fv[index]
        else:
            a, n = self.divided_diff()
            n1 = self.__fv[0]
            p = 1
            for i in range(n):
                p = p * (self.__x - self.__xv[i])
                n1 += a[0, i] * p
        return n1