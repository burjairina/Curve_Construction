import numpy as np

class HermiteNewton:
    def __init__(self, xv, fv, fvd, x):
        self.__xv = xv
        self.__fv = fv
        self.__fvd = fvd
        self.__x = x

    def divided_diff_double_nodes(self):
        n = len(self.__xv) -1
        z = np.zeros(2 * n + 2)
        f = np.zeros((2 * n + 2, 1))
        a = np.zeros((2 * n + 1, 2 * n + 1))

        z[0:2 * n + 1:2] = self.__xv
        z[1:2 * n + 2:2] = self.__xv
        f[0:2 * n + 1:2, 0] = self.__fv
        f[1:2 * n + 2:2, 0] = self.__fv
        a[0:2 * n + 1:2, 0] = self.__fvd

        for i in range(1, 2 * n + 1, 2):
            a[i, 0] = (f[i + 1, 0] - f[i, 0]) / (z[i + 1] - z[i])

        for j in range(1, 2 * n + 1):
            for i in range(2 * n + 1 - j):
                a[i, j] = (a[i + 1, j - 1] - a[i, j - 1]) / (z[i + j + 1] - z[i])

        return a, z, n

    def hermite_newton(self):
        if self.__x in self.__xv:
            index = np.where(self.__xv == self.__x)[0][0]
            nh = self.__fv[index]
        else:
            a, z, n = self.divided_diff_double_nodes()
            nh = self.__fv[0]
            p = 1
            for i in range(2 * n + 1):
                p *= (self.__x - z[i])
                nh = nh + a[0, i] * p
        return nh