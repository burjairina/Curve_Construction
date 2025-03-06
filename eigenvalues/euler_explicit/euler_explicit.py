import numpy as np

class EulerExplicit:
    def __init__(self, x ,y0):
        self.__x = x
        self.__y0 = y0
        #self.__lambdaa = lambdaa

    def euler_explicit(self, f):
        n = len(self.__x)
        h = self.__x[1] - self.__x[0]
        print('\nh =', h)
        y = np.zeros(n)
        y[0] = self.__y0
        for i in range(n - 2):
            y[i + 1] = y[i] + h * f.f(self.__x[i], y[i])
        return y

    def heunn(self, f):
        n = len(self.__x)
        h = self.__x[1] - self.__x[0]
        w = np.zeros(n)
        w[0] = self.__y0
        for i in range(n - 2):
            wint = w[i] + h * f.f(self.__x[i], w[i])
            w[i + 1] = w[i] + 0.5 * h * f.f(self.__x[i], w[i]) + 0.5 * h * f.f(self.__x[i] + h, wint)
        return w

    def runge(self, f):
        n = len(self.__x)
        h = self.__x[1] - self.__x[0]
        z = np.zeros(n)
        z[0] = self.__y0
        for i in range(n - 2):
            zint = z[i] + 0.5 * h * f.f(self.__x[i], z[i])
            z[i + 1] = z[i] + h * f.f(self.__x[i] + 0.5 * h, zint)
        return z