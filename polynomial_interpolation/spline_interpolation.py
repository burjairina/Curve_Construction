import numpy as np


class SplineInterpolation:
    def __init__(self, d, fd, x):
        self.__d = d
        self.__fd = fd
        self.__x = x

    def spline_linear(self):
        n = len(self.__d) - 1
        s = np.zeros(n + 1)
        if self.__x < self.__d[1]:
            s[0] = (self.__d[1] - self.__x) / (self.__d[1] - self.__d[0])
        if self.__x > self.__d[n]:
            s[n] = (self.__x - self.__d[n]) / (self.__d[n + 1] - self.__d[n])
        for k in range(1, n + 1):
            if self.__d[k - 1] <= self.__x <= self.__d[k]:
                s[k] = (self.__x - self.__d[k - 1]) / (self.__d[k] - self.__d[k - 1])
            elif self.__d[k] <= self.__x <= self.__d[k + 1]:
                s[k] = (self.__d[k + 1] - self.__x) / (self.__d[k + 1] - self.__d[k])
        s1 = sum(self.__fd * s)
        return s1