import numpy as np

class Bezier:
    def __init__(self, t, px, py):
        self.__t = t
        self.__px = px
        self.__py = py

    def bernstein_polynomial(self, n):
        b = np.zeros(n + 1)
        t1 = 1 - self.__t
        b[0] = 1
        for j in range (1, n +1):
            saved = 0
            for k in range(j):
                temp = b[k]
                b[k] = saved + t1 * temp
                saved = self.__t * temp
            b[j] = saved
        return b

    def bezier_curve(self):
        n = len(self.__px) - 1
        b = self.bernstein_polynomial(n)
        x = np.sum(self.__px * b)
        y = np.sum(self.__py * b)
        return x, y