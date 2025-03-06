import numpy as np

class LagrangeClassic:
    def __init__(self, xv, fv, x):
        self.__xv = xv
        self.__fv = fv
        self.__x = x

    def fundamental_lagrange(self):
        n = len(self.__xv) - 1
        l = np.zeros(n + 1)
        if len(self.__xv) != len(np.unique(self.__xv)):
            print('there are repeated elements in vector nodes')
            return None, None
        elif self.__x in self.__xv:
            index = np.where(self.__xv == self.__x)[0][0]
            l[index] = 1
        else:
            for i in range (n + 1):
                p1 = 1
                p2 = 1
                for j in range (n + 1):
                    if j != i:
                        p1 = p1 * (self.__x - self.__xv[j])
                        p2 = p2 * (self.__xv[i] - self.__xv[j])
                l[i] = p1 / p2
        return l, n

    def lagrange_classic(self):
        if self.__x in self.__xv:
            index = np.where(self.__xv == self.__x)[0][0] if np.any(self.__xv == self.__x) else 0
            l1 = self.__fv[index]
        else:
            l, n = self.fundamental_lagrange()
            l1 = 0
            for i in range (n + 1):
                l1 =l1 + self.__fv[i] * l[i]
        return l1