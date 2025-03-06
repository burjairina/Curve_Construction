import numpy as np

class LagrangeBarycentric:
    def __init__(self, xv, fv, x):
        self.__xv = xv
        self.__fv = fv
        self.__x = x

    def barycentric_weights(self):
        n = len(self.__xv) - 1
        w = np.ones(n + 1)
        if len(self.__xv) != len(np.unique(self.__xv)):
            print('there are repeated elements in vector nodes')
            return None, None
        else:
            for k in range (n + 1):
                p = 1
                for j in range (n + 1):
                    if j != k:
                        p = p * (self.__xv[k] - self.__xv[j])
                w[k] = 1 / p
        return w, n

    def lagrange_barycentric(self):
        if self.__x in self.__xv:
            index = np.where(self.__xv == self.__x)[0][0] if np.any(self.__xv == self.__x) else 0
            b = self.__fv[index]
        else:
            w, n = self.barycentric_weights()
            if w is None:  # Handle the case of repeated nodes
                return None
            s1 = 0
            s2 = 0
            for i in range (n + 1):
                c = w[i] / (self.__x - self.__xv[i])
                s1 = s1 + c * self.__fv[i]
                s2 = s2 + c
            b = s1 / s2
        return b