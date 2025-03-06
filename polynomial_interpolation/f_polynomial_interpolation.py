import numpy as np

class FPI:
    def __init__(self, ex):
        self.__ex = ex

    def f(self, x):
        if self.__ex == 1:
            y = np.sin(x)
        elif self.__ex == 2:
            y = 1 / ( 1 + x ** 2)
        else:
            y=1 / ( 1 + x ** 2)
        return y

    def f_title(self):
        if self.__ex == 1:
            q1 = '$f(x)=sin(x)$'
            q2 = 'Equidistant Nodes'
        elif self.__ex == 2:
            q1 = r"$f(x) = \frac{1}{1 + x^2}$"
            q2 = 'Equidistant Nodes'
        else:
            q1 = r"$f(x) = \frac{1}{1 + x^2}$"
            q2 = 'Chebyshev Nodes'
        return q1, q2

    def nodes_function(self):
        if self.__ex == 1:
            xv = np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2)
            fv = np.sin(xv)
        elif self.__ex == 2:
            xv = np.arange(-5, 6)
            fv = 1 /(1 + xv ** 2)
        else:
            #Chebyshev Nodes on [a, b]
            nc = 15
            a = -5
            b = 5
            xv = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(1, nc + 1) - 1) / (2 * nc) * np.pi)
            fv = 1 / (1 + xv ** 2)
        return xv, fv

    def nodes_function_derivative(self):
        if self.__ex == 1:
            xv = np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2)
            fv = np.sin(xv)
            fvd = np.cos(xv)
        elif self.__ex == 2:
            xv = np.arange(-5, 6)
            fv = 1 /(1 + xv ** 2)
            fvd = 2 * xv / ((1 + xv ** 2) ** 2)
        else:
            #Chebyshev Nodes on [a, b]
            nc = 15
            a = -5
            b = 5
            xv = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(1, nc + 1) - 1) / (2 * nc) * np.pi)
            fv = 1 / (1 + xv ** 2)
            fvd = -2 * xv / ((1 + xv ** 2) ** 2)
        return xv, fv, fvd