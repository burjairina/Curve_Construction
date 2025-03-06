import numpy as np
import time as t

class IterativeMethods:
    def __init__(self, a, b, xo, e):
        self.__a = a
        self.__b = b
        self.__xo = xo
        self.__e = e

    def gauss_seidel_method(self):
        print('Gauss - Seidel Method\n')
        start_time = float(t.perf_counter())
        if self.__b.ndim == 2 and self.__b.shape[1] == 1:
            self.__b = self.__b.T
        n = len(self.__b)
        d = 1
        k = 0
        while d > self.__e and k <= 500:
            xn = np.zeros(n)
            for i in range (n):
                s1 = 0
                s2 = 0
                for j in range (i):
                    s1 += self.__a[i, j] * xn[j]
                for j in range (i + 1, n):
                    s2 += self.__a[i, j] * self.__xo[j]
                xn[i] = (self.__b[i] - s1 -s2) / self.__a[i, i]
            d = np.linalg.norm(xn - self.__xo, np.inf)
            k += 1
            self.__xo = xn.copy()
        sol = self.__xo
        steps = k
        if k > 500:
            print('the method fails to converge in 500 steps\n')

        timeb = float(t.perf_counter()) - start_time

        print('The approx. sol. is ', sol, 'obtained in ', steps, 'steps')
        print('The execution time for Gauss - Seidel Method is ', timeb, '\n\n')
        return

    def jacobi_method(self):
        print('Jacobi Method\n')
        start_time = float(t.perf_counter())
        if self.__b.ndim == 2 and self.__b.shape[1] == 1:
            self.__b = self.__b.T
        n = len(self.__b)
        d = 1
        k = 0
        while d > self.__e and k <= 500:
            xn = np.zeros(n)
            for i in range(n):
                s = 0
                for j in range(n):
                    if j != i:
                        s += self.__a[i, j] * self.__xo[j]
                xn[i] = (self.__b[i] - s) / self.__a[i, i]
            d = np.linalg.norm(xn - self.__xo, np.inf)
            k += 1
            self.__xo = xn.copy()
        sol = self.__xo
        steps = k
        if k > 500:
            print('the method fails to converge in 500 steps\n')

        timeb = float(t.perf_counter()) - start_time

        print('The approx. sol. is ', sol, ' obtained in ', steps,  'steps')
        print('The execution time for Jacobi Method is', timeb, '\n\n')
        return