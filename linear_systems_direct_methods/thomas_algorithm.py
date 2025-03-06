import numpy as np
import time as t

class ThomasAlgorithm:
    def __init__(self, a, b, c, r, ag, exact):
        self.__a = a
        self.__b = b
        self.__c = c
        self.__r = r
        self.__ag = ag
        self.__exact = exact

    def thomas_alg(self):
        start_time = float(t.perf_counter())
        n = len(self.__a)
        d = np.zeros(n)
        y = np.zeros(n)
        x = np.zeros(n)

        d[0] = self.__c[0] / self.__a[0]
        y[0] = self.__r[0] / self.__a[0]

        for i in range (n - 1):
            num = self.__a[i + 1] - self.__b[i + 1] * d[i]
            d[i + 1] = self.__c[i + 1] / num
            y[i + 1] = (self.__r[i + 1] - self.__b[i + 1] * y[i]) / num

        x[n - 1] = y[n - 1]
        for i in range (n - 2 , -1, -1):
            x[i] = y[i] - d[i] * x[i + 1]

        timeb = float(t.perf_counter()) - start_time

        print('\n\nThomas Algorithm\n')
        print(f'The execution time for Thomas Algorithm is {timeb:.7f}')
        errorth = np.linalg.norm(self.__exact - x, np.inf)
        print(f'The error for Thomas Algorithm is {errorth:.15f}\n')
        return

    def elim_gauss_piv_partial(self):
        start_time = float(t.perf_counter())
        n = len(self.__r)
        self.__ag = np.column_stack((self.__ag, self.__r))
        x = np.zeros(n)
        for i in range (n - 1):
            ma = abs(self.__ag[i, i])
            p = i
            for j in range (i, n - 1):
                if ma < abs(self.__ag[j , i]):
                    ma = abs(self.__ag[j , i])
                    p = j
            if self.__ag[p, i] == 0:
                print('The system does not have a solution\n')
            if p != i:
                aux = self.__ag[p, :].copy()
                self.__ag[p, :] = self.__ag[i, :]
                self.__ag[i, :] = aux
            for j in range(i + 1, n ):
                m = self.__ag[j, i] / self.__ag[i, i]
                self.__ag[j, :] -= m * self.__ag[i, :]
        if self.__ag[n - 1, n - 1] == 0:
            print('The system does not have a solution\n')
        else:
            x[n - 1] = self.__ag[n - 1 , n] / self.__ag[n - 1, n - 1]
            for i in range(n - 2, -1, -1):
                x[i] = (self.__ag[i, n] - np.sum(self.__ag[i, i + 1:n] * x[i + 1:n])) / self.__ag[i, i]
        timeb = float(t.perf_counter()) - start_time

        print('Gauss Elimination Algorithm\n')
        print(f'The execution time for Gauss Elimination Algorithm is {timeb:.7f}')
        erroreg = np.linalg.norm(self.__exact - x, np.inf)
        print(f'The error for Gauss Elimination Algorithm is {erroreg:.15f}\n')
        return