import numpy as np
import time as t

class BisectionMethod:
    def __init__(self, a, b, e):
        self.__a = a
        self.__b = b
        self.__e = e

    def bisection_method(self, funct):
        start_time = float(t.perf_counter())

        m = self.__a + (self.__b - self.__a) / 2
        d = abs(self.__b - self.__a)
        k = 1
        while d > self.__e and k <= 100:
            if np.sign(funct.f(self.__a)) * np.sign(funct.f(m)) < 0:
                self.__b = m
            else:
                self.__a = m
            m = self.__a + (self.__b - self.__a) / 2
            d = abs(self.__b - self.__a)
            k = k + 1
            if funct.f(m) == 0:
                break
        if k > 100:
            print("the method does not converge in 100 steps\n")

        timeb = float(t.perf_counter()) - start_time

        print('Bisection Method\n')
        print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
        print(f"The execution time for Bisection Method is {timeb:.7f}\n\n")
