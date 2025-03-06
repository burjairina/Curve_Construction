import time as t

class FixedPointMethod:
    def __init__(self, x0, e):
        self.__x0 = x0
        self.__e = e

    def fixed_point_method(self, f):
        print('Fixed Point Method\n')
        start_time = float(t.perf_counter())

        x1 = f.g(self.__x0)
        k = 1
        d = abs(x1 - self.__x0)
        if f.f(x1) == 0:
            m = x1
        elif d < self.__e:
            m = x1
        else:
            while d > self.__e and k <= 100:
                x1 = f.g(self.__x0)
                d = abs(x1 - self.__x0)
                k += 1
                self.__x0 = x1
            m = x1
            if k>100:
                print('the method fails to converge in 100 steps\n')

        timeb = float(t.perf_counter()) - start_time

        print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
        print(f"The execution time for Fixed Point Method is {timeb:.7f}\n\n")
