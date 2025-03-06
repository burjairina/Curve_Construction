import time as t

class NewtonMethod:
    def __init__(self, x0, e):
        self.__x0 = x0
        self.__e = e

    def newton_method(self, funct):
        start_time = float(t.perf_counter())

        if funct.fd(self.__x0) == 0:
            print('Change initial value\n')
            m = []
            k = 1

            timeb = float(t.perf_counter()) - start_time
            print('Newton Method\n')
            print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
            print(f"The execution time for Newton Method is {timeb:.7f}")
            return
        else:
            xn = self.__x0 - funct.f(self.__x0) / funct.fd(self.__x0)
            k = 1
        d = abs(xn - self.__x0)
        if funct.f(xn) == 0:
            m = xn
        elif d < self.__e:
            m = xn
        else:
            xn = 1
            while d > self.__e and k <= 100:
                self.__x0 = xn
                xn = self.__x0 - funct.f(self.__x0) / funct.fd(self.__x0)
                d = abs(xn - self.__x0)
                k = k + 1
            m = xn
            if k > 100:
                print("the method does not converge in 100 steps\n")

        timeb = float(t.perf_counter()) - start_time

        print('Newton Method\n')
        print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
        print(f"The execution time for Newton Method is {timeb:.7f}\n\n")