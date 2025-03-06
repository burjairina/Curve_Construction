import time as t

class SecantMethod:
    def __init__(self, x0, x1, e):
        self.__x0 = x0
        self.__x1 = x1
        self.__e = e

    def secant_method(self, funct):
        start_time = float(t.perf_counter())


        if funct.f(self.__x0) * funct.f(self.__x1) >= 0:
            print('change initial values\n')
            m = []
            k = 1

            timeb = float(t.perf_counter()) - start_time
            print('Secant Method\n')
            print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
            print(f"The execution time for Secant Method is {timeb:.7f}")
            return
        else:
            x2 = self.__x1 - funct.f(self.__x1) * (self.__x1 - self.__x0) / (funct.f(self.__x1)- funct.f(self.__x0))
            k = 1
        d = abs(x2 - self.__x1)
        if funct.f(x2) == 0:
            m = x2

            timeb = float(t.perf_counter()) - start_time
            print('Newton Method\n')
            print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
            print(f"The execution time for Secant Method is {timeb:.7f}")
            return
        elif d < self.__e:
            m = x2

            timeb = float(t.perf_counter()) - start_time
            print('Newton Method\n')
            print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
            print(f"The execution time for Secant Method is {timeb:.7f}")
            return
        else:
            while d > self.__e and k <= 100:
                self.__x0 = self.__x1
                self.__x1 = x2
                x2 = self.__x1 - funct.f(self.__x1) * (self.__x1 - self.__x0) / (funct.f(self.__x1)- funct.f(self.__x0))
                d = abs(x2 - self.__x1)
                k = k + 1
            m = x2
            if k >100:
                print("the method does not converge in 100 steps\n")

            timeb = float(t.perf_counter()) - start_time

            print('Secant Method\n')
            print(f"The approx. sol. is {m:.5f} obtained in {k} steps")
            print(f"The execution time for Secant Method is {timeb:.7f}\n\n")