class NumericalIntegration:
    def __init__(self, a, b, n):
        self.__a = a
        self.__b = b
        self.__n = n

    def trap_rule(self, f):
        h = (self.__b - self.__a) / self.__n
        t = 1 / 2 * (f.f(self.__a) + f.f(self.__b))
        for i in range (self.__n - 1):
            t += f.f(self.__a + (i + 1) * h)
        t *= h
        print('Trapezoidal Rule')
        print('Approximation of the integral, using ', self.__n +1, ' nodes is ', t, '\n\n')
        return

    def midp_rule(self, f):
        h = (self.__b - self.__a) / self.__n
        m = 0
        for i in range(self.__n):
            m += f.f(self.__a + (2 * i - 1) * h / 2)
        m *= h
        print('Midpoint Rule')
        print('Approximation of the integral, using ', self.__n +1, ' nodes is ', m, '\n\n')
        return

    def simpson_rule(self, f):
        if self.__n % 2 != 0:
            print('give an even value for the number n')
            s = []
        else:
            h = (self.__b - self.__a) / self.__n
            s1 = 0
            s2 = 0
            for i in range(int(self.__n / 2) - 1):
                s1 += f.f(self.__a + 2 * (i + 1) * h)
                s2 += f.f(self.__a + (2 * (i + 1) - 1) * h)
            s2 += f.f(self.__a + (self.__n - 1) * h)
            s = (f.f(self.__a) + 2 * s1 + 4 * s2 + f.f(self.__b)) * h / 3
        print('Simpson Rule')
        print('Approximation of the integral, using ', self.__n + 1, ' nodes is ', s, '\n\n')
        return

    def simpson_3_8_rule(self, f):
        if self.__n % 3 != 0:
            print('give an even value for the number n')
            s = []
        else:
            h = (self.__b - self.__a) / self.__n
            s1 = 0
            s2 = 0
            for i in range(1, self.__n):
                if i % 3 == 0:
                    s1 += f.f(self.__a + i * h)
                else:
                    s2 += f.f(self.__a + i * h)
            s = (f.f(self.__a) + 2 * s1 + 3 * s2 + f.f(self.__b)) * 3 * h / 8
        print('Simpson 3/8 Rule')
        print('Approximation of the integral, using ', self.__n + 1, ' nodes is ', s, '\n\n')
        return
