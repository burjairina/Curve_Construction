class FFP:
    def __init__(self, ex, eg):
        self.__ex = ex
        self.__eg = eg

    def f(self, x):
        if self.__ex == 1:
            y = x ** 3 + 4 * x ** 2 - 10
        elif self.__ex == 2:
            y = x ** 3 - 2 * x + 1
        else:
            y = x ** 4 + 2 * x ** 2 - x - 3
        return y

    def f_title(self):
        if self.__ex == 1:
            q = r"$f(x) = x^3 + 4x^2 - 10$"
        elif self.__ex == 2:
            q = r"$f(x) = x^3 - 2x + 1$"
        else:
            q = r"$f(x) = x^4 + 2x^2 - x - 3$"
        return q

    def lim_int(self):
        if self.__ex == 1:
            a = 1
            b = 2
            x0 = 2
        elif self.__ex == 2:
            a = 3 / 4
            b = 2
            x0 = 1.5
        else:
            a = 0.5
            b = 2
            x0 = 1.5
        return a, b, x0

    def g(self, x):
        if self.__ex == 1:
            if self.__eg == 1:
                y = x - x ** 3 - 4 * x ** 2 + 10
            elif self.__eg == 2:
                y = (10 / x - 4 * x) ** (1 / 2)
            elif self.__eg == 3:
                y = 1 /2 * (10 -x ** 3) ** (1 / 2)
            elif self.__eg == 4:
                y = (10 / (4 + x)) ** (1 / 2)
            else:
                y = x - (x ** 3 + 4 * x ** 2 - 10) / (3 * x ** 2 + 8 * x)
        elif self.__ex == 2:
            if self.__eg == 1:
                y = (x ** 3 + 1) / 2
            elif self.__eg == 2:
                y = 2 / x - 1 / (x ** 2)
            elif self.__eg == 3:
                y = (2 - 1 / x) ** (1 / 2)
            else:
                y = -(1 - 2 * x) ** (1 / 3)
        else:
            if self.__eg == 1:
                y = (3 + x - 2 * x ** 2) ** (1 / 4)
            elif self.__eg == 2:
                y = ((x + 3 - x ** 4) / 2) ** (1 / 2)
            elif self.__eg == 3:
                y = ((x + 3 ) / (x ** 2 + 2)) **(1 / 2)
            else:
                y = (3 * x ** 4 + 2 * x ** 2 + 3) / (4 * x ** 3 + 4 * x - 1)
        return y