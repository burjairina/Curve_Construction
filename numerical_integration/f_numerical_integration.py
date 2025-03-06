import numpy as np

class FNI:
    def __init__(self, __ex):
        self.__ex = __ex

    def f(self, x):
        if self.__ex == 1:
            y = np.cos(x)
        elif self.__ex == 2:
            y = x ** 2 + 3 * x
        else:
            y = x * np.exp(-x) * np.cos(2 * x)
        return y

    def f_title(self):
        if self.__ex == 1:
            q = r'$\displaystyle \int^{\frac{\pi}{2}}_0 \cos(x)\,dx=1$'
        elif self.__ex == 2:
            q = r'$\displaystyle \int^{1}_0 (x^2+3\cdot x)\,dx=\frac{11}{6}$'
        else:
            q = r'$\displaystyle \int^{2\pi}_0 x \cdot e^{-x} \cdot \cos(x)\,dx=\frac{3(e^{-2\pi}-1)-10\pi e^{-2\pi}}{25}$'
        return q

    def lim_int(self):
        if self.__ex == 1:
            a = 0
            b = np.pi / 2
            exact = 1
        elif self.__ex == 2:
            a = 0
            b = 1
            exact = 11 / 6
        else:
            a = 0
            b = 2 * np.pi
            exact = (3 * (np.exp(-2 * np.pi) - 1) - 10 * np.pi * np.exp(-2 * np.pi)) / 25
        return a, b, exact