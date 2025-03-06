import numpy as np

class DirectMethods:
    def __init__(self, b):
        self.__b = b

    def backward_subst(self, u):
        if self.__b.ndim == 2 and self.__b.shape[1] == 1:
            self.__b = self.__b.T
        n = len(self.__b)
        x = np.zeros(n)
        x[n - 1] = self.__b[n - 1] / u[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            s = 0
            for j in range(i + 1, n):
                s += u[i, j] * x[j]
            x[i] = (self.__b[i] - s) / u[i, i]
        return x

    def forward_subs(self, l):
        if self.__b.ndim == 2 and self.__b.shape[1] == 1:
            self.__b = self.__b.T
        n = len(self.__b)
        x = np.zeros(n)
        x[0] = self.__b[0] / l[0 , 0]
        for i in range (1, n):
            s = 0
            for j in range(i):
                s += l[i, j] * x[j]
            x[i] = (self.__b[i] - s) / l[i, i]
        return x

    def gaussian_el_piv(self, a):
        if self.__b.ndim == 2 and self.__b.shape[1] == 1:
            self.__b = self.__b.T
        n = len(self.__b)
        m1 = np.zeros((n , n))
        for i in range (n - 1):
            m = abs(a[i , i])
            p = i
            for j in range (i + 1, n ):
                if m < abs(a[j, i]):
                    m = abs(a[j, i])
                    p = j

            m = abs(a[p, i])
            if m == 0:
                print('the matrix a is singular\n')
                print('we do not have an unique solution\n')
                x = []
                return x, m1
            if p != i:
                aux = a[p, :].copy()
                a[p, :] = a[i, :]
                a[i, :] = aux
                auxb = self.__b[p]
                self.__b[p] = self.__b[i]
                self.__b[i] = auxb
            for j in range (i + 1, n):
                m1[j, i] = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m1[j, i] * a[i, :]
                self.__b[j] = self.__b[j] - m1[j, i] * self.__b[i]
        if a[n-1 ,n-1] == 0:
            print('we do not have unique solution\n')
            x = []
        else:
            x = self.backward_subst(a)
        return x, m1

    def gaussian_el_without_piv(self, a):
        if self.__b.ndim == 2 and self.__b.shape[1] == 1:
            self.__b = self.__b.T
        n = len(self.__b)
        m1 = np.zeros((n, n))
        for i in range (n - 1):
            for j in range (i + 1, n):
                m1[j, i] = a[j, i] / a[i, i]
                a[j, :] = a[j, :] - m1[j, i] * a[i, :]
                self.__b[j] = self.__b[j] - m1[j, i] * self.__b[i]
        if a[n - 1, n - 1] == 0:
            print('we do not have unique solution\n')
            xwp = []
        else:
            xwp = self.backward_subst(a)
        return xwp , m1