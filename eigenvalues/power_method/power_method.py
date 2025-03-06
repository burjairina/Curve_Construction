import numpy as np
import matplotlib.pyplot as plt
import time as t

class PowerMethod:
    def __init__(self, a, x0, e, valpmax):
        self.__a = a
        self.__x0 = x0
        self.__e = e
        self.__valpmax = valpmax

    def circle(self,xc, yc, r):
        theta = np.arange(0, 2 * np.pi, 0.001)
        nt = len(theta)
        x = np.zeros(nt)
        y = np.zeros(nt)
        for k in range (nt):
            x[k] = xc + r * np.cos(theta[k])
            y[k] = yc + r * np.sin(theta[k])
        return x, y

    def gershgorin_circles(self):
        n = len(self.__a[0 :])

        plt.figure('Row Gershgorin Circles')
        plt.title("Row Gershgorin Circles")
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        for i in range(n):
            xc = np.real(self.__a[i, i])
            yc = np.imag(self.__a[i, i])
            r = np.sum(np.abs(self.__a[i, :])) - np.abs(self.__a[i, i])
            x, y = self.circle(xc, yc, r)
            plt.plot(x, y, color='k', linewidth=1.5)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.show()

        # Column Gershgorin Circles
        plt.figure('Column Gershgorin Circles')
        plt.title("Column Gershgorin Circles")
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        for i in range(n):
            xc = np.real(self.__a[i, i])
            yc = np.imag(self.__a[i, i])
            r = np.sum(np.abs(self.__a[:, i])) - np.abs(self.__a[i, i])
            x, y = self.circle(xc, yc, r)
            plt.plot(x, y, color='k', linewidth=1.5)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.show()
        return

    def power_method(self):
        start_time = float(t.perf_counter())
        if self.__x0.ndim == 1 or self.__x0.shape[0] == 1:  # If x0 is 1D or a row vector
            self.__x0 = self.__x0.T
        self.__x0 /= np.linalg.norm(self.__x0)
        k = 0
        d = 1
        while d > self.__e and k < 100:
            y = self.__a @ self.__x0
            xn = y / np.linalg.norm(y)
            d = np.linalg.norm(xn - self.__x0)
            self.__x0 = xn
            k += 1
        x = self.__x0
        lambdaa = x.T @ self.__a @ x / (x.T @ x)

        timeb = float(t.perf_counter()) - start_time
        print('Power Method\n')
        print('The exact value of dominant eigenvalue is ', self.__valpmax, '\n')
        print('The approx value is ',lambdaa , ' obtained in', k,'steps\n')
        print('The execution time for Power Method is', timeb, '\n')