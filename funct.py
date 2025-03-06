import numpy as np
import matplotlib.pyplot as plt
import time

from nonlinear_equations.bisection_method import BisectionMethod
from nonlinear_equations.newton_method import NewtonMethod
from nonlinear_equations.secant_method import SecantMethod
from nonlinear_equations.fixed_point_method import FixedPointMethod
from nonlinear_equations.f_fixed_point import FFP
from nonlinear_equations.f_nonlinear_eq import FNE
from polynomial_interpolation.lagrange_classic import LagrangeClassic
from polynomial_interpolation.lagrange_newton import LagrangeNewton
from polynomial_interpolation.lagrange_barycentric import LagrangeBarycentric
from polynomial_interpolation.hermite_newton import HermiteNewton
from polynomial_interpolation.spline_interpolation import SplineInterpolation
from polynomial_interpolation.f_polynomial_interpolation import FPI
from polynomial_interpolation.f_spline_interpolation import FSI
from linear_systems_direct_methods.direct_methods import DirectMethods
from linear_systems_direct_methods.f_direct_methods import FDM
from linear_systems_direct_methods.thomas_algorithm import ThomasAlgorithm
from linear_systems_iterative_methods.iterative_methods import IterativeMethods
from linear_systems_iterative_methods.f_iterative_methods import FIM
from numerical_integration.numerical_integration import NumericalIntegration
from numerical_integration.f_numerical_integration import FNI
from eigenvalues.bezier.bezier import Bezier
from eigenvalues.power_method.power_method import PowerMethod
from eigenvalues.power_method.f_power_method import FPM
from eigenvalues.euler_explicit.euler_explicit import EulerExplicit
from eigenvalues.euler_explicit.f_euler_explicit import FEE


#NONLINEAR EQUATIONS
def nonlinear_eq(ex, exf, eg):
    #merge
    fneq = FNE(ex)
    a, b, x0n, x0s, x1s = fneq.lim_int()
    x = np.arange(a - 0.5, b + 0.5 + 0.05, 0.05)
    nx = len(x)
    y = fneq.f(x)
    z = np.zeros(nx)
    q = fneq.f_title()

    plt.figure('Nonlinear Equations')
    plt.plot(x, y, color='m', linewidth=1.5)
    plt.plot(x, z, color='k', linewidth=2)

    plt.axis((a - 0.6, b + 0.6, min(y) - 0.15, max(y) + 0.15))

    plt.title(f'The Graphic of {q}', fontsize=14)
    plt.xlabel(r'$x$', fontsize=12)  # r'' is used to denote a raw string for LaTeX
    plt.ylabel(r'$f(x)$', fontsize=12)

    plt.show()
    print('NUMERICAL METHODS FOR NONLINEAR EQUATIONS\n')

    e = 10 ** (-10)

    # Bisection Method
    BisectionMethod(a, b, e).bisection_method(fneq)

    # Newton Method
    NewtonMethod(x0n, e).newton_method(fneq)

    # Secant Method
    SecantMethod(x0s, x1s, e).secant_method(fneq)

    # Fixed Point Method
    ffp = FFP(exf, eg)
    q = ffp.f_title()
    a, b, x0 = ffp.lim_int()
    x = np.arange(a - 0.5, b + 0.5, 0.05)
    nx = len(x)
    y = ffp.f(x)
    z = np.zeros(nx)

    plt.figure('Fixed Point Method')
    plt.title(f'The Graphic of {q}', fontsize=14, verticalalignment='bottom')
    plt.xlabel(r'$x$', fontsize=12)  # Use raw string for LaTeX formatting
    plt.ylabel(r'$f(x)$', fontsize=12)
    plt.axis((a - 0.6, b + 0.6, min(y) - 0.15, max(y) + 0.15))

    plt.plot(x, y, color='m', linewidth=1.5)
    plt.plot(x, z, color='k', linewidth=2)

    plt.show()

    e1 = 10 ** (-10)
    FixedPointMethod(x0, e1).fixed_point_method(ffp)
    return


#POLYNOMYAL INTERPOLATION
def polynomial_interpolation(ex1, ex2, ex3):
    # Lagrange Interpolation - cred ca merge
    fpi = FPI(ex1)
    q1, q2 = fpi.f_title()
    xv, fv = fpi.nodes_function()
    x = np.linspace(np.min(xv), np.max(xv), int((np.max(xv) - np.min(xv)) / 0.05) + 1)
    # x = np.arange(np.min(xv), np.max(xv) + 0.05, 0.05)
    y = fpi.f(x)
    nx = len(x)
    l = np.zeros(nx)
    n = np.zeros(nx)
    b = np.zeros(nx)
    for k in range(nx):
        l[k] = LagrangeClassic(xv, fv, x[k]).lagrange_classic()
        n[k] = LagrangeNewton(xv, fv, x[k]).lagrange_newton()
        b[k] = LagrangeBarycentric(xv, fv, x[k]).lagrange_barycentric()
    plt.figure(1)

    plt.plot(xv, fv, 'g*', markersize=10, label='Data Points')  # Green stars
    plt.plot(x, y, color='r', linewidth=3)  # , label='y')  # Red line
    plt.plot(x, l, color='k', linewidth=2, linestyle=':')  # , label='L')  # Black dotted line
    plt.plot(x, n, color='b', linewidth=2, linestyle='-.')  # , label='N')  # Blue dash-dot line
    plt.plot(x, b, color='m', linewidth=2, linestyle='--')  # , label='B')

    plt.title(f'The Graphic of {q1}', fontsize=14)
    plt.ylabel(q2, fontsize=12)
    plt.show()

    plt.figure(2)

    log_error_l = np.log10(abs(l - y))
    log_error_n = np.log10(abs(n - y))
    log_error_b = np.log10(abs(b - y))

    plt.title(r'The $log_{10}$ (absolute error)', fontsize=14)  # Title with LaTeX
    plt.plot(x, log_error_l, color='k', linewidth=2, linestyle=':', label='Classic')  # Classic vs exact
    plt.plot(x, log_error_n, color='b', linewidth=2, linestyle='-.', label='Newton')  # Newton vs exact
    plt.plot(x, log_error_b, color='m', linewidth=2, linestyle='--', label='Barycentric')  # Barycentric vs exact

    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

    plt.figure(3)

    log_diff_ln = np.log10(np.abs(l - n))
    log_diff_lb = np.log10(np.abs(l - b))
    log_diff_bn = np.log10(np.abs(b - n))

    plt.title(r'The $log_{10}$ (absolute difference)', fontsize=14)  # Title with LaTeX
    plt.plot(x, log_diff_ln, color='b', label='Classic vs Newton')  # Classic vs Newton
    plt.plot(x, log_diff_lb, color='m', label='Classic vs Barycentric')  # Classic vs Barycentric
    plt.plot(x, log_diff_bn, color='k', label='Barycentric vs Newton')  # Barycentric vs Newton

    plt.legend(fontsize=12)
    plt.grid(True)  # Equivalent to box on
    plt.show()


    # Hermite interpolation - nu arata deloc log abs error si nici ex2
    fpi = FPI(ex2)
    q1, q2 = fpi.f_title()
    xv, fv, fvd = fpi.nodes_function_derivative()
    x = np.arange(min(xv), max(xv) + 0.01, 0.01)
    y = fpi.f(x)
    nx = len(x)
    nh = np.zeros(nx)

    for k in range(nx):
        nh[k] = HermiteNewton(xv, fv, fvd, x[k]).hermite_newton()

    plt.figure(1)
    plt.title(f'The Graphic of {q1}', fontsize=14)
    plt.plot(xv, fv, 'g*', markersize=10, label='Data Points')  # Data points
    plt.plot(x, y, color='r', linewidth=3, label='Original Function')  # Original function
    plt.plot(x, nh, color='b', linewidth=2, linestyle='-.', label='Hermite Interpolation')  # Hermite interpolation
    plt.ylabel(q2, fontsize=12)
    plt.xlabel('$x$', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Figure 2: Logarithmic absolute error
    plt.figure(2)
    plt.title(r'The $log_{10}$ (absolute error)', fontsize=14)
    error = np.abs(nh - y)
    # Avoid log10 of zero by masking or adding a small epsilon
    log_error = np.log10(error)
    plt.plot(x, log_error, color='k', linewidth=2, linestyle=':', label='Logarithmic Error')  # Error plot
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel(r'$\log_{10}(|Nh - y|)$', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Spline Interpolation
    fsi = FSI(ex3)
    d = fsi.division()
    nd = len(d)
    fd = np.zeros(nd)
    for k in range(nd):
        fd[k] = fsi.f(d[k])
    x = np.arange(min(d), max(d) + 0.01, 0.01)
    nx = len(x)
    y = np.zeros(nx)
    s = np.zeros(nx)
    for k in range(nx):
        y[k] = fsi.f(x[k])
        s[k] = SplineInterpolation(d, fd, x[k]).spline_linear()

    plt.figure('Spline Interpolation')
    plt.grid(True)
    plt.box(True)
    plt.plot(x, y, color='k', linewidth=2, label='Function')
    plt.plot(d, fd, 'go', markersize=6, markerfacecolor='g', label='Nodes')
    plt.plot(x, s, color='r', linewidth=1.5, label='Linear Spline')
    plt.legend()
    plt.show()

    return


#LINEAR SYSTEMS - DIRECT METHODS
def direct_methods(ex):
    #merge
    f = FDM(ex)
    a, b, exact = f.f()
    print('LINEAR SYSTEMS - DIRECT METHODS\n')
    print('the matrix of the system')
    print('A =', a, '\n')
    print('the right hand  side (as row vector) of the system')
    print("b =", b, '\n')
    print('the exact solution (as row vector) of the system')
    print("exact solution = ", exact, '\n')
    if np.allclose(a, np.tril(a)):
        x = DirectMethods(b).forward_subs(a)
        print('solution of the lower triangular system')
        print('aprox sol = ', x, '\n')
    elif np.allclose(a, np.triu(a)):
        x = DirectMethods(b).backward_subst(a)
        print('solution of the upper triangular system')
        print('aprox sol = ', x, '\n')
    else:
        a1 = a[:].copy()
        b1 = b[:].copy()
        x, m = DirectMethods(b1).gaussian_el_piv(a1)
        print('solution of the system using Gaussian elimination method')
        print('aprox sol = ', x, '\n')
        if f.get_ex() >= 4:
            a2 = a[:].copy()
            xwp, m = DirectMethods(b).gaussian_el_without_piv(a2)
            print('solution of system using Gaussian elimination method without pivoting strategies')
            print('A[0, 0] = ', a[0, 0], '\n')
            print('approx sol without pivoting strategies = ', xwp, '\n')
            print('the multipliers matrix')
            print('M = ', m, '\n')

    #Thomas Algorithm - merge
    print('\nThomas Algorithm\n')
    print('the system matrix is of type')
    t7 = np.diag(5 * np.ones(7)) + np.diag(-1 * np.ones(7 - 1), k=-1) + np.diag(-1 * np.ones(7 - 1), k=1)
    print(t7)
    n = 100
    exact = np.ones(n)
    r = np.ones(n) * 3
    r[0] = 4
    #r[1:n - 2] = 3
    r[n - 1] = 4

    a = 5 * np.ones(n)
    b = np.hstack([0, -np.ones(n-1)])
    c = np.hstack([-np.ones(n-1), 0])
    t = np.diag(5 * np.ones(n)) + np.diag(-np.ones(n - 1), -1) + np.diag(-np.ones(n - 1), 1)

    #Thomas Algorithm
    ThomasAlgorithm(a, b, c, r, t, exact).thomas_alg()

    #Gauss Elimination Algorithm
    ThomasAlgorithm(a, b, c, r, t, exact).elim_gauss_piv_partial()

    start_time = float(time.perf_counter())
    xm = np.linalg.solve(t, r)
    timpm = float(time.perf_counter()) - start_time

    print('mldivide function from Python\n')
    print(f'The execution time for mldivide function from Python is {timpm:.10f} seconds')
    eroarem = np.linalg.norm(exact - xm, np.inf)
    print(f'The error for mldivide function from Python is {eroarem:.15f}')
    return

#LINEAR SYSTEMS - ITERATIVE METHODS
def iterative_methods(ex):
    #merge pe toate cazurile
    f = FIM(ex)
    a, b, exact, xo = f.f()
    print('LINEAR SYSTEMS - ITERATIVE METHODS\n')
    print('the matrix of the system')
    print('A =', a, '\n')
    print('the right hand  side (as row vector) of the system')
    print("b =", b, '\n')
    print('the exact solution (as row vector) of the system')
    print('exact solution = ', exact, '\n')
    print('the initial value (as row vector) of the system')
    print('initial value = ', xo, '\n')
    e = 10 ** (-10)
    print('the constant in stop condition is ', e, '\n')

    #Jacobi Method
    IterativeMethods(a, b, xo, e).jacobi_method()

    #Gauss - Seidel Method
    IterativeMethods(a, b, xo, e).gauss_seidel_method()
    return

#NUMERICAL INTEGRATION
def numerical_integration(ex):
    #merge pe toate cazurile
    f = FNI(ex)
    n = 60
    a, b, exact = f.lim_int()
    print('NUMERICAL METHODS FOR NUMERICAL INTEGRATION\n')
    print('the exact value of the integral is ', exact, '\n')

    '''
    q = f.f_title()
    rcParams["text.usetex"] = True
    plt.figure(1)
    plt.gca().set_xticks([])  
    plt.gca().set_yticks([])  
    plt.box(on=True)  

    plt.text(0.5, 0.5, q, fontsize=20, ha='center', va='center')

    plt.show()
    '''

    #Trapezoidal Rule
    NumericalIntegration(a, b, n).trap_rule(f)

    #Midpoint Rule
    NumericalIntegration(a, b, n).midp_rule(f)

    #Simpson Rule
    NumericalIntegration(a, b, n).simpson_rule(f)

    #Simpson 3 / 8 Rule
    NumericalIntegration(a, b, n).simpson_3_8_rule(f)
    return

#EIGENVALUES
def eigenvalues(ex1, ex2):
    #Bezier Curve - merge
    t = np.arange(0, 1.01, 0.01)
    nt = len(t)
    px1 = np.array([3, 5, 5, 3])
    py1 = np.array([0, -3, 2, 2])
    px2 = np.array([3, 2, 1, 0, -1])
    py2 = np.array([2, 2, -3, -2, 4])
    x1 = np.zeros(nt)
    y1 = np.zeros(nt)
    x2 = np.zeros(nt)
    y2 = np.zeros(nt)
    for k in range(nt):
        x1[k], y1[k] = Bezier(t[k], px1, py1).bezier_curve()
        x2[k], y2[k] = Bezier(t[k], px2, py2).bezier_curve()
    plt.figure('Bezier Curve')
    plt.title("Bezier Curve")
    plt.plot(px1, py1, 'go', markersize=8, markerfacecolor='g', label="Control Points 1")
    plt.plot(px2, py2, 'rs', markersize=8, markerfacecolor='r', label="Control Points 2")
    plt.plot(x1, y1, color='k', linewidth=2, label="Bezier Curve 1")
    plt.plot(x2, y2, color='b', linewidth=2, label="Bezier Curve 2")
    plt.legend()
    plt.show()

    #Power Method - merge pe toate cazurile (ex3 arata diferit dar e corect)
    f1 = FPM(ex1)
    a, valpmax, x0 = f1.f()
    e = 10 ** (-10)
    PowerMethod(a, x0, e, valpmax).gershgorin_circles()
    PowerMethod(a, x0, e, valpmax).power_method()

    # Euler Explicit - merge pe toate cazurile
    f2 = FEE(ex2, 4)
    h = 0.01
    y0 = f2.initial_val()
    a, b = f2.lim_int()
    x = np.arange(a, b + h, h)
    sol_ex = f2.exact(x)
    y = EulerExplicit(x, y0).euler_explicit(f2)
    z = EulerExplicit(x, y0).runge(f2)
    w = EulerExplicit(x, y0).heunn(f2)

    error_ee = abs(y - sol_ex)
    error_r = abs(z - sol_ex)
    error_h = abs(w - sol_ex)

    plt.figure('Numerical methods for ODE-IVP')
    plt.plot(x, sol_ex, color='r', linewidth=1.5, label='Exact')  # Exact solution
    plt.plot(x, y, color='b', label='Euler-explicit')  # Euler-explicit solution
    plt.plot(x, z, color='g', label='Runge')  # Runge solution
    plt.plot(x, w, color='k', label='Heun')  # Heun solution
    plt.title('Numerical methods for ODE-IVP')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure('Approx. error for ODE-IVP')
    plt.plot(x, np.log10(error_ee), color='b', linewidth=1.5, label='Euler-explicit')  # Euler-explicit error
    plt.plot(x, np.log10(error_r), color='g', linewidth=1.5, label='Runge')  # Runge error
    plt.plot(x, np.log10(error_h), color='k', linewidth=1.5, label='Heun')  # Heun error
    plt.title('Approx. error for ODE-IVP')
    plt.xlabel('x')
    plt.ylabel('lg(absolute error)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

#SIGNATURE
def signature():
    pink = [1, 0.75, 0.8]
    t = np.arange(0, 1.01, 0.01)
    nt = len(t)
    strokes = [
        # "i"
        np.array([[0.25, 0], [0.25, 1]]),
        np.array([[0.25, 1.15], [0.05, 1.25], [0.15, 1.35], [0.25, 1.25]]),
        np.array([[0.25, 1.25], [0.35, 1.35], [0.45, 1.25], [0.25, 1.15]]),
        # "r"
        np.array([[0.5, 0], [0.5, 1], [0.5, 0.9], [0.5, 1], [0.5, 0.9], [0.75, 1.25], [1, 1]]),
        # "i"
        np.array([[1.25, 0], [1.25, 1]]),
        np.array([[1.25, 1.15], [1.05, 1.25], [1.15, 1.35], [1.25, 1.25]]),
        np.array([[1.25, 1.25], [1.35, 1.35], [1.45, 1.25], [1.25, 1.15]]),
        # "n"
        np.array([[1.5, 0], [1.5, 0.5], [1.5, 0.75], [1.75, 1.75], [2, 0.75], [2, 0.5], [2, 0]]),
        np.array([[2, 0], [2, 0.5], [2, 0.75], [2.25, 1.75], [2.5, 0.75], [2.5, 0.5], [2.5, 0]]),
        # "a"
        np.array(
            [[3.42175, 0.5], [3, -1], [2, 0.5], [3, 2.75], [3.5, 0.5], [3.5, 0.5], [3.3, 0.37], [3.5, 0.20], [4, 0]]),
    ]
    plt.figure('Bezier Signature')
    plt.title("Bezier Signature")
    for stroke in strokes:
        px = stroke[:, 0]
        py = stroke[:, 1]

        x = np.zeros(nt)
        y = np.zeros(nt)

        for k in range(nt):
            x[k], y[k] = Bezier(t[k], px, py).bezier_curve()

        plt.plot(x, y, color=pink, linewidth=2, label="Bezier Curve")

    plt.show()
    return