from scipy.optimize import fsolve
import math
import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib import mlab
import scipy.optimize
from scipy.optimize import leastsq, broyden1, fsolve


def task_2_1():
    def func(x):
        return 2 ** x + 5 * x - 3

    def sys_func(p):
        x, y = p
        return math.sin(x + 2) - 1.5 - y, math.cos(y - 2) + x - 0.5

    x = broyden1(func, 0, f_tol=1e-6)
    print("Решение нелинейного уравнения")
    print(x)

    x, y = fsolve(sys_func, (1, 1))
    print("Решение системы нелинейных уравнений")
    print(x, y)

def task_2_2_1():
    print('Задание 2.2.1')
    T = np.array([20, 22, 24, 26, 28])
    R = np.array([40, 22.0, 12.0, 6.6, 3.6])

    def func(arguments, T):
        return arguments[0] * T + arguments[1]

    def err_func(arguments, T, R):
        return R - func(arguments, T)

    arguments0 = np.array([0, 0])
    arguments0, tmp = leastsq(err_func, arguments0, args=(T, R))
    a, b = arguments0[0], arguments0[1]
    print('Function', 'R=', arguments0[0], '*T+', arguments0[1])
    print('R при T = 21 :', a * 21. + b)

    for i in range(len(T)):
        print('ошибка при i', i, '=', R[i] - T[i] * arguments0[0] - arguments0[1])
    plt.scatter(T, R, c='g')
    ax = plt.gca()
    plt.plot(T, T * arguments0[0] + arguments0[1],label="Task2.2.1")
    plt.legend(loc='upper right', frameon=False)
    plt.show()


def task_2_2_2():
    print()
    print('Задание 2.2.2')
    T = np.array([0, 0.83008, 1.66016, 2.49024, 3.32032, 4.1504])
    N = np.array([1000594, 942916, 892598, 840004, 795709, 748860])
    N_0 = N[0]

    def func(arguments, T, N_0):
        return np.log(N_0) - arguments * T

    def err_func(arguments, T, N):
        return np.log(N) - func(arguments, T, N_0)

    arguments0 = 0
    arguments0, tmp = leastsq(err_func, arguments0, args=(T, N))
    print('Function', 'N(t)=N(0)*e^(-(', *arguments0,')* t)')
    print('T(1/2) =', *(np.log(2) / arguments0))

    for i in range(len(T)):
        print('ошибка при i', i, '=', np.log(N[i]) - (np.log(N_0) - arguments0 * T[i]))

    plt.scatter(T, np.log(N), c='g')
    plt.plot(T, np.log(N_0) - arguments0 * T,label="Task2.2.1")
    plt.legend(loc='upper right', frameon=False)
    plt.show()


task_2_1()
task_2_2_1()
task_2_2_2()
