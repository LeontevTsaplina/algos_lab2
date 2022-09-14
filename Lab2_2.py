import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from typing import Callable


def lin_func(x: list, a: float, b: float) -> np.array:
    """
    Function for finding result of linear function for every x

    :param x: list of elements
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type a: float
    :type b: float
    :return: array of results of counting
    :rtype: np.array
    """

    return a * np.array(x) + b


def ration_func(x: list, a: float, b: float) -> np.array:
    """
    Function for finding result of rational function for every x

    :param x: list of elements
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type a: float
    :type b: float
    :return: array of results of counting
    :rtype: np.array
    """

    return a / (1 + b * np.array(x))


def errors_func(function: Callable, x: list, a: float, b: float, y: list) -> np.array:
    """
    Function for finding errors function of input function

    :param function: input function
    :param x: list of elements
    :param a: coefficient a
    :param b: coefficient b
    :param y: list of results
    :type function: Callable
    :type x: list
    :type a: float
    :type b: float
    :type y: list
    :return: array of results of counting
    :rtype: np.array
    """

    return np.sum((function(x, a, b) - y) ** 2)


def enumeration(func: Callable, x: list, y: list) -> (float, float):
    """
    Function of exhaustive search of minimum of input function

    :param func: input function
    :param x: list of x
    :param y: list of y
    :return: best params a and b to minimize function
    :rtype: tuple
    """

    min_error = np.inf
    best_a = np.inf
    best_b = np.inf

    for a_i in np.linspace(-1, 1, 100):
        for b_j in np.linspace(-1, 1, 100):
            current_error = errors_func(func, x, a_i, b_j, y)

            if current_error < min_error:
                min_error = current_error
                best_a = a_i
                best_b = b_j

    return best_a, best_b


def gauss(func: Callable, x: list, y: list, eps: float = 0.001) -> (float, float):
    """
    Function of gauss's method to find minimum of input function

    :param func: input function
    :param x: list of x
    :param y: list of y
    :param eps: epsilon
    :type func: Callable
    :type x: list
    :type y: list
    :type eps: float
    :return: best params a and b to minimize function
    :rtype: tuple
    """

    a_cur = 100
    b_cur = 100
    a_next = 0
    b_next = 0

    a_vars = np.linspace(-1, 1, 100)
    b_vars = np.linspace(-1, 1, 100)

    while (abs(a_next - a_cur) > eps) or (abs(b_next - b_cur) > eps) or (
            abs(errors_func(func, x, a_next, b_next, y) - errors_func(func, x, a_cur, b_cur, y)) > eps):
        a_cur = a_next
        b_cur = b_next

        results = []
        for a_i in a_vars:
            results.append(errors_func(func, x, a_i, b_cur, y))

        a_next = a_vars[np.argmin(results)]

        results = []
        for b_i in b_vars:
            results.append(errors_func(func, x, a_next, b_i, y))

        b_next = b_vars[np.argmin(results)]

    return a_next, b_next


def nelder_mead(func: Callable, eps: float = 0.001) -> (float, float):
    """
    Function of Nelder Mead's method to find minimum of input function

    :param func: input function
    :param eps: epsilon
    :type func: Callable
    :type eps: float
    :return: best params a and b to minimize function
    :rtype: tuple
    """

    best = minimize(func, x0=[0, 0], method='Nelder-Mead', tol=eps)
    return best['x'][0], best['x'][1]


# Creating of data
alpha = random.uniform(0.00000001, 0.99999999)
beta = random.uniform(0.00000001, 0.99999999)

x_list = [k / 100 for k in range(101)]
y_list = [alpha * xk + beta + random.normalvariate(0, 1) for xk in x_list]


def errors_func_lin_neldler(params: list) -> np.array:
    """
    Function for finding errors function of linear function for Nelder Mead's method (because of specific of scipy)

    :param params: params a and b
    :type params: list
    :return: errors sum
    :rtype: np.array
    """

    a = params[0]
    b = params[1]
    return np.sum((a * np.array(x_list) + b - np.array(y_list)) ** 2)


def errors_func_ration_neldler(params):
    """
    Function for finding errors function of rational function for Nelder Mead's method (because of specific of scipy)

    :param params: params a and b
    :type params: list
    :return: errors sum
    :rtype: np.array
    """

    a = params[0]
    b = params[1]
    return np.sum(((a / (1 + b * np.array(x_list))) - np.array(y_list)) ** 2)


# Plot results for linear function
plt.scatter(x_list, y_list, color='orange', s=10)
plt.plot(x_list, lin_func(x_list, *enumeration(lin_func, x_list, y_list)), label="enumeration lin")
plt.plot(x_list, lin_func(x_list, *gauss(lin_func, x_list, y_list)), label="gauss lin")
plt.plot(x_list, lin_func(x_list, *nelder_mead(errors_func_lin_neldler)), label="nelder_mead lin")
plt.legend()
plt.show()


# Plot results for rational function
plt.scatter(x_list, y_list, color='orange', s=10)
plt.plot(x_list, ration_func(x_list, *enumeration(ration_func, x_list, y_list)), label="enumeration ration")
plt.plot(x_list, ration_func(x_list, *gauss(ration_func, x_list, y_list)), label="gauss ration")
plt.plot(x_list, ration_func(x_list, *nelder_mead(errors_func_ration_neldler)), label="nelder_mead ration")
plt.legend()
plt.show()
