import random
import math
import matplotlib.pyplot as plt
from typing import Callable


def enumeration(func: Callable, xmin: int, xmax: int, epsilon: float) -> (int, int, int):
    """
    Function for finding the minimum of the function by exhaustive search method

    :param func: function
    :param xmin: min x
    :param xmax: max x
    :param epsilon: epsilon
    :type func: Callable
    :type xmin: int
    :type xmax: int
    :type epsilon: float
    :return: x: f(x) -> min, number of iterations, number of function count
    :rtype: tuple
    """

    n = int((xmax - xmin) / epsilon)

    x = [(xmin + i * (xmax - xmin) / n) for i in range(n + 1)]

    y = [func(x[i]) for i in range(len(x))]

    return x[y.index(min(y))], n, n


def dichotomy(func: Callable, xmin: int, xmax: int, epsilon: float) -> (int, int, int):
    """
    Function for finding the minimum of the function by dichotomy method

    :param func: function
    :param xmin: min x
    :param xmax: max x
    :param epsilon: epsilon
    :type func: Callable
    :type xmin: int
    :type xmax: int
    :type epsilon: float
    :return: x: f(x) -> min, number of iterations, number of function count
    :rtype: tuple
    """

    a = xmin
    b = xmax
    delta = random.uniform(0, epsilon)
    iterations = 0
    func_count = 0

    while abs(b - a) >= epsilon:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        if func(x1) < func(x2):
            b = x2
        else:
            a = x1

        iterations += 1
        func_count += 2

    if func(a) < func(b):
        return a, iterations, func_count

    return b, iterations, func_count


def golden_section(func: Callable, xmin: int, xmax: int, epsilon: float) -> (int, int, int):
    """
    Function for finding the minimum of the function by golden section method

    :param func: function
    :param xmin: min x
    :param xmax: max x
    :param epsilon: epsilon
    :type func: Callable
    :type xmin: int
    :type xmax: int
    :type epsilon: float
    :return: x: f(x) -> min, number of iterations, number of function count
    :rtype: tuple
    """

    a = xmin
    b = xmax

    x1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
    x2 = b + (math.sqrt(5) - 3) / 2 * (b - a)

    if func(x1) < func(x2):
        b = x2
        x2 = x1
        x1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
    else:
        a = x1
        x1 = x2
        x2 = b + (math.sqrt(5) - 3) / 2 * (b - a)

    iterations = 1
    func_count = 2

    while abs(b - a) >= epsilon:

        if func(x1) < func(x2):
            b = x2
            x2 = x1
            x1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = b + (math.sqrt(5) - 3) / 2 * (b - a)

        iterations += 1
        func_count += 1

    if func(a) < func(b):
        return a, iterations, func_count

    return b, iterations, func_count


def cube(x: float) -> float:
    """
    Function returns x^3

    :param x: param
    :type x: float
    :return: x^3
    :rtype: float
    """

    return x ** 3


def absolute(x: float) -> float:
    """
    Function returns |x - 0.2|

    :param x: param
    :type x: float
    :return: |x - 0.2|
    :rtype: float
    """

    return abs(x - 0.2)


def sinus(x: float) -> float:
    """
    Function returns x * sin(1 / x)
    :param x: param
    :type x: float
    :return: x * sin(1 / x)
    :rtype: float
    """

    return x * math.sin(1 / x)


# Finding of all params for all methods
cube_min_enumeration, cube_iters_enumeration, cube_func_count_enumeration = enumeration(cube, 0, 1, 0.001)
absolute_min_enumeration, absolute_iters_enumeration, absolute_func_count_enumeration = enumeration(absolute, 0, 1,
                                                                                                    0.001)
sinus_min_enumeration, sinus_iters_enumeration, sinus_func_count_enumeration = enumeration(sinus, 0.01, 1, 0.001)

cube_min_dichotomy, cube_iters_dichotomy, cube_func_count_dichotomy = dichotomy(cube, 0, 1, 0.001)
absolute_min_dichotomy, absolute_iters_dichotomy, absolute_func_count_dichotomy = dichotomy(absolute, 0, 1, 0.001)
sinus_min_dichotomy, sinus_iters_dichotomy, sinus_func_count_dichotomy = dichotomy(sinus, 0.01, 1, 0.001)

cube_min_golden, cube_iters_golden, cube_func_count_golden = golden_section(cube, 0, 1, 0.001)
absolute_min_golden, absolute_iters_golden, absolute_func_count_golden = golden_section(absolute, 0, 1, 0.001)
sinus_min_golden, sinus_iters_golden, sinus_func_count_golden = golden_section(sinus, 0.01, 1, 0.001)

# Printing results
print("Enumeration")
print("x: f(x) = x^3 -> min = {}\niterations_count = {}\nfunc_count = {}\n".format(cube_min_enumeration,
                                                                                   cube_iters_enumeration,
                                                                                   cube_func_count_enumeration))
print("x: f(x) = |x - 0.2| -> min = {}\niterations_count = {}\nfunc_count = {}\n".format(absolute_min_enumeration,
                                                                                         absolute_iters_enumeration,
                                                                                         absolute_func_count_enumeration))
print("x: f(x) = x * sin(1/x) = {}\niterations_count = {}\nfunc_count = {}\n".format(sinus_min_enumeration,
                                                                                     sinus_iters_enumeration,
                                                                                     sinus_func_count_enumeration))

print("Dichotomy")
print("x: f(x) = x^3 -> min = {}\niterations_count = {}\nfunc_count = {}\n".format(cube_min_dichotomy,
                                                                                   cube_iters_dichotomy,
                                                                                   cube_func_count_dichotomy))
print("x: f(x) = |x - 0.2| -> min = {}\niterations_count = {}\nfunc_count = {}\n".format(absolute_min_dichotomy,
                                                                                         absolute_iters_dichotomy,
                                                                                         absolute_func_count_dichotomy))
print("x: f(x) = x * sin(1/x) = {}\niterations_count = {}\nfunc_count = {}\n".format(sinus_min_dichotomy,
                                                                                     sinus_iters_dichotomy,
                                                                                     sinus_func_count_dichotomy))

print("Golden section")
print("x: f(x) = x^3 -> min = {}\niterations_count = {}\nfunc_count = {}\n".format(cube_min_golden,
                                                                                   cube_iters_golden,
                                                                                   cube_func_count_golden))
print("x: f(x) = |x - 0.2| -> min = {}\niterations_count = {}\nfunc_count = {}\n".format(absolute_min_golden,
                                                                                         absolute_iters_golden,
                                                                                         absolute_func_count_golden))
print("x: f(x) = x * sin(1/x) = {}\niterations_count = {}\nfunc_count = {}\n".format(sinus_min_golden,
                                                                                     sinus_iters_golden,
                                                                                     sinus_func_count_golden))

# Plot nums of iterations
plt.bar([0.05, 0.15, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95], [cube_iters_enumeration, cube_iters_dichotomy,
                                                              cube_iters_golden, absolute_iters_enumeration,
                                                              absolute_iters_dichotomy, absolute_iters_golden,
                                                              sinus_iters_enumeration, sinus_iters_dichotomy,
                                                              sinus_iters_golden], width=0.1,
        color=['#C6D8FF', '#FED6BC', '#C3FBD8'], edgecolor='#000', linewidth=0.5,
        tick_label=['x^3 enumer', 'x^3 dichotomy', 'x^3 golden', '|x - 0.2| enumer', '|x - 0.2| dichotomy',
                    '|x - 0.2| golden', 'x*sin(1/x) enumer', 'x*sin(1/x) dichotomy', 'x*sin(1/x) golden'])
plt.title("Iterations comparison")
plt.xticks(rotation=45)
plt.show()

# Plot function count
plt.bar([0.05, 0.15, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95], [cube_func_count_enumeration, cube_func_count_dichotomy,
                                                              cube_func_count_golden, absolute_func_count_enumeration,
                                                              absolute_func_count_dichotomy, absolute_func_count_golden,
                                                              sinus_func_count_enumeration, sinus_func_count_dichotomy,
                                                              sinus_func_count_golden], width=0.1,
        color=['#C6D8FF', '#FED6BC', '#C3FBD8'], edgecolor='#000', linewidth=0.5,
        tick_label=['x^3 enumer', 'x^3 dichotomy', 'x^3 golden', '|x - 0.2| enumer', '|x - 0.2| dichotomy',
                    '|x - 0.2| golden', 'x*sin(1/x) enumer', 'x*sin(1/x) dichotomy', 'x*sin(1/x) golden'])
plt.title("Functions count comparison")
plt.xticks(rotation=45)
plt.show()
