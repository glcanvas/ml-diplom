from functools import partial
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sympy import *


def get_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def one_dimensional(f, a0, b0, eps, max_iter_num, x1_f, x2_f):
    """
    абстракция поверх одномерного поиска, возвращает массив сближений для левой и правой границы и кол-во итерации
    :param f:  функция, которую исследуем
    :param a0: левая граница
    :param b0: правая граница
    :param eps: епсилон
    :param max_iter_num: максимальное кол-во итераций
    :param x1_f: функция которая выбирает левую границу
    :param x2_f: функция которая выбирает правую границу
    :return: возвращает массив сближений для левой и правой границы и кол-во итерации
    """
    a = [a0]
    b = [b0]
    iter_num = 1
    while abs(f(b[-1]) - f(a[-1])) > eps and iter_num < max_iter_num:
        x1 = x1_f(a[-1], b[-1], iter_num)
        x2 = x2_f(a[-1], b[-1], iter_num)
        if f(x1) < f(x2):
            a.append(a[-1])
            b.append(x2)
        elif f(x1) > f(x2):
            a.append(x1)
            b.append(b[-1])
        else:
            break
        iter_num += 1
    return a, b, iter_num


def dichotomy(f, a0, b0, eps, max_iter_num):
    """
    x1 = (a + b) / 2 - delta
    x2 = (a + b) / 2 + delta
    :param f:
    :param a0:
    :param b0:
    :param eps:
    :param max_iter_num:
    :return:
    """
    delta = eps / 3  # Should be less than eps / 2
    x1_f = lambda a, b, _: (a + b) / 2 - delta
    x2_f = lambda a, b, _: (a + b) / 2 + delta
    return one_dimensional(f, a0, b0, eps, max_iter_num, x1_f, x2_f)


def gold_section(*args):
    phi = (1 + np.sqrt(5)) / 2
    x1_f = lambda a, b, _: b - (b - a) / phi
    x2_f = lambda a, b, _: a + (b - a) / phi
    return one_dimensional(*args, x1_f, x2_f)


def fibonacci(f, a0, b0, eps, max_iter_num):
    fibs = [1, 1, 2]

    def add_fib():
        fibs.append(fibs[-1] + fibs[-2])

    n = 0
    while (b0 - a0) >= fibs[n + 2] * eps:
        n += 1
        add_fib()

    fib_np2 = fibs[n + 2]

    x1_f = lambda a, _, iter_num: a + (b0 - a0) * fibs[n - iter_num + 1] / fib_np2
    x2_f = lambda a, _, iter_num: a + (b0 - a0) * fibs[n - iter_num + 2] / fib_np2
    return one_dimensional(f, a0, b0, eps, max_iter_num, x1_f, x2_f)


one_dimensional_methods = [dichotomy, gold_section, fibonacci]


#  1.2 Тестируемая функция и действительное значение минимума:
def parabola(x):
    return 1.5 * x ** 2 - 5 * x + 15.5


# 1.5 Изменение отрезка при переходе к следующему интервалу по номеру итерации
def interval_relationship(a0, b0, eps, max_iter_num):
    def inner(odm):
        a, b, iter_num = odm(parabola, a0, b0, eps, max_iter_num)
        return [(b[i] - a[i]) / (b[i - 1] - a[i - 1]) for i in range(1, iter_num)]

    return inner


def draw_initial_function():
    begin = -100.0
    end = 100.0

    xs = np.linspace(begin, end, 10000)
    ys = parabola(xs)
    plt.plot(xs, ys)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('$f(x) = 1.5x^2 - 5 * x + 15.5$')
    plt.show()
    print(f'x_min = {xs[ys.argmin()]}')


def draw_iteration_accuracy():
    # 1.3 Графики зависимости количества итераций от точности:
    begin = -100.0
    end = 100.0

    a0, b0 = begin, end
    eps_range = np.arange(0.01, 0.5, 0.01)
    max_iter_num = 100
    print(eps_range)
    for odm in one_dimensional_methods:
        iter_nums = []
        for eps in eps_range:
            _, _, iter_num = odm(parabola, a0, b0, eps, max_iter_num)
            iter_nums.append(iter_num)
        plt.plot(eps_range, iter_nums)
        plt.title(get_name(odm))
        plt.xlabel('$\epsilon$')
        plt.ylabel('Iterations number')
        plt.show()


def draw_interval_by_iteration_number():
    # 1.4 Графики интервалов по номеру итерации:
    begin = -100.0
    end = 100.0

    a0, b0 = begin, end
    eps = 1e-5
    max_iter_num = 100
    for odm in one_dimensional_methods:
        a, b, iter_num = odm(parabola, a0, b0, eps, max_iter_num)
        iters = range(0, iter_num)
        plt.step(iters, a, label='a border')
        plt.step(iters, b, label='b border')
        plt.fill_between(iters, a, b, step='pre', alpha=0.1)
        plt.title(get_name(odm))
        plt.xlabel('Iteration number')
        plt.ylabel('$x$')
        plt.legend()
        plt.show()

