import mo.hw1.task1 as t1
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


# %% md

### 2. Реализуйте метод градиентного спуска и процедуру линейного поиска. Оцените, как меняется скорость сходимости, если для поиска величины шага использовать различные методы одномерного поиска.

# %% md

#### 2.1 Реализация процедуры линейного поиска:

# %%

# для минимизации альфы на к-ом шаге
# т.е. условный наискорейший градиентный спуск
def line_search(f, x_cur, df_x):
    grad = []
    for i in range(0, len(df_x)):
        grad.append(df_x[i](*x_cur))
    alpha = 0.5
    beta = 0.9
    stp = 1.0
    grad = np.array(grad)
    p = np.dot(grad, grad)
    # формула 9, стр 4 условие армихо
    while (f(*x_cur) - (f(*(x_cur - stp * np.array(grad))) + alpha * stp * p)) < 0:
        stp *= beta
    return stp


# %% md

#### 2.2 Реализация градиентного спуска:

# %%

def to_step_arg(f, x, x_prime, lmd):
    args = []
    dim = len(x)
    for i in range(dim):
        args.append(x[i] - lmd * x_prime[i](*x))
    return f(*args)


def get_symbol(dim):
    return list(map(Symbol, {
        1: ['x'],
        2: ['x', 'y'],
        3: ['x', 'y', 'z']
    }.get(dim, ['x' + str(i) for i in range(1, dim + 1)])))


def gradient_descent(f, *x, eps, max_iter_num, step_f):
    dim = len(x)
    symbol = get_symbol(dim)

    x_prime = [lambdify(symbol, f(*symbol).diff(smb)) for smb in symbol]

    x_cur = [*x]
    xs = [x_cur]
    iter_num = 0
    while True:
        if step_f is line_search:
            step = step_f(f, np.array(x_cur), x_prime)
        else:
            step = step_f(lambda lmd: to_step_arg(f, x_cur, x_prime, lmd))

        x_next = [0] * dim
        for i in range(dim):
            x_next[i] = x_cur[i] - step * x_prime[i](*x_cur)
        xs.append(x_next)

        if abs(f(*x_next) - f(*x_cur)) < eps or iter_num == max_iter_num:
            return np.array(xs), iter_num

        x_cur = x_next.copy()
        iter_num += 1


