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


def get_symbol(dim):
    return list(map(Symbol, {
        1: ['x'],
        2: ['x', 'y'],
        3: ['x', 'y', 'z']
    }.get(dim, ['x' + str(i) for i in range(1, dim + 1)])))


# %%

### 5. Реализуйте метод Ньютона.

# %%

#### 5.1 Реализация метода Ньютона:

# %%

def newton_descent(f, *x, eps, max_iter_num):
    dim = len(x)
    symbol = get_symbol(dim)

    x_prime = [lambdify(symbol, f(*symbol).diff(smb)) for smb in symbol]
    x_y_prime = [[] for _ in range(dim)]
    for i in range(dim):
        for smb in symbol:
            x_y_prime[i].append(lambdify(symbol, x_prime[i](*symbol).diff(smb)))

    x_cur = [arg for arg in x]
    iter_num = 0
    while True:
        step = 1  # Constant step

        A = [[0 for _ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                A[i][j] = x_y_prime[i][j](*x_cur)

        b = [0 for _ in range(dim)]
        for i in range(dim):
            b[i] = x_prime[i](*x_cur)

        c, low = cho_factor(A)
        d = cho_solve((c, low), b)

        x_next = [0] * dim
        for i in range(dim):
            x_next[i] = x_cur[i] - step * d[i]

        if abs(f(*x_next) - f(*x_cur)) < eps or iter_num == max_iter_num:
            return x_next, iter_num

        x_cur = x_next.copy()
        iter_num += 1

# %%

#### 5.2 Тестируемая функция из пункта **2.2**. Точка минимума функции согласно методу Ньютона:
