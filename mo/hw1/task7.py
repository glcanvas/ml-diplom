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
import mo.hw1.task1 as t1
import mo.hw1.task2 as t2
import mo.hw1.task6 as t6


def get_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def create_matrix(condition_number, n):
    r = sqrt(condition_number)
    A = np.random.randn(n, n)
    u, s, v = np.linalg.svd(A)
    h, l = np.max(s), np.min(s)

    def f(x):
        return h * (1 - ((r - 1) / r) / (h - l) * (h - x))

    new_s = f(s)
    new_A = (u * new_s) @ v.T
    new_A = new_A @ new_A.T

    return new_A


def number_of_iters(cond, n_vars, step_chooser=t1.dichotomy, n_checks=10):
    all_iters = 0
    for _ in range(n_checks):
        A = create_matrix(cond, n_vars)
        b = np.random.randn(n_vars)
        init_x = np.random.randn(n_vars)

        def func(*args):
            x = np.array(args)
            return x.dot(A).dot(x) - b.dot(x)

        _, iter_num = t2.gradient_descent(func, *init_x, eps=1e-3, max_iter_num=20,
                                          step_f=partial(t6.to_step_f, step_chooser))

        all_iters += iter_num
    return all_iters / n_checks


# %%

def draw_condition():
    n_vars = list(range(2, 6))
    condition_numbers = np.linspace(1, 10, 5)
    plt.figure()
    for n in n_vars:
        iter_numbers = [number_of_iters(cond, n) for cond in condition_numbers]
        plt.plot(condition_numbers, iter_numbers, label=f'n={n}')

    plt.xlabel('$\mu$')
    plt.ylabel('$T(n, k)$')
    plt.legend()
    plt.show()
