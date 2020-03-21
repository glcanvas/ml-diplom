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
import mo.hw1.task2 as t2
import mo.hw1.task1 as t1

a0, b0 = -100, 100
eps = 1e-3
max_iter_num = 100


def get_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def rosenbrock(x, y):
    return 100 * ((y - x ** 2) ** 2) + ((1 - x) ** 2)


def to_step_f(one_dimensional_method, f):
    a, _, _ = one_dimensional_method(f, a0, b0, 1e-5, 100)
    return a[-1]


methods = t1.one_dimensional_methods + [t2.line_search]


def draw_contour(func):
    bds_x = [-12, 12]
    bds_y = [-5, 15]

    nx = np.linspace(*bds_x, 500)
    ny = np.linspace(*bds_y, 500)

    xs, ys = np.meshgrid(nx, ny)

    zs = func(xs, ys)

    plt.subplots(figsize=(12, 21))
    plt.contour(xs, ys, zs, levels=500, alpha=0.5, cmap='summer')

    y_min_idx, x_min_idx = np.unravel_index(zs.argmin(), zs.shape)
    min_p = (nx[x_min_idx], ny[y_min_idx])
    plt.plot(min_p[0], min_p[1], 'o')
    plt.annotate('min', min_p)

    x0, y0 = [2, 5, 8, 11], [3, 5, 7, 9]

    for i, method in enumerate(methods):
        step_f = partial(to_step_f, method) if method != t2.line_search else method
        grad_pts, iter_num = t2.gradient_descent(func, x0[i], y0[i], eps=1e-5, max_iter_num=50, step_f=step_f)
        x_grads, y_grads = grad_pts.T
        name = get_name(method)
        plt.plot(x_grads, y_grads, label=name)
        print(f"found min={grad_pts[-1]} by {name}, iterations={iter_num}")

    plt.axis(bds_x + bds_y)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(f"{get_name(func)} contour with gradient steps")
    plt.legend()
    plt.show()

    print(f'real min={min_p}')
