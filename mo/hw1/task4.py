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


def get_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def straignh(f):
    x_prev = np.random.uniform(low=-100, high=100)
    delta = 0.1

    if f(x_prev) > f(x_prev + delta):
        x_cur = x_prev + delta
        h = delta
        # elif f(x_prev) > f(x_prev - delta):
    else:
        x_cur = x_prev - delta
        h = -delta

    while True:
        h *= 2
        x_next = x_cur + h

        if f(x_cur) > f(x_next):
            x_prev = x_cur
            x_cur = x_next
        else:
            return x_prev, x_next


def logistic_q(X, y, lmd, w):
    w_range = range(len(w))
    objects_num = X.shape[0]

    alpha = 0
    for i in range(objects_num):
        y_cur = y[i]

        sum_ = 0
        for j in w_range:
            x_cur = X[i][j]
            w_cur = w[j]
            sum_ += x_cur * w_cur

        alpha += np.logaddexp(0, -y_cur * sum_)

    sum_ = 0
    for i in w_range:
        sum_ += w[i] ** 2
    reg = lmd * sum_ / 2

    return alpha + reg


def construct_f(X, y, lmd, w, h):
    w_range = range(len(w))
    arg = w.copy()
    objects_num, _ = X.shape

    for i in range(objects_num):
        y_cur = y[i]

        sum_ = 0
        for j in w_range:
            x_cur = X[i][j]
            w_cur = w[j]
            sum_ += x_cur * w_cur

        alpha = expit(-y_cur * sum_)
        for j in w_range:
            x_cur = X[i][j]
            w_cur = w[j]
            arg[j] -= h * (lmd * w_cur + (y_cur * x_cur * w_cur) * (alpha - 1))

    return logistic_q(X, y, lmd, arg)


def logistic_gradient(X, y, lmd, w_begin, eps, max_iter_num):
    objects_num, features_num = X.shape
    w_cur = w_begin
    iter_num = 1

    while True:
        f = partial(construct_f, X, y, lmd, w_cur)
        a0, b0 = straignh(f)
        a, _, _ = t1.dichotomy(f, a0, b0, 0.01, 100)
        h = a[-1]

        w_next = w_cur.copy()

        for i in range(objects_num):
            y_cur = y[i]
            sum_ = 0
            for j in range(features_num):
                x_cur = X[i][j]
                w_cur_ = w_cur[j]
                sum_ += x_cur * w_cur_
            alpha = expit(-y_cur * sum_)
            for j in range(features_num):
                x_cur = X[i][j]
                w_cur_ = w_cur[j]
                w_next[j] -= h * (lmd * w_cur_ + (y_cur * x_cur * w_cur_) * (alpha - 1))

        if abs(logistic_q(X, y, lmd, w_next) - logistic_q(X, y, lmd, w_cur)) < eps or iter_num == max_iter_num:
            return w_next

        w_cur = w_next.copy()
        iter_num += 1


def get_classifier(w, dec_lim):
    def predict(x):
        sum_ = -dec_lim
        for i in range(len(w)):
            sum_ += w[i] * x[i]
        return 1 if sum_ > 0 else 0

    return predict
