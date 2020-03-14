# %%

# градиентный спуск
# x_k+1 = x_k - a * f'(x_k)
import math

f = lambda x: x * x - 4
f_grad = lambda x: 2 * x

EPS = 1e-10


def simple_grad_descent(x_k, alpha, step):
    x_k_1 = x_k - alpha * f(x_k)
    st = 0
    while st < step and math.fabs(f(x_k) - f(x_k_1)) > EPS:
        x_k = x_k_1
        x_k_1 = x_k - alpha * f(x_k)
        st += 1
    return x_k, st


def grad_descent_with_div(x_k, alpha, div, step):
    x_k_1 = x_k - alpha * f(x_k)
    st = 0
    while st < step and math.fabs(f(x_k) - f(x_k_1)) > EPS:
        x_k = x_k_1
        x_k_1 = x_k - alpha * f(x_k)
        st += 1
        alpha /= div
    return x_k, st


def fast_gradient_descent(x_k, alpha, step):
    l_k = f(x_k - alpha * f_grad(x_k))
    x_k_1 = x_k - l_k * f(x_k)
    st = 0
    while st < step and math.fabs(f(x_k) - f(x_k_1)) > EPS:
        x_k = x_k_1
        x_k_1 = x_k - alpha * f(x_k)
        st += 1
        # alph# /= div
    return x_k, st


import numpy as np

if __name__ == "__main__":
    print(np.random.uniform(1, 99))
