from math import ceil, floor

import numpy as np
import scipy.optimize as opt


def split(i, m, A, b, b0, t) -> tuple:
    a0 = np.zeros(m)
    a0[i] = t
    new_A = np.append(A, [a0], axis=0)
    new_b = np.append(b, b0)
    return new_A, new_b


def branch_solver(n, m, A, b, c):
    res = opt.linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
    print("vector is {}".format(res.x))
    b_res = res.x
    answer = c @ b_res

    if len(list(filter(lambda x: abs(x - int(x)) < 0.000000001, b_res))) == len(b_res):
        return answer, b_res

    for i in range(m):
        if not abs(b_res[i] - int(b_res[i])) < 0.000000001:
            right_A, right_b = split(i, m, A, b, floor(b_res[i]), 1)
            left_A, left_b = split(i, m, A, b, -ceil(b_res[i]), -1)

            o1 = opt.linprog(c=c, A_ub=right_A, b_ub=right_b, bounds=(0, None))
            x1 = o1.x
            o2 = opt.linprog(c=c, A_ub=left_A, b_ub=left_b, bounds=(0, None))
            x2 = o2.x

            if o1.success is False and o2.success is False:
                return None, None

            s1 = c @ x1
            s2 = c @ x2

            print('Split at index = {}'.format(i))
            print('Max at right = {}'.format(s1))
            print('Max at left = {}'.format(s2))
            if (o1.success and o2.success and s1 <= s2) or (o2.success == False):
                print('Move from {} -> {} (right)'.format(b, right_b))
                answer = branch_solver(n + 1, m, right_A, right_b, c)
            elif o2.success:
                print('Move from {} -> {} (left)'.format(b, left_b))
                answer = branch_solver(n + 1, m, left_A, left_b, c)
            break
    return answer


if __name__ == "__main__":

    n, m = 3, 5
    A = np.array([[1, 2, -1, 2, 4],
                  [0, -1, 2, 1, 3],
                  [1, -3, 2, 2, 0]], dtype='float')
    b = [1, 3, 4]
    c = [1, -3, 2, 1, 4]

    print(branch_solver(n, m, A, b, c))

    print("=" * 50)
    n, m = 3, 5
    A = np.array([[-1, 3, 0, 2, 1],
                  [2, -1, 1, 2, 3],
                  [1, -1, 2, 1, 0]], dtype='float')
    b = [1, 2, 4]
    c = [-1, -3, 2, 1, 4]

    print(branch_solver(n, m, A, b, c))
    print("=" * 50)
    n, m = 2, 2
    A = np.array([[1, 8],[0, 9]], dtype='float')
    b = [5, 8]
    c = [1, 16]

    print(branch_solver(n, m, A, b, c))
