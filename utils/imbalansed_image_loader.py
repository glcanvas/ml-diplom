import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

import scipy.optimize as opt
from utils import property as P
import torch
import numpy as np
import utils.image_loader as il


def calculate_stat(lst):
    acc = torch.zeros(5)
    for _, _, k in lst:
        acc += k
    return acc


def balance_dataset(data_set, data_size, marked_size):
    pows = torch.zeros(5)
    for i in range(5):
        pows[i] = 2 ** i

    train_split = {}
    mask_dict = {}
    for a, b, k in data_set:
        idx = int((k * pows).sum().data)
        mask_dict.setdefault(idx, 0)
        mask_dict[idx] += 1
        train_split.setdefault(idx, [])
        train_split[idx].append((a, b, k))

    mask_dict_keys = sorted(mask_dict.keys())
    for i in mask_dict_keys:
        P.write_to_log("".join(reversed(format(i, 'b').zfill(5))), mask_dict[i])

    sm_list = [0, 0, 0, 0, 0]
    for k, v in mask_dict.items():
        key_ = "".join(reversed(format(k, 'b').zfill(5)))
        for ill in range(5):
            exists = 1 if key_[ill] == '1' else 0
            sm_list[ill] += v * exists
    print(sm_list)

    A = [[], [], [], [], [], []]
    for key_idx, key in enumerate(mask_dict_keys):
        key_ = "".join(reversed(format(key, 'b').zfill(5)))
        for ill in range(5):
            exists = 1 if key_[ill] == '1' else 0
            A[ill].append(mask_dict[key] * exists)
        A[5].append(mask_dict[key])

    b = []
    for i in range(5):
        b.append(marked_size)
    b.append(data_size)

    bounds = [(mask_dict[mdk] / data_size, None) for mdk in mask_dict_keys]

    c = [-mask_dict[i] for i in mask_dict_keys]
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    P.write_to_log(A)
    P.write_to_log(b)
    P.write_to_log(c)

    res = opt.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='simplex')
    P.write_to_log(res)
    accumulate_dict = {}
    for idx, i in enumerate(res.x):
        P.write_to_log("{} {:.5f} {} {:.5f} ".format("".join(reversed(format(mask_dict_keys[idx], 'b').zfill(5))),
                                                     i, mask_dict[mask_dict_keys[idx]],
                                                     mask_dict[mask_dict_keys[idx]] * i))
        accumulate_dict[mask_dict_keys[idx]] = mask_dict[mask_dict_keys[idx]] * i + 1
    sm = 0
    sm_list = [0, 0, 0, 0, 0]
    for x, k in zip(res.x, mask_dict_keys):
        sm += x * mask_dict[k]
        key_ = "".join(reversed(format(k, 'b').zfill(5)))
        for ill in range(5):
            exists = 1 if key_[ill] == '1' else 0
            sm_list[ill] += x * mask_dict[k] * exists
    P.write_to_log(sm_list)
    P.write_to_log(sm)

    result_list = []

    for idx, lst in train_split.items():
        needed_cnt = int(accumulate_dict[idx])
        for i in range(needed_cnt):
            result_list.append(lst[i % len(lst)])

    return result_list


def load_balanced_dataset(train_size: int, seed: int, image_size: int):
    train_set, test_set = il.load_data(train_size, seed, image_size)

    train_set = balance_dataset(train_set, train_size, train_size // 2)
    test_set = balance_dataset(test_set, len(test_set), len(test_set) // 2)

    P.write_to_log("========")
    P.write_to_log("balanced TRAIN size: ", calculate_stat(train_set), " full size: ", len(train_set))
    P.write_to_log("balanced TEST size: ", calculate_stat(test_set), " full size: ", len(test_set))
    return train_set, test_set


if __name__ == "__main__":
    # 433, 198, 389
    res1, _ = load_balanced_dataset(100, 389, 224)
    print(res1)