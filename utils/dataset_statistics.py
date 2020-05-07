import torch
import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

import utils.image_loader as il


def count_statistics():
    loader = il.DatasetLoader.initial()
    all_data = il.prepare_data(loader.load_tensors(None, None))
    pows = torch.zeros(5)
    for i in range(5):
        pows[i] = 2 ** i

    stat_dict = {}
    for _, _, l in all_data:
        idx = int((l * pows).sum().data)
        stat_dict.setdefault(idx, 0)
        stat_dict[idx] += 1

    for i in sorted(stat_dict.keys()):
        print("".join(reversed(format(i, 'b').zfill(5))), stat_dict[i])


if __name__ == "__main__":
    count_statistics()


"""
00000 514
10000 15
01000 36
00100 229
01100 16
00010 160
10010 16
01010 18
11010 4
00110 44
10110 3
01110 16
00001 817
10001 24
01001 52
00101 259
10101 5
01101 24
00011 206
10011 31
01011 17
11011 2
00111 81
01111 5
^     -- первое
 ^    -- второе
  ^   -- третье
   ^  -- четвертое
    ^ -- пятое
заболевания по классам
"""
