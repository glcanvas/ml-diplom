import torch
import sys
import cv2

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

import utils.image_loader as il
import utils.property as p

def count_statistics():
    loader = il.DatasetLoader.initial()
    image_paths, tensors = loader.load_tensors(None, None, load_extend=True)
    tensors = il.prepare_data(tensors)

    for i in image_paths:
        i['input'] = i['input'].replace("/cached", "").replace(".torch", ".jpg")
        for j in p.labels_attributes:
            i[j] = i[j].replace("/cached", "").replace(".torch", ".png")

    for paths, (_, _, labels) in zip(image_paths, tensors):
        pass
    """
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
    """


a = """00000 514
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
01111 5"""
if __name__ == "__main__":
    #count_statistics()
    indexes = a.split()[::2]
    values = a.split()[1::2]
    print(indexes)
    print(values)

    sum_list = [0 for _ in range(5)]
    for i, j in zip(indexes, values):
        for idx, k in enumerate(i):
            if k == '1':
                sum_list[idx] += int(j)

    print(sum_list)

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


set size: 2594, set by classes: tensor([ 100.,  190.,  682.,  603., 1523.])
TEST set size: 794, test set by classes: tensor([ 34.,  64., 215., 174., 475.])
TRAIN set size: 1800, train set by classes: tensor([  66.,  126.,  467.,  429., 1048.])   

сначала увеличивать резмерность 1ого класса (до 2000 элементов), затем 2ого до 2000 элементов
основываться буду на мапе выше 
"""
