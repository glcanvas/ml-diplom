import torch
import sys
import cv2
import albumentations as A
import numpy as np

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

import utils.image_loader as il
import utils.property as p

repeat_list = """10000 20
01000 15
00100 6
01100 15
10010 20
01010 15
11010 20
10110 20
01110 15
10001 20
01001 15
10101 20
10011 20
11011 20"""

IMG_AUG = A.Compose([A.GaussNoise(), A.RandomRotate90(), A.Blur(), A.RandomResizedCrop(767, 1022)])


def aug_dataset():
    loader = il.DatasetLoader.initial()
    image_paths, tensors = loader.load_tensors(None, None, load_extend=True)
    tensors = il.prepare_data(tensors)

    indexes = list(map(lambda x: int(x, 2), repeat_list.split()[::2]))
    values = repeat_list.split()[1::2]

    for i in image_paths:
        i['input'] = i['input'].replace("/cached", "").replace(".torch", ".jpg")
        for j in p.labels_attributes:
            i[j] = i[j].replace("/cached", "").replace(".torch", ".png")

    pows = torch.zeros(5)
    for i in range(5):
        pows[i] = i ** 2

    for paths, (_, _, label) in zip(image_paths, tensors):
        idx = int((pows * label).sum().data)
        if idx not in indexes:
            continue
        input = cv2.imread(paths['input'])
        masks = []
        for j in p.labels_attributes:
            masks.append(cv2.imread(paths[j], cv2.IMREAD_GRAYSCALE))
        masks = np.array(masks).reshape((767, 1022, 5))
        print("ok")
        #IMG_AUG


if __name__ == "__main__":
    aug_dataset()

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


Добрый день!
подскажите, пожалуйста, как можно быстро посчитать следующую задачу:
есть набор "векторов" 00001, 01010 ... все вектора размера 5
я хочу узнать количество путей и сам путь, которым можно добраться в ячейку [1000][1000][1000][1000][1000]
"ходить" можно только прибавляя описанные выше вектора не ограниченное количество раз.
Первое что приходит в голову это динамика -- но для массива выше она слишком долго будет работать,
есть ли способ решить эту задачу быстрее ? 
"""
