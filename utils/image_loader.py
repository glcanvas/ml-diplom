import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")
import os
import torch
import torch.utils.data.dataset
from torchvision import transforms
from PIL import Image
from utils import property as P
import torch.nn.functional as F
import random


composite = transforms.Compose([
    #  transforms.ToTensor(),
    transforms.Scale((P.image_size, P.image_size)),
    transforms.ToTensor(),
    #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normalization = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DatasetLoader:
    """
    Load ISIC files as images and cache them, or load from cache files
    """

    @staticmethod
    def initial():
        return DatasetLoader(P.data_inputs_path, P.data_labels_path, P.cache_data_inputs_path, P.cache_data_labels_path)

    def __init__(self, input_path: str = None, target_path: str = None, cache_input_path: str = None,
                 cache_target_path: str = None, image_suffixes: list = ['.jpg', '.png', P.cached_extension]):
        self.input_path = input_path
        self.target_path = target_path
        self.image_suffixes = image_suffixes

        # create cached path if not exists
        if cache_input_path is None:
            cache_input_path = os.path.join(input_path, "cached")
            os.makedirs(cache_input_path, exist_ok=True)
            print("created cache input dir: {}".format(cache_input_path))
            P.write_to_log("created cache input dir: {}".format(cache_input_path))
        if cache_target_path is None:
            cache_target_path = os.path.join(target_path, "cached")
            os.makedirs(cache_target_path, exist_ok=True)
            print("created cache target dir: {}".format(cache_target_path))
            P.write_to_log("created cache target dir: {}".format(cache_target_path))
        self.cache_input_path = cache_input_path
        self.cache_target_path = cache_target_path

        self.data = None

    def __get_input_image_list(self, input_path) -> list:
        inputs = self.__get_id_value(input_path)
        inputs.sort(key=lambda x: int(x[0]))
        return inputs

    def __get_target_image_list(self, target_path) -> list:
        targets = self.__get_id_value(target_path)
        targets = list(map(lambda x: (x[0].split(P.attribute)[0], x[0].split(P.attribute)[1], x[1]), targets))
        reduced = dict()
        for identifier, label, path in targets:
            if identifier in reduced:
                reduced[identifier][label] = path
            else:
                reduced[identifier] = {label: path}
                reduced[identifier]['id'] = identifier
        targets = [reduced[i] for i in reduced]
        targets.sort(key=lambda x: int(x['id']))
        return targets

    def __merge_data(self, input_path, target_path) -> list:
        inputs = self.__get_input_image_list(input_path)
        targets = self.__get_target_image_list(target_path)
        print("ok")

        def composite_zips(x):
            inp = x[0]
            tar = x[1]
            tar[P.input_attribute] = inp[1]
            return tar

        return list(map(composite_zips, zip(inputs, targets)))

    def __get_id_value(self, absolute_path: str) -> list:
        arrays = []
        for _, _, f in os.walk(absolute_path):
            for file in f:
                for suffix in self.image_suffixes:
                    if file.endswith(suffix) and file.startswith(P.prefix):
                        identifier = file.split(suffix)[0].split(P.prefix)[1]
                        arrays.append((identifier, os.path.join(absolute_path, file)))
        return arrays

    def save_images_to_tensors(self):
        # print(self.input_path, self.target_path)
        data = self.__merge_data(self.input_path, self.target_path)
        data_len = len(data)
        for idx, dct in enumerate(data):
            for item in P.labels_attributes:
                self.__save_torch(dct, item, self.cache_target_path)
            self.__save_torch(dct, P.input_attribute, self.cache_input_path)
            print("=" * 10)
            print("save: {} of {} elements".format(idx, data_len))
            P.write_to_log("=" * 10)
            P.write_to_log("save: {} of {} elements".format(idx, data_len))
        print("all saved successfully")
        P.write_to_log("all saved successfully")

    def load_tensors(self, lower_bound: int, upper_bound: int, load_first_segments: int = 10 ** 20,
                     image_size: int = 224,
                     use_norm: bool = False, load_extend: bool = False):
        if self.data is None:
            self.data = self.__merge_data(self.cache_input_path, self.cache_target_path)
        data_len = len(self.data)
        print(data_len)
        lower_bound = 0 if lower_bound is None else lower_bound
        upper_bound = data_len if upper_bound is None else upper_bound
        result = []
        data = self.data[lower_bound:upper_bound]
        loads = 0

        for idx, dct in enumerate(data):
            torch_dict = dict()
            torch_dict['id'] = dct['id']
            # здесь как обсуждалось
            # я помечу только первые N сегментов как существующие, остальные при загрузке
            # я заменю  НА ПУСТЫЕ МАТРИЦЫ
            torch_dict['is_segments'] = False
            if loads < load_first_segments:
                torch_dict['is_segments'] = True
                loads += 1
            for item in P.labels_attributes:
                torch_dict[item] = torch.load(dct[item])
                if image_size != 224:
                    torch_dict[item] = F.upsample(torch.tensor([torch_dict[item].tolist()]),
                                                  (image_size, image_size))[0]

            if use_norm:
                torch_dict[P.input_attribute] = normalization(torch.load(dct[P.input_attribute]))
            else:
                torch_dict[P.input_attribute] = torch.load(dct[P.input_attribute])

            if image_size != 224:
                torch_dict[P.input_attribute] = F.upsample(torch.tensor([torch_dict[P.input_attribute].tolist()]),
                                                           (image_size, image_size))[0]
            # normalization(torch.load(dct[P.input_attribute]))
            result.append(torch_dict)
            # print("left:{}, current:{}, right:{} processed".format(lower_bound, idx, upper_bound))
            # P.write_to_log("left:{}, current:{}, right:{} processed".format(lower_bound, idx, upper_bound))
        if load_extend:
            return data, result
        return result

    @staticmethod
    def __save_torch(dct, item, path):
        # print(dct)
        # print(item, dct[item])
        # print(type(Image.open(dct[item])))
        # print(Image.open(dct[item]))
        composited_image = composite(Image.open(dct[item]))
        name = os.path.basename(dct[item])[:-4] + P.cached_extension
        torch.save(composited_image, os.path.join(path, name))


def prepare_data(data: list):
    result = []
    for item in range(0, len(data)):
        dct = data[item]
        input_data = dct[P.input_attribute]
        target_data = dict()
        for i in P.labels_attributes:
            target_data[i] = dct[i]
        target_data = __cache_targets(target_data)
        # tensor of input data
        # tensor of segments
        # tensor of labels answer
        segm, labl = split_targets(target_data)
        # ничего не загружаю
        if not dct['is_segments']:
            segm = -1  # torch.zeros([5, 1, 224, 224]).float()
        result.append((input_data, segm, labl))
    return result


def __cache_targets(dct: dict):
    for i in P.labels_attributes:
        ill_tag = i + '_value'
        if ill_tag not in dct:
            dct[ill_tag] = True if dct[i].sum().item() > 0 else False
    return dct


def split_targets(dct: dict):
    segments = None
    # not exist ill, exist ill
    labels = []
    # trusted = None
    for idx, i in enumerate(P.labels_attributes):
        segments = dct[i] if segments is None else torch.cat((segments, dct[i]))
        # trusted = dct[i]
        ill_tag = i + '_value'
        if dct[ill_tag]:
            labels.append(1)
        else:
            labels.append(0)
    return segments, torch.tensor(labels).float()


def count_size(x):
    cnt = torch.zeros(5)
    for _, _, i in x:
        cnt += i
    return cnt


def load_data(train_size: int, seed: int, image_size: int):
    loader = DatasetLoader.initial()
    all_data = prepare_data(loader.load_tensors(0, train_size * 2, 10 ** 20, image_size))
    # all_data = prepare_data(loader.load_tensors(None, None))
    log = "set size: {}, set by classes: {}".format(len(all_data), count_size(all_data))
    P.write_to_log(log)
    random.Random(seed).shuffle(all_data)
    test_set = all_data[train_size:]
    train_set = all_data[:train_size]
    log = "TEST set size: {}, test set by classes: {}".format(len(test_set), count_size(test_set))
    P.write_to_log(log)

    train_count = count_size(train_set)
    log = "TRAIN set size: {}, train set by classes: {}".format(len(train_set), train_count)
    P.write_to_log(log)

    return train_set, test_set, train_count


def load_data_batch(offset: int, size: int) -> list:
    return prepare_data(DatasetLoader.initial().load_tensors(offset, offset + size))


class ImageDataset(torch.utils.data.Dataset):
    """
    data_set is result of execution DatasetLoader.load_tensors()

    возвращается кортеж:
    (исходная картинка| 5 сегмент. масок| 10 элментов для заболевани)

    i -- если 1 то нет заболевания
    i + 1 --  если 1 то есть заболевание
    """

    def __init__(self, data_set, from_honest: bool = True):
        if from_honest:
            self.data_set = data_set
        else:
            self.data_set = prepare_data(data_set)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        return self.data_set[item]


def create_torch_tensors():
    loader = DatasetLoader(P.data_inputs_path, P.data_labels_path)
    loader.save_images_to_tensors()


if __name__ == "__main__":
    loader = DatasetLoader(P.data_inputs_path, P.data_labels_path)
    loader.save_images_to_tensors()
