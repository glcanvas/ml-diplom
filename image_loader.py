import os
import torch
import random
import torch.utils.data.dataset
from torchvision import transforms

prefix = 'ISIC_'
attribute = '_attribute_'
image_size = 224
data_inputs_path = "/home/nikita/PycharmProjects/ISIC2018_Task1-2_Training_Input"
data_labels_path = "/home/nikita/PycharmProjects/ISIC2018_Task2_Training_GroundTruth_v3"

composite = transforms.Compose([
    transforms.Scale((image_size, image_size)),
    transforms.ToTensor(),
])


class DatasetLoader:
    """
    Load ISIC files as images and cache them, or load from cache files
    """

    def __init__(self, input_path: str, target_path: str, cache_input_path: str = None, cache_target_path: str = None,
                 image_suffixes: list = ['.jpg', '.png']):
        self.input_path = input_path
        self.target_path = target_path
        self.cache_input_path = cache_input_path
        self.cache_target_path = cache_target_path
        self.image_suffixes = image_suffixes

    def _load_from_cache(self):
        pass

    def get_input_image_list(self) -> list:
        inputs = self.__get_id_value(self.input_path)
        inputs.sort(key=lambda x: int(x[0]))
        return inputs

    def get_target_image_list(self) -> list:
        targets = self.__get_id_value(self.target_path)
        targets = list(map(lambda x: (x[0].split(attribute)[0], x[0].split(attribute)[1], x[1]), targets))
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

    def merge_data(self) -> list:
        inputs = self.get_input_image_list()
        targets = self.get_target_image_list()

        def composite_zips(x):
            inp = x[0]
            tar = x[1]
            tar['input'] = inp[1]
            return tar
        return list(map(composite_zips, zip(inputs, targets)))

    def __get_id_value(self, absolute_path: str) -> list:
        arrays = []
        for _, _, f in os.walk(absolute_path):
            for file in f:
                for suffix in self.image_suffixes:
                    if file.endswith(suffix) and file.startswith(prefix):
                        identifier = file.split(suffix)[0].split(prefix)[1]
                        arrays.append((identifier, os.path.join(absolute_path, file)))
        return arrays

    def prepare_images(self):
        pass


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self):
        pass


if __name__ == "__main__":
    dl = DatasetLoader(data_inputs_path, data_labels_path)
    l = dl.merge_data()
    for i in l:
        print(i)
