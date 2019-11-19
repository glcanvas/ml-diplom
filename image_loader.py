import os
import torch
import torch.utils.data.dataset
from torchvision import transforms
from PIL import Image

prefix = 'ISIC_'
attribute = '_attribute_'
image_size = 224
data_inputs_path = "/home/nikita/PycharmProjects/ISIC2018_Task1-2_Training_Input"
data_labels_path = "/home/nikita/PycharmProjects/ISIC2018_Task2_Training_GroundTruth_v3"

cache_data_inputs_path = "/home/nikita/PycharmProjects/ISIC2018_Task1-2_Training_Input/cached"
cache_data_labels_path = "/home/nikita/PycharmProjects/ISIC2018_Task2_Training_GroundTruth_v3/cached"

composite = transforms.Compose([
    transforms.Scale((image_size, image_size)),
    transforms.ToTensor(),
])

labels_attributes = [
    'streaks',
    'negative_network',
    'milia_like_cyst',
    'globules',
    'pigment_network'
]
input_attribute = 'input'
cached_extension = '.torch'


class DatasetLoader:
    """
    Load ISIC files as images and cache them, or load from cache files
    """

    def __init__(self, input_path: str = None, target_path: str = None, cache_input_path: str = None,
                 cache_target_path: str = None, image_suffixes: list = ['.jpg', '.png', cached_extension]):
        self.input_path = input_path
        self.target_path = target_path
        self.image_suffixes = image_suffixes

        # create cached path if not exists
        if cache_input_path is None:
            cache_input_path = os.path.join(input_path, "cached")
            os.makedirs(cache_input_path, exist_ok=True)
            print("created cache input dir: {}".format(cache_input_path))
        if cache_target_path is None:
            cache_target_path = os.path.join(target_path, "cached")
            os.makedirs(cache_target_path, exist_ok=True)
            print("created cache target dir: {}".format(cache_target_path))
        self.cache_input_path = cache_input_path
        self.cache_target_path = cache_target_path

    def __get_input_image_list(self, input_path) -> list:
        inputs = self.__get_id_value(input_path)
        inputs.sort(key=lambda x: int(x[0]))
        return inputs

    def __get_target_image_list(self, target_path) -> list:
        targets = self.__get_id_value(target_path)
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

    def __merge_data(self, input_path, target_path) -> list:
        inputs = self.__get_input_image_list(input_path)
        targets = self.__get_target_image_list(target_path)

        def composite_zips(x):
            inp = x[0]
            tar = x[1]
            tar[input_attribute] = inp[1]
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

    def save_images_to_tensors(self):
        data = self.__merge_data(self.input_path, self.target_path)
        data_len = len(data)
        for idx, dct in enumerate(data):
            for item in labels_attributes:
                self.__save_torch(dct, item, self.cache_target_path)
            self.__save_torch(dct, input_attribute, self.cache_input_path)
            print("=" * 10)
            print("save: {} of {} elements".format(idx, data_len))
        print("all saved successfully")

    def load_tensors(self, lower_bound=None, upper_bound=None):
        data = self.__merge_data(self.cache_input_path, self.cache_target_path)
        data_len = len(data)
        lower_bound = 0 if lower_bound is None else lower_bound
        upper_bound = data_len if upper_bound is None else upper_bound
        result = []

        for idx, dct in enumerate(data):
            if lower_bound <= idx < upper_bound:
                torch_dict = dict()
                torch_dict['id'] = dct['id']
                for item in labels_attributes:
                    torch_dict[item] = torch.load(dct[item])
                torch_dict[input_attribute] = torch.load(dct[input_attribute])
                result.append(torch_dict)

        return result

    @staticmethod
    def __save_torch(dct, item, path):
        composited_image = composite(Image.open(dct[item]))
        name = os.path.basename(dct[item])[:-4] + cached_extension
        torch.save(composited_image, os.path.join(path, name))


class ImageDataset(torch.utils.data.Dataset):
    """
    data_set is result of execution DatasetLoader.load_tensors()
    """

    def __init__(self, data_set):
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        dct = self.data_set[item]
        input_data = dct[input_attribute]
        target_data = dict()
        for i in labels_attributes:
            target_data[i] = dct[i]
        return input_data, target_data


if __name__ == "__main__":
    dl = DatasetLoader(data_inputs_path, data_labels_path, cache_data_inputs_path, cache_data_labels_path)
    # dl.save_images_to_tensors()
    res = dl.load_tensors(0, 10)
    print(res)
    # l = dl.merge_data()
    # for i in l:
    #    print(i)
