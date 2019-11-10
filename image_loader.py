from sklearn.utils import shuffle
import torch.utils.data.dataset
import os
from torchvision import transforms
from PIL import Image
import torch
from multiprocessing.pool import ThreadPool
from properties import classes, field_id, field_train, image_size, num_workers

name_delimiter = '_.'
prefix = 'ISIC_'
attribute = '_attribute_'

composite = transforms.Compose([
    transforms.Scale((image_size, image_size)),
    transforms.ToTensor(),
])


def images_filter(dir_path: str, suffix: str):
    arrays = []
    for _, _, f in os.walk(dir_path):
        for file in f:
            if not file.endswith(suffix):
                continue
            id = file.split(suffix)[0].split(prefix)[1]
            arrays.append((id, os.path.join(dir_path, file)))
    return arrays


def map_inputs_labels(data_inputs_path: str, data_labels_path: str):
    """
    composed inputs images and labels images to dict with attributes
    :return: dict<img_id, <path to original image, path to label image, attributes>>
    """
    inputs = images_filter(data_inputs_path, '.jpg')
    labels = images_filter(data_labels_path, '.png')
    # <id, attribute_name, path_to_file>
    labels_generator = map(lambda x: (x[0].split(attribute)[0], x[0].split(attribute)[1], x[1]), labels)

    labels_dict = dict()
    for ids, attr, path in labels_generator:
        if ids in labels_dict:
            labels_dict[ids][attr] = path
        else:
            labels_dict[ids] = {attr: path}

    result = []
    for ids, path in inputs:
        labels_dict[ids]['id'] = ids
        labels_dict[ids]['train'] = path
        result.append(labels_dict[ids])
    return result


def split_set(values: list, test_size: int, validate_size: int):
    """
    get test set, validate set from list of images, previous shuffled
    :param values: result of execute map_inputs_label
    :param test_size: train set size
    :param validate_size: validate set size
    :return: train set, validate set
    """
    shuffled = shuffle(values)
    return shuffled[0:test_size], shuffled[test_size:test_size + validate_size]


def map_cell(cell: dict):
    for i in classes:
        cell[i] = composite(Image.open(cell[i]))
    cell[field_train] = composite(Image.open(cell[field_train]))
    return cell


def prepare_data(*args):
    pool = ThreadPool(num_workers)
    for array in args:
        for v in array:
            pool.apply(map_cell, args=(v,))
    pool.close()
    pool.join()


def save_data(path: str, data: dict):
    ids = data[field_id]
    os.makedirs(os.path.join(path, ids), exist_ok=True)
    for i in classes:
        torch.save(data[i], os.path.join(path, ids, i))
    torch.save(data[field_train], os.path.join(path, ids, field_train))


# either train_path or test_path
def load_tensors(path: str):
    result = []
    for tensors in os.listdir(path):
        tensor_dir = os.path.join(path, tensors)
        if not os.path.isdir(tensor_dir):
            continue
        res = dict()
        res[field_id] = tensors
        for fl in os.listdir(tensor_dir):
            tensor_file = os.path.join(tensor_dir, fl)
            if not os.path.isfile(tensor_file):
                continue
            res[fl] = torch.load(tensor_file)
        result.append(res)
    return result


def remove_empty_mask(cell: dict):
    for c in classes:
        if cell[c].sum() > 1:
            cell.pop(c)
    return cell


def apply_resize(cell: dict):
    scale = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale((image_size, image_size)),
        transforms.ToTensor()
    ])
    for c in classes:
        cell[c] = scale(cell[c])
    cell[field_train] = scale(cell[field_train])
    return cell


class CustomDataset(torch.utils.data.dataset.Dataset):
    """
    data -- list with dict of result load_tensors
    """

    def __init__(self, data: list):
        self.data = data
        self.inputs = [i[field_train] for i in data]
        self.labels = [self.__item_to_tensor(i) for i in data]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    @staticmethod
    def __item_to_tensor(d: dict) -> torch.Tensor:
        t = torch.tensor([0 for _ in classes])
        for idx, cl in enumerate(classes):
            if cl in d:
                t[idx] = 1
        return t
