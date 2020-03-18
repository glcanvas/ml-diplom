import os
from collections import Set

import torch
import torch.utils.data.dataset
from torchvision import transforms
from PIL import Image
import copy

import property as P
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

composite = transforms.Compose([
    #  transforms.ToTensor(),
    transforms.Scale((P.image_size, P.image_size)),
    transforms.ToTensor(),
    #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normalization = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class VocDataLoader:

    def __init__(self, root_path: str, object_names: list, object_indexes: list):
        self.root_path = root_path
        self.object_indexes = object_indexes
        self.images_with_segment = self.__images_with_segment()
        self.image_objects = [self.__load_images_contains_object(i) for i in object_names]
        self.segmented_labels, self.unsegmented_labels = self.__merge_all_to_torch_tensor(self.images_with_segment,
                                                                                          self.image_objects)

        self.train_data = list(zip(map(lambda x: x[1], self.segmented_labels),
                                   self.__load_torch_images(self.segmented_labels, "JPEGImages_cached"),
                                   map(self.__convert_to_segment_tensor,
                                       self.__load_torch_images(self.segmented_labels, "SegmentationClass_cached"))))

        self.test_data = list(zip(map(lambda x: x[1], self.unsegmented_labels),
                                  self.__load_torch_images(self.unsegmented_labels, "JPEGImages_cached")))


        print("Done")

    def __show_by_idx(self, idx: int):
        plt.imshow(
            np.transpose(torch.cat((self.train_data[idx][2], self.train_data[idx][2], self.train_data[idx][2])).numpy(),
                         (1, 2, 0)))

    def __convert_to_segment_tensor(self, tensor_segment):
        list_of_segments = []
        for idx in self.object_indexes:
            indexes_value = idx / 255
            cloned = copy.deepcopy(tensor_segment)
            cloned[cloned >= indexes_value + P.VOC_EPS] = 0.0
            cloned[cloned <= indexes_value - P.VOC_EPS] = 0.0
            cloned[cloned != 0.0] = 1.0
            list_of_segments.append(cloned)
        return torch.cat(list_of_segments)

    def __images_with_segment(self) -> set:
        segment_label = os.path.join(self.root_path, "ImageSets", "Segmentation")
        segment_label = os.path.join(segment_label, "trainval.txt")
        with open(segment_label) as f:
            return set(map(lambda x: x.split()[0], f.readlines()))

    # 2010_001692 0.01569 0.02353
    def __load_images_contains_object(self, object_name: str) -> map:
        images_label = os.path.join(self.root_path, "ImageSets", "Main")
        images_label = os.path.join(images_label, object_name + "_trainval.txt")
        with open(images_label) as f:
            def inner(array: str) -> tuple:
                elements = array.split()
                return elements[0], int(elements[1])

            return {k: v for (k, v) in list(filter(lambda x: x[1] != 0, map(inner, f.readlines())))}

    # images_by_name -- list of list
    def __merge_all_to_torch_tensor(self, segmented_images: set, images_by_name: list) -> tuple:
        # first load objects with segment
        # then all without segment
        tensor_for_segment = self.__build_set(segmented_images, images_by_name)

        not_used_images_id = set()
        for obj in images_by_name:
            for index in obj:
                if index in segmented_images:
                    continue
                not_used_images_id.add(index)

        tensor_without_segment = self.__build_set(not_used_images_id, images_by_name)
        return tensor_for_segment, tensor_without_segment

    def __load_torch_images(self, lst: list, prefix) -> list:
        res = []
        dir = os.path.join(self.root_path, prefix)
        for name, label in lst:
            full_dir = os.path.join(dir, name + ".torch")
            torch_tensor = torch.load(full_dir)
            res.append(torch_tensor)
        return res

    @staticmethod
    def __build_set(segmented_images, images_by_name):
        tensor_for_segment = []
        for segment_id in segmented_images:
            tensor = torch.tensor([0.0 for _ in range(len(images_by_name))])
            for obj_idx, obj in enumerate(images_by_name):
                if segment_id in obj and obj[segment_id] == 1:
                    tensor[obj_idx] = 1.0
            tensor_for_segment.append((segment_id, tensor))
        return tensor_for_segment


def __get_full_file_paths(path: str) -> list:
    for _, _, file_names in os.walk(path):
        return list(map(lambda x: (x, os.path.join(path, x)),
                        filter(lambda x: x.endswith(".jpg") or x.endswith(".png"), file_names)))
    raise ValueError("No files found at {}".format(path))


def convert_images(jpeg_path: str, jpeg_cached_path: str, prefix: str):
    jpeg_files = __get_full_file_paths(jpeg_path)
    for index, (jpeg_file_name, jpeg_file_path) in enumerate(jpeg_files):
        torch_image = composite(Image.open(jpeg_file_path))
        cached_file_name = jpeg_file_name[:-4] + P.cached_extension
        cached_file_path = os.path.join(jpeg_cached_path, cached_file_name)
        torch.save(torch_image, cached_file_path)
        print("processed {} images {} of {}".format(prefix, index, len(jpeg_files)))


def convert_to_torch(path_to_voc: str):
    jpeg_path = os.path.join(path_to_voc, "JPEGImages")
    segment_class_path = os.path.join(path_to_voc, "SegmentationClass")

    jpeg_cached_path = os.path.join(path_to_voc, "JPEGImages_cached")
    segment_class_cached_path = os.path.join(path_to_voc, "SegmentationClass_cached")

    # create
    os.makedirs(jpeg_cached_path, exist_ok=True)
    os.makedirs(segment_class_cached_path, exist_ok=True)
    convert_images(jpeg_path, jpeg_cached_path, "jpeg")
    convert_images(segment_class_path, segment_class_cached_path, "segment")


if __name__ == "__main__":
    # convert_to_torch(P.voc_data_path)
    v = VocDataLoader(P.voc_data_path, ['person', 'aeroplane', 'chair', 'car', 'cow', 'horse'],
                      [15, 1, 9, 7, 10, 13])
