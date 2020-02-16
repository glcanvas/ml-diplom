import torch
import torch.optim as opt
import torch.nn as nn
import copy
import property as P
from datetime import datetime
import os
import utils
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-10
probability_threshold = 0.5


def scalar(tensor):
    return tensor.data.cpu().item()


class ATTENTION_MODULE_TRAIN:
    def __init__(self, sam_model: nn.Module = None,
                 train_segments_set=None,
                 test_set=None,
                 train_epochs: int = 100,
                 m_loss: nn.Module = nn.BCELoss(),
                 classes: int = None,
                 use_gpu: bool = True,
                 gpu_device: int = 0,
                 description: str = "sam",
                 change_lr_epochs: int = None,
                 class_number: int = None,
                 register_weights=None,
                 snapshot_elements_count: int = 11,
                 snapshot_dir: str = None):
        self.snapshot_elements_count = snapshot_elements_count
        self.snapshot_dir = snapshot_dir

        self.register_weights = register_weights

        self.class_number = class_number

        self.train_segments_set = train_segments_set
        self.test_set = test_set

        self.train_epochs = train_epochs

        self.sam_model = sam_model

        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        if use_gpu:
            self.sam_model = self.sam_model.cuda(self.gpu_device)

        self.current_epoch = 0

        self.description = description
        self.classes = classes

        self.m_loss = m_loss

        self.puller = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.change_lr_epochs = change_lr_epochs

    def train_attention_module(self):
        optimizer = torch.optim.Adam(self.register_weights("attention", self.sam_model), lr=1e-4)

        for epoch in range(1, self.train_epochs + 1):
            loss_m_sum = 0
            with_segments_elements = 0
            for images, segments, labels in self.train_segments_set:
                if self.class_number is not None:
                    segments = segments[:, self.class_number:self.class_number + 1, :, :]

                images, labels, segments = self.__convert_data_and_label(images, labels, segments)
                segments = self.puller(segments)
                optimizer.zero_grad()
                model_classification, model_segmentation = self.sam_model(images, segments)

                segmentation_loss = self.m_loss(model_segmentation, segments)
                segmentation_l1_loss = segmentation_loss
                segmentation_l1_loss.backward()

                loss_m_sum += scalar(segmentation_loss.sum())
                with_segments_elements += 1

                optimizer.step()
            P.write_to_log("TEST={}, Loss_M={:.5f}".format(epoch, loss_m_sum / (with_segments_elements + EPS)))
            self.__take_snapshot(self.train_segments_set, "TRAIN_{}".format(epoch))
            # self.__take_snapshot(self.test_set, "TEST_{}".format(epoch))

    def __convert_data_and_label(self, data, label, segments=None):
        if self.use_gpu:
            data = data.cuda(self.gpu_device)
            label = label.cuda(self.gpu_device)
            if segments is not None:
                segments = segments.cuda(self.gpu_device)
                return data, label, segments
            return data, label
        return data, label

    def __de_convert_data_and_label(self, data, label, segments=None):
        if self.use_gpu:
            data = data.cpu()
            label = label.cpu()
            if segments is not None:
                segments = segments.cpu()
                return data, label, segments
            return data, label

    def __take_snapshot(self, data_set, snapshot_name: str = None):
        cnt = 0
        model_segments_list = []
        trust_segments_list = []
        images_list = []

        for images, segments, labels in data_set:

            if self.class_number is not None:
                segments = segments[:, self.class_number:self.class_number + 1, :, :]

            images, labels, segments = self.__convert_data_and_label(images, labels, segments)
            segments = self.puller(segments)
            _, model_segments = self.sam_model(images, segments)
            cnt += segments.size(0)
            images, _, segments = self.__de_convert_data_and_label(images, labels, segments)
            model_segments = model_segments.cpu()
            for idx in range(segments.size(0)):
                images_list.append(images[idx])
                model_segments_list.append(model_segments[idx])
                trust_segments_list.append(segments[idx])

            if cnt >= self.snapshot_elements_count:
                break
        fig, axes = plt.subplots(len(images_list), model_segments_list[0].size(0) * 3 + 1, figsize=(50, 100))
        fig.tight_layout()
        # plt.subplots_adjust(bottom=0.5, top=2)
        for idx, img in enumerate(images_list):
            axes[idx][0].imshow(np.transpose(img.numpy(), (1, 2, 0)))

        for idx, (trist_ansv, model_ansv) in enumerate(zip(trust_segments_list, model_segments_list)):
            for class_number in range(trist_ansv.size(0)):
                a = model_ansv[class_number].detach().numpy()
                a = np.array([a] * 3)
                axes[idx][1 + class_number * 3].imshow(np.transpose(a, (1, 2, 0)))
                print("model        idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                     np.max(a),
                                                                                     np.min(a)))
                a = (a - np.min(a)) / (np.max(a) - np.min(a))
                axes[idx][1 + class_number * 3 + 1].imshow(np.transpose(a, (1, 2, 0)))
                print("model normed idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                     np.max(a), np.min(a)))

                a = trist_ansv[class_number].detach().numpy()
                a = np.array([a] * 3)
                axes[idx][1 + class_number * 3 + 2].imshow(np.transpose(a, (1, 2, 0)))
                print("trust        idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                     np.max(a),
                                                                                     np.min(a)))

                print("=" * 50)

                axes[idx][1 + class_number * 3].set(xlabel='model answer class: {}'.format(class_number))
                axes[idx][1 + class_number * 3 + 1].set(xlabel='model normed answer class: {}'.format(class_number))
                axes[idx][1 + class_number * 3 + 2].set(xlabel='trust answer class: {}'.format(class_number))
        print("=" * 50)
        print("=" * 50)
        print("=" * 50)
        print("=" * 50)
        print("=" * 50)
        plt.savefig(os.path.join(self.snapshot_dir, snapshot_name))
        plt.close(fig)
