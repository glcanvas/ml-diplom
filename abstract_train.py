"""
class with common functions
"""

import torch
import torch.nn as nn
from datetime import datetime
import os
import utils
import matplotlib.pyplot as plt
import numpy as np
import property as p


class AbstractTrain:
    """
    Common class for all trains which has attention module
    """
    PULLER = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=2, stride=2))

    def __init__(self, classes: int = None,
                 pre_train_epochs: int = 100,
                 train_epochs: int = 100,
                 save_train_logs_epochs: int = 5,
                 test_each_epoch: int = 5,
                 use_gpu: bool = True,
                 gpu_device: int = 0,
                 description: str = "am",
                 left_class_number: int = None,
                 right_class_number: int = None,
                 snapshot_elements_count: int = 11,
                 snapshot_dir: str = None,
                 classifier_learning_rate: float = None,
                 attention_module_learning_rate: float = None,
                 weight_decay: float = 0):
        self.classifier_learning_rate = classifier_learning_rate
        self.attention_module_learning_rate = attention_module_learning_rate

        self.snapshot_elements_count = snapshot_elements_count
        self.snapshot_dir = snapshot_dir

        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        self.description = description
        self.classes = classes

        self.left_class_number = left_class_number
        self.right_class_number = right_class_number
        # self.class_number = class_number

        self.pre_train_epochs = pre_train_epochs
        self.train_epochs = train_epochs
        self.save_train_logs_epochs = save_train_logs_epochs
        self.test_each_epoch = test_each_epoch

        self.weight_decay = weight_decay

        self.current_epoch = 0

        self.train_model_answers = [[] for _ in range(self.classes)]
        self.train_trust_answers = [[] for _ in range(self.classes)]
        self.train_probabilities = [[] for _ in range(self.classes)]

        self.test_model_answers = [[] for _ in range(self.classes)]
        self.test_trust_answers = [[] for _ in range(self.classes)]
        self.test_probabilities = [[] for _ in range(self.classes)]

    def test(self, model, test_set, l_loss, m_loss):
        loss_classification_sum = 0
        loss_segmentation_sum = 0
        accuracy_classification_sum = 0
        batch_count = 0
        for images, segments, labels in test_set:
            labels, segments = utils.reduce_to_class_number(self.left_class_number, self.right_class_number, labels,
                                                            segments)
            images, labels, segments = self.convert_data_and_label(images, labels, segments)
            segments = self.PULLER(segments)
            model_classification, model_segmentation = utils.wait_while_can_execute(model, images)

            classification_loss = l_loss(model_classification, labels)
            segmentation_loss = m_loss(model_segmentation, segments)

            output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                            labels.size(0))

            self.save_test_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += utils.scalar(cl_acc.sum())
            loss_classification_sum += utils.scalar(classification_loss.sum())
            loss_segmentation_sum += utils.scalar(segmentation_loss.sum())
            batch_count += 1
            self.de_convert_data_and_label(images, labels)
            torch.cuda.empty_cache()

        f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                         self.test_trust_answers,
                                                                                         self.test_model_answers)

        loss_classification_sum /= batch_count + p.EPS
        accuracy_classification_sum /= batch_count + p.EPS
        loss_segmentation_sum /= batch_count + p.EPS
        text = 'TEST Loss_CL={:.5f} Loss_M={:.5f} Accuracy_CL={:.5f} {} {} {} '.format(loss_classification_sum,
                                                                                       loss_segmentation_sum,
                                                                                       accuracy_classification_sum,
                                                                                       f_1_score_text,
                                                                                       recall_score_text,
                                                                                       precision_score_text)
        p.write_to_log(text)

        return loss_classification_sum, accuracy_classification_sum

    def train_classifier(self, model, l_loss, m_loss, optimizer, train_set):
        loss_classification_sum = 0
        loss_segmentation_sum = 0
        accuracy_classification_sum = 0
        batch_count = 0

        for images, segments, labels in train_set:
            labels, segments = utils.reduce_to_class_number(self.left_class_number, self.right_class_number, labels,
                                                            segments)
            images, labels, segments = self.convert_data_and_label(images, labels, segments)
            segments = self.PULLER(segments)

            # calculate and optimize model
            optimizer.zero_grad()

            model_classification, model_segmentation = utils.wait_while_can_execute(model, images)
            segmentation_loss = m_loss(model_segmentation, segments)
            classification_loss = l_loss(model_classification, labels)
            torch.cuda.empty_cache()
            classification_loss.backward()
            optimizer.step()

            output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                            labels.size(0))

            self.save_train_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += utils.scalar(cl_acc.sum())
            loss_classification_sum += utils.scalar(classification_loss.sum())
            loss_segmentation_sum += utils.scalar(segmentation_loss.sum())
            batch_count += 1
            self.de_convert_data_and_label(images, segments, labels)
            torch.cuda.empty_cache()

        return loss_classification_sum / (batch_count + p.EPS), accuracy_classification_sum / (
                batch_count + p.EPS), loss_segmentation_sum / (batch_count + p.EPS)

    def train_segments(self, model, l_loss, m_loss, optimizer: torch.optim.Adam, train_set):
        accuracy_classification_sum = 0
        loss_m_sum = 0
        loss_l1_sum = 0
        loss_classification_sum = 0
        batch_count = 0

        for images, segments, labels in train_set:
            labels, segments = utils.reduce_to_class_number(self.left_class_number, self.right_class_number, labels,
                                                            segments)
            images, labels, segments = self.convert_data_and_label(images, labels, segments)
            segments = self.PULLER(segments)
            optimizer.zero_grad()
            model_classification, model_segmentation = utils.wait_while_can_execute(model, images)

            classification_loss = l_loss(model_classification, labels)
            segmentation_loss = m_loss(model_segmentation, segments)

            torch.cuda.empty_cache()
            segmentation_loss.backward()
            optimizer.step()

            output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                            labels.size(0))
            self.save_train_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += utils.scalar(cl_acc.sum())
            loss_m_sum += utils.scalar(segmentation_loss.sum())
            loss_l1_sum += 0
            loss_classification_sum += utils.scalar(classification_loss.sum())
            batch_count += 1
            self.de_convert_data_and_label(images, labels, segments)
            torch.cuda.empty_cache()

        return accuracy_classification_sum / (batch_count + p.EPS), loss_m_sum / (
                batch_count + p.EPS), loss_l1_sum / (
                       batch_count + p.EPS), loss_classification_sum / (batch_count + p.EPS)

    def save_model(self, weights):
        name = self.description + "_date-" + datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S') + ".torch"
        try:
            saved_dir = os.path.join(p.base_data_dir, 'model_weights')
            os.makedirs(saved_dir, exist_ok=True)
            saved_file = os.path.join(saved_dir, name)
            torch.save(weights, saved_file)
            print("Save model: {}".format(name))
            p.write_to_log("Save model: {}".format(name))
        except Exception as e:
            print("Can't save model: {}".format(name), e)
            p.write_to_log("Can't save model: {}".format(name), e)

    def save_train_data(self, labels, output_cl, output_probability):
        output_cl = output_cl.cpu()
        output_probability = output_probability.cpu()
        labels = labels.cpu()
        for i in range(output_cl.shape[1]):
            self.train_trust_answers[i].extend(labels[:, i].tolist())
            self.train_model_answers[i].extend(output_cl[:, i].tolist())
            self.train_probabilities[i].extend(output_probability[:, i].tolist())

    def save_test_data(self, labels, output_cl, output_probability):
        output_cl = output_cl.cpu()
        output_probability = output_probability.cpu()
        labels = labels.cpu()
        for i in range(output_cl.shape[1]):
            self.test_trust_answers[i].extend(labels[:, i].tolist())
            self.test_model_answers[i].extend(output_cl[:, i].tolist())
            self.test_probabilities[i].extend(output_probability[:, i].tolist())

    def calculate_accuracy(self, labels, output_cl, batch_count):
        output_probability = output_cl.clone()
        output_cl[output_cl >= p.PROBABILITY_THRESHOLD] = 1
        output_cl[output_cl < p.PROBABILITY_THRESHOLD] = 0
        cl_acc = torch.eq(output_cl, labels).sum().float()
        cl_acc /= (batch_count * self.classes + p.EPS)
        return output_probability, output_cl, cl_acc

    def convert_data_and_label(self, data, label, segments=None):
        if self.use_gpu:
            data = data.cuda(self.gpu_device)
            label = label.cuda(self.gpu_device)
            if segments is not None:
                segments = segments.cuda(self.gpu_device)
                return data, label, segments
            return data, label
        return data, label

    def de_convert_data_and_label(self, data, label, segments=None):
        if self.use_gpu:
            data = data.cpu()
            label = label.cpu()
            if segments is not None:
                segments = segments.cpu()
                return data, label, segments
            return data, label

    def clear_temp_metrics(self):
        self.train_model_answers = [[] for _ in range(self.classes)]
        self.train_trust_answers = [[] for _ in range(self.classes)]
        self.train_probabilities = [[] for _ in range(self.classes)]

        self.test_model_answers = [[] for _ in range(self.classes)]
        self.test_trust_answers = [[] for _ in range(self.classes)]
        self.test_probabilities = [[] for _ in range(self.classes)]

    def take_snapshot(self, data_set, model, snapshot_name: str = None):
        cnt = 0
        model_segments_list = []
        trust_segments_list = []
        images_list = []

        for images, segments, labels in data_set:

            segments = segments[:, self.left_class_number:self.right_class_number, :, :]

            images, labels, segments = self.convert_data_and_label(images, labels, segments)
            segments = self.PULLER(segments)
            _, model_segmentation = utils.wait_while_can_execute(model, images)

            cnt += segments.size(0)
            images, _, segments = self.de_convert_data_and_label(images, labels, segments)
            model_segmentation = model_segmentation.cpu()
            for idx in range(segments.size(0)):
                images_list.append(images[idx])
                model_segments_list.append(model_segmentation[idx])
                trust_segments_list.append(segments[idx])

            if cnt >= self.snapshot_elements_count:
                break
        fig, axes = plt.subplots(len(images_list), model_segments_list[0].size(0) * 3 + 1, figsize=(50, 100))
        fig.tight_layout()
        for idx, img in enumerate(images_list):
            axes[idx][0].imshow(np.transpose(img.numpy(), (1, 2, 0)))

        for idx, (trist_answer, model_answer) in enumerate(zip(trust_segments_list, model_segments_list)):
            for class_number in range(trist_answer.size(0)):
                a = model_answer[class_number].detach().numpy()
                a = np.array([a] * 3)
                axes[idx][1 + class_number * 3].imshow(np.transpose(a, (1, 2, 0)))
                p.write_to_log(
                    "model        idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                   np.max(a),
                                                                                   np.min(a)))
                a = (a - np.min(a)) / (np.max(a) - np.min(a))
                axes[idx][1 + class_number * 3 + 1].imshow(np.transpose(a, (1, 2, 0)))
                p.write_to_log(
                    "model normed idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                   np.max(a), np.min(a)))

                a = trist_answer[class_number].detach().numpy()
                a = np.array([a] * 3)
                axes[idx][1 + class_number * 3 + 2].imshow(np.transpose(a, (1, 2, 0)))
                p.write_to_log(
                    "trust        idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                   np.max(a),
                                                                                   np.min(a)))

                p.write_to_log("=" * 50)

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


"""
ginec@laplas:~/ml2/ml-diplom/runners_alternate_one_loss$ 
nduginec@laplas:~/ml2/ml-diplom/runners_alternate_one_loss$ EXCEPTION CUDA out of memory. Tried to allocate 392.00 MiB (GPU 0; 10.92 GiB total capacity; 3.24 GiB already allocated; 216.69 MiB free; 3.33 GiB reserved in total by PyTorch)
<class 'RuntimeError'>
EXCEPTION
CUDA out of memory. Tried to allocate 392.00 MiB (GPU 0; 10.92 GiB total capacity; 3.24 GiB already allocated; 216.69 MiB free; 3.33 GiB reserved in total by PyTorch)
<class 'RuntimeError'>
  File "/home/nduginec/ml2/ml-diplom/main_alternate.py", line 75, in <module>
    traceback.print_stack()
Traceback (most recent call last):
  File "/home/nduginec/ml2/ml-diplom/main_alternate.py", line 77, in <module>
    raise e
  File "/home/nduginec/ml2/ml-diplom/main_alternate.py", line 69, in <module>
    sam_train.train()
  File "/home/nduginec/ml2/ml-diplom/alternate_attention_module_train.py", line 75, in train
    self.train_segments_set)
  File "/home/nduginec/ml2/ml-diplom/abstract_train.py", line 119, in train_classifier
    classification_loss.backward()
  File "/home/nduginec/nduginetc_env3/lib/python3.5/site-packages/torch/tensor.py", line 195, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/home/nduginec/nduginetc_env3/lib/python3.5/site-packages/torch/autograd/__init__.py", line 99, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: CUDA out of memory. Tried to allocate 392.00 MiB (GPU 0; 10.92 GiB total capacity; 3.24 GiB already allocated; 216.69 MiB free; 3.33 GiB reserved in total by PyTorch)
/home/nduginec/nduginetc_env3/lib/python3.5/site-packages/torchvision/transforms/transforms.py:220: UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead.
  "please use transforms.Resize
"""
