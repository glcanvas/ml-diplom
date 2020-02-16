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


#           sam branch
#     basis   ----------
# inp -----            -------- |||   ----------
#           ----------- merged  avg   classifier
#           classifier

def classifier_weights(model):
    a = list(model.classifier_branch.parameters())
    a.extend(list(model.merged_branch.parameters()))
    a.extend(list(model.avg_pool.parameters()))
    a.extend(list(model.classifier.parameters()))
    return a


def attention_module_weights(model):
    a = []  # list(model.basis.parameters())
    a.extend(list(model.sam_branch.parameters()))
    return a


def register_weights(weight_class, model):
    if weight_class == "classifier":
        return classifier_weights(model)
    elif weight_class == "attention":
        return attention_module_weights(model)
    raise BaseException("unrecognized param: " + weight_class)


def disable_gradient(weight_class, model):
    if weight_class == "classifier":
        for param in classifier_weights(model):
            param.requires_grad = False
        return
    elif weight_class == "attention":
        for param in attention_module_weights(model):
            param.requires_grad = False
        return
    raise BaseException("unrecognized param: " + weight_class)


def enable_gradient(weight_class, model):
    if weight_class == "classifier":
        for param in classifier_weights(model):
            param.requires_grad = True
        return
    elif weight_class == "attention":
        for param in attention_module_weights(model):
            param.requires_grad = True
        return
    raise BaseException("unrecognized param: " + weight_class)


def enable_gradient_model(model):
    for param in model.parameters():
        param.requires_grad = True


def disable_gradient_model(model):
    for param in model.parameters():
        param.requires_grad = False


"""
ДАННОЕ РЕШЕНИЕ НЕЧЕСТНОЕ!!!!!!!!!!!!!
ТАК КАК Я УМНОЖАЮ ВЕТКУ КЛАССИФИКАЦИИ НА НАСТОЯЩУЮ МАСКУ, А НЕ НА ТО ЧТО ПОЛУЧИЛОСЬ В attention модуле
"""


class SAM_TRAIN:

    def __init__(self, sam_model: nn.Module = None,
                 train_segments_set=None,
                 test_set=None,
                 l_loss: nn.Module = nn.BCELoss(),
                 m_loss: nn.Module = nn.BCELoss(),
                 classes: int = None,
                 pre_train_epochs: int = 100,
                 train_epochs: int = 100,
                 save_train_logs_epochs: int = 5,
                 test_each_epoch: int = 5,
                 use_gpu: bool = True,
                 gpu_device: int = 0,
                 description: str = "sam",
                 change_lr_epochs: int = None,
                 class_number: int = None,
                 snapshot_elements_count: int = 11,
                 snapshot_dir: str = None):

        self.snapshot_elements_count = snapshot_elements_count
        self.snapshot_dir = snapshot_dir

        self.class_number = class_number

        self.train_segments_set = train_segments_set
        self.test_set = test_set

        self.sam_model = sam_model

        self.pre_train_epochs = pre_train_epochs
        self.train_epochs = train_epochs
        self.save_train_logs_epochs = save_train_logs_epochs
        self.test_each_epoch = test_each_epoch

        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        if use_gpu:
            self.sam_model = self.sam_model.cuda(self.gpu_device)

        self.current_epoch = 0

        self.description = description
        self.classes = classes

        self.train_model_answers = [[] for _ in range(self.classes)]
        self.train_trust_answers = [[] for _ in range(self.classes)]
        self.train_probabilities = [[] for _ in range(self.classes)]

        self.test_model_answers = [[] for _ in range(self.classes)]
        self.test_trust_answers = [[] for _ in range(self.classes)]
        self.test_probabilities = [[] for _ in range(self.classes)]

        self.best_weights = None
        self.best_test_weights = None

        self.l_loss = l_loss
        self.m_loss = m_loss

        self.puller = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.change_lr_epochs = change_lr_epochs

    def train(self):

        learning_rate = 1e-6
        classifier_optimizer = torch.optim.Adam(register_weights("classifier", self.sam_model),
                                                lr=learning_rate)
        attention_module_optimizer = opt.SGD(register_weights("attention", self.sam_model), lr=0.1,
                                             momentum=0.9)

        self.best_weights = copy.deepcopy(self.sam_model.state_dict())
        best_loss = None
        best_test_loss = None

        for epoch in range(1, self.train_epochs + 1):
            self.current_epoch = epoch

            loss_m_sum = 0
            accuracy_classification_sum_segments = 0
            loss_classification_sum_segments = 0
            loss_l1_sum = 0

            # turn on gradients in model
            # enable_gradient_model(self.sam_model)
            classifier_optimizer = self.__apply_adaptive_learning(classifier_optimizer, learning_rate,
                                                                  self.current_epoch)

            div_flag = False

            if self.current_epoch <= self.pre_train_epochs:
                # loss_classification_sum_segments, accuracy_classification_sum_segments = self.__train_classifier(
                #    classifier_optimizer, self.train_segments_set)

                # disable attention module gradient
                disable_gradient("attention", self.sam_model)
                div_flag = True
                loss_classification_sum_classifier, accuracy_classification_sum_classifier = self.__train_classifier(
                    classifier_optimizer, self.train_segments_set)
            else:

                # disable attention module gradient
                # disable_gradient("attention", self.sam_model)
                loss_classification_sum_classifier, accuracy_classification_sum_classifier = self.__train_classifier(
                    classifier_optimizer, self.train_segments_set)

                # disable classifier branch
                # enable  attention module gradient
                # enable_gradient("attention", self.sam_model)
                # disable_gradient("classifier", self.sam_model)
                accuracy_classification_sum_segments, loss_m_sum, loss_l1_sum = self.__train_segments(
                    attention_module_optimizer, self.train_segments_set)

            accuracy_total = (accuracy_classification_sum_segments + accuracy_classification_sum_classifier) / (
                1 if div_flag else 2)
            classification_loss_total = loss_classification_sum_classifier
            loss_total = loss_classification_sum_classifier + loss_m_sum

            prefix = "PRETRAIN" if epoch <= self.pre_train_epochs else "TRAIN"
            f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                             self.train_trust_answers,
                                                                                             self.train_model_answers)

            text = "{}={} Loss_CL={:.5f} Loss_M={:.5f} Loss_L1={:.5f} Loss_Total={:.5f} Accuracy_CL={:.5f} " \
                   "{} {} {} ".format(prefix, self.current_epoch, classification_loss_total,
                                      loss_m_sum,
                                      loss_l1_sum,
                                      loss_total,
                                      accuracy_total,
                                      f_1_score_text,
                                      recall_score_text,
                                      precision_score_text)

            P.write_to_log(text)

            if self.current_epoch % self.test_each_epoch == 0:
                self.__take_snapshot(self.train_segments_set, "TRAIN_{}".format(epoch))
                self.__take_snapshot(self.test_set, "TEST_{}".format(epoch))
                test_loss, _ = self.test()
                if best_test_loss is None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.best_test_weights = copy.deepcopy(self.sam_model.state_dict())

            if best_loss is None or loss_total < best_loss:
                best_loss = loss_total
                self.best_weights = copy.deepcopy(self.sam_model.state_dict())

            self.train_model_answers = [[] for _ in range(self.classes)]
            self.train_trust_answers = [[] for _ in range(self.classes)]
            self.train_probabilities = [[] for _ in range(self.classes)]

            self.test_model_answers = [[] for _ in range(self.classes)]
            self.test_trust_answers = [[] for _ in range(self.classes)]
            self.test_probabilities = [[] for _ in range(self.classes)]

        self.__save_model(self.best_test_weights)
        self.__save_model(self.best_weights)

    def __train_classifier(self, optimizer: opt.SGD, train_set):
        loss_classification_sum = 0
        accuracy_classification_sum = 0
        without_segments_elements = 0
        for images, segments, labels in train_set:
            if self.class_number is not None:
                labels = labels[:, self.class_number:self.class_number + 1]
                segments = segments[:, self.class_number:self.class_number + 1, :, :]

            images, labels, segments = self.__convert_data_and_label(images, labels, segments)
            segments = self.puller(segments)

            # calculate and optimize model
            optimizer.zero_grad()
            model_classification, _ = self.sam_model(images, segments)
            classification_loss = self.l_loss(model_classification, labels)
            classification_loss.backward()
            optimizer.step()

            output_probability, output_cl, cl_acc = self.__calculate_accuracy(labels, model_classification,
                                                                              labels.size(0))

            self.__save_train_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += scalar(cl_acc.sum())
            loss_classification_sum += scalar(classification_loss.sum())
            without_segments_elements += 1  # labels.size(0)
            self.__de_convert_data_and_label(images, segments, labels)
            torch.cuda.empty_cache()

        return loss_classification_sum / (without_segments_elements + EPS), accuracy_classification_sum / (
                without_segments_elements + EPS)

    def __train_segments(self, optimizer: opt.SGD, train_set):
        accuracy_classification_sum = 0
        loss_m_sum = 0
        loss_l1_sum = 0
        with_segments_elements = 0

        for images, segments, labels in train_set:
            if self.class_number is not None:
                labels = labels[:, self.class_number:self.class_number + 1]
                segments = segments[:, self.class_number:self.class_number + 1, :, :]

            images, labels, segments = self.__convert_data_and_label(images, labels, segments)
            segments = self.puller(segments)
            optimizer.zero_grad()
            model_classification, model_segmentation = self.sam_model(images, segments)

            # classification_loss = self.l_loss(model_classification, labels)
            # classification_loss.backward(retain_graph=True)

            segmentation_loss = self.m_loss(model_segmentation, segments)
            # l1_loss = self.__add_l1_regularization_loss(register_weights("attention", self.sam_model))
            segmentation_l1_loss = segmentation_loss  # + l1_loss
            segmentation_l1_loss.backward()

            optimizer.step()

            output_probability, output_cl, cl_acc = self.__calculate_accuracy(labels, model_classification,
                                                                              labels.size(0))
            self.__save_train_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += scalar(cl_acc.sum())
            loss_m_sum += scalar(segmentation_loss.sum())
            loss_l1_sum += 0  # scalar(l1_loss.sum())
            with_segments_elements += 1  # labels.size(0)
            self.__de_convert_data_and_label(images, labels, segments)
            torch.cuda.empty_cache()

        return accuracy_classification_sum / (with_segments_elements + EPS), \
               loss_m_sum / (with_segments_elements + EPS), loss_l1_sum / (with_segments_elements + EPS)

    def test(self):
        loss_classification_sum = 0
        accuracy_classification_sum = 0
        without_segments_elements = 0
        for images, _segments, labels in self.test_set:
            if self.class_number is not None:
                labels = labels[:, self.class_number:self.class_number + 1]
                _segments = _segments[:, self.class_number:self.class_number + 1, :, :]
            images, labels, _segments = self.__convert_data_and_label(images, labels, _segments)
            _segments = self.puller(_segments)
            model_classification, _ = self.sam_model(images, _segments)
            classification_loss = self.l_loss(model_classification, labels)
            # classification_loss.backward()

            output_probability, output_cl, cl_acc = self.__calculate_accuracy(labels, model_classification,
                                                                              labels.size(0))

            self.__save_test_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += scalar(cl_acc.sum())
            loss_classification_sum += scalar(classification_loss.sum())
            without_segments_elements += 1  # labels.size(0)
            self.__de_convert_data_and_label(images, labels)
            torch.cuda.empty_cache()

        f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                         self.test_trust_answers,
                                                                                         self.test_model_answers)

        loss_classification_sum /= without_segments_elements + EPS
        accuracy_classification_sum /= without_segments_elements + EPS
        text = 'TEST Loss_CL={:.5f} Accuracy_CL={:.5f} {} {} {} '.format(loss_classification_sum,
                                                                         accuracy_classification_sum,
                                                                         f_1_score_text,
                                                                         recall_score_text,
                                                                         precision_score_text)
        P.write_to_log(text)

        return loss_classification_sum, accuracy_classification_sum

    def __save_model(self, weights):
        name = self.description + "_date-" + datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S') + ".torch"
        try:
            saved_dir = os.path.join(P.base_data_dir, 'sam_weights')
            os.makedirs(saved_dir, exist_ok=True)
            saved_file = os.path.join(saved_dir, name)
            torch.save(weights, saved_file)
            print("Save model: {}".format(name))
            P.write_to_log("Save model: {}".format(name))
        except Exception as e:
            print("Can't save model: {}".format(name), e)
            P.write_to_log("Can't save model: {}".format(name), e)

    def __save_train_data(self, labels, output_cl, output_probability):
        output_cl = output_cl.cpu()
        output_probability = output_probability.cpu()
        labels = labels.cpu()
        for i in range(output_cl.shape[1]):
            self.train_trust_answers[i].extend(labels[:, i].tolist())
            self.train_model_answers[i].extend(output_cl[:, i].tolist())
            self.train_probabilities[i].extend(output_probability[:, i].tolist())

    def __save_test_data(self, labels, output_cl, output_probability):
        output_cl = output_cl.cpu()
        output_probability = output_probability.cpu()
        labels = labels.cpu()
        for i in range(output_cl.shape[1]):
            self.test_trust_answers[i].extend(labels[:, i].tolist())
            self.test_model_answers[i].extend(output_cl[:, i].tolist())
            self.test_probabilities[i].extend(output_probability[:, i].tolist())

    def __calculate_accuracy(self, labels, output_cl, batch_size):
        output_probability = output_cl.clone()
        output_cl[output_cl >= probability_threshold] = 1
        output_cl[output_cl < probability_threshold] = 0
        cl_acc = torch.eq(output_cl, labels).sum().float()
        cl_acc /= (batch_size * self.classes + EPS)
        return output_probability, output_cl, cl_acc

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

    def __apply_adaptive_learning(self, optimizer, learning_rate, epoch):
        pow_epoch = epoch // self.change_lr_epochs
        if pow_epoch == 0:
            return optimizer
        return torch.optim.Adam(register_weights("classifier", self.sam_model),
                                lr=learning_rate * (0.1 ** pow_epoch))

    # def __add_l1_regularization_loss(self, weights):
    #    l2_loss = torch.tensor(0.)
    #    if self.use_gpu:
    #        l2_loss = l2_loss.cuda(self.gpu_device)
    #    for param in weights:
    #        l2_loss += torch.norm(param)
    #    return 0.5 * l2_loss

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
            _, model_segments = self.sam_model(images)
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
                print(
                    "model        idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                   np.max(a),
                                                                                   np.min(a)))
                a = (a - np.min(a)) / (np.max(a) - np.min(a))
                axes[idx][1 + class_number * 3 + 1].imshow(np.transpose(a, (1, 2, 0)))
                print(
                    "model normed idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
                                                                                   np.max(a), np.min(a)))

                a = trist_ansv[class_number].detach().numpy()
                a = np.array([a] * 3)
                axes[idx][1 + class_number * 3 + 2].imshow(np.transpose(a, (1, 2, 0)))
                print(
                    "trust        idx={}, class={}, sum={}, max={}, min={}".format(idx, class_number, np.sum(a),
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
