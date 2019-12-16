"""
classify dataset
"""

import torch
import torch.nn.functional as F
import torchvision.models as m
import property as P
import torch.nn as nn
import copy
from datetime import datetime
import os


def scalar(tensor):
    return tensor.data.cpu().item()


def send_to_gpu(*args) -> tuple:
    result = []
    for i in args:
        result.append(i.cuda())
    return (*result,)


def wrap_to_variable(*args) -> tuple:
    result = []
    for i in args:
        result.append(torch.autograd.Variable(i))
    return (*result,)


class Classifier:

    def __init__(self, classes: int, gpu=False, loss_classifier=None):
        self.gpu = gpu
        # здесь * 2 так как каждой метке соответсвует бинарное значение -- да, нет в самом деле я сделал так для
        # классификации так как сделать по другому не знаю
        self.classes = classes * 2
        self.model = m.vgg16(pretrained=True)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, self.classes)

        self.best_weights = copy.deepcopy(self.model.state_dict())
        self.best_test_weights = copy.deepcopy(self.model.state_dict())

        if loss_classifier is None:
            self.loss_classifier = torch.nn.BCEWithLogitsLoss()

        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
        else:
            self.tensor_source = torch

    def train(self, test_each_epochs: int, train_data_set, test_data_set, epochs, train_batch_size: int,
              test_batch_size: int,
              learning_rate=1e-6):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        best_accuracy = None
        best_test_accuracy = None

        for epoch in range(1, epochs + 1):
            total_loss_cl = 0
            total_cl_acc = 0
            for images, _, labels in train_data_set:
                if self.gpu:
                    images, labels = send_to_gpu(images, labels)
                images, labels = wrap_to_variable(images, labels)
                class_label = labels

                self.model.zero_grad()
                output_cl = self.model(images)

                grad_target = output_cl * class_label
                grad_target.backward(gradient=class_label * output_cl, retain_graph=True)

                loss_cl = self.loss_classifier(output_cl, class_label)

                loss_cl.backward()
                optimizer.step()

                total_loss_cl, total_cl_acc = self.__calculate_accuracy(output_cl, class_label,
                                                                        train_batch_size, loss_cl,
                                                                        total_loss_cl, total_cl_acc)

            if best_accuracy is None or total_loss_cl < best_accuracy:
                best_accuracy = total_loss_cl
                self.best_weights = copy.deepcopy(self.model.state_dict())

            train_size = len(train_data_set)
            text = '%i of %i EPOCHS, TEST Loss_CL: %f, Accuracy_CL: %f%%' % (epoch, epochs, total_loss_cl / train_size, (total_cl_acc / train_size) * 100.0)
            print(text)
            P.write_to_log(text)
            if epoch % test_each_epochs == 0:
                test_loss, _ = self.test(test_data_set, test_batch_size)
                if best_test_accuracy is None or test_loss < best_test_accuracy:
                    best_test_accuracy = test_loss
                    self.best_test_weights = copy.deepcopy(self.model.state_dict())
        self.save_model(self.best_test_weights, "classifier_test_weights")
        self.save_model(self.best_weights, "classifier_train_weights")

    def test(self, test_data_set, batch_size: int):
        test_total_loss_cl = 0
        test_total_cl_acc = 0
        for images, _, labels in test_data_set:
            if self.gpu:
                images, labels = send_to_gpu(images, labels)
            images, labels = wrap_to_variable(images, labels)
            class_label = labels

            output_cl = self.model(images)

            grad_target = output_cl * class_label
            grad_target.backward(gradient=class_label * output_cl, retain_graph=True)

            loss_cl = self.loss_classifier(output_cl, class_label)

            test_total_loss_cl, test_total_cl_acc = self.__calculate_accuracy(output_cl, class_label, batch_size,
                                                                              loss_cl, test_total_loss_cl,
                                                                              test_total_cl_acc)
        test_size = len(test_data_set)

        test_total_loss_cl /= test_size
        test_total_cl_acc = (test_total_cl_acc / test_size) * 100.0
        text = 'TRAIN Loss_CL: %f, Accuracy_CL: %f%%' % (test_total_loss_cl, test_total_cl_acc)
        print(text)
        P.write_to_log(text)

        return test_total_loss_cl, test_total_cl_acc

    def save_model(self, weights, name="classifier-model"):
        try:
            name = name + datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S') + ".torch"
            saved_dir = os.path.join(P.base_data_dir, 'classifier_weights')
            os.makedirs(saved_dir, exist_ok=True)
            saved_file = os.path.join(saved_dir, name)
            torch.save(weights, saved_file)
            print("Save model: {}".format(name))
            P.write_to_log("Save model: {}".format(name))
        except Exception as e:
            print("Can't save model: {}".format(name), e)
            P.write_to_log("Can't save model: {}".format(name), e)

    def __calculate_accuracy(self, output_cl, class_label, batch_size, loss_cl, total_loss_cl, total_cl_acc):
        output_cl = output_cl.view((batch_size, self.classes // 2, 2))
        class_label = class_label.view(batch_size, self.classes // 2, 2)
        # может софтмакс надо брать по dim=1 пока точность неочень
        _, output_cl_softmax_indexes = F.softmax(output_cl, dim=2).max(dim=2)
        _, label_indexes = class_label.max(dim=2)
        cl_acc = torch.eq(output_cl_softmax_indexes, label_indexes).sum()

        total_loss_cl += scalar(loss_cl.sum()) / (batch_size)
        total_cl_acc += scalar(cl_acc.sum() / (class_label.sum()))
        return total_loss_cl, total_cl_acc
