"""
classify dataset
"""

import torch
import torchvision.models as m
import property as P
import torch.nn as nn
import copy
from datetime import datetime
import os
import utils

probability_threshold = 0.5


def scalar(tensor):
    return tensor.data.cpu().item()


def send_to_gpu(*args) -> tuple:
    result = []
    for i in args:
        result.append(i.cuda())
    return (*result,)


def send_to_cpu(*args) -> tuple:
    result = []
    for i in args:
        result.append(i.cpu())
    return (*result,)


class Classifier:

    def __init__(self, description: str, classes: int, gpu=False, loss_classifier=None):
        self.gpu = gpu
        self.description = description
        # здесь * 2 так как каждой метке соответсвует бинарное значение -- да, нет в самом деле я сделал так для
        # классификации так как сделать по другому не знаю
        self.classes = classes
        self.model = m.vgg16(pretrained=True)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, self.classes)

        self.best_weights = copy.deepcopy(self.model.state_dict())
        self.best_test_weights = copy.deepcopy(self.model.state_dict())

        if loss_classifier is None:
            self.loss_classifier = torch.nn.BCELoss()

        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
        else:
            self.tensor_source = torch

        self.train_model_answers = [[] for _ in range(self.classes)]
        self.train_trust_answers = [[] for _ in range(self.classes)]
        self.train_probabilities = [[] for _ in range(self.classes)]

        self.test_model_answers = [[] for _ in range(self.classes)]
        self.test_trust_answers = [[] for _ in range(self.classes)]
        self.test_probabilities = [[] for _ in range(self.classes)]

    def train(self, epochs: int, test_each_epochs: int, save_test_roc_each_epochs: int, save_train_roc_each_epochs: int,
              train_data_set, test_data_set,
              learning_rate=1e-6):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        best_loss = None
        best_test_loss = None

        for epoch in range(1, epochs + 1):
            total_loss_cl = 0
            total_cl_acc = 0
            set_size = 0
            for images, _, labels in train_data_set:
                set_size += 1 # images.size(0)
                if self.gpu:
                    images, labels = send_to_gpu(images, labels)
                # images, labels = wrap_to_variable(images, labels)
                class_label = labels
                train_batch_size = labels.shape[0]
                self.model.zero_grad()
                output_cl = self.model(images)

                sigmoid = nn.Sigmoid()  # used for calculate accuracy
                output_cl = sigmoid(output_cl)
                loss_cl = self.loss_classifier(output_cl, class_label)

                loss_cl.backward()
                optimizer.step()

                total_loss_cl, total_cl_acc, output_cl, output_probability = self.__calculate_accuracy(output_cl,
                                                                                                       class_label,
                                                                                                       train_batch_size,
                                                                                                       loss_cl,
                                                                                                       total_loss_cl,
                                                                                                       total_cl_acc)

                labels = labels.cpu()
                output_cl = output_cl.cpu()
                output_probability = output_probability.cpu()
                for i in range(output_cl.shape[1]):
                    self.train_trust_answers[i].extend(labels[:, i].tolist())
                    self.train_model_answers[i].extend(output_cl[:, i].tolist())
                    self.train_probabilities[i].extend(output_probability[:, i].tolist())

                torch.cuda.empty_cache()

            if best_loss is None or total_loss_cl < best_loss:
                best_loss = total_loss_cl
                self.best_weights = copy.deepcopy(self.model.state_dict())

            f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                             self.train_trust_answers,
                                                                                             self.train_model_answers)
            text = "TRAIN={} Loss_CL={:.10f} Accuracy_CL_Percent={:.5f} {} {} {} ".format(epoch,
                                                                                          total_loss_cl / set_size,
                                                                                          total_cl_acc / set_size,
                                                                                          f_1_score_text,
                                                                                          recall_score_text,
                                                                                          precision_score_text)
            if epoch % save_train_roc_each_epochs == 0:
                auc_roc = "auc_roc="
                for idx, i in enumerate(self.train_trust_answers):
                    auc_roc += "trust_{}={}".format(idx, ",".join(list(map(lambda x: "{}".format(x), i))))
                for idx, i in enumerate(self.train_probabilities):
                    auc_roc += "prob_{}={}".format(idx, ",".join(list(map(lambda x: "{:.5f}".format(x), i))))
                text += auc_roc

            print(text)
            P.write_to_log(text)
            if epoch % test_each_epochs == 0:
                test_loss, _ = self.test(test_data_set, epoch, save_test_roc_each_epochs)
                if best_test_loss is None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.best_test_weights = copy.deepcopy(self.model.state_dict())

            self.train_model_answers = [[] for _ in range(self.classes)]
            self.train_trust_answers = [[] for _ in range(self.classes)]
            self.test_model_answers = [[] for _ in range(self.classes)]
            self.test_trust_answers = [[] for _ in range(self.classes)]
            self.train_probabilities = [[] for _ in range(self.classes)]
            self.test_probabilities = [[] for _ in range(self.classes)]

        self.save_model(self.best_test_weights, "classifier_test_weights")
        self.save_model(self.best_weights, "classifier_train_weights")

    def test(self, test_data_set, epoch: int, save_test_roc_each_epoch: int):
        test_total_loss_cl = 0
        test_total_cl_acc = 0
        test_size = 0
        for images, _, labels in test_data_set:
            test_size += 1 # images.size(0)
            if self.gpu:
                images, labels = send_to_gpu(images, labels)
            class_label = labels
            batch_size = labels.shape[0]
            output_cl = self.model(images)

            grad_target = output_cl * class_label
            grad_target.backward(gradient=class_label * output_cl, retain_graph=True)

            sigmoid = nn.Sigmoid()  # used for calculate accuracy
            output_cl = sigmoid(output_cl)
            loss_cl = self.loss_classifier(output_cl, class_label)

            test_total_loss_cl, test_total_cl_acc, output_cl, output_probability = self.__calculate_accuracy(output_cl,
                                                                                                             class_label,
                                                                                                             batch_size,
                                                                                                             loss_cl,
                                                                                                             test_total_loss_cl,
                                                                                                             test_total_cl_acc)
            labels = labels.cpu()
            output_cl = output_cl.cpu()
            output_probability = output_probability.cpu()
            for i in range(output_cl.shape[1]):
                self.test_trust_answers[i].extend(labels[:, i].tolist())
                self.test_model_answers[i].extend(output_cl[:, i].tolist())
                self.test_probabilities[i].extend(output_probability[:, i].tolist())

        # test_size = len(test_data_set)
        test_total_loss_cl /= test_size
        test_total_cl_acc /= test_size

        f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                         self.test_trust_answers,
                                                                                         self.test_model_answers)
        text = "TEST Loss_CL={:.10f} Accuracy_CL_Percent={:.5f} {} {} {} ".format(test_total_loss_cl,
                                                                                 test_total_cl_acc,
                                                                                 f_1_score_text,
                                                                                 recall_score_text,
                                                                                 precision_score_text)
        if epoch % save_test_roc_each_epoch == 0:
            auc_roc = "auc_roc="
            for idx, i in enumerate(self.test_trust_answers):
                auc_roc += "trust_{}={}".format(idx, ",".join(list(map(lambda x: "{}".format(x), i))))
            for idx, i in enumerate(self.test_probabilities):
                auc_roc += "prob_{}={}".format(idx, ",".join(list(map(lambda x: "{:.5f}".format(x), i))))
            text += auc_roc

        print(text)
        P.write_to_log(text)

        return test_total_loss_cl, test_total_cl_acc

    def save_model(self, weights, name="classifier-model"):
        try:
            name = name + self.description + datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S') + ".torch"
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
        output_probability = output_cl.clone()
        output_cl[output_cl >= probability_threshold] = 1
        output_cl[output_cl < probability_threshold] = 0
        cl_acc = torch.eq(output_cl, class_label).sum()

        total_loss_cl += scalar(loss_cl.sum()) / batch_size
        total_cl_acc += scalar(cl_acc.sum()) / (batch_size * self.classes)
        return total_loss_cl, total_cl_acc, output_cl, output_probability
