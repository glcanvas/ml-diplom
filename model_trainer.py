"""
train and validate model
"""
from models import *
import torch
import torch.nn.functional as F
import torchvision.models as m
import torch.nn as nn
import copy
from optimizers import *

EPS = 1e-10
probability_threshold = 0.5


class Trainer:

    def __init__(self, description: str,
                 classes: int = 5,
                 gradient_layer_name: str = "features.28",
                 gpu: bool = False,
                 device: int = 0,
                 use_am: bool = False,
                 alpha: float = 1.0,
                 omega: float = 10.0,
                 sigma: float = 0.5,
                 loss_classifier=None
                 ):
        self.description = description
        self.classes = classes
        self.gradient_layer_name = gradient_layer_name
        self.gpu = gpu
        self.device = device
        self.use_am = use_am

        net = m.vgg16(pretrained=True)
        num_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_features, classes)

        # misc parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha

        self.model = GAIN(net, gradient_layer_name, classes, self.use_am, self.gpu, self.device, self.alpha, self.omega,
                          self.sigma)

        if self.gpu:
            self.model = self.model.cuda(self.device)

        if loss_classifier is None:
            self.loss_classifier = torch.nn.BCELoss()  # before use
            self.logits_loss_classifier = torch.nn.BCEWithLogitsLoss()

        self.best_weights = None

    def train(self, gain_train_set,
              classifier_train_set,
              test_set,
              epochs: int = 100,
              test_each_epochs: int = 4,
              save_test_roc_each_epoch: int = 12,
              save_train_roc_each_epochs: int = 10,
              pre_train_epoch: int = 25,
              learning_rate: float = 1e-6):
        self.model.training = True

        self.best_weights = copy.deepcopy(self.model.state_dict())
        best_loss = None
        best_test_loss = None

        # optimizer = AdamW(self.model.parameters())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):

            loss_total_sum = 0
            loss_cl_sum = 0
            loss_am_sum = 0
            loss_e_sum = 0
            accuracy_sum = 0

            gain_set_size = 0
            classifier_set_size = 0

            for images, segments, labels in gain_train_set:
                gain_set_size += images.size(0)
                if self.gpu:
                    images = images.cuda(self.device)
                    segments = segments.cuda(self.device)
                    labels = labels.cuda(self.device)
                loss_total_sum_, loss_cl_sum_, loss_am_sum_, loss_e_sum_, accuracy_sum_ = self.__gain_branch(epoch,
                                                                                                             pre_train_epoch,
                                                                                                             images,
                                                                                                             segments,
                                                                                                             labels)
                loss_total_sum += loss_total_sum_
                loss_cl_sum += loss_cl_sum_
                loss_am_sum += loss_am_sum_
                loss_e_sum += loss_e_sum_
                accuracy_sum += accuracy_sum_

                optimizer.step()
                torch.cuda.empty_cache()

            for images, _, labels in classifier_train_set:
                classifier_set_size += images.size(0)
                if self.gpu:
                    images = images.cuda(self.device)
                    labels = labels.cuda(self.device)
                loss_cl_sum_, accuracy_sum_ = self.__classifier_branch(
                    images,
                    labels)
                loss_cl_sum += loss_cl_sum_
                accuracy_sum += accuracy_sum_

                optimizer.step()
                torch.cuda.empty_cache()

            prefix = "PRETRAIN" if epoch < pre_train_epoch else "TRAIN"
            text = "{}={} Loss_CL={:.5f} Loss_AM={:.5f} Loss_E={:.5f} Loss_Total={:.5f} Accuracy_CL={:.5f} ".format(
                prefix,
                epoch,
                loss_cl_sum / (gain_set_size + classifier_set_size + EPS),
                loss_am_sum / (gain_set_size + EPS),
                loss_e_sum / (gain_set_size + EPS),
                loss_total_sum / (gain_set_size + EPS),
                accuracy_sum / (gain_set_size + classifier_set_size + EPS) * 1000.0
                # ,
                # f_1_score_text,
                # recall_score_text,
                # precision_score_text
            )
            print(text)

    def __gain_branch(self, epoch_number, pre_train_epochs, images, segments, labels):
        loss_total_sum = 0
        loss_cl_sum = 0
        loss_am_sum = 0
        loss_e_sum = 0
        accuracy_sum = 0

        for illness in range(0, self.classes):
            logits, logits_am, heatmap = self.model(images, labels, illness)

            loss_cl = self.logits_loss_classifier(logits, labels)
            # WHAT ?????????????????????
            loss_am = F.softmax(logits_am)
            loss_am, _ = loss_am.max(dim=1)
            loss_am = loss_am.sum() / loss_am.size(0)
            # WHAT ?????????????????????

            loss_e = (segments - heatmap) @ (segments - heatmap)
            loss_e = loss_e.sum() / segments.size(0)

            total_loss = loss_cl * 0.5 + loss_am * 0.5 + loss_e * 0.9

            if epoch_number % 2 == 0:
                total_loss.backward()
            else:
                # total_loss.backward()
                loss_e.backward()

            loss_total_sum += total_loss.cpu().item()
            loss_cl_sum += loss_cl.cpu().item()
            loss_am_sum += loss_am.cpu().item()
            loss_e_sum += loss_e.cpu().item()
            accuracy_sum += self.__calculate_accuracy(labels, logits).cpu().item()

        loss_total_sum /= images.size(0)
        loss_cl_sum /= images.size(0)
        loss_am_sum /= images.size(0)
        loss_e_sum /= images.size(0)
        accuracy_sum /= images.size(0)

        return loss_total_sum, loss_cl_sum, loss_am_sum, loss_e_sum, accuracy_sum

    def __classifier_branch(self, images, labels):
        logits = self.model.forvard_(images, labels)
        loss_cl = self.logits_loss_classifier(logits, labels) * 10
        loss_cl.backward()

        return loss_cl.cpu().item(), self.__calculate_accuracy(labels, logits).cpu().item()

    def __calculate_accuracy(self, labels, logits):

        probability_layer = nn.Sigmoid()
        probabilities = probability_layer(logits)

        model_labels = probabilities.clone()
        model_labels[model_labels < probability_threshold] = 0
        model_labels[model_labels >= probability_threshold] = 1

        accuracy = torch.eq(model_labels, labels).sum().float()
        accuracy /= (labels.size(0) * self.classes)

        return accuracy
