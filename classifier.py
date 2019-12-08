"""
classify dataset
"""

import torch
import torch.nn.functional as F
import torchvision.models as m
import torch.nn as nn
import copy


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
        self.classes = classes
        self.model = m.vgg16(pretrained=True)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, classes)

        self.best_weights = copy.deepcopy(self.model.state_dict())

        if loss_classifier is None:
            self.loss_classifier = torch.nn.BCEWithLogitsLoss()

        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
        else:
            self.tensor_source = torch

    def train(self, train_data_set, epochs, batch_size, learning_rate=1e-6):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss_cl = 0
            total_cl_acc = 0
            for images, _, labels, _ in train_data_set:
                if self.gpu:
                    images, labels = send_to_gpu(images, labels)
                images, labels = wrap_to_variable(images, labels)
                for i in range(5):
                    class_label = labels[:, i, :]
                    output_cl = self.model(images)

                    grad_target = output_cl * class_label
                    grad_target.backward(gradient=class_label * output_cl, retain_graph=True)

                    self.model.zero_grad()
                    loss_cl = self.loss_classifier(output_cl, class_label)

                    loss_cl.backward()
                    optimizer.step()

                    _, output_cl_softmax_indexes = F.softmax(output_cl, dim=1).max(dim=1)
                    _, label_indexes = class_label.max(dim=1)
                    cl_acc = torch.eq(output_cl_softmax_indexes, label_indexes).sum()

                    total_loss_cl += scalar(loss_cl.sum()) / (batch_size * 5)
                    total_cl_acc += scalar(cl_acc.sum() / (class_label.sum() * 5))

            train_size = len(train_data_set)
            print('%i of %i EPOCHS, TEST Loss_CL: %f, Accuracy_CL: %f%%' %
                  (epoch, epochs, total_loss_cl / train_size, (total_cl_acc / train_size) * 100.0))
