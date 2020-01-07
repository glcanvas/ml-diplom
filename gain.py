"""
train and validate model
"""
import torch
import torch.nn.functional as F
import torchvision.models as m
import torch.nn as nn
import copy
from optimizers import *
import property as P

EPS = 1e-10
probability_threshold = 0.5


class GAIN(nn.Module):

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
        super(GAIN, self).__init__()
        self.description = description
        self.classes = classes
        self.gradient_layer_name = gradient_layer_name
        self.gpu = gpu
        self.device = device
        self.use_am = use_am

        net = m.vgg16(pretrained=True)
        num_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_features, classes)
        self.model = net

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None
        # Register hooks
        self._register_hooks(gradient_layer_name)

        # misc parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha

        if self.gpu:
            self.model = self.model.cuda(self.device)

        if loss_classifier is None:
            self.loss_classifier = torch.nn.BCELoss()  # before use
            self.logits_loss_classifier = torch.nn.BCEWithLogitsLoss()

        self.best_weights = None

    def _register_hooks(self, grad_layer):
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def train_model(self, gain_train_set,
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

        optimizer = AdamW(self.model.parameters())
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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
                                                                                                             labels,
                                                                                                             optimizer)
                loss_total_sum += loss_total_sum_
                loss_cl_sum += loss_cl_sum_
                loss_am_sum += loss_am_sum_
                loss_e_sum += loss_e_sum_
                accuracy_sum += accuracy_sum_

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
                accuracy_sum / (gain_set_size + classifier_set_size + EPS)
                # ,
                # f_1_score_text,
                # recall_score_text,
                # precision_score_text
            )
            print(text)
            P.write_to_log(text)
            if epoch % test_each_epochs == 0:
                self.test(test_set)

    def __gain_branch(self, epoch_number, pre_train_epochs, images, segments, labels, optimizer):
        loss_total_sum = 0
        loss_cl_sum = 0
        loss_am_sum = 0
        loss_e_sum = 0
        accuracy_sum = 0

        for illness in range(0, self.classes):
            logits, logits_am, heatmap = self.forward(images, labels, illness)

            loss_cl = self.logits_loss_classifier(logits, labels)
            # WHAT ?????????????????????
            loss_am = F.softmax(logits_am)
            loss_am, _ = loss_am.max(dim=1)
            loss_am = loss_am.sum() / loss_am.size(0)
            # WHAT ?????????????????????

            activation_layer = nn.Softmax()
            loss_e = (segments - activation_layer(heatmap)) * (segments - activation_layer(heatmap))
            loss_e = loss_e.sum() / segments.size(0)

            total_loss = loss_cl * 4.0 + loss_am * 0.5 + loss_e * 0.9

            if epoch_number < pre_train_epochs:
                loss_e.backward()
            else:
                total_loss.backward()

            loss_total_sum += total_loss.cpu().item()
            loss_cl_sum += loss_cl.cpu().item()
            loss_am_sum += loss_am.cpu().item()
            loss_e_sum += loss_e.cpu().item()
            accuracy_sum += self.__calculate_accuracy(labels, logits).cpu().item()

            optimizer.step()
            torch.cuda.empty_cache()

        loss_total_sum /= images.size(0)
        loss_cl_sum /= images.size(0)
        loss_am_sum /= images.size(0)
        loss_e_sum /= images.size(0)
        accuracy_sum /= images.size(0)

        return loss_total_sum, loss_cl_sum, loss_am_sum, loss_e_sum, accuracy_sum

    def forward(self, images, labels, index):

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to  unstable validation score. It is better to visualize attention maps at the testset

        with torch.enable_grad():

            _, _, img_h, img_w = images.size()

            self.model.train(True)
            logits = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            # TODO thought not needed here, it' will be ok if labels non-binary 
            # labels_ohe = self.__to_ohe(labels)

            # if self.gpu:
            #     labels_ohe = labels_ohe.cuda(self.device)
            # else:
            #    labels_ohe = labels_ohe.cpu()

            ones = torch.ones(labels.shape[0]).cpu()
            if self.gpu:
                ones = ones.cuda(self.device)
            # Здесь переписал так
            # так, как мы так обсуждали
            # gradient = logits * labels
            # grad_logits = logits * labels  # .sum()  # BS x num_classes
            # grad_logits = logits
            # grad_logits[:, index].backward(gradient=gradient[:, index], retain_graph=True)
            # TODO I suppose line bellow not needed here, but who knows
            # self.model.zero_grad()

            # gradient = logits * labels
            # gradient[:, index].backward(gradient=ones, retain_graph=True)
            gradient = logits * labels
            grad_logits = (logits * labels) # .sum()  # BS x num_classes
            grad_logits.backward(gradient=gradient, retain_graph=True)
            self.model.zero_grad()

        backward_features = self.backward_features  # BS x C x H x W

        # Eq 2
        fl = self.feed_forward_features  # BS x C x H x W

        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        Ac = F.upsample_bilinear(Ac, size=images.size()[2:])
        heatmap = Ac

        Ac_min = Ac.min()
        Ac_max = Ac.max()
        scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
        mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = images - images * mask

        if self.use_am:
            logits_am = self.model(masked_image)
        else:
            logits_am = torch.tensor([[0]]).float()
            if self.gpu:
                logits_am = logits_am.cuda(self.device)

        return logits, logits_am, heatmap

    def __classifier_branch(self, images, labels):
        with torch.enable_grad():
            self.model.train(True)
            logits = self.model(images)
            self.model.zero_grad()
            # gradient = logits * labels
            # grad_logits = (logits * labels).sum()  # BS x num_classes
            # https://stackoverflow.com/questions/57248777/backward-function-in-pytorch
            # grad_logits.backward(retain_graph=True)

            # self.model.zero_grad()
            loss_cl = self.logits_loss_classifier(logits, labels)
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

    def __to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.classes), requires_grad=True)
        labels = labels.long()
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        ohe = torch.autograd.Variable(ohe)

        return ohe

    def test(self, test_set):
        loss_cl_sum = 0
        accuracy_sum = 0
        test_set_size = 0
        for images, _, labels in test_set:
            test_set_size += images.size(0)
            if self.gpu:
                images = images.cuda(self.device)
                labels = labels.cuda(self.device)

            self.model.train(False)
            logits = self.model(images)
            loss_cl = self.logits_loss_classifier(logits, labels)
            loss_cl = loss_cl.cpu().item()
            accuracy = self.__calculate_accuracy(labels, logits).cpu().item()

            loss_cl_sum += loss_cl
            accuracy_sum += accuracy

        loss_cl_sum /= test_set_size
        accuracy_sum /= test_set_size
        text = "TEST Loss_CL={:.5f} Accuracy_CL={:.5f} ".format(loss_cl_sum, accuracy_sum)
        print(text)
        P.write_to_log(text)
        return loss_cl_sum, accuracy_sum
