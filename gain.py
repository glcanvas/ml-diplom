import torch
import torch.nn.functional as F
import torchvision.models as m
import torch.nn as nn
import copy
import property as P
from datetime import datetime
import os
import utils

EPS = 1e-10
probability_threshold = 0.5


def scalar(tensor):
    return tensor.data.cpu().item()


def reduce_boundaries(boundaries: int, batch_size=10, huge: int = -10000):
    # assume BSx1x224x224
    zeros = torch.zeros((batch_size, 1, 224, 224))
    zeros[:, :, 0:boundaries, :] = huge
    zeros[:, :, 224 - boundaries:224, :] = huge
    zeros[:, :, :, 0:boundaries] = huge
    zeros[:, :, :, 224 - boundaries:224] = huge
    return zeros


class AttentionGAIN:
    # 28
    def __init__(self, description: str, classes: int, gradient_layer_name="features.28", weights=None,
                 gpu=False,
                 device=0,
                 loss_classifier=None,
                 usage_am_loss=False,
                 alpha=1,
                 omega=10,
                 sigma=0.5):
        # validation
        if not gradient_layer_name:
            raise ValueError('Missing required argument gradient_layer_name')
        if gpu and device is None:
            raise ValueError('Missing required argument device, but gpu enable')

        # set gpu options
        self.gpu = gpu
        self.device = device
        self.description = description
        self.usage_am_loss = usage_am_loss

        self.classes = classes

        # define model
        self.model = m.vgg16(pretrained=True)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, self.classes)

        self.best_weights = None
        self.best_test_weights = None

        # define loss classifier for classifier path
        # create loss function
        if loss_classifier is None:
            self.loss_cl = torch.nn.BCELoss()
            self.gain_loss_cl = torch.nn.BCEWithLogitsLoss()

        if weights:
            self.model.load_state_dict(weights)

        self.minus_mask = reduce_boundaries(8)
        if self.gpu:
            self.model = self.model.cuda(self.device)
            self.tensor_source = torch.cuda
            self.minus_mask = self.minus_mask.cuda(self.device)
        else:
            self.tensor_source = torch

        # wire up our hooks for heatmap creation
        self._register_hooks(gradient_layer_name)

        # misc parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha

        self.train_model_answers = [[] for _ in range(self.classes)]
        self.train_trust_answers = [[] for _ in range(self.classes)]
        self.train_probabilities = [[] for _ in range(self.classes)]

        self.test_model_answers = [[] for _ in range(self.classes)]
        self.test_trust_answers = [[] for _ in range(self.classes)]
        self.test_probabilities = [[] for _ in range(self.classes)]

    def _register_hooks(self, layer_name):
        # this wires up a hook that stores both the activation and gradient of the conv layer we are interested in
        def forward_hook(module, input_, output_):
            self._last_activation = output_

        def backward_hook(module, grad_in, grad_out):
            self._last_grad = grad_out[0]

        # locate the layer that we are concerned about
        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == layer_name:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % layer_name)

    def forward(self, data, label, segments, train_batch_size: int, ill_index: int):
        data, label, segments = self._convert_data_and_label(data, label, segments)
        return self._forward(data, label, segments, train_batch_size, ill_index)

    def train(self, rds, epochs, test_each_epochs: int, save_test_roc_each_epochs: int, save_train_roc_each_epochs: int,
              pre_train_epoch: int = 25, learning_rate=1e-6):

        self.best_weights = copy.deepcopy(self.model.state_dict())
        best_loss = None
        best_test_loss = None

        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(1, epochs, 1):

            loss_cl_sum_segm = 0
            loss_am_sum_segm = 0
            acc_cl_sum_segm = 0
            loss_sum_segm = 0
            loss_e_sum_segm = 0

            loss_cl_sum_no_segm = 0
            acc_cl_sum_no_segm = 0

            with_segments_elements = 0
            without_segments_elements = 0

            for images, segments, labels in rds['train_segment']:
                # тренирую здесь с использованием сегментов
                with_segments_elements += images.shape[0]
                loss_sum_segm, loss_cl_sum_segm, loss_am_sum_segm, acc_cl_sum_segm, loss_e_sum_segm = self.__train_gain_branch(
                    epoch,
                    pre_train_epoch,
                    images,
                    segments,
                    labels,
                    opt,
                    loss_sum_segm,
                    loss_cl_sum_segm,
                    loss_am_sum_segm,
                    acc_cl_sum_segm,
                    loss_e_sum_segm)

            for images, segments, labels in rds['train_classifier']:
                without_segments_elements += images.shape[0]
                loss_cl_sum_no_segm, acc_cl_sum_no_segm = self.__train_classifier_branch(images, labels, opt,
                                                                                         loss_cl_sum_no_segm,
                                                                                         acc_cl_sum_no_segm)

            last_acc_total = acc_cl_sum_segm / self.classes
            last_acc_total += acc_cl_sum_no_segm
            last_acc_total /= (len(rds['train_segment']) + len(rds['train_classifier']))

            loss_cl_sum_total = loss_cl_sum_segm / self.classes
            loss_cl_sum_total += loss_cl_sum_no_segm
            loss_cl_sum_total /= (len(rds['train_segment']) + len(rds['train_classifier']))

            f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                             self.train_trust_answers,
                                                                                             self.train_model_answers)
            prefix = "PRETRAIN" if epoch < pre_train_epoch else "TRAIN"
            text = "{}={} Loss_CL={:.5f} Loss_AM={:.5f} Loss_E={:.5f} Loss_Total={:.5f} Accuracy_CL={:.5f} " \
                   "{} {} {} ".format(
                prefix,
                epoch,
                loss_cl_sum_total,
                loss_am_sum_segm / (with_segments_elements * self.classes + EPS),
                loss_e_sum_segm / (with_segments_elements * self.classes + EPS),
                loss_sum_segm / (with_segments_elements * self.classes + EPS),
                last_acc_total * 100.0,
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
                test_loss, _ = self.test(rds['test'], epoch, save_test_roc_each_epochs)
                if best_test_loss is None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.best_test_weights = copy.deepcopy(self.model.state_dict())

            if best_loss is None or loss_sum_segm < best_loss:
                best_loss = loss_sum_segm
                self.best_weights = copy.deepcopy(self.model.state_dict())

            self.train_model_answers = [[] for _ in range(self.classes)]
            self.train_trust_answers = [[] for _ in range(self.classes)]
            self.train_probabilities = [[] for _ in range(self.classes)]

            self.test_model_answers = [[] for _ in range(self.classes)]
            self.test_trust_answers = [[] for _ in range(self.classes)]
            self.test_probabilities = [[] for _ in range(self.classes)]

        self.save_model(self.best_test_weights, "gain_test_weights")
        self.save_model(self.best_weights, "gain_train_weights")

    def __train_gain_branch(self, current_epoch: int, pre_train_epoch: int, images, segments, labels,
                            optimizer, loss_sum, loss_cl_sum, am_sum, acc_cl_sum, e_sum):

        train_batch_size = images.shape[0]
        self.minus_mask = reduce_boundaries(8, train_batch_size)
        if self.gpu:
            images = images.cuda(self.device)  # bs x 3 x 224 x 224
            segments = segments.cuda(self.device)  # bs x 1 x 224 x 224
            labels = labels.cuda(self.device)  # bs x 5
            self.minus_mask = self.minus_mask.cuda(self.device)

        for ill_index in range(0, self.classes):
            total_loss, loss_cl, loss_am, loss_e, output_cl, output_probability, acc_cl, _, _, _ = \
                self.forward(images,
                             labels,
                             segments[:, ill_index],
                             train_batch_size,
                             ill_index)

            self.save_train_data(labels, output_cl, output_probability)

            loss_sum += scalar(total_loss)
            loss_cl_sum += scalar(loss_cl)
            am_sum += scalar(loss_am)
            acc_cl_sum += scalar(acc_cl)
            e_sum += scalar(loss_e)

            if current_epoch <= pre_train_epoch:
                loss_cl.backward()
            else:
                total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        return loss_sum, loss_cl_sum, am_sum, acc_cl_sum, e_sum

    def __train_classifier_branch(self, images, labels, optimizer, loss_cl_sum, acc_cl_sum):
        if self.gpu:
            images = images.cuda(self.device)  # bs x 3 x 224 x 224
            labels = labels.cuda(self.device)  # bs x 10
        output_cl_model = self.model(images)

        sigmoid = nn.Sigmoid()  # used for calculate accuracy
        output_cl = sigmoid(output_cl_model)
        loss_cl = self.loss_cl(output_cl, labels)

        loss_cl.backward()
        optimizer.step()
        self.model.zero_grad()

        batch_size = images.shape[0]

        output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, output_cl, batch_size)

        self.save_train_data(labels, output_cl, output_probability)

        # accumulate information
        acc_cl_sum += scalar(cl_acc.sum())
        loss_cl_sum += scalar(loss_cl.sum())

        return loss_cl_sum, acc_cl_sum

    # тестирование взять с классификатора(взял)
    def test(self, test_data_set, epoch: int, save_test_roc_each_epoch: int):
        test_total_loss_cl = 0
        test_total_cl_acc = 0
        for images, _, labels in test_data_set:
            batch_size = images.shape[0]
            if self.gpu:
                # images, labels = send_to_gpu(images, labels)
                images = images.cuda(self.device)
                labels = labels.cuda(self.device)

            output_cl_model = self.model(images)

            sigmoid = nn.Sigmoid()  # used for calculate accuracy
            output_cl = sigmoid(output_cl_model)
            loss_cl = self.loss_cl(output_cl, labels)

            output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, output_cl, batch_size)

            self.save_test_data(labels, output_cl, output_probability)

            test_total_loss_cl += scalar(loss_cl.sum()) / batch_size
            test_total_cl_acc += scalar(cl_acc)

        f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                         self.test_trust_answers,
                                                                                         self.test_model_answers)
        test_size = len(test_data_set)
        test_total_loss_cl /= test_size
        test_total_cl_acc = (test_total_cl_acc / test_size) * 100.0
        text = 'TEST Loss_CL={:.5f} Accuracy_CL={:.5f} {} {} {} '.format(test_total_loss_cl,
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

        P.write_to_log(text)
        print(text)

        return test_total_loss_cl, test_total_cl_acc

    def _attention_map_forward(self, data, label, ill_index: int):

        output_cl_model = self.model(data)
        # тут раньше была сумма, но не думаю что она нужна в данном случае
        grad_target = output_cl_model * label
        ones = torch.ones(label.shape[0])
        if self.gpu:
            ones = ones.cuda(self.device)

        sigmoid = nn.Sigmoid()  # used for calculate accuracy
        output_cl = sigmoid(output_cl_model)
        loss_cl = self.gain_loss_cl(output_cl_model, label)

        grad_target[:, ill_index].backward(gradient=ones, retain_graph=True)
        self.model.zero_grad()

        # Eq 1
        # grad = self._last_grad
        # w_c = F.avg_pool2d(grad, (grad.shape[-2], grad.shape[-1]), 1)
        # w_c_new_shape = (1, w_c.shape[0] * w_c.shape[1], w_c.shape[2], w_c.shape[3])
        # w_c = w_c.view(w_c_new_shape).unsqueeze(0)

        # Eq 2
        # TODO this support batching !!!
        # weights = self._last_activation
        # weights_new_shape = (1, weights.shape[0] * weights.shape[1], weights.shape[2], weights.shape[3])
        # weights = weights.view(weights_new_shape).unsqueeze(0)

        last_grad_GAP = F.adaptive_avg_pool2d(self._last_grad, 1)  # weights
        # self._last_activation = F.adaptive_avg_pool2d(self._last_activation, 1)  # fl
        A_c = torch.mul(self._last_activation, last_grad_GAP).sum(dim=1, keepdim=True)
        A_c = F.relu(A_c)
        A_c = F.upsample_bilinear(A_c, size=data.size()[2:])
        # gcam = F.relu(F.conv2d(weights, w_c))
        # A_c = F.upsample(gcam, size=data.size()[2:], mode='bilinear')

        return output_cl, loss_cl, A_c

    def _mask_image(self, gcam, image):
        gcam_min = gcam.min()
        gcam_max = gcam.max()
        scaled_gcam = (gcam - gcam_min) / (gcam_max - gcam_min + EPS)
        mask = F.sigmoid(self.omega * (scaled_gcam - self.sigma))
        masked_image = image - (image * mask)
        return masked_image, mask

    def _forward(self, data, label, segment, train_batch_size: int, ill_index: int):
        # TODO normalize elsewhere, this feels wrong (вроде здесь ок)
        output_cl, loss_cl, gcam = self._attention_map_forward(data, label, ill_index)

        # Eq 4
        # TODO this support batching
        I_star, mask = self._mask_image(gcam, data)

        if self.gpu:
            loss_am = torch.tensor([0.0]).cuda(self.device)
        else:
            loss_am = torch.tensor([0.0])

        if self.usage_am_loss:
            output_am = self.model(I_star)
            # Eq 5
            loss_am = F.sigmoid(output_am) * label
            loss_am = loss_am.sum() / label.sum().type(self.tensor_source.FloatTensor)

        updated_segment_mask = segment * self.omega + self.minus_mask
        loss_e = ((mask - updated_segment_mask) @ (mask - updated_segment_mask)).sum()

        # Eq 6
        total_loss = loss_cl + loss_e * 100 + self.alpha * loss_am

        output_probability, output_cl, cl_acc = self.calculate_accuracy(label, output_cl, train_batch_size)

        return total_loss, loss_cl, loss_am, loss_e, output_cl, output_probability, cl_acc, gcam, I_star, mask

    def _convert_data_and_label(self, data, label, segments):
        # converts our data and label over to optional gpu
        if self.gpu:
            data = data.cuda(self.device)
            label = label.cuda(self.device)
            segments = segments.cuda(self.device)

        return data, label, segments

    def save_model(self, weights, name="gain-model"):
        try:
            name = name + self.description + "_date-" + datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S') + ".torch"
            saved_dir = os.path.join(P.base_data_dir, 'gain_weights')
            os.makedirs(saved_dir, exist_ok=True)
            saved_file = os.path.join(saved_dir, name)
            torch.save(weights, saved_file)
            print("Save model: {}".format(name))
            P.write_to_log("Save model: {}".format(name))
        except Exception as e:
            print("Can't save model: {}".format(name), e)
            P.write_to_log("Can't save model: {}".format(name), e)

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

    def calculate_accuracy(self, labels, output_cl, batch_size):
        output_probability = output_cl.clone()
        output_cl[output_cl >= probability_threshold] = 1
        output_cl[output_cl < probability_threshold] = 0
        cl_acc = torch.eq(output_cl, labels).sum().float()
        cl_acc /= (batch_size * self.classes + EPS)
        return output_probability, output_cl, cl_acc
