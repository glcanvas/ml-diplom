import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import os
import re
import io
import json
import math
import models
from torchvision import transforms, models


def scalar(tensor):
    return tensor.data.cpu().item()


class AttentionGAIN:
    def __init__(self, model_type=None, gradient_layer_name=None, weights=None, heatmap_dir=None,
                 saved_model_dir=None, epoch=0, gpu=False, alpha=1, omega=10, sigma=0.5, labels=None,
                 input_channels=None, input_dims=None, batch_norm=True):

        # set gpu options
        self.gpu = gpu

        # define model
        self.model_type = model_type
        self.model = models.alexnet(
            pretrained=True)  # models.get_model(self.model_type, len(labels), batch_norm=batch_norm, num_channels=input_channels)
        if weights:
            self.model.load_state_dict(weights)
            self.epoch = epoch
        elif epoch > 0:
            raise ValueError('epoch_offset > 0, but no weights were supplied')

        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
        else:
            self.tensor_source = torch

        # wire up our hooks for heatmap creation
        self._register_hooks(gradient_layer_name)

        # create loss function
        # TODO make this configurable
        self.loss_cl = torch.nn.BCEWithLogitsLoss()

        # output directory setup
        self.heatmap_dir = heatmap_dir
        if self.heatmap_dir:
            self.heatmap_dir = os.path.abspath(self.heatmap_dir)

        self.saved_model_dir = saved_model_dir
        if self.saved_model_dir:
            self.saved_model_dir = os.path.abspath(saved_model_dir)

        # misc. parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha
        self.labels = labels
        self.input_channels = input_channels
        self.input_dims = input_dims
        self.epoch = epoch

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

    def forward(self, data, label):
        data, label = None  # self._convert_data_and_label(data, label)
        return self._forward(data, label)


    def train(self, rds, epochs, serialization_format, pretrain_epochs=10, learning_rate=1e-5,
              test_every_n_epochs=5, num_heatmaps=1):

        last_acc = 0
        max_acc = 0
        pretrain_finished = False
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for i in range(self.epoch, epochs, 1):
            self.epoch = i
            pretrain_finished = pretrain_finished or \
                                i > pretrain_epochs
            loss_cl_sum = 0
            loss_am_sum = 0
            acc_cl_sum = 0
            total_loss_sum = 0
            # train
            for sample in rds.datasets['train']:
                total_loss, loss_cl, loss_am, probs, acc_cl, A_c, _ = self.forward(sample['image'],
                                                                                   sample['label/onehot'])
                total_loss_sum += scalar(total_loss)
                loss_cl_sum += scalar(loss_cl)
                loss_am_sum += scalar(loss_am)
                acc_cl_sum += scalar(acc_cl)

                # Backprop selectively based on pretraining/training
                if pretrain_finished:
                    print_prefix = 'TRAIN'
                    total_loss.backward()
                else:
                    print_prefix = 'PRETRAIN'
                    loss_cl.backward()

                opt.step()
            train_size = len(rds.datasets['train'])
            last_acc = acc_cl_sum / train_size
            print('%s Epoch %i, Loss_CL: %f, Loss_AM: %f, Loss Total: %f, Accuracy_CL: %f%%' %
                  (print_prefix, (i + 1), loss_cl_sum / train_size, loss_am_sum / train_size,
                   total_loss_sum / train_size, last_acc * 100.0))

            if (i + 1) % test_every_n_epochs == 0:
                # test
                loss_cl_sum = 0
                loss_am_sum = 0
                acc_cl_sum = 0
                total_loss_sum = 0
                heatmap_count = 0
                for sample in rds.datasets['test']:
                    data = sample['image']
                    label_onehot = sample['label/onehot']
                    label = sample['label/idx']

                    # test
                    total_loss, loss_cl, loss_am, prob, acc_cl, A_c, I_star = self.forward(data, label_onehot)

                    total_loss_sum += scalar(total_loss)
                    loss_cl_sum += scalar(loss_cl)
                    loss_am_sum += scalar(loss_am)
                    acc_cl_sum += scalar(acc_cl)

                    if heatmap_count < num_heatmaps:
                        self._maybe_save_heatmap(data[0], label[0], A_c[0], I_star[0], i + 1, heatmap_count)
                        heatmap_count += 1

                test_size = len(rds.datasets['test'])
                avg_acc = acc_cl_sum / test_size
                print('TEST Loss_CL: %f, Loss_AM: %f, Loss_Total: %f, Accuracy_CL: %f%%' %
                      (loss_cl_sum / test_size, loss_am_sum / test_size, total_loss_sum / test_size, avg_acc * 100.0))

    def _attention_map_forward(self, data, label):
        output_cl = self.model(data)
        grad_target = (output_cl * label).sum()
        grad_target.backward(gradient=label * output_cl, retain_graph=True)

        self.model.zero_grad() 

        # Eq 1
        grad = self._last_grad
        w_c = F.avg_pool2d(self._last_grad, (self._last_grad.shape[-2], self._last_grad.shape[-1]), 1)
        w_c_new_shape = (w_c.shape[0] * w_c.shape[1], w_c.shape[2], w_c.shape[3])
        w_c = w_c.view(w_c_new_shape).unsqueeze(0)

        # Eq 2
        # TODO this doesn't support batching
        weights = self._last_activation
        weights_new_shape = (weights.shape[0] * weights.shape[1], weights.shape[2], weights.shape[3])
        weights = weights.view(weights_new_shape).unsqueeze(0)

        gcam = F.relu(F.conv2d(weights, w_c))
        A_c = F.upsample(gcam, size=data.size()[2:], mode='bilinear')

        loss_cl = self.loss_cl(output_cl, label)

        return output_cl, loss_cl, A_c

    def _forward(self, data, label):
        # TODO normalize elsewhere, this feels wrong
        output_cl, loss_cl, gcam = self._attention_map_forward(data, label)
        output_cl_softmax = F.softmax(output_cl, dim=1)

        # Eq 4
        # TODO this currently doesn't support batching, maybe add that
        I_star = self._mask_image(gcam, data)

        output_am = self.model(I_star)

        # Eq 5
        loss_am = F.sigmoid(output_am) * label
        loss_am = loss_am.sum() / label.sum().type(self.tensor_source.FloatTensor)

        # Eq 6
        total_loss = loss_cl + self.alpha * loss_am

        cl_acc = output_cl_softmax.max(dim=1)[1] == label.max(dim=1)[1]
        cl_acc = cl_acc.type(self.tensor_source.FloatTensor).mean()

        return total_loss, loss_cl, loss_am, output_cl_softmax, cl_acc, gcam, I_star

    def _mask_image(self, gcam, image):
        gcam_min = gcam.min()
        gcam_max = gcam.max()
        scaled_gcam = (gcam - gcam_min) / (gcam_max - gcam_min)
        # Eq 4
        mask = F.sigmoid(self.omega * (scaled_gcam - self.sigma)).squeeze()
        #Eq 3
        masked_image = image - (image * mask)


        return masked_image
