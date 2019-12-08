import torch
import torch.nn.functional as F
import torchvision.models as m
import torch.nn as nn
import visualize
import time
import copy


def scalar(tensor):
    return tensor.data.cpu().item()


def reduce_boundaries(boundaries: int):
    # assume BSx1x224x224
    zeros = torch.zeros((10, 1, 224, 224))
    for bs in zeros:
        for chanel in bs:
            for idx_i in range(0, 224):
                for idx_j in range(0, 224):
                    if idx_i < boundaries or idx_i + boundaries >= 224 or idx_j < boundaries or idx_j + boundaries >= 224:
                        chanel[idx_i][idx_j] = -10000
    return zeros


class AttentionGAIN:
    # 28
    def __init__(self, gradient_layer_name="features.28", weights=None, epoch=0, gpu=False, alpha=1, omega=10,
                 sigma=0.5):
        # validation
        if not gradient_layer_name:
            raise ValueError('Missing required argument gradient_layer_name')

        # set gpu options
        self.gpu = gpu

        # define model
        # self.model_type = model_type
        self.model = m.vgg16(pretrained=True)
        # models.get_model(self.model_type, len(labels), batch_norm=batch_norm, num_channels=input_channels)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 1)

        self.best_weights = None

        if weights:
            self.model.load_state_dict(weights)
            self.epoch = epoch
        elif epoch > 0:
            raise ValueError('epoch_offset > 0, but no weights were supplied')

        self.minus_mask = reduce_boundaries(8)
        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
            self.minus_mask = self.minus_mask.cuda()
        else:
            self.tensor_source = torch

        # wire up our hooks for heatmap creation
        self._register_hooks(gradient_layer_name)

        # create loss function
        # TODO make this configurable
        self.loss_cl = torch.nn.BCEWithLogitsLoss()

        # misc. parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha
        self.epoch = epoch

        self.epochs_trained = 0

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

    def forward(self, data, label, segments):
        data, label, segments = self._convert_data_and_label(data, label, segments)
        return self._forward(data, label, segments)

    def train(self, rds, epochs, pretrain_epochs=4, learning_rate=1e-6):
        # TODO dynamic optimizer selection
        # self._check_dataset_compatability(rds)
        self.best_weights = copy.deepcopy(self.model.state_dict())
        best_loss = 1e40

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
            loss_e_sum = 0
            print_prefix = None
            # train
            for images, segments, labels, _ in rds['train']:

                if self.gpu:
                    images = images.cuda()
                    segments = segments.cuda()
                    labels = labels.cuda()
                total_loss, loss_cl, loss_am, loss_e, probs, acc_cl, _, _, _ = self.forward(images, labels, segments)
                total_loss_sum += scalar(total_loss)
                loss_cl_sum += scalar(loss_cl)
                # loss_am_sum += scalar(loss_am)
                acc_cl_sum += scalar(acc_cl)
                loss_e_sum += scalar(loss_e)

                # Backprop selectively based on pretraining/training
                # if pretrain_finished:
                #    print_prefix = 'TRAIN'
                total_loss.backward()
                # else:
                #    print_prefix = 'PRETRAIN'
                #    loss_cl.backward()

                opt.step()
            train_size = len(rds['train'])
            last_acc = acc_cl_sum / train_size
            print('%s Epoch %i, Loss_CL: %f, Loss_AM: %f, Loss E: %f, Loss Total: %f, Accuracy_CL: %f%%' %
                  (print_prefix, (i + 1), loss_cl_sum / train_size, loss_am_sum / train_size, loss_e_sum / train_size,
                   total_loss_sum / train_size, last_acc * 100.0))
            if total_loss_sum < best_loss:
                best_loss = total_loss_sum
                self.best_weights = copy.deepcopy(self.model.state_dict())
        self.epochs_trained = epochs
        # torch.save(self.best_weights, "./weights" + str(time.time_ns()))

    def test(self, rds):

        self.model = m.vgg16()
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 1)

        self.model.load_state_dict(self.best_weights)
        self.model.eval()

        if self.gpu:
            self.model = self.model.cuda()

        # test
        loss_cl_sum = 0
        loss_am_sum = 0
        acc_cl_sum = 0
        total_loss_sum = 0
        for images, segments, labels, trust_image in rds['test']:
            # test
            if self.gpu:
                images = images.cuda()
                segments = segments.cuda()
                labels = labels.cuda()
            total_loss, loss_cl, loss_am, loss_e, prob, acc_cl, A_c, I_star, mask = self.forward(images, labels,
                                                                                                 segments)

            total_loss_sum += scalar(total_loss)
            loss_cl_sum += scalar(loss_cl)
            # loss_am_sum += scalar(loss_am)
            acc_cl_sum += scalar(acc_cl)

            visualize.visualize(images[0], I_star[0], A_c[0], mask[0], segments[0],
                                self.epochs_trained)

        test_size = len(rds['test'])
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
        # w_c = F.avg_pool2d(grad, (grad.shape[-2], grad.shape[-1]), 1)
        # w_c_new_shape = (1, w_c.shape[0] * w_c.shape[1], w_c.shape[2], w_c.shape[3])
        # w_c = w_c.view(w_c_new_shape).unsqueeze(0)

        # Eq 2
        # TODO this doesn't support batching
        # weights = self._last_activation
        # weights_new_shape = (1, weights.shape[0] * weights.shape[1], weights.shape[2], weights.shape[3])
        # weights = weights.view(weights_new_shape).unsqueeze(0)

        self._last_grad = self._last_grad  # weights
        self._last_activation = F.adaptive_avg_pool2d(self._last_activation, 1)  # fl
        A_c = torch.mul(self._last_activation, self._last_grad).sum(dim=1, keepdim=True)
        A_c = F.relu(A_c)
        A_c = F.upsample_bilinear(A_c, size=data.size()[2:])
        # gcam = F.relu(F.conv2d(weights, w_c))
        # A_c = F.upsample(gcam, size=data.size()[2:], mode='bilinear')

        loss_cl = self.loss_cl(output_cl, label)

        return output_cl, loss_cl, A_c

    def _mask_image(self, gcam, image):
        gcam_min = gcam.min()
        gcam_max = gcam.max()
        scaled_gcam = (gcam - gcam_min) / (gcam_max - gcam_min)
        mask = F.sigmoid(self.omega * (scaled_gcam - self.sigma))
        masked_image = image - (image * mask)
        return masked_image, mask

    def _forward(self, data, label, segment):
        # TODO normalize elsewhere, this feels wrong
        output_cl, loss_cl, gcam = self._attention_map_forward(data, label)
        output_cl_softmax = F.softmax(output_cl, dim=1)

        # Eq 4
        # TODO this currently doesn't support batching, maybe add that
        I_star, mask = self._mask_image(gcam, data)

        # output_am = self.model(I_star)

        # Eq 5
        # loss_am = F.sigmoid(output_am) * label
        # loss_am = loss_am.sum() / label.sum().type(self.tensor_source.FloatTensor)

        updated_segment_mask = segment * 50 + self.minus_mask
        loss_e = ((mask - updated_segment_mask) @ (mask - updated_segment_mask)).sum()

        # Eq 6
        total_loss = loss_cl + loss_e * 100  # + self.alpha * loss_am

        cl_acc = output_cl_softmax.max(dim=1)[1] == label.max(dim=1)[1]
        cl_acc = cl_acc.type(self.tensor_source.FloatTensor).mean()

        #                       loss_am
        return total_loss, loss_cl, 0, loss_e, output_cl_softmax, cl_acc, gcam, I_star, mask

    def _convert_data_and_label(self, data, label, segments):
        # converts our data and label over to variables, gpu optional
        if self.gpu:
            data = data.cuda()
            label = label.cuda()
            segments = segments.cuda()

        data = torch.autograd.Variable(data)
        label = torch.autograd.Variable(label)
        segments = torch.autograd.Variable(segments)
        return data, label, segments


"""
gcam.shape = {Size} torch.Size([10, 1, 224, 224])
image.shape = {Size} torch.Size([10, 3, 224, 224])
hmmm need proof that
"""
