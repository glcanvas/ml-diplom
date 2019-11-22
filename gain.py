import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torch

models.alexnet(pretrained=True)


class GAIN:

    def __init__(self, model, layer_name: str, gpu=False):
        self.model = model

        # set cuda
        self.gpu = gpu
        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
        else:
            self.tensor_source = torch

        self._last_activation = None
        self._last_grad = None

        self.__register_callbacks(layer_name)
        self.loss_cl = torch.nn.BCEWithLogitsLoss()

    def __register_callbacks(self, layer_name: str):

        def forward_callback(module, input_, output_):
            self._last_activation = output_

        def backward_callback(module, grad_in, grad_out):
            self._last_grad = grad_out[0]

        found_layer = False
        for idx, layer in self.model.named_modules():
            if idx == layer_name:
                found_layer = True
                layer.register_forward_hook(forward_callback)
                layer.register_backward_hook(backward_callback)
                break

        if not found_layer:
            raise ValueError("Not found layer:{}".format(layer_name))

    def russian_forward(self, input_image_data, mask_label_data, label_data):
        # build attention map
        output_classification = self.model(input_image_data)

        # required for get dx / dy for substitution to eq 1
        grad_target = (output_classification * label_data).sum()
        grad_target.backward(gradient=output_classification * label_data, retain_graph=True)

        self.model.zero_grad()

        # Eq 1
        # find w_c from evaluate grad model for specified layer
        layer_gradient = self._last_grad
        # TODO what dim has layer_gradient
        w_c = F.avg_pool2d(layer_gradient, (layer_gradient.shape[-2], layer_gradient.shape[-1]), 1)
        # TODO why mul not div
        w_c_new_shape = (w_c.shape[0] * w_c.shape[1], w_c.shape[2], w_c.shape[3])
        w_c = w_c.view(w_c_new_shape).unsqueeze(0)

        # Eq 2
        # reshape current weights for compability with w_c
        f_weights = self._last_activation
        f_weights_new_shape = (f_weights.shape[0] * f_weights.shape[1], f_weights.shape[2], f_weights.shape[3])
        f_weights = f_weights.view(f_weights_new_shape).unsqueeze(0)

        a_c = F.relu(F.conv2d(f_weights, w_c))
        # here for a_c maybe add upsample (augumentation), but not now

        # calculate old plain classification loss
        classification_loss = self.loss_cl(output_classification, label_data)

        # her we has output_classification, classification_loss, a_c

        i_start = self.__mask_image(input_image_data, a_c)


    sigma = 0.5
    omega = 0.8

    def __mask_image(self, data_image, a_c):
        # Eq 4
        t_a_c = F.sigmoid(self.omega * (a_c - self.sigma)).squeeze()
        # Eq 3
        masked_image = data_image - (data_image * t_a_c)
        return masked_image
