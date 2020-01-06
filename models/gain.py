import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet


class GAIN(nn.Module):
    def __init__(self, model,
                 grad_layer,
                 num_classes,
                 use_am: bool = False,
                 gpu: bool = False,
                 device: int = 0,
                 alpha: float = 1.0,
                 omega: float = 10.0,
                 sigma: float = 0.5,
                 ):
        super(GAIN, self).__init__()
        self.model = model

        self.grad_layer = grad_layer

        self.num_classes = num_classes

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        self.use_am = use_am
        self.gpu = gpu
        self.device = device
        # sigma, omega for making the soft-mask
        self.alpha = alpha
        self.omega = omega
        self.sigma = sigma

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

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=True)
        labels = labels.long()
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        ohe = torch.autograd.Variable(ohe)

        return ohe

    def forward(self, images, labels, index):

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.model.training

        with torch.enable_grad():

            _, _, img_h, img_w = images.size()

            self.model.train(True)
            logits = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            if not is_train:
                pred = F.softmax(logits).argmax(dim=1)
                labels_ohe = self._to_ohe(pred)
            else:
                labels_ohe = self._to_ohe(labels)

            if self.gpu:
                labels_ohe = labels_ohe.cuda(self.device)
            else:
                labels_ohe = labels_ohe.cpu()

            ones = torch.ones(labels.shape[0])
            if self.gpu:
                ones = ones.cuda(self.device)
            # Здесь переписал так
            # так, как мы так обсуждали
            gradient = logits * labels
            grad_logits = logits * labels  # .sum()  # BS x num_classes
            # grad_logits = logits
            grad_logits[:, index].backward(gradient=gradient[:, index], retain_graph=True)
            self.model.zero_grad()

        if is_train:
            self.model.train(True)
        else:
            self.model.train(False)
            self.model.eval()
            logits = self.model(images)

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

    def forvard_(self, images, labels):
        with torch.enable_grad():
            self.model.train(True)
            logits = self.model(images)
            self.model.zero_grad()
            gradient = logits * labels
            grad_logits = logits * labels.sum()  # BS x num_classes
            grad_logits.backward(gradient=gradient, retain_graph=True)
            self.model.zero_grad()
            return logits
