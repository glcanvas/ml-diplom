from torchvision import models
import torch.utils.data.dataset
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from model_processing.validate_model import ValidateModel


class TrainModel:
    def __init__(self, validator: ValidateModel, device='cpu'):
        self.device = device
        self.validator = validator

    def train_model(self, m: models.AlexNet, train_loader: torch.utils.data.DataLoader,
                    validate_loader: torch.utils.data.DataLoader,
                    criterion: nn.CrossEntropyLoss,
                    optimizer: Optimizer, epochs: int) -> models.AlexNet:

        self.validator.flush()

        for epoch in range(epochs):
            print("Epoch: {}/{}".format(epoch + 1, epochs))
            m = self._train_model_single_epoch(m, train_loader, criterion, optimizer)
            acc = self.validator.validate_model_single_epoch(m, validate_loader)
            print("acc: {}".format(acc))
            print("-" * 20)

        best_model = m.load_state_dict(self.validator.best_model_weights)
        print("best acc: {}".format(self.validator.best_accuracy))
        return best_model

    def _train_model_single_epoch(self, m: models.AlexNet, data_loader: torch.utils.data.DataLoader,
                                  criterion: nn.CrossEntropyLoss, optimizer: Optimizer) -> models.AlexNet:
        m.train()
        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.type(torch.FloatTensor).to(self.device)
            optimizer.zero_grad()

            model_result = m(inputs)
            loss = criterion(model_result, labels)
            loss.backward()
            optimizer.step()
        return m
