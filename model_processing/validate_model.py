from torchvision import models
import torch.utils.data.dataset
import torch
import copy
from sklearn.metrics import f1_score


class ValidateModel:
    def __init__(self, labels_number: int, device='cpu'):
        self.device = device
        self.labels_number = labels_number

        self.iterate_accuracy = []
        self.best_accuracy = 0.0
        self.best_model_weights = None

    def validate_model_single_epoch(self, m: models.AlexNet, data_loader: torch.utils.data.DataLoader) -> float:

        with torch.set_grad_enabled(False):
            trust_answer = [[] for _ in range(self.labels_number)]
            model_answer = [[] for _ in range(self.labels_number)]
            m.eval()
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                model_result = m(inputs)
                # all vectors size 5
                for i, t in enumerate(model_result):
                    # all classes
                    for j, v in enumerate(t):
                        dst_0 = abs(v.item())
                        dst_1 = abs(1 - v.item())
                        trust = labels[i][j].item()
                        ma = 1 if dst_1 < dst_0 else 0
                        trust_answer[j].append(trust)
                        model_answer[j].append(ma)

            res = sum([f1_score(t, m) for t, m in zip(trust_answer, model_answer)]) / self.labels_number
            if res > self.best_accuracy:
                self.best_accuracy = res
                self.best_model_weights = copy.deepcopy(m.state_dict())
            self.iterate_accuracy.append(res)
            return res

    def flush(self):
        self.iterate_accuracy = []
        self.best_accuracy = 0.0
        self.best_model_weights = None
