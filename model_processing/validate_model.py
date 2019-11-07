from torchvision import models
import torch.utils.data.dataset
import torch
import copy


class ValidateModel:
    def __init__(self, labels_number: int, device='cpu'):
        self.device = device
        self.labels_number = labels_number

        self.iterate_accuracy = []
        self.best_accuracy = 0.0
        self.best_model_weights = None

    def validate_model_single_epoch(self, m: models.AlexNet, data_loader: torch.utils.data.DataLoader) -> float:

        with torch.set_grad_enabled(False):
            confusion_matrix = torch \
                .tensor([[[0 for _ in range(2)] for _ in range(2)] for _ in range(self.labels_number)]) \
                .to(self.device)

            m.eval()
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # trust result
                for t in labels:
                    for idx, v in enumerate(t):
                        confusion_matrix[idx][v.item()][v.item()] += 1

                model_result = m(inputs)
                # all vectors size 5
                for i, t in enumerate(model_result):
                    # all classes
                    for j, v in enumerate(t):
                        dst_0 = abs(v.item())
                        dst_1 = abs(1 - v.item())
                        trust = labels[i][j].item()
                        model_answer = 1 if dst_1 < dst_0 else 0
                        confusion_matrix[j][model_answer][trust] += 1

            res = sum([self.__calculate_f_measure(i) for i in confusion_matrix]) / self.labels_number
            if res > self.best_accuracy:
                self.best_accuracy = res
                self.best_model_weights = copy.deepcopy(m.state_dict())
            self.iterate_accuracy.append(res)
            return res

    def flush(self):
        self.iterate_accuracy = []
        self.best_accuracy = 0.0
        self.best_model_weights = None

    @staticmethod
    def __calculate_f_measure(conf_matrix: torch.Tensor) -> float:
        precision_w = 0
        recall_w = 0
        sum_all = conf_matrix.sum().item()

        for i, t in enumerate(conf_matrix):
            c = 0
            p = 0
            for j, _ in enumerate(t):
                c += conf_matrix[i][j].item()
                p += conf_matrix[j][i].item()
            if p == 0:
                precision_w += 0
            else:
                precision_w += ((conf_matrix[i][i].item() * c) / p) / sum_all
            recall_w += conf_matrix[i][i].item() / sum_all
        return (2 * (precision_w * recall_w)) / (precision_w + recall_w)
