import torch
import torch.nn as nn
import copy
from utils import property as P, gradient_registers as gr
from utils import utils
from strategies import abstract_train as at


class AlternateModuleTrain(at.AbstractTrain):
    """
        implementation train where at first only am module train, then only classification
    """

    def __init__(self, am_model: nn.Module = None,
                 train_segments_set=None,
                 test_set=None,
                 l_loss: nn.Module = nn.BCELoss(),
                 m_loss: nn.Module = nn.BCEWithLogitsLoss(),
                 classes: int = None,
                 pre_train_epochs: int = 100,
                 train_epochs: int = 100,
                 save_train_logs_epochs: int = 5,
                 test_each_epoch: int = 5,
                 use_gpu: bool = True,
                 gpu_device: int = 0,
                 description: str = "am",
                 left_class_number: int = None,
                 right_class_number: int = None,
                 snapshot_elements_count: int = 11,
                 snapshot_dir: str = None,
                 classifier_learning_rate: float = None,
                 attention_module_learning_rate: float = None,
                 weight_decay: float = 0,
                 current_epoch: int = 1):

        super(AlternateModuleTrain, self).__init__(classes, pre_train_epochs, train_epochs, save_train_logs_epochs,
                                                   test_each_epoch, use_gpu,
                                                   gpu_device, description, left_class_number, right_class_number,
                                                   snapshot_elements_count, snapshot_dir,
                                                   classifier_learning_rate,
                                                   attention_module_learning_rate,
                                                   weight_decay,
                                                   current_epoch)

        self.train_segments_set = train_segments_set
        self.test_set = test_set

        self.am_model = am_model

        if use_gpu:
            self.am_model = self.am_model.cuda(self.gpu_device)

        self.best_weights = None
        self.best_test_weights = None

        self.l_loss = l_loss
        self.m_loss = m_loss

    def train(self, use_all_params: bool):

        if use_all_params:
            classifier_optimizer = torch.optim.Adam(self.am_model.parameters(), lr=self.classifier_learning_rate)
        else:
            classifier_optimizer = torch.optim.Adam(gr.register_weights("classifier", self.am_model),
                                                    self.classifier_learning_rate)
        attention_module_optimizer = torch.optim.Adam(gr.register_weights("attention", self.am_model),
                                                      lr=self.attention_module_learning_rate)

        self.best_weights = copy.deepcopy(self.am_model.state_dict())
        best_loss = None
        best_test_loss = None

        while self.current_epoch <= self.train_epochs:

            loss_m_sum = 0
            accuracy_classification_sum_segments = 0
            loss_classification_sum_from_segm = 0
            loss_l1_sum = 0

            div_flag = False

            if self.current_epoch <= self.pre_train_epochs:
                div_flag = True
                loss_classification_sum_classifier, accuracy_classification_sum_classifier, loss_segmentation_sum = \
                    self.train_classifier(self.am_model, self.l_loss, self.m_loss, classifier_optimizer,
                                          self.train_segments_set)
                classifier_optimizer.zero_grad()
            else:

                loss_classification_sum_classifier, accuracy_classification_sum_classifier, loss_segmentation_sum = \
                    self.train_classifier(self.am_model, self.l_loss, self.m_loss, classifier_optimizer,
                                          self.train_segments_set)
                classifier_optimizer.zero_grad()

                accuracy_classification_sum_segments, loss_m_sum, loss_l1_sum, loss_classification_sum_from_segm = \
                    self.train_segments(self.am_model, self.l_loss, self.m_loss, attention_module_optimizer,
                                        self.train_segments_set)
                attention_module_optimizer.zero_grad()

            accuracy_total = (accuracy_classification_sum_segments + accuracy_classification_sum_classifier) / (
                1 if div_flag else 2)
            classification_loss_total = (loss_classification_sum_classifier + loss_classification_sum_from_segm) / (
                1 if div_flag else 2)
            loss_m_sum = (loss_segmentation_sum + loss_m_sum) / (1 if div_flag else 2)

            loss_total = loss_classification_sum_classifier + loss_m_sum + loss_classification_sum_from_segm
            prefix = "PRETRAIN" if self.current_epoch <= self.pre_train_epochs else "TRAIN"
            f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                             self.train_trust_answers,
                                                                                             self.train_model_answers)

            text = "{}={} Loss_CL={:.5f} Loss_M={:.5f} Loss_L1={:.5f} Loss_Total={:.5f} Accuracy_CL={:.5f} " \
                   "{} {} {} ".format(prefix, self.current_epoch, classification_loss_total,
                                      loss_m_sum,
                                      loss_l1_sum,
                                      loss_total,
                                      accuracy_total,
                                      f_1_score_text,
                                      recall_score_text,
                                      precision_score_text)

            P.write_to_log(text)

            if self.current_epoch % self.test_each_epoch == 0:
                self.take_snapshot(self.train_segments_set, self.am_model, "TRAIN_{}".format(self.current_epoch))
                self.take_snapshot(self.test_set, self.am_model, "TEST_{}".format(self.current_epoch))
                test_loss, _ = self.test(self.am_model, self.test_set, self.l_loss, self.m_loss)
                if best_test_loss is None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.best_test_weights = copy.deepcopy(self.am_model.state_dict())

            if best_loss is None or loss_total < best_loss:
                best_loss = loss_total
                self.best_weights = copy.deepcopy(self.am_model.state_dict())

            self.clear_temp_metrics()
            self.current_epoch += 1

        self.save_model(self.best_test_weights)
        self.save_model(self.best_weights)
