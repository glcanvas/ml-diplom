import torch
import torch.nn as nn
from utils import property as P, vgg16_gradient_registers as gr, resnet_gradient_registers as rgr
from strategies import abstract_train as at
from utils import metrics_processor
from utils import model_utils
from utils import property as p


class SingleSimultaneousModuleTrain(at.AbstractTrain):
    """
        implementation train where at first only am module train, then only classification
    """

    def __init__(self, am_model: nn.Module = None,
                 is_vgg_model: bool = None,
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
                 current_epoch: int = 1,
                 puller: nn.Module = None):

        super(SingleSimultaneousModuleTrain, self).__init__(classes, pre_train_epochs, train_epochs,
                                                            save_train_logs_epochs,
                                                            test_each_epoch, use_gpu,
                                                            gpu_device, description, left_class_number,
                                                            right_class_number,
                                                            snapshot_elements_count, snapshot_dir,
                                                            classifier_learning_rate,
                                                            attention_module_learning_rate,
                                                            weight_decay,
                                                            current_epoch,
                                                            puller)

        self.train_segments_set = train_segments_set
        self.test_set = test_set

        self.am_model = am_model

        if use_gpu:
            self.am_model = self.am_model.cuda(self.gpu_device)

        self.best_weights = None
        self.best_test_weights = None

        self.l_loss = l_loss
        self.m_loss = m_loss
        self.is_vgg_model = is_vgg_model

    def train(self):
        if self.is_vgg_model:
            classifier_optimizer = torch.optim.Adam(gr.register_weights("classifier", self.am_model),
                                                    self.classifier_learning_rate)
            attention_module_optimizer = torch.optim.Adam(gr.register_weights("attention", self.am_model),
                                                          lr=self.attention_module_learning_rate)
        else:
            classifier_optimizer = torch.optim.Adam(rgr.register_weights("classifier", self.am_model),
                                                    self.classifier_learning_rate)
            attention_module_optimizer = torch.optim.Adam(rgr.register_weights("attention", self.am_model),
                                                          lr=self.attention_module_learning_rate)

        while self.current_epoch <= self.train_epochs:

            loss_m_sum = 0
            loss_l1_sum = 0

            loss_classification_sum = 0
            loss_segmentation_sum = 0
            accuracy_sum = 0
            batch_count = 0
            self.am_model.train(mode=True)
            for images, segments, labels in self.train_segments_set:
                labels, segments = model_utils.reduce_to_class_number(self.left_class_number, self.right_class_number,
                                                                      labels,
                                                                      segments)
                images, labels, segments = self.convert_data_and_label(images, labels, segments)
                segments = self.puller(segments)

                # calculate and optimize model
                classifier_optimizer.zero_grad()
                attention_module_optimizer.zero_grad()

                model_classification, model_segmentation = model_utils.wait_while_can_execute(self.am_model, images)
                segmentation_loss = self.m_loss(model_segmentation, segments)
                classification_loss = self.l_loss(model_classification, labels)
                # torch.cuda.empty_cache()
                classification_loss.backward(retain_graph=True)
                segmentation_loss.backward()

                classifier_optimizer.step()
                attention_module_optimizer.step()

                output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                                labels.size(0))

                classifier_optimizer.zero_grad()
                attention_module_optimizer.zero_grad()

                self.save_train_data(labels, output_cl, output_probability)

                # accumulate information
                accuracy_sum += model_utils.scalar(cl_acc.sum())
                loss_classification_sum += model_utils.scalar(classification_loss.sum())
                loss_segmentation_sum += model_utils.scalar(segmentation_loss.sum())
                batch_count += 1
                # self.de_convert_data_and_label(images, segments, labels)
                # torch.cuda.empty_cache()

            loss_classification_sum = loss_classification_sum / (batch_count + p.EPS)
            accuracy_sum = accuracy_sum / (batch_count + p.EPS)
            loss_segmentation_sum = loss_segmentation_sum / (batch_count + p.EPS)
            loss_total = loss_classification_sum + loss_m_sum + loss_segmentation_sum
            prefix = "PRETRAIN" if self.current_epoch <= self.pre_train_epochs else "TRAIN"
            f_1_score_text, recall_score_text, precision_score_text = metrics_processor.calculate_metric(self.classes,
                                                                                                         self.train_trust_answers,
                                                                                                         self.train_model_answers)

            text = "{}={} Loss_CL={:.5f} Loss_M={:.5f} Loss_L1={:.5f} Loss_Total={:.5f} Accuracy_CL={:.5f} " \
                   "{} {} {} ".format(prefix, self.current_epoch, loss_classification_sum,
                                      loss_m_sum,
                                      loss_l1_sum,
                                      loss_total,
                                      accuracy_sum,
                                      f_1_score_text,
                                      recall_score_text,
                                      precision_score_text)

            P.write_to_log(text)
            self.am_model.train(mode=False)
            if self.current_epoch % self.test_each_epoch == 0:
                test_loss, _ = self.test(self.am_model, self.test_set, self.l_loss, self.m_loss)
            if self.current_epoch % 200 == 0:
                self.take_snapshot(self.train_segments_set, self.am_model, "TRAIN_{}".format(self.current_epoch))
                self.take_snapshot(self.test_set, self.am_model, "TEST_{}".format(self.current_epoch))

            self.clear_temp_metrics()
            self.current_epoch += 1
