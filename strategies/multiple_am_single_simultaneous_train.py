import torch
import torch.nn as nn
from utils import property as P, vgg16_gradient_registers as gr, resnet_gradient_registers as rgr
from strategies import abstract_train as at
from utils import metrics_processor
from utils import model_utils
from utils import property as p


class MultipleAMSingleSimultaneousTrain(at.AbstractTrain):
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
                 current_epoch: int = 1,
                 puller: nn.Module = None,
                 use_mloss: bool = False):
        super(MultipleAMSingleSimultaneousTrain, self).__init__(classes, pre_train_epochs, train_epochs,
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
        self.use_mloss = use_mloss

    def train(self):
        optimizer = torch.optim.Adam(self.am_model.parameters(), self.classifier_learning_rate)

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
                segments_list = []
                for puller in self.puller:
                    segments_list.append(puller(segments))

                # calculate and optimize model
                optimizer.zero_grad()

                model_classification, model_segmentation = model_utils.wait_while_can_execute(self.am_model, images)
                classification_loss = self.l_loss(model_classification, labels)
                total_loss = classification_loss

                if self.use_mloss:
                    sum_segm_loss = None
                    for ms, sl in zip(model_segmentation, segments_list):
                        segmentation_loss = self.m_loss(ms, sl)
                        total_loss += segmentation_loss
                        if sum_segm_loss is None:
                            sum_segm_loss = segmentation_loss
                        else:
                            sum_segm_loss += segmentation_loss
                total_loss.backward()
                optimizer.step()

                output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                                labels.size(0))

                optimizer.zero_grad()

                self.save_train_data(labels, output_cl, output_probability)

                accuracy_sum += model_utils.scalar(cl_acc.sum())
                loss_classification_sum += model_utils.scalar(classification_loss.sum())
                if self.use_mloss:
                    loss_segmentation_sum += model_utils.scalar(sum_segm_loss.sum())
                batch_count += 1

            loss_classification_sum = loss_classification_sum / (batch_count + p.EPS)
            accuracy_sum = accuracy_sum / (batch_count + p.EPS)
            loss_segmentation_sum = loss_segmentation_sum / (batch_count + p.EPS)
            loss_total = loss_classification_sum + loss_m_sum + loss_segmentation_sum
            prefix = "TRAIN"
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

            if self.current_epoch % self.test_each_epoch == 0:
                test_loss, _ = self.test(self.am_model, self.test_set, self.l_loss, self.m_loss)

            self.clear_temp_metrics()
            self.current_epoch += 1

    def test(self, model, test_set, l_loss, m_loss):
        model.train(mode=False)
        loss_classification_sum = 0
        loss_segmentation_sum = 0
        accuracy_classification_sum = 0
        batch_count = 0
        for images, segments, labels in test_set:
            labels, segments = model_utils.reduce_to_class_number(self.left_class_number, self.right_class_number,
                                                                  labels,
                                                                  segments)
            images, labels, segments = self.convert_data_and_label(images, labels, segments)
            segments_list = []
            for puller in self.puller:
                segments_list.append(puller(segments))
            model_classification, model_segmentation = model_utils.wait_while_can_execute(model, images)

            classification_loss = l_loss(model_classification, labels)
            if self.use_mloss:
                sum_segm_loss = None
                for ms, sl in zip(model_segmentation, segments_list):
                    segmentation_loss = self.m_loss(ms, sl)
                    if sum_segm_loss is None:
                        sum_segm_loss = segmentation_loss
                    else:
                        sum_segm_loss += segmentation_loss

            output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                            labels.size(0))

            self.save_test_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += model_utils.scalar(cl_acc.sum())
            loss_classification_sum += model_utils.scalar(classification_loss.sum())
            if self.use_mloss:
                loss_segmentation_sum += model_utils.scalar(sum_segm_loss.sum())
            batch_count += 1
            # self.de_convert_data_and_label(images, labels)
            # torch.cuda.empty_cache()

        f_1_score_text, recall_score_text, precision_score_text = metrics_processor.calculate_metric(self.classes,
                                                                                                     self.test_trust_answers,
                                                                                                     self.test_model_answers)

        loss_classification_sum /= batch_count + p.EPS
        accuracy_classification_sum /= batch_count + p.EPS
        loss_segmentation_sum /= batch_count + p.EPS
        text = 'TEST={} Loss_CL={:.5f} Loss_M={:.5f} Accuracy_CL={:.5f} {} {} {} '.format(self.current_epoch,
                                                                                          loss_classification_sum,
                                                                                          loss_segmentation_sum,
                                                                                          accuracy_classification_sum,
                                                                                          f_1_score_text,
                                                                                          recall_score_text,
                                                                                          precision_score_text)
        p.write_to_log(text)
        model.train(mode=True)
        return loss_classification_sum, accuracy_classification_sum
