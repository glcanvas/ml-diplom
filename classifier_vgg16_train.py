"""
classify dataset
"""

import torch
import property as p
import torch.nn as nn
import copy
import utils
import abstract_train as at


class Classifier(at.AbstractTrain):
    """
    baseline classifier based on vgg16
    """

    def __init__(self, model: nn.Module = None,
                 train_segments_set=None,
                 test_set=None,
                 l_loss: nn.Module = nn.BCELoss(),
                 classes: int = None,
                 test_each_epoch: int = 5,
                 train_epochs: int = 100,
                 use_gpu: bool = True,
                 gpu_device: int = 0,
                 description: str = "sam",
                 left_class_number: int = None,
                 right_class_number: int = None):
        super(Classifier, self).__init__(classes, None, None, None, test_each_epoch, use_gpu, gpu_device, description,
                                         left_class_number, right_class_number, None, None)

        self.train_epochs = train_epochs
        self.train_segments_set = train_segments_set
        self.test_set = test_set

        self.best_weights = None
        self.best_test_weights = None

        self.l_loss = l_loss

        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        self.description = description

        self.classes = classes

        self.model = model
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, self.classes)

        self.best_weights = copy.deepcopy(self.model.state_dict())
        self.best_test_weights = copy.deepcopy(self.model.state_dict())

        if self.use_gpu:
            self.model = self.model.cuda(self.gpu_device, )

    def train(self, learning_rate=1e-6):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        best_loss = None
        best_test_loss = None

        for epoch in range(1, self.train_epochs):
            self.current_epoch = epoch

            loss_classification_sum = 0
            accuracy_classification_sum = 0
            batch_count = 0

            for images, segments, labels in self.train_segments_set:
                labels, segments = utils.reduce_to_class_number(self.left_class_number, self.right_class_number, labels,
                                                                segments)
                images, labels, segments = self.convert_data_and_label(images, labels, segments)
                segments = self.PULLER(segments)

                # calculate and optimize model
                optimizer.zero_grad()

                model_classification = utils.wait_while_can_execute_single(self.model, images)
                sigmoid = nn.Sigmoid()  # used for calculate accuracy
                model_classification = sigmoid(model_classification)

                classification_loss = self.l_loss(model_classification, labels)
                torch.cuda.empty_cache()
                classification_loss.backward()
                optimizer.step()

                output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                                labels.size(0))

                self.save_train_data(labels, output_cl, output_probability)

                # accumulate information
                accuracy_classification_sum += utils.scalar(cl_acc.sum())
                loss_classification_sum += utils.scalar(classification_loss.sum())
                batch_count += 1
                self.de_convert_data_and_label(images, segments, labels)
                torch.cuda.empty_cache()

            if best_loss is None or loss_classification_sum < best_loss:
                best_loss = loss_classification_sum
                self.best_weights = copy.deepcopy(self.model.state_dict())

            f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                             self.train_trust_answers,
                                                                                             self.train_model_answers)
            text = "TRAIN={} Loss_CL={:.10f} Accuracy_CL={:.5f} {} {} {} ".format(epoch,
                                                                                  loss_classification_sum / batch_count,
                                                                                  accuracy_classification_sum / batch_count,
                                                                                  f_1_score_text,
                                                                                  recall_score_text,
                                                                                  precision_score_text)
            p.write_to_log(text)
            if self.current_epoch % self.test_each_epoch == 0:
                test_loss, _ = self.test(self.model, self.test_set, self.l_loss)
                if best_test_loss is None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.best_test_weights = copy.deepcopy(self.model.state_dict())

            if best_loss is None or loss_classification_sum < best_loss:
                best_loss = loss_classification_sum
                self.best_weights = copy.deepcopy(self.model.state_dict())
            self.clear_temp_metrics()

        self.save_model(self.best_test_weights)
        self.save_model(self.best_weights)

    def test(self, model, test_set, l_loss, m_loss=None):
        loss_classification_sum = 0
        accuracy_classification_sum = 0
        batch_count = 0
        for images, segments, labels in test_set:
            labels, segments = utils.reduce_to_class_number(self.left_class_number, self.right_class_number, labels,
                                                            segments)
            images, labels, segments = self.convert_data_and_label(images, labels, segments)
            model_classification = utils.wait_while_can_execute_single(model, images)

            sigmoid = nn.Sigmoid()  # used for calculate accuracy
            model_classification = sigmoid(model_classification)
            classification_loss = l_loss(model_classification, labels)

            output_probability, output_cl, cl_acc = self.calculate_accuracy(labels, model_classification,
                                                                            labels.size(0))

            self.save_test_data(labels, output_cl, output_probability)

            # accumulate information
            accuracy_classification_sum += utils.scalar(cl_acc.sum())
            loss_classification_sum += utils.scalar(classification_loss.sum())
            batch_count += 1
            self.de_convert_data_and_label(images, labels)
            torch.cuda.empty_cache()

        f_1_score_text, recall_score_text, precision_score_text = utils.calculate_metric(self.classes,
                                                                                         self.test_trust_answers,
                                                                                         self.test_model_answers)

        loss_classification_sum /= batch_count + p.EPS
        accuracy_classification_sum /= batch_count + p.EPS
        text = 'TEST Loss_CL={:.5f} Accuracy_CL={:.5f} {} {} {} '.format(loss_classification_sum,
                                                                         accuracy_classification_sum,
                                                                         f_1_score_text,
                                                                         recall_score_text,
                                                                         precision_score_text)
        p.write_to_log(text)

        return loss_classification_sum, accuracy_classification_sum
