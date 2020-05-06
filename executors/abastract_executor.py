import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

import os
from utils import image_loader as il, property as P
from torch.utils.data import DataLoader
import traceback
import torch.nn as nn
import model.soft_f1_loss as f1loss


class AbstractExecutor:

    def __init__(self, parsed):
        self.gpu = int(parsed.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        self.gpu = 0
        self.parsed_description = parsed.description
        self.pre_train = int(parsed.pre_train)
        self.train_set_size = int(parsed.train_set)
        self.epochs = int(parsed.epochs)

        self.run_name = parsed.run_name
        self.algorithm_name = parsed.algorithm_name
        self.left_class_number = int(parsed.left_class_number)
        self.right_class_number = int(parsed.right_class_number)
        self.freeze_list = parsed.freeze_list
        self.classifier_learning_rate = float(parsed.classifier_learning_rate)
        self.attention_module_learning_rate = float(parsed.attention_module_learning_rate)
        self.is_freezen = False if str(parsed.is_freezen).lower() == "false" else True

        self.resnet_type = parsed.resnet_type
        self.inceptionv_type = parsed.inceptionv_type
        self.image_size = int(parsed.image_size)

        if str(parsed.classifier_loss_function).lower() == "bceloss":
            self.classifier_loss_function = nn.BCELoss()
        elif str(parsed.classifier_loss_function).lower() == "softf1":
            self.classifier_loss_function = f1loss.SoftF1Loss()
        else:
            raise Exception("loss {} not found".format(parsed.classifier_loss_function))

        if str(parsed.am_loss_function).lower() == "bceloss":
            self.am_loss_function = nn.BCELoss()
        elif str(parsed.am_loss_function).lower() == "softf1":
            self.am_loss_function = f1loss.SoftF1Loss()
        else:
            raise Exception("loss {} not found".format(parsed.am_loss_function))

        self.model_identifier = parsed.model_identifier
        self.execute_from_model = False if str(parsed.execute_from_model).lower() == "false" else True

        self.classes = self.right_class_number - self.left_class_number

        self.description = "description-{},train_set-{},epochs-{},l-{},r-{},clr-{},amlr-{},model_identifier-{}".format(
            self.parsed_description,
            self.train_set_size,
            self.epochs,
            self.left_class_number,
            self.right_class_number,
            self.classifier_learning_rate,
            self.attention_module_learning_rate,
            self.model_identifier
        )
        self.snapshots_path = None
        self.train_segments_set = None
        self.test_set = None
        self.model = None
        self.strategy = None

        self.initialize_logs()
        self.initialize_snapshots_dir()
        self.load_dataset()
        self.current_epoch = self.get_current_epoch()
        self.model_state_dict = self.load_model_from_saves()
        self.model = self.create_model()
        self.strategy = self.create_strategy()

    def create_model(self):
        pass

    def create_strategy(self):
        pass

    def train_strategy(self):
        pass

    def safe_train(self):
        try:
            self.train_strategy()
            exit(0)
        except BaseException as e:
            print("EXCEPTION", e)
            print(type(e))
            P.write_to_log("EXCEPTION", e, type(e))
            traceback.print_stack()

            P.save_raised_model(self.model, self.strategy.current_epoch, self.model_identifier, self.run_name,
                                self.algorithm_name)
            P.write_to_log("saved model, exception raised")
            exit(1)

    def initialize_logs(self):
        P.initialize_log_name(self.run_name, self.algorithm_name, self.description, self.model_identifier)

        P.write_to_log("description={}".format(self.description))
        P.write_to_log("classes={}".format(self.classes))
        P.write_to_log("run=" + self.run_name)
        P.write_to_log("algorithm_name=" + self.algorithm_name)

    def initialize_snapshots_dir(self):
        log_name, log_dir = os.path.basename(P.log)[:-4], os.path.dirname(P.log)

        self.snapshots_path = os.path.join(log_dir, log_name)
        os.makedirs(self.snapshots_path, exist_ok=True)

    def load_dataset(self):
        segments_set, test_set = il.load_data(self.train_set_size, self.model_identifier, self.image_size)

        self.train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=3, shuffle=True)
        print("ok")
        self.test_set = DataLoader(il.ImageDataset(test_set), batch_size=3)
        print("ok")

    def get_current_epoch(self) -> int:
        if self.execute_from_model:
            model_state_dict, current_epoch = P.load_latest_model(self.model_identifier, self.run_name,
                                                                  self.algorithm_name)
            if model_state_dict is None:
                exit(0)
                #raise Exception(
                #    "not found model for current epoch: model_identifier: {}, run_name: {}, algorithm_name: {}"
                #        .format(self.model_identifier, self.run_name, self.algorithm_name))
            return current_epoch
        return 1

    def load_model_from_saves(self):
        if self.execute_from_model:
            model_state_dict, current_epoch = P.load_latest_model(self.model_identifier, self.run_name,
                                                                  self.algorithm_name)
            if model_state_dict is None:
                exit(0)
                #raise Exception(
                #    "not found model for current epoch: model_identifier: {}, run_name: {}, algorithm_name: {}"
                #        .format(self.model_identifier, self.run_name, self.algorithm_name))
            return model_state_dict
        return None
