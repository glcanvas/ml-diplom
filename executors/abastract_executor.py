import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")
import traceback
import torchvision.models as m
import os
from utils import image_loader as il, property as P, imbalansed_image_loader as imbalanced
from model import vgg_with_am_model as am_model
from model import vgg_with_shifted_am_model as am_model_shift
from model import resnet_with_am_model as resnet_am_model
from model import resnet18_cbam as resnet18cbam
from model import resnet34_cbam as resnet34cbam
from model import connection_block as cb
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
        self.cbam_use_mloss = False if str(parsed.cbam_use_mloss).lower() == "false" else True

        self.model_type = str(parsed.model_type).lower()
        self.is_vgg16_model = True if "vgg" in self.model_type else False

        self.image_size = int(parsed.image_size)

        if str(parsed.classifier_loss_function).lower() == "bceloss":
            self.classifier_loss_function = nn.BCELoss()
        elif str(parsed.classifier_loss_function).lower() == "softf1":
            self.classifier_loss_function = f1loss.SoftF1Loss()
        else:
            raise Exception("classifier loss {} not found".format(parsed.classifier_loss_function))

        if str(parsed.am_model).lower() == "sum":
            self.am_model_type = parsed.am_model
        elif str(parsed.am_model).lower() == "product":
            self.am_model_type = parsed.am_model
        elif str(parsed.am_model).lower() == "sum_shift":
            self.am_model_type = parsed.am_model
        elif str(parsed.am_model).lower() == "product_shift":
            self.am_model_type = parsed.am_model
        elif str(parsed.am_model).lower() == "cbam":
            self.am_model_type = parsed.am_model
        else:
            raise Exception("model {} not found".format(parsed.am_model))

        if str(parsed.am_loss_function).lower() == "bceloss":
            self.am_loss_function = nn.BCELoss()
        elif str(parsed.am_loss_function).lower() == "softf1":
            self.am_loss_function = f1loss.SoftF1Loss()
        else:
            raise Exception("am loss {} not found".format(parsed.am_loss_function))

        if str(parsed.dataset_type).lower() == "balanced":
            self.dataset_type = "balanced"
        else:
            self.dataset_type = "imbalanced"
        self.model_identifier = parsed.model_identifier
        self.execute_from_model = False if str(parsed.execute_from_model).lower() == "false" else True

        self.train_batch_size = int(parsed.train_batch_size)
        self.test_batch_size = int(parsed.test_batch_size)

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
        self.puller = None
        self.strategy = None

        self.initialize_logs()
        self.initialize_snapshots_dir()
        self.load_dataset()
        self.current_epoch = self.get_current_epoch()
        self.model_state_dict = self.load_model_from_saves()
        self.model = self.create_model()
        self.strategy = self.create_strategy()
        P.write_to_log("incoming args = {}".format(parsed))

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
            if isinstance(e, SystemExit) and e.args[0] == 0:
                P.write_to_log("Exit with 0")
                return
            print("EXCEPTION", e)
            print(type(e))
            P.write_to_log("EXCEPTION", e, type(e))
            P.write_to_log(traceback.extract_tb(e.__traceback__))

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

        if self.dataset_type != "balanced":
            segments_set, test_set = il.load_data(self.train_set_size, self.model_identifier, self.image_size)
        else:
            segments_set, test_set = imbalanced.load_balanced_dataset(self.train_set_size, self.model_identifier,
                                                                      self.image_size)
        self.train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=self.train_batch_size,
                                             shuffle=True)
        print("ok")
        self.test_set = DataLoader(il.ImageDataset(test_set), batch_size=self.test_batch_size)
        print("ok")

    def get_current_epoch(self) -> int:
        if self.execute_from_model:
            model_state_dict, current_epoch = P.load_latest_model(self.model_identifier, self.run_name,
                                                                  self.algorithm_name)
            if model_state_dict is None:
                exit(0)
                # raise Exception(
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
                # raise Exception(
                #    "not found model for current epoch: model_identifier: {}, run_name: {}, algorithm_name: {}"
                #        .format(self.model_identifier, self.run_name, self.algorithm_name))
            return model_state_dict
        return None

    def build_model(self):
        pass

    def select_resnet_model(self):
        if self.model_type == "resnet18":
            model = m.resnet18(pretrained=True)
        elif self.model_type == "resnet34":
            model = m.resnet34(pretrained=True)
        elif self.model_type == "resnet50":
            model = m.resnet50(pretrained=True)
        elif self.model_type == "resnet101":
            model = m.resnet101(pretrained=True)
        elif self.model_type == "resnet152":
            model = m.resnet152(pretrained=True)
        else:
            raise Exception("Not exist model with name: {}".format(self.model))
        return model

    def build_baseline_resnet_model(self):
        model = self.select_resnet_model()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.classes)
        return model

    def build_baseline_vgg_model(self):
        model = m.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, self.classes)
        return model

    def build_vgg16_with_am_model(self):
        if self.am_model_type == "sum":
            self.model, self.puller = am_model.build_attention_module_model(self.classes, cb.ConnectionSumBlock())
        elif self.am_model_type == "product":
            self.model, self.puller = am_model.build_attention_module_model(self.classes, cb.ConnectionProductBlock())
        elif self.am_model_type == "sum_shift":
            self.model, self.puller = am_model_shift.build_attention_module_model(self.classes, cb.ConnectionSumBlock())
        elif self.am_model_type == "product_shift":
            self.model, self.puller = am_model_shift.build_attention_module_model(self.classes,
                                                                                  cb.ConnectionProductBlock())
        return self.model

    def build_resnet_with_am_model(self):
        if self.model_type != "resnet18" and self.model_type != "resnet34":
            raise Exception("Model {} not supported".format(self.model_type))
        model = self.select_resnet_model()
        if self.am_model_type == "sum":
            self.model, self.puller = resnet_am_model.build_attention_module_model(self.classes, model,
                                                                                   cb.ConnectionSumBlock())
        elif self.am_model_type == "product":
            self.model, self.puller = resnet_am_model.build_attention_module_model(self.classes, model,
                                                                                   cb.ConnectionProductBlock())
        return self.model

    def build_model_with_am(self):
        if "vgg" in self.model_type:
            return self.build_vgg16_with_am_model()
        else:
            return self.build_resnet_with_am_model()

    def build_model_without_am(self):
        if "vgg" in self.model_type:
            return self.build_baseline_vgg_model()
        else:
            return self.build_baseline_resnet_model()

    def build_resnet_with_cbam(self):
        if "resnet34" in self.model_type:
            self.model, self.puller = resnet34cbam.build_resnet34_with_cbam(self.classes)
        if "resnet18" in self.model_type:
            self.model, self.puller = resnet18cbam.build_resnet18_with_cbam(self.classes)
