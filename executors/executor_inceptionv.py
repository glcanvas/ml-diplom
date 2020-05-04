from utils import property as P
import sys
import torchvision.models as m
from strategies import vgg16_baseline_strategy as cl
import torch.nn as nn
from executors.abastract_executor import AbstractExecutor
from model import googlenet
from model import inceptionv3

class InceptionBaselineExecutor(AbstractExecutor):
    def __init__(self, parsed):
        super(InceptionBaselineExecutor, self).__init__(parsed)

    def create_model(self):
        if self.inceptionv_type == "inceptionv3":
            self.model = inceptionv3.inception_v3(pretrained=True)
        elif self.inceptionv_type == "inceptionv1":
            self.model = googlenet.inceptionv1(pretrained=True)
        else:
            raise Exception("Not exist model with name: {}".format(self.resnet_type))
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.classes)
        P.write_to_log(self.model)

        if self.execute_from_model:
            self.model.load_state_dict(self.model_state_dict)
            P.write_to_log("recovery model:", self.model, "current epoch = {}".format(self.current_epoch))
        return self.model

    def create_strategy(self):
        self.strategy = cl.Classifier(self.model,
                                      self.train_segments_set,
                                      self.test_set,
                                      classes=self.classes,
                                      test_each_epoch=4,
                                      gpu_device=self.gpu,
                                      train_epochs=self.epochs,
                                      left_class_number=self.left_class_number,
                                      right_class_number=self.right_class_number,
                                      description=self.run_name + "_" + self.description,
                                      classifier_learning_rate=self.classifier_learning_rate,
                                      attention_module_learning_rate=self.attention_module_learning_rate,
                                      is_freezen=self.is_freezen,
                                      current_epoch=self.current_epoch)
        return self.strategy

    def train_strategy(self):
        self.strategy.train()


def execute(args=None):
    parsed = P.parse_input_commands().parse_args(sys.argv[1:]) if args is None else args

    alternate = InceptionBaselineExecutor(parsed)
    alternate.safe_train()


if __name__ == "__main__":
    execute(None)
