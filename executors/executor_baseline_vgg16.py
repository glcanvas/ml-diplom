import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")

from utils import property as P
import sys
import torchvision.models as m
from strategies import vgg16_baseline_strategy as cl
import torch.nn as nn

from executors.abastract_executor import AbstractExecutor


class VGG16BaselineExecutor(AbstractExecutor):
    def __init__(self, parsed):
        super(VGG16BaselineExecutor, self).__init__(parsed)

    def create_model(self):
        self.model = m.vgg16(pretrained=True)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, self.classes)
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

    alternate = VGG16BaselineExecutor(parsed)
    alternate.safe_train()


if __name__ == "__main__":
    execute(None)
