import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

from utils import property as P
import sys
from model import vgg_with_am_product_model as product_am
from model import vgg_with_am_sum_model as sum_am
from strategies import single_simultaneous_train as st
from executors.abastract_executor import AbstractExecutor


class SimultaneousExecutor(AbstractExecutor):
    def __init__(self, parsed):
        super(SimultaneousExecutor, self).__init__(parsed)

    def create_model(self):
        if self.am_model_type == "sum":
            self.model = sum_am.build_attention_module_model(self.classes)
        elif self.am_model_type == "product":
            self.model = product_am.build_attention_module_model(self.classes)
        P.write_to_log(self.model)
        if self.execute_from_model:
            self.model.load_state_dict(self.model_state_dict)
            P.write_to_log("recovery model:", self.model, "current epoch = {}".format(self.current_epoch))
        return self.model

    def create_strategy(self):
        self.strategy = st.SingleSimultaneousModuleTrain(self.model, self.train_segments_set, self.test_set,
                                                         classes=self.classes,
                                                         pre_train_epochs=self.pre_train,
                                                         gpu_device=self.gpu,
                                                         train_epochs=self.epochs,
                                                         save_train_logs_epochs=4,
                                                         l_loss=self.classifier_loss_function,
                                                         m_loss=self.am_loss_function,
                                                         test_each_epoch=4,
                                                         left_class_number=self.left_class_number,
                                                         right_class_number=self.right_class_number,
                                                         description=self.run_name + "_" + self.description,
                                                         snapshot_elements_count=20,
                                                         snapshot_dir=self.snapshots_path,
                                                         classifier_learning_rate=self.classifier_learning_rate,
                                                         attention_module_learning_rate=self.attention_module_learning_rate,
                                                         current_epoch=self.current_epoch)
        return self.strategy

    def train_strategy(self):
        self.strategy.train()


def execute(args=None):
    parsed = P.parse_input_commands().parse_args(sys.argv[1:]) if args is None else args

    alternate = SimultaneousExecutor(parsed)
    alternate.safe_train()


if __name__ == "__main__":
    execute(None)
