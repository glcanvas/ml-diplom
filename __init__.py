"""# flake8: noqa
from catalyst.dl import registry
from models import *
from callbacks import *
from optimizers import *


# Register models
registry.Model(GAIN)
registry.Model(GCAM)

# Register callbacks
registry.Callback(GAINCriterionCallback)
# registry.Callback(GAINSaveHeatmapCallback)
# registry.Callback(GCAMSaveHeatmapCallback)
registry.Callback(GAINMaskCriterionCallback)

# Register criterions

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)
"""
import image_loader as il
from torch.utils.data import DataLoader
from models import *
import property as P
import sys

import model_trainer as mt

if __name__ == "__main__":
    # parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    # gpu = int(parsed.gpu)
    # parsed_description = parsed.description
    train_left = 0  # int(parsed.train_left)
    train_right = 100  # int(parsed.train_right)
    test_left = 150  # int(parsed.test_left)
    test_right = 250  # int(parsed.test_right)
    train_segments_count = 50  # int(parsed.segments)
    use_am_loss = False  # parsed.am_loss.lower() == "true"
    pre_train = 2  # int(parsed.pre_train)
    """
    description = "{}_train_left-{},train_segments-{},train_right-{},test_left-{},test_right-{},am_loss-{}," \
                  "pre_train-{}" \
        .format(parsed_description,
                train_left,
                train_segments_count,
                train_right,
                test_left,
                test_right,
                use_am_loss,
                pre_train
                )
    """
    P.initialize_log_name("gain_ngx")

    trainer = mt.Trainer("AAA", gpu=True)

    loader = il.DatasetLoader.initial()
    train_segments = loader.load_tensors(train_left, train_segments_count, train_segments_count)
    train_classifier = loader.load_tensors(train_segments_count, train_right, 0)
    test = loader.load_tensors(test_left, test_right)

    train_segments_set = DataLoader(il.ImageDataset(train_segments), batch_size=10, shuffle=True)
    train_classifier_set = DataLoader(il.ImageDataset(train_classifier), batch_size=10, shuffle=True)
    test_set = DataLoader(il.ImageDataset(test), batch_size=10)

    trainer.train(train_segments_set, train_classifier_set, test_set, pre_train_epoch=3)
