# flake8: noqa
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from callbacks import *
from optimizers import *
import image_processing as il

from torch.utils.data import DataLoader
from models.gain import *

# Register models
registry.Model(GAIN)
registry.Model(GCAM)

# Register callbacks
registry.Callback(GAINCriterionCallback)
registry.Callback(GAINSaveHeatmapCallback)
registry.Callback(GCAMSaveHeatmapCallback)
registry.Callback(GAINMaskCriterionCallback)

# Register criterions

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)

gain = GAIN('features.10', 2)

loader = il.DatasetLoader.initial()
train = loader.load_tensors(0, 100)
test = loader.load_tensors(400, 450)

train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

if __name__ == "__init__":
    for inputs, segments, labels, _ in train_set:
        inputs = inputs.cpu()
        labels = labels.cpu()
        logits, logits_am, heatmap = gain.forward(inputs, labels)

        print(logits)
