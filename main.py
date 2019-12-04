import image_processing as il
import torch.nn as nn
from torch.utils.data import DataLoader
from models.gain import *

gain = GAIN('features.10', 2)

loader = il.DatasetLoader.initial()
train = loader.load_tensors(0, 100)
test = loader.load_tensors(400, 450)

train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

loss = nn.CrossEntropyLoss()


if __name__ == "__main__":
    for inputs, segments, labels, _ in train_set:
        inputs = inputs.cpu()
        labels = labels.cpu()
        logits, logits_am, heatmap = gain.forward(inputs, labels)
        loss(labels.float(), logits)
        print(logits)
