import gain
import image_loader as il
from torch.utils.data import DataLoader
import classifier
import torch
import torch.nn.functional as F
import torch.nn as nn

# gain_model = gain.AttentionGAIN(gpu=True)
clf = classifier.Classifier(6, gpu=True)

loader = il.DatasetLoader.initial()
train = loader.load_tensors(0, 50)
test = loader.load_tensors(0, 4)

train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
test_set = DataLoader(il.ImageDataset(test), batch_size=1, shuffle=True, num_workers=0)

rds = {'train': train_set, 'test': train_set}

if __name__ == "__main__":
    # a = torch.tensor([[0.5, 0.25, 0.10, 0.15], [0.1, 0.25, 0.5, 0.15]])
    # print(F.softmax(a, dim=1).max(dim=1))

    clf.train(train_set, test_set, 100, 10, 1)
    #    gain_model.train(rds, 100)
    #    gain_model.test(rds)
