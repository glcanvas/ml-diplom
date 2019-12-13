import torch
import gain
import image_loader as il
from torch.utils.data import DataLoader
import classifier
import torch.nn.functional as F
import torch.nn as nn


gain_model = gain.AttentionGAIN(classes=5, gpu=True)
# clf = classifier.Classifier(5, gpu=True)

loader = il.DatasetLoader.initial()
train = loader.load_tensors(0, 50)
test = loader.load_tensors(55, 70)

train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
test_set = DataLoader(il.ImageDataset(test), batch_size=1, shuffle=True, num_workers=0)

rds = {'train': train_set, 'test': test_set}

if __name__ == "__main__":
    # a = torch.tensor([[0.5, 0.25, 0.10, 0.15], [0.1, 0.25, 0.5, 0.15]])
    # print(F.softmax(a, dim=1).max(dim=1))

    # clf.train(train_set, test_set, 100, 10, 1)
    gain_model.train(rds, 100, train_batch_size=10)
    # gain_model.test(rds)
    """
    t = torch.tensor([[10, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 21, 34, 44, 45, 76, 17, 38, 49, 10]])
    x = t.view((2, 5, 2))
    print(t)
    print(x)
    print(x.max(dim=2))
    """