import image_loader as il
from torch.utils.data import DataLoader
import property as P
import sys
import torchvision.models as m
import torch.nn as nn
import traceback
import sam_model as ss
import sam_train as st

classes = 1

if __name__ == "__main__":
    P.initialize_log_name("_" + "AA")
    loader = il.DatasetLoader.initial()
    train_segments = loader.load_tensors(0, 3, 4)
    train_classifier = loader.load_tensors(0, 3, 0)
    test = loader.load_tensors(0, 4)

    train_segments_set = DataLoader(il.ImageDataset(train_segments), batch_size=10, shuffle=True)
    train_classifier_set = DataLoader(il.ImageDataset(train_classifier), batch_size=10, shuffle=True)
    test_set = DataLoader(il.ImageDataset(test), batch_size=10)

    sam_branch = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 1)),
        nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 1, kernel_size=(3, 3), padding=(1, 1)),
        nn.Sigmoid()
    )
    model = m.vgg16(pretrained=True)
    basis_branch = model.features[:4]

    # parallel
    classifier_branch = model.features[4:16]

    merged_branch = model.features[16:]
    avg_pool = model.avgpool
    classifier = nn.Sequential(*model.classifier,
                               nn.Linear(1000, classes),
                               nn.Sigmoid())

    sam_model = ss.SAM(basis_branch,
                       sam_branch,
                       classifier_branch,
                       merged_branch,
                       avg_pool,
                       classifier)
    # print(sam_model)
    sam_train = st.SAM_TRAIN(sam_model, train_classifier_set, train_segments_set, test_set, classes=classes,
                             pre_train_epochs=2,
                             train_epochs=15,
                             save_train_logs_epochs=5,
                             test_each_epoch=5)
    sam_train.train()
