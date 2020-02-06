import image_loader as il
from torch.utils.data import DataLoader
import property as P
import sys
import torchvision.models as m
import torch.nn as nn
import traceback
import sam_model as ss
import sam_train as st
import cbam_model as cbam

classes = 5

if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    parsed_description = parsed.description
    # train_left = int(parsed.train_left)
    # train_right = int(parsed.train_right)
    # test_left = int(parsed.test_left)
    # test_right = int(parsed.test_right)
    # train_segments_count = int(parsed.segments)
    pre_train = int(parsed.pre_train)
    # use_am_loss = parsed.am_loss.lower() == "true"
    # gradient_layer_name = parsed.gradient_layer_name
    # from_gradient_layer = parsed.from_gradient_layer.lower() == "true"

    classifier = float(parsed.classifier)
    segments = float(parsed.segments)
    test = float(parsed.test)

    epochs = int(parsed.epochs)
    change_lr_epochs = int(parsed.change_lr)

    description = "{}_segments-{},classifier-{},tests-{},pre_train-{}," \
                  "lr_epoch-{}".format(parsed_description,
                                       segments, classifier, test,
                                       pre_train,
                                       change_lr_epochs
                                       )

    P.initialize_log_name("_" + description)

    try:
        """loader = il.DatasetLoader.initial()
        print("A")
        train_segments = loader.load_tensors(train_left, train_segments_count, train_segments_count)
        print("A")
        train_classifier = loader.load_tensors(train_segments_count, train_right, 0)
        print("A")
        test = loader.load_tensors(test_left, test_right)
        print("A")"""
        classifier_set, segments_set, test_set = il.load_all_data(classifier, segments, test)

        train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=5, shuffle=True)
        print("A")
        train_classifier_set = DataLoader(il.ImageDataset(classifier_set), batch_size=5, shuffle=True)
        print("A")
        test_set = DataLoader(il.ImageDataset(test_set), batch_size=5)
        print("A")

        sam_branch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            cbam.CBAM(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 5, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )
        model = m.vgg16(pretrained=True)
        basis_branch = model.features[:4]

        # parallel
        classifier_branch = model.features[4:16]

        merged_branch = model.features[16:]
        merged_branch[1] = nn.Conv2d(1280, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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
        print(sam_model)
        sam_train = st.SAM_TRAIN(sam_model, train_classifier_set, train_segments_set, test_set, classes=classes,
                                 pre_train_epochs=pre_train,
                                 gpu_device=gpu,
                                 train_epochs=epochs,
                                 save_train_logs_epochs=4,
                                 test_each_epoch=4,
                                 change_lr_epochs=change_lr_epochs)
        sam_train.train()

    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
