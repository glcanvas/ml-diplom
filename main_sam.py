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
    pre_train = int(parsed.pre_train)
    train_set_size = int(parsed.train_set)
    epochs = int(parsed.epochs)
    change_lr_epochs = int(parsed.change_lr)
    run_name = parsed.run_name
    algorithm_name = parsed.algorithm_name

    description = "description-{},train_set-{},pre_train_epochs-{},update_lr_epoch-{},epochs-{},classes-{}".format(
        parsed_description,
        train_set_size,
        pre_train,
        change_lr_epochs,
        epochs,
        classes
    )

    P.initialize_log_name(run_name, algorithm_name, description)

    P.write_to_log("description={}".format(description))
    P.write_to_log("classes={}".format(classes))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    try:
        segments_set, test_set = il.load_data_2(train_set_size)

        train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=5, shuffle=True)
        print("ok")
        test_set = DataLoader(il.ImageDataset(test_set), batch_size=5)
        print("ok")

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
        merged_branch = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            *merged_branch
        )

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

        P.write_to_log(sam_model)
        sam_train = st.SAM_TRAIN(sam_model, train_segments_set, test_set, classes=classes,
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
