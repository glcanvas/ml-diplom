import image_loader as il
from torch.utils.data import DataLoader
import property as P
import sys
import torchvision.models as m
import torch.nn as nn
import traceback
import sam_model as ss
import attention_module_train as amt
import os

classes = 5
class_number = None


def register_weights(weight_class, model):
    if weight_class == "classifier":
        a = list(model.classifier_branch.parameters())
        a.extend(list(model.merged_branch.parameters()))
        a.extend(list(model.avg_pool.parameters()))
        a.extend(list(model.classifier.parameters()))
        return a
    elif weight_class == "attention":
        a = list(model.basis.parameters())
        a.extend(list(model.sam_branch.parameters()))
        return a
    raise BaseException("unrecognized param: " + weight_class)


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
    use_class_number = int(parsed.use_class_number)
    if use_class_number != -1:
        classes = 1
        class_number = use_class_number

    description = "ATTENTION_MODULE_ONLY_description-{},train_set-{},pre_train_epochs-{}," \
                  "update_lr_epoch-{},epochs-{},class_number-{}".format(parsed_description,
                                                                        train_set_size,
                                                                        pre_train,
                                                                        change_lr_epochs,
                                                                        epochs,
                                                                        class_number
                                                                        )

    P.initialize_log_name(run_name, algorithm_name, description)

    P.write_to_log("description={}".format(description))
    P.write_to_log("classes={}".format(classes))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    log_name, log_dir = os.path.basename(P.log)[:-4], os.path.dirname(P.log)

    snapshots_path = os.path.join(log_dir, log_name)
    os.makedirs(snapshots_path, exist_ok=True)

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
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, classes, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )
        model = m.vgg16(pretrained=True)
        basis_branch = model.features[:4]

        # parallel
        classifier_branch = model.features[4:16]

        merged_branch = model.features[16:]
        merged_branch = nn.Sequential(
            nn.Conv2d(256 * classes, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
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
        sam_train = amt.ATTENTION_MODULE_TRAIN(sam_model, train_segments_set, test_set, classes=classes,
                                               gpu_device=gpu,
                                               train_epochs=epochs,
                                               change_lr_epochs=change_lr_epochs,
                                               class_number=class_number,
                                               description=run_name + "_" + description,
                                               register_weights=register_weights,
                                               snapshot_elements_count=10,
                                               snapshot_dir=snapshots_path)
        sam_train.train_attention_module()

    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
