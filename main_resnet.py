import image_loader as il
from torch.utils.data import DataLoader
import property as P
import sys
import torchvision.models as m
import traceback
import classifier_train as cl
import os
import torch.nn as nn

#
#
# lr = 1e-4
# weight_decay=0.01

if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu = 0
    parsed_description = parsed.description
    train_set_size = int(parsed.train_set)
    epochs = int(parsed.epochs)
    run_name = parsed.run_name
    algorithm_name = parsed.algorithm_name
    left_class_number = int(parsed.left_class_number)
    right_class_number = int(parsed.right_class_number)
    classes = right_class_number - left_class_number
    classifier_learning_rate = float(parsed.classifier_learning_rate)
    attention_module_learning_rate = float(parsed.attention_module_learning_rate)
    weight_decay = float(parsed.weight_decay)

    seed = int(parsed.seed)
    description = "description-{},train_set-{},epochs-{},l-{},r-{},clr-{},amlr-{},seed-{},weight_decay-{}".format(
        parsed_description,
        train_set_size,
        epochs,
        left_class_number,
        right_class_number,
        classifier_learning_rate,
        attention_module_learning_rate,
        seed,
        weight_decay
    )

    P.initialize_log_name(run_name, algorithm_name, description)

    P.write_to_log("description={}".format(description))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    try:
        segments_set, test_set = il.load_data(train_set_size, seed)

        train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=5, shuffle=True)
        print("ok")
        test_set = DataLoader(il.ImageDataset(test_set), batch_size=5)
        print("ok")

        model = m.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, classes)
        P.write_to_log(model)

        classifier = cl.Classifier(model,
                                   train_segments_set,
                                   test_set,
                                   classes=classes,
                                   test_each_epoch=4,
                                   gpu_device=gpu,
                                   train_epochs=epochs,
                                   left_class_number=left_class_number,
                                   right_class_number=right_class_number,
                                   description=run_name + "_" + description,
                                   classifier_learning_rate=classifier_learning_rate,
                                   attention_module_learning_rate=attention_module_learning_rate,
                                   weight_decay=weight_decay)
        classifier.train()

    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
