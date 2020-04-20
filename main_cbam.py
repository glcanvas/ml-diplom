import image_loader as il
from torch.utils.data import DataLoader
import property as P
import sys
import torchvision.models as m
import traceback
import classifier_train as cl
import os
import torch.nn as nn
import build_cbam_bam_model as bm


class VggWithCbam(nn.Module):

    def __init__(self):
        super(VggWithCbam, self).__init__()


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
    is_freezen = False if str(parsed.is_freezen).lower() == "false" else True

    time_stamp = parsed.time_stamp
    execute_from_model = False if str(parsed.execute_from_model).lower() == "false" else True

    seed = int(parsed.seed)
    description = "description-{},train_set-{},epochs-{},l-{},r-{},clr-{},amlr-{},seed-{}".format(
        parsed_description,
        train_set_size,
        epochs,
        left_class_number,
        right_class_number,
        classifier_learning_rate,
        attention_module_learning_rate,
        seed
    )

    P.initialize_log_name(run_name, algorithm_name, description, time_stamp)

    P.write_to_log("description={}".format(description))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    segments_set, test_set = il.load_data(train_set_size, seed)

    train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=5, shuffle=True)
    print("ok")
    test_set = DataLoader(il.ImageDataset(test_set), batch_size=5)
    print("ok")

    # model = m.vgg16(pretrained=True)
    # num_features = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_features, classes)
    model = bm.build_cbam_model(classes)

    current_epoch = 1
    if execute_from_model:
        model_state_dict, current_epoch = P.load_latest_model(time_stamp, run_name, algorithm_name)
        if model_state_dict is None:
            exit(0)
        model.load_state_dict(model_state_dict)
        P.write_to_log("recovery model:", model, "current epoch = {}".format(current_epoch))
    else:
        P.write_to_log("begin model:", model, "current epoch = {}".format(current_epoch))

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
                               is_freezen=is_freezen,
                               current_epoch=current_epoch)
    #try:
    classifier.train()
    exit(0)
    """except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        P.save_raised_model(classifier.model, classifier.current_epoch, time_stamp, run_name, algorithm_name)
        P.write_to_log("saved model, exception raised")
        exit(1)
    """