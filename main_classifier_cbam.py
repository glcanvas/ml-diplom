import image_loader as il
from torch.utils.data import DataLoader
import classifier
import property as P
import sys
import torchvision.models as m
import torch.nn as nn
import main_cbam as mb
import traceback

if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    parsed_description = parsed.description
    train_left = int(parsed.train_left)
    train_right = int(parsed.train_right)
    test_left = int(parsed.test_left)
    test_right = int(parsed.test_right)
    train_segments_count = int(parsed.segments)
    use_am_loss = parsed.am_loss.lower() == "true"
    pre_train = int(parsed.pre_train)
    gradient_layer_name = parsed.gradient_layer_name
    from_gradient_layer = parsed.from_gradient_layer.lower() == "true"
    epochs = int(parsed.epochs)
    description = "BAM_CLASSIFIER_{}_train_left-{},train_right-{},test_left-{},test_right-{}" \
        .format(parsed_description,
                train_left,
                train_right,
                test_left,
                test_right
                )

    P.initialize_log_name("classifier_with_cbam" + description)
    model = mb.build_model(m.vgg16(pretrained=True), [3, 8, 15, 22, 29], [2, 7, 14, 21, 28])
    try:
        clf = classifier.Classifier(description, 5, gpu=True, device=gpu)

        loader = il.DatasetLoader.initial()
        train = loader.load_tensors(train_left, train_right)
        test = loader.load_tensors(test_left, test_right)

        train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
        test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

        clf.train(epochs, 4, 4, 10, train_set, test_set)
    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        trace = traceback.format_exc()
        print(trace)
        P.write_to_log(trace)
        raise e
