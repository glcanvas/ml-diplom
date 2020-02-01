import image_loader as il
from torch.utils.data import DataLoader
import gain
import property as P
import sys
import torchvision.models as m
import torch.nn as nn
import bam_model as bm
import traceback


def build_model(module: nn.Module, after_indexes: list, use_dim_indexes: list, feature: str = 'features'):
    feature_seq = module.__getattr__(feature)
    layers = []
    cnt = 0
    for idx, layer in enumerate(feature_seq):
        layers.append(layer)
        if idx in after_indexes:
            output_dim = feature_seq[use_dim_indexes[cnt]].out_channels
            cnt += 1
            layers.append(bm.BAM(output_dim))
            pass

    module.__setattr__(feature, nn.Sequential(*layers))
    return module


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

    description = "{}_train_left-{},train_segments-{},train_right-{},test_left-{},test_right-{},am_loss-{}," \
                  "pre_train-{}_gradient_layer_name-{}_from_gradient_layer-{}" \
        .format(parsed_description,
                train_left,
                train_segments_count,
                train_right,
                test_left,
                test_right,
                use_am_loss,
                pre_train,
                gradient_layer_name,
                from_gradient_layer
                )

    P.initialize_log_name("_" + description)

    try:
        model = build_model(m.vgg16(pretrained=True), [3, 8, 15, 22, 29], [2, 7, 14, 21, 28])

        gain = gain.AttentionGAIN(description, 5, gpu=True, model=model, device=gpu,
                                  gradient_layer_name=gradient_layer_name, from_gradient_layer=from_gradient_layer,
                                  usage_am_loss=use_am_loss)

        loader = il.DatasetLoader.initial()
        train_segments = loader.load_tensors(train_left, train_segments_count, train_segments_count)
        train_classifier = loader.load_tensors(train_segments_count, train_right, 0)
        test = loader.load_tensors(test_left, test_right)

        train_segments_set = DataLoader(il.ImageDataset(train_segments), batch_size=10, shuffle=True)
        train_classifier_set = DataLoader(il.ImageDataset(train_classifier), batch_size=10, shuffle=True)
        test_set = DataLoader(il.ImageDataset(test), batch_size=10)

        gain.train({'train_segment': train_segments_set, 'train_classifier': train_classifier_set, 'test': test_set},
                   epochs,
                   1,
                   4,
                   10,
                   pre_train)
    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
