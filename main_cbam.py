import image_loader as il
from torch.utils.data import DataLoader
import gain
import property as P
import sys
import torchvision.models as m
import torch.nn as nn
import cbam_model as cbm
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
            layers.append(cbm.CBAM(output_dim))
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

    description = "BAM_{}_train_left-{},train_segments-{},train_right-{},test_left-{},test_right-{},am_loss-{}," \
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

    P.initialize_log_name("gain_with_сbam" + description)

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
                   4,
                   4,
                   10,
                   pre_train)
    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()
        
        raise e

"""
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    here!
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    here!
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    here!
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    here!
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    here!
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""

"""
/home/nikita/anaconda3/envs/ml-diplom/bin/python /home/nikita/PycharmProjects/ml-diplom/main_cbam.py
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): BAM(
      (channel_att): ChannelGate(
        (gate_c): Sequential(
          (flatten): Flatten()
          (gate_c_fc_0): Linear(in_features=64, out_features=4, bias=True)
          (gate_c_bn_1): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_c_relu_1): ReLU()
          (gate_c_fc_final): Linear(in_features=4, out_features=64, bias=True)
        )
      )
      (spatial_att): SpatialGate(
        (gate_s): Sequential(
          (gate_s_conv_reduce0): Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1))
          (gate_s_bn_reduce0): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_reduce0): ReLU()
          (gate_s_conv_di_0): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_0): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_0): ReLU()
          (gate_s_conv_di_1): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_1): ReLU()
          (gate_s_conv_final): Conv2d(4, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): BAM(
      (channel_att): ChannelGate(
        (gate_c): Sequential(
          (flatten): Flatten()
          (gate_c_fc_0): Linear(in_features=128, out_features=8, bias=True)
          (gate_c_bn_1): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_c_relu_1): ReLU()
          (gate_c_fc_final): Linear(in_features=8, out_features=128, bias=True)
        )
      )
      (spatial_att): SpatialGate(
        (gate_s): Sequential(
          (gate_s_conv_reduce0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1))
          (gate_s_bn_reduce0): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_reduce0): ReLU()
          (gate_s_conv_di_0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_0): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_0): ReLU()
          (gate_s_conv_di_1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_1): ReLU()
          (gate_s_conv_final): Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): BAM(
      (channel_att): ChannelGate(
        (gate_c): Sequential(
          (flatten): Flatten()
          (gate_c_fc_0): Linear(in_features=256, out_features=16, bias=True)
          (gate_c_bn_1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_c_relu_1): ReLU()
          (gate_c_fc_final): Linear(in_features=16, out_features=256, bias=True)
        )
      )
      (spatial_att): SpatialGate(
        (gate_s): Sequential(
          (gate_s_conv_reduce0): Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
          (gate_s_bn_reduce0): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_reduce0): ReLU()
          (gate_s_conv_di_0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_0): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_0): ReLU()
          (gate_s_conv_di_1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_1): ReLU()
          (gate_s_conv_final): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (20): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): ReLU(inplace=True)
    (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (23): ReLU(inplace=True)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): BAM(
      (channel_att): ChannelGate(
        (gate_c): Sequential(
          (flatten): Flatten()
          (gate_c_fc_0): Linear(in_features=512, out_features=32, bias=True)
          (gate_c_bn_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_c_relu_1): ReLU()
          (gate_c_fc_final): Linear(in_features=32, out_features=512, bias=True)
        )
      )
      (spatial_att): SpatialGate(
        (gate_s): Sequential(
          (gate_s_conv_reduce0): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
          (gate_s_bn_reduce0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_reduce0): ReLU()
          (gate_s_conv_di_0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_0): ReLU()
          (gate_s_conv_di_1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_1): ReLU()
          (gate_s_conv_final): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace=True)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace=True)
    (34): BAM(
      (channel_att): ChannelGate(
        (gate_c): Sequential(
          (flatten): Flatten()
          (gate_c_fc_0): Linear(in_features=512, out_features=32, bias=True)
          (gate_c_bn_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_c_relu_1): ReLU()
          (gate_c_fc_final): Linear(in_features=32, out_features=512, bias=True)
        )
      )
      (spatial_att): SpatialGate(
        (gate_s): Sequential(
          (gate_s_conv_reduce0): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
          (gate_s_bn_reduce0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_reduce0): ReLU()
          (gate_s_conv_di_0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_0): ReLU()
          (gate_s_conv_di_1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (gate_s_bn_di_1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (gate_s_relu_di_1): ReLU()
          (gate_s_conv_final): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (35): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)



"""
